"""The Node/Chromium sidecar of the browser backends.

Python keeps credentials, the trusted scheduler and cleanup; a Node process (streameval/providers/js/) bundles
the vendor's page with its pinned JS SDK, runs it in headless Chromium (Playwright) with fake media devices, serves
only the job's verified assets and forwards JSON-line commands to ``window.arena[op](args)``:

    stdin   {"id", "op", "args"}                stdout  {"event", ...}   and   {"event": "reply", "id", "ok", ...}

Page times are performance.now() milliseconds, mapped to the host clock by a min-RTT round trip (clock_sync).

    STREAMEVAL_NODE             node executable (default: node on PATH)
    STREAMEVAL_BROWSER_RUNTIME  folder whose node_modules holds playwright, esbuild and the vendor SDKs
                                  (default: streameval/providers/js after `npm ci` there)
    STREAMEVAL_CHROMIUM         Chromium executable (default: Playwright's own build, `npx playwright install
                                  chromium`; pinned Playwright 1.63.0 / chromium-1243)
"""
from __future__ import annotations

import json
from pathlib import Path
import queue
import subprocess
import threading
import time

from ..config import env

JS = Path(__file__).resolve().parent / "js"
BROWSER_FLAGS = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", "--mute-audio",
                 "--autoplay-policy=no-user-gesture-required", "--disable-background-timer-throttling",
                 "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows"]


def node() -> str:
    return env("STREAMEVAL_NODE", "node")


def runtime() -> Path:
    return Path(env("STREAMEVAL_BROWSER_RUNTIME") or JS).expanduser()


def chromium() -> str | None:
    return env("STREAMEVAL_CHROMIUM")


def check_runtime() -> Path:
    root = runtime()
    if not (root / "node_modules/playwright").is_dir() or not (root / "node_modules/esbuild").is_dir():
        raise RuntimeError(f"browser runtime not installed in {root}: run `npm ci && npx playwright install chromium` "
                           "in streameval/providers/js (or set STREAMEVAL_BROWSER_RUNTIME)")
    return root


def installed(package: str) -> dict:
    """The lockfile entry of an installed package of the runtime ({} when absent)."""
    lock = runtime() / "package-lock.json"
    if not lock.is_file():
        return {}
    return json.loads(lock.read_text()).get("packages", {}).get("node_modules/" + package) or {}


class BrowserError(RuntimeError):
    pass


class Browser:
    """One sidecar process for one session; `script` is the Node entry point in providers/js/."""

    def __init__(self, directory: Path, launch: dict, script: str = "edit_browser.mjs"):
        directory.mkdir(parents=True, mode=0o700)
        root = check_runtime()
        launch = {"out": str(directory), "runtime": str(root), "chromium": chromium(), "flags": BROWSER_FLAGS,
                  **launch}
        path = directory / "launch.json"
        path.write_text(json.dumps(launch, indent=1))
        self.stderr = (directory / "browser_stderr.log").open("w")
        self.proc = subprocess.Popen([node(), str(JS / script), str(path)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr, text=True, bufsize=1, cwd=str(JS))
        self.events: queue.Queue = queue.Queue()
        self.replies, self.seq, self.lock = {}, 0, threading.Lock()
        self.cv = threading.Condition()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for raw in self.proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            msg["host_ns"] = time.monotonic_ns()
            if msg.get("event") == "reply":
                with self.cv:
                    self.replies[msg["id"]] = msg
                    self.cv.notify_all()
            else:
                self.events.put(msg)
        self.events.put({"event": "process_exited", "host_ns": time.monotonic_ns()})

    def wait_event(self, name: str, timeout: float) -> dict:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                msg = self.events.get(timeout=max(0.01, end - time.monotonic()))
            except queue.Empty:
                break
            if msg.get("event") == name:
                return msg
            if msg.get("event") in {"fatal", "process_exited"}:
                raise BrowserError("browser failed before " + name + ": " + str(msg.get("error")))
        raise BrowserError("timed out waiting for " + name)

    def call(self, op: str, args: dict | None = None, timeout: float = 30):
        """(result, host ns sent, host ns of the reply)."""
        with self.lock:
            self.seq += 1
            ident = self.seq
            sent = time.monotonic_ns()
            self.proc.stdin.write(json.dumps({"id": ident, "op": op, "args": args or {}}) + "\n")
            self.proc.stdin.flush()
        with self.cv:
            if not self.cv.wait_for(lambda: ident in self.replies, timeout):
                raise BrowserError(op + " timed out")
            reply = self.replies.pop(ident)
        if not reply["ok"]:
            raise BrowserError(op + ": " + reply.get("error", "failed"))
        return reply["result"], sent, reply["host_ns"]

    def clock_sync(self, n: int = 9) -> dict:
        samples = []
        for _ in range(n):
            result, sent, received = self.call("clock", timeout=5)
            samples.append({"host_send_ns": sent, "host_reply_ns": received, "page_ms": result["page_ms"]})
        best = min(samples, key=lambda s: s["host_reply_ns"] - s["host_send_ns"])
        offset = (best["host_send_ns"] + best["host_reply_ns"]) / 2 - best["page_ms"] * 1e6
        return {"samples": samples, "offset_ns": round(offset),
                "uncertainty_ns": (best["host_reply_ns"] - best["host_send_ns"]) // 2,
                "method": "min-RTT round trip; host = time.monotonic_ns, page = performance.now"}

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"op": "close"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=20)
        except Exception:
            self.proc.kill()
        self.stderr.close()
