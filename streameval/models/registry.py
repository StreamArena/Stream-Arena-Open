"""Model registry: one ``system.json`` per model declares what it is, which tracks it runs and how to start it.

Searched in ``models/selfhost/*/system.json``, ``models/providers/*/system.json``, ``examples/*/system.json``, the directories or files listed in
STREAMEVAL_MODELS (``os.pathsep``-separated) and any path given with ``--registry``. A file holds one entry or a
list of entries. Minimal entries:

    in-process Python model                         model server (any language, docs/protocol.md)
    {"system_id": "my_model",                       {"model": "flashhead", "system_id": "soulx_flashhead",
     "model_id": "my_model", "name": "My Model",     "model_id": "soulx_flashhead", "name": "SoulX FlashHead",
     "provider": "My Lab",                           "provider": "Soul AI Lab", "system_kind": "native",
     "system_kind": "native",                        "tracks": ["digital_human_audio"],
     "tracks": ["instruction_edit"],                 "portrait_policy": "benchmark_portrait",
     "portrait_policy": null,                        "server": {"cmd": "python server.py --port {port}",
     "python": "model.py:MyModel"}                              "port": 8765}}

``python``: ``module:Class`` or ``file.py:Class`` (relative to the system.json), constructed with ``kwargs``.
``server``: launch command (``--launch``) with placeholders {port} {python} {model_dir} {weights} {env}; weights and
env live under STREAMEVAL_MODEL_ROOT/<model>/ (default ~/.cache/streameval/models). ``url``: an already running server.
``backend``: a ``streameval.providers.base.SessionBackend`` class (``file.py:Class``, constructed with ``kwargs``)
that runs whole sessions with its own media stack (a vendor's browser SDK, say) and writes the run contract itself
(docs/protocol.md); models/providers/ holds the vendor backends. A vendor system (a backend, or a ``python``
model class that also inherits ``streameval.providers.account.Account``) declares its credentials, cost and
session cap; ``account(entry)`` returns it without making any request.

Optional keys: ``ops`` (operations the model supports, default every operation of its tracks), ``unsupported_ops``,
``output`` {width, height, fps}, ``push`` {max_side, max_fps} for the edited source, ``composition``
{input: audio|text, audio_out: client|server, asr, llm, tts, tts_voice} for renderer-only avatars, ``seed``,
``input_adapters``, ``model_version``, ``weights``, ``steps``, ``gpu``, ``system_ids`` {track: id}, ``notes``.
Every public key is frozen into the system configuration recorded with each run. Per-track system ids are
``<system_id>_<suffix>`` (tracks.TRACKS suffix) unless ``system_ids`` names them.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import time

from .. import REPO
from ..benchmark.tracks import OPTIONAL_OPS, SAMPLE_MS, TRACK_OPS, TRACKS
from ..config import env
from ..evaluation import input_contract, input_condition, policy, policy_binding
from ..util import digest

KINDS = ("native", "composition")
PORTRAIT_POLICIES = ("benchmark_portrait", "provider_stock", None)
REQUIRED = ("system_id", "model_id", "name", "provider", "system_kind", "tracks", "portrait_policy")
DEFAULT_OUTPUT = {"width": 512, "height": 512, "fps": 25}
DEFAULT_PUSH = {"max_side": 1280, "max_fps": 30}
DEFAULT_SEED = 20260929
READY_TIMEOUT_S = policy()["execution"]["ready_timeout_s"]


def validate(entry: dict, where: str = "") -> dict:
    missing = [k for k in REQUIRED if k not in entry]
    if missing:
        raise ValueError(f"{where}: registry entry lacks {missing}")
    if entry["system_kind"] not in KINDS:
        raise ValueError(f"{where}: system_kind must be one of {KINDS}")
    unknown = [t for t in entry["tracks"] if t not in TRACKS]
    if unknown or not entry["tracks"]:
        raise ValueError(f"{where}: unknown tracks {unknown}; choose from {sorted(TRACKS)}")
    if entry["portrait_policy"] not in PORTRAIT_POLICIES:
        raise ValueError(f"{where}: portrait_policy must be one of {PORTRAIT_POLICIES}")
    if entry["system_kind"] == "composition":
        bad = [t for t in entry["tracks"] if TRACKS[t]["kind"] != "avatar"]
        if bad:
            raise ValueError(f"{where}: composition systems are renderer-only avatars; not {bad}")
    if sum(k in entry for k in ("python", "server", "url", "backend")) != 1:
        raise ValueError(f"{where}: give exactly one of python (in-process class), server (launch command), url or "
                         "backend (session backend class)")
    if "server" in entry and ("cmd" not in entry["server"] or "port" not in entry["server"]):
        raise ValueError(f"{where}: server needs cmd and port")
    for op in entry.get("ops") or []:
        if not any(op in TRACK_OPS[t] + OPTIONAL_OPS.get(t, []) for t in entry["tracks"]):
            raise ValueError(f"{where}: op {op!r} belongs to none of the registered tracks")
    if entry.get("ready_timeout_s", READY_TIMEOUT_S) != READY_TIMEOUT_S:
        raise ValueError("readiness timeout is fixed by the shared evaluation policy")
    for track in entry["tracks"]:
        input_contract(entry, track)
    return entry


def _entries(path: Path) -> list[dict]:
    data = json.loads(path.read_text())
    return [validate(e, str(path)) for e in (data if isinstance(data, list) else [data])]


def search_paths(extra=()) -> list[Path]:
    files = [f for d in ("models/selfhost", "models/providers", "examples") for f in sorted((REPO / d).glob("*/system.json"))]
    for item in [*(env("STREAMEVAL_MODELS") or "").split(os.pathsep), *map(str, extra)]:
        if not item:
            continue
        p = Path(item).expanduser()
        files += [p] if p.is_file() else sorted(p.glob("*/system.json")) + sorted(p.glob("system.json"))
    seen, out = set(), []
    for f in files:
        if f.resolve() not in seen:
            seen.add(f.resolve())
            out.append(f)
    return out


def models(extra=()) -> dict[str, dict]:
    """system_id prefix -> entry (with the private key _dir: the folder of its system.json)."""
    found = {}
    for path in search_paths(extra):
        for entry in _entries(path):
            key = entry["system_id"]
            if key in found:
                raise SystemExit(f"duplicate system_id {key!r}: {path} and {found[key]['_dir']}/system.json both "
                                 "register it; give your copy its own system_id")
            found[key] = {**entry, "_dir": str(path.parent.resolve())}
    return found


def system_ids(entry: dict) -> dict[str, str]:
    """track -> system_id for one entry."""
    explicit = entry.get("system_ids") or {}
    return {t: explicit.get(t, f"{entry['system_id']}_{TRACKS[t]['suffix']}") for t in entry["tracks"]}


def systems(extra=()) -> dict[str, dict]:
    """system_id -> {"entry", "track", **TRACKS[track]} for every registered track."""
    table = {}
    for entry in models(extra).values():
        for track, sid in system_ids(entry).items():
            if sid in table:
                raise SystemExit(f"duplicate system_id {sid!r}: registered by {entry['_dir']} and "
                                 f"{table[sid]['entry']['_dir']}")
            table[sid] = {"entry": entry, "track": track, **TRACKS[track]}
    return table


def spec_of(system_id: str, extra=()) -> dict:
    table = systems(extra)
    if system_id not in table:
        raise SystemExit(f"unknown system {system_id}; registered: {sorted(table)}")
    return table[system_id]


def adhoc(track: str, *, python: str | None = None, url: str | None = None, name: str = "adhoc") -> tuple[str, dict]:
    """(system_id, spec) of an unregistered model given on the command line (--model / --url with --track)."""
    if track not in TRACKS:
        raise SystemExit(f"unknown track {track}; choose from {sorted(TRACKS)}")
    entry = {"system_id": name, "model_id": name, "name": name, "provider": "unregistered", "system_kind": "native",
             "tracks": [track], "portrait_policy": "benchmark_portrait" if TRACKS[track]["kind"] == "avatar" else None,
             **({"python": python} if python else {"url": url})}
    entry = {**validate(entry, "command line"), "_dir": str(Path.cwd())}
    return system_ids(entry)[track], {"entry": entry, "track": track, **TRACKS[track]}


def ops(entry: dict, track: str) -> list[str]:
    base = entry.get("ops")
    if base is None:
        base = TRACK_OPS[track]
    drop = set(entry.get("unsupported_ops") or [])
    return [o for o in base if o not in drop and o in TRACK_OPS[track] + OPTIONAL_OPS.get(track, [])]


def composition(entry: dict) -> dict | None:
    if entry["system_kind"] != "composition":
        return None
    comp = entry.get("composition") or {}
    return {"input": comp.get("input", "audio"), "audio_out": comp.get("audio_out", "client")}


def public(entry: dict) -> dict:
    return {k: v for k, v in entry.items() if not k.startswith("_")}


def system_config(system_id: str, spec: dict) -> dict:
    """The frozen configuration of one system on one track, recorded in every binding.json and batch plan."""
    from ..runtime.dialogue import components
    entry, track, kind, group = spec["entry"], spec["track"], spec["kind"], spec["group"]
    pub = public(entry)
    transport = ("in-process Python (" + entry["python"] + ")" if "python" in entry
                 else "session backend (" + entry["backend"] + ")" if "backend" in entry
                 else "WebSocket streameval-selfhost/1 (streameval.runtime.protocol)")
    config = {
        "system_id": system_id, "model_id": entry["model_id"], "name": entry["name"], "provider": entry["provider"],
        "group": group, "input_profile": spec["input_profile"], "track": track,
        "system_kind": entry["system_kind"], "portrait_policy": entry["portrait_policy"] if kind == "avatar" else None,
        "components": components(entry), "session_cap_ms": None,
        "registry_entry": pub, "registry_entry_sha256": digest(pub),
        "ops": ops(entry, track),
        "model_version": entry.get("model_version"), "weights": entry.get("weights"), "steps": entry.get("steps"),
        "gpu": entry.get("gpu"), "output": entry.get("output") or DEFAULT_OUTPUT,
        "source_push": ({"policy": "source resolution scaled down to push.max_side (never upscaled), "
                                   "fps = min(round(source fps), push.max_fps); first frame held before the epoch; "
                                   "ffmpeg mjpeg decode, one JPEG per frame on an absolute clock",
                         **DEFAULT_PUSH, **(entry.get("push") or {})} if kind == "editing" else None),
        "seed": {"base": entry.get("seed", DEFAULT_SEED), "per_session": "base + repeat"},
        "client": {"transport": transport,
                   "capture": "first received frame per activity bin (receive time, no display); audio_out 48 kHz "
                              "mono int16 with online VAD",
                   "audio_in": "16 kHz mono s16le, 10 ms chunks on an absolute clock, silence between clips"},
        "readiness": ("ready + first frame, then first frame >= 4 s old and 1500 ms without speech (at most 25 s)"
                      if kind == "avatar" else "ready + first frame"),
        "unsupported": "events outside `ops` are released, recorded as dispatch status 'unsupported' without a model "
                       "call; an ack 'unsupported' is recorded the same way; both are scored as failures",
        "sampling": {"frame_sample_ms": SAMPLE_MS[group], "frame_selection": "first received frame per activity bin"},
    }
    vendor = account(entry)
    if vendor is not None:
        # a vendor adapter describes its own client (transport, capture, readiness, source push, session cap)
        config.update(vendor.system_config(system_id, spec))
        config["client"] = {"transport": transport, **config["client"]}
        if vendor.session_cap_s is not None and config["session_cap_ms"] is None:
            config["session_cap_ms"] = int(vendor.session_cap_s * 1000)
    contract = input_contract(entry, track)
    config.update(evaluation_policy=policy_binding(), ready_timeout_s=READY_TIMEOUT_S, input_contract=contract,
                  input_condition=input_condition(contract))
    return config


# ---- building a model -------------------------------------------------------------------------------

def load_class(spec: str, base: Path | None = None):
    """``pkg.module:Class`` or ``path/to/file.py:Class`` (relative to base)."""
    target, _, name = spec.rpartition(":")
    if not target or not name:
        raise ValueError(f"expected module:Class or file.py:Class, got {spec!r}")
    if target.endswith(".py"):
        path = (Path(base or ".") / target).resolve()
        module_name = "streameval_user_" + digest(str(path))[:12]
        if module_name not in sys.modules:
            spec_ = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec_)
            sys.modules[module_name] = module
            spec_.loader.exec_module(module)
        module = sys.modules[module_name]
    else:
        module = importlib.import_module(target)
    return getattr(module, name)


def build_backend(entry: dict):
    """The SessionBackend of a ``backend`` entry."""
    return load_class(entry["backend"], Path(entry.get("_dir", ".")))(**(entry.get("kwargs") or {}))


def account(entry: dict):
    """The Account of a vendor system (providers/account.py): its backend, or its python model when that class is
    an Account; None for every other system. Constructing one makes no request."""
    from ..providers.account import Account
    if "backend" in entry:
        return build_backend(entry)
    if "python" in entry:
        cls = load_class(entry["python"], Path(entry.get("_dir", ".")))
        if isinstance(cls, type) and issubclass(cls, Account):
            return cls(**(entry.get("kwargs") or {}))
    return None


def server_url(entry: dict, port: int | None = None) -> str:
    return entry.get("url") or f"ws://127.0.0.1:{port or entry['server']['port']}/session"


def build_model(entry: dict, url: str | None = None):
    """A fresh StreamingModel for one session."""
    if "python" in entry and url is None:
        return load_class(entry["python"], Path(entry.get("_dir", ".")))(**(entry.get("kwargs") or {}))
    from .websocket import WebSocketModel
    return WebSocketModel(url or server_url(entry))


# ---- launching a registered server ------------------------------------------------------------------

def model_root() -> Path:
    return Path(env("STREAMEVAL_MODEL_ROOT", "~/.cache/streameval/models")).expanduser()


def resolve_cmd(entry: dict, port: int | None = None, python: str | None = None) -> tuple[list[str], Path]:
    model_dir = Path(entry["_dir"])
    root = model_root() / entry.get("model", entry["system_id"])
    values = {"port": port or entry["server"]["port"], "model_dir": str(model_dir), "weights": str(root / "weights"),
              "env": str(root / "env"), "python": python or "python"}
    tokens = [t.format(**values) for t in shlex.split(entry["server"]["cmd"])]
    if python and tokens and tokens[0] == "python":
        tokens[0] = python
    return tokens, model_dir / entry["server"].get("cwd", ".")


def health(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> str | None:
    """'ok' when GET /health answers 200, 'open' when the port accepts connections otherwise, None when closed."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.settimeout(timeout)
            s.sendall(f"GET /health HTTP/1.1\r\nHost: {host}:{port}\r\nConnection: close\r\n\r\n".encode())
            head = s.recv(64)
            return "ok" if head.startswith(b"HTTP/1.") and b" 200 " in head[:16] else "open"
    except OSError:
        return None


def wait_health(port: int, timeout_s: float, proc: subprocess.Popen | None = None) -> str:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode} before it was healthy")
        state = health(port)
        if state:
            return state
        time.sleep(0.5)
    raise TimeoutError(f"server on port {port} not healthy after {timeout_s:g} s")


class LaunchedServer:
    """Start a registered server command locally and wait for GET /health; stop() terminates it."""

    def __init__(self, entry: dict, port: int | None = None, log: Path | None = None):
        self.port = port or entry["server"]["port"]
        tokens, cwd = resolve_cmd(entry, self.port, env("STREAMEVAL_SERVER_PYTHON") or sys.executable)
        self.log = open(log, "w") if log else subprocess.DEVNULL
        self.proc = subprocess.Popen(tokens, cwd=str(cwd), stdout=self.log, stderr=subprocess.STDOUT)
        try:
            wait_health(self.port, float(entry["server"].get("health_timeout_s", 900)), self.proc)
        except BaseException:
            self.stop()
            raise

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/session"

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.log is not subprocess.DEVNULL:
            self.log.close()
