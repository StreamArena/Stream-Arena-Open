"""XMax X2.0 (realtime video editing) through the official JS SDK in headless Chromium.

Every released event becomes one ``session.set({prompt, refImageUrl})``; reference images are uploaded with the
SDK file client at dispatch time (the upload is part of the measured response). The account key stays in this
process; the page only receives a temporary key bounded in time and credits.

XMAX_REGION selects the Open API host and the matching SDK build: ``global`` (default, @xmaxai/sdk-global) or
``cn`` (@xmaxai/sdk). Both 1.1.13 builds are the same source and model catalog and differ only in the compiled
region (API/telemetry host, upload endpoint); the page imports @xmaxai/sdk-global and the cn run bundles it with
an esbuild alias.
"""
from __future__ import annotations

from streameval.providers.browser import installed
from streameval.providers.edit import READY_MAX_S, EditBackend

HOSTS = {"global": "https://api.xmax.ai/open/api/v1", "cn": "https://api.xmaxai.com/open/api/v1"}
SDKS = {"global": {"package": "@xmaxai/sdk-global", "version": "1.1.13",
                   "integrity": "sha512-xHwGiq+0/FesuFBnP+qozFO3DzzfbfKduc4/m5DAzechIRfOnx7ARu0J1tpU5OtNXe3CTKuKKzYPs03bc414fw=="},
        "cn": {"package": "@xmaxai/sdk", "version": "1.1.13",
               "integrity": "sha512-en9IAc9jY6MZdKF5CC4Os/DsvtaPF3EFwWhmPVIIjW23L0GsJ5QR9/Uvnp5vbCOw9MpfVGWwhRofMozoiQb4jQ=="}}
PAGE_IMPORT = "@xmaxai/sdk-global"   # module imported by streameval/providers/js/edit_page_xmax.mjs
# SDK 1.1.13: MAX_UPLOAD_VIDEO_PIXELS = 1_280_000 (sizes aligned to 32), RTC_PUBLISH_FPS_WEB = 30
PUSH = {"kind": "max_pixels", "max_pixels": 1_280_000, "align": 32, "max_fps": 30,
        "bitrate_kbps": {"high": 5000, "low": 3000}}
MODEL = "x2.0"
BALANCE_MARGIN = 120   # credits kept above one session's worst case before starting it


class XMax(EditBackend):
    page = "xmax_x2"
    push = PUSH
    credentials_env = {"api_key": "XMAX_API_KEY", "region": "XMAX_REGION"}
    cost = {"unit": "credits", "per_active_second": 1.0, "overhead_s": 30}

    def region(self) -> str:
        region = self.credentials().get("region") or "global"
        if region not in HOSTS:
            raise ValueError(f"unknown XMAX_REGION {region!r}; expected one of {sorted(HOSTS)}")
        return region

    def system_config(self, system_id: str, spec: dict) -> dict:
        region = self.region()
        config = super().system_config(system_id, spec)
        config["client"] = {**config["client"], "transport": "XMax JS SDK in headless Chromium (Playwright)",
                            "sdk": SDKS[region], "telemetry": False, "audio": {"publish": False, "subscribe": False}}
        return {**config, "model": MODEL, "region": region, "api_origin": HOSTS[region].rsplit("/open", 1)[0],
                "product": "client.realtime.connect(MediaStream) + session.set({prompt, refImageUrl})",
                "credential": "backend-minted temporary key (expireSeconds = duration + 480, pointsLimit = "
                              "duration + 120); the account key never reaches the browser",
                "session": ("connect with context {prompt, refImageUrl: uploaded reference A}, autoStart; epoch "
                            "after the first presented output frame" if spec["track"] == "reference_edit" else
                            "connect with autoStart=false; e000 at t=0 starts generation"),
                "reference_upload": ("client.files.uploadImage(released image) at dispatch; refImageUrl=null when "
                                     "the planned state has no reference" if spec["track"] == "reference_edit"
                                     else None),
                "source_push": {"policy": "native size if below 1.28 MP, else the largest proportional 32-aligned "
                                          "size below 1.28 MP (SDK MAX_UPLOAD_VIDEO_PIXELS); never upscaled (the "
                                          "SDK's own <0.6 MP upscale is bypassed by passing stream.width/height)",
                                "fps": "round(source fps) capped at 30 (SDK web publish rate)",
                                "max_kbps": "5000 if an edge >= 1600 px else 3000", "content_hint": "detail",
                                "pipeline": "canvas at the pushed size redrawn per decoded source frame; first "
                                            "frame held before the epoch", **PUSH},
                "stop": "session.stopGeneration + disconnect in page; backend DELETE /session/{uid} and GET "
                        "confirmation (status CLOSED)"}

    def expects_output_before_epoch(self, track: str) -> bool:
        # instruction cases start generating with their first event at t=0
        return track == "reference_edit"

    # ---- network -------------------------------------------------------------------------------------
    def api(self, method: str, endpoint: str, body=None) -> dict:
        import requests
        response = requests.request(method, HOSTS[self.region()] + endpoint, json=body, timeout=30,
                                    headers={"Content-Type": "application/json",
                                             "X-Api-Key": self.credentials()["api_key"]})
        response.raise_for_status()
        return response.json()

    def balance(self):
        value = (self.api("GET", "/balance").get("data") or {}).get("currentBalance")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("unexpected XMax balance response")
        return value

    def preflight(self, job) -> None:
        super().preflight(job)
        sdk = SDKS[self.region()]
        entry = installed(sdk["package"])
        if (entry.get("version"), entry.get("integrity")) != (sdk["version"], sdk["integrity"]):
            raise RuntimeError(f"{sdk['package']}@{sdk['version']} with the pinned integrity is not installed in "
                               "the browser runtime (npm ci in streameval/providers/js)")
        job.balance_before = self.balance()
        need = job.case["duration_ms"] // 1000 + BALANCE_MARGIN
        if job.balance_before < need:
            raise RuntimeError(f"XMax balance below {need} credits")

    def browser_launch(self, job) -> dict:
        package = SDKS[self.region()]["package"]
        return {} if package == PAGE_IMPORT else {"alias": {PAGE_IMPORT: package}}

    def mint(self, job) -> dict:
        seconds = job.case["duration_ms"] // 1000
        data = self.api("POST", "/temporary-api-key", {"expireSeconds": seconds + READY_MAX_S + 360,
                                                       "pointsLimit": seconds + 120})
        return {"key": data["data"]["temporaryApiKey"]}

    def connect_args(self, job, initial) -> dict:
        push = job.push
        context = None
        if initial is not None:
            context = {"prompt": initial.provider_prompt, "ref_asset": initial.after.reference.asset_id}
        return {"model": MODEL, "context": context, "auto_start": context is not None,
                "stream": {"width": push["width"], "height": push["height"], "fps": push["fps"],
                           "maxKbps": push["max_kbps"], "contentHint": "detail"}}

    def set_args(self, plan) -> dict:
        after, before = plan.after.reference, plan.before.reference
        return {"prompt": plan.provider_prompt, "ref_asset": after.asset_id if after else None,
                "clear_ref": after is None and before is not None}

    def close_receipt(self, job, connected, finished) -> dict:
        receipt = {"termination_confirmed": False, "provider_credits_cost": None,
                   "page_disconnect": (finished or {}).get("disconnected")}
        uid = (connected or {}).get("session_uid")
        if uid and not self.missing_credentials():
            try:
                data = self.api("DELETE", "/session/" + uid).get("data") or {}
                if data.get("status") != "CLOSED":
                    data = self.api("GET", "/session/" + uid).get("data") or {}
                receipt["close"] = {k: data.get(k) for k in ("status", "closeReason", "closeTimestamp")}
                receipt["termination_confirmed"] = data.get("status") == "CLOSED"
            except Exception as exc:
                receipt["close"] = {"error_type": type(exc).__name__}
        elif not uid:
            receipt["close"] = "not_created_or_unknown"
        before = getattr(job, "balance_before", None)
        if before is not None:
            try:
                receipt["provider_credits_cost"] = before - self.balance()
                receipt["credits_cost_basis"] = "account balance delta (exact only at concurrency 1)"
            except Exception:
                pass
        return receipt
