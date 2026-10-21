"""Account: the credentials, cost and session limits of a vendor system, shared by both kinds of vendor adapter.

A vendor system is either a SessionBackend (providers/base.py, B layer) or a StreamingModel that also inherits
Account (A layer, e.g. the RTC avatars of providers/rtc.py). Either way the batch runner reads the same facts:

    missing_credentials()   environment variable names a session needs; a run is refused (exit 3) without them
    estimate(case)          the estimated cost of one session of `case` in cost["unit"] (None: unknown)
    session_cap_s           the longest session the vendor allows (None: no known cap); longer cases are kept and
                            fail in the denominator
    system_config(...)      keys of the frozen system configuration the adapter fixes, merged over the registry's

Credentials come from environment variables only (a git-ignored .env works too) and are never written to an
artifact. Constructing an Account makes no request.
"""
from __future__ import annotations

from ..config import env

# the shortest wait from the first output frame to the activity epoch of an avatar (runtime.harness readiness)
READINESS_MIN_S = 4.0 + 1.5


class Account:
    """Credentials, cost estimate and session cap of a vendor system. Mixed into every vendor adapter
    (SessionBackend, RtcAvatar), which overrides the class attributes below; read by the batch runner."""

    # credential field -> environment variable; required_credentials must all be set to run a session
    credentials_env: dict[str, str] = {}
    required_credentials: tuple[str, ...] = ("api_key",)
    # cost estimate of one session: (case seconds + overhead_s) * per_active_second, in unit; None: unknown
    cost: dict = {"unit": None, "per_active_second": None, "overhead_s": 0}
    session_cap_s: float | None = None

    def credentials(self) -> dict:
        """Account values from the environment (never recorded)."""
        return {field: env(var) for field, var in self.credentials_env.items()}

    def missing_credentials(self) -> list[str]:
        """Names of the unset environment variables a session needs."""
        values = self.credentials()
        return [self.credentials_env[f] for f in self.required_credentials if not values.get(f)]

    def estimate(self, case: dict) -> float | None:
        rate = self.cost.get("per_active_second")
        if rate is None:
            return None
        return (case["duration_ms"] / 1000 + self.cost.get("overhead_s", 0)) * rate

    def exceeds_session_cap(self, case: dict, kind: str) -> bool:
        """The case cannot finish inside the vendor's session cap (avatars also spend the readiness wait in it)."""
        if self.session_cap_s is None:
            return False
        return case["duration_ms"] / 1000 + (READINESS_MIN_S if kind == "avatar" else 0) > self.session_cap_s

    def system_config(self, system_id: str, spec: dict) -> dict:
        """Keys of the frozen system configuration this adapter fixes (client, readiness, session cap, ...); merged
        over the registry's defaults."""
        return {}
