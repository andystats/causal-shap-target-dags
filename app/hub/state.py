"""Stage lifecycle: one immutable result per stage, staleness by fingerprint.

A stage result remembers the fingerprint of the inputs that produced it. When
the live inputs drift (new data, new graph, a moved slider), the stored result
stops matching and the UI shows a stale chip — it never recomputes on its own,
because four expensive downstream stages re-running on every slider twitch is
worse than an amber badge.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

EMPTY = "empty"
RUNNING = "running"
OK = "ok"
ERROR = "error"

STAGES = ("data", "naive", "discover", "flags", "surgery", "attribute", "policy")


@dataclass(frozen=True)
class StageResult:
    status: str = EMPTY
    payload: object | None = None
    fingerprint: str = ""       # of the inputs that produced this result
    error: str = ""
    traceback: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == OK

    def stale_against(self, live_fingerprint: str) -> bool:
        return self.ok and self.fingerprint != live_fingerprint


def fingerprint(*parts: object) -> str:
    payload = "|".join(repr(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
