"""Durable case records: the missing "did this actually happen" layer.

A pipeline that returns a card to whoever called it is a function, not a
product. This module is what turns one investigation into a case with a
lifecycle: received, investigating, a terminal outcome, and (when the policy
in ``notify.py`` says so) a delivered notification that can be checked
independently later, not just trusted because the pipeline said so.

``CaseStore`` is a directory of one JSON file per case -- simple, zero AWS
dependency beyond what the rest of the pipeline already needs, and what every
offline test, local run, and the deployed agent all use. It is not durable
across AgentCore container recycling: files written inside the container are
not guaranteed to survive it. That tradeoff is accepted rather than worked
around here -- see "Known limitations" in the README.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path


class CaseStatus(str, Enum):
    RECEIVED = "received"
    INVESTIGATING = "investigating"
    COMPLETE = "complete"
    FAILED = "failed"


class NotificationChannel(str, Enum):
    NONE = "none"
    ADVISORY = "advisory"
    STANDARD = "standard"
    URGENT = "urgent"


@dataclass
class NotificationRecord:
    """Two different claims, deliberately not collapsed into one boolean.

    A suppressed case (channel=NONE) has a real, complete outcome — the
    notify/suppress decision was made and persisted — but nothing was ever
    sent to the user. Calling that "delivered" (the previous shape of this
    record) reads as if a notification went out when none did.
    """

    channel: NotificationChannel
    # The notify-or-suppress decision itself was made and this record
    # persisted -- true for every case that reaches a terminal outcome,
    # suppressed or not. This is what verify_delivered checks.
    decision_recorded: bool = False
    # An actual notification (standard or urgent) was sent -- false for a
    # suppressed case, since suppression means nothing was sent.
    notification_delivered: bool = False
    delivered_at: str | None = None
    detail: str | None = None


@dataclass
class CaseRecord:
    """One message's full lifecycle, from arrival to (maybe) a delivered alert."""

    case_id: str
    received_at: str
    status: CaseStatus
    message_text: str
    triage: dict | None = None
    card: dict | None = None
    ml_screening: dict | None = None
    notification: NotificationRecord | None = None
    error: str | None = None
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        if self.notification is not None:
            d["notification"]["channel"] = self.notification.channel.value
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "CaseRecord":
        d = dict(d)
        d["status"] = CaseStatus(d["status"])
        if d.get("notification") is not None:
            n = dict(d["notification"])
            n["channel"] = NotificationChannel(n["channel"])
            d["notification"] = NotificationRecord(**n)
        return cls(**d)


def new_case_id() -> str:
    return f"case-{uuid.uuid4().hex[:12]}"


class CaseStore:
    """A directory of one JSON file per case, keyed by case_id."""

    def __init__(self, base_dir: str | None = None) -> None:
        # A relative default (e.g. ".smishsentinel_data") resolves against the
        # process's cwd, which inside the deployed container is /app -- the
        # application's own source directory, not writable at runtime. The
        # system temp directory is reliably writable in every environment
        # this actually runs in, local or containerized.
        default_dir = str(Path(tempfile.gettempdir()) / "smishsentinel_data")
        self.base_dir = Path(base_dir or os.environ.get("SMISH_STORE_DIR", default_dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: str) -> Path:
        # case_id is always our own uuid-derived value (new_case_id()), never
        # caller-supplied, so this isn't a path-traversal boundary -- but the
        # check costs nothing and documents the invariant.
        if "/" in case_id or "\\" in case_id or ".." in case_id:
            raise ValueError(f"unsafe case_id: {case_id!r}")
        return self.base_dir / f"{case_id}.json"

    def save(self, record: CaseRecord) -> None:
        record.updated_at = datetime.now(UTC).isoformat()
        path = self._path(record.case_id)
        # Write-then-rename: a reader never observes a half-written file, and
        # a crash mid-write leaves the previous version intact rather than a
        # corrupt one. The tmp filename carries a random suffix, not just the
        # case_id, so two callers saving the same case_id in close succession
        # -- e.g. webui.py's request handler persisting a fresh RECEIVED
        # record just before the background thread it starts saves its own
        # -- never share one tmp file and clobber each other's write
        # mid-flight; each writes and renames its own.
        tmp_path = path.with_suffix(f".{uuid.uuid4().hex[:8]}.tmp")
        tmp_path.write_text(json.dumps(record.to_json_dict(), indent=2), encoding="utf-8")

        # os.replace (MoveFileExW on Windows) can transiently raise
        # PermissionError/WinError 5 when the destination was just created
        # a moment ago and is still momentarily held by real-time antivirus
        # or search indexing -- observed in practice from exactly the
        # concurrent-save pattern above. Retried briefly rather than treated
        # as fatal: it clears within milliseconds, and this is a local file,
        # not a real permissions problem. A no-op on platforms that don't
        # exhibit this (the first attempt always succeeds there).
        for attempt in range(5):
            try:
                tmp_path.replace(path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02 * (attempt + 1))

    def get(self, case_id: str) -> CaseRecord | None:
        path = self._path(case_id)
        if not path.exists():
            return None
        return CaseRecord.from_json_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_recent(self, limit: int = 20) -> list[CaseRecord]:
        files = sorted(self.base_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [
            CaseRecord.from_json_dict(json.loads(p.read_text(encoding="utf-8")))
            for p in files[:limit]
        ]


