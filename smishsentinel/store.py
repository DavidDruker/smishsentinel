"""Durable case records: the missing "did this actually happen" layer.

A pipeline that returns a card to whoever called it is a function, not a
product. This module is what turns one investigation into a case with a
lifecycle: received, investigating, a terminal outcome, and (when the policy
in ``notify.py`` says so) a delivered notification that can be checked
independently later, not just trusted because the pipeline said so.

Backend: a directory of one JSON file per case. This is an honest,
deliberately small choice for a hackathon-scoped submission, not a production
design — the interface below (``save`` / ``get`` / ``list_recent``) is the
seam a real deployment would swap for DynamoDB or S3 without touching any
caller. Documented as a known limitation rather than left implicit: files
written inside an AgentCore Runtime container are not guaranteed to survive
container recycling, so this persists within a running session, not
durably across deployments.
"""

from __future__ import annotations

import json
import os
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
    STANDARD = "standard"
    URGENT = "urgent"


@dataclass
class NotificationRecord:
    channel: NotificationChannel
    delivered: bool = False
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
        self.base_dir = Path(base_dir or os.environ.get("SMISH_STORE_DIR", ".smishsentinel_data"))
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
        # corrupt one.
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(record.to_json_dict(), indent=2), encoding="utf-8")
        tmp_path.replace(path)

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
