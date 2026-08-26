"""Durable case records: the missing "did this actually happen" layer.

A pipeline that returns a card to whoever called it is a function, not a
product. This module is what turns one investigation into a case with a
lifecycle: received, investigating, a terminal outcome, and (when the policy
in ``notify.py`` says so) a delivered notification that can be checked
independently later, not just trusted because the pipeline said so.

Two backends, same interface (``save`` / ``get`` / ``list_recent``), so
nothing above this module needs to know which is active:

- ``CaseStore`` — a directory of one JSON file per case. Simple, zero AWS
  dependency, what every offline test and local run uses. Not durable across
  AgentCore container recycling: files written inside the container are not
  guaranteed to survive it.
- ``DynamoDBCaseStore`` — the production swap the interface above was always
  meant to allow. Persists independently of the container, at the cost of
  needing a real table and an execution role permitted to use it (see
  ``docs/agentcore-execution-role-dynamodb-policy.json`` and README's
  "Running it").

``get_case_store()`` picks between them based on whether ``SMISH_CASE_TABLE``
is set, and is what ``inbox.py``, ``app.py``, and ``webui.py`` actually call.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any


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


def _decimals_to_native(value: Any) -> Any:
    """DynamoDB returns numbers as Decimal, not int/float -- CaseRecord has
    no numeric fields today, but converting defensively means a stray number
    anywhere in a nested dict (triage/card, both free-form JSON dicts from
    Pydantic) can never silently produce a Decimal where JSON serialization
    or an equality check expects a plain number."""
    if isinstance(value, list):
        return [_decimals_to_native(v) for v in value]
    if isinstance(value, dict):
        return {k: _decimals_to_native(v) for k, v in value.items()}
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    return value


class DynamoDBCaseStore:
    """The production persistence swap ``CaseStore``'s interface was always
    meant to allow -- see this module's docstring. Same
    save/get/list_recent contract; nothing above this class needs to know
    which backend is active.

    Deliberately simple at this submission's scale: ``list_recent`` does a
    full table ``Scan`` and sorts client-side by ``updated_at`` rather than
    requiring a GSI. That's fine for a demo inbox of a handful of cases and
    wrong for a production-scale table -- stated here rather than left
    implicit, the same way ``CaseStore``'s own limitations are.

    The execution role this runs under needs its own DynamoDB permissions;
    AgentCore's auto-created role does not include them by default (Bedrock,
    logs, and X-Ray only). See
    ``docs/agentcore-execution-role-dynamodb-policy.json``.
    """

    def __init__(
        self,
        table_name: str | None = None,
        *,
        resource: Any = None,
        region_name: str | None = None,
    ) -> None:
        table_name = table_name or os.environ.get("SMISH_CASE_TABLE")
        if not table_name:
            raise ValueError(
                "DynamoDBCaseStore requires a table name -- pass table_name "
                "or set SMISH_CASE_TABLE."
            )
        self.table_name = table_name
        if resource is None:
            import boto3

            resource = boto3.resource(
                "dynamodb", region_name=region_name or os.environ.get("AWS_REGION", "us-east-1")
            )
        self._table = resource.Table(table_name)

    def save(self, record: CaseRecord) -> None:
        record.updated_at = datetime.now(UTC).isoformat()
        self._table.put_item(Item=record.to_json_dict())

    def get(self, case_id: str) -> CaseRecord | None:
        response = self._table.get_item(Key={"case_id": case_id})
        item = response.get("Item")
        if item is None:
            return None
        return CaseRecord.from_json_dict(_decimals_to_native(item))

    def list_recent(self, limit: int = 20) -> list[CaseRecord]:
        items: list[dict] = []
        response = self._table.scan()
        items.extend(response.get("Items", []))
        while "LastEvaluatedKey" in response:
            response = self._table.scan(ExclusiveStartKey=response["LastEvaluatedKey"])
            items.extend(response.get("Items", []))

        items.sort(key=lambda i: i.get("updated_at", ""), reverse=True)
        return [CaseRecord.from_json_dict(_decimals_to_native(i)) for i in items[:limit]]


def get_case_store() -> "CaseStore | DynamoDBCaseStore":
    """The store this process should actually use, chosen once per call
    rather than hardcoded by any caller: ``DynamoDBCaseStore`` when
    ``SMISH_CASE_TABLE`` is set (the real production path -- persistence
    surviving AgentCore container recycling, which ``CaseStore`` cannot
    offer), the local JSON ``CaseStore`` otherwise (every offline test, and
    local development without a table). ``inbox.py``, ``app.py``, and
    ``webui.py`` all call this rather than constructing a backend directly,
    so the same code runs unchanged against either.
    """
    table_name = os.environ.get("SMISH_CASE_TABLE")
    if table_name:
        return DynamoDBCaseStore(table_name=table_name)
    return CaseStore()
