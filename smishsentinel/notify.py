"""Deterministic notify-vs-suppress policy, delivery, and verification.

Three separate responsibilities that must not blur together:

1. ``decide`` — a pure function from (triage, card) to a channel. No model
   call, no judgment beyond a fixed rule table, because the one thing worse
   than a bad policy is a policy nobody can audit.
2. ``deliver`` — actually carries out that decision and records that it did.
3. ``verify_delivered`` — answers "did this case's user-facing action really
   happen" from the persisted record, independent of what any pipeline stage
   claimed in the moment.

What "delivery" means here is scoped honestly to what a backend-only
submission can actually do: there is no phone in this loop, so there is no
push notification. Delivery means writing a durable, queryable notification
record and emitting a clear log line — the equivalent signal a real channel
integration (SNS, a mobile push service) would consume downstream. Claiming
more than that would be describing a product this submission does not build.
"""

from __future__ import annotations

from datetime import UTC, datetime

from .schemas import EvidenceCard, RiskLevel, TriageResult
from .store import CaseRecord, NotificationChannel, NotificationRecord


def decide(triage: TriageResult, card: EvidenceCard | None) -> NotificationChannel:
    """Fixed rule table, not a judgment call.

    - No investigation was warranted: the quiet path is correct and common,
      by design (see agent.py's triage prompt) -- suppress.
    - Investigated and the card is missing (should not happen if the
      pipeline ran to completion, but a defensive default matters more than
      an elegant one): treat as urgent rather than silently dropping it.
    - Investigated and risk_level is high: urgent.
    - Investigated at all, any other outcome: standard. There is no
      "verified safe" branch here, deliberately -- see EvidenceCard's own
      invariant that no verdict means safety. Every investigated case
      reaches the user in some form; the only question is how loudly.
    """
    if not triage.warrants_investigation:
        return NotificationChannel.NONE
    if card is None:
        return NotificationChannel.URGENT
    if card.risk_level == RiskLevel.HIGH:
        return NotificationChannel.URGENT
    return NotificationChannel.STANDARD


def deliver(record: CaseRecord, channel: NotificationChannel) -> NotificationRecord:
    """Carry out the decision and produce the record that proves it happened.

    The "delivery" itself is a log line plus the returned record -- see the
    module docstring for why that is the honest scope of "delivered" here.
    """
    if channel == NotificationChannel.NONE:
        notification = NotificationRecord(channel=channel, delivered=True, detail="suppressed")
        print(f"[notify] case={record.case_id} SUPPRESSED (no investigation warranted)")
        return notification

    headline = record.card["headline"] if record.card else "Investigation could not complete."
    detail = f"{channel.value.upper()}: {headline}"
    print(f"[notify] case={record.case_id} {detail}")

    return NotificationRecord(
        channel=channel,
        delivered=True,
        delivered_at=datetime.now(UTC).isoformat(),
        detail=detail,
    )


def verify_delivered(record: CaseRecord | None) -> bool:
    """Ground truth for "did the user-facing action actually occur" -- reads
    the persisted record rather than trusting an in-memory return value that
    could reflect a step that ran but never actually got written down."""
    return (
        record is not None
        and record.status.value == "complete"
        and record.notification is not None
        and record.notification.delivered
    )
