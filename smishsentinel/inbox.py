"""The end-to-end action: a synthetic inbox trigger through to a verified,
persisted, delivered-or-suppressed outcome.

A pipeline that returns a card to whoever called it is a function; running
this is what makes it an agent that does something end to end. There is no
real inbox to watch here — no phone, no carrier integration, nothing this
submission is claiming to have built — so the trigger is a small, fixed set
of synthetic messages standing in for "new messages arrived." Everything
after that point is real: each message gets a real case ID, moves through
real status transitions, is investigated by the real pipeline, is persisted
via the real store, and has a real notify-or-suppress decision made and
recorded against it. run_inbox_cycle() is what a demo actually runs.

One message in the fixed set (SUSPICIOUS_DEAD_LINK below) exists specifically
to exercise the evidence-collection-failure path deterministically, since
that is the path most likely to be handled wrong: a source that could not be
checked is not the same thing as a source that contradicts the message, and
the synthesis prompt now says so explicitly (see agent.py) — but a rule in a
prompt is only worth as much as the case that actually exercises it.
"""

from __future__ import annotations

import time

from .agent import investigate
from .config import INBOX_CYCLE_DEADLINE_SECONDS
from .notify import decide, deliver
from .schemas import TriageResult
from .store import (
    CaseRecord,
    CaseStatus,
    CaseStore,
    DynamoDBCaseStore,
    NotificationChannel,
    get_case_store,
    new_case_id,
)

SYNTHETIC_INBOX: list[str] = [
    # Ordinary, no claimed organization, no consequential action -- the
    # quiet path should be the outcome for both of these.
    "Hey are we still meeting for coffee at 10 tomorrow?",
    "Your package from Purolator has been delivered to your front door.",
    # Claimed brand, credential/payment pressure, hostname mismatch -- should
    # investigate and land on a real, cited verdict.
    "RBC Royal Bank: We noticed unusual sign-in activity on your account. "
    "Verify now or it will be suspended in 24 hours: "
    "http://rbc-online-secure.info/verify",
    # No link at all -- requested action is a phone call, tests the pipeline
    # against a case with nothing to fetch regardless of domain-lock status.
    "CRA NOTICE: You owe $487.32 in unpaid taxes. Call 1-647-555-0199 "
    "immediately to avoid legal action and a warrant for your arrest.",
    # Deliberately unreachable claimed link -- exercises the
    # evidence-collection-failure path specifically. Correct behaviour is an
    # honestly hedged verdict driven by the real behavioural signals present
    # (hostname mismatch, urgency), not one that treats the dead link itself
    # as proof of anything.
    "Amazon: Your order #4471 payment failed. Update your payment method "
    "within 12 hours to avoid cancellation: "
    "http://amazon-billing-update-portal-2026.info/pay",
]


def investigate_one_message(
    message: str,
    store: CaseStore | DynamoDBCaseStore,
    *,
    case_id: str | None = None,
) -> CaseRecord:
    """Run a single message through the full lifecycle: received ->
    investigating -> complete/failed, persisted at every stage transition
    rather than only at the end.

    This is the per-message body ``run_inbox_cycle`` loops over, pulled out
    so a caller that already has one specific message in hand -- webui.py's
    "investigate this message" form, for instance -- gets the exact same
    real lifecycle, persistence, and notify-or-suppress decision without
    faking up a one-item inbox to get it.
    """
    record = CaseRecord(
        case_id=case_id or new_case_id(),
        received_at=_now(),
        status=CaseStatus.RECEIVED,
        message_text=message,
    )
    store.save(record)

    record.status = CaseStatus.INVESTIGATING
    store.save(record)

    try:
        result = investigate(message)
    except Exception as exc:  # noqa: BLE001 - a failed case is a real, visible outcome, not a crash
        record.status = CaseStatus.FAILED
        record.error = f"{type(exc).__name__}: {exc}"
        # A failure that never notifies anyone is worse than a wrong
        # verdict -- the user is left thinking the message was checked
        # when it silently wasn't. Treated as urgent for the same reason
        # notify.decide treats a missing card as urgent: a defensive
        # default matters more than an elegant one.
        record.notification = deliver(record, NotificationChannel.URGENT)
        store.save(record)
        return record

    triage: TriageResult = result["triage"]
    card = result["card"]

    record.triage = triage.model_dump(mode="json")
    record.card = card.model_dump(mode="json") if card else None

    channel = decide(triage, card)
    record.notification = deliver(record, channel)
    record.status = CaseStatus.COMPLETE
    store.save(record)
    return record


def run_inbox_cycle(
    messages: list[str] | None = None,
    store: CaseStore | DynamoDBCaseStore | None = None,
    *,
    deadline_seconds: float = INBOX_CYCLE_DEADLINE_SECONDS,
) -> list[CaseRecord]:
    """Run every message in the synthetic inbox through the full lifecycle.

    Each message becomes a case that is persisted at every stage, not just at
    the end, so a case that fails partway through leaves a real record of
    where it got to rather than silently vanishing. Returns the finalized
    records; ``notify.verify_delivered`` is the independent check that what
    this function claims to have done actually landed in the store.

    Per-stage budgets in config.py bound one message's model calls; they
    don't bound the whole cycle. ``deadline_seconds`` is that missing ceiling
    -- checked before each message starts, not mid-investigation, since
    Strands agents don't expose a way to preempt a call already in flight.
    Once the deadline has passed, every remaining message is failed without
    ever calling the model, with a real, persisted, notified record -- not
    silently dropped and not run anyway.
    """
    store = store or get_case_store()
    records: list[CaseRecord] = []
    start = time.monotonic()

    for message in messages if messages is not None else SYNTHETIC_INBOX:
        if time.monotonic() - start > deadline_seconds:
            record = CaseRecord(
                case_id=new_case_id(),
                received_at=_now(),
                status=CaseStatus.RECEIVED,
                message_text=message,
            )
            store.save(record)
            record.status = CaseStatus.FAILED
            record.error = (
                f"Cycle deadline of {deadline_seconds}s exceeded before this "
                "message could be investigated; skipped without a model call."
            )
            record.notification = deliver(record, NotificationChannel.URGENT)
            store.save(record)
            records.append(record)
            continue

        records.append(investigate_one_message(message, store))

    return records


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
