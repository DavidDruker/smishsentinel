"""End-to-end smoke test against live Bedrock.

Runs two messages through triage: one that should be investigated and one that
should not. Proves the agent wiring, structured output, and the quiet path all
work before any of the expensive stages are trusted.

Usage:  python smoke_test.py
"""

from __future__ import annotations

import sys

from smishsentinel.agent import build_triage_agent
from smishsentinel.config import TRIAGE_BUDGET, TRIAGE_MODEL
from smishsentinel.safety import wrap_untrusted
from smishsentinel.schemas import TriageResult

SUSPICIOUS = (
    "Canada Post: Your parcel is held pending an unpaid redelivery fee of "
    "$2.99. Pay within 24 hours to avoid return to sender: "
    "http://canadapost-redelivery.xyz/pay"
)

ORDINARY = (
    "Hey, running about 10 minutes late for dinner. Order me the usual if the "
    "waiter comes by."
)


def run(label: str, message: str, expect_investigation: bool) -> bool:
    print(f"\n{'=' * 66}\n{label}\n{'=' * 66}")
    print(f"message: {message[:90]}{'...' if len(message) > 90 else ''}")

    agent = build_triage_agent()
    result = agent(
        "Assess this message for triage.\n\n"
        + wrap_untrusted(message, source="user_submitted_message"),
        structured_output_model=TriageResult,
        limits=TRIAGE_BUDGET.as_limits(),
    )
    triage: TriageResult = result.structured_output

    print(f"\n  warrants_investigation : {triage.warrants_investigation}")
    print(f"  claimed_organization   : {triage.claimed_organization}")
    print(f"  requested_action       : {triage.requested_action.value}")
    print(f"  visible_hostname       : {triage.visible_hostname}")
    print(f"  urgency_signals        : {triage.urgency_signals}")
    print(f"  reasoning              : {triage.reasoning}")

    ok = triage.warrants_investigation == expect_investigation
    print(f"\n  => {'PASS' if ok else 'FAIL'} "
          f"(expected warrants_investigation={expect_investigation})")
    return ok


def main() -> int:
    print(f"model: {TRIAGE_MODEL}")
    results = [
        run("SUSPICIOUS — should investigate", SUSPICIOUS, True),
        run("ORDINARY — should stay quiet", ORDINARY, False),
    ]
    print(f"\n{'=' * 66}")
    print(f"{sum(results)}/{len(results)} passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
