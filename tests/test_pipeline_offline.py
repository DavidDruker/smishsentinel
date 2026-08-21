"""Pipeline tests that run without AWS credentials or network.

Two reasons this exists beyond convenience. First, the citation-enforcement
logic is the part of the system that constrains the model, so it must be
testable independently of any model. Second, a judge cloning this repo should
be able to run the test suite without an AWS account.
"""

from __future__ import annotations

import unittest

from smishsentinel.agent import _enforce_citations
from smishsentinel.schemas import (
    ClaimAssessment,
    ClaimStatus,
    EvidenceCard,
    EvidenceItem,
    RequestedAction,
    RiskLevel,
    Verdict,
)
from smishsentinel.tools.evidence import current_context, reset_context


def _card(**overrides) -> EvidenceCard:
    """An evidence card with sensible defaults, overridable per test."""
    base = {
        "verdict": Verdict.OFFICIAL_CONTRADICTION,
        "risk_level": RiskLevel.HIGH,
        "headline": "Do not pay this fee; the sender is not who it claims to be.",
        "claimed_identity": "Canada Post",
        "requested_action": RequestedAction.MAKE_PAYMENT,
        "observed_behaviour": ["Manufactured a 24-hour deadline."],
        "verified_facts": ["Canada Post does not request fees by text (E1)."],
        "inferences": ["The message is likely part of a redelivery-fee campaign."],
        "unresolved": [],
        "claim_assessments": [],
        "evidence": [],
        "safe_next_action": "Check tracking in the official Canada Post app.",
    }
    base.update(overrides)
    return EvidenceCard(**base)


def _evidence(evidence_id: str, url: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_url=url,
        source_controller="Canada Post",
        is_first_party=True,
        quoted_text="We will never request payment by text message.",
        retrieved_at="2026-08-20T12:00:00+00:00",
    )


class TestCitationEnforcement(unittest.TestCase):
    """The fetch ledger, not the model, decides what evidence exists."""

    def setUp(self) -> None:
        reset_context()

    def test_genuine_citation_survives(self) -> None:
        context = reset_context()
        context.record("https://canadapost.ca/fraud", "https://canadapost.ca/fraud", 200)

        card = _card(evidence=[_evidence("E1", "https://canadapost.ca/fraud")])
        result = _enforce_citations(card, context.retrieved_urls())

        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.verdict, Verdict.OFFICIAL_CONTRADICTION)

    def test_hallucinated_evidence_is_stripped(self) -> None:
        """A citation to a page never fetched must not survive."""
        context = reset_context()  # nothing fetched

        card = _card(evidence=[_evidence("E1", "https://canadapost.ca/invented")])
        result = _enforce_citations(card, context.retrieved_urls())

        self.assertEqual(result.evidence, [])

    def test_verdict_downgrades_when_its_evidence_vanishes(self) -> None:
        """An evidence-backed verdict cannot stand on stripped evidence."""
        context = reset_context()  # nothing fetched

        card = _card(
            verdict=Verdict.OFFICIAL_CONTRADICTION,
            evidence=[_evidence("E1", "https://canadapost.ca/invented")],
        )
        result = _enforce_citations(card, context.retrieved_urls())

        self.assertEqual(result.verdict, Verdict.INSUFFICIENT_EVIDENCE)
        self.assertTrue(result.unresolved, "downgrade must explain itself")

    def test_claim_assessment_loses_phantom_support(self) -> None:
        context = reset_context()
        context.record("https://canadapost.ca/fraud", "https://canadapost.ca/fraud", 200)

        card = _card(
            evidence=[_evidence("E1", "https://canadapost.ca/fraud")],
            claim_assessments=[
                ClaimAssessment(
                    claim_id="C1",
                    status=ClaimStatus.CONTRADICTED,
                    supporting_evidence_ids=["E1", "E7"],  # E7 never fetched
                    rationale="Official page contradicts the fee claim.",
                )
            ],
        )
        result = _enforce_citations(card, context.retrieved_urls())

        self.assertEqual(result.claim_assessments[0].supporting_evidence_ids, ["E1"])

    def test_suspicious_verdict_survives_without_evidence(self) -> None:
        """Behavioural suspicion needs no citation, so it must not downgrade."""
        context = reset_context()

        card = _card(verdict=Verdict.SUSPICIOUS_UNCONFIRMED, evidence=[])
        result = _enforce_citations(card, context.retrieved_urls())

        self.assertEqual(result.verdict, Verdict.SUSPICIOUS_UNCONFIRMED)


class TestFetchBudget(unittest.TestCase):
    def test_budget_depletes_and_floors_at_zero(self) -> None:
        context = reset_context()
        start = context.remaining()
        self.assertGreater(start, 0)

        for i in range(start + 3):
            context.record(f"https://example.com/{i}", f"https://example.com/{i}", 200)

        self.assertEqual(context.remaining(), 0)

    def test_evidence_ids_are_sequential(self) -> None:
        context = reset_context()
        ids = [
            context.record(f"https://example.com/{i}", f"https://example.com/{i}", 200)
            for i in range(3)
        ]
        self.assertEqual(ids, ["E1", "E2", "E3"])

    def test_reset_clears_prior_case(self) -> None:
        """Case state must not leak between messages."""
        context = reset_context()
        context.record("https://example.com/a", "https://example.com/a", 200)
        self.assertEqual(len(current_context().fetch_log), 1)

        reset_context()
        self.assertEqual(len(current_context().fetch_log), 0)


class TestSchemaInvariants(unittest.TestCase):
    def test_no_verdict_asserts_safety(self) -> None:
        """The core product promise, enforced as a test rather than a comment."""
        self.assertFalse(_card().is_safe_claim())
        for verdict in Verdict:
            self.assertNotIn("safe", verdict.value)

    def test_reassuring_verdict_is_named_honestly(self) -> None:
        """The weakest reassurance must not read as a clean bill of health."""
        self.assertEqual(Verdict.NO_CONTRADICTION_FOUND.value, "no_contradiction_found")


if __name__ == "__main__":
    unittest.main()
