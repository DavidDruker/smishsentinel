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
from smishsentinel.tools.evidence import current_context, evidence_dump, reset_context


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


_REAL_QUOTE = "We will never request payment by text message."


def _evidence(
    evidence_id: str,
    url: str,
    quoted_text: str = _REAL_QUOTE,
    is_first_party: bool = True,
    source_controller: str = "Canada Post",
) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        source_url=url,
        source_controller=source_controller,
        is_first_party=is_first_party,
        quoted_text=quoted_text,
        retrieved_at="2026-08-20T12:00:00+00:00",
    )


class TestCitationEnforcement(unittest.TestCase):
    """The fetch ledger, not the model, decides what evidence exists.

    These specifically cover the gap a reviewer found in an earlier version:
    the old check only confirmed an evidence_id had been fetched *at some
    point*, so a fabricated quotation and a fabricated URL both survived as
    long as they reused a real ID — even an ID that represented a 404. Every
    class of that forgery gets its own test below, named for the forgery it
    blocks.
    """

    def setUp(self) -> None:
        self.context = reset_context()
        self.context.official_domain = "canadapost.ca"
        self.context.claimed_organization = "Canada Post"

    def test_genuine_citation_survives(self) -> None:
        self.context.record(
            "https://canadapost.ca/fraud", "https://canadapost.ca/fraud", 200,
            text=_REAL_QUOTE, is_first_party=True,
        )

        card = _card(evidence=[_evidence("E1", "https://canadapost.ca/fraud")])
        result = _enforce_citations(card)

        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.verdict, Verdict.OFFICIAL_CONTRADICTION)

    def test_hallucinated_evidence_id_is_stripped(self) -> None:
        """A citation to an ID that was never fetched at all must not survive."""
        card = _card(evidence=[_evidence("E1", "https://canadapost.ca/invented")])
        result = _enforce_citations(card)

        self.assertEqual(result.evidence, [])

    def test_fabricated_quote_on_a_real_id_is_stripped(self) -> None:
        """The exact reviewer-reported gap: a real ID, but an invented quote.

        E1 really was fetched and really does contain text — just not the
        text the card claims to quote from it.
        """
        self.context.record(
            "https://canadapost.ca/fraud", "https://canadapost.ca/fraud", 200,
            text="Canada Post never asks for banking information by SMS.",
            is_first_party=True,
        )

        card = _card(
            verdict=Verdict.OFFICIAL_CONTRADICTION,
            evidence=[_evidence(
                "E1", "https://canadapost.ca/fraud",
                quoted_text="Canada Post confirms this $2.99 fee is legitimate.",
            )],
        )
        result = _enforce_citations(card)

        self.assertEqual(result.evidence, [])
        self.assertEqual(result.verdict, Verdict.INSUFFICIENT_EVIDENCE)

    def test_citation_against_a_404_is_stripped_even_with_a_real_id(self) -> None:
        """The other half of the reviewer's repro: reusing an ID that exists
        in the ledger, but represents a failed fetch with no real content."""
        self.context.record(
            "https://canadapost.ca/gone", "https://canadapost.ca/gone", 404,
            is_first_party=True,
        )

        card = _card(
            verdict=Verdict.OFFICIAL_CONTRADICTION,
            evidence=[_evidence(
                "E1", "https://canadapost.ca/gone",
                quoted_text="Canada Post confirms this $2.99 fee is legitimate.",
            )],
        )
        result = _enforce_citations(card)

        self.assertEqual(result.evidence, [])
        self.assertEqual(result.verdict, Verdict.INSUFFICIENT_EVIDENCE)

    def test_citation_with_mismatched_url_is_stripped(self) -> None:
        """A real, successful fetch at E1 -- but the card cites E1 while
        claiming a different source_url than what was actually fetched."""
        self.context.record(
            "https://canadapost.ca/fraud", "https://canadapost.ca/fraud", 200,
            text=_REAL_QUOTE, is_first_party=True,
        )

        card = _card(evidence=[_evidence("E1", "https://canadapost.ca/totally-different-page")])
        result = _enforce_citations(card)

        self.assertEqual(result.evidence, [])

    def test_url_citation_accepts_either_requested_or_final_redirect_url(self) -> None:
        """Not a forgery: citing the post-redirect URL for a fetch that
        redirected must still be accepted."""
        self.context.record(
            "https://canadapost.ca/old-path",
            "https://www.canadapost-postescanada.ca/new-path",
            200, text=_REAL_QUOTE, is_first_party=True,
        )

        card = _card(evidence=[_evidence(
            "E1", "https://www.canadapost-postescanada.ca/new-path",
        )])
        result = _enforce_citations(card)

        self.assertEqual(len(result.evidence), 1)

    def test_is_first_party_and_controller_are_overridden_not_trusted(self) -> None:
        """The model can claim whatever it wants in these two fields; the
        ledger's domain-lock result is what actually lands in the card."""
        self.context.record(
            "https://canadapost.ca/fraud", "https://canadapost.ca/fraud", 200,
            text=_REAL_QUOTE, is_first_party=False,  # e.g. redirected off-domain
        )

        card = _card(evidence=[_evidence(
            "E1", "https://canadapost.ca/fraud",
            is_first_party=True, source_controller="Canada Post",  # the model's (wrong) claim
        )])
        result = _enforce_citations(card)

        self.assertEqual(len(result.evidence), 1)
        self.assertFalse(result.evidence[0].is_first_party)
        self.assertEqual(result.evidence[0].source_controller, "unverified")

    def test_verdict_downgrades_when_its_evidence_vanishes(self) -> None:
        """An evidence-backed verdict cannot stand on stripped evidence."""
        card = _card(
            verdict=Verdict.OFFICIAL_CONTRADICTION,
            evidence=[_evidence("E1", "https://canadapost.ca/invented")],
        )
        result = _enforce_citations(card)

        self.assertEqual(result.verdict, Verdict.INSUFFICIENT_EVIDENCE)
        self.assertTrue(result.unresolved, "downgrade must explain itself")

    def test_claim_assessment_loses_phantom_support(self) -> None:
        self.context.record(
            "https://canadapost.ca/fraud", "https://canadapost.ca/fraud", 200,
            text=_REAL_QUOTE, is_first_party=True,
        )

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
        result = _enforce_citations(card)

        self.assertEqual(result.claim_assessments[0].supporting_evidence_ids, ["E1"])

    def test_verified_fact_citing_a_failed_fetch_is_dropped(self) -> None:
        """verified_facts is free text with inline (E<n>) markers, not a
        structured field -- it must be held to the same standard."""
        self.context.record(
            "https://canadapost.ca/gone", "https://canadapost.ca/gone", 404,
            is_first_party=True,
        )

        card = _card(verified_facts=["Canada Post confirms this fee is standard (E1)."])
        result = _enforce_citations(card)

        self.assertEqual(result.verified_facts, [])
        self.assertTrue(
            any("failed verification" in u for u in result.unresolved),
            result.unresolved,
        )

    def test_verified_fact_with_no_citation_is_left_alone(self) -> None:
        """A fact with no (E<n>) marker at all isn't a citation claim and
        shouldn't be touched by citation enforcement."""
        card = _card(verified_facts=["The message uses a .xyz domain."])
        result = _enforce_citations(card)

        self.assertEqual(result.verified_facts, ["The message uses a .xyz domain."])

    def test_suspicious_verdict_survives_without_evidence(self) -> None:
        """Behavioural suspicion needs no citation, so it must not downgrade."""
        card = _card(verdict=Verdict.SUSPICIOUS_UNCONFIRMED, evidence=[])
        result = _enforce_citations(card)

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


class TestEvidenceDump(unittest.TestCase):
    """evidence_dump() is what lets synthesis quote real page text instead of
    the investigator's paraphrase of it — see the regression this guards
    against in agent.py's module docstring / commit history: a live run
    correctly refused to fabricate a quote because the synthesist had never
    been given anything to quote from.
    """

    def test_empty_context_says_so_plainly(self) -> None:
        reset_context()
        self.assertIn("No page text", evidence_dump())

    def test_dump_includes_the_actual_retrieved_text(self) -> None:
        context = reset_context()
        eid = context.record(
            "https://canadapost.ca/security",
            "https://canadapost.ca/security",
            200,
            text="We will never ask you to pay a redelivery fee by text message.",
        )
        dump = evidence_dump()
        self.assertIn(eid, dump)
        self.assertIn("canadapost.ca/security", dump)
        self.assertIn("never ask you to pay a redelivery fee", dump)

    def test_records_without_text_are_omitted_not_fabricated(self) -> None:
        """A 404 or empty-body fetch is recorded (it consumed budget) but has
        no text — the dump must not paper over that with a placeholder a
        model could mistake for real content."""
        context = reset_context()
        context.record("https://canadapost.ca/gone", "https://canadapost.ca/gone", 404)
        dump = evidence_dump()
        self.assertNotIn("canadapost.ca/gone", dump)


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
