"""Structured output contracts for SmishSentinel.

Every stage of the pipeline returns a validated Pydantic model rather than free
text. This is deliberate: the product claim is "we show you the evidence," and
an evidence card is only auditable if its fields are typed, enumerated, and
impossible to quietly omit.

The most important design decision in this file is that "we found no evidence
of a problem" and "we verified this is safe" are different values of different
fields. A missing contradiction is not a clean bill of health, and the schema
refuses to let the model collapse the two.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    """Terminal assessment for a message.

    Ordered from most to least actionable. ``INSUFFICIENT_EVIDENCE`` is a
    first-class outcome, not a failure mode: abstaining is correct when the
    organization publishes nothing that bears on the claim.
    """

    KNOWN_MALICIOUS = "known_malicious"
    OFFICIAL_CONTRADICTION = "official_contradiction"
    SUSPICIOUS_UNCONFIRMED = "suspicious_unconfirmed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_CONTRADICTION_FOUND = "no_contradiction_found"


class RiskLevel(str, Enum):
    """How loudly the agent should interrupt, if at all."""

    HIGH = "high"
    ELEVATED = "elevated"
    UNCLEAR = "unclear"
    QUIET = "quiet"


class ClaimStatus(str, Enum):
    """How first-party evidence bears on a single extracted claim."""

    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    NOT_ADDRESSED = "not_addressed"
    PARTIALLY_SUPPORTED = "partially_supported"
    SUPPORTED = "supported"
    SOURCE_UNUSABLE = "source_unusable"


class RequestedAction(str, Enum):
    """What the message wants the recipient to do.

    Scam messages are defined more by the action they demand than by their
    wording, so this is the primary behavioural axis for triage.
    """

    CLICK_LINK = "click_link"
    CALL_NUMBER = "call_number"
    REPLY_WITH_DATA = "reply_with_data"
    MAKE_PAYMENT = "make_payment"
    SHARE_CREDENTIAL = "share_credential"
    INSTALL_APP = "install_app"
    MOVE_TO_CHANNEL = "move_to_channel"
    NONE_DETECTED = "none_detected"


# --------------------------------------------------------------------------
# Stage 1 — triage
# --------------------------------------------------------------------------


class TriageResult(BaseModel):
    """Cheap first pass deciding whether a message deserves investigation.

    This gate is what makes the agent quiet. Most messages end here and the
    user is never notified.
    """

    warrants_investigation: bool = Field(
        description=(
            "True only if this message both claims an identity worth verifying "
            "and asks the recipient to take a consequential action."
        )
    )
    claimed_organization: str | None = Field(
        default=None,
        description=(
            "The organization the message presents itself as, verbatim as named "
            "in the message. Null if the message names no organization."
        ),
    )
    requested_action: RequestedAction = Field(
        description="The most consequential action the message asks for."
    )
    urgency_signals: list[str] = Field(
        default_factory=list,
        description=(
            "Short verbatim phrases from the message that manufacture time "
            "pressure, threat, or secrecy. Empty if none."
        ),
    )
    visible_hostname: str | None = Field(
        default=None,
        description=(
            "The hostname visible in the message's link, if any. Extracted "
            "literally; never resolved, expanded, or followed."
        ),
    )
    reasoning: str = Field(
        description="One or two sentences explaining the triage decision."
    )


# --------------------------------------------------------------------------
# Stage 2 — claim extraction
# --------------------------------------------------------------------------


class ExtractedClaim(BaseModel):
    """A single checkable assertion the message makes."""

    claim_id: str = Field(description="Stable identifier, e.g. 'C1'.")
    claim_text: str = Field(
        description="The assertion restated as a neutral, checkable proposition."
    )
    why_it_matters: str = Field(
        description="What the recipient risks if this claim is false."
    )


class ClaimSet(BaseModel):
    """All checkable claims extracted from one message."""

    claims: list[ExtractedClaim] = Field(
        description="Between one and four claims, most consequential first."
    )


# --------------------------------------------------------------------------
# Stage 3 — first-party evidence
# --------------------------------------------------------------------------


class EvidenceItem(BaseModel):
    """One retrieved piece of first-party evidence.

    ``source_url`` must be a page actually fetched during this investigation.
    Unfetched or recalled-from-memory sources are not evidence and must not
    appear here.
    """

    evidence_id: str = Field(description="Stable identifier, e.g. 'E1'.")
    source_url: str = Field(description="Exact URL fetched.")
    source_controller: str = Field(
        description=(
            "The entity that controls this domain, e.g. 'Canada Post' or "
            "'Government of Canada'. This is what makes evidence first-party. "
            "The pipeline overwrites this after synthesis from the domain lock "
            "recorded during investigation — fill it in as your best answer, "
            "but do not expect this field to be trusted as written."
        )
    )
    is_first_party: bool = Field(
        description=(
            "True only if source_controller is the same organization the "
            "message claims to be from. The pipeline overwrites this after "
            "synthesis based on whether the fetch actually matched the locked "
            "official domain — it is not taken from this field as written."
        )
    )
    quoted_text: str = Field(
        description=(
            "A bounded verbatim excerpt (max ~300 chars) from the page that "
            "bears on a claim. Never a paraphrase. Checked against the actual "
            "retrieved text after synthesis; a quote that cannot be found "
            "verbatim in what was really fetched causes this entire evidence "
            "item to be dropped from the card, not silently corrected."
        )
    )
    retrieved_at: str = Field(description="ISO-8601 UTC retrieval timestamp.")


class ClaimAssessment(BaseModel):
    """How the retrieved evidence bears on one claim."""

    claim_id: str
    status: ClaimStatus
    supporting_evidence_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence IDs that drove this status. Must be empty when status is "
            "NOT_ADDRESSED — that status means no evidence spoke to the claim."
        ),
    )
    rationale: str = Field(
        description="Why the evidence yields this status, in one or two sentences."
    )


# --------------------------------------------------------------------------
# Stage 4 — the evidence card
# --------------------------------------------------------------------------


class EvidenceCard(BaseModel):
    """The user-facing product of an investigation.

    Deliberately separates four layers that a scam score collapses into one
    number: what the message did, what was independently verified, what was
    inferred from that, and what remains unknown.
    """

    verdict: Verdict
    risk_level: RiskLevel
    headline: str = Field(
        description=(
            "One sentence a worried person can act on, leading with the "
            "behavioural risk rather than a score. Plain language, no jargon."
        )
    )

    claimed_identity: str | None = Field(
        default=None, description="Who the message presents itself as."
    )
    requested_action: RequestedAction
    observed_behaviour: list[str] = Field(
        default_factory=list,
        description=(
            "What the message itself does — urgency, impersonation cues, "
            "hostname mismatch. Observations about the message, not the world."
        ),
    )

    verified_facts: list[str] = Field(
        default_factory=list,
        description=(
            "Statements established by retrieved first-party evidence. Each "
            "must cite an evidence ID inline, e.g. '(E1)'."
        ),
    )
    inferences: list[str] = Field(
        default_factory=list,
        description=(
            "Conclusions reasoned from the facts. Never presented as verified."
        ),
    )
    unresolved: list[str] = Field(
        default_factory=list,
        description=(
            "What could not be established, and why it matters. This field "
            "must not be empty when verdict is INSUFFICIENT_EVIDENCE."
        ),
    )

    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)

    safe_next_action: str = Field(
        description=(
            "A concrete verification path found independently of the message. "
            "Must never cite a link, phone number, or address from the message "
            "itself."
        )
    )

    def is_safe_claim(self) -> bool:
        """Whether this card asserts safety. It never does, by construction.

        Kept as an explicit method so the invariant is testable rather than
        merely documented: no verdict value means "this message is safe."
        """
        return False
