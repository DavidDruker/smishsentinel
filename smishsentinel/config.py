"""Model and budget configuration.

Model IDs are inference-profile IDs (the ``us.`` prefix). Bare model IDs such
as ``anthropic.claude-sonnet-4-6`` return AccessDeniedException on this account
even though they appear in ``list-foundation-models``; the profile ID is what
actually resolves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE", "agentsforhumans")

# Verified working on this account.
MODEL_HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Requires the Anthropic use-case form to be submitted in the Bedrock console.
# Until then these raise ResourceNotFoundException.
MODEL_SONNET = "us.anthropic.claude-sonnet-4-6"

# Triage runs on every message, so it uses the cheap model by default; the
# reasoning-heavy stages prefer Sonnet and fall back when it is unavailable.
TRIAGE_MODEL = os.environ.get("SMISH_TRIAGE_MODEL", MODEL_HAIKU)
REASONING_MODEL = os.environ.get("SMISH_REASONING_MODEL", MODEL_HAIKU)


@dataclass(frozen=True)
class StageBudget:
    """Per-stage ceiling on model work.

    Passed to ``Agent.__call__(limits=...)``. Without these a single crafted
    message could drive an unbounded tool loop — the denial-of-wallet risk that
    matters for a service that anyone can send input to.
    """

    turns: int
    output_tokens: int

    def as_limits(self) -> dict[str, int]:
        return {"turns": self.turns, "output_tokens": self.output_tokens}


# Triage must stay cheap: it runs on every message, including the majority
# that end there.
TRIAGE_BUDGET = StageBudget(turns=2, output_tokens=800)

# Investigation is allowed real tool loops, but still bounded.
INVESTIGATION_BUDGET = StageBudget(turns=12, output_tokens=6000)

# Synthesis reasons over already-gathered evidence; no tools needed.
SYNTHESIS_BUDGET = StageBudget(turns=3, output_tokens=3000)

# Hard ceiling on evidence fetches per case, independent of model behaviour.
MAX_FETCHES_PER_CASE = 6
