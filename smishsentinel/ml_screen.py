"""A statistical second opinion for messages triage lets go quiet.

Triage's gate (see agent.py's _TRIAGE_PROMPT) requires BOTH an identifiable
organization and a consequential action -- deliberately, since the
investigation stage that follows can only verify an organization's claim,
never judge risk in the abstract. A real category of scam text has neither:
a bare "you have WON, call 0900-xxx-xxx", no brand claimed at all. Triage is
right to wave that through by its own rule, and the investigation stage
would have nothing to check even if it ran -- there is no organization to
resolve against the registry. That gap is structural, not a triage bug.

This module is the different kind of check that gap needs: a classical
TF-IDF + linear SVM classifier, trained offline on labelled ham/spam/
smishing text (see research_dump/training_data/ and scratch_export_model.py,
not part of this package), run only on messages that already failed the
triage gate. It never produces a verdict, never claims evidence, and is
tuned to favour recall over precision on purpose -- see MLScreeningResult's
docstring for why a positive here means something categorically weaker than
an EvidenceCard, and notify.py for what a positive actually leads to
(NotificationChannel.ADVISORY, never STANDARD/URGENT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .schemas import MLScreeningResult

_ARTIFACT_PATH = Path(__file__).parent / "ml_models" / "smishing_screener.joblib"

_cache: dict[str, Any] = {}


def _load() -> dict[str, Any]:
    """Load once per process; every subsequent screen() call reuses it.

    Not lazy-and-silent: a missing or corrupt artifact raises immediately
    and loudly on first use, the same stance config.py takes on an
    unavailable model -- a screening step that silently no-ops would be far
    worse than one that fails to start.
    """
    if "artifact" not in _cache:
        artifact = joblib.load(_ARTIFACT_PATH)
        _cache["artifact"] = artifact
        _cache["version"] = artifact["metadata"]["trained_at"]
    return _cache["artifact"]


def screen(text: str) -> MLScreeningResult:
    """Score one message against the trained screener.

    Returns a result regardless of outcome -- ``flagged`` carries the
    decision, there is no separate "did we screen it" boolean, because this
    always runs synchronously and has no failure mode short of the process
    itself being broken (see _load).
    """
    artifact = _load()
    probability = float(artifact["pipeline"].predict_proba([text])[0, 1])
    threshold = float(artifact["threshold"])
    return MLScreeningResult(
        flagged=probability >= threshold,
        probability=probability,
        threshold=threshold,
        model_version=_cache["version"],
    )
