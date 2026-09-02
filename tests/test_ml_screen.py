"""Offline tests for the statistical screener -- the real trained artifact,
not a mock. It's a local classical model (no network, no Bedrock, loads in
well under a second), so there's no cost reason to fake it here the way the
four Bedrock-backed agents are faked elsewhere; what's worth proving is that
loading and scoring actually works and returns the shape the rest of the
pipeline depends on.
"""

from __future__ import annotations

import unittest

from smishsentinel.ml_screen import screen
from smishsentinel.schemas import MLScreeningResult

_CLEARLY_SCAM_LIKE = (
    "URGENT! You have WON a guaranteed cash prize of $1000! To claim, call "
    "09051234567 NOW before this offer expires. Standard rates apply."
)
_CLEARLY_BENIGN = "Hey, are we still on for lunch tomorrow at noon?"


class TestMLScreen(unittest.TestCase):
    def test_returns_a_well_formed_result(self) -> None:
        result = screen(_CLEARLY_BENIGN)
        self.assertIsInstance(result, MLScreeningResult)
        self.assertIsInstance(result.flagged, bool)
        self.assertTrue(0.0 <= result.probability <= 1.0)
        self.assertTrue(0.0 <= result.threshold <= 1.0)
        self.assertTrue(result.model_version)

    def test_flagged_matches_probability_vs_threshold(self) -> None:
        for text in (_CLEARLY_SCAM_LIKE, _CLEARLY_BENIGN):
            with self.subTest(text=text):
                result = screen(text)
                self.assertEqual(result.flagged, result.probability >= result.threshold)

    def test_scam_like_template_text_scores_higher_than_benign_text(self) -> None:
        """Not a claim that every scam is caught -- see ml_screen.py's own
        docstring on what this model is and isn't -- just that the trained
        artifact separates an obvious case from an obvious non-case at all,
        which is the minimum bar for "the artifact loaded and isn't inert"."""
        scam_result = screen(_CLEARLY_SCAM_LIKE)
        benign_result = screen(_CLEARLY_BENIGN)
        self.assertGreater(scam_result.probability, benign_result.probability)
        self.assertTrue(scam_result.flagged)
        self.assertFalse(benign_result.flagged)

    def test_same_text_scores_identically_across_calls(self) -> None:
        """Deterministic: no randomness at inference time, unlike the
        Bedrock-backed stages."""
        first = screen(_CLEARLY_SCAM_LIKE)
        second = screen(_CLEARLY_SCAM_LIKE)
        self.assertEqual(first.probability, second.probability)


if __name__ == "__main__":
    unittest.main()
