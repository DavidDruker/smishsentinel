"""Tests for the curated organization -> domain registry.

This is the correctness half of domain enforcement: test_evidence_tools.py's
TestDomainLock proves the pipeline is *consistent* with whatever domain gets
locked, but consistency alone can't catch a wrong domain. These tests cover
the registry itself -- the thing that decides which domain is correct in the
first place, and that an organization outside it produces an honest
"unknown" rather than a guess.
"""

from __future__ import annotations

import unittest

from smishsentinel.registry import all_organizations, resolve


class TestResolve(unittest.TestCase):
    def test_resolves_by_canonical_name(self) -> None:
        record = resolve("Canada Post")
        self.assertIsNotNone(record)
        self.assertEqual(record.canonical_name, "Canada Post")
        self.assertEqual(record.domain, "canadapost-postescanada.ca")

    def test_resolves_by_alias(self) -> None:
        record = resolve("Postes Canada")
        self.assertIsNotNone(record)
        self.assertEqual(record.canonical_name, "Canada Post")

    def test_resolution_is_case_and_punctuation_insensitive(self) -> None:
        record = resolve("  rbc royal-bank!! ")
        self.assertIsNotNone(record)
        self.assertEqual(record.canonical_name, "RBC Royal Bank")

    def test_unregistered_organization_returns_none_not_a_guess(self) -> None:
        self.assertIsNone(resolve("Totally Fictitious Bank of Nowhere"))

    def test_empty_and_none_input_return_none(self) -> None:
        self.assertIsNone(resolve(""))
        self.assertIsNone(resolve(None))  # type: ignore[arg-type]

    def test_near_miss_does_not_fuzzy_match(self) -> None:
        """A near-miss must not silently resolve to the wrong organization —
        the whole point of a curated registry is that matching is exact."""
        self.assertIsNone(resolve("Canada Postal Service"))

    def test_every_record_has_a_nonempty_domain(self) -> None:
        for record in all_organizations():
            with self.subTest(org=record.canonical_name):
                self.assertTrue(record.domain)

    def test_registry_covers_at_least_ten_organizations(self) -> None:
        self.assertGreaterEqual(len(all_organizations()), 10)

    def test_canonical_names_are_unique(self) -> None:
        names = [r.canonical_name for r in all_organizations()]
        self.assertEqual(len(names), len(set(names)))

    def test_no_alias_collides_with_a_different_organizations_domain(self) -> None:
        """Every name form (canonical or alias) must resolve to a record
        whose own domain -- never a different organization's."""
        for record in all_organizations():
            for name in (record.canonical_name, *record.aliases):
                with self.subTest(name=name):
                    self.assertEqual(resolve(name).domain, record.domain)


if __name__ == "__main__":
    unittest.main()
