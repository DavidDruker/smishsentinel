"""Tests for the domain-lock enforcement and per-request state isolation in
smishsentinel.tools.evidence.

Two things get dedicated coverage here because a reviewer specifically found
them broken in an earlier version:

1. "Official source" was model-asserted, not enforced — fetch_official_page
   would fetch any public URL the model chose, and is_first_party/
   source_controller were free-text fields nothing checked. TestDomainLock
   covers the fix: a domain must be locked before any fetch or hostname
   comparison, and every fetch is checked against it before the request is
   even made.
2. The evidence ledger was a module-global variable. AgentCore runs
   invocation handlers in a thread pool, so two concurrent investigations
   could read and write the same ledger. TestConcurrentIsolation proves two
   genuinely-overlapping investigations, running in separate threads, cannot
   see each other's state — which a bare global could not have guaranteed,
   and which this test would have caught if run against that version.
"""

from __future__ import annotations

import threading
import unittest
from unittest import mock

from smishsentinel.tools.evidence import (
    compare_hostname_to_domain,
    current_context,
    fetch_official_page,
    reset_context,
    set_official_domain,
)


class TestDomainLock(unittest.TestCase):
    def setUp(self) -> None:
        reset_context()

    def test_first_call_locks_the_domain(self) -> None:
        result = set_official_domain("Canada Post", "canadapost.ca")
        self.assertIn("LOCKED", result)
        self.assertEqual(current_context().official_domain, "canadapost.ca")

    def test_locking_the_same_domain_again_is_idempotent(self) -> None:
        set_official_domain("Canada Post", "canadapost.ca")
        result = set_official_domain("Canada Post", "www.canadapost.ca")  # normalizes the same
        self.assertIn("ALREADY_LOCKED", result)
        self.assertEqual(current_context().official_domain, "canadapost.ca")

    def test_locking_a_different_domain_is_rejected(self) -> None:
        """A model cannot relabel an unrelated domain as official mid-case."""
        set_official_domain("Canada Post", "canadapost.ca")
        result = set_official_domain("Canada Post", "some-other-site.com")

        self.assertIn("REJECTED", result)
        self.assertEqual(current_context().official_domain, "canadapost.ca")

    def test_fetch_is_blocked_before_any_domain_is_locked(self) -> None:
        with mock.patch("smishsentinel.tools.evidence.safe_fetch") as mock_fetch:
            result = fetch_official_page("https://canadapost.ca/fraud")

        self.assertIn("BLOCKED", result)
        mock_fetch.assert_not_called()

    def test_fetch_off_the_locked_domain_is_blocked_before_any_request(self) -> None:
        """The core of the fix: the model cannot fetch an arbitrary domain and
        have it counted as first-party just by asserting so afterward — it
        cannot even fetch it as evidence in the first place."""
        set_official_domain("Canada Post", "canadapost.ca")

        with mock.patch("smishsentinel.tools.evidence.safe_fetch") as mock_fetch:
            result = fetch_official_page("https://totally-unrelated-site.com/page")

        self.assertIn("BLOCKED", result)
        self.assertIn("totally-unrelated-site.com", result)
        mock_fetch.assert_not_called()

    def test_fetch_on_a_subdomain_of_the_locked_domain_is_allowed(self) -> None:
        set_official_domain("Canada Post", "canadapost.ca")

        with mock.patch("smishsentinel.tools.evidence.safe_fetch") as mock_fetch:
            from smishsentinel.safety import FetchResult
            mock_fetch.return_value = FetchResult(
                url="https://track.canadapost.ca/fraud",
                final_url="https://track.canadapost.ca/fraud",
                status=200, text="Some real page text.", truncated=False,
            )
            fetch_official_page("https://track.canadapost.ca/fraud")

        mock_fetch.assert_called_once()

    def test_hostname_comparison_is_blocked_before_any_domain_is_locked(self) -> None:
        result = compare_hostname_to_domain("canadapost-fake.xyz")
        self.assertIn("BLOCKED", result)

    def test_hostname_comparison_uses_the_locked_domain(self) -> None:
        set_official_domain("Canada Post", "canadapost.ca")
        result = compare_hostname_to_domain("canadapost-fake.xyz")
        self.assertIn("NO_MATCH", result)

    def test_recorded_fetch_first_party_flag_reflects_domain_match(self) -> None:
        """Integration-shaped: a real fetch on the locked domain is recorded
        as first-party without the tool itself ever asking the model."""
        set_official_domain("Canada Post", "canadapost.ca")

        with mock.patch("smishsentinel.tools.evidence.safe_fetch") as mock_fetch:
            from smishsentinel.safety import FetchResult
            mock_fetch.return_value = FetchResult(
                url="https://canadapost.ca/fraud",
                final_url="https://canadapost.ca/fraud",
                status=200, text="Real page text.", truncated=False,
            )
            fetch_official_page("https://canadapost.ca/fraud")

        entry = current_context().fetch_log[0]
        self.assertTrue(entry["is_first_party"])


class TestConcurrentIsolation(unittest.TestCase):
    """Proves two genuinely-overlapping investigations cannot see each
    other's evidence ledger.

    This is deliberately adversarial about timing: a Barrier forces both
    threads to call reset_context() at effectively the same moment, then both
    record a fetch, then both read back their own context -- the shape most
    likely to expose shared mutable state. With a bare module-global (the
    original implementation), this is flaky-to-reliably-broken depending on
    scheduling; with contextvars, each thread has its own default context and
    this passes deterministically.
    """

    def test_two_threads_do_not_share_a_ledger(self) -> None:
        barrier = threading.Barrier(2)
        results: dict[str, list[str]] = {}
        errors: list[BaseException] = []

        def run_case(name: str, url: str) -> None:
            try:
                barrier.wait(timeout=5)  # maximize actual overlap
                ctx = reset_context()
                ctx.record(url, url, 200, text=f"page for {name}")
                barrier.wait(timeout=5)  # both threads have reset before either reads
                # If state were shared, this could now see the other thread's URL.
                results[name] = [entry["final_url"] for entry in current_context().fetch_log]
            except BaseException as exc:  # noqa: BLE001 - surface in the main thread
                errors.append(exc)

        t1 = threading.Thread(target=run_case, args=("A", "https://a.example.com/1"))
        t2 = threading.Thread(target=run_case, args=("B", "https://b.example.com/1"))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(results["A"], ["https://a.example.com/1"])
        self.assertEqual(results["B"], ["https://b.example.com/1"])

    def test_many_concurrent_cases_each_keep_exactly_their_own_fetches(self) -> None:
        """Same property at higher concurrency, without a hand-tuned barrier,
        to catch anything the two-thread case might get lucky on."""
        n = 12
        barrier = threading.Barrier(n)
        outcomes: dict[int, bool] = {}

        def run_case(i: int) -> None:
            barrier.wait(timeout=5)
            ctx = reset_context()
            url = f"https://case-{i}.example.com/page"
            ctx.record(url, url, 200, text=f"text-{i}")
            ctx.record(url, url, 200, text=f"text-{i}-again")
            log = current_context().fetch_log
            outcomes[i] = (
                len(log) == 2
                and all(entry["final_url"] == url for entry in log)
            )

        threads = [threading.Thread(target=run_case, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(outcomes), n)
        self.assertTrue(all(outcomes.values()), outcomes)


if __name__ == "__main__":
    unittest.main()
