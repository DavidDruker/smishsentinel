"""Tests for the local demo web UI (webui.py).

Two layers, tested separately: the pure data-shaping/rendering functions
(no HTTP involved, fast and exhaustive), and a real end-to-end round trip
against a live server bound to localhost -- proving the actual HTTP layer
works, not just the string templates in isolation. Everything here is
loopback-only; nothing reaches the network or needs AWS credentials.
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from smishsentinel.notify import decide, deliver
from smishsentinel.schemas import RequestedAction, RiskLevel, TriageResult
from smishsentinel.store import CaseRecord, CaseStatus, CaseStore, new_case_id
from smishsentinel.webui import (
    case_summary,
    make_app,
    recent_case_summaries,
    render_case_page,
    render_inbox_page,
)


def _triage(warrants: bool = True) -> TriageResult:
    return TriageResult(
        warrants_investigation=warrants,
        claimed_organization="Canada Post" if warrants else None,
        requested_action=RequestedAction.MAKE_PAYMENT if warrants else RequestedAction.NONE_DETECTED,
        urgency_signals=[],
        visible_hostname=None,
        reasoning="test fixture",
    )


def _investigated_record() -> CaseRecord:
    record = CaseRecord(
        case_id=new_case_id(),
        received_at="2026-08-20T12:00:00+00:00",
        status=CaseStatus.COMPLETE,
        message_text="Canada Post: pay your redelivery fee now: http://scam.example/pay",
    )
    triage = _triage(True)
    card = {
        "verdict": "official_contradiction",
        "risk_level": "high",
        "headline": "Do not pay this fee.",
        "observed_behaviour": ["Manufactured urgency."],
        "verified_facts": ["Canada Post never asks for payment by text (E1)."],
        "inferences": ["Likely a redelivery-fee scam."],
        "unresolved": [],
        "evidence": [
            {
                "evidence_id": "E1",
                "source_url": "https://www.canadapost-postescanada.ca/fraud",
                "source_controller": "Canada Post",
                "is_first_party": True,
                "quoted_text": "We never ask for payment by text.",
            }
        ],
        "safe_next_action": "Check tracking in the official Canada Post app.",
    }
    record.triage = triage.model_dump(mode="json")
    record.card = card
    channel = decide(triage, mock.Mock(risk_level=RiskLevel.HIGH))
    record.notification = deliver(record, channel)
    return record


def _quiet_record() -> CaseRecord:
    record = CaseRecord(
        case_id=new_case_id(),
        received_at="2026-08-20T12:00:00+00:00",
        status=CaseStatus.COMPLETE,
        message_text="hey are we still on for lunch?",
    )
    triage = _triage(False)
    record.triage = triage.model_dump(mode="json")
    record.notification = deliver(record, decide(triage, None))
    return record


class TestCaseSummary(unittest.TestCase):
    def test_investigated_case_summary_carries_headline_and_channel(self) -> None:
        summary = case_summary(_investigated_record())
        self.assertTrue(summary["investigated"])
        self.assertEqual(summary["channel"], "urgent")
        self.assertIn("Do not pay", summary["headline"])

    def test_quiet_case_summary_has_no_headline(self) -> None:
        summary = case_summary(_quiet_record())
        self.assertFalse(summary["investigated"])
        self.assertEqual(summary["channel"], "none")
        self.assertIsNone(summary["headline"])

    def test_failed_case_summary_carries_the_error(self) -> None:
        record = CaseRecord(
            case_id=new_case_id(), received_at="2026-01-01T00:00:00+00:00",
            status=CaseStatus.FAILED, message_text="x", error="RuntimeError: boom",
        )
        summary = case_summary(record)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("boom", summary["error"])


class TestRendering(unittest.TestCase):
    """The rendering functions must never crash on real card shapes and must
    escape message-derived content rather than interpolate it raw."""

    def test_inbox_page_renders_with_no_cases(self) -> None:
        page = render_inbox_page([])
        self.assertIn("No cases yet", page)

    def test_inbox_page_lists_investigated_and_quiet_cases(self) -> None:
        cases = [case_summary(_investigated_record()), case_summary(_quiet_record())]
        page = render_inbox_page(cases)
        self.assertIn("Do not pay this fee.", page)
        self.assertIn("URGENT", page.upper())

    def test_case_page_includes_clickable_evidence_source(self) -> None:
        page = render_case_page(_investigated_record())
        self.assertIn('href="https://www.canadapost-postescanada.ca/fraud"', page)
        self.assertIn("Check tracking in the official Canada Post app.", page)

    def test_case_page_for_quiet_case_says_no_investigation(self) -> None:
        page = render_case_page(_quiet_record())
        self.assertIn("did not warrant investigation", page)

    def test_message_text_is_escaped_not_interpolated_raw(self) -> None:
        """A message containing HTML-significant characters must not be
        able to inject markup into the rendered page."""
        record = CaseRecord(
            case_id=new_case_id(), received_at="2026-01-01T00:00:00+00:00",
            status=CaseStatus.COMPLETE,
            message_text="<script>alert(1)</script>",
        )
        page = render_case_page(record)
        self.assertNotIn("<script>alert(1)</script>", page)
        self.assertIn("&lt;script&gt;", page)


class _LiveServerCase(unittest.TestCase):
    """Shared scaffolding for tests that need a real running server bound to
    loopback -- not itself a test case (no test_ methods), just the setup
    TestLiveServer and TestInvestigateEndpoint both build on."""

    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.store = CaseStore(base_dir=self.tmpdir.name)
        record = _investigated_record()
        self.store.save(record)
        self.record = record
        self.server = make_app(store=self.store, port=0)  # port=0 -> OS picks a free port
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._shutdown)

    def _shutdown(self) -> None:
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()

    def _get(self, path: str) -> tuple[int, bytes]:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}", timeout=5) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

    def _post(self, path: str, body: bytes = b"", content_type: str = "application/json") -> tuple[int, bytes]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=body, method="POST",
            headers={"Content-Type": content_type},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()

class TestLiveServer(_LiveServerCase):
    """A real request/response round trip against the stdlib HTTP server."""

    def test_index_page_lists_the_seeded_case(self) -> None:
        status, body = self._get("/")
        self.assertEqual(status, 200)
        self.assertIn(self.record.case_id.encode(), body)

    def test_api_cases_returns_valid_json(self) -> None:
        status, body = self._get("/api/cases")
        self.assertEqual(status, 200)
        cases = json.loads(body)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["case_id"], self.record.case_id)

    def test_case_detail_page_returns_the_evidence_card(self) -> None:
        status, body = self._get(f"/case/{self.record.case_id}")
        self.assertEqual(status, 200)
        self.assertIn(b"Do not pay this fee.", body)

    def test_unknown_case_is_a_404(self) -> None:
        status, _ = self._get("/case/case-doesnotexist")
        self.assertEqual(status, 404)


class TestInvestigateEndpoint(_LiveServerCase):
    """POST /investigate -- the custom-message path. investigate() is
    mocked, same reasoning as test_inbox_pipeline.py: it needs live
    Bedrock, and what's under test here is the HTTP/validation/persistence
    layer around it, not the model's judgment."""

    def _wait_for_status(self, case_id: str, status: str, timeout: float = 5) -> CaseRecord:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.store.get(case_id)
            if record is not None and record.status.value == status:
                return record
            time.sleep(0.02)
        self.fail(f"case {case_id} did not reach status={status!r} within {timeout}s")

    def test_persists_immediately_and_completes_in_the_background(self) -> None:
        card = mock.Mock(risk_level=RiskLevel.HIGH)
        card.model_dump.return_value = {"headline": "Do not click.", "verdict": "suspicious_unconfirmed"}
        triage = TriageResult(
            warrants_investigation=True, claimed_organization="Acme",
            requested_action=RequestedAction.CLICK_LINK, urgency_signals=[],
            visible_hostname=None, reasoning="test",
        )
        # The mock must stay active for the whole background-thread window,
        # not just for _post() -- investigate_one_message() runs in a thread
        # started by the request handler, which can easily still be running
        # (or not yet started) after _post() returns and an outer `with`
        # would already have unpatched investigate() back to the real,
        # live-Bedrock-calling function. Losing this race once during
        # development made a real model call from what should be a fully
        # offline test -- keeping _wait_for_status inside the patch is what
        # actually closes that gap, not just tidiness.
        with mock.patch("smishsentinel.inbox.investigate") as mock_investigate:
            mock_investigate.return_value = {"investigated": True, "triage": triage, "card": card}
            status, body = self._post(
                "/investigate", json.dumps({"message": "a suspicious message"}).encode()
            )
            self.assertEqual(status, 202)
            case_id = json.loads(body)["case_id"]
            # Persisted synchronously -- present the instant the response comes back.
            self.assertIsNotNone(self.store.get(case_id))
            record = self._wait_for_status(case_id, "complete")

        self.assertEqual(record.notification.channel.value, "urgent")
        self.assertEqual(record.card["headline"], "Do not click.")

    def test_rejects_blank_message(self) -> None:
        status, body = self._post("/investigate", json.dumps({"message": "   "}).encode())
        self.assertEqual(status, 422)
        self.assertIn("error", json.loads(body))

    def test_rejects_missing_message_field(self) -> None:
        status, _ = self._post("/investigate", json.dumps({}).encode())
        self.assertEqual(status, 422)

    def test_rejects_non_string_message(self) -> None:
        status, _ = self._post("/investigate", json.dumps({"message": ["not", "a", "string"]}).encode())
        self.assertEqual(status, 422)

    def test_rejects_oversized_message(self) -> None:
        status, _ = self._post("/investigate", json.dumps({"message": "x" * 5000}).encode())
        self.assertEqual(status, 422)

    def test_rejects_invalid_json_body(self) -> None:
        status, body = self._post("/investigate", b"not json")
        self.assertEqual(status, 400)
        self.assertIn("error", json.loads(body))


if __name__ == "__main__":
    unittest.main()
