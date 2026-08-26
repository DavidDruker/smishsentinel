"""Offline tests for the end-to-end action: store persistence, the
notify/suppress policy, delivery, and the inbox cycle's status transitions.

investigate() is mocked throughout -- it needs real Bedrock access, and none
of what's tested here is about whether the model reasons well, only about
whether the surrounding machinery (persistence, policy, delivery,
verification, failure handling) does what it claims regardless of what the
model returns.
"""

from __future__ import annotations

import tempfile
import unittest
from unittest import mock

from smishsentinel.inbox import run_inbox_cycle
from smishsentinel.notify import decide, deliver, verify_delivered, verify_notification_sent
from smishsentinel.schemas import RequestedAction, RiskLevel, TriageResult, Verdict
from smishsentinel.store import CaseRecord, CaseStatus, CaseStore, NotificationChannel, new_case_id


def _triage(warrants: bool) -> TriageResult:
    return TriageResult(
        warrants_investigation=warrants,
        claimed_organization="Canada Post" if warrants else None,
        requested_action=RequestedAction.CLICK_LINK if warrants else RequestedAction.NONE_DETECTED,
        urgency_signals=[],
        visible_hostname=None,
        reasoning="test fixture",
    )


class TestCaseStore(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = CaseStore(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_save_and_get_roundtrip(self) -> None:
        record = CaseRecord(
            case_id=new_case_id(), received_at="2026-01-01T00:00:00+00:00",
            status=CaseStatus.RECEIVED, message_text="hello",
        )
        self.store.save(record)

        loaded = self.store.get(record.case_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.case_id, record.case_id)
        self.assertEqual(loaded.status, CaseStatus.RECEIVED)
        self.assertEqual(loaded.message_text, "hello")

    def test_get_missing_case_returns_none(self) -> None:
        self.assertIsNone(self.store.get("case-doesnotexist"))

    def test_unsafe_case_id_is_rejected(self) -> None:
        record = CaseRecord(
            case_id="../../etc/passwd", received_at="2026-01-01T00:00:00+00:00",
            status=CaseStatus.RECEIVED, message_text="x",
        )
        with self.assertRaises(ValueError):
            self.store.save(record)

    def test_list_recent_orders_newest_first(self) -> None:
        import time

        ids = []
        for i in range(3):
            record = CaseRecord(
                case_id=new_case_id(), received_at="2026-01-01T00:00:00+00:00",
                status=CaseStatus.RECEIVED, message_text=f"msg-{i}",
            )
            self.store.save(record)
            ids.append(record.case_id)
            time.sleep(0.01)  # ensure distinct mtimes

        recent = self.store.list_recent(limit=10)
        self.assertEqual([r.case_id for r in recent], list(reversed(ids)))

    def test_status_and_notification_survive_a_roundtrip(self) -> None:
        record = CaseRecord(
            case_id=new_case_id(), received_at="2026-01-01T00:00:00+00:00",
            status=CaseStatus.COMPLETE, message_text="x",
        )
        record.notification = deliver(record, NotificationChannel.URGENT)
        self.store.save(record)

        loaded = self.store.get(record.case_id)
        self.assertEqual(loaded.status, CaseStatus.COMPLETE)
        self.assertEqual(loaded.notification.channel, NotificationChannel.URGENT)
        self.assertTrue(loaded.notification.decision_recorded)
        self.assertTrue(loaded.notification.notification_delivered)


class TestNotifyPolicy(unittest.TestCase):
    def test_suppresses_when_no_investigation_warranted(self) -> None:
        self.assertEqual(decide(_triage(False), None), NotificationChannel.NONE)

    def test_urgent_when_card_missing_despite_investigation(self) -> None:
        """Defensive default: an investigated case with no card is a gap in
        the pipeline, not something to suppress silently."""
        self.assertEqual(decide(_triage(True), None), NotificationChannel.URGENT)

    def test_urgent_on_high_risk_card(self) -> None:
        card = mock.Mock(risk_level=RiskLevel.HIGH)
        self.assertEqual(decide(_triage(True), card), NotificationChannel.URGENT)

    def test_standard_on_non_high_risk_investigated_card(self) -> None:
        for level in (RiskLevel.ELEVATED, RiskLevel.UNCLEAR, RiskLevel.QUIET):
            card = mock.Mock(risk_level=level)
            with self.subTest(level=level):
                self.assertEqual(decide(_triage(True), card), NotificationChannel.STANDARD)


class TestDeliveryAndVerification(unittest.TestCase):
    def _record(self, status: CaseStatus) -> CaseRecord:
        return CaseRecord(
            case_id=new_case_id(), received_at="2026-01-01T00:00:00+00:00",
            status=status, message_text="x",
        )

    def test_suppression_is_recorded_as_a_decision_but_not_a_delivery(self) -> None:
        """A suppressed case still has a real, checkable outcome -- the
        record proves the decision was made and executed, not silently
        dropped -- but it must not claim a notification went out when
        nothing was actually sent."""
        record = self._record(CaseStatus.COMPLETE)
        notification = deliver(record, NotificationChannel.NONE)
        self.assertTrue(notification.decision_recorded)
        self.assertFalse(notification.notification_delivered)
        self.assertEqual(notification.detail, "suppressed")

    def test_notify_delivery_includes_the_headline(self) -> None:
        record = self._record(CaseStatus.COMPLETE)
        record.card = {"headline": "Do not click this link."}
        notification = deliver(record, NotificationChannel.URGENT)
        self.assertTrue(notification.decision_recorded)
        self.assertTrue(notification.notification_delivered)
        self.assertIn("Do not click this link.", notification.detail)
        self.assertIsNotNone(notification.delivered_at)

    def test_verify_delivered_false_for_missing_record(self) -> None:
        self.assertFalse(verify_delivered(None))

    def test_verify_delivered_false_when_status_not_complete(self) -> None:
        record = self._record(CaseStatus.INVESTIGATING)
        record.notification = deliver(record, NotificationChannel.URGENT)
        self.assertFalse(verify_delivered(record))

    def test_verify_delivered_false_when_no_notification_recorded(self) -> None:
        record = self._record(CaseStatus.COMPLETE)
        self.assertFalse(verify_delivered(record))

    def test_verify_delivered_true_for_a_genuinely_complete_case(self) -> None:
        record = self._record(CaseStatus.COMPLETE)
        record.notification = deliver(record, NotificationChannel.STANDARD)
        self.assertTrue(verify_delivered(record))

    def test_verify_notification_sent_is_false_for_a_suppressed_case(self) -> None:
        """The distinction verify_delivered can't make on its own: a
        suppressed case completed correctly (verify_delivered=True) but no
        notification actually reached anyone (verify_notification_sent=False)."""
        record = self._record(CaseStatus.COMPLETE)
        record.notification = deliver(record, NotificationChannel.NONE)
        self.assertTrue(verify_delivered(record))
        self.assertFalse(verify_notification_sent(record))

    def test_verify_notification_sent_is_true_for_a_real_delivery(self) -> None:
        record = self._record(CaseStatus.COMPLETE)
        record.notification = deliver(record, NotificationChannel.URGENT)
        self.assertTrue(verify_notification_sent(record))


class TestInboxCycle(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.store = CaseStore(base_dir=self.tmpdir.name)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def test_quiet_message_ends_suppressed_and_persisted(self) -> None:
        with mock.patch("smishsentinel.inbox.investigate") as mock_investigate:
            mock_investigate.return_value = {"investigated": False, "triage": _triage(False), "card": None}
            records = run_inbox_cycle(messages=["hey, running late"], store=self.store)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.status, CaseStatus.COMPLETE)
        self.assertEqual(record.notification.channel, NotificationChannel.NONE)
        self.assertTrue(verify_delivered(self.store.get(record.case_id)))

    def test_cycle_deadline_fails_remaining_messages_without_a_model_call(self) -> None:
        """The overall-cycle latency ceiling: an exceeded deadline fails every
        remaining message outright rather than continuing to burn model
        calls and wall-clock time. deadline_seconds=-1 guarantees the check
        trips before the very first message, so investigate() must never be
        called at all."""
        with mock.patch("smishsentinel.inbox.investigate") as mock_investigate:
            records = run_inbox_cycle(
                messages=["a", "b", "c"], store=self.store, deadline_seconds=-1,
            )

        mock_investigate.assert_not_called()
        self.assertEqual(len(records), 3)
        for record in records:
            self.assertEqual(record.status, CaseStatus.FAILED)
            self.assertIn("deadline", record.error.lower())
            self.assertIsNotNone(record.notification)
            self.assertEqual(record.notification.channel, NotificationChannel.URGENT)
            self.assertTrue(record.notification.notification_delivered)

    def test_investigated_message_ends_notified_and_persisted(self) -> None:
        card = mock.Mock(risk_level=RiskLevel.HIGH)
        card.model_dump.return_value = {"headline": "Do not click.", "verdict": Verdict.SUSPICIOUS_UNCONFIRMED.value}

        with mock.patch("smishsentinel.inbox.investigate") as mock_investigate:
            mock_investigate.return_value = {"investigated": True, "triage": _triage(True), "card": card}
            records = run_inbox_cycle(messages=["suspicious message"], store=self.store)

        record = records[0]
        self.assertEqual(record.status, CaseStatus.COMPLETE)
        self.assertEqual(record.notification.channel, NotificationChannel.URGENT)
        self.assertTrue(verify_delivered(self.store.get(record.case_id)))

    def test_pipeline_exception_ends_failed_not_crashed(self) -> None:
        """A message that blows up the pipeline must leave a real, queryable
        failure record -- not an unhandled exception that silently drops the
        case, and not a false COMPLETE."""
        with mock.patch("smishsentinel.inbox.investigate") as mock_investigate:
            mock_investigate.side_effect = RuntimeError("simulated Bedrock outage")
            records = run_inbox_cycle(messages=["will blow up"], store=self.store)

        record = records[0]
        self.assertEqual(record.status, CaseStatus.FAILED)
        self.assertIn("simulated Bedrock outage", record.error)
        loaded = self.store.get(record.case_id)
        self.assertEqual(loaded.status, CaseStatus.FAILED)
        # status stays FAILED, not COMPLETE, so verify_delivered is correctly
        # false regardless of the notification below -- it answers "did the
        # pipeline finish," and it didn't.
        self.assertFalse(verify_delivered(loaded))

    def test_pipeline_exception_still_notifies_the_user(self) -> None:
        """The gap a reviewer found: a failed investigation used to leave a
        FAILED record with no notification at all -- the user was never told
        their message couldn't be checked. A failure to investigate must
        reach the user, not vanish silently."""
        with mock.patch("smishsentinel.inbox.investigate") as mock_investigate:
            mock_investigate.side_effect = RuntimeError("simulated Bedrock outage")
            records = run_inbox_cycle(messages=["will blow up"], store=self.store)

        record = records[0]
        self.assertIsNotNone(record.notification)
        self.assertEqual(record.notification.channel, NotificationChannel.URGENT)
        self.assertTrue(record.notification.notification_delivered)

        loaded = self.store.get(record.case_id)
        self.assertIsNotNone(loaded.notification)
        self.assertTrue(loaded.notification.notification_delivered)

    def test_multi_message_cycle_persists_every_case_independently(self) -> None:
        with mock.patch("smishsentinel.inbox.investigate") as mock_investigate:
            mock_investigate.return_value = {"investigated": False, "triage": _triage(False), "card": None}
            records = run_inbox_cycle(messages=["a", "b", "c"], store=self.store)

        self.assertEqual(len(records), 3)
        self.assertEqual(len({r.case_id for r in records}), 3)  # no ID collisions
        self.assertEqual(len(self.store.list_recent(limit=10)), 3)


if __name__ == "__main__":
    unittest.main()
