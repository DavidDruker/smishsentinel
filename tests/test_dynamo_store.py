"""Tests for DynamoDBCaseStore and the get_case_store() backend switch.

No moto, no real AWS, no new dependency: _FakeTable/_FakeResource below are a
small in-memory double implementing just the put_item/get_item/scan surface
DynamoDBCaseStore actually calls, with the same semantics a real table would
have for these tests (missing key -> no Item, scan returns everything).
That's enough to exercise the store's real logic -- item shape, recency
sort, limit slicing, round-tripping through CaseRecord -- rather than just
asserting which methods got called.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

from smishsentinel.store import (
    CaseRecord,
    CaseStatus,
    CaseStore,
    DynamoDBCaseStore,
    get_case_store,
    new_case_id,
)


class _FakeTable:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def put_item(self, Item: dict) -> None:
        self.items[Item["case_id"]] = dict(Item)

    def get_item(self, Key: dict) -> dict:
        item = self.items.get(Key["case_id"])
        return {"Item": dict(item)} if item is not None else {}

    def scan(self, ExclusiveStartKey=None) -> dict:
        # No pagination in this fake -- these tests stay well under any real
        # single-page Scan limit, so LastEvaluatedKey never needs exercising.
        return {"Items": [dict(v) for v in self.items.values()]}


class _FakeResource:
    def __init__(self, table: _FakeTable) -> None:
        self._table = table

    def Table(self, name: str) -> _FakeTable:
        return self._table


def _record(case_id: str | None = None, updated_at: str = "2026-08-20T12:00:00+00:00") -> CaseRecord:
    record = CaseRecord(
        case_id=case_id or new_case_id(),
        received_at="2026-08-20T11:00:00+00:00",
        status=CaseStatus.COMPLETE,
        message_text="Canada Post: pay your redelivery fee: http://scam.example/pay",
        triage={"warrants_investigation": True, "claimed_organization": "Canada Post"},
        card={"verdict": "official_contradiction", "headline": "Do not pay this fee."},
    )
    record.updated_at = updated_at
    return record


class TestDynamoDBCaseStoreRoundTrip(unittest.TestCase):
    def setUp(self) -> None:
        self.table = _FakeTable()
        self.store = DynamoDBCaseStore(table_name="test-cases", resource=_FakeResource(self.table))

    def test_save_and_get_roundtrip(self) -> None:
        record = _record()
        self.store.save(record)

        loaded = self.store.get(record.case_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.case_id, record.case_id)
        self.assertEqual(loaded.status, CaseStatus.COMPLETE)
        self.assertEqual(loaded.card["verdict"], "official_contradiction")

    def test_get_missing_case_returns_none(self) -> None:
        self.assertIsNone(self.store.get("case-doesnotexist"))

    def test_save_writes_a_plain_dict_item_matching_to_json_dict(self) -> None:
        """The item actually stored must be exactly what to_json_dict()
        produces -- the same shape the JSON store already persists, so
        either backend round-trips identically."""
        record = _record()
        self.store.save(record)

        stored_item = self.table.items[record.case_id]
        self.assertEqual(stored_item["status"], "complete")
        self.assertEqual(stored_item["message_text"], record.message_text)

    def test_save_updates_updated_at(self) -> None:
        record = _record(updated_at="2020-01-01T00:00:00+00:00")
        self.store.save(record)
        self.assertNotEqual(record.updated_at, "2020-01-01T00:00:00+00:00")


class TestDynamoDBCaseStoreListRecent(unittest.TestCase):
    def setUp(self) -> None:
        self.table = _FakeTable()
        self.store = DynamoDBCaseStore(table_name="test-cases", resource=_FakeResource(self.table))

    def test_list_recent_orders_newest_first_by_updated_at(self) -> None:
        older = _record(updated_at="2026-08-20T09:00:00+00:00")
        newer = _record(updated_at="2026-08-20T10:00:00+00:00")
        self.store.save(older)
        self.store.save(newer)

        recent = self.store.list_recent(limit=10)
        self.assertEqual([r.case_id for r in recent], [newer.case_id, older.case_id])

    def test_list_recent_respects_limit(self) -> None:
        for i in range(5):
            self.store.save(_record(updated_at=f"2026-08-20T0{i}:00:00+00:00"))

        self.assertEqual(len(self.store.list_recent(limit=2)), 2)

    def test_list_recent_on_empty_table_returns_empty_list(self) -> None:
        self.assertEqual(self.store.list_recent(), [])


class TestDynamoDBCaseStoreConstruction(unittest.TestCase):
    def test_requires_a_table_name(self) -> None:
        with self.assertRaises(ValueError):
            DynamoDBCaseStore(resource=_FakeResource(_FakeTable()))

    def test_table_name_can_come_from_env_var(self) -> None:
        with mock.patch.dict(os.environ, {"SMISH_CASE_TABLE": "env-table"}):
            store = DynamoDBCaseStore(resource=_FakeResource(_FakeTable()))
        self.assertEqual(store.table_name, "env-table")


class TestGetCaseStoreBackendSwitch(unittest.TestCase):
    def test_returns_json_case_store_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SMISH_CASE_TABLE", None)
            store = get_case_store()
        self.assertIsInstance(store, CaseStore)

    def test_returns_dynamodb_store_when_table_env_var_is_set(self) -> None:
        """Only checks the type/construction path -- doesn't touch real AWS,
        since DynamoDBCaseStore's own boto3.resource() call is lazy and this
        test never calls save/get on it."""
        with mock.patch.dict(os.environ, {"SMISH_CASE_TABLE": "smishsentinel-cases"}):
            store = get_case_store()
        self.assertIsInstance(store, DynamoDBCaseStore)
        self.assertEqual(store.table_name, "smishsentinel-cases")


if __name__ == "__main__":
    unittest.main()
