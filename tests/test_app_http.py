"""HTTP-boundary tests for app.py, run in-process via Starlette's TestClient
-- no live server, no network, no AWS credentials needed.

Only the invalid-payload paths are covered here, deliberately: those are the
only paths that don't reach investigate() (which needs real Bedrock access),
so this file stays runnable offline like the rest of the suite. It exists
because a reviewer found that invalid requests were coming back as HTTP 200
with a Python-repr string smuggled into the JSON body instead of a clean 4xx
-- a bug in exactly this boundary, and one the offline pipeline tests
couldn't have caught since they never go through the HTTP layer at all.
"""

from __future__ import annotations

import json
import unittest

from starlette.testclient import TestClient

from app import app


class TestInvalidRequestHandling(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def _assert_clean_422(self, response) -> dict:
        """A 4xx status, and a body that's genuinely valid JSON throughout --
        not just JSON-shaped with a stringified Python object inside it."""
        self.assertEqual(response.status_code, 422)
        # json.loads succeeding at all is most of the assertion: the original
        # bug didn't produce invalid JSON, it produced valid JSON containing a
        # str(SomePythonException) value, which round-trips through json.loads
        # just fine and has to be checked for structurally, not just parsed.
        body = json.loads(response.text)
        self.assertIn("error", body)
        return body

    def test_missing_text_field_is_a_clean_422(self) -> None:
        response = self.client.post("/invocations", json={"wrong_field": "hello"})
        body = self._assert_clean_422(response)
        fields = body["error"]["fields"]
        self.assertEqual(fields[0]["field"], "text")
        self.assertEqual(fields[0]["type"], "missing")
        # The exact regression: pydantic's raw errors() can carry a `ctx` key
        # holding the original exception object, which isn't JSON-native.
        # A clean, fixed response has no such key anywhere in the body.
        self.assertNotIn("ctx", response.text)

    def test_empty_text_is_a_clean_422(self) -> None:
        response = self.client.post("/invocations", json={"text": ""})
        body = self._assert_clean_422(response)
        self.assertEqual(body["error"]["fields"][0]["type"], "value_error")

    def test_non_string_text_is_a_clean_422_not_a_silent_200(self) -> None:
        """The CoreBreak-shaped attack: a caller sends a list instead of a
        string for `text`. This must never reach an agent, and the framework
        must report it as an error the caller can see -- not as a quiet 200,
        which is what a plain returned dict (rather than a raised
        HTTPException) produces regardless of its contents."""
        response = self.client.post("/invocations", json={"text": ["not", "a", "string"]})
        body = self._assert_clean_422(response)
        self.assertEqual(body["error"]["fields"][0]["type"], "string_type")

    def test_text_over_length_limit_is_a_clean_422(self) -> None:
        response = self.client.post("/invocations", json={"text": "x" * 5000})
        self._assert_clean_422(response)

    def test_ping_is_healthy(self) -> None:
        response = self.client.get("/ping")
        self.assertEqual(response.status_code, 200)
        self.assertIn("status", response.json())


if __name__ == "__main__":
    unittest.main()
