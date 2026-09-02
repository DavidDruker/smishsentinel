"""AgentCore entrypoint.

Deliberately thin: this file's only job is the boundary between an untrusted
caller and the pipeline in ``smishsentinel.agent``. Everything here exists to
satisfy AgentCore's contract (``POST /invocations``, ``GET /ping``) and to
close the CoreBreak gap documented in the project's hackathon skill — the
open-source Strands event loop will execute a ``toolUse`` block found as the
latest message without ever calling the model, so the boundary accepts
nothing but a plain string and builds every agent's message history itself.
"""

from __future__ import annotations

from bedrock_agentcore import BedrockAgentCoreApp
from pydantic import BaseModel, ValidationError, field_validator
from starlette.exceptions import HTTPException

from smishsentinel.agent import investigate
from smishsentinel.inbox import run_inbox_cycle
from smishsentinel.notify import verify_delivered, verify_notification_sent
from smishsentinel.store import get_case_store

app = BedrockAgentCoreApp()


class InvocationRequest(BaseModel):
    """The only shape a caller is allowed to send.

    Rejecting anything but a plain string at this boundary is what actually
    closes the CoreBreak gap for this app — not a system-prompt instruction,
    which the vulnerability bypasses entirely since the model never runs.
    """

    text: str

    @field_validator("text")
    @classmethod
    def must_be_nonempty_plain_string(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("text must be a non-empty string")
        if len(v) > 4000:
            raise ValueError("text exceeds the 4000-character limit")
        return v


def _serialize(result: dict) -> dict:
    """Turn the pipeline's Pydantic objects into a JSON-serializable dict."""
    triage = result["triage"]
    card = result["card"]
    ml_screening = result.get("ml_screening")
    return {
        "investigated": result["investigated"],
        "triage": triage.model_dump(mode="json") if triage else None,
        "card": card.model_dump(mode="json") if card else None,
        "ml_screening": ml_screening.model_dump(mode="json") if ml_screening else None,
    }


def _clean_validation_detail(exc: ValidationError) -> list[dict[str, object]]:
    """Reduce pydantic's error list to JSON-native fields only.

    ``ValidationError.errors()`` can include a ``ctx`` entry holding the
    original Python exception object (e.g. a ``ValueError``), and an
    ``input`` entry echoing back whatever the caller sent — neither is
    guaranteed JSON-serializable, and letting either through was exactly what
    made an invalid request come back as a 200 with a Python-repr string
    smuggled into the JSON body instead of a clean 4xx: the framework's
    serializer degrades to `str(obj)` on the whole payload the moment any one
    field fails to encode, rather than failing just that field.
    """
    return [
        {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"], "type": e["type"]}
        for e in exc.errors()
    ]


def _run_inbox_demo() -> dict:
    """The end-to-end action: synthetic inbox -> investigation -> persisted,
    notified-or-suppressed, independently-verified outcome for each case.

    ``verify_delivered`` re-reads each case from the store rather than
    trusting the in-memory record ``run_inbox_cycle`` returns — the point is
    to confirm the persisted state actually says what the pipeline claims,
    not just that the pipeline claims it.
    """
    store = get_case_store()
    records = run_inbox_cycle(store=store)
    return {
        "action": "run_inbox_cycle",
        "cases": [
            {
                "case_id": r.case_id,
                "status": r.status.value,
                "investigated": r.triage is not None and r.triage.get("warrants_investigation", False),
                "verdict": r.card["verdict"] if r.card else None,
                "notification_channel": r.notification.channel.value if r.notification else None,
                "verified_delivered": verify_delivered(store.get(r.case_id)),
                "notification_actually_sent": verify_notification_sent(store.get(r.case_id)),
            }
            for r in records
        ],
    }


@app.entrypoint
def invoke(payload: dict) -> dict:
    """AgentCore calls this with whatever JSON body the caller POSTed.

    Two shapes are accepted: ``{"action": "run_inbox_cycle"}`` runs the
    end-to-end synthetic-inbox demo (see smishsentinel/inbox.py), and
    everything else is validated as a single-message analysis request.
    ``payload`` is caller-controlled and therefore untrusted either way: the
    action check only matches one exact literal string, and the text path is
    validated into ``InvocationRequest`` before anything else happens, so
    only a resulting plain string is ever handed to an agent.

    Raising ``HTTPException`` (rather than returning an error dict) is what
    actually produces a real 4xx: this app's underlying framework only sets a
    non-200 status when the handler raises HTTPException or returns a
    Response object directly — a plain returned dict always serializes as a
    200, whatever it contains.
    """
    if isinstance(payload, dict) and payload.get("action") == "run_inbox_cycle":
        return _run_inbox_demo()

    try:
        request = InvocationRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={"reason": "invalid_request", "fields": _clean_validation_detail(exc)},
        ) from exc

    result = investigate(request.text)
    return _serialize(result)


if __name__ == "__main__":
    app.run()
