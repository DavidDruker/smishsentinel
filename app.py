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

from smishsentinel.agent import investigate

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
    return {
        "investigated": result["investigated"],
        "triage": triage.model_dump(mode="json") if triage else None,
        "card": card.model_dump(mode="json") if card else None,
    }


@app.entrypoint
def invoke(payload: dict) -> dict:
    """AgentCore calls this with whatever JSON body the caller POSTed.

    ``payload`` is caller-controlled and therefore untrusted: it is validated
    into ``InvocationRequest`` before anything else happens, and only the
    resulting plain string is ever handed to an agent.
    """
    try:
        request = InvocationRequest.model_validate(payload)
    except ValidationError as exc:
        return {"error": "invalid_request", "detail": exc.errors()}

    result = investigate(request.text)
    return _serialize(result)


if __name__ == "__main__":
    app.run()
