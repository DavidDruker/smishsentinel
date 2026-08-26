"""Deterministic stand-ins for the four Strands agents in agent.py.

Not mocks of Strands internals -- these are plain objects that satisfy the
exact call shape ``investigate()`` uses (callable with a prompt and, where
applicable, ``structured_output_model``/``limits`` keywords) and nothing
more. They exist so ``tests/test_deterministic_eval.py`` can run the real
``investigate()`` orchestration -- real tool calls, real domain-registry
resolution, real ``_enforce_citations`` -- against scripted responses instead
of live Bedrock calls, closing the gap between "each stage is unit-tested in
isolation" and "the pipeline that chains them together has any offline
coverage at all."
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from smishsentinel.safety import FetchResult

# Recorded official-page fixtures, keyed by URL. Stands in for a live fetch:
# tests patch smishsentinel.tools.evidence.safe_fetch to look these up rather
# than making a real HTTP request, so "evidence retrieval" is exercised for
# real (the actual fetch_official_page tool, the actual domain-lock check)
# without any network access.
RECORDED_PAGES: dict[str, FetchResult] = {
    "https://www.canadapost-postescanada.ca/cpc/en/support/articles/security-and-fraud-prevention.page": FetchResult(
        url="https://www.canadapost-postescanada.ca/cpc/en/support/articles/security-and-fraud-prevention.page",
        final_url="https://www.canadapost-postescanada.ca/cpc/en/support/articles/security-and-fraud-prevention.page",
        status=200,
        text=(
            "Canada Post will never request payment or personal information "
            "by text message. If you receive a text asking for payment to "
            "release a parcel, do not click any links -- report it to our "
            "fraud prevention team."
        ),
        truncated=False,
    ),
}


def recorded_safe_fetch(url: str) -> FetchResult:
    """A drop-in replacement for smishsentinel.safety.safe_fetch that serves
    only the recorded fixtures above. Raises like a real 404 would look to
    the caller (via a FetchResult with status>=400) for anything else, so a
    scripted investigator that goes off-script fails loudly rather than
    silently hitting the real network."""
    if url in RECORDED_PAGES:
        return RECORDED_PAGES[url]
    return FetchResult(url=url, final_url=url, status=404, text="", truncated=False)


@dataclass
class _StructuredResult:
    structured_output: Any


class FakeStructuredAgent:
    """Stands in for the triage, claim-extraction, and synthesis agents --
    each of which is called as ``agent(prompt, structured_output_model=X,
    limits=Y)`` and read via ``.structured_output``. Always returns the same
    canned object regardless of prompt content, since these tests are
    proving the pipeline's plumbing, not a model's judgment."""

    def __init__(self, response: Any) -> None:
        self._response = response

    def __call__(self, prompt: str, *, structured_output_model=None, limits=None):
        return _StructuredResult(self._response)


class ScriptedInvestigatorAgent:
    """Stands in for the investigator agent -- the one stage that can't just
    return a canned value, because the whole point of testing it is that it
    actually drives the real tools (set_official_domain, fetch_official_page,
    compare_hostname_to_domain, report_fetch_ledger) exactly as a live model
    would, just via a fixed script instead of reasoning about what to call.

    ``actions`` is a list of ``(tool_callable, args)`` pairs run in order
    when this agent is invoked; the return value mimics what
    ``investigate()`` does with a real investigator's result (used only via
    ``str()`` in the synthesis prompt), so any short string works.
    """

    def __init__(self, actions: list[tuple[Callable[..., str], tuple]], summary: str) -> None:
        self._actions = actions
        self._summary = summary

    def __call__(self, prompt: str, *, limits=None) -> str:
        for tool_fn, args in self._actions:
            tool_fn(*args)
        return self._summary
