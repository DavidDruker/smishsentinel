# Disclosures

## AI coding assistance

Built during the hackathon submission period using Claude Code as an AI
coding assistant throughout — the pipeline, tools, tests, documentation, and
deployment configuration were all written fresh for this submission, not
adapted from a pre-existing project. Third-party dependencies (the Strands
Agents SDK, boto3, Bedrock/AgentCore, pydantic, starlette, scikit-learn,
joblib) are standard tools, not incorporated prior work. Dataset cleaning for
the ML screener (deduplication of the training corpus) was also done with
Claude's assistance — see [`model-card.md`](model-card.md) for what that
involved and the resulting counts.

## Relationship to prior independent research

Separately from this submission, the author previously completed an
independent research project, *Verified-Source Evidence for LLM-Based
Smishing Detection* (David Druker, University of Toronto, August 2026) — a
reproduction and extension of the published SmishX system, evaluated with
paired McNemar significance tests and a manual evidence-grounding audit.

**This project was built fresh, during the competition period; the prior
research supplied problem understanding and experimental lessons, not code.**
Concretely, two findings from that research directly shaped design decisions
here, and are stated plainly rather than left implicit:

- That research's evidence-grounding audit found 0 of 283 explanations from
  either evaluated system had every claim actually supported by visible
  evidence — a negative result for relying on prompting alone to keep an
  LLM's explanations grounded. This project's citation-enforcement layer
  (`_enforce_citations` in [`agent.py`](../smishsentinel/agent.py)) takes a
  different approach to the same problem: every citation is checked
  deterministically against a real fetch ledger, rather than trusted because
  the model was asked to ground its claims.
- That research identified a recurring decision-policy defect: unresolved
  evidence (a link that could not be checked) being treated as equivalent to
  malicious evidence. This project's synthesis prompt and schema encode that
  distinction explicitly (`ClaimStatus.NOT_ADDRESSED`/`SOURCE_UNUSABLE` vs.
  `CONTRADICTED`; see `EvidenceCard`'s docstring in
  [`schemas.py`](../smishsentinel/schemas.py)) rather than leaving it to a
  single prompt instruction to enforce.

No code, prompts, or data from the prior research were reused directly; the
two systems are independent implementations. The prior research is not
itself part of this submission and is disclosed here for transparency about
where the underlying problem understanding came from.
