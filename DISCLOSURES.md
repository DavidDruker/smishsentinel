# Disclosures

Required reading alongside the hackathon's "New Projects Only" clause: *"The
work described and submitted must have been built during the Submission
Period... but must disclose any other pre-existing code or work incorporated
into the Project."*

## Prior work that informed this project's design

The core idea this project tests — that an SMS-scam analyzer should check
what the claimed organization actually publishes, rather than pattern-match
on message wording or trust a model's own judgment about legitimacy — was
explored in earlier, independent personal research conducted before this
hackathon's Submission Period. That research is not incorporated into this
codebase: no code, prompts, datasets, or written material from it appears
here. What carried over is the underlying concept and specific lessons about
what goes wrong with it, most concretely a known failure mode where a system
given first-party evidence retrieval starts treating an unresolved or
inaccessible source as equivalent to a contradicting one — a defect this
project's synthesis prompt explicitly guards against (see
`ARCHITECTURE.md`). Every line of code, every prompt, every schema, and the
entire Strands Agents SDK implementation in this repository was built fresh
during the Submission Period.

That prior research is not cited anywhere in this repository as evidence
that this project's approach works, and it should not be read that way. Its
own findings were a mixed, largely-null result on a related but materially
different pipeline, and are not evidence about this system's behavior.

## Third-party references

- [SmishX](https://github.com/yizhu-joy/SmishX) (SOUPS 2025, MIT license) —
  a peer-reviewed system pairing an LLM with link/brand-context retrieval
  for SMS-phishing detection. Conceptually adjacent prior art in the same
  problem space; no code from it is used in this repository. Referenced in
  hackathon planning materials, not incorporated here.
- The Strands Agents SDK (Apache 2.0) and Amazon Bedrock AgentCore are the
  required frameworks this project is built on, per `requirements.txt` and
  the hackathon rules.

## Data

The five messages in `smishsentinel/inbox.py`'s synthetic inbox are
originals, written for this project — not drawn from any dataset, real
message archive, or third-party corpus. No personal data, real phone
numbers, or real organizational content appears in this repository.
