# SmishSentinel

An AI agent that investigates suspicious text messages by checking what the
organization they claim to be from actually publishes — not a black-box risk
score, and not silence disguised as safety.

Built for the [Agents for Humans Hackathon](https://agentsforhumans.devpost.com/) (Everyday Agents track) on
[AWS's Strands Agents SDK](https://strandsagents.com/), deployed to [Amazon Bedrock AgentCore Runtime](https://docs.aws.amazon.com/bedrock-agentcore/).

## What it does

Most texts don't need a second look. SmishSentinel's first pass is silent on
purpose — an ordinary message, a delivery notice, a friend confirming plans,
gets no response at all. It only investigates a message that both claims to
be from an identifiable organization *and* asks for something consequential:
a click, a call, a payment, a code.

When it does investigate, it doesn't guess. It locks in the organization's
real official domain, fetches that organization's own published pages, and
checks whether what they actually say lines up with what the message
claims. Every fact in the final evidence card traces back to a real,
independently-verified citation — see [Design decisions worth knowing about](#design-decisions-worth-knowing-about)
for exactly what "verified" means here and how it's enforced in code, not
just claimed in a prompt.

There is no verdict that means "safe." The weakest possible outcome —
retrieving an organization's material and finding no conflict — is reported
as exactly that, not as reassurance. See [`schemas.py`](smishsentinel/schemas.py) for the full verdict
taxonomy and why it's shaped this way.

## Try it

Two ways to invoke the deployed agent, or run either locally (see
[Running it](#running-it)):

**Analyze one message:**
```json
{"text": "Canada Post: Your parcel is held pending an unpaid redelivery fee of $2.99. Pay within 24 hours to avoid return to sender: http://canadapost-redelivery.xyz/pay"}
```

**Run the full end-to-end demo** — a synthetic inbox of five messages moves
through triage, investigation, persistence, and a notify-or-suppress
decision, with every outcome independently re-verified from storage
afterward:
```json
{"action": "run_inbox_cycle"}
```

## Architecture

Three specialist agents, composed rather than merged, plus a code layer that
trusts none of them by default. Full breakdown, the Strands-specific design
choices, and the reasoning behind the split: [`ARCHITECTURE.md`](ARCHITECTURE.md).

```
message → triage (silent gate) → claim extraction → investigation (the only
stage with network access) → synthesis → deterministic citation verification
→ persisted case → notify/suppress policy → delivery → independent
verification that delivery actually happened
```

## Design decisions worth knowing about

These aren't incidental implementation details — they're the parts of this
project that took the most iteration, including fixes made in response to an
external review that found real gaps. Documented here rather than just in
commit messages because a judge testing the citation guarantee, the
concurrency safety, or the domain enforcement should be able to find the
reasoning without archaeology.

- **Citations are verified, not trusted.** A model can hallucinate a
  quotation or reuse a real evidence ID with invented content. Every citation
  in the final card is checked against the fetch ledger: the ID was actually
  fetched, the cited URL matches what was fetched, the fetch succeeded
  (a 404 can't be cited as content), and the quoted text appears verbatim in
  what was actually retrieved. `is_first_party` and `source_controller` are
  overwritten from a deterministic domain lock after synthesis, never taken
  from the model as written. See `_enforce_citations` in
  [`agent.py`](smishsentinel/agent.py) and the tests in
  [`test_pipeline_offline.py`](tests/test_pipeline_offline.py) that name each
  forgery shape it blocks.
- **"Official" is enforced, not asserted.** The investigator must call
  `set_official_domain` before it can fetch anything or compare the
  message's hostname against anything. That domain locks once per case;
  `fetch_official_page` refuses — before making any request — any URL that
  isn't on the locked domain or a subdomain of it. See
  [`tools/evidence.py`](smishsentinel/tools/evidence.py).
- **Concurrent investigations can't corrupt each other.** AgentCore runs
  invocation handlers via a thread pool (visible directly in this project's
  own CloudWatch stack traces). Per-case state lives in a
  `contextvars.ContextVar`, not a module global, and
  [`test_evidence_tools.py`](tests/test_evidence_tools.py) proves it with a
  genuinely concurrent, `Barrier`-forced multi-thread test rather than a
  single-threaded assumption.
- **An unresolved link is not evidence of wrongdoing.** The synthesis prompt
  states this explicitly: a source that couldn't be checked is a different
  thing from a source that contradicts the message, and treating them the
  same is a known failure mode worth naming rather than leaving implicit.
- **The CoreBreak mitigation is a validation boundary, not a prompt
  instruction.** The Strands event loop will execute a `toolUse` block found
  as the latest message without a model call in between — a disclosed,
  unpatched gap in the open-source SDK. `app.py`'s entrypoint accepts only a
  validated plain string; there is no code path where caller-supplied
  structured content reaches an agent's message history.
- **"Delivery" is scoped honestly.** There's no phone in this loop. A
  notification here means a durable, queryable record plus a log line — the
  signal a real channel integration would consume downstream — not a push
  notification that doesn't exist in this submission. See
  [`notify.py`](smishsentinel/notify.py).

## Running it

Requires Python 3.10+, an AWS account with Bedrock model access enabled for
Anthropic models in your target region (including the account-level
"Anthropic use case details" attestation — see the AWS Console under Bedrock
→ Model catalog if a live call fails with `ResourceNotFoundException`
mentioning it), and AWS credentials available to boto3 (a named profile via
`AWS_PROFILE`, or the default credential chain).

```bash
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env        # then edit AWS_PROFILE / AWS_REGION for your account
```

**Run the offline test suite** — no AWS credentials or network needed, and
this is most of the test suite by design:
```bash
python -m unittest discover -s tests
```

**Run it locally:**
```bash
python app.py
# in another terminal:
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{"text": "Canada Post: ..."}'
```

**Deploy to Bedrock AgentCore:** this project uses the (deprecated but
functional) `bedrock-agentcore-starter-toolkit` CLI run inside WSL2 on
Windows, specifically to avoid a documented, unexplained native-Windows hang
in the current recommended CLI's `configure` step. Either toolchain works on
Linux/macOS directly:
```bash
pip install bedrock-agentcore-starter-toolkit
agentcore configure -e app.py -r us-east-1
agentcore deploy   # builds ARM64 via CodeBuild -- no local Docker needed
agentcore invoke '{"text": "..."}'
```

## Repository layout

```
app.py                          AgentCore entrypoint: HTTP boundary, CoreBreak-safe validation
smishsentinel/
  agent.py                      The three-stage pipeline and citation verification
  schemas.py                    Structured contracts -- see the verdict taxonomy and why
  safety.py                     SSRF guard (peer-IP validated, not just DNS-checked), redaction
  store.py                      Case persistence
  notify.py                     Deterministic notify/suppress policy, delivery, verification
  inbox.py                      The synthetic inbox trigger and end-to-end orchestration
  tools/evidence.py             Fetch, hostname comparison, and the domain-lock enforcement
tests/                          76 tests; everything except the live-only smoke test runs offline
ARCHITECTURE.md                 Diagrams and the reasoning behind the agent/tool split
docs/agentcore-iam-bootstrap-policy.json   The IAM policy AgentCore's auto-role-creation needs
```

## Known limitations

Stated plainly rather than left for a judge to discover:

- No curated organization → domain registry. The investigator determines the
  official domain from the model's own knowledge; the code enforces
  *consistency* against that declared domain, not that the declaration
  itself is correct. A convincing but wrong domain declaration is not
  currently caught.
- Case persistence is a local JSON-file store, appropriate for this
  submission's scope but not durable across AgentCore container recycling.
  The store/notify interfaces are the seam a production deployment would
  swap for DynamoDB or S3 without touching business logic.
- One organization per case. A message plausibly involving two unrelated
  organizations is handled by locking the one the requested action is
  actually about and noting the other as a limitation, not by evaluating
  both.
- No adversarial or prompt-injection red-teaming beyond the specific,
  disclosed defenses in the code (SSRF peer-IP validation, CoreBreak input
  validation, untrusted-content wrapping).

## License

MIT — see [`LICENSE`](LICENSE).
