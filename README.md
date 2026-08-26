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

**Watch it happen** — a small local web UI reads the same case store this
demo populates:
```bash
python -m smishsentinel.webui
# open http://127.0.0.1:8090/, click "Run demo inbox cycle"
```
Inbox message → background investigation → quiet-or-surfaced status →
evidence card with clickable, independently-fetched sources → a safe next
action. Most messages stay quiet on purpose; only consequential cases
surface with a colored badge. See [`webui.py`](smishsentinel/webui.py) for
why this is a separate local process from the deployed agent rather than
more routes on `app.py`.

## Architecture

Four specialist agents, composed rather than merged, plus a code layer that
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
- **"Official" is enforced, not asserted — and now the domain itself is
  looked up, not guessed.** The investigator must call `set_official_domain`
  before it can fetch anything or compare the message's hostname against
  anything, but that call takes only an organization name: the domain is
  resolved from a curated registry ([`registry.py`](smishsentinel/registry.py),
  [`data/organizations.json`](data/organizations.json)), never supplied or
  guessed by the model. That domain locks once per case; `fetch_official_page`
  refuses — before making any request — any URL that isn't on the locked
  domain or a subdomain of it. An organization outside the registry can't be
  locked at all, so the investigation abstains honestly
  (`insufficient_evidence`) instead of guessing at a plausible-looking
  domain. See [`tools/evidence.py`](smishsentinel/tools/evidence.py).
- **A downgraded verdict can't leave the rest of the card asserting the
  conclusion that was just withdrawn.** Citation enforcement doesn't stop at
  stripping bad evidence: a claim assessment needs *every* cited ID to
  verify, not merely one out of several, and `OFFICIAL_CONTRADICTION`
  specifically requires a verified `CONTRADICTED` claim assessment behind it
  — surviving evidence that isn't linked to a contradicted claim doesn't
  justify the verdict. Whenever any of that forces the top-level verdict to
  change, `risk_level`, `headline`, and `inferences` are reconciled to the
  new verdict in the same pass, so a card can't end up saying
  `insufficient_evidence` while its headline still states a contradiction.
  See `_reconcile_after_downgrade` in [`agent.py`](smishsentinel/agent.py).
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
  [`test_adversarial.py`](tests/test_adversarial.py) goes a step further and
  proves it offline: even a fully-hijacked, worst-case model output that
  complies with an injected instruction can't fabricate a citation or
  suppress a notification for a message that was actually investigated.
- **The pipeline itself has offline test coverage, not just its stages in
  isolation.** Citation enforcement, the domain lock, and the registry each
  have dedicated tests, but none of them call `investigate()` — the function
  that actually chains triage, claim extraction, investigation, and
  synthesis together. [`test_deterministic_eval.py`](tests/test_deterministic_eval.py)
  runs the real `investigate()`, the real tool calls, and the real
  `_enforce_citations` against injectable fake agents and recorded page
  fixtures ([`tests/fakes.py`](tests/fakes.py)) instead of live Bedrock, so a
  wiring bug between stages fails offline instead of only showing up against
  a real account. `smoke_test.py` remains the separate, manually-run live
  counterpart.
- **"Delivery" is scoped honestly, and suppression isn't called delivery.**
  There's no phone in this loop. A notification here means a durable,
  queryable record plus a log line — the signal a real channel integration
  would consume downstream — not a push notification that doesn't exist in
  this submission. `NotificationRecord` also separates two different claims
  that used to be one boolean: `decision_recorded` (the notify-or-suppress
  decision was made and persisted — true for every completed case) and
  `notification_delivered` (an actual notification was sent — false for a
  suppressed case, since nothing was). See [`notify.py`](smishsentinel/notify.py).
- **A failed investigation still notifies.** An exception during
  investigation used to leave a `FAILED` case record with no notification at
  all — a real gap, since the user was never told their message couldn't be
  checked. It's now treated as urgent, the same defensive default already
  used when a card is unexpectedly missing. See [`inbox.py`](smishsentinel/inbox.py).
- **An inbox cycle can't run unbounded.** Per-stage turn/token budgets in
  [`config.py`](smishsentinel/config.py) cap one message; they don't cap a
  whole `run_inbox_cycle()` of several messages. `INBOX_CYCLE_DEADLINE_SECONDS`
  is that missing ceiling — once elapsed time in a cycle passes it, every
  remaining message fails without a model call rather than continuing to
  spend time and money.
- **Persistence survives container recycling, when configured to.** The
  local JSON `CaseStore` was always documented as non-durable across
  AgentCore container recycling. `DynamoDBCaseStore` is the production swap
  the store/notify interfaces were built to allow — same `save`/`get`/
  `list_recent` contract, so `inbox.py`, `app.py`, and `webui.py` never need
  to know which backend is active. `get_case_store()` picks DynamoDB when
  `SMISH_CASE_TABLE` is set and the local JSON store otherwise (every
  offline test, and local development without a table). See
  [`store.py`](smishsentinel/store.py) and "Optional: durable persistence
  with DynamoDB" below for setup.

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

**Optional: durable persistence with DynamoDB.** By default this runs with
the local JSON `CaseStore` (zero setup, but doesn't survive AgentCore
container recycling). To switch to `DynamoDBCaseStore` instead:

```bash
# 1. Create the table (on-demand billing, no capacity planning needed for this scale)
aws dynamodb create-table \
  --table-name smishsentinel-cases \
  --attribute-definitions AttributeName=case_id,AttributeType=S \
  --key-schema AttributeName=case_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region us-east-1

# 2. Grant the deployed agent's execution role access to it -- the
#    auto-created role does NOT include this by default (Bedrock + logs +
#    X-Ray + ECR only). Fill in the placeholders in the policy file first.
aws iam put-role-policy \
  --role-name AmazonBedrockAgentCoreSDKRuntime-us-east-1-<your-hash> \
  --policy-name SmishSentinelCaseTableAccess \
  --policy-document file://docs/agentcore-execution-role-dynamodb-policy.json

# 3. Set the table name as an env var on the deployed agent and redeploy
agentcore deploy --env SMISH_CASE_TABLE=smishsentinel-cases
```

Locally, `SMISH_CASE_TABLE=smishsentinel-cases python app.py` switches the
same way (your own AWS credentials need equivalent DynamoDB permissions on
that table). Unset it, or don't set it at all, to keep using the JSON store
— that's what every offline test and `webui.py`'s local demo do.

## Repository layout

```
app.py                          AgentCore entrypoint: HTTP boundary, CoreBreak-safe validation
smishsentinel/
  agent.py                      The four-stage pipeline and citation/verdict reconciliation
  schemas.py                    Structured contracts -- see the verdict taxonomy and why
  safety.py                     SSRF guard (peer-IP validated, not just DNS-checked), redaction
  registry.py                   Curated organization -> official-domain registry
  store.py                      Case persistence -- local JSON store or DynamoDB, same interface
  notify.py                     Deterministic notify/suppress policy, delivery, verification
  inbox.py                      The synthetic inbox trigger and end-to-end orchestration
  webui.py                      Local demo UI: inbox -> evidence card -> safe action (stdlib-only)
  tools/evidence.py             Fetch, hostname comparison, and the domain-lock enforcement
data/organizations.json         The registry's data: ~15 curated organizations and their domains
tests/                          128 tests; everything except the live-only smoke test runs offline
  fakes.py                      Injectable fake agents + recorded page fixtures, not a test file itself
ARCHITECTURE.md                 Diagrams and the reasoning behind the agent/tool split
docs/agentcore-iam-bootstrap-policy.json           IAM policy for the deploying user (auto-role-creation)
docs/agentcore-execution-role-dynamodb-policy.json IAM policy for the deployed agent's own DynamoDB access
.github/workflows/tests.yml     CI: installs pinned dependencies, runs the offline suite on push/PR
```

## Known limitations

Stated plainly rather than left for a judge to discover:

- The organization registry is curated and deliberately small (around fifteen
  organizations — Canada Post, CRA, the major Canadian banks and telecoms,
  Amazon and the major couriers, a few others). An organization outside it
  produces an honest abstention (`insufficient_evidence`) rather than a
  guessed domain, which is the intended failure mode, but it does mean the
  agent can verify claims only about organizations someone has curated in
  advance — not a general-purpose entity-resolution system. See
  [`registry.py`](smishsentinel/registry.py) and
  [`data/organizations.json`](data/organizations.json).
- Case persistence defaults to a local JSON-file store — zero setup, but not
  durable across AgentCore container recycling. `DynamoDBCaseStore` is the
  durable alternative and is fully implemented ([`store.py`](smishsentinel/store.py)),
  but it's opt-in (`SMISH_CASE_TABLE`) because it needs a real table and an
  execution-role permission grant this repo can't create on your behalf —
  see "Optional: durable persistence with DynamoDB" above.
  `DynamoDBCaseStore.list_recent` does a full table Scan and sorts
  client-side rather than requiring a GSI on `updated_at`; fine at this
  submission's scale, not how you'd do it at production scale.
- One organization per case. A message plausibly involving two unrelated
  organizations is handled by locking the one the requested action is
  actually about and noting the other as a limitation, not by evaluating
  both.
- Adversarial coverage is real but narrow: [`test_adversarial.py`](tests/test_adversarial.py)
  proves the deterministic layers (citation verification, notify policy,
  the schema's no-safe-verdict invariant) hold even against a worst-case,
  fully-hijacked model output, and the specific, disclosed defenses in the
  code (SSRF peer-IP validation, CoreBreak input validation,
  untrusted-content wrapping) are each tested directly. This is not
  systematic red-teaming or a jailbreak corpus against the live model.

## License

MIT — see [`LICENSE`](LICENSE).
