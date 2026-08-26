# Evaluation

This is a systems/functional verification report: does the pipeline do what
it claims, correctly and safely, under real conditions. It is explicitly
**not** a statistical accuracy evaluation — there is no labeled benchmark
run here, no precision/recall/F1 number, and this document makes no
detection-accuracy claim. Given the project's own stance that no verdict
here means "safe," a false confidence in an unmeasured accuracy number would
be a worse failure than admitting one wasn't measured. A properly
constructed, independently-sourced labeled cohort is future work, not
something this submission's timeline supported doing rigorously.

## Automated test suite

76 tests, 75 of which run fully offline (no AWS credentials, no network) —
verifiable with `python -m unittest discover -s tests`.

| File | What it verifies |
|---|---|
| `test_safety.py` | SSRF guard rejects private ranges, cloud metadata endpoints, non-HTTP schemes, unusual ports; a live TLS loopback test proves the guarded HTTPS connection actually completes real requests, not just that it blocks bad ones; prompt-injection content wrapping. |
| `test_pipeline_offline.py` | Citation verification: nine tests specifically named for the forgery shape they block (fabricated quote on a real ID, citation against a 404, mismatched URL, phantom ID, `is_first_party` overridden regardless of what the model claims). |
| `test_evidence_tools.py` | The domain-lock: fetch/comparison blocked before a domain is locked, blocked when the URL doesn't match, allowed on subdomains. A genuine multi-threaded concurrency test, forced to overlap via a `Barrier`, proving two simultaneous investigations cannot see each other's evidence ledger. |
| `test_inbox_pipeline.py` | Store persistence round-trips; every branch of the notify policy; delivery and independent verification; the inbox cycle's success, suppression, and failure paths (`investigate()` mocked — this file tests the surrounding machinery, not model reasoning). |
| `test_app_http.py` | The HTTP boundary returns real 4xx status codes (not a 200 with an error-shaped body) for missing fields, empty text, non-string `text` (the CoreBreak-shaped attack), and over-length input — via an in-process Starlette `TestClient`, no live server needed. |

The one live-only path (an end-to-end smoke test against real Bedrock) is
excluded from the offline count by design, not oversight — it needs real AWS
credentials and costs real (small) money per run.

## Live deployment verification

Manually verified against the actual deployed AgentCore Runtime, not just
locally:

- **Ordinary message** ("running late for dinner") → triage correctly stayed
  silent, no investigation, no card.
- **Phishing message with a real, live domain mismatch** (a fake Canada Post
  redelivery-fee message) → investigated, fetched Canada Post's real
  website, correctly identified the hostname mismatch, produced a card with
  a genuine citation (a real evidence ID, a real quoted excerpt from the
  actually-retrieved page) rather than an empty evidence list.
- **End-to-end action** (`{"action": "run_inbox_cycle"}`) → all five
  synthetic messages persisted, correctly suppressed or notified, and
  independently re-verified from the store afterward. See the case-by-case
  result in the commit history around the `run_inbox_cycle` feature for the
  exact response.

## Defects found during this verification process, and fixed

Listed because finding and fixing these is itself part of what an evaluation
should show, not something to bury in commit messages:

1. **DNS-rebinding TOCTOU** in the original SSRF guard — validated a
   hostname's DNS answer, then let a separate HTTP call re-resolve
   independently at connect time. Fixed by validating the live connected
   socket's peer address instead of a prior DNS lookup.
2. **A dead code path in redirect handling** — with automatic redirects
   disabled, this urllib version raises `HTTPError` for 3xx responses rather
   than returning them normally, so the redirect-handling branch inside the
   success path never actually ran; every redirect silently returned an
   empty result. Caught by a live TLS loopback test, not by inspection.
3. **HTTPS fetches were completely broken** — a guarded connection handler
   referenced a urllib internal attribute that doesn't exist on this Python
   version. The HTTP-only wiring test stayed green throughout and masked
   this entirely, since virtually every real evidence source is HTTPS.
4. **Retrieved evidence wasn't reaching synthesis** — the synthesis stage
   only saw the investigator's own narrative summary of a fetch, not the
   actual page text, so a successful fetch produced no usable citation. The
   model correctly refused to fabricate a quote it didn't have; the fix was
   giving it the real text.
5. **Citation verification checked only ID membership** (found via external
   review) — a fabricated quote or URL survived by reusing any real evidence
   ID, including one representing a 404. Rewritten to check URL match, fetch
   success, and verbatim quote presence per item.
6. **The evidence ledger was a shared module-global** (found via external
   review) — a real concurrency bug under AgentCore's thread-pooled
   invocation handling, confirmed directly in this project's own CloudWatch
   stack traces. Fixed with `contextvars.ContextVar`.
7. **"Official" domain was model-asserted with nothing enforcing it** (found
   via external review) — added the `set_official_domain` lock described in
   `ARCHITECTURE.md`.
8. **The case store's default path wasn't writable in the deployed
   container** — a relative default resolved against `/app`, the
   application's own source directory. Caught on the first live invocation
   of the end-to-end action; fixed by defaulting under the system temp
   directory.

## What this evaluation does not cover

- No adversarial red-teaming beyond the specific, disclosed defenses in the
  code itself (SSRF peer-IP validation, CoreBreak input validation,
  untrusted-content wrapping for fetched pages).
- No measurement of detection accuracy, false-positive rate, or
  false-negative rate against any labeled dataset.
- No load testing beyond the deliberate multi-thread concurrency test in
  `test_evidence_tools.py`.
- No evaluation of whether the evidence card format actually changes a
  real user's behavior — that's a product question this systems-level
  submission doesn't attempt to answer.
