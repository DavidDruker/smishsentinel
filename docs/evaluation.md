# Evaluation history

How the triage-plus-classifier design was reached, and what the evidence for
it actually supports. Written in the same claim-ladder spirit as the
project's own prior research (see [`disclosures.md`](disclosures.md)):
supported, suggestive, and unconfirmed are kept separate rather than
collapsed into one headline number.

## 1. Baseline: LLM-only triage against real data

Two evaluation runs against the deployed agent, using real messages from the
Mishra & Soni SMS phishing dataset (see [`model-card.md`](model-card.md) for
licensing): 600 randomly sampled cases, then an independently-drawn 500-case
run with zero text overlap with the first.

These runs surfaced two real triage-prompt defects, both confirmed against
the false negatives that motivated them and both fixed by editing
`_TRIAGE_PROMPT` in [`agent.py`](../smishsentinel/agent.py):

1. Triage was overriding its own "consequential action" rule with an
   unauthorized judgment about whether an action "felt risky" or "seemed
   like ordinary commercial activity" — a verifiability question it has no
   tools to answer, and not one it was asked to answer.
2. Triage was overriding its own "identifiable organization" rule (a name is
   given) with an unauthorized judgment about whether that name sounded like
   a real, established company — again, a question for the registry lookup
   downstream, not for a tool-free gate.

Both fixes were verified by redeploying and re-running the specific false
negatives they targeted; both flipped to correctly flagged.

## 2. The remaining gap was structural, not a prompt bug

After both fixes, false negatives persisted specifically on messages naming
**no organization at all** — classic premium-rate/reverse-billing scam text
("you have WON, call this number"). Triage's own rule is correct to let
these through: it explicitly requires both a named organization and a
consequential action, and the investigation stage that follows can only
verify an organizational claim against the registry — it has nothing to
check when no organization is named. No prompt wording closes a gap that is
about what the downstream tools can do, not about model judgment. This is
what motivated the classifier: a different kind of check for a category of
message the rule-based gate cannot help by construction.

## 3. Hybrid evaluation: 750 fresh cases

**Sample**: 750 messages, stratified across ham/spam/smishing, drawn from the
combined pool (Mishra & Soni + UCI SMS Spam Collection + the Munoz & Islam
balanced set) with the classifier's own training rows excluded. Includes
messages from the earlier 600/500-case evaluation pool — those were never
seen by the classifier during training, only by the LLM agent, so re-running
them is a valid test of the new combined system.

**Scoring**: `predicted_positive = triage.warrants_investigation OR
ml_screening.flagged` (hybrid) vs. `predicted_positive = triage.warrants_investigation`
alone (baseline), against ground truth (`smishing` = positive, `ham` =
negative; `spam` tracked separately, out of this project's stated scope).

| | Triage alone | Hybrid |
|---|---|---|
| Recall | 50.3% | 99.7% |
| Precision | 100.0% | 98.7% |
| Accuracy | 68.8% | 99.0% |
| False negatives | 153 | 1 |

**Paired significance**: an exact two-sided McNemar test (matching the
statistical methodology in the project's prior research) on the same 491
cases: 152 cases correct only under the hybrid system, 4 correct only under
triage alone, **p = 5.3 × 10⁻⁴⁰**. This is a real, strong paired result — not
a borderline one requiring a fresh holdout to interpret.

**Run health**: 750/750 completed, 4 errors (0.5%) — 1 client-side response
issue, 3 server-side transient errors with no application-level traceback in
either CloudWatch log stream for the deployed agent, meaning the failures
did not reach the pipeline code. Not clearly attributable to the new code
path.

## 4. What is and isn't established

Following the same claim-ladder discipline as the project's prior research:

- **Supported**: on this 750-case sample, the hybrid system's classification
  decisions are significantly better than triage alone (McNemar
  p = 5.3 × 10⁻⁴⁰). Both triage prompt fixes are independently confirmed
  against the specific false negatives they targeted.
- **Preliminary, not established**: the size of the recall improvement.
  70 of the 156 rescued cases (44.9%) closely resemble a message already in
  the classifier's training data (see [`model-card.md`](model-card.md)'s
  Limitations) — meaning a real portion of the headline gain reflects
  recognizing a known template, not confirmed generalization to unfamiliar
  scam phrasing. Directionally positive on lower-similarity messages too,
  but not yet confirmed on a fully independent, time-separated sample.
- **Not tested**: adversarial robustness (a rewritten message specifically
  targeting the classifier's lexical features), deployed user behavior, and
  whether the citation-enforcement layer's grounding guarantees hold up
  under a manual audit of the kind the project's prior research applied to
  SmishX (see [`disclosures.md`](disclosures.md)) — that audit has not yet
  been run against this system.

Raw results: `research_dump/eval_runs/eval_750_results.json` and
`eval_750_scored.json` (local, gitignored alongside every other evaluation
dataset in this project — not part of the submission's tracked files).
