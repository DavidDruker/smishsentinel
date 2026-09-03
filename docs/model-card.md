# Model card — smishing screener (`smishsentinel/ml_models/smishing_screener.joblib`)

Follows the standard model-card convention (Mitchell et al., 2019) so this artifact is
auditable on its own terms, separately from the agent pipeline it plugs into.

## What this model is

A classical TF-IDF (word 1–2 grams) + linear SVM classifier, Platt-calibrated
(`CalibratedClassifierCV`, `method="sigmoid"`, 5-fold), trained to distinguish
ham from "flag-worthy" (smishing or spam) SMS text.

## Intended use

Runs **only** on messages that already failed triage's gate in
[`agent.py`](../smishsentinel/agent.py) — i.e., messages with no identifiable
organization and consequential action named together. It never produces a
verdict and is never treated as evidence: a positive result reaches
`NotificationChannel.ADVISORY`, a category structurally distinct from an
investigated, citation-backed case (`STANDARD`/`URGENT`). See
`MLScreeningResult` in [`schemas.py`](../smishsentinel/schemas.py) and
`notify.decide()` for the enforced separation.

Not intended for: standalone use as a verdict-producing system, non-English
text, or as a defense against an adversary deliberately rewriting messages to
evade a lexical fingerprint (see Limitations).

## Training data

| Source | Rows contributed (post-dedup) | License |
|---|---|---|
| Mishra, S. & Soni, D. (2022). *SMS Phishing Dataset for Machine Learning and Pattern Recognition*. Mendeley Data, V1. DOI: [10.17632/f45bkkt8pr.1](https://doi.org/10.17632/f45bkkt8pr.1) | ham/smishing/spam, largest single source | CC BY 4.0 |
| Almeida, T. & Hidalgo, J. (2011). *SMS Spam Collection*. UCI Machine Learning Repository. [archive.ics.uci.edu/dataset/228](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | ham/spam only (no smishing subclass) | CC BY 4.0 |
| Munoz, M. & Islam, M. (2025). *A Balanced Dataset for Spam and Smishing Detection using LLMs*. Mendeley Data, V1. DOI: [10.17632/vmg875v4xs.1](https://doi.org/10.17632/vmg875v4xs.1) | ham/smishing/spam | CC BY 4.0 |

All three licenses were checked directly against each dataset's own published
license grant (not inferred) and explicitly permit sharing, copying, and
**modifying** the data for any purpose, provided attribution is given, a link
to the license is provided, and changes are indicated — which this notice
does. **Redistributing this derived `.joblib` artifact is therefore
permitted** under all three source licenses; no source restricts derivative
or commercial use. Further permission would only be needed for content
within a source identified as belonging to a still-further third party,
which none of the three licenses flag.

**Synthetic/mixed-provenance data**: the Munoz & Islam set's own documentation
states it was produced by "a trained Large Language Model... or alternatively,
collect[ing] real examples" — its own methodology does not distinguish which
rows are which. It is used here, but flagged as lower-trust than the other
two sources for that reason (see Limitations).

## Data processing

- **Label mapping**: the three-class label (`ham`/`spam`/`smishing`) is
  collapsed to binary for training — `ham` = negative, `spam` or `smishing` =
  positive ("flag-worthy") — matching the operational question this model
  answers (does this look like something worth a second look), not a
  fine-grained taxonomy.
- **Deduplication**: raw sources overlap heavily (UCI's ham/spam messages are
  ~87% duplicated inside Mishra & Soni; the Munoz & Islam set overlaps
  Mishra & Soni ~52%, consistent with its stated LLM-assisted construction).
  Exact-text duplicates were removed first (21,736 raw rows → 9,843). A
  second pass, done with Claude's assistance, identified and collapsed
  near-duplicate messages — the same bulk-scam template repeated with a
  phone number, name, or word changed, which exact-text matching does not
  catch — down to **5,060 rows** (ham 4,004 / smishing 599 / spam 457).
  Ham barely moved in that second pass; smishing and spam each dropped ~75%,
  indicating most of the raw volume in those classes was repeated campaign
  templates rather than independent examples.
- **Held-out separation from agent evaluation**: 1,093 messages already used
  to evaluate the deployed LLM agent (independent of this classifier) were
  excluded from training entirely, so they remain available as a clean check
  of the combined pipeline on cases the classifier has never seen.

## Train / validation / threshold-selection separation

- **Model comparison** (choosing SVM over Naive Bayes and Logistic
  Regression): stratified 80/20 split, TF-IDF fit on the training portion
  only.
- **Final shipped model**: refit on all 5,060 rows (maximizing data used),
  after model choice was already fixed from the step above — the 80/20 split
  is not reused for the artifact that ships.
- **Threshold selection**: the operating threshold is chosen via 5-fold
  cross-validation (`cross_val_predict`, TF-IDF refit inside each fold) on
  the full 5,060-row set, targeting recall ≥ 0.98 — never on the same split
  used for model comparison, and never on data the shipped model was
  evaluated against downstream.
- Full training/threshold-selection code: [`train_screener.py`](../smishsentinel/ml_models/train_screener.py)
  and [`build_training_data.py`](../smishsentinel/ml_models/build_training_data.py).

## Threshold

**0.166** (probability), chosen because the project's stated priority is
recall over precision — a missed smishing message (false negative) is judged
worse than an unnecessary advisory flag (false positive), since a positive
result here only ever triggers a lighter review, never an autonomous action.
Cross-validated estimate at this threshold: precision 90.2%, recall 98.0%,
smishing-specific recall 99.7% (see the artifact's own stored metadata for
the exact figures at export time).

## Limitations

- **A real portion of measured recall gains reflect template recognition,
  not pure generalization.** In a 750-message fresh evaluation (see
  [`evaluation.md`](evaluation.md)), of the messages the classifier correctly
  rescued that triage had missed, **70 of 156 (44.9%) had a ≥0.9 text
  similarity to a training example** — i.e., a near-duplicate of a campaign
  template already seen in training, not a genuinely novel message. Stripped
  to only the messages with low similarity to training (a fresher-looking
  test), recall was still substantially higher than triage alone, but this
  is reported as a **preliminary** result pending a fully independent,
  time-separated evaluation, not a confirmed generalization claim.
- **Classical bag-of-words, not adversarially robust.** Strong against the
  templated, lazily-varied scam text that dominates the training sources;
  weak by construction against an adversary who deliberately rewrites a
  message to avoid a lexical fingerprint.
- **English-only.**
- **Mixed-provenance source data**: one of the three training sources is
  partly LLM-generated by its own account (see above), with no per-row flag
  distinguishing synthetic from real examples.
- **No standalone accuracy claim**: this model was never evaluated as a
  freestanding classifier against an external benchmark — every reported
  number is in the context of the hybrid pipeline (triage + this model), per
  [`evaluation.md`](evaluation.md).

## Artifact metadata

Stored inside the artifact itself (`joblib.load(...)['metadata']`):
training timestamp, training-row count, per-label counts, the
cross-validated threshold-selection estimate, and a plain-text model
description. Load and inspect directly rather than trusting this document
alone for those figures — they are the source of truth.
