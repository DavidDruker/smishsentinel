"""Builds research_dump/training_data/combined_labeled_sms.csv, the input
train_screener.py trains on. Committed for the same reason that script is:
methodology transparency, even though the raw source datasets and the
research_dump/ output both live outside git (see train_screener.py's
docstring). Consolidates three third-party labelled SMS datasets (Mishra &
Soni's Mendeley SMS phishing dataset, the UCI SMS Spam Collection, and a
GWU-published "balanced" Mendeley set of mixed LLM-generated/real messages),
removes exact duplicates giving priority to the cleanest-provenance source,
and holds out anything that overlaps the message sets already used to
evaluate the deployed LLM agent (research_dump/eval_runs/) so that overlap
never leaks into classifier training.

Deliberately not the whole cleaning pipeline: a further pass collapsing
near-duplicate campaign templates (the same scam message with a phone
number, name, or word changed -- these datasets are full of them) runs on
this script's output before train_screener.py sees it. Not included here.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent

# Raw source archives -- downloaded locally, not committed (see .gitignore's
# *.csv rule); re-running this script requires them to exist at these paths.
_MISHRA_PATH = (
    REPO_ROOT / "scratch_dataset" / "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION"
    / "Dataset_5971.csv"
)
_BALANCED_PATH = REPO_ROOT / "scratch_dataset_balanced" / "A Balanced Dataset for Spam and Smishing Detection" / "Dataset_10191.csv"
_UCI_PATH = REPO_ROOT / "scratch_dataset_uci" / "SMSSpamCollection"


def _load_mishra_soni() -> list[dict]:
    with open(_MISHRA_PATH, encoding="cp1252") as f:
        reader = csv.reader(f)
        next(reader)
        return [
            {"source": "mishra_soni", "label": r[0].strip().lower(), "text": r[1], "synthetic": False}
            for r in reader if r and r[1].strip()
        ]


def _load_gwu_balanced() -> list[dict]:
    for enc in ("cp1252", "utf-8"):
        try:
            with open(_BALANCED_PATH, encoding=enc) as f:
                reader = csv.reader(f)
                next(reader)
                rows = [
                    {"source": "gwu_balanced", "label": r[0].strip().lower(), "text": r[1], "synthetic": True}
                    for r in reader if r and len(r) > 1 and r[1].strip()
                ]
            if "�" not in " ".join(r["text"] for r in rows[:200]):
                return rows
        except UnicodeDecodeError:
            continue
    return rows


def _load_uci() -> list[dict]:
    for enc in ("utf-8", "latin-1"):
        try:
            rows = []
            with open(_UCI_PATH, encoding=enc) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    label, _, text = line.partition("\t")
                    rows.append({"source": "uci_spam_collection", "label": label.strip().lower(), "text": text, "synthetic": False})
            return rows
        except UnicodeDecodeError:
            continue
    return []


def main() -> None:
    all_rows = _load_mishra_soni() + _load_gwu_balanced() + _load_uci()

    print("=== Raw counts by source x label (before any dedup) ===")
    for (source, label), n in sorted(Counter((r["source"], r["label"]) for r in all_rows).items()):
        print(f"  {source:<22} {label:<10} {n}")
    print(f"  TOTAL raw rows: {len(all_rows)}")

    # Exact-duplicate removal: keep first occurrence, priority = cleanest-provenance source first.
    priority = ["mishra_soni", "uci_spam_collection", "gwu_balanced"]
    seen: set[str] = set()
    deduped = []
    for r in sorted(all_rows, key=lambda r: priority.index(r["source"])):
        key = r["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    print(f"\nAfter exact-duplicate removal: {len(deduped)} rows (removed {len(all_rows) - len(deduped)})")

    # Hold out anything overlapping the agent-eval sample sets already scored
    # against the deployed LLM agent -- keeps them available later as a clean
    # check of whether the classifier catches what that eval run's triage missed.
    eval_dir = REPO_ROOT / "research_dump" / "eval_runs"
    with open(eval_dir / "eval_600_results.json", encoding="utf-8") as f:
        eval_600_texts = {r["text"].strip() for r in json.load(f)}
    with open(eval_dir / "eval_500_independent_results.json", encoding="utf-8") as f:
        eval_500_texts = {r["text"].strip() for r in json.load(f)}
    eval_texts_all = eval_600_texts | eval_500_texts

    training_final = [r for r in deduped if r["text"].strip() not in eval_texts_all]
    eval_holdout = [r for r in deduped if r["text"].strip() in eval_texts_all]
    print(f"Training pool: {len(training_final)}  |  eval-overlap holdout: {len(eval_holdout)}")

    out_dir = REPO_ROOT / "research_dump" / "training_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(path: Path, rows: list[dict]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "label", "text", "synthetic_or_mixed_provenance"])
            for r in rows:
                writer.writerow([r["source"], r["label"], r["text"], r["synthetic"]])

    _write(out_dir / "combined_labeled_sms.csv", training_final)
    _write(out_dir / "eval_overlap_holdout.csv", eval_holdout)
    print(f"\nSaved {len(training_final)} rows to {out_dir / 'combined_labeled_sms.csv'} "
          f"(before the near-duplicate-template collapse -- see this module's docstring)")


if __name__ == "__main__":
    main()
