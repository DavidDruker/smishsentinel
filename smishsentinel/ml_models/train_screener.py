"""Trains smishing_screener.joblib -- the artifact ml_screen.py loads at
runtime. Committed for methodology transparency even though its input data
isn't: the training corpus (research_dump/training_data/combined_labeled_sms.csv,
~5,060 rows built from Mishra & Soni's Mendeley SMS phishing dataset plus the
UCI SMS Spam Collection, near-duplicate campaign templates collapsed) lives
in this project's gitignored research_dump/ the same way every other
evaluation dataset in this repo does -- it's derived from third-party
academic datasets, not original data, and isn't part of what's submitted.
Re-running this script requires that file to exist locally; the label
counts and cross-validated estimate below are recorded in the artifact's own
metadata regardless, so what the shipped model was actually trained and
validated on is auditable without needing to reproduce it byte for byte.

Not run automatically by anything -- this is a build step, run by hand
whenever the training corpus changes, not part of the request path or the
test suite.
"""
from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import fbeta_score, precision_score, precision_recall_curve, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).parent.parent.parent
DATA = REPO_ROOT / "research_dump" / "training_data" / "combined_labeled_sms.csv"
OUT_PATH = Path(__file__).parent / "smishing_screener.joblib"

TARGET_RECALL = 0.98  # recall weighted over precision: see MLScreeningResult's docstring on why


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("svm", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=5000, random_state=42), method="sigmoid", cv=5
        )),
    ])


def main() -> None:
    with open(DATA, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    texts = [r["text"] for r in rows]
    labels_3class = [r["label"] for r in rows]
    y = np.array([0 if lbl == "ham" else 1 for lbl in labels_3class])
    is_smishing = np.array([lbl == "smishing" for lbl in labels_3class])

    print(f"Training pool: {len(texts)} rows "
          f"({sum(y == 0)} ham, {sum(y == 1)} flag-worthy, {is_smishing.sum()} of which smishing)")

    # Honest out-of-fold probabilities: TF-IDF refit inside each outer fold,
    # not fit once on the full data first -- otherwise the vectorizer's
    # vocabulary/IDF weights would have seen every fold's held-out text
    # before that fold is scored.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba = cross_val_predict(make_pipeline(), texts, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y, oof_proba)
    candidates = np.where(recalls >= TARGET_RECALL)[0]
    best_idx = candidates[np.argmax(precisions[candidates])]
    threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])

    pred_at_threshold = (oof_proba >= threshold).astype(int)
    cv_precision = precision_score(y, pred_at_threshold, zero_division=0)
    cv_recall = recall_score(y, pred_at_threshold, zero_division=0)
    cv_f2 = fbeta_score(y, pred_at_threshold, beta=2, zero_division=0)
    cv_smishing_recall = pred_at_threshold[is_smishing].mean()

    print(f"\nCross-validated (5-fold) estimate at threshold={threshold:.4f}:")
    print(f"  precision={cv_precision:.3f} recall={cv_recall:.3f} F2={cv_f2:.3f} "
          f"smishing-recall={cv_smishing_recall:.3f}")

    # Refit on ALL data for the artifact actually shipped -- the CV pass above
    # is only ever used to pick an honest threshold, never as the model itself.
    final_pipeline = make_pipeline()
    final_pipeline.fit(texts, y)

    artifact = {
        "pipeline": final_pipeline,
        "threshold": threshold,
        "metadata": {
            "trained_at": datetime.now(UTC).isoformat(),
            "training_rows": len(texts),
            "label_counts": {lbl: labels_3class.count(lbl) for lbl in ("ham", "spam", "smishing")},
            "cv_estimate": {
                "precision": round(cv_precision, 4),
                "recall": round(cv_recall, 4),
                "f2": round(cv_f2, 4),
                "smishing_recall": round(cv_smishing_recall, 4),
                "folds": 5,
                "target_recall": TARGET_RECALL,
            },
            "model": "TF-IDF(1,2-gram) + LinearSVC(class_weight=balanced), Platt-calibrated",
        },
    }
    joblib.dump(artifact, OUT_PATH)
    print(f"\nSaved artifact to {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.1f} KB)")
    print(json.dumps(artifact["metadata"], indent=2))


if __name__ == "__main__":
    main()
