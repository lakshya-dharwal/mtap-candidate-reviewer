"""Primary model: regularized logistic regression on CNA features.

Trains -2 (positive) vs {0,1,2} (reference) using the ~150-gene GISTIC panel
(MTAP excluded). Reports cross-validated and held-out ROC-AUC, PR-AUC,
sensitivity, specificity, F1, and confusion matrix. Saves the model refit on
all labeled data (for scoring the -1 set), plus metrics.json and provenance.json.

`train_and_evaluate` takes an explicit feature list so Phase 5 ablations can
re-run it on gene subsets without changing anything else.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import data_prep

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_VERSION = "0.1.0"
RANDOM_STATE = 42


def make_pipeline():
    """StandardScaler + L2 logistic regression, class-weight balanced."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2", C=1.0, class_weight="balanced",
            solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])


def _metrics_at_threshold(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1_score(y_true, pred, zero_division=0), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def train_and_evaluate(X, y, feature_genes, label="main", random_state=RANDOM_STATE):
    """Train + evaluate on the labeled subset restricted to `feature_genes`.

    Returns (metrics_dict, pipeline_fit_on_all_labeled).
    """
    labeled = y.notna()
    Xl = X.loc[labeled, feature_genes]
    yl = y.loc[labeled].astype(int)
    n_pos, n_neg = int((yl == 1).sum()), int((yl == 0).sum())

    # Cross-validated out-of-fold probabilities (whole labeled set).
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    oof = cross_val_predict(
        make_pipeline(), Xl, yl, cv=skf, method="predict_proba"
    )[:, 1]
    cv_roc = roc_auc_score(yl, oof)
    cv_pr = average_precision_score(yl, oof)
    cv_thr = _metrics_at_threshold(yl.values, oof)

    # Held-out 80/20 for an independent confusion matrix.
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xl, yl, test_size=0.20, stratify=yl, random_state=random_state
    )
    pipe = make_pipeline().fit(X_tr, y_tr)
    te_proba = pipe.predict_proba(X_te)[:, 1]
    test_roc = roc_auc_score(y_te, te_proba)
    test_pr = average_precision_score(y_te, te_proba)
    test_thr = _metrics_at_threshold(y_te.values, te_proba)

    metrics = {
        "label": label,
        "n_features": len(feature_genes),
        "n_positive": n_pos,
        "n_reference": n_neg,
        "cv_roc_auc": round(cv_roc, 4),
        "cv_pr_auc": round(cv_pr, 4),
        "cv_5fold": cv_thr,
        "test_n": int(len(y_te)),
        "test_roc_auc": round(test_roc, 4),
        "test_pr_auc": round(test_pr, 4),
        "test_at_0.5": test_thr,
    }

    # Refit on ALL labeled data for inference scoring of the -1 set.
    final = make_pipeline().fit(Xl, yl)
    return metrics, final


def print_metrics(m):
    print(f"\n=== model: {m['label']} "
          f"(features={m['n_features']}, pos={m['n_positive']}, ref={m['n_reference']}) ===")
    print(f"  CV(5-fold)  ROC-AUC={m['cv_roc_auc']}  PR-AUC={m['cv_pr_auc']}  "
          f"sens={m['cv_5fold']['sensitivity']}  spec={m['cv_5fold']['specificity']}  "
          f"F1={m['cv_5fold']['f1']}")
    cm = m["cv_5fold"]["confusion_matrix"]
    print(f"  CV confusion: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"  Test(20%,n={m['test_n']})  ROC-AUC={m['test_roc_auc']}  "
          f"PR-AUC={m['test_pr_auc']}  sens={m['test_at_0.5']['sensitivity']}  "
          f"spec={m['test_at_0.5']['specificity']}  F1={m['test_at_0.5']['f1']}")
    tcm = m["test_at_0.5"]["confusion_matrix"]
    print(f"  Test confusion: TN={tcm['tn']} FP={tcm['fp']} FN={tcm['fn']} TP={tcm['tp']}")


def run_ablations():
    """Phase 5: retrain the primary model on three feature sets, report side by side.

    Identical setup each time (L2, class_weight=balanced, stratified 80/20,
    random_state=42, 5-fold CV); only the feature columns change.
    """
    import genes as G

    os.makedirs(MODELS_DIR, exist_ok=True)
    ds = data_prep.build_dataset()
    data_prep.summarize(ds)
    all_feats = ds["feature_genes"]  # MTAP already excluded

    # 9p21 neighborhood present in the panel (MTAP is the label, not a feature).
    nine_p21_features = [g for g in G.NINE_P21_SYMBOLS
                         if g != "MTAP" and g in all_feats]

    configs = [
        ("with_CDKN2A", all_feats, []),
        ("without_CDKN2A", [g for g in all_feats if g != "CDKN2A"], ["CDKN2A"]),
        ("without_9p21_neighborhood",
         [g for g in all_feats if g not in nine_p21_features],
         list(nine_p21_features)),
    ]

    results = []
    for label, feats, dropped in configs:
        m, _ = train_and_evaluate(ds["X"], ds["y"], feats, label=label)
        m["dropped_genes"] = dropped
        results.append(m)
        print_metrics(m)

    # Side-by-side table.
    print("\n=== THREE-WAY ABLATION (side by side) ===")
    hdr = f"{'model':<28}{'feats':>6}{'CV ROC':>9}{'CV PR':>8}{'sens':>7}{'spec':>7}{'F1':>7}{'test ROC':>10}"
    print(hdr)
    print("-" * len(hdr))
    for m in results:
        cv = m["cv_5fold"]
        print(f"{m['label']:<28}{m['n_features']:>6}{m['cv_roc_auc']:>9}{m['cv_pr_auc']:>8}"
              f"{cv['sensitivity']:>7}{cv['specificity']:>7}{cv['f1']:>7}{m['test_roc_auc']:>10}")

    out = {
        "cohort": ds["study"],
        "random_seed": RANDOM_STATE,
        "model": "L2 logistic regression (StandardScaler), class_weight=balanced",
        "nine_p21_features_dropped_in_model3": list(nine_p21_features),
        "models": results,
    }
    with open(os.path.join(MODELS_DIR, "ablation_metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[model] saved ablation_metrics.json -> {MODELS_DIR}")
    return out


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    ds = data_prep.build_dataset()
    data_prep.summarize(ds)

    metrics, final = train_and_evaluate(
        ds["X"], ds["y"], ds["feature_genes"], label="main_with_CDKN2A"
    )
    print_metrics(metrics)

    # Persist.
    import joblib
    joblib.dump(
        {"pipeline": final, "feature_genes": ds["feature_genes"],
         "model_version": MODEL_VERSION},
        os.path.join(MODELS_DIR, "model_main.joblib"),
    )
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    role = ds["role"]
    provenance = {
        "cohort": ds["study"],
        "data_pull_date": "2026-07-07",
        "run_date": _dt.date.today().isoformat(),
        "random_seed": RANDOM_STATE,
        "model_version": MODEL_VERSION,
        "model": "L2 logistic regression (StandardScaler), class_weight=balanced",
        "n_features": len(ds["feature_genes"]),
        "sample_counts": {
            "labelable_total": int(len(ds["X"])),
            "positive_-2": int((role == "positive").sum()),
            "reference_0_1_2": int((role == "reference").sum()),
            "ambiguous_-1": int((role == "ambiguous").sum()),
        },
        "exclusions_logged": int(len(ds["exclusions"])),
        "label_definition": {
            "-2": "positive (confirmed homozygous deletion)",
            "0,1,2": "reference (copy-number non-deleted)",
            "-1": "held out, scored at inference (ambiguous shallow loss)",
        },
        "leakage_controls": "MTAP excluded from features; MTAP expression not used as feature",
    }
    with open(os.path.join(MODELS_DIR, "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)
    print(f"\n[model] saved model_main.joblib, metrics.json, provenance.json -> {MODELS_DIR}")


if __name__ == "__main__":
    import sys
    if "--ablations" in sys.argv:
        run_ablations()
    else:
        main()
