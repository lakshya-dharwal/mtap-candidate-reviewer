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
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
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


def bootstrap_auc_ci(y_true, scores, n_boot=1000, seed=RANDOM_STATE, alpha=0.05):
    """Percentile bootstrap 95% CI for ROC-AUC and PR-AUC over (y_true, scores).

    Resamples the already-computed CV out-of-fold predictions (no retraining) —
    cheap, and gives an honest sense of estimate uncertainty given the small
    positive count instead of only reporting a single point estimate.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    roc_boot, pr_boot = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.min() == yb.max():
            continue  # degenerate resample (all one class) — skip
        roc_boot.append(roc_auc_score(yb, scores[idx]))
        pr_boot.append(average_precision_score(yb, scores[idx]))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "roc_auc_ci95": [round(float(np.percentile(roc_boot, lo)), 4),
                        round(float(np.percentile(roc_boot, hi)), 4)],
        "pr_auc_ci95": [round(float(np.percentile(pr_boot, lo)), 4),
                       round(float(np.percentile(pr_boot, hi)), 4)],
        "n_boot": len(roc_boot),
    }


def youden_optimal_threshold(y_true, scores):
    """Threshold maximizing Youden's J (sensitivity + specificity - 1).

    class_weight='balanced' shifts the natural decision boundary away from 0.5;
    this reports the data-driven optimal cut alongside the fixed-0.5 metrics
    rather than treating 0.5 as if it were principled.
    """
    fpr, tpr, thr = roc_curve(y_true, scores)
    j = tpr - fpr
    best = int(np.argmax(j))
    # roc_curve's first threshold is +inf by convention; guard against it.
    t = float(thr[best]) if np.isfinite(thr[best]) else float(np.max(scores))
    return t, j[best]


def train_and_evaluate(X, y, feature_genes, label="main", random_state=RANDOM_STATE,
                       capture_curves=False):
    """Train + evaluate on the labeled subset restricted to `feature_genes`.

    Returns (metrics_dict, pipeline_fit_on_all_labeled). If capture_curves is
    True, adds ROC/PR curve arrays (from the CV out-of-fold predictions) to the
    metrics dict — purely additive, does not affect any scalar metric.
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
    cv_ci = bootstrap_auc_ci(yl.values, oof, seed=random_state)
    youden_t, youden_j = youden_optimal_threshold(yl.values, oof)
    cv_youden = _metrics_at_threshold(yl.values, oof, thr=youden_t)

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
        "cv_auc_ci95": cv_ci,
        "cv_5fold": cv_thr,
        "cv_youden_threshold": round(youden_t, 4),
        "cv_youden_j": round(float(youden_j), 4),
        "cv_at_youden": cv_youden,
        "test_n": int(len(y_te)),
        "test_roc_auc": round(test_roc, 4),
        "test_pr_auc": round(test_pr, 4),
        "test_at_0.5": test_thr,
    }

    if capture_curves:
        fpr, tpr, _ = roc_curve(yl, oof)
        prec, rec, _ = precision_recall_curve(yl, oof)
        metrics["roc_curve"] = {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()}
        metrics["pr_curve"] = {"precision": prec.round(5).tolist(),
                               "recall": rec.round(5).tolist()}

    # Refit on ALL labeled data for inference scoring of the -1 set.
    final = make_pipeline().fit(Xl, yl)
    return metrics, final


def print_metrics(m):
    print(f"\n=== model: {m['label']} "
          f"(features={m['n_features']}, pos={m['n_positive']}, ref={m['n_reference']}) ===")
    ci = m.get("cv_auc_ci95", {})
    print(f"  CV(5-fold)  ROC-AUC={m['cv_roc_auc']} (95% CI {ci.get('roc_auc_ci95')})  "
          f"PR-AUC={m['cv_pr_auc']} (95% CI {ci.get('pr_auc_ci95')})  "
          f"sens={m['cv_5fold']['sensitivity']}  spec={m['cv_5fold']['specificity']}  "
          f"F1={m['cv_5fold']['f1']}")
    if "cv_youden_threshold" in m:
        yj = m["cv_at_youden"]
        print(f"  Youden-J optimal threshold={m['cv_youden_threshold']} (J={m['cv_youden_j']})  "
              f"sens={yj['sensitivity']}  spec={yj['specificity']}  F1={yj['f1']}  "
              f"[vs fixed 0.5 above]")
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

    # Permutation/null control: is the AUC drop specific to 9p21, or would
    # dropping ANY 2 genes from the panel do about the same? Repeatedly drop 2
    # random non-9p21 genes and rebuild the CV ROC-AUC distribution.
    n_drop = len(nine_p21_features)
    n_reps = 20
    rng = np.random.default_rng(RANDOM_STATE)
    non_9p21 = [g for g in all_feats if g not in nine_p21_features]
    null_aucs = []
    for i in range(n_reps):
        drop = rng.choice(non_9p21, size=n_drop, replace=False).tolist()
        feats_i = [g for g in all_feats if g not in drop]
        mi, _ = train_and_evaluate(ds["X"], ds["y"], feats_i,
                                   label=f"null_drop_{i}", random_state=RANDOM_STATE)
        null_aucs.append(mi["cv_roc_auc"])
    null_aucs = np.array(null_aucs)
    true_auc = next(m["cv_roc_auc"] for m in results
                    if m["label"] == "without_9p21_neighborhood")
    percentile_of_true = float((null_aucs <= true_auc).mean() * 100)
    permutation_control = {
        "n_replicates": n_reps,
        "n_genes_dropped_each_rep": n_drop,
        "null_auc_mean": round(float(null_aucs.mean()), 4),
        "null_auc_std": round(float(null_aucs.std()), 4),
        "null_auc_min": round(float(null_aucs.min()), 4),
        "null_auc_max": round(float(null_aucs.max()), 4),
        "without_9p21_auc": true_auc,
        "without_9p21_percentile_within_null": round(percentile_of_true, 1),
        "interpretation": (
            f"Dropping the 2 9p21 genes (CV ROC-AUC {true_auc}) falls at the "
            f"{percentile_of_true:.0f}th percentile of {n_reps} random 2-gene drops "
            f"(null mean {null_aucs.mean():.4f} +/- {null_aucs.std():.4f}). "
            f"{'Lower than essentially all random drops' if percentile_of_true <= 5 else 'Not clearly distinguishable from a random 2-gene drop'} "
            f"of this size."
        ),
    }
    print(f"\n=== PERMUTATION CONTROL ({n_reps} random 2-gene drops) ===")
    print(f"  null AUC: mean={permutation_control['null_auc_mean']} "
          f"std={permutation_control['null_auc_std']} "
          f"range=[{permutation_control['null_auc_min']}, {permutation_control['null_auc_max']}]")
    print(f"  without_9p21 AUC={true_auc} -> percentile {percentile_of_true:.0f} within null")
    print(f"  {permutation_control['interpretation']}")

    out = {
        "cohort": ds["study"],
        "random_seed": RANDOM_STATE,
        "model": "L2 logistic regression (StandardScaler), class_weight=balanced",
        "nine_p21_features_dropped_in_model3": list(nine_p21_features),
        "models": results,
        "permutation_control": permutation_control,
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
        ds["X"], ds["y"], ds["feature_genes"], label="main_with_CDKN2A",
        capture_curves=True,
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
    # Derive the actual pull date from the earliest cached raw-data file's mtime
    # rather than a hardcoded literal — a hardcoded date silently goes stale the
    # moment the cache is rebuilt on a different day.
    raw_cache_files = [
        os.path.join(data_prep.CACHE_DIR, f"cna_panel_{ds['study']}.csv"),
        os.path.join(data_prep.CACHE_DIR, f"merged_{ds['study']}.csv"),
    ]
    existing_mtimes = [os.path.getmtime(p) for p in raw_cache_files if os.path.exists(p)]
    data_pull_date = (_dt.date.fromtimestamp(min(existing_mtimes)).isoformat()
                      if existing_mtimes else None)
    provenance = {
        "cohort": ds["study"],
        "data_pull_date": data_pull_date,
        "data_pull_date_source": "min(mtime of cached raw-data CSVs)",
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
        "reference_class_composition": {
            "note": "the 'reference' class pools 3 distinct GISTIC calls — "
                    "gained/amplified tumors are not copy-number-neutral",
            "neutral_0": int((ds["mtap_cna"] == 0).sum()),
            "gain_1": int((ds["mtap_cna"] == 1).sum()),
            "amplified_2": int((ds["mtap_cna"] == 2).sum()),
        },
        "exclusions_logged": int(len(ds["exclusions"])),
        "label_definition": {
            "-2": "positive (confirmed homozygous deletion)",
            "0,1,2": "reference (copy-number non-deleted)",
            "-1": "held out, scored at inference (ambiguous shallow loss)",
        },
        "leakage_controls": "MTAP excluded from features; MTAP expression not used as feature",
        "deployed_vs_evaluated_model": (
            "The CV and held-out-test metrics in metrics.json come from models "
            "fit inside train_and_evaluate's internal splits. The model actually "
            "saved to model_main.joblib and used to score the -1 queue is a THIRD "
            "fit — refit on 100% of labeled data — which has no independent "
            "holdout of its own. Its true generalization is only estimated via "
            "the CV number above, not directly measured."
        ),
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
