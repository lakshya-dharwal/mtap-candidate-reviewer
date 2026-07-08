"""Stage 2: deletion-burden baseline.

Answers the sharpest critique of the full model — "is it just learning overall
deletion load, not MTAP-specific context?" — by training a logistic model on
only two crude burden features and comparing it to the full 150-gene model.

Features (from the existing cached CNA matrix, no new pull):
  - genome_deletion_burden : fraction of the 150 panel genes with GISTIC < 0
  - chr9p_deletion_burden  : fraction of the panel's chr-9p genes with GISTIC < 0

Same label (-2 vs reference) and the SAME training/eval routine as the primary
model (model.train_and_evaluate), so the comparison is apples-to-apples.
"""

from __future__ import annotations

import json
import os

import pandas as pd

import data_prep
import genes as G
import model as M

OUT_PATH = os.path.join(M.MODELS_DIR, "deletion_burden_metrics.json")


def compute_burden_features(ds):
    """Return DataFrame (index=sampleId) with the two burden features."""
    X = ds["X"]
    feats = ds["feature_genes"]
    genome_burden = (X[feats] < 0).sum(axis=1) / len(feats)
    chr9p = [g for g in G.CHR9P_SYMBOLS if g in feats]
    chr9p_burden = (X[chr9p] < 0).sum(axis=1) / len(chr9p)
    return pd.DataFrame({
        "genome_deletion_burden": genome_burden,
        "chr9p_deletion_burden": chr9p_burden,
    }, index=X.index)


def main():
    os.makedirs(M.MODELS_DIR, exist_ok=True)
    ds = data_prep.build_dataset()
    burden = compute_burden_features(ds)
    burden_feats = ["genome_deletion_burden", "chr9p_deletion_burden"]

    metrics, _ = M.train_and_evaluate(
        burden, ds["y"], burden_feats,
        label="deletion_burden_baseline", capture_curves=True,
    )

    # Compare to the already-saved full (primary) model.
    full = json.load(open(os.path.join(M.MODELS_DIR, "metrics.json")))
    full_auc, base_auc = full["cv_roc_auc"], metrics["cv_roc_auc"]
    beats = full_auc > base_auc
    statement = (
        f"The full {full['n_features']}-gene model {'outperforms' if beats else 'does NOT outperform'} "
        f"the deletion-burden baseline (CV ROC-AUC {full_auc} vs {base_auc}), "
        f"{'indicating it learns MTAP-relevant context beyond overall deletion load.' if beats else 'suggesting its signal may largely reflect overall deletion load.'}"
    )
    metrics["comparison_to_full_model"] = {
        "full_model_cv_roc_auc": full_auc,
        "full_model_cv_pr_auc": full["cv_pr_auc"],
        "baseline_cv_roc_auc": base_auc,
        "baseline_cv_pr_auc": metrics["cv_pr_auc"],
        "full_model_outperforms": bool(beats),
        "statement": statement,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    cv = metrics["cv_5fold"]
    print(f"=== deletion-burden baseline (features={metrics['n_features']}, "
          f"pos={metrics['n_positive']}, ref={metrics['n_reference']}) ===")
    print(f"  CV(5-fold)  ROC-AUC={base_auc}  PR-AUC={metrics['cv_pr_auc']}  "
          f"sens={cv['sensitivity']}  spec={cv['specificity']}  F1={cv['f1']}")
    print(f"  full model CV ROC-AUC={full_auc}")
    print(f"  -> {statement}")
    print(f"  saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
