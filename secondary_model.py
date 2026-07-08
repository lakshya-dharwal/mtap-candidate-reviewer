"""Secondary model: XGBoost comparison to the primary logistic model.

Same task, same 150-gene feature set (incl. CDKN2A), same stratified 80/20 +
5-fold CV, same seed as the primary model — only the estimator changes. This is
a nonlinear comparison point, not the hero model (the regularized logistic
regression stays primary; it is more defensible with small positive counts).

Saves models/secondary_xgboost_metrics.json with the same shape as the primary
metrics (CV ROC-AUC/PR-AUC, CV sensitivity/specificity/F1, ROC/PR curve arrays).
"""

from __future__ import annotations

import json
import os

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier

import data_prep
import model as M

RANDOM_STATE = M.RANDOM_STATE


def make_xgb(n_pos, n_neg):
    return XGBClassifier(
        max_depth=3,
        n_estimators=200,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=n_neg / n_pos,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=2,
    )


def main():
    os.makedirs(M.MODELS_DIR, exist_ok=True)
    ds = data_prep.build_dataset()
    feat = ds["feature_genes"]

    labeled = ds["y"].notna()
    Xl = ds["X"].loc[labeled, feat]
    yl = ds["y"].loc[labeled].astype(int)
    n_pos, n_neg = int((yl == 1).sum()), int((yl == 0).sum())

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = cross_val_predict(make_xgb(n_pos, n_neg), Xl, yl, cv=skf,
                            method="predict_proba")[:, 1]

    cv_roc = roc_auc_score(yl, oof)
    cv_pr = average_precision_score(yl, oof)
    thr = M._metrics_at_threshold(yl.values, oof)
    fpr, tpr, _ = roc_curve(yl, oof)
    prec, rec, _ = precision_recall_curve(yl, oof)

    metrics = {
        "label": "secondary_xgboost",
        "model": "XGBoost (max_depth=3, scale_pos_weight=n_neg/n_pos)",
        "n_features": len(feat),
        "n_positive": n_pos,
        "n_reference": n_neg,
        "cv_roc_auc": round(cv_roc, 4),
        "cv_pr_auc": round(cv_pr, 4),
        "cv_5fold": thr,
        "roc_curve": {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()},
        "pr_curve": {"precision": prec.round(5).tolist(), "recall": rec.round(5).tolist()},
    }
    out_path = os.path.join(M.MODELS_DIR, "secondary_xgboost_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"=== secondary XGBoost (features={len(feat)}, pos={n_pos}, ref={n_neg}) ===")
    print(f"  CV(5-fold)  ROC-AUC={metrics['cv_roc_auc']}  PR-AUC={metrics['cv_pr_auc']}  "
          f"sens={thr['sensitivity']}  spec={thr['specificity']}  F1={thr['f1']}")
    print(f"  saved -> {out_path}")


if __name__ == "__main__":
    main()
