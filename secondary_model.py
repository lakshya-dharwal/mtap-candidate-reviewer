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

import joblib
import numpy as np
import pandas as pd
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
import score_queue

RANDOM_STATE = M.RANDOM_STATE
XGB_MODEL_PATH = os.path.join(M.MODELS_DIR, "model_secondary_xgb.joblib")
XGB_QUEUE_PATH = os.path.join(data_prep.CACHE_DIR,
                              f"review_queue_xgb_{data_prep.STUDY}.csv")
AGREEMENT_PATH = os.path.join(M.MODELS_DIR, "queue_model_agreement.json")


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

    # --- Score the -1 queue with XGBoost too, and check agreement with the
    # deployed logistic queue. If the two models disagree substantially on
    # which -1 cases are high-priority, that's worth knowing before trusting
    # either ranking. ---
    final_xgb = make_xgb(n_pos, n_neg).fit(Xl, yl)
    joblib.dump({"model": final_xgb, "feature_genes": feat}, XGB_MODEL_PATH)

    amb = ds["role"] == "ambiguous"
    X_amb = ds["X"].loc[amb, feat]
    xgb_scores = final_xgb.predict_proba(X_amb)[:, 1]
    xgb_q = pd.DataFrame({"sampleId": X_amb.index, "xgb_score": xgb_scores})
    xgb_q["xgb_tier"] = score_queue.assign_tiers(xgb_q["xgb_score"]).values
    xgb_q.to_csv(XGB_QUEUE_PATH, index=False)

    log_q = score_queue.build_queue(ds)[["sampleId", "model_probability", "review_tier"]]
    merged = log_q.merge(xgb_q, on="sampleId")
    spearman = merged["model_probability"].corr(merged["xgb_score"], method="spearman")
    tier_agree = (merged["review_tier"] == merged["xgb_tier"]).mean()
    high_log = set(merged.loc[merged["review_tier"] == "High", "sampleId"])
    high_xgb = set(merged.loc[merged["xgb_tier"] == "High", "sampleId"])
    jaccard_high = (len(high_log & high_xgb) / len(high_log | high_xgb)
                    if (high_log | high_xgb) else float("nan"))

    agreement = {
        "n_scored": int(len(merged)),
        "spearman_score_correlation": round(float(spearman), 4),
        "tier_exact_agreement_rate": round(float(tier_agree), 4),
        "high_tier_jaccard": round(float(jaccard_high), 4),
        "n_high_logistic": len(high_log), "n_high_xgboost": len(high_xgb),
        "n_high_both": len(high_log & high_xgb),
        "note": ("Logistic (primary) is the deployed queue model. This checks "
                 "whether the higher-CV-AUC XGBoost model would rank the same "
                 "-1 cases as high priority — low agreement would mean the "
                 "choice of model materially changes who gets reviewed first."),
    }
    with open(AGREEMENT_PATH, "w") as f:
        json.dump(agreement, f, indent=2)
    print(f"\n=== queue agreement: logistic vs XGBoost on {len(merged)} ambiguous cases ===")
    print(f"  Spearman score correlation: {agreement['spearman_score_correlation']}")
    print(f"  Exact tier agreement: {agreement['tier_exact_agreement_rate']:.0%}")
    print(f"  High-tier Jaccard overlap: {agreement['high_tier_jaccard']:.0%} "
          f"({agreement['n_high_both']} shared of {agreement['n_high_logistic']} "
          f"logistic / {agreement['n_high_xgboost']} xgboost)")
    print(f"  saved -> {AGREEMENT_PATH}")


if __name__ == "__main__":
    main()
