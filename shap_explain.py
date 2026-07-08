"""SHAP explanations for the primary logistic model.

The model is a StandardScaler + L2 logistic regression pipeline, so a linear
SHAP explainer is exact. Provides:
  - global feature importance (mean |SHAP| across the labeled set)
  - per-patient top contributing features ("why flagged") for the -1 queue

SHAP values are in log-odds space toward the positive (homozygous-deletion)
class; a positive value pushes a case toward "resembles confirmed deletion".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def _explainer(pipeline, background_X, feature_genes):
    scaler = pipeline.named_steps["scale"]
    clf = pipeline.named_steps["clf"]
    bg_scaled = scaler.transform(background_X[feature_genes])
    # GISTIC CNA features are highly correlated (co-deletion blocks — e.g.
    # CDKN2A/CDKN2B move together in ~80-90% of 9p21-loss cases). The default
    # ("interventional") SHAP mode assumes feature independence and can split
    # credit between correlated features somewhat arbitrarily. Explicitly use
    # "correlation_dependent", which estimates the background covariance from
    # bg_scaled, so credit among correlated genes reflects their joint
    # distribution rather than an independence assumption.
    explainer = shap.LinearExplainer(
        clf, bg_scaled, feature_perturbation="correlation_dependent")
    return explainer, scaler


def global_importance(pipeline, background_X, feature_genes):
    """Return DataFrame(gene, mean_abs_shap) sorted descending."""
    expl, scaler = _explainer(pipeline, background_X, feature_genes)
    sv = expl.shap_values(scaler.transform(background_X[feature_genes]))
    imp = np.abs(sv).mean(axis=0)
    return (pd.DataFrame({"gene": feature_genes, "mean_abs_shap": imp})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True))


def per_patient_top_features(pipeline, X_query, feature_genes,
                             background_X, top_n=3):
    """Return Series (index=sampleId) of a 'why flagged' string per patient.

    Lists the top_n features by SHAP magnitude, signed toward the positive class,
    e.g. "CDKN2B↓ (-1), FGFR3↑ (+1), RB1↓ (0)". Sign shows push direction; the
    parenthetical is the patient's GISTIC value for that gene.
    """
    expl, scaler = _explainer(pipeline, background_X, feature_genes)
    Xq = X_query[feature_genes]
    sv = expl.shap_values(scaler.transform(Xq))
    sv = np.asarray(sv)

    out = {}
    for i, sid in enumerate(Xq.index):
        contrib = sv[i]
        order = np.argsort(np.abs(contrib))[::-1][:top_n]
        parts = []
        for j in order:
            gene = feature_genes[j]
            arrow = "↑" if contrib[j] > 0 else "↓"  # push toward/away positive
            gistic = int(Xq.iloc[i][gene])
            parts.append(f"{gene}{arrow} ({gistic:+d})")
        out[sid] = ", ".join(parts)
    return pd.Series(out, name="why_flagged")


if __name__ == "__main__":
    import joblib
    import os
    import data_prep
    import model as M
    import score_queue

    ds = data_prep.build_dataset()
    bundle = joblib.load(os.path.join(M.MODELS_DIR, "model_main.joblib"))
    pipe, feat = bundle["pipeline"], bundle["feature_genes"]
    labeled = ds["y"].notna()

    gi = global_importance(pipe, ds["X"].loc[labeled], feat)
    print("Top 10 global features:")
    print(gi.head(10).to_string(index=False))

    q = score_queue.build_queue(ds)
    amb = ds["role"] == "ambiguous"
    why = per_patient_top_features(pipe, ds["X"].loc[amb], feat, ds["X"].loc[labeled])
    print("\nSample 'why flagged' (top of queue):")
    print(why.reindex(q["sampleId"].head()).to_string())
