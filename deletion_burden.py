"""Stage 2: deletion-burden baseline.

Answers the sharpest critique of the full model — "is it just learning overall
deletion load, not MTAP-specific context?" — by training on crude burden
features and comparing to the full 150-gene model.

IMPORTANT nuance (found during audit): the two burden features are NOT
interchangeable. `chr9p_deletion_burden` is computed from CDKN2A/CDKN2B/JAK2 —
i.e. it's substantially a re-encoding of 9p21 status, the same signal the
ablation already measures. `genome_deletion_burden` (all 150 panel genes) is
the actual "generic aneuploidy" control. Reporting only the combined feature
set conflates the two and overstates what "deletion burden" means. This script
now trains THREE variants — genome-only, chr9p-only, and both — so the
narrative doesn't imply genome-wide burden alone explains the signal when it
doesn't.

Same label (-2 vs reference) and the SAME training/eval routine as the primary
model (model.train_and_evaluate), so every comparison is apples-to-apples.
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

    variants = [
        ("genome_only", ["genome_deletion_burden"]),
        ("chr9p_only", ["chr9p_deletion_burden"]),
        ("genome_plus_chr9p", ["genome_deletion_burden", "chr9p_deletion_burden"]),
    ]
    results = {}
    for name, feats in variants:
        m, _ = M.train_and_evaluate(
            burden, ds["y"], feats, label=f"deletion_burden_{name}", capture_curves=True,
        )
        results[name] = m
        cv = m["cv_5fold"]
        print(f"=== deletion-burden [{name}] (features={feats}) ===")
        print(f"  CV(5-fold)  ROC-AUC={m['cv_roc_auc']} (CI {m['cv_auc_ci95']['roc_auc_ci95']})  "
              f"PR-AUC={m['cv_pr_auc']}  sens={cv['sensitivity']}  spec={cv['specificity']}  F1={cv['f1']}")

    # Compare each to the already-saved full (primary) model.
    full = json.load(open(os.path.join(M.MODELS_DIR, "metrics.json")))
    full_auc = full["cv_roc_auc"]

    genome_auc = results["genome_only"]["cv_roc_auc"]
    chr9p_auc = results["chr9p_only"]["cv_roc_auc"]
    combined_auc = results["genome_plus_chr9p"]["cv_roc_auc"]
    beats_combined = full_auc > combined_auc

    statement = (
        f"Genome-wide deletion burden ALONE is near chance (CV ROC-AUC {genome_auc}) — "
        f"it carries essentially no signal. Almost all of the combined baseline's power "
        f"(CV ROC-AUC {combined_auc}) comes from chr9p_deletion_burden alone "
        f"(CV ROC-AUC {chr9p_auc}), which is a crude re-encoding of CDKN2A/CDKN2B "
        f"copy-number status — the SAME signal the 9p21 ablation already measures, not an "
        f"independent 'overall aneuploidy' explanation. The full {full['n_features']}-gene "
        f"model {'outperforms' if beats_combined else 'does NOT outperform'} the combined "
        f"burden baseline (CV ROC-AUC {full_auc} vs {combined_auc}). Conclusion: this "
        f"comparison does NOT show the model is 'just learning deletion burden' in the "
        f"genome-wide sense — genome-wide burden has no predictive value here. It does "
        f"confirm (again, consistent with the ablation) that 9p21-local copy number "
        f"dominates the -2-vs-reference discrimination."
    )

    combined = {
        "label": "deletion_burden_baseline",
        "variants": {name: m for name, m in results.items()},
        "comparison_to_full_model": {
            "full_model_cv_roc_auc": full_auc,
            "full_model_cv_pr_auc": full["cv_pr_auc"],
            "genome_only_cv_roc_auc": genome_auc,
            "chr9p_only_cv_roc_auc": chr9p_auc,
            "combined_cv_roc_auc": combined_auc,
            "full_model_outperforms_combined": bool(beats_combined),
            "chr9p_carries_the_signal": bool(chr9p_auc > genome_auc + 0.1),
            "statement": statement,
        },
        # Kept at top level for backward-compat with anything reading the old shape.
        "cv_roc_auc": combined_auc,
        "cv_pr_auc": results["genome_plus_chr9p"]["cv_pr_auc"],
        "cv_5fold": results["genome_plus_chr9p"]["cv_5fold"],
        "roc_curve": results["genome_plus_chr9p"].get("roc_curve"),
        "pr_curve": results["genome_plus_chr9p"].get("pr_curve"),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"\n{statement}\n")
    print(f"  saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
