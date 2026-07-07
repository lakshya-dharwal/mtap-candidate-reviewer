"""Phase 4: orthogonal expression validation (the money figure).

The review queue is ranked by CNA-model probability only. Here MTAP RNA-seq is
brought in as INDEPENDENT evidence: if the high-priority -1 cases genuinely
resemble confirmed-deleted tumors, they should show lower MTAP expression than
the low-priority -1 cases.

Four groups (expected MTAP expression, low -> high):
  -2 homozygous deletion  <  -1 high-priority  <~  -1 low-priority  <  0/1/2 reference

RSEM values are right-skewed, so expression is log2(x+1)-transformed before
plotting and testing. The reported statistic is a Mann-Whitney U between the
high-priority and low-priority -1 groups.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import data_prep  # noqa: E402
import score_queue  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FIG_PATH = os.path.join(FIG_DIR, f"expression_validation_{data_prep.STUDY}.png")
STATS_PATH = os.path.join(FIG_DIR, f"expression_validation_{data_prep.STUDY}.json")


def load_expression(study=data_prep.STUDY):
    """Return Series sampleId -> log2(MTAP RSEM + 1)."""
    merged = pd.read_csv(
        os.path.join(data_prep.CACHE_DIR, f"merged_{study}.csv")
    ).set_index("sampleId")
    expr = merged["MTAP_rna"].dropna()
    return np.log2(expr + 1.0)


def build_groups(ds, queue, expr):
    role = ds["role"]
    ref_ids = role.index[role == "reference"]
    pos_ids = role.index[role == "positive"]
    hi_ids = queue.loc[queue["review_tier"] == "High", "sampleId"]
    lo_ids = queue.loc[queue["review_tier"] == "Low", "sampleId"]

    def vals(ids):
        return expr.reindex(ids).dropna()

    return {
        "-2 homozygous\ndeletion": vals(pos_ids),
        "-1 high-priority\n(High tier)": vals(hi_ids),
        "-1 low-priority\n(Low tier)": vals(lo_ids),
        "0/1/2\nreference": vals(ref_ids),
    }


def main():
    ds = data_prep.build_dataset()
    queue = score_queue.build_queue(ds)
    expr = load_expression()
    groups = build_groups(ds, queue, expr)

    labels = list(groups.keys())
    data = [groups[k].values for k in labels]
    medians = {k: round(float(np.median(v)), 4) if len(v) else None
               for k, v in groups.items()}
    ns = {k: int(len(v)) for k, v in groups.items()}

    hi = groups["-1 high-priority\n(High tier)"].values
    lo = groups["-1 low-priority\n(Low tier)"].values
    U, p = mannwhitneyu(hi, lo, alternative="two-sided")

    # --- figure ---
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=[f"{k}\n(n={ns[k]})" for k in labels],
                    showfliers=False, patch_artist=True)
    colors = ["#B03A2E", "#E67E22", "#F1C40F", "#2E86C1"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    for i, d in enumerate(data, start=1):
        jitter = np.random.default_rng(42).normal(0, 0.05, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, s=8, color="black", alpha=0.3)
    ax.set_ylabel("MTAP expression  log2(RSEM + 1)")
    ax.set_title("MTAP expression by CNA group and review priority\n"
                 f"(Mann-Whitney high vs low -1: U={U:.0f}, p={p:.3g})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150)
    plt.close(fig)

    stats = {
        "cohort": data_prep.STUDY,
        "expression_transform": "log2(RSEM + 1)",
        "group_n": ns,
        "group_median_log2": medians,
        "mannwhitney_high_vs_low_ambiguous": {
            "U": float(U), "p_value": float(p),
            "n_high": int(len(hi)), "n_low": int(len(lo)),
            "alternative": "two-sided",
        },
        "figure": os.path.basename(FIG_PATH),
    }
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== Phase 4: expression validation ===")
    for k in labels:
        print(f"  {k.replace(chr(10),' '):<32} n={ns[k]:>3}  median log2={medians[k]}")
    print(f"\n  Mann-Whitney (-1 High vs -1 Low): U={U:.0f}  p={p:.3g}  "
          f"(n_high={len(hi)}, n_low={len(lo)})")
    print(f"  figure -> {FIG_PATH}")
    print(f"  stats  -> {STATS_PATH}")


if __name__ == "__main__":
    main()
