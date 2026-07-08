"""Phase 3: score the held-out ambiguous (-1) set into a tiered review queue.

The queue is ranked purely by the CNA-model score of resembling a confirmed
homozygous-deletion (-2) case. Expression is NOT part of this score — it is
applied later as orthogonal validation (Phase 4).

Tiers are percentile-based over the scored -1 set itself (not fixed score
cutoffs): top 20% -> High, next 30% -> Medium, bottom 50% -> Low. This always
yields a usable, proportionate queue regardless of the score distribution.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

import data_prep
import model as M

QUEUE_PATH = os.path.join(data_prep.CACHE_DIR, f"review_queue_{data_prep.STUDY}.csv")


def assign_tiers(scores):
    """Percentile-based tiers over the scored set: top 20% High, next 30% Medium,
    bottom 50% Low. Returns a Series aligned to `scores`."""
    scores = pd.Series(scores)
    q80 = np.percentile(scores, 80)
    q50 = np.percentile(scores, 50)
    return scores.map(lambda s: "High" if s >= q80 else "Medium" if s >= q50 else "Low")


def build_queue(ds=None):
    if ds is None:
        ds = data_prep.build_dataset()
    bundle = joblib.load(os.path.join(M.MODELS_DIR, "model_main.joblib"))
    pipe, feat = bundle["pipeline"], bundle["feature_genes"]

    amb = ds["role"] == "ambiguous"
    X_amb = ds["X"].loc[amb, feat]
    proba = pipe.predict_proba(X_amb)[:, 1]

    q = pd.DataFrame({"sampleId": X_amb.index, "model_probability": proba})
    q["review_tier"] = assign_tiers(q["model_probability"]).values
    # 9p21 context columns (audit trail; MTAP CNA is -1 for all by definition).
    ctx = ds["X"].loc[amb, ["CDKN2A", "CDKN2B"]].rename(
        columns={"CDKN2A": "CDKN2A_cna", "CDKN2B": "CDKN2B_cna"})
    q = q.merge(ctx, left_on="sampleId", right_index=True)
    q["MTAP_cna"] = ds["mtap_cna"].loc[amb].values
    q = q.sort_values("model_probability", ascending=False).reset_index(drop=True)
    return q


def main():
    ds = data_prep.build_dataset()
    q = build_queue(ds)
    q.to_csv(QUEUE_PATH, index=False)

    n = len(q)
    counts = q["review_tier"].value_counts()
    print(f"\n=== Review queue: {n} ambiguous (-1) cases scored ===")
    ranges = {"High": "(top 20%)", "Medium": "(next 30%)", "Low": "(bottom 50%)"}
    for tier in ["High", "Medium", "Low"]:
        c = int(counts.get(tier, 0))
        print(f"  {tier:<7} {ranges[tier]:<13} {c:>4}  ({c/n:.0%})")
    print(f"\n  saved -> {QUEUE_PATH}")
    print(f"\n  top of queue:")
    print(q.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
