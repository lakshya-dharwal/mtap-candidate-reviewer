"""Build the CNA feature matrix, labels, and exclusion log for the reviewer.

Labels come from MTAP GISTIC copy-number:
  -2         -> positive  (confirmed homozygous deletion)
  {0, 1, 2}  -> negative  (copy-number non-deleted reference)
  -1         -> held out, scored at inference (ambiguous shallow loss)

MTAP is the label and is NEVER a feature (no leakage). MTAP expression is
validation-only and never enters this matrix.

The ~150-gene panel's GISTIC values are the features. Phase 1 only cached the
three 9p21 genes, so the panel CNA is pulled once here and cached; every later
run is cache-only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

import cbioportal_client as cc
import genes as G

STUDY = "blca_tcga_pan_can_atlas_2018"
CACHE_DIR = cc.CACHE_DIR
EXCLUSIONS_PATH = os.path.join(CACHE_DIR, f"exclusions_{STUDY}.csv")


# --------------------------------------------------------------------------- #
# Panel CNA (one-time pull, then cache-only)
# --------------------------------------------------------------------------- #
def load_or_build_panel_cna(study=STUDY):
    """Return (panel_cna_long, gene_map) for the ~150-gene panel.

    panel_cna_long: sampleId, entrezGeneId, value
    gene_map:       hugoGeneSymbol, entrezGeneId
    """
    cache_path = os.path.join(CACHE_DIR, f"cna_panel_{study}.csv")
    gene_map_path = os.path.join(CACHE_DIR, "gene_map.csv")

    # This short-circuit intentionally avoids ANY network call (not even
    # cc.connect()) when the cache is warm, per the project's "never hit the
    # API again during dev" design constraint. It validates the panel
    # content-hash itself (no network needed for that, and a DIFFERENT sig
    # file from cc.fetch_molecular_data's own study+profile_id+entrez_ids sig,
    # since profile_id isn't known without a network call) — otherwise a
    # changed PANEL_SYMBOLS would silently keep serving a stale panel forever.
    panel_sig_path = cache_path + ".panelsig"
    panel_sig = cc._sig(study, sorted(G.PANEL_SYMBOLS))
    cache_ok = os.path.exists(cache_path) and os.path.exists(gene_map_path)
    sig_ok = (os.path.exists(panel_sig_path)
             and open(panel_sig_path).read().strip() == panel_sig)
    if cache_ok and sig_ok:
        return pd.read_csv(cache_path), pd.read_csv(gene_map_path)
    if cache_ok and not os.path.exists(panel_sig_path):
        # Pre-existing cache from before this check existed — grandfather it
        # in rather than force an unnecessary re-pull, and backfill the sig.
        with open(panel_sig_path, "w") as f:
            f.write(panel_sig)
        return pd.read_csv(cache_path), pd.read_csv(gene_map_path)

    client = cc.connect()
    gene_map = cc.resolve_symbols_to_entrez(client, G.PANEL_SYMBOLS)
    requested = set(G.PANEL_SYMBOLS)
    resolved = set(gene_map["hugoGeneSymbol"])
    missing = sorted(requested - resolved)
    if missing:
        print(f"[data_prep] {len(missing)} panel symbols unresolved (dropped): {missing}")

    cna_profile = cc.pick_profile(client, study, "gistic")
    entrez_ids = gene_map["entrezGeneId"].tolist()
    panel_cna = cc.fetch_molecular_data(
        client, study, cna_profile, entrez_ids, f"cna_panel_{study}"
    )
    with open(panel_sig_path, "w") as f:
        f.write(panel_sig)
    return panel_cna, gene_map


# --------------------------------------------------------------------------- #
# Feature matrix + labels
# --------------------------------------------------------------------------- #
def build_dataset(study=STUDY):
    """Build the modeling dataset.

    Returns a dict with:
      X            : DataFrame (sampleId index, Hugo-symbol feature columns; MTAP excluded)
      mtap_cna     : Series of MTAP GISTIC value per sample
      role         : Series in {"positive", "reference", "ambiguous"}
      y            : Series (1=positive, 0=reference, NaN=ambiguous) aligned to X
      feature_genes: list of feature column names
      exclusions   : DataFrame log of dropped samples/features and why
    """
    panel_cna, gene_map = load_or_build_panel_cna(study)
    entrez_to_hugo = dict(zip(gene_map["entrezGeneId"], gene_map["hugoGeneSymbol"]))

    exclusions = []  # list of {level, id, reason, n}

    # Wide matrix: samples x Hugo symbol
    panel_cna = panel_cna.copy()
    panel_cna["hugo"] = panel_cna["entrezGeneId"].map(entrez_to_hugo)
    wide = panel_cna.pivot_table(
        index="sampleId", columns="hugo", values="value", aggfunc="first"
    )

    # --- MTAP is the label; pull it out, then remove from features ---------- #
    if "MTAP" not in wide.columns:
        raise RuntimeError("MTAP missing from panel CNA — cannot label.")
    mtap_cna = wide["MTAP"].copy()

    # Samples with no MTAP call cannot be labeled -> exclude.
    no_mtap = mtap_cna.isna()
    if no_mtap.any():
        for sid in wide.index[no_mtap]:
            exclusions.append({"level": "sample", "id": sid,
                               "reason": "no MTAP GISTIC call (cannot label)", "n": 1})
        wide = wide[~no_mtap]
        mtap_cna = mtap_cna[~no_mtap]

    features = wide.drop(columns=["MTAP"])
    exclusions.append({"level": "feature", "id": "MTAP",
                       "reason": "label gene — excluded from features (no leakage)", "n": 1})

    # --- Handle sparse feature columns ------------------------------------- #
    # GISTIC is normally complete; any residual NaN is neutral copy-number (0).
    n_missing_cells = int(features.isna().sum().sum())
    if n_missing_cells:
        exclusions.append({"level": "cell", "id": "features",
                           "reason": "missing GISTIC cell imputed as 0 (neutral)",
                           "n": n_missing_cells})
    features = features.fillna(0).astype(int)

    # --- Roles / labels ----------------------------------------------------- #
    def role_of(v):
        v = int(v)
        if v == -2:
            return "positive"
        if v == -1:
            return "ambiguous"
        return "reference"

    role = mtap_cna.map(role_of)
    y = role.map({"positive": 1, "reference": 0, "ambiguous": np.nan})

    exclusions_df = pd.DataFrame(exclusions, columns=["level", "id", "reason", "n"])
    os.makedirs(CACHE_DIR, exist_ok=True)
    exclusions_df.to_csv(EXCLUSIONS_PATH, index=False)

    return {
        "X": features,
        "mtap_cna": mtap_cna,
        "role": role,
        "y": y,
        "feature_genes": list(features.columns),
        "exclusions": exclusions_df,
        "gene_map": gene_map,
        "study": study,
    }


def summarize(ds):
    role = ds["role"]
    print(f"\n[data_prep] study={ds['study']}")
    print(f"  samples (labelable): {len(ds['X'])}")
    print(f"  features:            {len(ds['feature_genes'])} genes (MTAP excluded)")
    print(f"  positive (-2):       {(role=='positive').sum()}")
    print(f"  reference (0/1/2):   {(role=='reference').sum()}")
    print(f"  ambiguous (-1):      {(role=='ambiguous').sum()}")
    print(f"  exclusions logged:   {len(ds['exclusions'])} rows -> {EXCLUSIONS_PATH}")


if __name__ == "__main__":
    ds = build_dataset()
    summarize(ds)
