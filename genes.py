"""Entrez gene IDs and gene sets for the MTAP-loss candidate reviewer.

Only the 0th-step essentials are populated. PANEL and NINE_P21 are stubs used by
later steps (feature matrix, ablations) and are intentionally minimal for now.
"""

# Label / core 9p21 genes (0th step uses these three).
MTAP = 4507
CDKN2A = 1029
CDKN2B = 1030

# The three genes pulled in the 0th step, name -> Entrez.
CORE_GENES = {
    "MTAP": MTAP,
    "CDKN2A": CDKN2A,
    "CDKN2B": CDKN2B,
}

# 9p21 neighborhood (Entrez) — expanded in the "whole 9p21" ablation step.
NINE_P21 = {
    "MTAP": MTAP,
    "CDKN2A": CDKN2A,
    "CDKN2B": CDKN2B,
}

# ~150 recurrently-altered gene panel for the CNA feature matrix.
# Stubbed for later steps; populated when the model is built.
PANEL = {}
