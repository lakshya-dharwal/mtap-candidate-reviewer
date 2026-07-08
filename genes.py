"""Entrez gene IDs and gene sets for the MTAP-loss candidate reviewer.

The CNA feature matrix is built from PANEL_SYMBOLS — a curated set of ~150
recurrently copy-number-altered cancer genes (drivers amplified or deleted
across TCGA cohorts, plus bladder-relevant drivers). Hugo symbols are resolved
to Entrez IDs via the cBioPortal /genes endpoint at build time (see
cbioportal_client.resolve_symbols_to_entrez), so no Entrez IDs are hand-typed
here except the three 9p21 core genes used as the label / neighborhood.

PANEL PROVENANCE (honesty note, added post-audit): this list was hand-curated
from general cancer-driver knowledge (well-known tumor suppressors + oncogenes
recurrently altered across TCGA pan-cancer studies), NOT derived from an
empirical GISTIC significant-peaks run on this specific bladder cohort, nor
sourced from a single cited driver list (e.g. COSMIC Cancer Gene Census,
Bailey et al. 2018 Cell "Comprehensive Characterization of Cancer Driver
Genes"). It is a reasonable, defensible panel but not an empirically-derived
or externally-audited one — a rigorous follow-up would re-derive it from this
cohort's own GISTIC peaks or a cited census list and check the ablation/
deletion-burden results are robust to the swap.
"""

# --- Label / core 9p21 genes (Entrez, verified in Phase 1) ------------------ #
MTAP = 4507
CDKN2A = 1029
CDKN2B = 1030

CORE_GENES = {"MTAP": MTAP, "CDKN2A": CDKN2A, "CDKN2B": CDKN2B}

# 9p21 neighborhood removed in the harshest ablation. MTAP is the label (never a
# feature); CDKN2A/CDKN2B are the physically-adjacent co-deleted genes.
NINE_P21_SYMBOLS = ["MTAP", "CDKN2A", "CDKN2B"]

# Chromosome-9p genes present in the 150-feature panel, used for the
# chr-9p deletion-burden baseline. MTAP is deliberately excluded (it is the
# label — consistent with this project's project-wide no-leakage rule), so this
# lists the panel's non-label 9p genes: CDKN2A/CDKN2B (9p21.3) and JAK2 (9p24.1).
CHR9P_SYMBOLS = ["CDKN2A", "CDKN2B", "JAK2"]

# --- ~150-gene recurrently-altered cancer panel (Hugo symbols) -------------- #
# Deduplicated at load time. Resolved to Entrez via the API and cached.
PANEL_SYMBOLS = list(dict.fromkeys([
    # 9p21 neighborhood (kept in main model; MTAP excluded from features later)
    "MTAP", "CDKN2A", "CDKN2B",
    # Core tumor suppressors (recurrently deleted)
    "TP53", "RB1", "PTEN", "APC", "SMAD4", "SMAD2", "SMAD3", "NF1", "NF2",
    "VHL", "STK11", "KEAP1", "ARID1A", "ARID1B", "ARID2", "PBRM1", "BAP1",
    "SETD2", "KMT2C", "KMT2D", "KDM6A", "CREBBP", "EP300", "ATM", "ATR",
    "BRCA1", "BRCA2", "PALB2", "CHEK2", "FANCA", "MLH1", "MSH2", "MSH6",
    "PMS2", "FBXW7", "NOTCH1", "NOTCH2", "FAT1", "TSC1", "TSC2", "DICER1",
    "WT1", "CTNNB1", "AXIN1", "DAXX", "ATRX", "MEN1", "RNF43", "ZNRF3",
    "PTCH1", "SUFU", "DNMT3A", "TET2", "ASXL1", "RUNX1", "CEBPA", "PHF6",
    "BCOR", "STAG2", "CDKN2C", "CDKN1A", "CDKN1B", "SMARCA4", "SMARCB1",
    "CDH1", "MAP2K4", "PPP2R1A", "ERCC2", "ELF3", "RXRA", "KLF5", "FOXA1",
    "NFE2L2", "TXNIP", "ZFP36L1", "SPOP", "CIC", "FUBP1",
    # Oncogenes (recurrently amplified)
    "EGFR", "ERBB2", "ERBB3", "ERBB4", "MET", "FGFR1", "FGFR2", "FGFR3",
    "FGFR4", "KRAS", "NRAS", "HRAS", "BRAF", "RAF1", "PIK3CA", "PIK3CB",
    "PIK3R1", "AKT1", "AKT2", "AKT3", "MTOR", "MYC", "MYCN", "MYCL",
    "CCND1", "CCND2", "CCND3", "CCNE1", "CDK4", "CDK6", "MDM2", "MDM4",
    "SOX2", "TERT", "KIT", "PDGFRA", "PDGFRB", "KDR", "IGF1R", "ALK",
    "ROS1", "RET", "NTRK1", "NTRK2", "NTRK3", "JAK2", "FLT3", "ABL1",
    "MCL1", "BCL2", "BCL2L1", "AR", "ESR1", "PPARG", "E2F3", "GATA3",
    "GATA6", "NKX2-1", "AURKA", "YES1", "SRC", "EZH2", "IDH1", "IDH2",
    "KDM5A", "GNAS", "GNAQ", "GNA11", "MYB", "CCNE1", "TFE3",
]))
