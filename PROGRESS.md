# Build Progress — MTAP-Loss Candidate Reviewer

## 2026-07-07 — Phase 1 (data pull + viability) — COMPLETE
Cohort `blca_tcga_pan_can_atlas_2018`. Merged MTAP/CDKN2A/CDKN2B GISTIC CNA +
MTAP RNA-seq (`rna_seq_v2_mrna`, RSEM). Auto-picked profiles via substring; no
hardcoding. bravado uses the Swagger-2.0 endpoint `/api/v2/api-docs` (the v3
endpoint is OpenAPI-3 and silently returns None); plain-`requests` fallback in
place.
- Samples with MTAP GISTIC: 408 · with matched MTAP RNA-seq: 404
- **-2 positives: 106 · -1 ambiguous: 125 (all with expression) · {0,1,2} reference: 177**
- Gate PASSED (want ≥15 pos, ≥20 ambiguous-with-expression). Proceed with bladder.

## 2026-07-07 — Phase 2 (data_prep + model) — COMPLETE
**Deviation noted (conservative choice):** Phase 1 only cached the 3 core genes,
which is not a usable feature matrix. Per the build gotcha, populated
`genes.py` with a curated ~150-gene recurrently-altered cancer panel (Hugo
symbols) and did a **one-time** GISTIC CNA pull for that panel, cached to
`cache/cna_panel_*.csv` + `cache/gene_map.csv`. All later runs are cache-only.
All 151 requested symbols resolved to Entrez; 0 dropped.

- Feature matrix: **408 samples × 150 features** (MTAP excluded from features —
  no leakage). No samples silently dropped; GISTIC complete (0 missing cells).
- Labels: 106 positive (-2) / 177 reference (0,1,2) / 125 ambiguous (-1 held out).
- Model: L2 logistic regression, `StandardScaler`, `class_weight='balanced'`,
  stratified 80/20, `random_state=42`, 5-fold CV.
- Exclusions logged: `cache/exclusions_*.csv` (MTAP as label; imputation policy).
- Artifacts: `models/model_main.joblib`, `models/metrics.json`,
  `models/provenance.json`.

**Phase 2 metrics (main model, WITH CDKN2A, n_pos=106, n_ref=177):**
| Split | ROC-AUC | PR-AUC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| CV 5-fold | 0.9716 | 0.919 | 0.934 | 0.932 | 0.912 |
| Test 20% (n=57) | 0.9762 | 0.9628 | 0.810 | 0.944 | 0.850 |
- CV confusion: TN=165 FP=12 FN=7 TP=99. Test confusion: TN=34 FP=2 FN=4 TP=17.

Next: Phase 3 (score -1 set into tiered queue), Phase 4 (expression validation).
