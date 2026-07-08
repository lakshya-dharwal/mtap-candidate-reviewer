# MTAP-Loss Candidate Reviewer

An **expression-validated review queue for ambiguous MTAP shallow-loss tumors.**

The tool trains a machine-learning model on unambiguous MTAP homozygous
deletions, then **ranks the ambiguous shallow-loss (`-1`) tumors whose
copy-number profile resembles confirmed-deleted cases**, and validates that the
high-priority ones actually show suppressed MTAP expression.

> It ranks MTAP shallow-loss tumors whose CNA profiles resemble confirmed
> homozygous-deleted cases, then validates whether those high-priority cases
> show suppressed MTAP expression. It does **not** claim to detect or find
> "missed MTAP-deficient patients" — it prioritizes candidates for orthogonal
> human review.

## Cohort
Bladder — `blca_tcga_pan_can_atlas_2018` (TCGA PanCancer Atlas). CNA from the
GISTIC 2.0 thresholded profile; expression from `rna_seq_v2_mrna` (RSEM
normalized counts). Prevalence figures are verified from the actual pulled
cohort, not literature estimates.

## Labels (from MTAP GISTIC discrete copy-number)
| MTAP GISTIC | Role | Name |
|---|---|---|
| `-2` | train positive | confirmed homozygous deletion |
| `0,1,2` | train negative | copy-number non-deleted reference |
| `-1` | held out, scored at inference | ambiguous shallow loss |

No leakage: MTAP is the label and is never a feature; MTAP expression is
validation-only and never a feature.

## Pipeline
`cbioportal_client.py` (fetch + cache) → `data_prep.py` (feature matrix, labels,
exclusions) → `model.py` (L2 logistic regression, primary) → tiered review queue
→ `validate_expression.py` (four-group expression comparison).

## Honest limitations
- Public TCGA data, not IDEAYA assay data.
- "Candidate" = CNA-profile resemblance to confirmed-deleted cases, validated by
  expression — **not** IHC/FISH-confirmed truth.
- Small positive counts widen confidence intervals; every AUC is now reported
  with a bootstrap 95% CI (`models/metrics.json`'s `cv_auc_ci95`), not just a
  point estimate.
- The tool prioritizes cases for orthogonal human review; it does not make
  clinical calls.
- Prevalence figures are verified from the actual pulled cohort, not literature
  estimates.
- Tumor purity is **not modeled**. Bulk CNA/expression are diluted by
  low-purity samples; cBioPortal's clinical API for this study does not expose
  an ABSOLUTE purity attribute (checked directly — none of its 60 clinical
  attributes match), so this is an acknowledged, currently unaddressed gap
  rather than a silently-ignored one.
- The `-2`-vs-reference discrimination is dominated by 9p21-local copy number
  (CDKN2A/CDKN2B), confirmed two ways: the three-way ablation (AUC 0.97→0.78
  dropping the neighborhood, and a 20-replicate random-2-gene permutation
  control shows that drop is far below the null range, not a generic
  "any-2-features" effect — see `permutation_control` in
  `models/ablation_metrics.json`) and a deletion-burden baseline
  (`deletion_burden.py`) that shows genome-wide deletion burden alone is at
  chance (CV ROC-AUC 0.50) while a crude chr9p-only burden feature alone
  reaches 0.98. The model is not adding much beyond 9p21 status on this cohort.
- The primary (logistic) and secondary (XGBoost) models only agree on 58% of
  tier assignments for the ambiguous −1 queue (47% Jaccard overlap on the High
  tier) — see `models/queue_model_agreement.json`. The choice of model
  materially changes who gets reviewed first; this is not yet reconciled.
- The "reference" class pools GISTIC 0 (neutral, n=118), 1 (gain, n=58), and 2
  (amplified, n=1) — a third of "reference" tumors are copy-gained, not
  copy-neutral (`models/provenance.json`'s `reference_class_composition`).
- The 150-gene panel (`genes.py`) is hand-curated from general cancer-driver
  knowledge, not empirically derived from this cohort's own GISTIC peaks or a
  single cited driver census.
- No repeated CV / multiple random seeds — every model here shares
  `random_state=42` for both the CV folds and the held-out split, so there's
  no estimate of seed-to-seed variance on top of the bootstrap CI.

## Key references
- MTAP–MAT2A/PRMT5 vulnerability, CDKN2A co-deletion 80–90%: Mavrakis et al.,
  *Cell Reports* 2016 (ScienceDirect S2211124716302996)
- 9p21 pan-cancer frequencies (MTAP homdel ~9.3%): Nature Communications 2021,
  s41467-021-25894-9
- NGS misses MTAP-deficient cases IHC catches: *Journal of Thoracic Oncology*
  2025, S1556-0864(25)01018-4
- Heterozygous loss can be functionally MTAP-deficient (~71% protein loss):
  mesothelioma CNV/IHC study, PMC10605896
- Expression tracks copy number (validation basis): PTEN pan-cancer PMC10050165;
  UCSC Xena GISTIC/expression docs
- cBioPortal API + GISTIC formats: docs.cbioportal.org/web-api-and-clients
- IDEAYA pipeline (IDE397 MAT2A, IDE892 PRMT5): ideayabio.com/pipeline
- XGBoost imbalance (scale_pos_weight): xgboost.readthedocs.io; SHAP
  TreeExplainer: shap.readthedocs.io

## Engineering / reproducibility risks
- `bravado` is pinned to cBioPortal's **legacy `/api/v2/api-docs`** Swagger 2.0
  endpoint (`cbioportal_client.py`) because `/api/v3` is OpenAPI 3 and bravado
  silently returns `None` from every call against it. v2 is not the
  documented/maintained API and could be deprecated without notice — the
  `requests`-based fallback hits v3 directly and isn't affected.
- All dependencies are now pinned (`requirements.txt`) — unpinned deps already
  broke this build twice mid-project (a matplotlib 3.11 API rename, a
  streamlit deprecation).
- Cache files are now content-hash-gated (`cbioportal_client._sig`/`_cache_valid`)
  so a changed `PANEL_SYMBOLS` or cohort triggers a re-pull instead of silently
  serving a stale panel; network fetches retry with exponential backoff.
- `tests/test_pipeline_invariants.py` covers label/feature invariants, queue
  tier proportions, and cross-artifact consistency — run with `pytest tests/ -v`.

## Reproduce
```bash
pip install -r requirements.txt
python model.py                 # data prep + primary model + saved artifacts
python model.py --ablations     # three-way ablation + permutation control
python secondary_model.py       # XGBoost comparison + queue agreement check
python deletion_burden.py       # deletion-burden baseline (genome/chr9p/combined)
python score_queue.py           # tiered review queue for the -1 set
python validate_expression.py   # four-group expression figure + Mann-Whitney
pytest tests/ -v                # pipeline invariant checks
```
All steps are cache-only after the initial pull (`cache/`).
