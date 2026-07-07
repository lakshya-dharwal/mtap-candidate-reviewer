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
- Small positive counts widen confidence intervals; counts (n=) are reported
  alongside every metric.
- The tool prioritizes cases for orthogonal human review; it does not make
  clinical calls.
- Prevalence figures are verified from the actual pulled cohort, not literature
  estimates.

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

## Reproduce
```bash
pip install -r requirements.txt
python model.py                 # data prep + primary model + saved artifacts
python score_queue.py           # tiered review queue for the -1 set
python validate_expression.py   # four-group expression figure + Mann-Whitney
```
All steps are cache-only after the initial pull (`cache/`).
