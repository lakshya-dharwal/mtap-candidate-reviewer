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

## 2026-07-07 — Phase 3 (score -1 set → tiered queue) — COMPLETE
Scored all 125 ambiguous (-1) cases with the model refit on all labeled data.
Ranked by `model_probability` (resemblance to -2). Tiers: >0.75 High,
0.50–0.75 Medium, <0.50 Low. Queue cached to
`cache/review_queue_*.csv` with 9p21 context columns (MTAP/CDKN2A/CDKN2B GISTIC).

**Tier counts (n=125):**
| Tier | Range | n | % |
|---|---|---|---|
| High | >0.75 | 50 | 40% |
| Medium | 0.50–0.75 | 20 | 16% |
| Low | <0.50 | 55 | 44% |

## 2026-07-07 — Phase 4 (expression validation) — COMPLETE
Orthogonal check: MTAP RNA-seq (RSEM), log2(x+1)-transformed. Four groups; the
queue score does NOT use expression. Fix applied: matplotlib 3.11 renamed
`boxplot(labels=...)` to `tick_labels=...`. Figure + stats saved to `models/`.

**Group medians (MTAP expression, log2(RSEM+1)):**
| Group | n | median log2 |
|---|---|---|
| -2 homozygous deletion | 106 | 5.9231 |
| -1 high-priority (High tier) | 50 | 9.5214 |
| -1 low-priority (Low tier) | 55 | 9.6571 |
| 0/1/2 reference | 173 | 10.2485 |

**Mann-Whitney, -1 High vs -1 Low:** U=1126, **p=0.111** (n_high=50, n_low=55, two-sided).

Artifacts: `models/expression_validation_*.png`, `models/expression_validation_*.json`.

---

## ⛔ HARD STOP #1 — awaiting review before Phase 5

The two verdict numbers for the morning:
1. **AUC with CDKN2A:** CV ROC-AUC 0.9716 / PR-AUC 0.919 (test 0.976 / 0.963).
   The without-CDKN2A ablation is Phase 5 — not yet run, so whether AUC *survives
   dropping CDKN2A* is still unknown.
2. **Expression separation, high vs low -1:** medians 9.52 vs 9.66, Mann-Whitney
   **p=0.111** (not below 0.05). The -2 group (median 5.92) is clearly separated
   from all -1 and reference groups; the high- vs low-priority -1 separation is
   small on this cohort.

Reporting numbers only, not judging them. Stopping here per instructions.
Next (on "continue"): Phase 5 ablations (with / without CDKN2A / without 9p21).

## 2026-07-07 — Phase 5 (three-way ablation) — COMPLETE
Primary logistic model retrained three times, identical setup each time (L2,
`class_weight='balanced'`, stratified 80/20, `random_state=42`, 5-fold CV);
only the feature columns change. Saved to `models/ablation_metrics.json`.

**Genes dropped in model 3 (whole 9p21 neighborhood):** `CDKN2A`, `CDKN2B`.
Panel check: the only other chr9p gene present is `JAK2` (9p24.1), which is NOT
part of the 9p21.3 neighborhood and was kept. MTAP is the label and is already
excluded from all feature sets.

**Three-way table (n_pos=106, n_ref=177):**
| Model | feats | CV ROC-AUC | CV PR-AUC | Sens | Spec | F1 | Test ROC-AUC |
|---|---|---|---|---|---|---|---|
| with CDKN2A | 150 | 0.9716 | 0.9190 | 0.9340 | 0.9322 | 0.9124 | 0.9762 |
| without CDKN2A | 149 | 0.9464 | 0.8827 | 0.8774 | 0.9379 | 0.8857 | 0.9074 |
| without 9p21 (CDKN2A+CDKN2B) | 148 | 0.7751 | 0.6659 | 0.6604 | 0.7853 | 0.6542 | 0.6772 |

Numbers reported without interpretation, per instructions.

---

## ⛔ HARD STOP #2 (pre-Phase-6) — awaiting review
Verdict number #1 (does AUC survive dropping CDKN2A) is now answered by the
table above. Stopping here per instructions. Next (on "continue"): Phase 6
(Streamlit app wrapping Phases 2–5).

## 2026-07-07 — Phases 6 + 7 + 7.5 (Streamlit UI + integrity spine + restyle) — COMPLETE
**Note:** the restyle request assumed Phases 6/7 were already built, but the
session had stopped at HARD STOP #2 — `app.py` did not exist. Built Phases 6 and
7 together with the IDEAYA styling applied from the start (conservative,
intent-serving choice). No data/model/metric changed; the UI only renders cached
artifacts.

- `shap_explain.py`: exact linear SHAP (StandardScaler + L2 logistic). Global
  importance (top: CDKN2A 2.41, CDKN2B 1.74) + per-patient "why flagged" strings.
- `app.py` panels: cohort selector + prevalence stat cards; model-performance
  three-way ablation table + cards; expression-validation figure + medians +
  Mann-Whitney; enriched review queue (MTAP/CDKN2A/CDKN2B GISTIC, MTAP expr
  percentile, model probability progress bar, tier, why-flagged SHAP) with tier
  filter; provenance/audit panel; exclusions-log panel.
- **Phase 7.5 IDEAYA restyle:** `.streamlit/config.toml` base theme
  (primaryColor #1E9BD7, bg #EEF4F8) + injected CSS — navy hero banner, Poppins
  headings, ALL-CAPS letter-spaced eyebrows with rotating accent rule-lines
  (#2B8A8A/#5BA84A/#A3C644), white rounded cards with soft shadows. Styled in the
  spirit of IDEAYA's brand; no logo/affiliation claimed.

**Fixes noted:** (1) streamlit was left half-installed by the Phase-1
out-of-disk error (files present, metadata missing) — reinstalled cleanly
(1.59.0). (2) Replaced deprecated `use_container_width=True` with
`width="stretch"`. Verified via `streamlit.testing.v1.AppTest`: runs with no
exceptions, 3 dataframes / 18 metrics, tier-filter interaction works.

Pushed to origin (github.com/lakshya-dharwal/mtap-candidate-reviewer).
Per instructions: STOP. Phase 8 (pan-cancer, XGBoost secondary, concordance)
NOT built — it is the cut list.

## 2026-07-08 — Stage 1 (tabbed demo UI + honesty fixes) — COMPLETE
Reorganized the single-scroll app into 5 tabs; no data/metric dropped. Small
honesty fixes plus additive new artifacts (curves, confusion matrix, XGBoost
comparison). Verified via AppTest (no exceptions, 5 tabs, 5 dataframes, 24
metrics; values match: CV ROC-AUC 0.9716, AUC-without-9p21 0.7751, 125 queued,
expression p=0.111) and a live `streamlit run` (health 200, no log errors).

- **Percentile tiers** (`score_queue.py`): top 20% High / next 30% Medium /
  bottom 50% Low → 25 / 38 / 62 of the 125 ambiguous cases (was fixed 0.75/0.50
  score cutoffs). API/columns unchanged.
- **ROC/PR curves** (`model.py`): additive `capture_curves` param stores CV
  out-of-fold curve arrays in `metrics.json`. Primary scalar numbers unchanged
  (same seed/data) — CV ROC-AUC still 0.9716.
- **XGBoost secondary** (`secondary_model.py`, new): same 150 features / folds /
  seed. CV ROC-AUC 0.9861, PR-AUC 0.9621 → `models/secondary_xgboost_metrics.json`.
  Comparison only; logistic stays primary.
- **UI honesty**: "probability" → "model score" everywhere; tier captions now
  "(top 20%)/(next 30%)/(bottom 50%)"; expression stated as a null result.
- **New panels**: Overview (purpose, headline-results strip, Known Limitations,
  Model Sanity Checks — deletion-burden line file-gated to "(pending)"),
  How It Works, and in Validation: ROC/PR plots, confusion matrix, global-SHAP
  bar, XGBoost card. Raw-data preview in Data & Provenance.
- **Stage-2 safety**: the sanity-check line + burden card are gated on
  `models/deletion_burden_metrics.json` existing — Stage 2 needs zero app.py
  edits, and deleting that file fully reverts the UI.

Note (Py 3.11): fixed an f-string with backslash-escaped quotes inside `{}`
(SyntaxError pre-3.12) by moving the span literal out of the expression.

Next: Stage 2 (deletion-burden baseline).
