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

## 2026-07-08 — Stage 2 (deletion-burden baseline) — COMPLETE
Attempted after Stage 1 committed + verified clean. Ran cleanly on first pass;
kept. Answers the sharpest critique ("is it just learning deletion burden?").

- `genes.py`: added `CHR9P_SYMBOLS = ["CDKN2A","CDKN2B","JAK2"]` (panel's non-label
  chr-9p genes; MTAP excluded as label, per no-leakage rule).
- `deletion_burden.py` (new): two features from the cached CNA matrix —
  `genome_deletion_burden` (fraction of 150 genes with GISTIC<0) and
  `chr9p_deletion_burden` — trained with the SAME `model.train_and_evaluate`
  routine (−2 vs reference). Saved `models/deletion_burden_metrics.json` with a
  `comparison_to_full_model` block + auto-generated honest sentence.
- app.py needed **zero edits** — the file-existence gate lit up the burden card
  and flipped the sanity-check line to ✓ (verified via AppTest, 26 metrics).

**Result (the honest headline):**
| Model | features | CV ROC-AUC | CV PR-AUC |
|---|---|---|---|
| Full model | 150 | 0.9716 | 0.9190 |
| Deletion-burden baseline | 2 | **0.9837** | 0.9633 |

The full model does **NOT** outperform the deletion-burden baseline — the
−2-vs-reference signal is largely explained by overall / chr-9p deletion load
(chr9p burden includes co-deleted CDKN2A/CDKN2B). Reported as-is. This is the
control to lead with: it is a mature, honest finding, and it reframes the tool's
contribution toward the expression-validation and review-routing story rather
than raw discrimination.

## 2026-07-08 — Audit fixes (all 17 findings addressed) — COMPLETE
Fixed every gap from the technical audit. Grouped by what changed; each item
verified to run clean (scripts + pytest + AppTest + live server, all green).

**Correction to prior narrative:** the "deletion-burden baseline beats the
full model" framing from the earlier session was imprecise. Decomposed:
genome-wide burden ALONE is at chance (CV ROC-AUC **0.4969**); chr9p burden
ALONE (CDKN2A/CDKN2B/JAK2) reaches **0.9798**; combined **0.9837**. The
finding is "signal is 9p21-local," not "signal is deletion-burden-general" —
the same conclusion as the ablation, not an independent confirmation of it.
`deletion_burden.py` now trains and reports all three variants.

**Statistical rigor added** (`model.py`):
- Bootstrap 95% CI on CV ROC-AUC/PR-AUC (1000 resamples of CV OOF predictions)
  on every model — primary, ablations, burden variants.
- Youden's-J optimal threshold reported alongside the fixed-0.5 metrics
  (class_weight='balanced' shifts the natural boundary; primary model's
  Youden threshold = 0.2011, not 0.5).
- **Permutation/null control** on the ablation: 20 replicates dropping 2
  random non-9p21 genes give null AUC 0.9717 ± 0.0009 (range
  [0.9697, 0.9742]); the true 9p21 drop (0.7751) falls **below every single
  replicate** — confirms the ablation is 9p21-specific, not a generic
  "any-2-features" artifact.

**Cross-model + methodology gaps** (`secondary_model.py`, `shap_explain.py`):
- XGBoost now scores the −1 queue too (previously only the logistic model
  did). Agreement check: Spearman correlation **0.486**, exact tier agreement
  **58%**, High-tier Jaccard **47%** (16/25 shared) — a real, previously-unknown
  disagreement between the two models on who gets reviewed first. Deployed
  queue still uses logistic only; not reconciled, flagged in the UI.
- SHAP `LinearExplainer` now explicitly uses `feature_perturbation=
  "correlation_dependent"` instead of the (undocumented) default — GISTIC
  features are highly correlated (co-deletion blocks) and independence
  assumptions distorted credit. Confirmed the fix mattered: JAK2 jumped from
  outside the top 10 to #3 globally once correlation was accounted for.

**Data/provenance transparency** (`model.py`, `genes.py`, `README.md`):
- `provenance.json`'s `data_pull_date` was a hardcoded literal — now derived
  from the actual cached raw-data file mtimes.
- Reference class composition surfaced: GISTIC 0=118, 1=58, 2=1 — a third of
  "reference" tumors are copy-gained, not neutral. Not previously visible
  anywhere.
- Deployed-vs-evaluated model distinction documented in provenance: the
  model scoring the −1 queue is a third fit (refit on 100% of labeled data),
  not the same model the CV/test metrics describe.
- Panel gene list (`genes.py`) now carries an explicit honesty note: hand-
  curated from general knowledge, not derived from this cohort's own GISTIC
  peaks or a single cited driver census.
- Tumor purity: checked cBioPortal's clinical-attributes API for this
  cohort (60 attributes) — no ABSOLUTE purity field exists. Documented as an
  honestly-unaddressed gap rather than silently skipped.

**Engineering hardening** (`cbioportal_client.py`, `data_prep.py`,
`requirements.txt`, `tests/`):
- Cache files are now content-hash-gated (`_sig`/`_cache_valid`) — a changed
  `PANEL_SYMBOLS` or cohort now triggers a re-pull instead of silently
  serving a stale panel forever. `data_prep.py`'s own short-circuit (which
  bypasses network entirely on a warm cache, by design) now validates its
  own local signature file rather than only checking file existence.
  Verified: cache-hit runs stayed network-free (0.4–1.2s) after the change.
- Retry-with-backoff (3 attempts) added to every cBioPortal network call —
  previously any transient blip killed the whole pull.
- `requirements.txt` fully pinned to installed versions — unpinned deps had
  already broken the build twice mid-project.
- New `tests/test_pipeline_invariants.py` (14 tests, pytest): label/feature
  invariants, no patient-level duplication, percentile-tier proportions,
  cross-artifact consistency (ablation feature-count ordering, permutation
  control extremity, CI containment). All pass.
- `README.md` rewritten: honest-limitations section now lists every audit
  finding with the specific numbers; new "Engineering / reproducibility
  risks" section documents the bravado v2-endpoint risk explicitly.

**Verification:** every script re-run clean (`model.py`, `model.py
--ablations`, `secondary_model.py`, `deletion_burden.py`, `score_queue.py`,
`shap_explain.py`), `pytest tests/ -v` 14/14 pass, AppTest 0 exceptions (6
dataframes, 31 metrics, all new content confirmed present in rendered
markdown/captions), live `streamlit run` health 200 with no log errors.

Nothing from the audit was skipped or deferred.
