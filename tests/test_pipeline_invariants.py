"""Regression tests for pipeline invariants — the gap audit found ZERO
automated tests existed; all verification was manual script runs + eyeballing
printed numbers. These are cheap, cache-only checks (no network) that would
catch a broken rebuild before it reaches the app.

Run: pytest tests/ -v
"""

import json
import os

import pandas as pd
import pytest

import data_prep
import genes as G
import model as M

MODELS_DIR = M.MODELS_DIR


@pytest.fixture(scope="module")
def ds():
    return data_prep.build_dataset()


# --------------------------------------------------------------------------- #
# data_prep invariants
# --------------------------------------------------------------------------- #
def test_mtap_excluded_from_features(ds):
    assert "MTAP" not in ds["feature_genes"], "MTAP is the label — must never be a feature"


def test_roles_are_mutually_exclusive_and_complete(ds):
    roles = ds["role"]
    assert set(roles.unique()) <= {"positive", "reference", "ambiguous"}
    assert len(roles) == len(ds["X"]), "every labelable sample must have exactly one role"


def test_y_matches_role(ds):
    role, y = ds["role"], ds["y"]
    assert (y[role == "positive"] == 1).all()
    assert (y[role == "reference"] == 0).all()
    assert y[role == "ambiguous"].isna().all()


def test_no_missing_gistic_cells(ds):
    assert ds["X"].isna().sum().sum() == 0, "features must be fully imputed by this point"


def test_no_patient_level_duplication(ds):
    patients = ds["X"].index.str.slice(0, 12)
    assert patients.nunique() == len(ds["X"]), \
        "duplicate patient barcodes would leak across train/test/CV splits"


def test_feature_count_matches_panel_minus_label(ds):
    # PANEL_SYMBOLS may include duplicates/unresolvable symbols; feature count
    # should be at most len(PANEL_SYMBOLS) - 1 (MTAP) and reasonably close to it.
    assert 1 < len(ds["feature_genes"]) <= len(set(G.PANEL_SYMBOLS)) - 1


# --------------------------------------------------------------------------- #
# score_queue invariants
# --------------------------------------------------------------------------- #
def test_percentile_tiers_sum_to_ambiguous_count(ds):
    import score_queue
    q = score_queue.build_queue(ds)
    assert len(q) == int((ds["role"] == "ambiguous").sum())
    assert set(q["review_tier"].unique()) <= {"High", "Medium", "Low"}
    counts = q["review_tier"].value_counts()
    # Roughly 20/30/50 split (percentile-based, so not exact due to ties/rounding).
    assert abs(counts.get("High", 0) / len(q) - 0.20) < 0.05
    assert abs(counts.get("Low", 0) / len(q) - 0.50) < 0.05


def test_queue_scores_in_unit_interval(ds):
    import score_queue
    q = score_queue.build_queue(ds)
    assert q["model_probability"].between(0, 1).all()


# --------------------------------------------------------------------------- #
# Saved-artifact sanity checks (only run if the artifact exists — these are
# regression checks on the LAST run's output, not a mandate to always exist).
# --------------------------------------------------------------------------- #
def _load(name):
    p = os.path.join(MODELS_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


def test_primary_metrics_internally_consistent():
    m = _load("metrics.json")
    if m is None:
        pytest.skip("metrics.json not built yet")
    assert m["n_positive"] + m["n_reference"] == 283  # 106 + 177, this cohort
    assert 0 <= m["cv_roc_auc"] <= 1
    cm = m["cv_5fold"]["confusion_matrix"]
    assert cm["tp"] + cm["fn"] == m["n_positive"]
    assert cm["tn"] + cm["fp"] == m["n_reference"]
    ci = m["cv_auc_ci95"]["roc_auc_ci95"]
    assert ci[0] <= m["cv_roc_auc"] <= ci[1], "point estimate should fall within its own CI"


def test_ablation_feature_counts_decrease_monotonically():
    ab = _load("ablation_metrics.json")
    if ab is None:
        pytest.skip("ablation_metrics.json not built yet")
    counts = [m["n_features"] for m in ab["models"]]
    assert counts == sorted(counts, reverse=True), \
        "with_CDKN2A > without_CDKN2A > without_9p21 in feature count"


def test_permutation_control_flags_9p21_as_extreme():
    ab = _load("ablation_metrics.json")
    if ab is None or "permutation_control" not in ab:
        pytest.skip("permutation_control not built yet")
    pc = ab["permutation_control"]
    # The whole point of this control: the true 9p21 drop should be a clear
    # outlier vs random 2-gene drops, not indistinguishable from noise.
    assert pc["without_9p21_auc"] < pc["null_auc_min"], \
        "9p21 ablation AUC should fall below the random-drop null range"


def test_deletion_burden_decomposition_present():
    db = _load("deletion_burden_metrics.json")
    if db is None:
        pytest.skip("deletion_burden_metrics.json not built yet")
    assert "variants" in db, "must report genome-only/chr9p-only/combined separately"
    assert set(db["variants"]) == {"genome_only", "chr9p_only", "genome_plus_chr9p"}


def test_queue_agreement_artifact_present():
    agree = _load("queue_model_agreement.json")
    if agree is None:
        pytest.skip("queue_model_agreement.json not built yet")
    assert 0 <= agree["tier_exact_agreement_rate"] <= 1
    assert -1 <= agree["spearman_score_correlation"] <= 1


def test_exclusions_log_is_written():
    assert os.path.exists(data_prep.EXCLUSIONS_PATH)
    ex = pd.read_csv(data_prep.EXCLUSIONS_PATH)
    assert "reason" in ex.columns
