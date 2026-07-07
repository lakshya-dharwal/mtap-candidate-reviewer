"""MTAP-Loss Candidate Reviewer — Streamlit UI (Phases 6 + 7).

Wraps the built pipeline: cohort prevalence, model performance (three-way
ablation), orthogonal expression validation, the enriched review queue (9p21
context + expression percentile + model probability + tier + "why flagged"
SHAP), and the data-integrity spine (provenance/audit + exclusions log).

Styling is IDEAYA-inspired (restrained corporate-scientific look); it does not
impersonate IDEAYA or claim affiliation. This file renders cached artifacts and
does not change any data, model, or metric.
"""

from __future__ import annotations

import json
import os

import joblib
import pandas as pd
import streamlit as st

import data_prep
import model as M
import score_queue
import shap_explain
import validate_expression as VE

STUDY = data_prep.STUDY
MODELS_DIR = M.MODELS_DIR

# --- IDEAYA brand palette --------------------------------------------------- #
NAVY = "#0C2D48"
LIGHT = "#EEF4F8"
BLUE = "#1E9BD7"
INK = "#1A2E3B"
GREEN = "#5BA84A"
AMBER = "#D99A2B"
ACCENTS = ["#2B8A8A", "#5BA84A", "#A3C644"]  # rotate over sections

st.set_page_config(page_title="MTAP-Loss Candidate Reviewer",
                   page_icon="🧬", layout="wide")


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}
    .stApp {{ background: {LIGHT}; }}
    .block-container {{ padding-top: 1.5rem; max-width: 1180px; }}

    h1, h2, h3, h4 {{ font-family: 'Poppins', sans-serif; font-weight: 600; color: {NAVY}; }}

    /* Navy hero banner */
    .hero {{
        background: linear-gradient(135deg, {NAVY} 0%, #0D3556 100%);
        border-radius: 14px; padding: 2.1rem 2.4rem; margin-bottom: 1.8rem;
        box-shadow: 0 6px 22px rgba(12,45,72,0.18);
    }}
    .hero h1 {{ color: #FFFFFF; font-size: 2.05rem; font-weight: 700; margin: 0; }}
    .hero p {{ color: #C7DCEC; font-size: 1.05rem; margin: .5rem 0 0 0; font-family:'Inter'; }}

    /* Section eyebrow + accent rule-line */
    .section-head {{ margin: 2.1rem 0 .6rem 0; }}
    .accent-bar {{ height: 4px; width: 54px; border-radius: 3px; margin-bottom: .55rem; }}
    .eyebrow {{
        font-family: 'Poppins', sans-serif; font-weight: 600; letter-spacing: .13em;
        text-transform: uppercase; font-size: .82rem; color: #4A6478;
    }}
    .section-title {{ font-family:'Poppins'; font-weight:600; font-size:1.35rem;
        color:{NAVY}; margin:.1rem 0 .2rem 0; }}
    .section-sub {{ color:#5B7488; font-size:.92rem; margin-bottom:.4rem; }}

    /* Metric cards */
    div[data-testid="stMetric"] {{
        background: #FFFFFF; border-radius: 12px; padding: 1rem 1.15rem;
        box-shadow: 0 2px 10px rgba(12,45,72,0.07); border: 1px solid #E1EAF1;
    }}
    div[data-testid="stMetricLabel"] p {{ font-weight:600; color:#5B7488;
        text-transform:uppercase; letter-spacing:.05em; font-size:.72rem; }}
    div[data-testid="stMetricValue"] {{ color:{NAVY}; font-family:'Poppins'; }}

    /* Generic card wrapper */
    .card {{ background:#FFFFFF; border-radius:12px; padding:1.2rem 1.4rem;
        box-shadow:0 2px 10px rgba(12,45,72,0.07); border:1px solid #E1EAF1; }}

    .stDataFrame {{ border-radius:10px; overflow:hidden; }}
    a, .stMarkdown a {{ color:{BLUE}; }}
    footer {{ visibility:hidden; }}
    </style>
    """, unsafe_allow_html=True)


def section(eyebrow_label, title, accent, subtitle=None):
    sub = f'<div class="section-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="section-head">
        <div class="accent-bar" style="background:{accent};"></div>
        <div class="eyebrow">{eyebrow_label}</div>
        <div class="section-title">{title}</div>
        {sub}
    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data loading (cache-only; artifacts built in earlier phases)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model and building review queue…")
def load_all():
    ds = data_prep.build_dataset()
    bundle = joblib.load(os.path.join(MODELS_DIR, "model_main.joblib"))
    pipe, feat = bundle["pipeline"], bundle["feature_genes"]

    # Enriched queue: base + expression percentile + why-flagged (SHAP).
    q = score_queue.build_queue(ds)
    merged = pd.read_csv(
        os.path.join(data_prep.CACHE_DIR, f"merged_{STUDY}.csv")).set_index("sampleId")
    expr_pct = (merged["MTAP_rna"].rank(pct=True) * 100).round(1)
    q["MTAP_expr_pctile"] = q["sampleId"].map(expr_pct)

    labeled = ds["y"].notna()
    amb = ds["role"] == "ambiguous"
    why = shap_explain.per_patient_top_features(
        pipe, ds["X"].loc[amb], feat, ds["X"].loc[labeled])
    q["why_flagged"] = q["sampleId"].map(why)

    def _load_json(name):
        p = os.path.join(MODELS_DIR, name)
        return json.load(open(p)) if os.path.exists(p) else None

    return {
        "ds": ds,
        "queue": q,
        "metrics": _load_json("metrics.json"),
        "ablation": _load_json("ablation_metrics.json"),
        "provenance": _load_json("provenance.json"),
        "expr_stats": _load_json(f"expression_validation_{STUDY}.json"),
        "expr_fig": os.path.join(MODELS_DIR, f"expression_validation_{STUDY}.png"),
        "exclusions": pd.read_csv(data_prep.EXCLUSIONS_PATH)
        if os.path.exists(data_prep.EXCLUSIONS_PATH) else pd.DataFrame(),
    }


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def main():
    inject_css()
    st.markdown("""
    <div class="hero">
        <h1>MTAP-Loss Candidate Reviewer</h1>
        <p>An expression-validated review queue for ambiguous MTAP shallow-loss tumors</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_all()
    ds, q = data["ds"], data["queue"]
    role = ds["role"]

    # ---- Cohort & prevalence ---- #
    section("COHORT & PREVALENCE", "Bladder — blca_tcga_pan_can_atlas_2018", ACCENTS[0],
            "Prevalence verified from the pulled cohort, not literature estimates.")
    st.selectbox("Cohort", [STUDY], index=0,
                 help="Additional cohorts (PAAD/GBM) are future work.")
    c = st.columns(4)
    c[0].metric("Confirmed −2 (positive)", int((role == "positive").sum()))
    c[1].metric("Ambiguous −1 (queue)", int((role == "ambiguous").sum()))
    c[2].metric("Reference 0/1/2", int((role == "reference").sum()))
    c[3].metric("Features (MTAP excluded)", len(ds["feature_genes"]))

    # ---- Model performance / ablations ---- #
    section("MODEL PERFORMANCE", "Three-way ablation", ACCENTS[1],
            "L2 logistic regression, class-weight balanced, 5-fold CV, seed 42. "
            "MTAP is the label and never a feature.")
    ab = data["ablation"]
    if ab:
        rows = []
        for m in ab["models"]:
            cv = m["cv_5fold"]
            rows.append({
                "Model": m["label"].replace("_", " "),
                "Features": m["n_features"],
                "CV ROC-AUC": m["cv_roc_auc"], "CV PR-AUC": m["cv_pr_auc"],
                "Sensitivity": cv["sensitivity"], "Specificity": cv["specificity"],
                "F1": cv["f1"], "Test ROC-AUC": m["test_roc_auc"],
            })
        cols = st.columns(3)
        for i, m in enumerate(ab["models"]):
            cols[i].metric(m["label"].replace("_", " "), m["cv_roc_auc"],
                           help="CV 5-fold ROC-AUC")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Model 3 drops the 9p21 neighborhood: "
                   f"{', '.join(ab['nine_p21_features_dropped_in_model3'])}. "
                   f"n_positive={ab['models'][0]['n_positive']}, "
                   f"n_reference={ab['models'][0]['n_reference']}.")

    # ---- Expression validation ---- #
    section("EXPRESSION VALIDATION", "Orthogonal evidence: MTAP RNA-seq", ACCENTS[2],
            "The queue is ranked by CNA-model probability only; expression is shown "
            "as independent validation, not part of the score.")
    es = data["expr_stats"]
    left, right = st.columns([3, 2])
    if os.path.exists(data["expr_fig"]):
        left.image(data["expr_fig"], width="stretch")
    if es:
        med = es["group_median_log2"]
        for k, v in med.items():
            right.metric(k.replace("\n", " "), v, help="median log2(RSEM+1)")
        mw = es["mannwhitney_high_vs_low_ambiguous"]
        p = mw["p_value"]
        color = GREEN if p < 0.05 else AMBER
        right.markdown(
            f"<div class='card'>Mann-Whitney (−1 High vs −1 Low): "
            f"<b>p = {p:.3g}</b> "
            f"<span style='color:{color};font-weight:600;'>"
            f"({'p&lt;0.05' if p < 0.05 else 'n.s.'})</span><br>"
            f"<span style='color:#5B7488;font-size:.85rem;'>"
            f"U={mw['U']:.0f}, n_high={mw['n_high']}, n_low={mw['n_low']}</span></div>",
            unsafe_allow_html=True)

    # ---- Review queue ---- #
    section("REVIEW QUEUE", "Ambiguous −1 cases ranked for orthogonal review", ACCENTS[0],
            "Ranked by model probability of resembling confirmed −2 deletions.")
    tcounts = q["review_tier"].value_counts()
    tc = st.columns(3)
    tc[0].metric("High (>0.75)", int(tcounts.get("High", 0)))
    tc[1].metric("Medium (0.50–0.75)", int(tcounts.get("Medium", 0)))
    tc[2].metric("Low (<0.50)", int(tcounts.get("Low", 0)))

    tiers = st.multiselect("Filter by review tier", ["High", "Medium", "Low"],
                           default=["High", "Medium", "Low"])
    view = q[q["review_tier"].isin(tiers)].copy()
    view = view.rename(columns={
        "sampleId": "Sample", "model_probability": "Model prob.",
        "review_tier": "Tier", "MTAP_cna": "MTAP", "CDKN2A_cna": "CDKN2A",
        "CDKN2B_cna": "CDKN2B", "MTAP_expr_pctile": "MTAP expr %ile",
        "why_flagged": "Why flagged (top SHAP)"})
    view["Model prob."] = view["Model prob."].round(3)
    st.dataframe(
        view[["Sample", "Model prob.", "Tier", "MTAP", "CDKN2A", "CDKN2B",
              "MTAP expr %ile", "Why flagged (top SHAP)"]],
        hide_index=True, width="stretch",
        column_config={"Model prob.": st.column_config.ProgressColumn(
            "Model prob.", min_value=0.0, max_value=1.0, format="%.3f")})
    st.caption(f"n = {len(view)} shown of {len(q)} ambiguous cases. Arrows in "
               "'why flagged' show push toward (↑) / away from (↓) resembling a "
               "confirmed deletion; parenthetical is the patient's GISTIC value.")

    # ---- Data-integrity spine ---- #
    section("DATA PROVENANCE", "Audit & inspection readiness", ACCENTS[1])
    prov = data["provenance"]
    if prov:
        pc = st.columns(4)
        pc[0].metric("Cohort", prov["cohort"].split("_")[0].upper())
        pc[1].metric("Data pull date", prov["data_pull_date"])
        pc[2].metric("Random seed", prov["random_seed"])
        pc[3].metric("Model version", prov["model_version"])
        with st.expander("Full provenance record"):
            st.json(prov)

    section("EXCLUSIONS LOG", "What was dropped and why", ACCENTS[2])
    ex = data["exclusions"]
    if not ex.empty:
        st.dataframe(ex, hide_index=True, width="stretch")
    st.caption("Clean-label discipline: high-confidence −2 / reference set trains "
               "the model; the ambiguous −1 middle is quarantined for review.")


if __name__ == "__main__":
    main()
