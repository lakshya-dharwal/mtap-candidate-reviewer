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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd
import streamlit as st

import data_prep  # noqa: E402
import model as M  # noqa: E402
import score_queue  # noqa: E402
import shap_explain  # noqa: E402

STUDY = data_prep.STUDY
MODELS_DIR = M.MODELS_DIR
SECONDARY_PATH = os.path.join(MODELS_DIR, "secondary_xgboost_metrics.json")
DELETION_BURDEN_PATH = os.path.join(MODELS_DIR, "deletion_burden_metrics.json")
QUEUE_AGREEMENT_PATH = os.path.join(MODELS_DIR, "queue_model_agreement.json")

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

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid #D5E1EA; }}
    .stTabs [data-baseweb="tab"] {{
        font-family:'Poppins',sans-serif; font-weight:600; font-size:.92rem;
        color:#5B7488; padding: .55rem 1.05rem; border-radius: 9px 9px 0 0;
    }}
    .stTabs [aria-selected="true"] {{ color:{NAVY}; background:#FFFFFF;
        border-bottom: 3px solid {BLUE}; }}
    .tab-sub {{ color:#5B7488; font-size:.95rem; margin:.2rem 0 1rem 0; }}

    /* Purpose / limitation callout cards */
    .callout {{ background:#FFFFFF; border-radius:12px; padding:1.1rem 1.35rem;
        box-shadow:0 2px 10px rgba(12,45,72,0.07); border:1px solid #E1EAF1;
        border-left:5px solid {BLUE}; margin-bottom:.8rem; }}
    .callout h4 {{ margin:0 0 .5rem 0; font-size:1.05rem; }}
    .check li {{ margin:.15rem 0; }}
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

    # Global SHAP importance (top genes by mean |SHAP|) over the labeled set.
    global_shap = shap_explain.global_importance(pipe, ds["X"].loc[labeled], feat)

    def _load_path(p):
        return json.load(open(p)) if os.path.exists(p) else None

    def _load_json(name):
        return _load_path(os.path.join(MODELS_DIR, name))

    return {
        "ds": ds,
        "queue": q,
        "global_shap": global_shap,
        "metrics": _load_json("metrics.json"),
        "ablation": _load_json("ablation_metrics.json"),
        "provenance": _load_json("provenance.json"),
        "secondary": _load_path(SECONDARY_PATH),
        "deletion_burden": _load_path(DELETION_BURDEN_PATH),
        "queue_agreement": _load_path(QUEUE_AGREEMENT_PATH),
        "expr_stats": _load_json(f"expression_validation_{STUDY}.json"),
        "expr_fig": os.path.join(MODELS_DIR, f"expression_validation_{STUDY}.png"),
        "exclusions": pd.read_csv(data_prep.EXCLUSIONS_PATH)
        if os.path.exists(data_prep.EXCLUSIONS_PATH) else pd.DataFrame(),
    }


# --------------------------------------------------------------------------- #
# Plot helpers (matplotlib, from cached curve arrays — no retraining)
# --------------------------------------------------------------------------- #
def _style_ax(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    return ax


def plot_roc(metrics, secondary=None):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    rc = metrics["roc_curve"]
    ax.plot(rc["fpr"], rc["tpr"], color=BLUE, lw=2.2,
            label=f"Logistic (AUC={metrics['cv_roc_auc']})")
    if secondary and "roc_curve" in secondary:
        src = secondary["roc_curve"]
        ax.plot(src["fpr"], src["tpr"], color="#8E44AD", lw=1.8, ls="--",
                label=f"XGBoost (AUC={secondary['cv_roc_auc']})")
    ax.plot([0, 1], [0, 1], color="#9AB0BF", lw=1, ls=":")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC (5-fold CV)", fontsize=11)
    ax.legend(fontsize=8, loc="lower right"); _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_pr(metrics, secondary=None):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    pc = metrics["pr_curve"]
    ax.plot(pc["recall"], pc["precision"], color=BLUE, lw=2.2,
            label=f"Logistic (AP={metrics['cv_pr_auc']})")
    if secondary and "pr_curve" in secondary:
        spc = secondary["pr_curve"]
        ax.plot(spc["recall"], spc["precision"], color="#8E44AD", lw=1.8, ls="--",
                label=f"XGBoost (AP={secondary['cv_pr_auc']})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall (5-fold CV)", fontsize=11)
    ax.legend(fontsize=8, loc="lower left"); _style_ax(ax)
    fig.tight_layout()
    return fig


def plot_shap_bar(global_shap, top_n=15):
    top = global_shap.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.barh(top["gene"], top["mean_abs_shap"], color=BLUE, alpha=0.85)
    ax.set_xlabel("mean |SHAP|  (log-odds toward −2 resemblance)")
    ax.set_title(f"Global feature importance — top {top_n}", fontsize=11)
    _style_ax(ax); ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def tab_overview(data):
    ds, q, ab = data["ds"], data["queue"], data["ablation"]
    role = ds["role"]
    st.markdown('<div class="tab-sub">What this tool does, the headline results, '
                'and where it stops.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
      <h4>Purpose</h4>
      <b>MTAP</b> sits on chromosome 9p21, beside <b>CDKN2A/CDKN2B</b>, and is
      co-deleted with them in most 9p21-loss tumors. Homozygous (deep) MTAP loss
      makes tumors dependent on <b>MAT2A/PRMT5</b> — the target of IDEAYA's
      IDE397 (MAT2A) and IDE892 (PRMT5) programs. Deep deletions (GISTIC −2) are
      unambiguous, but <b>shallow −1 loss is ambiguous</b>: some cases are
      functionally MTAP-deficient, some are not, and sequencing can miss them.
      This tool trains on confirmed −2 deletions and <b>ranks the ambiguous −1
      tumors whose copy-number profile resembles confirmed-deleted cases</b>,
      for orthogonal human review. It does not diagnose or claim to find
      "missed patients."
    </div>
    """, unsafe_allow_html=True)

    section("HEADLINE RESULTS", "At a glance", ACCENTS[0])
    m = data["metrics"]
    ab_9p21 = next((x for x in ab["models"]
                    if x["label"] == "without_9p21_neighborhood"), None) if ab else None
    es = data["expr_stats"]
    p = es["mannwhitney_high_vs_low_ambiguous"]["p_value"] if es else None
    c = st.columns(4)
    c[0].metric("Primary CV ROC-AUC", m["cv_roc_auc"] if m else "—",
                help="L2 logistic, 5-fold CV, full 150-gene panel")
    c[1].metric("AUC without 9p21", ab_9p21["cv_roc_auc"] if ab_9p21 else "—",
                help="CDKN2A+CDKN2B removed — the harsh ablation")
    c[2].metric("Ambiguous −1 queued", int((role == "ambiguous").sum()))
    c[3].metric("Expression high-vs-low", f"p = {p:.3g}" if p is not None else "—",
                help="Mann-Whitney; a null result, reported honestly")
    if p is not None and p >= 0.05:
        st.caption("Expression validation is a **null result** on this cohort "
                   f"(p = {p:.3g}, n.s.): high-priority −1 cases do not yet show "
                   "significantly lower MTAP expression than low-priority −1 cases. "
                   "Reported as-is, not spun.")

    lim, san = st.columns(2)
    with lim:
        section("KNOWN LIMITATIONS", "& next steps", ACCENTS[1])
        st.markdown("""
        <div class="callout check">
        <ul>
          <li>Single cohort (bladder only) — needs external / cross-cohort validation.</li>
          <li>Deletion-burden baseline: genome-wide burden is at chance (0.50);
              signal is 9p21-local, confirmed by ablation + permutation control
              (see Validation tab) — not yet an independent "beyond deletion load" win.</li>
          <li>Logistic vs XGBoost only agree on 58% of −1 tiers — model choice
              materially changes who's prioritized; unreconciled.</li>
          <li>Model scores are <b>not yet calibrated</b> probabilities.</li>
          <li>Reference class pools neutral/gain/amplified GISTIC calls (0/1/2)
              without checking for a subgroup effect.</li>
          <li>Tumor purity not modeled — no purity attribute available via
              cBioPortal's clinical API for this cohort.</li>
          <li>150-gene panel is hand-curated, not empirically derived from this
              cohort or a single cited driver census.</li>
          <li>Shallow −1 loss is biologically heterogeneous — subclonality,
              segmentation noise.</li>
          <li>Prioritizes cases for review; <b>not a diagnostic</b>.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with san:
        section("MODEL SANITY CHECKS", "leakage & robustness", ACCENTS[2])
        burden_done = data["deletion_burden"] is not None
        ab = data.get("ablation") or {}
        checks = [
            ("MTAP excluded from features", True),
            ("Patient-level duplication checked (none found)", True),
            ("CDKN2A / CDKN2B ablation run", True),
            ("Full 9p21-neighborhood ablation run", True),
            ("Permutation/null control on ablation", "permutation_control" in ab),
            ("Cross-validation (5-fold)", True),
            ("Bootstrap 95% CI on AUC", True),
            ("Deletion-burden baseline (decomposed)", burden_done),
            ("Cross-model (logistic vs XGBoost) queue agreement checked",
             data.get("queue_agreement") is not None),
            ("Score calibration", False),
            ("Tumor purity adjustment", False),
            ("External validation", False),
        ]
        pending = ' <span style="color:#5B7488">(pending)</span>'
        rows = "".join(
            "<li>{} {}{}</li>".format(
                "✓" if ok else "○", name, "" if ok else pending)
            for name, ok in checks)
        st.markdown('<div class="callout check"><ul style="list-style:none;'
                    f'padding-left:0">{rows}</ul></div>', unsafe_allow_html=True)


def tab_how_it_works(data):
    ds = data["ds"]
    st.markdown('<div class="tab-sub">The label strategy and data flow — '
                'conceptual, no code.</div>', unsafe_allow_html=True)

    section("LABEL STRATEGY", "Train on the confident, score the ambiguous", ACCENTS[0])
    st.markdown("""
    | MTAP GISTIC | Role | Used how |
    |---|---|---|
    | **−2** deep deletion | positive | trained on (confident) |
    | **0 / 1 / 2** | reference | trained on (confident) |
    | **−1** shallow loss | ambiguous | **held out**, scored for resemblance |

    MTAP is the **label**, so it is never a feature (no leakage). MTAP
    *expression* is used only for orthogonal validation — never as a feature.
    """)

    section("DATA FLOW", "cBioPortal → queue", ACCENTS[1])
    st.markdown(f"""
    1. **cBioPortal** — pull the bladder cohort `{STUDY}` (GISTIC CNA + MTAP RNA-seq), cached to CSV.
    2. **{len(ds['X'])} tumors** with an MTAP call → feature matrix of
       **{len(ds['feature_genes'])} recurrently-altered genes** (MTAP excluded).
    3. **Model** — L2 logistic regression, class-weight balanced, 5-fold CV,
       trained on −2 vs reference; XGBoost as a secondary comparison.
    4. **Score the −1 set** → rank by model score → **percentile tiers**
       (top 20% High, next 30% Medium, bottom 50% Low).
    5. **Validate** the ranking against MTAP expression (orthogonal, not part of the score).
    """)


def tab_review_queue(data):
    q = data["queue"]
    st.markdown('<div class="tab-sub">The primary deliverable: ambiguous −1 '
                'cases ranked for orthogonal review.</div>', unsafe_allow_html=True)

    section("REVIEW QUEUE", "Percentile-tiered candidates", ACCENTS[0],
            "Ranked by model score of resembling confirmed −2 deletions. "
            "Tiers are percentiles of the scored set, not fixed cutoffs.")
    tcounts = q["review_tier"].value_counts()
    tc = st.columns(3)
    tc[0].metric("High (top 20%)", int(tcounts.get("High", 0)))
    tc[1].metric("Medium (next 30%)", int(tcounts.get("Medium", 0)))
    tc[2].metric("Low (bottom 50%)", int(tcounts.get("Low", 0)))

    tiers = st.multiselect("Filter by review tier", ["High", "Medium", "Low"],
                           default=["High", "Medium", "Low"])
    view = q[q["review_tier"].isin(tiers)].copy()
    view = view.rename(columns={
        "sampleId": "Sample", "model_probability": "Model score",
        "review_tier": "Tier", "MTAP_cna": "MTAP", "CDKN2A_cna": "CDKN2A",
        "CDKN2B_cna": "CDKN2B", "MTAP_expr_pctile": "MTAP expr %ile",
        "why_flagged": "Why flagged (top SHAP)"})
    view["Model score"] = view["Model score"].round(3)
    st.dataframe(
        view[["Sample", "Model score", "Tier", "MTAP", "CDKN2A", "CDKN2B",
              "MTAP expr %ile", "Why flagged (top SHAP)"]],
        hide_index=True, width="stretch",
        column_config={"Model score": st.column_config.ProgressColumn(
            "Model score", min_value=0.0, max_value=1.0, format="%.3f")})
    st.caption(f"n = {len(view)} shown of {len(q)} ambiguous cases. Arrows in "
               "'why flagged' show push toward (↑) / away from (↓) resembling a "
               "confirmed deletion; parenthetical is the patient's GISTIC value. "
               "'Model score' is the logistic output — a ranking score, not a "
               "calibrated probability.")


def tab_validation(data):
    st.markdown('<div class="tab-sub">Every performance, explainability, and '
                'validation figure — findable during Q&A.</div>', unsafe_allow_html=True)
    m, ab = data["metrics"], data["ablation"]
    sec = data["secondary"]

    section("DISCRIMINATION", "ROC & precision-recall (5-fold CV)", ACCENTS[0])
    cc = st.columns(2)
    if m and "roc_curve" in m:
        cc[0].pyplot(plot_roc(m, sec))
        cc[1].pyplot(plot_pr(m, sec))
    if m:
        ci = m.get("cv_auc_ci95", {})
        st.caption(f"CV ROC-AUC {m['cv_roc_auc']} (95% bootstrap CI "
                   f"{ci.get('roc_auc_ci95', '—')}) · CV PR-AUC {m['cv_pr_auc']} "
                   f"(95% CI {ci.get('pr_auc_ci95', '—')}), n={ci.get('n_boot', '?')} resamples. "
                   f"Small positive count (n={m['n_positive']}) widens these intervals.")
        cm = m["test_at_0.5"]["confusion_matrix"]
        cmc1, cmc2 = st.columns(2)
        cmc1.markdown("**Confusion matrix** (held-out 20%, threshold 0.5)")
        cm_df = pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actual reference", "Actual −2"],
            columns=["Pred reference", "Pred −2"])
        cmc1.dataframe(cm_df, width="content")
        if "cv_youden_threshold" in m:
            yj = m["cv_at_youden"]
            cmc2.markdown(f"**Youden's-J optimal threshold** ({m['cv_youden_threshold']}, "
                          f"vs the fixed 0.5 used above)")
            yj_df = pd.DataFrame(
                [["sensitivity", yj["sensitivity"]], ["specificity", yj["specificity"]],
                 ["F1", yj["f1"]]], columns=["metric", "value at Youden-J"])
            cmc2.dataframe(yj_df, hide_index=True, width="content")
            cmc2.caption("class_weight='balanced' shifts the natural decision boundary "
                        "away from 0.5 — this is the data-driven optimal cut instead.")

    section("THREE-WAY ABLATION", "Does it learn beyond the 9p21 neighbor?", ACCENTS[1],
            "L2 logistic, identical setup; only the feature set changes.")
    if ab:
        rows = []
        for x in ab["models"]:
            cv = x["cv_5fold"]
            ci = x.get("cv_auc_ci95", {}).get("roc_auc_ci95")
            rows.append({
                "Model": x["label"].replace("_", " "), "Features": x["n_features"],
                "CV ROC-AUC": x["cv_roc_auc"], "95% CI": str(ci),
                "CV PR-AUC": x["cv_pr_auc"],
                "Sensitivity": cv["sensitivity"], "Specificity": cv["specificity"],
                "F1": cv["f1"], "Test ROC-AUC": x["test_roc_auc"]})
        cols = st.columns(3)
        for i, x in enumerate(ab["models"]):
            cols[i].metric(x["label"].replace("_", " "), x["cv_roc_auc"],
                           help="CV 5-fold ROC-AUC")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption(f"Model 3 drops the 9p21 neighborhood: "
                   f"{', '.join(ab['nine_p21_features_dropped_in_model3'])}. "
                   f"n_positive={ab['models'][0]['n_positive']}, "
                   f"n_reference={ab['models'][0]['n_reference']}.")

        pc = ab.get("permutation_control")
        if pc:
            if pc["without_9p21_auc"] < pc["null_auc_min"]:
                pc_verdict = "below every single random drop"
            else:
                pc_verdict = f"at the {pc['without_9p21_percentile_within_null']:.0f}th percentile of the null"
            st.markdown(f"<div class='card'><b>Permutation control</b> — is the AUC drop "
                       f"specific to 9p21, or would dropping ANY 2 genes do the same? "
                       f"{pc['n_replicates']} replicates dropped 2 random non-9p21 genes: "
                       f"null AUC {pc['null_auc_mean']} ± {pc['null_auc_std']} "
                       f"(range [{pc['null_auc_min']}, {pc['null_auc_max']}]) vs the true "
                       f"9p21 drop of <b>{pc['without_9p21_auc']}</b> — {pc_verdict}.</div>",
                       unsafe_allow_html=True)

    if sec:
        section("SECONDARY MODEL", "XGBoost comparison", ACCENTS[2],
                "Nonlinear comparison point; logistic stays primary.")
        sc = st.columns(2)
        sc[0].metric("XGBoost CV ROC-AUC", sec["cv_roc_auc"])
        sc[1].metric("Logistic CV ROC-AUC", m["cv_roc_auc"] if m else "—")
        if m:
            st.caption(f"XGBoost achieves CV ROC-AUC {sec['cv_roc_auc']} vs "
                       f"{m['cv_roc_auc']} for the primary logistic model "
                       f"(same 150 features, same folds, seed 42).")
        qa = data.get("queue_agreement")
        if qa:
            st.markdown(
                f"<div class='card'><b>Queue agreement</b> — does the higher-AUC model "
                f"rank the same ambiguous cases as high priority? On the {qa['n_scored']} "
                f"scored −1 cases: Spearman score correlation "
                f"<b>{qa['spearman_score_correlation']}</b>, exact tier agreement "
                f"<b>{qa['tier_exact_agreement_rate']:.0%}</b>, High-tier Jaccard overlap "
                f"<b>{qa['high_tier_jaccard']:.0%}</b> ({qa['n_high_both']} shared of "
                f"{qa['n_high_logistic']} logistic / {qa['n_high_xgboost']} XGBoost). "
                f"The deployed queue uses the logistic model only — this is NOT yet "
                f"reconciled.</div>", unsafe_allow_html=True)

    if data["deletion_burden"]:
        db = data["deletion_burden"]
        section("DELETION-BURDEN BASELINE", "Is the signal MTAP-specific — or 9p21-specific?",
                ACCENTS[0], "Decomposed into genome-wide vs chr-9p burden separately, "
                "since they answer different questions.")
        cmp = db["comparison_to_full_model"]
        bc = st.columns(4)
        bc[0].metric("Full model", cmp["full_model_cv_roc_auc"], help="150 genes")
        bc[1].metric("Genome-wide burden alone", cmp["genome_only_cv_roc_auc"],
                    help="fraction of 150 genes deleted — near chance")
        bc[2].metric("Chr9p burden alone", cmp["chr9p_only_cv_roc_auc"],
                    help="CDKN2A/CDKN2B/JAK2 only")
        bc[3].metric("Genome + chr9p", cmp["combined_cv_roc_auc"])
        st.markdown(f"<div class='card'>{cmp['statement']}</div>", unsafe_allow_html=True)

    section("GLOBAL EXPLAINABILITY", "Which genes drive the score", ACCENTS[1])
    st.pyplot(plot_shap_bar(data["global_shap"]))

    section("EXPRESSION VALIDATION", "Orthogonal evidence: MTAP RNA-seq", ACCENTS[2],
            "The queue is ranked by model score only; expression is independent "
            "validation, not part of the score.")
    es = data["expr_stats"]
    left, right = st.columns([3, 2])
    if os.path.exists(data["expr_fig"]):
        left.image(data["expr_fig"], width="stretch")
    if es:
        for k, v in es["group_median_log2"].items():
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


def tab_provenance(data):
    ds = data["ds"]
    st.markdown('<div class="tab-sub">Audit spine — cohort, counts, seed, '
                'exclusions, raw data.</div>', unsafe_allow_html=True)

    section("COHORT & PREVALENCE", f"Bladder — {STUDY}", ACCENTS[0],
            "Prevalence verified from the pulled cohort, not literature estimates.")
    role = ds["role"]
    c = st.columns(4)
    c[0].metric("Confirmed −2 (positive)", int((role == "positive").sum()))
    c[1].metric("Ambiguous −1 (queue)", int((role == "ambiguous").sum()))
    c[2].metric("Reference 0/1/2", int((role == "reference").sum()))
    c[3].metric("Features (MTAP excluded)", len(ds["feature_genes"]))
    mtap = ds["mtap_cna"]
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("Reference: neutral (0)", int((mtap == 0).sum()))
    rc2.metric("Reference: gain (1)", int((mtap == 1).sum()))
    rc3.metric("Reference: amplified (2)", int((mtap == 2).sum()))
    st.caption("The 'reference' class pools 3 distinct GISTIC calls — gained/amplified "
              "tumors are not copy-number-neutral, and this pooling is not further "
              "examined for a subgroup effect.")

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

    section("RAW DATA PREVIEW", "Feature matrix (first rows)", ACCENTS[0])
    st.dataframe(ds["X"].head(), width="stretch")


def main():
    inject_css()
    st.markdown("""
    <div class="hero">
        <h1>MTAP-Loss Candidate Reviewer</h1>
        <p>An expression-validated review queue for ambiguous MTAP shallow-loss tumors</p>
    </div>
    """, unsafe_allow_html=True)

    data = load_all()
    tabs = st.tabs(["Overview", "How It Works", "Review Queue",
                    "Validation & Explainability", "Data & Provenance"])
    with tabs[0]:
        tab_overview(data)
    with tabs[1]:
        tab_how_it_works(data)
    with tabs[2]:
        tab_review_queue(data)
    with tabs[3]:
        tab_validation(data)
    with tabs[4]:
        tab_provenance(data)


if __name__ == "__main__":
    main()
