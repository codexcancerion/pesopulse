"""
PesoPulse: MIT 118 Final Group Project (Team 3)
Local Price Coordinating Council (DTI & DA) Supply-Shock & Inflation Intelligence
Data: PSA OpenSTAT 2M (8 Joined Tables across 101 Provinces & HUCs, Jan 2018 to Oct 2026)
Engine: DuckDB SQL + Plotly + Ridge (All-Income & Bottom-30%) / XGBoost / Quantile GBR / SARIMAX
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix, mean_absolute_error, r2_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb

st.set_page_config(
    page_title="PesoPulse | LPCC Provincial Inflation & Supply-Shock Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PANEL_PARQUET = DATA_DIR / "cpi_provincial_ml_panel.parquet"
DIV_PARQUET = DATA_DIR / "cpi_division_monthly.parquet"
REG_CSV = DATA_DIR / "cpi_regional_monthly.csv"
SUMMARY_JSON = DATA_DIR / "analytics_summary.json"
SHAP_CSV = DATA_DIR / "treeshap_feature_importance.csv"

SECTIONS = [
    "1. Household Price Exposure",
    "2. Provincial Hotspots & Flows",
    "3. Next-Month Forecasts & Alerts",
    "4. LPCC Supply-Shock Simulator",
]

FEATURE_LABELS = {
    "headline_yoy_pct": "Current Headline Inflation (%)",
    "yoy_roll3_mean": "3-Month Inflation Momentum (%)",
    "natl_core_yoy_pct": "National Core Baseline (%)",
    "month_sin": "Seasonal Harvest Cycle (Sine)",
    "month_cos": "Seasonal Harvest Cycle (Cosine)",
    "gwpi_all_yoy_lag1": "Prior-Month Wholesale Index (GWPI %)",
    "gwpi_fuel_yoy_lag1": "Prior-Month Wholesale Fuel (GWPI %)",
    "headline_mom_pct": "Month-on-Month Price Jump (%)",
    "personal_fuel_yoy_pct": "Diesel & Gasoline Inflation (%)",
    "transport_yoy_pct": "Transport Sector Inflation (%)",
    "yoy_lag1": "1-Month Lagged Inflation (%)",
    "yoy_lag2": "2-Month Lagged Inflation (%)",
    "yoy_lag3": "3-Month Lagged Inflation (%)",
    "yoy_lag6": "6-Month Lagged Inflation (%)",
    "yoy_lag12": "12-Month Base-Effect Lag (%)",
    "yoy_roll6_std": "6-Month Price Volatility (SD)",
    "food_bev_yoy_pct": "Food & Beverages Inflation (%)",
    "rice_cpi_yoy_pct": "Provincial Rice CPI Inflation (%)",
    "elec_gas_fuel_yoy_pct": "Electricity, Gas & LPG Inflation (%)",
    "restaurants_accomm_yoy_pct": "Restaurants & Dining Inflation (%)",
    "food_bev_weight": "Local Food Basket Weight (%)",
    "rice_weight": "Local Rice Basket Weight (%)",
    "elec_gas_fuel_weight": "Local Utilities Basket Weight (%)",
    "b30_inflation_spread_pp": "Bottom-30% Inequality Gap (pp)",
    "ppp_index": "Purchasing Power of 2018 Peso",
    "retail_rice_wm_php": "Well-Milled Rice Price (PHP/kg)",
    "retail_rice_wm_yoy_pct": "Retail Rice Price Growth (%)",
    "latitude": "Province Latitude",
    "longitude": "Province Longitude",
}


def humanize_feature(raw_name: str) -> str:
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    return raw_name.replace("cat__", "").replace("num__", "").replace("_", " ").title()


@st.cache_resource
def get_duckdb():
    con = duckdb.connect(":memory:")
    p_panel = str(PANEL_PARQUET).replace("\\", "/")
    p_div = str(DIV_PARQUET).replace("\\", "/")
    p_reg = str(REG_CSV).replace("\\", "/")
    con.execute(f"CREATE VIEW prov_panel AS SELECT * FROM read_parquet('{p_panel}')")
    con.execute(f"CREATE VIEW div_monthly AS SELECT * FROM read_parquet('{p_div}')")
    con.execute(f"CREATE VIEW reg_monthly AS SELECT * FROM read_csv_auto('{p_reg}', header=True)")
    return con


@st.cache_resource
def train_interactive_models():
    con = get_duckdb()
    df = con.execute("""
        SELECT * FROM prov_panel
        WHERE headline_yoy_pct IS NOT NULL
          AND target_yoy_t1 IS NOT NULL
          AND b30_headline_yoy_pct IS NOT NULL
          AND yoy_lag1 IS NOT NULL
          AND yoy_lag3 IS NOT NULL
          AND yoy_lag12 IS NOT NULL
        ORDER BY clean_geo, obs_date
    """).df()
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    df["target_b30_t1"] = df.groupby("clean_geo")["b30_headline_yoy_pct"].shift(-1)
    df = df.dropna(subset=["target_b30_t1"]).sort_values(["obs_date", "clean_geo"]).reset_index(drop=True)

    num_feats = [
        "month_sin", "month_cos", "latitude", "longitude",
        "headline_yoy_pct", "headline_mom_pct",
        "yoy_lag1", "yoy_lag2", "yoy_lag3", "yoy_lag6", "yoy_lag12",
        "yoy_roll3_mean", "yoy_roll6_std",
        "food_bev_yoy_pct", "rice_cpi_yoy_pct",
        "elec_gas_fuel_yoy_pct", "transport_yoy_pct", "personal_fuel_yoy_pct",
        "restaurants_accomm_yoy_pct",
        "food_bev_weight", "rice_weight", "elec_gas_fuel_weight",
        "b30_inflation_spread_pp", "ppp_index",
        "natl_core_yoy_pct", "gwpi_all_yoy_lag1", "gwpi_fuel_yoy_lag1",
        "retail_rice_wm_php", "retail_rice_wm_yoy_pct",
    ]
    cat_feats = ["parent_region", "island_group"]
    X = df[num_feats + cat_feats].copy()
    y_reg = df["target_yoy_t1"].values
    y_b30 = df["target_b30_t1"].values
    y_clf = df["target_breach_4pct_t1"].astype(int).values

    tr_mask = df["year"] <= 2024
    te_mask = df["year"] >= 2025

    def make_pre():
        return ColumnTransformer([
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ])

    ridge = Pipeline([("pre", make_pre()), ("m", Ridge(alpha=10.0))])
    b30_ridge = Pipeline([("pre", make_pre()), ("m", Ridge(alpha=10.0))])
    clf = Pipeline([
        ("pre", make_pre()),
        ("m", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric="logloss", n_jobs=-1,
        )),
    ])

    ridge.fit(X[tr_mask], y_reg[tr_mask])
    b30_ridge.fit(X[tr_mask], y_b30[tr_mask])
    clf.fit(X[tr_mask], y_clf[tr_mask])

    X_tr_enc = ridge.named_steps["pre"].transform(X[tr_mask])
    q10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.10, n_estimators=120, max_depth=4, learning_rate=0.06, random_state=42
    ).fit(X_tr_enc, y_reg[tr_mask])
    q90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=120, max_depth=4, learning_rate=0.06, random_state=42
    ).fit(X_tr_enc, y_reg[tr_mask])

    preds_te = ridge.predict(X[te_mask])
    b30_preds_te = b30_ridge.predict(X[te_mask])
    probs_te = clf.predict_proba(X[te_mask])[:, 1]

    return {
        "df": df,
        "num_feats": num_feats,
        "cat_feats": cat_feats,
        "ridge": ridge,
        "b30_ridge": b30_ridge,
        "clf": clf,
        "q10": q10,
        "q90": q90,
        "y_te_reg": y_reg[te_mask],
        "preds_te": preds_te,
        "y_te_b30": y_b30[te_mask],
        "b30_preds_te": b30_preds_te,
        "y_te_clf": y_clf[te_mask],
        "probs_te": probs_te,
        "ridge_mae": float(mean_absolute_error(y_reg[te_mask], preds_te)),
        "ridge_r2": float(r2_score(y_reg[te_mask], preds_te)),
        "b30_mae": float(mean_absolute_error(y_b30[te_mask], b30_preds_te)),
        "b30_r2": float(r2_score(y_b30[te_mask], b30_preds_te)),
        "clf_auc": float(roc_auc_score(y_clf[te_mask], probs_te)),
        "clf_acc": float(( (probs_te >= 0.5).astype(int) == y_clf[te_mask] ).mean()),
        "te_meta": df.loc[
            te_mask,
            ["clean_geo", "parent_region", "island_group", "obs_date", "headline_yoy_pct", "target_yoy_t1"],
        ].copy(),
    }


def init_state(regs_all: list[str]):
    defaults = {
        "chapter_idx": 0,
        "scroll_to_top": False,
        "sim_prov": "City of Baguio",
        "sim_rice": 8.0,
        "sim_util": 5.0,
        "sim_fuel": 6.0,
        "flt_islands": ["Luzon", "Visayas", "Mindanao"],
        "flt_regs": regs_all,
        "flt_years": (2019, 2026),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def go_to_section(idx: int, target_prov: str | None = None):
    bounded = max(0, min(len(SECTIONS) - 1, idx))
    st.session_state["chapter_idx"] = bounded
    st.session_state["scroll_to_top"] = True
    if target_prov:
        st.session_state["sim_prov"] = target_prov


def apply_shock_preset(rice: float, util: float, fuel: float):
    st.session_state["sim_rice"] = rice
    st.session_state["sim_util"] = util
    st.session_state["sim_fuel"] = fuel


def reset_global_filters(regs_all: list[str]):
    st.session_state["flt_islands"] = ["Luzon", "Visayas", "Mindanao"]
    st.session_state["flt_regs"] = regs_all
    st.session_state["flt_years"] = (2019, 2026)


def compute_province_shock(
    con: duckdb.DuckDBPyConnection,
    eng: dict,
    prov_name: str,
    rice_shock: float,
    util_shock: float,
    fuel_shock: float,
) -> dict:
    safe_prov = prov_name.replace("'", "''")
    prov_hist = con.execute(f"""
        SELECT * FROM prov_panel
        WHERE clean_geo = '{safe_prov}' AND headline_yoy_pct IS NOT NULL
        ORDER BY obs_date DESC LIMIT 18
    """).df().sort_values("obs_date")
    prov_hist["obs_date"] = pd.to_datetime(prov_hist["obs_date"])
    latest_row = prov_hist.iloc[-1:].copy()

    X_cols = eng["num_feats"] + eng["cat_feats"]
    X_base = latest_row[X_cols].copy()
    X_sim = X_base.copy()

    w_rice = float(latest_row["rice_weight"].iloc[0] or 9.0) / 100.0
    w_util = float(latest_row["elec_gas_fuel_weight"].iloc[0] or 4.5) / 100.0
    w_fuel = float(latest_row["personal_fuel_weight"].iloc[0] or 2.5) / 100.0
    w_food = max(float(latest_row["food_bev_weight"].iloc[0] or 38.0) / 100.0, 0.1)
    b30_rice_w = float(latest_row["b30_rice_weight"].iloc[0] or (w_rice * 180.0)) / 100.0

    direct_acct_pp = w_rice * rice_shock + w_util * util_shock + w_fuel * fuel_shock
    b30_extra_pp = (b30_rice_w - w_rice) * rice_shock
    b30_direct_pp = direct_acct_pp + b30_extra_pp

    X_sim["rice_cpi_yoy_pct"] = X_sim["rice_cpi_yoy_pct"].fillna(0) + rice_shock
    X_sim["food_bev_yoy_pct"] = X_sim["food_bev_yoy_pct"].fillna(0) + rice_shock * (w_rice / w_food)
    X_sim["elec_gas_fuel_yoy_pct"] = X_sim["elec_gas_fuel_yoy_pct"].fillna(0) + util_shock
    X_sim["personal_fuel_yoy_pct"] = X_sim["personal_fuel_yoy_pct"].fillna(0) + fuel_shock
    X_sim["transport_yoy_pct"] = X_sim["transport_yoy_pct"].fillna(0) + fuel_shock * 0.45
    X_sim["headline_yoy_pct"] = X_sim["headline_yoy_pct"].fillna(0) + direct_acct_pp
    X_sim["b30_inflation_spread_pp"] = X_sim["b30_inflation_spread_pp"].fillna(0) + b30_extra_pp

    base_pred = float(eng["ridge"].predict(X_base)[0])
    sim_pred = float(eng["ridge"].predict(X_sim)[0])
    b30_base_pred = float(eng["b30_ridge"].predict(X_base)[0])
    b30_sim_pred = float(eng["b30_ridge"].predict(X_sim)[0])

    X_sim_enc = eng["ridge"].named_steps["pre"].transform(X_sim)
    sim_q10 = float(eng["q10"].predict(X_sim_enc)[0])
    sim_q90 = float(eng["q90"].predict(X_sim_enc)[0])
    base_prob = float(eng["clf"].predict_proba(X_base)[0, 1])
    sim_prob = float(eng["clf"].predict_proba(X_sim)[0, 1])

    return {
        "prov_hist": prov_hist,
        "latest_row": latest_row,
        "w_rice_pct": w_rice * 100.0,
        "b30_rice_w_pct": b30_rice_w * 100.0,
        "direct_acct_pp": direct_acct_pp,
        "b30_extra_pp": b30_extra_pp,
        "b30_direct_pp": b30_direct_pp,
        "base_pred": base_pred,
        "sim_pred": sim_pred,
        "b30_base_pred": b30_base_pred,
        "b30_sim_pred": b30_sim_pred,
        "sim_q10": sim_q10,
        "sim_q90": sim_q90,
        "base_prob": base_prob,
        "sim_prob": sim_prob,
    }


def render_section_footer(
    step_num: int,
    summary_line: str,
    prev_idx: int | None = None,
    prev_label: str | None = None,
    next_idx: int | None = None,
    next_label: str | None = None,
    next_primary: bool = True,
):
    st.markdown("")
    with st.container(border=True):
        st.caption(summary_line)
        b_left, b_spacer, b_right = st.columns([1.2, 1.2, 1.6])
        with b_left:
            if prev_idx is not None and prev_label:
                st.button(
                    prev_label,
                    key=f"btn_prev_{step_num}",
                    on_click=go_to_section,
                    args=(prev_idx,),
                    width="stretch",
                )
        with b_right:
            if next_idx is not None and next_label:
                st.button(
                    next_label,
                    key=f"btn_next_{step_num}",
                    type="primary" if next_primary else "secondary",
                    on_click=go_to_section,
                    args=(next_idx,),
                    width="stretch",
                )


def main():
    con = get_duckdb()
    eng = train_interactive_models()
    summary = json.loads(SUMMARY_JSON.read_text(encoding="utf-8")) if SUMMARY_JSON.exists() else {}

    regs_all = con.execute("SELECT DISTINCT parent_region FROM prov_panel ORDER BY 1").df()["parent_region"].tolist()
    init_state(regs_all)

    if st.session_state.get("scroll_to_top"):
        components.html(
            """
            <script>
            const mainSec = window.parent.document.querySelector('section.main');
            if (mainSec) { mainSec.scrollTo({top: 0, behavior: 'smooth'}); }
            window.parent.scrollTo({top: 0, behavior: 'smooth'});
            </script>
            """,
            height=0,
        )
        st.session_state["scroll_to_top"] = False

    # ------------------------------------------------------------------
    # Sidebar: Geographic & Forecast Controls
    # ------------------------------------------------------------------
    st.sidebar.header("Monitoring Scope")
    st.sidebar.caption("Filter provinces and years. Every view aggregates live inside DuckDB.")

    islands = ["Luzon", "Visayas", "Mindanao"]
    sel_islands = st.sidebar.multiselect(
        "Island Cluster",
        islands,
        key="flt_islands",
        help="Filter provincial monitoring across Luzon, Visayas, and Mindanao.",
    )
    sel_regs = st.sidebar.multiselect(
        "Regions",
        regs_all,
        key="flt_regs",
        help="Focus on specific regions across all 18 official PSA administrative regions.",
    )
    yr_range = st.sidebar.slider(
        "Observation Years",
        2019,
        2026,
        key="flt_years",
        help="Year-on-year monthly inflation from Jan 2019 through Oct 2026 (2018=100 base).",
    )

    if len(sel_islands) < 3 or len(sel_regs) < len(regs_all) or yr_range != (2019, 2026):
        st.sidebar.button(
            "Reset Filters",
            on_click=reset_global_filters,
            args=(regs_all,),
            width="stretch",
        )

    st.sidebar.divider()
    st.sidebar.subheader("Alert & Horizon Controls")
    tau = st.sidebar.slider(
        ">4.0% Price-Alert Cutoff (τ)",
        0.15,
        0.85,
        0.45,
        0.02,
        help="Probability threshold for triggering an LPCC price-monitoring alert. Lower values catch more surges early.",
    )
    horizon_mo = st.sidebar.slider(
        "SARIMAX Horizon (Months)",
        3,
        12,
        6,
        1,
        help="Number of months projected ahead in the regional uncertainty fan chart.",
    )

    isl_sql = ", ".join([f"'{x}'" for x in (sel_islands or islands)])
    reg_sql = ", ".join([f"'{x}'" for x in (sel_regs or regs_all)])
    where_sql = (
        f"WHERE headline_yoy_pct IS NOT NULL "
        f"AND island_group IN ({isl_sql}) "
        f"AND parent_region IN ({reg_sql}) "
        f"AND year BETWEEN {yr_range[0]} AND {yr_range[1]}"
    )

    # ------------------------------------------------------------------
    # Header & Glossary
    # ------------------------------------------------------------------
    h_left, h_right = st.columns([3.4, 1.0])
    with h_left:
        st.title("PesoPulse: Local Price Council Shock & Inflation Monitor")
        st.markdown(
            "Built for **DTI and Department of Agriculture Local Price Coordinating Councils** across `101` "
            "provinces and cities to estimate how sudden jumps in **rice, electricity, and diesel** hit ordinary "
            "and **Bottom-30% low-income households** next month."
        )
    with h_right:
        st.markdown("")
        with st.popover("Glossary", width="stretch"):
            st.markdown(
                """
                **Key Indicators**
                - **LPCC:** Local Price Coordinating Council under DTI, DA, and LGUs monitoring retail staples and SRP compliance.
                - **CPI (`2018=100`) & YoY (`%`):** Consumer Price Index and percentage price change versus the same month one year ago.
                - **Percentage Points (`pp`):** Direct arithmetic difference between two inflation rates.
                - **Bottom-30% Households:** Lowest-income `30%` of families, whose budgets spend far more on rice and basic food.
                - **`4.0%` Price-Alert Line:** Official national upper inflation benchmark used as a local council intervention trigger.
                - **TreeSHAP & Fan Charts:** Driver attribution showing what moves next-month forecasts, paired with `80%` and `95%` intervals.
                """
            )
        st.caption(
            f"`{len(sel_islands or islands)}` island groups · "
            f"`{len(sel_regs or regs_all)}` regions · `{yr_range[0]}–{yr_range[1]}`"
        )

    # ------------------------------------------------------------------
    # Focused 4-KPI Strip
    # ------------------------------------------------------------------
    kpi_sql = f"""
        SELECT
            COUNT(*) AS n_obs,
            COUNT(DISTINCT clean_geo) AS n_geos,
            ROUND(AVG(headline_yoy_pct), 2) AS mean_yoy,
            ROUND(AVG(b30_headline_yoy_pct), 2) AS mean_b30,
            ROUND(AVG(is_breach_bsp_4pct) * 100.0, 1) AS breach_pct
        FROM prov_panel {where_sql}
    """
    kpi = con.execute(kpi_sql).df().iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Monitored Localities",
        f"{int(kpi['n_geos'])}",
        f"{int(kpi['n_obs']):,} province-months",
        border=True,
        help="Subnational leaf units monitored across 100 provinces and highly urbanized cities plus NCR.",
    )
    c2.metric(
        "All-Income Mean Inflation",
        f"{kpi['mean_yoy']:.2f}%",
        "Ordinary Household Basket",
        border=True,
        help="Average year-on-year CPI inflation across all income households in the active selection.",
    )
    c3.metric(
        "Bottom-30% Mean Inflation",
        f"{kpi['mean_b30']:.2f}%",
        f"{(kpi['mean_b30'] - kpi['mean_yoy']):+.2f} pp vs Ordinary",
        delta_color="inverse",
        border=True,
        help="Average inflation faced by the lowest-income 30% of families; positive deltas indicate regressive price pressure.",
    )
    c4.metric(
        ">4.0% Alert Frequency",
        f"{kpi['breach_pct']:.1f}%",
        "Months Above National Ceiling",
        delta_color="off",
        border=True,
        help="Share of province-months exceeding the 4.0% official upper price-stability benchmark.",
    )

    # ------------------------------------------------------------------
    # Connected 4-Section Navigation Bar
    # ------------------------------------------------------------------
    active_idx = st.session_state["chapter_idx"]
    nav_cols = st.columns(len(SECTIONS))
    for i, sec_label in enumerate(SECTIONS):
        with nav_cols[i]:
            st.button(
                sec_label,
                key=f"top_chap_{i}",
                type="primary" if i == active_idx else "secondary",
                on_click=go_to_section,
                args=(i,),
                width="stretch",
            )

    audit_queries: list[tuple[str, int, str]] = [("LPCC KPI Summary", 1, kpi_sql)]

    # ==================================================================
    # SECTION 1: HOUSEHOLD PRICE EXPOSURE (5 ESSENTIAL BASIC CHARTS)
    # ==================================================================
    if active_idx == 0:
        st.subheader("Why Rice & Staple Surges Hit Low-Income Families Hardest")
        st.caption(
            "National averages dipped to 0.9% in mid-2025 before surging back to 7.8% in October 2026, yet "
            "provincial food and rice swings consistently push low-income inflation above the general rate."
        )

        # Basic 1: Line Chart
        ts_sql = f"""
            SELECT
                obs_date,
                ROUND(AVG(headline_yoy_pct), 2) AS Headline_YoY,
                ROUND(AVG(b30_headline_yoy_pct), 2) AS Bottom30_YoY,
                ROUND(AVG(rice_cpi_yoy_pct), 2) AS Rice_YoY
            FROM prov_panel {where_sql}
            GROUP BY obs_date ORDER BY obs_date
        """
        ts_df = con.execute(ts_sql).df()
        audit_queries.append(("Monthly Household & Rice Timeline", len(ts_df), ts_sql))

        fig_ts = go.Figure()
        fig_ts.add_hrect(
            y0=2.0, y1=4.0,
            fillcolor="rgba(5,150,105,0.12)", line_width=0,
            annotation_text="2.0%–4.0% Stability Band",
            annotation_position="top left",
        )
        for col, label, color, dash in [
            ("Headline_YoY", "Ordinary Households (All-Income)", "#0F172A", "solid"),
            ("Bottom30_YoY", "Bottom-30% Low-Income Households", "#DC2626", "dash"),
            ("Rice_YoY", "Provincial Rice CPI", "#D97706", "solid"),
        ]:
            fig_ts.add_trace(go.Scatter(
                x=ts_df["obs_date"], y=ts_df[col], mode="lines", name=label,
                line=dict(color=color, width=2.4, dash=dash),
            ))
        fig_ts.update_layout(
            title="Monthly Ordinary vs. Bottom-30% Inflation & Rice Price Waves",
            height=370, margin=dict(t=45, b=20), yaxis_title="Year-on-Year Inflation (%)", hovermode="x unified",
        )
        st.plotly_chart(fig_ts, width="stretch")
        st.caption(
            "Amber rice surges above `20%` pull the red dashed Bottom-30% curve well above ordinary household inflation."
        )

        # Basic 2 & 3: Histogram + Regional Bar
        col_a, col_b = st.columns(2)
        with col_a:
            dist_sql = f"""
                SELECT ROUND(headline_yoy_pct, 1) AS bin_yoy, COUNT(*) AS cnt
                FROM prov_panel {where_sql}
                GROUP BY 1 ORDER BY 1
            """
            dist_df = con.execute(dist_sql).df()
            audit_queries.append(("Provincial Distribution Histogram", len(dist_df), dist_sql))
            tukey_hi = float(summary.get("eda", {}).get("tukey_upper_fence", 11.25))
            fig_hist = px.bar(
                dist_df, x="bin_yoy", y="cnt",
                title="Provincial Inflation Distribution & Shock Outliers",
                labels={"bin_yoy": "Headline YoY Inflation (%)", "cnt": "Province-Month Count"},
                color_discrete_sequence=["#1D4ED8"],
            )
            fig_hist.add_vline(x=4.0, line_dash="dash", line_color="#DC2626", annotation_text="4% Alert")
            fig_hist.add_vline(
                x=tukey_hi, line_dash="dot", line_color="#D97706",
                annotation_text=f"Tukey Fence ({tukey_hi}%)",
            )
            fig_hist.update_layout(height=355, margin=dict(t=45, b=20))
            st.plotly_chart(fig_hist, width="stretch")
            st.caption(
                f"Right-skewed distribution (`Skew = 0.50`, `Kurtosis = 3.25`); readings above `{tukey_hi}%` capture "
                "severe localized supply shocks."
            )

        with col_b:
            reg_sql_q = f"""
                SELECT parent_region,
                       ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY headline_yoy_pct), 2) AS median_yoy,
                       ROUND(AVG(b30_inflation_spread_pp), 2) AS avg_b30_spread
                FROM prov_panel {where_sql}
                GROUP BY parent_region ORDER BY median_yoy DESC
            """
            reg_box = con.execute(reg_sql_q).df()
            audit_queries.append(("Regional Median & Bottom-30% Gap", len(reg_box), reg_sql_q))
            fig_bar = px.bar(
                reg_box, x="median_yoy", y="parent_region", color="avg_b30_spread", orientation="h",
                color_continuous_scale="Viridis",
                title="Regional Median Inflation Shaded by Bottom-30% Penalty",
                labels={"median_yoy": "Median YoY (%)", "parent_region": "", "avg_b30_spread": "B30 Gap (pp)"},
            )
            fig_bar.add_vline(x=4.0, line_dash="dash", line_color="#DC2626")
            fig_bar.update_layout(height=355, margin=dict(t=45, b=20))
            st.plotly_chart(fig_bar, width="stretch")
            st.caption("Bicol and MIMAROPA run hotter than NCR, with brighter bars marking steeper low-income pressure.")

        # Basic 4 & 5: Rice Scatter + Basket Share Donuts
        col_c, col_d = st.columns(2)
        with col_c:
            scat_sql = f"""
                SELECT clean_geo, island_group,
                       ROUND(AVG(retail_rice_wm_yoy_pct), 2) AS rice_retail_yoy,
                       ROUND(AVG(headline_yoy_pct), 2) AS headline_yoy,
                       ROUND(AVG(rice_weight), 1) AS rice_wt_pct
                FROM prov_panel {where_sql}
                  AND retail_rice_wm_yoy_pct IS NOT NULL
                GROUP BY 1, 2
            """
            scat_df = con.execute(scat_sql).df()
            audit_queries.append(("Retail Rice vs Headline Scatter", len(scat_df), scat_sql))
            fig_scat = px.scatter(
                scat_df, x="rice_retail_yoy", y="headline_yoy", color="island_group",
                size="rice_wt_pct", hover_name="clean_geo",
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Local Retail Rice Growth vs. Headline Inflation (101 Units)",
                labels={
                    "rice_retail_yoy": "Mean Retail Rice YoY (%)",
                    "headline_yoy": "Mean Headline YoY (%)",
                    "island_group": "Island",
                    "rice_wt_pct": "Rice Weight (%)",
                },
            )
            if len(scat_df) >= 3:
                x_vals = scat_df["rice_retail_yoy"].to_numpy(dtype=float)
                y_vals = scat_df["headline_yoy"].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 40)
                fig_scat.add_trace(go.Scatter(
                    x=x_line, y=slope * x_line + intercept, mode="lines",
                    name="OLS Fit (r = 0.45)", line=dict(color="#0F172A", width=2, dash="dash"),
                ))
            fig_scat.update_layout(height=355, margin=dict(t=45, b=20))
            st.plotly_chart(fig_scat, width="stretch")
            st.caption("Faster local retail rice inflation translates directly into higher provincial CPI (`r = +0.45`).")

        with col_d:
            bsk_sql = f"""
                SELECT
                    ROUND(AVG(rice_weight), 1) AS gen_rice,
                    ROUND(AVG(GREATEST(food_bev_weight - rice_weight, 0)), 1) AS gen_other_food,
                    ROUND(100.0 - AVG(food_bev_weight), 1) AS gen_non_food,
                    ROUND(AVG(b30_rice_weight), 1) AS b30_rice,
                    ROUND(AVG(GREATEST(b30_food_bev_weight - b30_rice_weight, 0)), 1) AS b30_other_food,
                    ROUND(100.0 - AVG(b30_food_bev_weight), 1) AS b30_non_food
                FROM prov_panel {where_sql}
                  AND food_bev_weight IS NOT NULL
                  AND b30_food_bev_weight IS NOT NULL
            """
            bsk = con.execute(bsk_sql).df().iloc[0]
            audit_queries.append(("Household Basket Weights", 1, bsk_sql))
            labels_b = ["Rice", "Other Food", "Power, Fuel & Other"]
            colors_b = ["#D97706", "#0D9488", "#334155"]
            fig_pie = make_subplots(
                rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                subplot_titles=["Ordinary Basket", "Bottom-30% Basket"],
            )
            fig_pie.add_trace(
                go.Pie(
                    labels=labels_b,
                    values=[bsk["gen_rice"], bsk["gen_other_food"], bsk["gen_non_food"]],
                    hole=0.48, marker=dict(colors=colors_b), textinfo="percent",
                ),
                row=1, col=1,
            )
            fig_pie.add_trace(
                go.Pie(
                    labels=labels_b,
                    values=[bsk["b30_rice"], bsk["b30_other_food"], bsk["b30_non_food"]],
                    hole=0.48, marker=dict(colors=colors_b), textinfo="percent",
                ),
                row=1, col=2,
            )
            fig_pie.update_layout(
                title="Budget Shares: Ordinary vs. Bottom-30% Households",
                height=355, margin=dict(t=55, b=20), legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig_pie, width="stretch")
            st.caption(
                f"Low-income budgets devote `{bsk['b30_rice']:.1f}%` to rice alone versus `{bsk['gen_rice']:.1f}%` "
                "for ordinary households, magnifying every staple price spike."
            )

        # Compact 3-Card Statistical Verification
        if summary:
            ht = summary.get("hypothesis_tests", {})
            s1, s2, s3 = st.columns(3)
            s1.metric(
                "Regional Divergence (ANOVA)",
                f"F = {ht.get('H1_regional_anova_F')}",
                "p < 0.001 across 18 regions",
                border=True,
                help="One-Way ANOVA confirms local inflation rates differ significantly across Philippine regions.",
            )
            s2.metric(
                "Low-Income Penalty (Food >5%)",
                f"+{ht.get('H2_high_food_b30_diff_pp')} pp",
                f"Paired t = {ht.get('H2_high_food_paired_t')} (6x normal gap)",
                delta_color="inverse",
                border=True,
                help="When food inflation exceeds 5%, Bottom-30% households pay +0.95 pp higher inflation than ordinary families.",
            )
            s3.metric(
                "Upstream Rice & Wholesale Link",
                f"r = +{ht.get('H3_rice_yoy_pearson_r')}",
                f"Lagged GWPI r = +{ht.get('H3_gwpi_lag1_pearson_r')}",
                border=True,
                help="Local retail rice growth and prior-month wholesale prices strongly lead provincial headline CPI.",
            )

        render_section_footer(
            step_num=1,
            summary_line=(
                "Rice and staple swings widen the low-income inflation penalty nearly sixfold across diverging regions. "
                "Next, pinpoint which specific provinces and supply channels carry the heaviest local burden."
            ),
            next_idx=1,
            next_label="Provincial Hotspots & Flows →",
        )

    # ==================================================================
    # SECTION 2: PROVINCIAL HOTSPOTS & FLOWS (5 ESSENTIAL ADVANCED CHARTS)
    # ==================================================================
    elif active_idx == 1:
        st.subheader("Where Local Price Pressure Clusters & How Shocks Travel")
        st.caption(
            "Pinpoint high-breach provinces and trace how food, electricity, and transport fuel transmit through "
            "local household baskets across 101 localities."
        )

        geo_sql = f"""
            SELECT
                clean_geo, parent_region, island_group, latitude, longitude,
                ROUND(AVG(headline_yoy_pct), 2) AS mean_yoy,
                ROUND(AVG(b30_inflation_spread_pp), 2) AS mean_b30_spread,
                ROUND(AVG(is_breach_bsp_4pct) * 100.0, 1) AS breach_freq_pct,
                ROUND(AVG(rice_cpi_yoy_pct), 2) AS mean_rice_yoy,
                ROUND(AVG(elec_gas_fuel_yoy_pct), 2) AS mean_util_yoy,
                ROUND(AVG(transport_yoy_pct), 2) AS mean_trans_yoy
            FROM prov_panel {where_sql}
            GROUP BY 1, 2, 3, 4, 5
            HAVING latitude IS NOT NULL
        """
        geo_agg = con.execute(geo_sql).df()
        audit_queries.append(("Provincial Map & Multi-Driver Profiles", len(geo_agg), geo_sql))

        c_map, c_sun = st.columns(2)
        with c_map:
            fig_geo = px.scatter_map(
                geo_agg, lat="latitude", lon="longitude",
                color="mean_yoy", size="breach_freq_pct",
                hover_name="clean_geo",
                hover_data={
                    "parent_region": True,
                    "mean_yoy": ":.2f",
                    "breach_freq_pct": ":.1f",
                    "mean_b30_spread": ":+.2f",
                    "latitude": False,
                    "longitude": False,
                },
                color_continuous_scale="Viridis", zoom=4.6, center={"lat": 12.8, "lon": 122.2},
                map_style="carto-positron",
                title="Provincial Inflation Intensity & >4.0% Alert Frequency",
                labels={"mean_yoy": "Mean YoY (%)", "breach_freq_pct": "Alert %"},
            )
            fig_geo.update_layout(height=450, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig_geo, width="stretch")
            st.caption(
                "Outer-island markets such as Batanes (`5.25%`), Dinagat Islands (`5.07%`), and Occidental Mindoro "
                "cross the `4.0%` alert line most frequently."
            )

        with c_sun:
            sun_sql = f"""
                SELECT
                    island_group, parent_region, coicop_name,
                    ROUND(AVG(ABS(weighted_pp_contrib)) + 0.01, 3) AS abs_pp_contrib,
                    ROUND(AVG(official_yoy_pct), 2) AS avg_yoy
                FROM div_monthly
                WHERE is_subnational_leaf = TRUE
                  AND island_group IN ({isl_sql})
                  AND parent_region IN ({reg_sql})
                  AND year BETWEEN {yr_range[0]} AND {yr_range[1]}
                  AND coicop_code IN ('01','02','03','04','05','06','07','08','09','10','11','12','13')
                  AND weighted_pp_contrib IS NOT NULL
                GROUP BY 1, 2, 3
            """
            sun_df = con.execute(sun_sql).df()
            audit_queries.append(("Regional Spending Hierarchy", len(sun_df), sun_sql))
            fig_sun = px.sunburst(
                sun_df, path=["island_group", "parent_region", "coicop_name"],
                values="abs_pp_contrib", color="avg_yoy", color_continuous_scale="Viridis",
                title="Island, Region & Commodity Contribution Hierarchy",
                labels={"avg_yoy": "Category YoY (%)"},
            )
            fig_sun.update_layout(height=450, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig_sun, width="stretch")
            st.caption("Select any island or region ring to inspect which basket divisions drive local inflation.")

        c_san, c_par = st.columns(2)
        with c_san:
            san_sql = f"""
                SELECT
                    CASE
                        WHEN coicop_code IN ('01') THEN 'Rice & Food Supply'
                        WHEN coicop_code IN ('04','07') THEN 'Electricity, LPG & Diesel'
                        ELSE 'Other Goods & Services'
                    END AS driver_channel,
                    coicop_name,
                    island_group,
                    ROUND(SUM(GREATEST(weighted_pp_contrib, 0.01)), 2) AS flow_pp
                FROM div_monthly
                WHERE is_subnational_leaf = TRUE
                  AND island_group IN ({isl_sql})
                  AND parent_region IN ({reg_sql})
                  AND year BETWEEN {yr_range[0]} AND {yr_range[1]}
                  AND coicop_code IN ('01','04','07','11','02','06','10')
                  AND weighted_pp_contrib IS NOT NULL
                GROUP BY 1, 2, 3
            """
            san_df = con.execute(san_sql).df()
            audit_queries.append(("Cost-Channel Sankey Transmission", len(san_df), san_sql))
            nodes = list(dict.fromkeys(
                san_df["driver_channel"].tolist() + san_df["coicop_name"].tolist() + san_df["island_group"].tolist()
            ))
            idx_map = {n: i for i, n in enumerate(nodes)}
            l1 = san_df.groupby(["driver_channel", "coicop_name"])["flow_pp"].sum().reset_index()
            l2 = san_df.groupby(["coicop_name", "island_group"])["flow_pp"].sum().reset_index()
            fig_san = go.Figure(go.Sankey(
                node=dict(pad=14, thickness=16, label=nodes, color="#1E293B"),
                link=dict(
                    source=[idx_map[x] for x in l1["driver_channel"]] + [idx_map[x] for x in l2["coicop_name"]],
                    target=[idx_map[x] for x in l1["coicop_name"]] + [idx_map[x] for x in l2["island_group"]],
                    value=l1["flow_pp"].tolist() + l2["flow_pp"].tolist(),
                ),
            ))
            fig_san.update_layout(
                title="Supply-Channel Flow Into Household Baskets & Islands",
                height=420, margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_san, width="stretch")
            st.caption(
                "Food and rice (`48%`) together with electricity, LPG, and transport fuel (`28%`) generate over "
                "three-quarters of household inflation pressure."
            )

        with c_par:
            fig_par = px.parallel_coordinates(
                geo_agg,
                dimensions=["mean_yoy", "mean_rice_yoy", "mean_util_yoy", "mean_trans_yoy", "mean_b30_spread"],
                color="mean_yoy", color_continuous_scale="Viridis",
                labels={
                    "mean_yoy": "Headline %",
                    "mean_rice_yoy": "Rice %",
                    "mean_util_yoy": "Power/LPG %",
                    "mean_trans_yoy": "Transport %",
                    "mean_b30_spread": "B30 Gap (pp)",
                },
                title="Compound Shock Profile Across 101 Localities",
            )
            fig_par.update_layout(height=420, margin=dict(t=45, b=10))
            st.plotly_chart(fig_par, width="stretch")
            st.caption("Drag vertically along any axis to isolate provinces facing simultaneous rice, utility, and fuel pressure.")

        st_sql = f"""
            SELECT obs_date, coicop_name, ROUND(AVG(weighted_pp_contrib), 3) AS avg_contrib_pp
            FROM div_monthly
            WHERE is_subnational_leaf = TRUE
              AND island_group IN ({isl_sql})
              AND parent_region IN ({reg_sql})
              AND year BETWEEN {yr_range[0]} AND {yr_range[1]}
              AND coicop_code IN ('01','04','07','11','02','06')
              AND weighted_pp_contrib IS NOT NULL
            GROUP BY 1, 2 ORDER BY 1, 2
        """
        st_df = con.execute(st_sql).df()
        audit_queries.append(("Category Contribution Stream", len(st_df), st_sql))
        fig_stream = px.area(
            st_df, x="obs_date", y="avg_contrib_pp", color="coicop_name",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Weighted Inflation Contributions by Spending Division (2019–2026)",
            labels={"avg_contrib_pp": "Weighted Contribution (pp)", "obs_date": "", "coicop_name": "Division"},
        )
        fig_stream.update_layout(height=345, margin=dict(t=40, b=10), hovermode="x unified")
        st.plotly_chart(fig_stream, width="stretch")
        st.caption("Food dominated the 2023 spike, cooled during mid-2025, and surged again alongside transport in 2026.")

        render_section_footer(
            step_num=2,
            summary_line=(
                "Food, utilities, and transport account for over three-quarters of local price pressure, clustering "
                "heavily in outer islands. Next, review our blind 2025–2026 forecasts and early alert engine."
            ),
            prev_idx=0,
            prev_label="← Household Price Exposure",
            next_idx=2,
            next_label="Next-Month Forecasts & Alerts →",
        )

    # ==================================================================
    # SECTION 3: NEXT-MONTH FORECASTS & ALERTS (FOCUSED PREDICTIVE & XAI)
    # ==================================================================
    elif active_idx == 2:
        st.subheader("Blind 2025–2026 Forecasts, Drivers & Early Price Alerts")
        st.caption(
            "Models train strictly on 2020–2024 and are tested blind on unseen 2025–2026 provincial months so "
            "councils see honest accuracy across sharp price reversals."
        )

        # Compact 3-Metric Out-of-Time Benchmark Strip
        b1, b2, b3 = st.columns(3)
        b1.metric(
            "Ordinary CPI Forecast (Ridge)",
            f"{eng['ridge_mae']:.2f} pp MAE",
            f"Holdout R² = {eng['ridge_r2']:.3f}",
            border=True,
            help="Best out-of-time regression model across 2025–2026; linear shrinkage handles regime swings better than tree leaves.",
        )
        b2.metric(
            "Bottom-30% Forecast (Ridge)",
            f"{eng['b30_mae']:.2f} pp MAE",
            f"Holdout R² = {eng['b30_r2']:.3f}",
            border=True,
            help="Dedicated Ridge model trained directly on next-month Bottom-30% household inflation.",
        )
        b3.metric(
            ">4.0% Alert Classifier (XGBoost)",
            f"{eng['clf_acc']:.1%} Accuracy",
            f"ROC-AUC = {eng['clf_auc']:.3f}",
            border=True,
            help="Flags provinces likely to cross the 4.0% upper price-stability line one month in advance.",
        )

        # 1. Dynamic SARIMAX Fan Chart
        sel_fc_geo = st.selectbox(
            "Regional Series for Multi-Month Uncertainty Fan Chart",
            ["PHILIPPINES"] + regs_all,
            help="Fits SARIMAX(1,0,1)(1,0,1,12) with 80% and 95% confidence bands; adjust horizon in the sidebar.",
        )
        fan_sql = f"""
            SELECT obs_date, headline_yoy_pct
            FROM reg_monthly
            WHERE region = '{sel_fc_geo}' AND headline_yoy_pct IS NOT NULL
            ORDER BY obs_date
        """
        geo_ts = con.execute(fan_sql).df()
        audit_queries.append(("Regional SARIMAX Series", len(geo_ts), fan_sql))
        geo_ts["obs_date"] = pd.to_datetime(geo_ts["obs_date"])
        s_series = geo_ts.set_index("obs_date")["headline_yoy_pct"].asfreq("MS")
        sar_fit = SARIMAX(
            s_series, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12),
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        fc = sar_fit.get_forecast(steps=horizon_mo)
        ci80 = fc.conf_int(alpha=0.20)
        ci95 = fc.conf_int(alpha=0.05)
        fwd_dates = pd.date_range(s_series.index.max() + pd.DateOffset(months=1), periods=horizon_mo, freq="MS")

        fig_fan = go.Figure()
        fig_fan.add_hrect(
            y0=2.0, y1=4.0, fillcolor="rgba(5,150,105,0.11)", line_width=0,
            annotation_text="2%–4% Stability Band", annotation_position="top left",
        )
        fig_fan.add_trace(go.Scatter(
            x=s_series.index[-36:], y=s_series.values[-36:], mode="lines",
            name="Observed YoY (%)", line=dict(color="#0F172A", width=2.4),
        ))
        fig_fan.add_trace(go.Scatter(
            x=list(fwd_dates) + list(fwd_dates[::-1]),
            y=list(ci95.iloc[:, 1]) + list(ci95.iloc[:, 0][::-1]),
            fill="toself", fillcolor="rgba(217,119,6,0.18)", line=dict(width=0),
            name="95% Prediction Band",
        ))
        fig_fan.add_trace(go.Scatter(
            x=list(fwd_dates) + list(fwd_dates[::-1]),
            y=list(ci80.iloc[:, 1]) + list(ci80.iloc[:, 0][::-1]),
            fill="toself", fillcolor="rgba(217,119,6,0.35)", line=dict(width=0),
            name="80% Prediction Band",
        ))
        fig_fan.add_trace(go.Scatter(
            x=fwd_dates, y=fc.predicted_mean.values, mode="lines+markers",
            name=f"{horizon_mo}-Mo Forecast", line=dict(color="#EA580C", width=2.5),
        ))
        fig_fan.update_layout(
            height=360,
            title=f"Multi-Month Forecast & Uncertainty Bands: {sel_fc_geo} ({horizon_mo} Months)",
            yaxis_title="Year-on-Year Inflation (%)", margin=dict(t=45, b=20), hovermode="x unified",
        )
        st.plotly_chart(fig_fan, width="stretch")

        # 2. XAI TreeSHAP + Interactive ROC Alert Cutoff
        c_xai, c_roc = st.columns(2)
        with c_xai:
            if SHAP_CSV.exists():
                sh_df = pd.read_csv(SHAP_CSV).head(8).iloc[::-1].copy()
                sh_df["readable_feature"] = sh_df["feature"].map(humanize_feature)
                fig_sh = px.bar(
                    sh_df, x="mean_abs_shap", y="readable_feature", orientation="h",
                    hover_data={"feature": True, "mean_abs_shap": ":.3f", "readable_feature": False},
                    title="Top 8 Forecast Drivers (TreeSHAP)",
                    color_discrete_sequence=["#0D9488"],
                )
                fig_sh.update_layout(
                    height=355, yaxis_title="", xaxis_title="Mean |SHAP| Impact (pp)",
                    margin=dict(t=45, b=20),
                )
                st.plotly_chart(fig_sh, width="stretch")

        with c_roc:
            y_true_c = eng["y_te_clf"]
            probs_c = eng["probs_te"]
            pred_c = (probs_c >= tau).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true_c, pred_c).ravel()
            prec_v = tp / max(tp + fp, 1)
            rec_v = tp / max(tp + fn, 1)
            fpr, tpr, thresh_arr = roc_curve(y_true_c, probs_c)
            idx_tau = int(np.argmin(np.abs(thresh_arr - tau)))

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"XGBoost Alert (AUC = {eng['clf_auc']:.3f})", line=dict(color="#1D4ED8", width=2.6),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[fpr[idx_tau]], y=[tpr[idx_tau]], mode="markers",
                name=f"Active Cutoff (τ = {tau:.2f})",
                marker=dict(color="#DC2626", size=12, symbol="star"),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random",
                line=dict(dash="dash", color="#64748B"),
            ))
            fig_roc.update_layout(
                height=295, title=">4.0% Early-Alert ROC Curve",
                xaxis_title="False Alarm Rate", yaxis_title="Caught Surge Rate",
                margin=dict(t=45, b=20),
            )
            st.plotly_chart(fig_roc, width="stretch")
            st.caption(
                f"**At τ = `{tau:.2f}`:** Caught `{tp}` · False Alarms `{fp}` · Missed `{fn}` · "
                f"Precision `{prec_v:.1%}` · Recall `{rec_v:.1%}`"
            )

        # 3. Cross-Filtered Abrupt Shock Inspector + Direct Simulator Handoff
        st.divider()
        st.markdown("##### Abrupt Supply-Shock Misses (2025–2026 Holdout)")
        err_df = eng["te_meta"].copy()
        err_df["predicted_yoy_t1"] = np.round(eng["preds_te"], 2)
        err_df["abs_error_pp"] = np.round(np.abs(err_df["target_yoy_t1"] - err_df["predicted_yoy_t1"]), 2)
        top_err = err_df.sort_values("abs_error_pp", ascending=False).head(8)

        ce1, ce2 = st.columns([1.05, 1.35])
        with ce1:
            st.dataframe(
                top_err[["clean_geo", "obs_date", "target_yoy_t1", "predicted_yoy_t1", "abs_error_pp"]].rename(
                    columns={
                        "clean_geo": "Province / City",
                        "obs_date": "Month",
                        "target_yoy_t1": "Actual %",
                        "predicted_yoy_t1": "Forecast %",
                        "abs_error_pp": "Shock Miss (pp)",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        with ce2:
            err_geos = list(dict.fromkeys(top_err["clean_geo"].tolist()))
            sel_err_geo = st.selectbox(
                "Inspect Shock-Prone Locality",
                err_geos,
                index=0,
                help="Highlights months where sudden local supply shocks pushed inflation 2.0 pp or more away from momentum.",
            )
            sub_err = err_df[err_df["clean_geo"] == sel_err_geo].sort_values("obs_date")
            big_pts = sub_err[sub_err["abs_error_pp"] >= 2.0]
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(
                x=sub_err["obs_date"], y=sub_err["target_yoy_t1"], mode="lines+markers",
                name="Actual Next-Month YoY", line=dict(color="#0F172A", width=2.2),
            ))
            fig_err.add_trace(go.Scatter(
                x=sub_err["obs_date"], y=sub_err["predicted_yoy_t1"], mode="lines+markers",
                name="Ridge Forecast", line=dict(color="#2563EB", dash="dot", width=2.2),
            ))
            if len(big_pts):
                fig_err.add_trace(go.Scatter(
                    x=big_pts["obs_date"], y=big_pts["target_yoy_t1"], mode="markers",
                    name="Abrupt Shock (>=2 pp)",
                    marker=dict(color="#DC2626", size=11, symbol="diamond"),
                ))
            fig_err.update_layout(
                height=275, title=f"Holdout Trajectory: {sel_err_geo}",
                yaxis_title="YoY Inflation (%)", margin=dict(t=40, b=15),
            )
            st.plotly_chart(fig_err, width="stretch")
            st.button(
                f"Stress-Test {sel_err_geo} in LPCC Simulator →",
                key="btn_handoff_err_prov",
                on_click=go_to_section,
                args=(3, sel_err_geo),
                width="stretch",
            )

        render_section_footer(
            step_num=3,
            summary_line=(
                "Momentum models catch regular trends and >4.0% alerts, while sudden rice, power, and diesel shocks "
                "require proactive local scenario testing before prices hit public markets."
            ),
            prev_idx=1,
            prev_label="← Provincial Hotspots & Flows",
            next_idx=3,
            next_label="LPCC Supply-Shock Simulator →",
        )

    # ==================================================================
    # SECTION 4: LPCC SUPPLY-SHOCK SIMULATOR (ORDINARY VS BOTTOM-30% ML)
    # ==================================================================
    else:
        st.subheader("LPCC Next-Month Shock Simulator: Ordinary vs. Bottom-30% Households")
        st.caption(
            "Estimate how a sudden jump in rice, electricity, or diesel prices passes through local CPI weights into "
            "next-month forecasts for ordinary and low-income families."
        )

        p1, p2, p3, p4 = st.columns(4)
        p1.button(
            "Rice Supply Spike (+12 pp)",
            on_click=apply_shock_preset, args=(12.0, 2.0, 2.0), width="stretch",
            help="Simulates a domestic palay shortfall or import delay pushing retail rice up sharply.",
        )
        p2.button(
            "Power & Diesel Surge",
            on_click=apply_shock_preset, args=(3.0, 9.0, 11.0), width="stretch",
            help="Simulates concurrent electricity/LPG hikes and diesel transport fare pass-through.",
        )
        p3.button(
            "Kadiwa & Tariff Relief",
            on_click=apply_shock_preset, args=(-8.0, -2.0, -4.0), width="stretch",
            help="Simulates NFA/Kadiwa rice buffer releases, tariff cuts, and fuel rollbacks.",
        )
        p4.button(
            "Reset Baseline Shock",
            on_click=apply_shock_preset, args=(8.0, 5.0, 6.0), width="stretch",
            help="Restores the standard scenario (+8 pp Rice, +5 pp Electricity/LPG, +6 pp Diesel).",
        )

        provs = sorted(eng["df"]["clean_geo"].unique().tolist())
        if st.session_state["sim_prov"] not in provs:
            st.session_state["sim_prov"] = "City of Baguio" if "City of Baguio" in provs else provs[0]

        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            sel_prov = st.selectbox(
                "Target Province or City",
                provs,
                key="sim_prov",
                help="Primary locality evaluated with its official PSA All-Income and Bottom-30% basket weights.",
            )
        default_cmp = "Camarines Sur" if sel_prov != "Camarines Sur" and "Camarines Sur" in provs else (
            "City of Baguio" if sel_prov != "City of Baguio" and "City of Baguio" in provs else provs[0]
        )
        with sel_col2:
            cmp_prov = st.selectbox(
                "Comparison Province or City",
                provs,
                index=provs.index(default_cmp),
                help="Contrasts how the exact same price shock impacts a second locality with different basket weights.",
            )

        s1, s2, s3 = st.columns(3)
        rice_shock = s1.slider(
            "Rice Price Shock (Δ pp)", -15.0, 25.0, key="sim_rice", step=0.5,
            help="Projected percentage-point jump or drop in local rice inflation.",
        )
        util_shock = s2.slider(
            "Electricity & LPG Shock (Δ pp)", -15.0, 25.0, key="sim_util", step=0.5,
            help="Projected percentage-point shift in household electricity, gas, and cooking fuel.",
        )
        fuel_shock = s3.slider(
            "Diesel & Pump Fuel Shock (Δ pp)", -15.0, 25.0, key="sim_fuel", step=0.5,
            help="Projected pump fuel shock including 45% spillover into public transport fares.",
        )

        res_main = compute_province_shock(con, eng, sel_prov, rice_shock, util_shock, fuel_shock)
        res_cmp = compute_province_shock(con, eng, cmp_prov, rice_shock, util_shock, fuel_shock)
        audit_queries.append((
            f"Simulator Local Series ({sel_prov} & {cmp_prov})",
            len(res_main["prov_hist"]) + len(res_cmp["prov_hist"]),
            f"SELECT * FROM prov_panel WHERE clean_geo IN ('{sel_prov}', '{cmp_prov}') ORDER BY obs_date DESC LIMIT 18",
        ))

        prov_hist = res_main["prov_hist"]
        latest_row = res_main["latest_row"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Ordinary Household Forecast",
            f"{res_main['sim_pred']:.2f}%",
            f"{res_main['sim_pred'] - res_main['base_pred']:+.2f} pp vs baseline ({res_main['base_pred']:.2f}%)",
            delta_color="inverse",
            border=True,
            help=f"Next-month All-Income Ridge forecast; 80% uncertainty band spans {res_main['sim_q10']:.2f}% to {res_main['sim_q90']:.2f}%.",
        )
        m2.metric(
            "Bottom-30% Household Forecast",
            f"{res_main['b30_sim_pred']:.2f}%",
            f"{res_main['b30_sim_pred'] - res_main['b30_base_pred']:+.2f} pp vs baseline ({res_main['b30_base_pred']:.2f}%)",
            delta_color="inverse",
            border=True,
            help="Dedicated Bottom-30% Ridge ML forecast capturing how low-income families absorb rice, utility, and fuel shocks.",
        )
        m3.metric(
            "Direct Basket Pass-Through",
            f"{res_main['direct_acct_pp']:+.2f} pp",
            f"Bottom-30% Extra Rice Hit: {res_main['b30_extra_pp']:+.2f} pp",
            delta_color="inverse",
            border=True,
            help=(
                f"{sel_prov} rice weights: {res_main['w_rice_pct']:.1f}% ordinary basket vs "
                f"{res_main['b30_rice_w_pct']:.1f}% Bottom-30% basket."
            ),
        )
        m4.metric(
            ">4.0% Alert Probability",
            f"{res_main['sim_prob']:.1%}",
            f"{(res_main['sim_prob'] - res_main['base_prob']) * 100:+.1f} pp vs baseline",
            delta_color="inverse",
            border=True,
            help="XGBoost probability that next month's provincial inflation exceeds the 4.0% alert ceiling.",
        )

        v_left, v_right = st.columns([1.45, 1.0])
        with v_left:
            next_dt = prov_hist["obs_date"].max() + pd.DateOffset(months=1)
            fig_sim = go.Figure()
            fig_sim.add_hrect(
                y0=2.0, y1=4.0, fillcolor="rgba(5,150,105,0.11)", line_width=0,
                annotation_text="2%–4% Stability Band", annotation_position="top left",
            )
            fig_sim.add_trace(go.Scatter(
                x=prov_hist["obs_date"], y=prov_hist["headline_yoy_pct"], mode="lines+markers",
                name="Observed Ordinary YoY (%)", line=dict(color="#0F172A", width=2.2),
            ))
            fig_sim.add_trace(go.Scatter(
                x=prov_hist["obs_date"], y=prov_hist["b30_headline_yoy_pct"], mode="lines",
                name="Observed Bottom-30% YoY (%)", line=dict(color="#D97706", dash="dot", width=1.9),
            ))
            fig_sim.add_trace(go.Scatter(
                x=[prov_hist["obs_date"].max(), next_dt],
                y=[prov_hist["headline_yoy_pct"].iloc[-1], res_main["sim_pred"]],
                mode="lines+markers", name="Shocked Ordinary Forecast",
                line=dict(color="#2563EB", width=2.6),
            ))
            fig_sim.add_trace(go.Scatter(
                x=[prov_hist["obs_date"].max(), next_dt],
                y=[prov_hist["b30_headline_yoy_pct"].iloc[-1], res_main["b30_sim_pred"]],
                mode="lines+markers", name="Shocked Bottom-30% Forecast",
                line=dict(color="#DC2626", width=2.6),
            ))
            fig_sim.add_trace(go.Scatter(
                x=[next_dt, next_dt], y=[res_main["sim_q10"], res_main["sim_q90"]],
                mode="lines", name="80% Forecast Interval",
                line=dict(color="#2563EB", width=8), opacity=0.28,
            ))
            fig_sim.update_layout(
                height=360,
                title=f"Next-Month Household Impact: {sel_prov} ({latest_row['parent_region'].iloc[0]})",
                yaxis_title="YoY Inflation (%)", margin=dict(t=45, b=20), hovermode="x unified",
            )
            st.plotly_chart(fig_sim, width="stretch")

        with v_right:
            cmp_df = pd.DataFrame([
                {"Locality": sel_prov, "Household Group": "Ordinary Basket Pass-Through", "Impact_pp": round(res_main["direct_acct_pp"], 2)},
                {"Locality": sel_prov, "Household Group": "Bottom-30% Basket Pass-Through", "Impact_pp": round(res_main["b30_direct_pp"], 2)},
                {"Locality": cmp_prov, "Household Group": "Ordinary Basket Pass-Through", "Impact_pp": round(res_cmp["direct_acct_pp"], 2)},
                {"Locality": cmp_prov, "Household Group": "Bottom-30% Basket Pass-Through", "Impact_pp": round(res_cmp["b30_direct_pp"], 2)},
            ])
            fig_cmp = px.bar(
                cmp_df, x="Locality", y="Impact_pp", color="Household Group", barmode="group",
                color_discrete_sequence=["#1D4ED8", "#DC2626"],
                title="Same Shock Across Two Localities",
                labels={"Impact_pp": "Direct Price Shift (pp)", "Locality": ""},
            )
            fig_cmp.update_layout(height=360, margin=dict(t=45, b=20), legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig_cmp, width="stretch")

        if res_main["sim_prob"] >= 0.50 or res_main["sim_pred"] > 4.0 or res_main["b30_sim_pred"] > 4.0:
            st.warning(
                f"**LPCC Action Advisory for {sel_prov}:** Ordinary inflation is projected at **`{res_main['sim_pred']:.2f}%`** "
                f"and Bottom-30% inflation at **`{res_main['b30_sim_pred']:.2f}%`** (`{res_main['sim_prob']:.1%}` alert "
                f"probability), with low-income rice share at **`{res_main['b30_rice_w_pct']:.1f}%`** versus "
                f"**`{res_main['w_rice_pct']:.1f}%`** overall. Recommended DTI/DA action: activate LPCC SRP inspections, "
                "deploy Kadiwa rolling stores and NFA buffer rice, and target food or transport relief to Bottom-30% households."
            )
        else:
            st.success(
                f"**Stability Advisory for {sel_prov}:** Projected ordinary inflation (`{res_main['sim_pred']:.2f}%`) "
                f"and Bottom-30% inflation (`{res_main['b30_sim_pred']:.2f}%`) remain within the `4.0%` benchmark "
                f"(`{res_main['sim_prob']:.1%}` alert probability). Maintain routine market monitoring."
            )

        render_section_footer(
            step_num=4,
            summary_line=(
                "Local basket weights determine how identical rice, electricity, and diesel shocks translate into "
                "next-month household inflation and LPCC intervention priorities."
            ),
            prev_idx=2,
            prev_label="← Next-Month Forecasts & Alerts",
            next_idx=0,
            next_label="Back to Household Price Exposure",
            next_primary=False,
        )

    # ------------------------------------------------------------------
    # DuckDB SQL Audit Expander
    # ------------------------------------------------------------------
    st.markdown("")
    with st.expander("DuckDB SQL Audit Trail (<1,000 Aggregated Rows per View)"):
        st.caption(
            "Every chart executes parameterized SQL inside DuckDB over `2.2 MB` ZSTD Parquet marts so strictly fewer "
            "than `1,000` aggregated rows reach Plotly."
        )
        for q_title, row_cnt, q_sql in audit_queries:
            st.markdown(f"**{q_title}** · `{row_cnt:,}` rows")
            st.code(q_sql.strip(), language="sql")


if __name__ == "__main__":
    main()
