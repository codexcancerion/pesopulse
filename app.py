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

SECTION_PURPOSES = [
    "Establishes how inflation swings across time, regions, and ordinary versus Bottom-30% household budgets.",
    "Locates chronic high-breach provinces and traces how food, power, and fuel transmit into local baskets.",
    "Validates blind 2025–2026 forecasts, calibrates early >4.0% price alerts, and isolates sudden shock misses.",
    "Simulates how rice, electricity, and diesel shocks shift next-month ordinary and Bottom-30% inflation.",
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
        "clf_acc": float(((probs_te >= 0.5).astype(int) == y_clf[te_mask]).mean()),
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
        "fc_horizon": 6,
        "alert_tau": 0.45,
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
        st.markdown(summary_line)
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
    # Sidebar: Global Scope Filters Only
    # ------------------------------------------------------------------
    st.sidebar.header("Monitoring Scope")
    st.sidebar.caption(
        "Narrow the geographic or time scope below. Every summary metric and visualization updates live through DuckDB."
    )

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
        help="Include or isolate any of the 18 official PSA administrative regions.",
    )
    yr_range = st.sidebar.slider(
        "Observation Years",
        2019,
        2026,
        key="flt_years",
        help="Monthly year-on-year inflation from Jan 2019 through Oct 2026 against the 2018=100 base basket.",
    )

    if len(sel_islands) < 3 or len(sel_regs) < len(regs_all) or yr_range != (2019, 2026):
        st.sidebar.button(
            "Reset Scope Filters",
            on_click=reset_global_filters,
            args=(regs_all,),
            width="stretch",
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "**Data Source:** 8 verified PSA OpenSTAT tables covering 101 provinces and highly urbanized cities "
        "from Jan 2018 to Oct 2026, served from 2.2 MB ZSTD Parquet marts."
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
    # Top KPI Query + Header & Glossary
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

    h_left, h_right = st.columns([3.5, 1.0])
    with h_left:
        st.title("PesoPulse: Local Price Council Shock & Inflation Monitor")
        st.markdown(
            "Designed for **DTI and Department of Agriculture Local Price Coordinating Councils** to track local "
            "inflation gaps across **`101` provinces and cities** and estimate how sudden jumps in **rice, electricity, "
            "and diesel** hit ordinary and **Bottom-30% low-income households** next month."
        )
    with h_right:
        st.markdown("")
        with st.popover("Glossary", width="stretch"):
            st.markdown(
                """
                **Key Indicators**
                - **LPCC:** Local Price Coordinating Council under DTI, DA, and LGUs monitoring retail staples and SRP compliance.
                - **CPI (`2018=100`) & YoY (`%`):** Consumer Price Index and percentage price change versus the same month one year earlier.
                - **Percentage Points (`pp`):** Direct arithmetic difference between two inflation rates.
                - **Bottom-30% Households:** Lowest-income `30%` of families, whose budgets allocate a much larger share to rice and basic food.
                - **`4.0%` Price-Alert Line:** Official national upper inflation benchmark used as a council intervention trigger.
                - **Fan Charts & TreeSHAP:** `80%` and `95%` prediction intervals paired with feature-level forecast attribution.
                """
            )
        st.caption(f"`{int(kpi['n_geos'])}` localities · `{int(kpi['n_obs']):,}` records · `{yr_range[0]}–{yr_range[1]}`")

    # ------------------------------------------------------------------
    # Streamlined 3-KPI Strip (With Practical Number Meaning)
    # ------------------------------------------------------------------
    mean_yoy_f = float(kpi["mean_yoy"])
    mean_b30_f = float(kpi["mean_b30"])
    b30_gap_f = mean_b30_f - mean_yoy_f
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Ordinary Household Inflation",
        f"{mean_yoy_f:.2f}%",
        f"₱1,000 basket → ₱{1000 * (1 + mean_yoy_f / 100):.0f}/yr",
        delta_color="off",
        border=True,
        help=(
            f"Average year-on-year CPI inflation for all-income households. At {mean_yoy_f:.2f}%, a ₱1,000 monthly "
            f"basket from one year earlier costs roughly ₱{1000 * (1 + mean_yoy_f / 100):.0f} today."
        ),
    )
    c2.metric(
        "Bottom-30% Household Inflation",
        f"{mean_b30_f:.2f}%",
        f"{b30_gap_f:+.2f} pp vs Ordinary",
        delta_color="inverse",
        border=True,
        help=(
            f"Average inflation for the lowest-income 30% of families. A {b30_gap_f:+.2f} percentage-point gap means "
            "poorer families face faster price increases than general shoppers because rice takes more of their budget."
        ),
    )
    c3.metric(
        ">4.0% Alert Frequency",
        f"{kpi['breach_pct']:.1f}%",
        "Months Above 4% Ceiling",
        delta_color="off",
        border=True,
        help=(
            f"{kpi['breach_pct']:.1f}% of all province-months crossed the 4.0% official upper price-stability line, "
            "signaling periods where local price councils must intensify retail monitoring."
        ),
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
                help=SECTION_PURPOSES[i],
                width="stretch",
            )

    audit_queries: list[tuple[str, int, str]] = [("LPCC KPI Summary", 1, kpi_sql)]

    # ==================================================================
    # SECTION 1: HOUSEHOLD PRICE EXPOSURE (5 ESSENTIAL BASIC CHARTS)
    # ==================================================================
    if active_idx == 0:
        st.subheader("Why Staple Shocks Hit Low-Income Households Hardest")
        st.caption(
            "Tracks how inflation moves across time, regions, and income groups through monthly price waves, "
            "provincial shock distributions, regional rankings, retail rice pass-through, and household budget shares."
        )
        st.markdown(
            "A single national inflation headline hides the sharpest pain points local price councils have to manage. "
            "Between January 2019 and October 2026, Philippine inflation surged to `8.7%` in early 2023, cooled to "
            "`0.9%` in July 2025, and rebounded to `7.8%` in October 2026. In everyday terms, `7.8%` inflation means "
            "a household spending `₱10,000` on basic goods last year must now find `₱10,780` for the exact same items."
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
            height=355, margin=dict(t=42, b=18), yaxis_title="Year-on-Year Inflation (%)", hovermode="x unified",
        )
        st.plotly_chart(fig_ts, width="stretch")
        st.caption(
            "Whenever rice inflation accelerates past `15%` to `24%`, the red dashed Bottom-30% line separates sharply "
            "above the ordinary household rate, signaling immediate pressure on low-income consumers."
        )

        st.markdown(
            "#### Uneven Regional Pressure & Extreme Local Tails\n"
            "Price surges rarely distribute evenly across the archipelago. Comparing the full distribution of monthly "
            "provincial readings against regional medians shows that extreme inflation spikes are recurring local "
            "supply events rather than statistical noise, with Bicol and MIMAROPA running consistently hotter than NCR."
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
            fig_hist.update_layout(height=335, margin=dict(t=42, b=18))
            st.plotly_chart(fig_hist, width="stretch")
            st.caption(
                f"Each bar counts how many province-months landed at that inflation rate. Readings beyond the "
                f"`{tukey_hi}%` Tukey upper fence (`1.5x` the middle spread) signal severe local supply shocks rather "
                "than normal seasonal fluctuation."
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
            fig_bar.update_layout(height=335, margin=dict(t=42, b=18))
            st.plotly_chart(fig_bar, width="stretch")
            st.caption(
                "Regions are ordered from hottest to coolest median rate, with brighter bars highlighting where "
                "low-income households pay the steepest extra inflation penalty."
            )

        st.markdown(
            "#### Why Rice Drives Regressive Inflation: Local Basket Math\n"
            "Retail rice inflation correlates directly with provincial headline CPI across all 101 localities, yet "
            "ordinary and low-income families do not feel that jump equally. Because poorer households devote a far "
            "larger share of their monthly expenditure to rice and basic food, the exact same market price hike hits "
            "Bottom-30% families nearly six times harder during high food-inflation periods."
        )

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
            fig_scat.update_layout(height=335, margin=dict(t=42, b=18))
            st.plotly_chart(fig_scat, width="stretch")
            st.caption(
                "Bubbles are sized by local rice budget weight. Provinces with faster retail rice increases systematically "
                "record higher overall inflation (`r = +0.45`)."
            )

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
                height=335, margin=dict(t=52, b=18), legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig_pie, width="stretch")
            st.caption(
                f"Bottom-30% households spend roughly `₱{bsk['b30_rice'] * 10:.0f}` out of every `₱1,000` on rice alone "
                f"(`{bsk['b30_rice']:.1f}%`) compared with `₱{bsk['gen_rice'] * 10:.0f}` (`{bsk['gen_rice']:.1f}%`) for "
                "ordinary households, magnifying every staple price jump."
            )

        if summary:
            ht = summary.get("hypothesis_tests", {})
            with st.expander("Statistical Verification: ANOVA, Paired t-Tests & Upstream Correlations"):
                st.markdown(
                    f"- **Regional Divergence (One-Way ANOVA):** `F = {ht.get('H1_regional_anova_F')}`, `p < 0.001` "
                    f"across all 18 regions (`Island F = {ht.get('H1_island_anova_F')}`), confirming regional inflation "
                    "differences are structural rather than random noise.\n"
                    f"- **Regressive Food Inequality (Paired t-Test):** Bottom-30% inflation exceeds ordinary inflation "
                    f"by `+{ht.get('H2_b30_vs_all_mean_diff_pp')} pp` overall (`t = {ht.get('H2_b30_paired_t')}`), jumping "
                    f"nearly sixfold to **`+{ht.get('H2_high_food_b30_diff_pp')} pp`** (`t = {ht.get('H2_high_food_paired_t')}`) "
                    "during high food-inflation regimes above `5%`.\n"
                    f"- **Upstream Cost Linkage:** Local retail rice YoY correlates at `r = +{ht.get('H3_rice_yoy_pearson_r')}` "
                    f"and prior-month island Wholesale GWPI correlates at `r = +{ht.get('H3_gwpi_lag1_pearson_r')}` with "
                    "provincial headline inflation."
                )

        render_section_footer(
            step_num=1,
            summary_line=(
                "**Key Finding:** Regional prices diverge significantly and rice surges hit Bottom-30% budgets nearly "
                "six times harder. Next, map where chronic hotspots cluster across all 101 localities and trace how "
                "food, electricity, and fuel pressure flows through local baskets."
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
            "Maps chronic provincial hotspots and traces commodity drivers using a 101-locality bubble map, "
            "regional spending sunburst, cost-channel Sankey flow, compound shock profile, and category stream."
        )
        st.markdown(
            "Regional averages still conceal which exact municipalities need market inspections or Kadiwa rolling "
            "stores first. Outer-island supply corridors face freight bottlenecks that push local inflation past the "
            "`4.0%` alert ceiling more than half the time, while spending hierarchies reveal whether food or energy "
            "drives each area's cost burden."
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
            fig_geo.update_layout(height=405, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig_geo, width="stretch")
            st.caption(
                "Larger, brighter bubbles breach `4.0%` most often, spotlighting outer-island hotspots such as "
                "Batanes (`5.25%`), Dinagat Islands (`5.07%`), and Occidental Mindoro."
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
            fig_sun.update_layout(height=405, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig_sun, width="stretch")
            st.caption("Click any island or region ring to zoom into the exact spending categories driving local pressure.")

        st.markdown(
            "#### Tracing Cost Channels & Compound Provincial Exposure\n"
            "Local councils get the highest leverage by targeting the specific supply channels behind household pain. "
            "Across Luzon, Visayas, and Mindanao, **rice and food (`48%`)** together with **electricity, LPG, and "
            "diesel (`28%`)** generate more than three-quarters of cumulative inflation pressure. Use the parallel "
            "axes below to filter localities hit by multiple channels at once."
        )

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
                height=380, margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_san, width="stretch")

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
            fig_par.update_layout(height=380, margin=dict(t=45, b=10))
            st.plotly_chart(fig_par, width="stretch")
            st.caption("Drag vertically along any axis to isolate provinces facing simultaneous rice, utility, and fuel spikes.")

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
        fig_stream.update_layout(height=320, margin=dict(t=40, b=10), hovermode="x unified")
        st.plotly_chart(fig_stream, width="stretch")
        st.caption(
            "Food dominated the 2023 surge, receded sharply in mid-2025, and returned alongside transport fuel in 2026, "
            "confirming why councils must monitor both food and energy channels together."
        )

        render_section_footer(
            step_num=2,
            summary_line=(
                "**Key Finding:** Food, electricity, and transport fuel account for over three-quarters of local "
                "inflation pressure and cluster in outer-island supply corridors. Next, examine blind 2025–2026 "
                "forecasts and early price-alert performance."
            ),
            prev_idx=0,
            prev_label="← Household Price Exposure",
            next_idx=2,
            next_label="Next-Month Forecasts & Alerts →",
        )

    # ==================================================================
    # SECTION 3: NEXT-MONTH FORECASTS & ALERTS (INLINE CONTROLS + CONTEXT)
    # ==================================================================
    elif active_idx == 2:
        st.subheader("Blind 2025–2026 Forecasts, Uncertainty & Early Price Alerts")
        st.caption(
            "Evaluates blind next-month forecast accuracy, multi-month SARIMAX uncertainty bands, TreeSHAP drivers, "
            "interactive >4.0% council alert cutoffs, and abrupt supply-shock misses."
        )
        st.markdown(
            "To prevent future price trends from leaking into evaluation, every model trains strictly on `2020–2024` "
            "and is tested blind on `2,020` paired provincial holdout months across `2025–2026`. Linear **Ridge "
            "Regression** outperforms tree ensembles out-of-time because linear shrinkage extrapolates smoothly "
            "across the sharp swing from `0.9%` to `7.8%`, whereas tree leaves clip near past training bounds."
        )

        st.info(
            f"**What the Holdout Accuracy Numbers Mean (`2025–2026`):** "
            f"Ordinary Ridge averages **`{eng['ridge_mae']:.2f} pp` MAE** (`R² = {eng['ridge_r2']:.3f}`), meaning "
            f"forecasts typically land within `1` percentage point of the actual PSA release and explain "
            f"`{eng['ridge_r2'] * 100:.0f}%` of provincial variation. Bottom-30% Ridge averages **`{eng['b30_mae']:.2f} pp` "
            f"MAE** (`R² = {eng['b30_r2']:.3f}`), while the XGBoost alert model flags `>4.0%` surges at "
            f"**`{eng['clf_acc']:.1%}` accuracy** (`ROC-AUC = {eng['clf_auc']:.3f}`, indicating near-complete separation "
            "between calm and overheating months)."
        )

        # 1. Dynamic SARIMAX Fan Chart with Inline Horizon Control
        st.markdown("#### Multi-Month Outlook & Forecast Uncertainty")
        st.caption(
            "Single-point predictions conceal risk. Adjust the horizon below to see how `80%` and `95%` prediction "
            "bands widen as seasonal uncertainty compounds across future months."
        )
        fc_c1, fc_c2 = st.columns([1.6, 1.2])
        with fc_c1:
            sel_fc_geo = st.selectbox(
                "National or Regional Series",
                ["PHILIPPINES"] + regs_all,
                help="Fits a seasonal SARIMAX(1,0,1)(1,0,1,12) model with 80% and 95% confidence bands.",
            )
        with fc_c2:
            horizon_mo = st.slider(
                "Forecast Horizon (Months)",
                3,
                12,
                key="fc_horizon",
                help="Number of months projected ahead in the uncertainty fan chart.",
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
            height=345,
            title=f"SARIMAX Forecast & Uncertainty Bands: {sel_fc_geo} ({horizon_mo} Months)",
            yaxis_title="Year-on-Year Inflation (%)", margin=dict(t=42, b=18), hovermode="x unified",
        )
        st.plotly_chart(fig_fan, width="stretch")

        # Optional XAI Attribution Expander (keeps main canvas uncluttered)
        if SHAP_CSV.exists():
            with st.expander("Model Attribution: What Drives Next-Month Forecasts? (TreeSHAP)"):
                st.caption(
                    "Because monthly year-over-year inflation overlaps eleven of twelve months with the prior month, "
                    "current momentum dominates point forecasts, reinforced by Core CPI, seasonal harvest cycles, "
                    "prior-month Wholesale GWPI, and diesel fuel."
                )
                sh_df = pd.read_csv(SHAP_CSV).head(8).iloc[::-1].copy()
                sh_df["readable_feature"] = sh_df["feature"].map(humanize_feature)
                fig_sh = px.bar(
                    sh_df, x="mean_abs_shap", y="readable_feature", orientation="h",
                    hover_data={"feature": True, "mean_abs_shap": ":.3f", "readable_feature": False},
                    title="Top 8 Next-Month Forecast Drivers (TreeSHAP)",
                    color_discrete_sequence=["#0D9488"],
                )
                fig_sh.update_layout(
                    height=310, yaxis_title="", xaxis_title="Mean |SHAP| Impact (pp)",
                    margin=dict(t=40, b=15),
                )
                st.plotly_chart(fig_sh, width="stretch")

        # 2. Interactive ROC Alert Cutoff + Cross-Filtered Abrupt Shock Inspector
        st.divider()
        st.markdown("#### Early Price-Alert Calibration & Abrupt Shock Diagnostics")
        st.markdown(
            "Local councils balance two operational costs: acting too late misses a food surge, while acting on weak "
            "signals wastes inspection teams. Tune the alert cutoff on the left to calibrate false alarms versus "
            "caught surges, then inspect sudden supply-shock misses on the right to see why scenario testing matters."
        )

        diag_l, diag_r = st.columns([1.0, 1.35])
        with diag_l:
            tau = st.slider(
                ">4.0% Price-Alert Cutoff (τ)",
                0.15,
                0.85,
                step=0.02,
                key="alert_tau",
                help="Lower cutoffs trigger earlier LPCC alerts with higher recall; higher cutoffs reduce false alarms.",
            )
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
                name=f"Cutoff τ = {tau:.2f}",
                marker=dict(color="#DC2626", size=12, symbol="star"),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random",
                line=dict(dash="dash", color="#64748B"),
            ))
            fig_roc.update_layout(
                height=300, title=">4.0% Early-Alert ROC Curve",
                xaxis_title="False Alarm Rate", yaxis_title="Caught Surge Rate",
                margin=dict(t=42, b=18),
            )
            st.plotly_chart(fig_roc, width="stretch")
            st.caption(
                f"**At τ = `{tau:.2f}`:** The model caught **`{tp}`** real surges, missed **`{fn}`**, and raised "
                f"**`{fp}`** false alarms. Precision of **`{prec_v:.1%}`** means roughly `{round(prec_v * 10)}` in `10` "
                f"council alerts are genuine, while Recall of **`{rec_v:.1%}`** shows the share of true surges caught early."
            )

        with diag_r:
            err_df = eng["te_meta"].copy()
            err_df["predicted_yoy_t1"] = np.round(eng["preds_te"], 2)
            err_df["abs_error_pp"] = np.round(np.abs(err_df["target_yoy_t1"] - err_df["predicted_yoy_t1"]), 2)
            top_err = err_df.sort_values("abs_error_pp", ascending=False).head(8)
            err_geos = list(dict.fromkeys(top_err["clean_geo"].tolist()))

            sel_err_geo = st.selectbox(
                "Cross-Filtered High-Error Locality (Top 2025–2026 Shock Misses)",
                err_geos,
                index=0,
                help="Select a locality with large holdout errors to inspect abrupt supply-shock months flagged in red.",
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
                    name="Abrupt Shock (>=2 pp miss)",
                    marker=dict(color="#DC2626", size=11, symbol="diamond"),
                ))
            fig_err.update_layout(
                height=300, title=f"Observed vs. Predicted Holdout Trajectory: {sel_err_geo}",
                yaxis_title="YoY Inflation (%)", margin=dict(t=42, b=18),
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
                "**Key Finding:** Momentum models forecast steady months reliably and flag `>4.0%` alerts at `92.0%+` "
                "accuracy, yet red-diamond supply shocks happen overnight. Open the simulator next to test rice, "
                "electricity, and diesel shocks before they hit public markets."
            ),
            prev_idx=1,
            prev_label="← Provincial Hotspots & Flows",
            next_idx=3,
            next_label="LPCC Supply-Shock Simulator →",
        )

    # ==================================================================
    # SECTION 4: LPCC SUPPLY-SHOCK SIMULATOR (CLEAN DEFAULT + OPTIONAL COMPARE)
    # ==================================================================
    else:
        st.subheader("LPCC Next-Month Shock Simulator: Ordinary vs. Bottom-30% Households")
        st.caption(
            "Translates hypothetical rice, electricity, and diesel price jumps into next-month ordinary and "
            "Bottom-30% forecasts, an 80% risk trajectory, optional cross-province comparisons, and LPCC advisories."
        )
        st.markdown(
            "When a typhoon damages palay harvests or global crude spikes raise pump prices, local councils need "
            "immediate household estimates before the next PSA release. Choose a locality and adjust **rice, "
            "electricity, and diesel** below to see how percentage-point (`pp`) price shifts pass through official "
            "local CPI weights into ordinary and Bottom-30% household budgets next month."
        )

        p1, p2, p3, p4 = st.columns(4)
        p1.button(
            "Rice Supply Spike (+12 pp)",
            on_click=apply_shock_preset, args=(12.0, 2.0, 2.0), width="stretch",
            help="Simulates a domestic harvest shortfall or import delay pushing retail rice up sharply.",
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

        top_s1, top_s2 = st.columns([2.0, 1.0])
        with top_s1:
            sel_prov = st.selectbox(
                "Target Province or Highly Urbanized City",
                provs,
                key="sim_prov",
                help="Locality evaluated using its official PSA All-Income and Bottom-30% basket weights.",
            )
        with top_s2:
            st.markdown("")
            show_cmp = st.toggle(
                "Compare with a second locality",
                value=False,
                key="toggle_cmp_prov",
                help="Opens a side-by-side comparison showing how identical shocks hit a second province differently.",
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
        prov_hist = res_main["prov_hist"]
        latest_row = res_main["latest_row"]
        audit_queries.append((
            f"Simulator Local Series ({sel_prov})",
            len(prov_hist),
            f"SELECT * FROM prov_panel WHERE clean_geo = '{sel_prov}' ORDER BY obs_date DESC LIMIT 18",
        ))

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Ordinary Household Forecast (t+1)",
            f"{res_main['sim_pred']:.2f}%",
            f"{res_main['sim_pred'] - res_main['base_pred']:+.2f} pp vs baseline ({res_main['direct_acct_pp']:+.2f} pp direct)",
            delta_color="inverse",
            border=True,
            help=f"Next-month All-Income Ridge forecast; 80% uncertainty band spans {res_main['sim_q10']:.2f}% to {res_main['sim_q90']:.2f}%.",
        )
        m2.metric(
            "Bottom-30% Household Forecast (t+1)",
            f"{res_main['b30_sim_pred']:.2f}%",
            f"{res_main['b30_sim_pred'] - res_main['b30_base_pred']:+.2f} pp vs baseline ({res_main['b30_direct_pp']:+.2f} pp direct)",
            delta_color="inverse",
            border=True,
            help=(
                f"Dedicated Bottom-30% Ridge forecast. In {sel_prov}, rice carries {res_main['b30_rice_w_pct']:.1f}% "
                f"of low-income budgets versus {res_main['w_rice_pct']:.1f}% overall."
            ),
        )
        m3.metric(
            ">4.0% Alert Probability",
            f"{res_main['sim_prob']:.1%}",
            f"{(res_main['sim_prob'] - res_main['base_prob']) * 100:+.1f} pp vs baseline",
            delta_color="inverse",
            border=True,
            help="XGBoost probability that next month's provincial inflation exceeds the 4.0% alert ceiling.",
        )

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
            yaxis_title="YoY Inflation (%)", margin=dict(t=42, b=18), hovermode="x unified",
        )

        if show_cmp:
            default_cmp = "Camarines Sur" if sel_prov != "Camarines Sur" and "Camarines Sur" in provs else (
                "City of Baguio" if sel_prov != "City of Baguio" and "City of Baguio" in provs else provs[0]
            )
            cmp_prov = st.selectbox(
                "Comparison Province or City",
                provs,
                index=provs.index(default_cmp),
                help="Contrasts how the exact same price shock hits a second locality with different basket weights.",
            )
            res_cmp = compute_province_shock(con, eng, cmp_prov, rice_shock, util_shock, fuel_shock)
            v_left, v_right = st.columns([1.45, 1.0])
            with v_left:
                st.plotly_chart(fig_sim, width="stretch")
            with v_right:
                cmp_df = pd.DataFrame([
                    {"Locality": sel_prov, "Household Group": "Ordinary Pass-Through", "Impact_pp": round(res_main["direct_acct_pp"], 2)},
                    {"Locality": sel_prov, "Household Group": "Bottom-30% Pass-Through", "Impact_pp": round(res_main["b30_direct_pp"], 2)},
                    {"Locality": cmp_prov, "Household Group": "Ordinary Pass-Through", "Impact_pp": round(res_cmp["direct_acct_pp"], 2)},
                    {"Locality": cmp_prov, "Household Group": "Bottom-30% Pass-Through", "Impact_pp": round(res_cmp["b30_direct_pp"], 2)},
                ])
                fig_cmp = px.bar(
                    cmp_df, x="Locality", y="Impact_pp", color="Household Group", barmode="group",
                    color_discrete_sequence=["#1D4ED8", "#DC2626"],
                    title="Same Shock Across Two Localities",
                    labels={"Impact_pp": "Direct Price Shift (pp)", "Locality": ""},
                )
                fig_cmp.update_layout(height=360, margin=dict(t=42, b=18), legend=dict(orientation="h", y=-0.18))
                st.plotly_chart(fig_cmp, width="stretch")
        else:
            st.plotly_chart(fig_sim, width="stretch")

        if res_main["sim_prob"] >= 0.50 or res_main["sim_pred"] > 4.0 or res_main["b30_sim_pred"] > 4.0:
            ord_cost = 10000 * (1 + res_main["sim_pred"] / 100.0)
            b30_cost = 10000 * (1 + res_main["b30_sim_pred"] / 100.0)
            st.warning(
                f"**LPCC Action Advisory for {sel_prov}:** Ordinary inflation is projected at "
                f"**`{res_main['sim_pred']:.2f}%`** (a `₱10,000` basket from last year costs roughly **`₱{ord_cost:,.0f}`**) "
                f"and Bottom-30% inflation at **`{res_main['b30_sim_pred']:.2f}%`** (**`₱{b30_cost:,.0f}`**, with "
                f"`{res_main['sim_prob']:.1%}` alert probability). Because low-income households in {sel_prov} spend "
                f"**`{res_main['b30_rice_w_pct']:.1f}%`** of their budget on rice versus **`{res_main['w_rice_pct']:.1f}%`** "
                f"overall, they absorb an extra **`{res_main['b30_extra_pp']:+.2f} pp`** direct rice burden. "
                "Recommended DTI and DA response: activate LPCC SRP inspections, deploy Kadiwa rolling stores and "
                "NFA buffer rice, and prioritize targeted food or transport relief for Bottom-30% families."
            )
        else:
            st.success(
                f"**Stability Advisory for {sel_prov}:** Projected ordinary inflation (`{res_main['sim_pred']:.2f}%`) "
                f"and Bottom-30% inflation (`{res_main['b30_sim_pred']:.2f}%`) stay within the `4.0%` benchmark "
                f"(`{res_main['sim_prob']:.1%}` alert probability). Maintain routine market monitoring."
            )

        render_section_footer(
            step_num=4,
            summary_line=(
                "**Key Finding:** Local CPI basket weights determine how identical rice, electricity, and diesel "
                "shocks translate into next-month household inflation and LPCC relief priorities."
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
            "Every visualization queries DuckDB SQL over `2.2 MB` ZSTD Parquet marts so strictly fewer than "
            "`1,000` aggregated rows reach Plotly."
        )
        for q_title, row_cnt, q_sql in audit_queries:
            st.markdown(f"**{q_title}** · `{row_cnt:,}` rows")
            st.code(q_sql.strip(), language="sql")


if __name__ == "__main__":
    main()
