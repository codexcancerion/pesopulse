"""
PesoPulse: MIT 118 Final Group Project (Team 3)
Philippine Provincial Inflation Analytics, Forecasting & Supply-Shock Simulator
Data: PSA OpenSTAT 2M (8 Joined Tables across 101 Provinces and HUCs, Jan 2018 to Oct 2026)
Engine: DuckDB SQL + Plotly + Ridge / XGBoost / Quantile GBR / SARIMAX
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
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import streamlit as st
import streamlit.components.v1 as components
import xgboost as xgb

st.set_page_config(
    page_title="PesoPulse | Philippine Provincial Inflation & Policy Intelligence",
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
    "1. National vs. Local Reality",
    "2. Provincial Hotspots & Flows",
    "3. Forecasts & Breach Risk",
    "4. Supply-Shock Simulator",
]

FEATURE_LABELS = {
    "headline_yoy_pct": "Current Headline Inflation (%)",
    "yoy_roll3_mean": "3-Month Inflation Momentum (%)",
    "natl_core_yoy_pct": "National Core Inflation (%)",
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
    cleaned = raw_name.replace("cat__", "").replace("num__", "").replace("_", " ")
    return cleaned.title()


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
          AND yoy_lag1 IS NOT NULL
          AND yoy_lag3 IS NOT NULL
          AND yoy_lag12 IS NOT NULL
        ORDER BY obs_date, clean_geo
    """).df()
    df["obs_date"] = pd.to_datetime(df["obs_date"])

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
    y_clf = df["target_breach_4pct_t1"].astype(int).values

    tr_mask = df["year"] <= 2024
    te_mask = df["year"] >= 2025

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
    ])
    ridge = Pipeline([("pre", pre), ("m", Ridge(alpha=10.0))])
    reg = Pipeline([
        ("pre", pre),
        ("m", xgb.XGBRegressor(
            n_estimators=220, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1,
        )),
    ])
    clf = Pipeline([
        ("pre", pre),
        ("m", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, eval_metric="logloss", n_jobs=-1,
        )),
    ])

    ridge.fit(X[tr_mask], y_reg[tr_mask])
    reg.fit(X[tr_mask], y_reg[tr_mask])
    clf.fit(X[tr_mask], y_clf[tr_mask])

    X_tr_enc = reg.named_steps["pre"].transform(X[tr_mask])
    q10 = GradientBoostingRegressor(
        loss="quantile", alpha=0.10, n_estimators=120, max_depth=4, learning_rate=0.06, random_state=42
    ).fit(X_tr_enc, y_reg[tr_mask])
    q90 = GradientBoostingRegressor(
        loss="quantile", alpha=0.90, n_estimators=120, max_depth=4, learning_rate=0.06, random_state=42
    ).fit(X_tr_enc, y_reg[tr_mask])

    probs_te = clf.predict_proba(X[te_mask])[:, 1]
    preds_te = ridge.predict(X[te_mask])
    return {
        "df": df,
        "num_feats": num_feats,
        "cat_feats": cat_feats,
        "ridge": ridge,
        "reg": reg,
        "clf": clf,
        "q10": q10,
        "q90": q90,
        "y_te_reg": y_reg[te_mask],
        "preds_te": preds_te,
        "y_te_clf": y_clf[te_mask],
        "probs_te": probs_te,
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
        "sim_gwpi": 3.0,
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


def apply_shock_preset(rice: float, util: float, fuel: float, gwpi: float):
    st.session_state["sim_rice"] = rice
    st.session_state["sim_util"] = util
    st.session_state["sim_fuel"] = fuel
    st.session_state["sim_gwpi"] = gwpi


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
    gwpi_shock: float,
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

    direct_acct_pp = w_rice * rice_shock + w_util * util_shock + w_fuel * fuel_shock
    X_sim["rice_cpi_yoy_pct"] = X_sim["rice_cpi_yoy_pct"].fillna(0) + rice_shock
    X_sim["food_bev_yoy_pct"] = X_sim["food_bev_yoy_pct"].fillna(0) + rice_shock * (w_rice / w_food)
    X_sim["elec_gas_fuel_yoy_pct"] = X_sim["elec_gas_fuel_yoy_pct"].fillna(0) + util_shock
    X_sim["personal_fuel_yoy_pct"] = X_sim["personal_fuel_yoy_pct"].fillna(0) + fuel_shock
    X_sim["transport_yoy_pct"] = X_sim["transport_yoy_pct"].fillna(0) + fuel_shock * 0.45
    X_sim["gwpi_all_yoy_lag1"] = X_sim["gwpi_all_yoy_lag1"].fillna(0) + gwpi_shock
    X_sim["headline_yoy_pct"] = X_sim["headline_yoy_pct"].fillna(0) + direct_acct_pp

    base_pred = float(eng["ridge"].predict(X_base)[0])
    sim_pred = float(eng["ridge"].predict(X_sim)[0])
    xgb_sim_pred = float(eng["reg"].predict(X_sim)[0])
    X_sim_enc = eng["reg"].named_steps["pre"].transform(X_sim)
    sim_q10 = float(eng["q10"].predict(X_sim_enc)[0])
    sim_q90 = float(eng["q90"].predict(X_sim_enc)[0])
    base_prob = float(eng["clf"].predict_proba(X_base)[0, 1])
    sim_prob = float(eng["clf"].predict_proba(X_sim)[0, 1])

    b30_rice_w = float(latest_row["b30_rice_weight"].iloc[0] or (w_rice * 180.0)) / 100.0
    b30_extra_pp = (b30_rice_w - w_rice) * rice_shock

    return {
        "prov_hist": prov_hist,
        "latest_row": latest_row,
        "w_rice_pct": w_rice * 100.0,
        "b30_rice_w_pct": b30_rice_w * 100.0,
        "direct_acct_pp": direct_acct_pp,
        "b30_extra_pp": b30_extra_pp,
        "base_pred": base_pred,
        "sim_pred": sim_pred,
        "xgb_sim_pred": xgb_sim_pred,
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
        b_left, b_spacer, b_right = st.columns([1.15, 1.3, 1.55])
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
    # Sidebar: Scope Filters & Forecast Parameters
    # ------------------------------------------------------------------
    st.sidebar.header("Scope & Filters")
    st.sidebar.caption("Filter by geography or period. All metrics and charts query DuckDB live.")

    islands = ["Luzon", "Visayas", "Mindanao"]
    sel_islands = st.sidebar.multiselect(
        "Island Cluster",
        islands,
        key="flt_islands",
        help="Filter provinces and cities across Luzon, Visayas, and Mindanao.",
    )
    sel_regs = st.sidebar.multiselect(
        "Regions",
        regs_all,
        key="flt_regs",
        help="Include or isolate any of the 18 official administrative regions, including NIR and BARMM.",
    )
    yr_range = st.sidebar.slider(
        "Observation Years",
        2019,
        2026,
        key="flt_years",
        help="Year-on-year inflation spans Jan 2019 through Oct 2026 against the 2018=100 PSA base basket.",
    )

    if len(sel_islands) < 3 or len(sel_regs) < len(regs_all) or yr_range != (2019, 2026):
        st.sidebar.button(
            "Reset Filters",
            on_click=reset_global_filters,
            args=(regs_all,),
            width="stretch",
        )

    st.sidebar.divider()
    st.sidebar.subheader("Forecast Parameters")
    tau = st.sidebar.slider(
        "BSP >4.0% Alert Threshold (τ)",
        0.15,
        0.85,
        0.45,
        0.02,
        help=(
            "Probability cutoff for flagging an inflation breach in Forecasts & Breach Risk. "
            "Lower cutoffs catch more breaches early at the cost of extra false alarms."
        ),
    )
    horizon_mo = st.sidebar.slider(
        "SARIMAX Horizon (Months)",
        3,
        12,
        6,
        1,
        help="Forward projection window for the regional SARIMAX fan chart.",
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
        st.title("PesoPulse: Provincial Inflation & Policy Intelligence")
        st.markdown(
            "National inflation headlines mask sharp local price gaps across the Philippines. "
            "Examine how prices diverge across `101` provinces and cities, why staple food surges hit "
            "low-income households hardest, where target breaches are building, and how supply shocks "
            "pass through local baskets."
        )
    with h_right:
        st.markdown("")
        with st.popover("Glossary", width="stretch"):
            st.markdown(
                """
                **Key Indicators & Methods**
                - **CPI (`2018=100`):** Consumer Price Index measuring the cost of a fixed household basket relative to 2018.
                - **YoY Inflation (`%`):** Price change compared with the same month one year earlier.
                - **Percentage Points (`pp`):** Direct arithmetic difference between two rates, such as `4.0%` to `5.5%` (`+1.5 pp`).
                - **BSP `2.0%`–`4.0%` Target:** Bangko Sentral ng Pilipinas price-stability band; readings above `4.0%` breach the upper ceiling.
                - **Bottom-30% Households:** Lowest-income `30%` of families, whose budgets allocate far more to rice and food.
                - **Purchasing Power (`PPP`):** Real value of a `2018` peso today; `₱0.72` means `28` centavos of value was lost.
                - **GWPI:** General Wholesale Price Index tracking upstream bulk prices before retail pass-through.
                - **TreeSHAP & Fan Charts:** Feature-level forecast attribution paired with `80%` and `95%` prediction intervals.
                """
            )
        st.caption(
            f"`{len(sel_islands or islands)}` island groups · "
            f"`{len(sel_regs or regs_all)}` regions · `{yr_range[0]}–{yr_range[1]}`"
        )

    # ------------------------------------------------------------------
    # Top KPI Strip
    # ------------------------------------------------------------------
    kpi_sql = f"""
        SELECT
            COUNT(*) AS n_obs,
            COUNT(DISTINCT clean_geo) AS n_geos,
            ROUND(AVG(headline_yoy_pct), 2) AS mean_yoy,
            ROUND(AVG(b30_headline_yoy_pct), 2) AS mean_b30,
            ROUND(AVG(is_breach_bsp_4pct) * 100.0, 1) AS breach_pct,
            ROUND(AVG(ppp_index), 2) AS mean_ppp
        FROM prov_panel {where_sql}
    """
    kpi = con.execute(kpi_sql).df().iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Subnational Observations",
        f"{int(kpi['n_obs']):,}",
        f"{int(kpi['n_geos'])} Provinces & HUCs",
        border=True,
        help="Monthly observations across 100 provinces/highly urbanized cities plus NCR after stripping aggregate summary rows.",
    )
    c2.metric(
        "Mean Headline Inflation",
        f"{kpi['mean_yoy']:.2f}%",
        "All-Income Basket",
        border=True,
        help="Average year-on-year CPI (2018=100) inflation across the active geographic and time selection.",
    )
    c3.metric(
        "Bottom-30% Inflation",
        f"{kpi['mean_b30']:.2f}%",
        f"{(kpi['mean_b30'] - kpi['mean_yoy']):+.2f} pp vs General",
        delta_color="inverse",
        border=True,
        help="Average inflation faced by the lowest-income 30% of households relative to the all-income basket.",
    )
    c4.metric(
        ">4.0% Target Breaches",
        f"{kpi['breach_pct']:.1f}%",
        "Months Above BSP Ceiling",
        delta_color="off",
        border=True,
        help="Share of province-months where local headline inflation exceeded the 4.0% BSP upper target.",
    )
    c5.metric(
        "Real Value of ₱1.00 (2018)",
        f"₱{kpi['mean_ppp']:.2f}",
        f"-₱{(1.0 - float(kpi['mean_ppp'])):.2f} eroded",
        delta_color="inverse",
        border=True,
        help="Purchasing Power of the Peso (2018=1.00), measuring how much 2018 purchasing value remains.",
    )

    # ------------------------------------------------------------------
    # Section Navigation Bar
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

    audit_queries: list[tuple[str, int, str]] = [("Global KPI Aggregation", 1, kpi_sql)]

    # ==================================================================
    # SECTION 1: NATIONAL VS. LOCAL REALITY
    # ==================================================================
    if active_idx == 0:
        st.subheader("National Cycles, Regional Gaps & Low-Income Exposure")
        st.caption(
            "Headline inflation peaked at 8.7% in early 2023, eased to 0.9% in July 2025, and rebounded to 7.8% in "
            "October 2026. Beneath that national cycle sit wide regional disparities and a regressive food burden."
        )

        ts_sql = f"""
            SELECT
                obs_date,
                ROUND(AVG(headline_yoy_pct), 2) AS Headline_YoY,
                ROUND(AVG(b30_headline_yoy_pct), 2) AS Bottom30_YoY,
                ROUND(AVG(natl_core_yoy_pct), 2) AS Natl_Core_YoY,
                ROUND(AVG(rice_cpi_yoy_pct), 2) AS Rice_YoY
            FROM prov_panel {where_sql}
            GROUP BY obs_date ORDER BY obs_date
        """
        ts_df = con.execute(ts_sql).df()
        audit_queries.append(("Monthly Inflation Timeline", len(ts_df), ts_sql))

        fig_ts = go.Figure()
        fig_ts.add_hrect(
            y0=2.0, y1=4.0,
            fillcolor="rgba(5,150,105,0.12)", line_width=0,
            annotation_text="BSP Target Band (2.0%–4.0%)",
            annotation_position="top left",
        )
        for col, label, color, dash in [
            ("Headline_YoY", "Headline CPI (All Income)", "#0F172A", "solid"),
            ("Bottom30_YoY", "Bottom-30% Low-Income CPI", "#DC2626", "dash"),
            ("Natl_Core_YoY", "Core CPI (Excludes Food & Energy)", "#2563EB", "dot"),
            ("Rice_YoY", "Provincial Rice CPI", "#D97706", "solid"),
        ]:
            fig_ts.add_trace(go.Scatter(
                x=ts_df["obs_date"], y=ts_df[col], mode="lines", name=label,
                line=dict(color=color, width=2.3, dash=dash),
            ))
        fig_ts.update_layout(
            title="Monthly Headline, Core, Bottom-30% & Rice Inflation Against the BSP Target",
            height=385, margin=dict(t=45, b=20), yaxis_title="Year-on-Year Inflation (%)", hovermode="x unified",
        )
        st.plotly_chart(fig_ts, width="stretch")
        st.caption(
            "Amber rice surges above `20%` in 2024 pull the red dashed Bottom-30% curve consistently above overall "
            "headline inflation, while non-food Core CPI remains comparatively stable."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            dist_sql = f"""
                SELECT ROUND(headline_yoy_pct, 1) AS bin_yoy, COUNT(*) AS cnt
                FROM prov_panel {where_sql}
                GROUP BY 1 ORDER BY 1
            """
            dist_df = con.execute(dist_sql).df()
            audit_queries.append(("Binned Distribution Histogram", len(dist_df), dist_sql))
            tukey_hi = float(summary.get("eda", {}).get("tukey_upper_fence", 11.25))
            fig_hist = px.bar(
                dist_df, x="bin_yoy", y="cnt",
                title="Provincial Inflation Distribution & Tail Extremes",
                labels={"bin_yoy": "Headline YoY Inflation (%)", "cnt": "Province-Month Count"},
                color_discrete_sequence=["#1D4ED8"],
            )
            fig_hist.add_vline(x=4.0, line_dash="dash", line_color="#DC2626", annotation_text="BSP 4%")
            fig_hist.add_vline(
                x=tukey_hi, line_dash="dot", line_color="#D97706",
                annotation_text=f"Tukey Fence ({tukey_hi}%)",
            )
            fig_hist.update_layout(height=365, margin=dict(t=45, b=20))
            st.plotly_chart(fig_hist, width="stretch")
            st.caption(
                f"Positively skewed (`Skew = 0.50`, `Kurtosis = 3.25`); readings past the `{tukey_hi}%` Tukey upper "
                "fence reflect acute island and commodity supply shocks."
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
            audit_queries.append(("Regional Median & Inequality Bar", len(reg_box), reg_sql_q))
            fig_bar = px.bar(
                reg_box, x="median_yoy", y="parent_region", color="avg_b30_spread", orientation="h",
                color_continuous_scale="Viridis",
                title="Regional Median Inflation & Bottom-30% Spread",
                labels={"median_yoy": "Median YoY (%)", "parent_region": "", "avg_b30_spread": "B30 Gap (pp)"},
            )
            fig_bar.add_vline(x=4.0, line_dash="dash", line_color="#DC2626")
            fig_bar.update_layout(height=365, margin=dict(t=45, b=20))
            st.plotly_chart(fig_bar, width="stretch")
            st.caption(
                "Bicol and MIMAROPA run substantially hotter than NCR, with brighter Viridis shading marking steeper "
                "extra inflation on low-income households."
            )

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
            audit_queries.append(("Provincial Rice vs Headline Scatter", len(scat_df), scat_sql))
            fig_scat = px.scatter(
                scat_df, x="rice_retail_yoy", y="headline_yoy", color="island_group",
                size="rice_wt_pct", hover_name="clean_geo",
                color_discrete_sequence=px.colors.qualitative.Safe,
                title="Retail Rice Price Growth vs. Headline CPI by Province",
                labels={
                    "rice_retail_yoy": "Mean Retail Rice Price YoY (%)",
                    "headline_yoy": "Mean Headline Inflation (%)",
                    "island_group": "Island Group",
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
                    name="OLS Trend (r = 0.45)", line=dict(color="#0F172A", width=2, dash="dash"),
                ))
            fig_scat.update_layout(height=365, margin=dict(t=45, b=20))
            st.plotly_chart(fig_scat, width="stretch")
            st.caption(
                "Each point represents one province sized by its rice expenditure weight, confirming clear retail "
                "rice pass-through into local headline inflation (`r = +0.45`)."
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
            audit_queries.append(("Household Basket Share Donuts", 1, bsk_sql))
            labels_b = ["Rice", "Other Food & Drinks", "Housing, Power, Fuel & Other"]
            colors_b = ["#D97706", "#0D9488", "#334155"]
            fig_pie = make_subplots(
                rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                subplot_titles=["All-Income Basket", "Bottom-30% Basket"],
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
                title="Expenditure Weights: All-Income vs. Bottom-30% Households",
                height=365, margin=dict(t=55, b=20), legend=dict(orientation="h", y=-0.08),
            )
            st.plotly_chart(fig_pie, width="stretch")
            st.caption(
                f"Bottom-30% households spend `{bsk['b30_rice']:.1f}%` of their total budget on rice versus "
                f"`{bsk['gen_rice']:.1f}%` overall, explaining why staple price spikes hit low-income families hardest."
            )

        corr_sql = f"""
            SELECT
                headline_yoy_pct AS Headline,
                b30_headline_yoy_pct AS Bottom30,
                food_bev_yoy_pct AS Food,
                rice_cpi_yoy_pct AS Rice,
                elec_gas_fuel_yoy_pct AS Utilities,
                transport_yoy_pct AS Transport,
                gwpi_all_yoy_lag1 AS Wholesale_Lag1
            FROM prov_panel {where_sql}
            USING SAMPLE 950 ROWS (reservoir, 42)
        """
        corr_sample = con.execute(corr_sql).df()
        audit_queries.append(("Driver Correlation Matrix Sample", len(corr_sample), corr_sql))
        corr_mat = corr_sample.corr(numeric_only=True).round(2)
        fig_corr = px.imshow(
            corr_mat, text_auto=True, color_continuous_scale="Viridis", zmin=-0.2, zmax=1.0,
            title="Cross-Driver Correlation Matrix: Upstream Wholesale, Staples & Headline CPI",
            aspect="auto",
        )
        fig_corr.update_layout(height=335, margin=dict(t=45, b=15))
        st.plotly_chart(fig_corr, width="stretch")

        if summary:
            ht = summary.get("hypothesis_tests", {})
            eda = summary.get("eda", {})
            st.markdown("##### Statistical Diagnostics")
            s1, s2, s3, s4 = st.columns(4)
            s1.metric(
                "Regional One-Way ANOVA",
                f"F = {ht.get('H1_regional_anova_F')}",
                "p < 0.001",
                border=True,
                help="Confirms statistically significant inflation differences across the 18 regions.",
            )
            s2.metric(
                "Bottom-30% Spread (All Months)",
                f"+{ht.get('H2_b30_vs_all_mean_diff_pp')} pp",
                f"t = {ht.get('H2_b30_paired_t')}",
                delta_color="inverse",
                border=True,
                help="Paired t-test showing low-income households face systematically higher monthly inflation.",
            )
            s3.metric(
                "Bottom-30% Spread (Food >5%)",
                f"+{ht.get('H2_high_food_b30_diff_pp')} pp",
                f"t = {ht.get('H2_high_food_paired_t')}",
                delta_color="inverse",
                border=True,
                help="During high food-inflation regimes above 5%, the low-income inflation penalty widens nearly sixfold.",
            )
            s4.metric(
                "Lagged Wholesale Correlation",
                f"r = +{ht.get('H3_gwpi_lag1_pearson_r')}",
                f"Tukey Fence: {eda.get('tukey_upper_fence')}%",
                border=True,
                help="Prior-month island Wholesale Price Index growth leads next-month provincial CPI.",
            )

        render_section_footer(
            step_num=1,
            summary_line=(
                "Regional medians diverge significantly across the country, and staple food surges widen the "
                "low-income inflation gap nearly sixfold. Next, examine where specific provincial hotspots cluster "
                "and how cost pressures transmit across spending categories."
            ),
            next_idx=1,
            next_label="Provincial Hotspots & Flows →",
        )

    # ==================================================================
    # SECTION 2: PROVINCIAL HOTSPOTS & FLOWS
    # ==================================================================
    elif active_idx == 1:
        st.subheader("Subnational Hotspots & Supply-to-Basket Transmission")
        st.caption(
            "Coordinate geography, basket hierarchies, and multi-channel cost flows across all 101 provinces and "
            "highly urbanized cities."
        )

        geo_sql = f"""
            SELECT
                clean_geo, parent_region, island_group, latitude, longitude,
                ROUND(AVG(headline_yoy_pct), 2) AS mean_yoy,
                ROUND(AVG(b30_inflation_spread_pp), 2) AS mean_b30_spread,
                ROUND(AVG(is_breach_bsp_4pct) * 100.0, 1) AS breach_freq_pct,
                ROUND(AVG(food_bev_yoy_pct), 2) AS mean_food_yoy,
                ROUND(AVG(rice_cpi_yoy_pct), 2) AS mean_rice_yoy,
                ROUND(AVG(elec_gas_fuel_yoy_pct), 2) AS mean_util_yoy,
                ROUND(AVG(transport_yoy_pct), 2) AS mean_trans_yoy,
                ROUND(MIN(ppp_index), 2) AS min_ppp
            FROM prov_panel {where_sql}
            GROUP BY 1, 2, 3, 4, 5
            HAVING latitude IS NOT NULL
        """
        geo_agg = con.execute(geo_sql).df()
        audit_queries.append(("Provincial Geospatial & Parallel Summary", len(geo_agg), geo_sql))

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
                    "min_ppp": ":.2f",
                    "latitude": False,
                    "longitude": False,
                },
                color_continuous_scale="Viridis", zoom=4.6, center={"lat": 12.8, "lon": 122.2},
                map_style="carto-positron",
                title="Provincial Mean Inflation & >4.0% Breach Frequency",
                labels={"mean_yoy": "Mean YoY (%)", "breach_freq_pct": "Breach %"},
            )
            fig_geo.update_layout(height=460, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig_geo, width="stretch")
            st.caption(
                "Bubble size encodes `>4.0%` breach frequency; outer-island shipping corridors such as Batanes "
                "(`5.25%`), Dinagat Islands (`5.07%`), and Occidental Mindoro face the most persistent pressure."
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
            audit_queries.append(("Hierarchical Sunburst Decomposition", len(sun_df), sun_sql))
            fig_sun = px.sunburst(
                sun_df, path=["island_group", "parent_region", "coicop_name"],
                values="abs_pp_contrib", color="avg_yoy", color_continuous_scale="Viridis",
                title="Regional & Spending-Division Contribution Hierarchy",
                labels={"avg_yoy": "Category YoY (%)"},
            )
            fig_sun.update_layout(height=460, margin=dict(t=40, b=10, l=0, r=0))
            st.plotly_chart(fig_sun, width="stretch")
            st.caption("Select any island or region ring to drill down into local COICOP category contributions.")

        c_san, c_par = st.columns(2)
        with c_san:
            san_sql = f"""
                SELECT
                    CASE
                        WHEN coicop_code IN ('01') THEN 'Food Supply & Rice'
                        WHEN coicop_code IN ('04','07') THEN 'Utilities & Transport Fuel'
                        ELSE 'Core Goods & Services'
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
            audit_queries.append(("Sankey Supply Transmission", len(san_df), san_sql))
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
                title="Cost-Push Transmission: Supply Channels to Basket Divisions to Islands",
                height=430, margin=dict(t=40, b=10),
            )
            st.plotly_chart(fig_san, width="stretch")
            st.caption(
                "Food and rice (`48%`) alongside housing utilities and transport (`28%`) generate over three-quarters "
                "of cumulative subnational inflation pressure."
            )

        with c_par:
            fig_par = px.parallel_coordinates(
                geo_agg,
                dimensions=[
                    "mean_yoy", "mean_food_yoy", "mean_rice_yoy",
                    "mean_util_yoy", "mean_trans_yoy", "mean_b30_spread",
                ],
                color="mean_yoy", color_continuous_scale="Viridis",
                labels={
                    "mean_yoy": "Headline %",
                    "mean_food_yoy": "Food %",
                    "mean_rice_yoy": "Rice %",
                    "mean_util_yoy": "Power/Fuel %",
                    "mean_trans_yoy": "Transport %",
                    "mean_b30_spread": "B30 Gap (pp)",
                },
                title="Multi-Driver Provincial Risk Profiles (101 Units)",
            )
            fig_par.update_layout(height=430, margin=dict(t=45, b=10))
            st.plotly_chart(fig_par, width="stretch")
            st.caption(
                "Drag vertically along any axis to filter provinces experiencing simultaneous rice, utility, and "
                "low-income spread pressure."
            )

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
        audit_queries.append(("Stream-Style Category Contributions", len(st_df), st_sql))
        fig_stream = px.area(
            st_df, x="obs_date", y="avg_contrib_pp", color="coicop_name",
            color_discrete_sequence=px.colors.qualitative.Safe,
            title="Category Percentage-Point Contributions Over Time (2019–2026)",
            labels={"avg_contrib_pp": "Weighted Contribution (pp)", "obs_date": "", "coicop_name": "Division"},
        )
        fig_stream.update_layout(height=360, margin=dict(t=40, b=10), hovermode="x unified")
        st.plotly_chart(fig_stream, width="stretch")
        st.caption(
            "Food drove the 2023 surge, contracted sharply during the 2025 disinflation, and accelerated again "
            "alongside transport in 2026."
        )

        render_section_footer(
            step_num=2,
            summary_line=(
                "Outer islands and food-intensive provinces absorb the most frequent target breaches. "
                "Next, evaluate out-of-time forecast accuracy, prediction intervals, and early breach alerts."
            ),
            prev_idx=0,
            prev_label="← National vs. Local Reality",
            next_idx=2,
            next_label="Forecasts & Breach Risk →",
        )

    # ==================================================================
    # SECTION 3: FORECASTS & BREACH RISK
    # ==================================================================
    elif active_idx == 2:
        st.subheader("Out-of-Time Forecasting, Uncertainty Bands & Feature Attribution")
        st.caption(
            "All models train strictly on 2020–2024 and are evaluated blind across 2,121 provincial observations from "
            "Jan 2025 to Sep 2026, covering both the 0.9% trough and the 7.8% rebound."
        )

        lb_col1, lb_col2 = st.columns([1.35, 1.0])
        with lb_col1:
            st.markdown("##### `2025–2026` Out-of-Time Holdout Leaderboard")
            if summary and "regression_oot_2025_2026" in summary:
                reg_lb = pd.DataFrame(summary["regression_oot_2025_2026"]).T.reset_index().rename(
                    columns={
                        "index": "Model",
                        "CV_MAE": "2020–24 CV MAE (pp)",
                        "MAE": "Holdout MAE (pp)",
                        "RMSE": "Holdout RMSE (pp)",
                        "R2": "Holdout R²",
                    }
                )
                st.dataframe(reg_lb, width="stretch", hide_index=True)
        with lb_col2:
            st.info(
                "**Regime-Shift Takeaway:** Tree ensembles clip predictions inside historical `2020–2024` leaf bounds "
                "when inflation reverses sharply. **Ridge Regression** (`0.93 pp` MAE, `0.874` R²) extrapolates "
                "cleanly across turning points, while **XGBoost Classifier** flags `>4.0%` breaches at `92.6%` "
                "accuracy (`0.976` ROC-AUC)."
            )

        f_top1, f_top2 = st.columns([1.6, 1.4])
        with f_top1:
            sel_fc_geo = st.selectbox(
                "National or Regional Series (SARIMAX Fan Chart)",
                ["PHILIPPINES"] + regs_all,
                help="Fits a seasonal SARIMAX(1,0,1)(1,0,1,12) model with 80% and 95% confidence bands.",
            )
        with f_top2:
            st.caption(
                f"Active horizon: `{horizon_mo}` months (adjust **SARIMAX Horizon** in the sidebar to inspect "
                "interval widening)."
            )

        fan_sql = f"""
            SELECT obs_date, headline_yoy_pct
            FROM reg_monthly
            WHERE region = '{sel_fc_geo}' AND headline_yoy_pct IS NOT NULL
            ORDER BY obs_date
        """
        geo_ts = con.execute(fan_sql).df()
        audit_queries.append(("Regional SARIMAX Time Series", len(geo_ts), fan_sql))
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
            annotation_text="BSP 2%–4% Target", annotation_position="top left",
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
            name=f"SARIMAX {horizon_mo}-Mo Forecast", line=dict(color="#EA580C", width=2.5),
        ))
        fig_fan.update_layout(
            height=380,
            title=f"SARIMAX Forecast & Prediction Intervals: {sel_fc_geo} ({horizon_mo}-Month Horizon, AIC = {sar_fit.aic:.1f})",
            yaxis_title="Year-on-Year Inflation (%)", margin=dict(t=45, b=20), hovermode="x unified",
        )
        st.plotly_chart(fig_fan, width="stretch")

        c_xai, c_roc = st.columns(2)
        with c_xai:
            if SHAP_CSV.exists():
                sh_df = pd.read_csv(SHAP_CSV).head(10).iloc[::-1].copy()
                sh_df["readable_feature"] = sh_df["feature"].map(humanize_feature)
                fig_sh = px.bar(
                    sh_df, x="mean_abs_shap", y="readable_feature", orientation="h",
                    hover_data={"feature": True, "mean_abs_shap": ":.3f", "readable_feature": False},
                    title="TreeSHAP Feature Attribution (Next-Month Forecast)",
                    color_discrete_sequence=["#0D9488"],
                )
                fig_sh.update_layout(
                    height=380, yaxis_title="", xaxis_title="Mean |SHAP| Contribution (pp)",
                    margin=dict(t=45, b=20),
                )
                st.plotly_chart(fig_sh, width="stretch")
                st.caption(
                    "Recent inflation momentum leads attribution, reinforced by national Core CPI, seasonal harvest "
                    "cycles, lagged island Wholesale GWPI, and transport fuel."
                )

        with c_roc:
            y_true_c = eng["y_te_clf"]
            probs_c = eng["probs_te"]
            pred_c = (probs_c >= tau).astype(int)
            cm = confusion_matrix(y_true_c, pred_c)
            tn, fp, fn, tp = cm.ravel()
            prec_v = tp / max(tp + fp, 1)
            rec_v = tp / max(tp + fn, 1)
            auc_v = roc_auc_score(y_true_c, probs_c)
            fpr, tpr, thresh_arr = roc_curve(y_true_c, probs_c)
            idx_tau = int(np.argmin(np.abs(thresh_arr - tau)))

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"XGBoost >4% Classifier (AUC = {auc_v:.3f})", line=dict(color="#1D4ED8", width=2.6),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[fpr[idx_tau]], y=[tpr[idx_tau]], mode="markers",
                name=f"Active Cutoff (τ = {tau:.2f})",
                marker=dict(color="#DC2626", size=12, symbol="star"),
            ))
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline",
                line=dict(dash="dash", color="#64748B"),
            ))
            fig_roc.update_layout(
                height=290, title="ROC Diagnostic: >4.0% BSP Breach Classifier",
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                margin=dict(t=45, b=20),
            )
            st.plotly_chart(fig_roc, width="stretch")
            st.caption(
                f"**Cutoff τ = `{tau:.2f}`:** Caught Breaches `{tp}` · False Alarms `{fp}` · Missed `{fn}` · "
                f"Safe `{tn}` (`Precision = {prec_v:.1%}`, `Recall = {rec_v:.1%}`)"
            )

        st.divider()
        st.markdown("##### Holdout Error Diagnostics & Cross-Filtered Trajectory")
        st.caption(
            "Inspect the largest `2025–2026` provincial prediction misses on the left to view observed versus "
            "forecast trajectories and shock months on the right."
        )
        err_df = eng["te_meta"].copy()
        err_df["predicted_yoy_t1"] = np.round(eng["preds_te"], 2)
        err_df["abs_error_pp"] = np.round(np.abs(err_df["target_yoy_t1"] - err_df["predicted_yoy_t1"]), 2)
        top_err = err_df.sort_values("abs_error_pp", ascending=False).head(10)

        ce1, ce2 = st.columns([1.05, 1.35])
        with ce1:
            st.dataframe(
                top_err[["clean_geo", "parent_region", "obs_date", "target_yoy_t1", "predicted_yoy_t1", "abs_error_pp"]].rename(
                    columns={
                        "clean_geo": "Province / HUC",
                        "parent_region": "Region",
                        "obs_date": "Month",
                        "target_yoy_t1": "Actual %",
                        "predicted_yoy_t1": "Predicted %",
                        "abs_error_pp": "Error (pp)",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
        with ce2:
            err_geos = list(dict.fromkeys(top_err["clean_geo"].tolist()))
            sel_err_geo = st.selectbox(
                "Province Holdout Trajectory",
                err_geos,
                index=0,
                help="Highlights holdout months where abrupt local price jumps produced forecast errors of 2.0 pp or more.",
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
                    name="Shock Month (>=2 pp error)",
                    marker=dict(color="#DC2626", size=11, symbol="diamond"),
                ))
            fig_err.update_layout(
                height=290, title=f"Observed vs. Predicted Holdout Series: {sel_err_geo}",
                yaxis_title="YoY Inflation (%)", margin=dict(t=40, b=15),
            )
            st.plotly_chart(fig_err, width="stretch")
            st.button(
                f"Simulate Supply Shocks for {sel_err_geo} →",
                key="btn_handoff_err_prov",
                on_click=go_to_section,
                args=(3, sel_err_geo),
                width="stretch",
            )

        render_section_footer(
            step_num=3,
            summary_line=(
                "Statistical lag models capture momentum accurately, whereas sudden typhoon, tariff, or fuel shocks "
                "create sharp single-month jumps that require forward scenario stress-testing."
            ),
            prev_idx=1,
            prev_label="← Provincial Hotspots & Flows",
            next_idx=3,
            next_label="Supply-Shock Simulator →",
        )

    # ==================================================================
    # SECTION 4: SUPPLY-SHOCK SIMULATOR
    # ==================================================================
    else:
        st.subheader("Supply-Shock & Low-Income Equity Simulator")
        st.caption(
            "Test how sudden rice, electricity, fuel, and wholesale price shifts pass through local provincial "
            "baskets, next-month ML forecasts, and Bottom-30% household budgets."
        )

        st.markdown("##### Scenario Presets")
        p1, p2, p3, p4 = st.columns(4)
        p1.button(
            "Rice Supply Spike (+12 pp)",
            on_click=apply_shock_preset, args=(12.0, 2.0, 2.0, 3.0), width="stretch",
            help="Applies a sharp domestic harvest shortfall or import bottleneck on retail rice prices.",
        )
        p2.button(
            "Energy & Fuel Surge",
            on_click=apply_shock_preset, args=(3.0, 9.0, 11.0, 6.0), width="stretch",
            help="Applies concurrent electricity, LPG, and diesel transport rate increases.",
        )
        p3.button(
            "Tariff & Buffer Relief",
            on_click=apply_shock_preset, args=(-8.0, -2.0, -4.0, -2.0), width="stretch",
            help="Applies Kadiwa/NFA buffer-stock releases, tariff reductions, and pump fuel rollbacks.",
        )
        p4.button(
            "Reset Baseline Shock",
            on_click=apply_shock_preset, args=(8.0, 5.0, 6.0, 3.0), width="stretch",
            help="Restores the default benchmark shock (+8 pp Rice, +5 pp Utilities, +6 pp Fuel, +3 pp GWPI).",
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
                help="Primary province or HUC evaluated using its official PSA basket weights and latest observations.",
            )
        default_cmp = "Camarines Sur" if sel_prov != "Camarines Sur" and "Camarines Sur" in provs else (
            "City of Baguio" if sel_prov != "City of Baguio" and "City of Baguio" in provs else provs[0]
        )
        with sel_col2:
            cmp_prov = st.selectbox(
                "Benchmark Comparison Province",
                provs,
                index=provs.index(default_cmp),
                help="Contrasts how the identical price shock translates into a second province's local basket.",
            )

        s1, s2, s3, s4 = st.columns(4)
        rice_shock = s1.slider(
            "Rice Price Shock (Δ pp)", -15.0, 25.0, key="sim_rice", step=0.5,
            help="Percentage-point change in local year-on-year rice inflation.",
        )
        util_shock = s2.slider(
            "Electricity & LPG Shock (Δ pp)", -15.0, 25.0, key="sim_util", step=0.5,
            help="Percentage-point shift in household electricity, gas, and cooking fuel inflation.",
        )
        fuel_shock = s3.slider(
            "Diesel & Pump Fuel Shock (Δ pp)", -15.0, 25.0, key="sim_fuel", step=0.5,
            help="Percentage-point pump fuel shock with 45% pass-through into public transport fares.",
        )
        gwpi_shock = s4.slider(
            "Wholesale GWPI Spillover (Δ pp)", -10.0, 20.0, key="sim_gwpi", step=0.5,
            help="Upstream island wholesale inflation shift feeding into next-month retail prices.",
        )

        res_main = compute_province_shock(con, eng, sel_prov, rice_shock, util_shock, fuel_shock, gwpi_shock)
        res_cmp = compute_province_shock(con, eng, cmp_prov, rice_shock, util_shock, fuel_shock, gwpi_shock)
        audit_queries.append((
            f"Simulator Province History ({sel_prov} & {cmp_prov})",
            len(res_main["prov_hist"]) + len(res_cmp["prov_hist"]),
            f"SELECT * FROM prov_panel WHERE clean_geo IN ('{sel_prov}', '{cmp_prov}') ORDER BY obs_date DESC LIMIT 18",
        ))

        prov_hist = res_main["prov_hist"]
        latest_row = res_main["latest_row"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Baseline Next-Month Forecast",
            f"{res_main['base_pred']:.2f}%",
            f"Latest ({latest_row['obs_date'].iloc[0].strftime('%b %Y')}): {float(latest_row['headline_yoy_pct'].iloc[0]):.2f}%",
            delta_color="off",
            border=True,
            help="Unshocked Ridge Regression forecast for next month.",
        )
        m2.metric(
            "Shocked Ridge Forecast (t+1)",
            f"{res_main['sim_pred']:.2f}%",
            f"{res_main['sim_pred'] - res_main['base_pred']:+.2f} pp (XGB: {res_main['xgb_sim_pred']:.2f}%)",
            delta_color="inverse",
            border=True,
            help=f"Post-shock forecast; 80% Quantile interval spans {res_main['sim_q10']:.2f}% to {res_main['sim_q90']:.2f}%.",
        )
        m3.metric(
            "Direct Basket Pass-Through",
            f"{res_main['direct_acct_pp']:+.2f} pp",
            f"Bottom-30% Extra Rice Hit: {res_main['b30_extra_pp']:+.2f} pp",
            delta_color="inverse",
            border=True,
            help=(
                f"Direct accounting shift from {sel_prov}'s 2018 weights ({res_main['w_rice_pct']:.1f}% general rice "
                f"share vs {res_main['b30_rice_w_pct']:.1f}% Bottom-30% rice share)."
            ),
        )
        m4.metric(
            ">4.0% Breach Probability",
            f"{res_main['sim_prob']:.1%}",
            f"{(res_main['sim_prob'] - res_main['base_prob']) * 100:+.1f} pp vs Baseline",
            delta_color="inverse",
            border=True,
            help="XGBoost Classifier probability of exceeding the 4.0% BSP target ceiling next month.",
        )

        v_left, v_right = st.columns([1.45, 1.0])
        with v_left:
            next_dt = prov_hist["obs_date"].max() + pd.DateOffset(months=1)
            fig_sim = go.Figure()
            fig_sim.add_hrect(
                y0=2.0, y1=4.0, fillcolor="rgba(5,150,105,0.11)", line_width=0,
                annotation_text="BSP 2%–4% Target", annotation_position="top left",
            )
            fig_sim.add_trace(go.Scatter(
                x=prov_hist["obs_date"], y=prov_hist["headline_yoy_pct"], mode="lines+markers",
                name="Observed YoY (%)", line=dict(color="#0F172A", width=2.3),
            ))
            fig_sim.add_trace(go.Scatter(
                x=[prov_hist["obs_date"].max(), next_dt],
                y=[prov_hist["headline_yoy_pct"].iloc[-1], res_main["base_pred"]],
                mode="lines+markers", name="Baseline Forecast",
                line=dict(color="#2563EB", dash="dot", width=2.3),
            ))
            fig_sim.add_trace(go.Scatter(
                x=[prov_hist["obs_date"].max(), next_dt],
                y=[prov_hist["headline_yoy_pct"].iloc[-1], res_main["sim_pred"]],
                mode="lines+markers", name="Shock Scenario Forecast",
                line=dict(color="#DC2626", width=2.7),
            ))
            fig_sim.add_trace(go.Scatter(
                x=[next_dt, next_dt], y=[res_main["sim_q10"], res_main["sim_q90"]],
                mode="lines", name="Simulated 80% Interval",
                line=dict(color="#DC2626", width=8), opacity=0.35,
            ))
            fig_sim.update_layout(
                height=360,
                title=f"Simulated Trajectory & 80% Interval: {sel_prov} ({latest_row['parent_region'].iloc[0]})",
                yaxis_title="YoY Inflation (%)", margin=dict(t=45, b=20), hovermode="x unified",
            )
            st.plotly_chart(fig_sim, width="stretch")

        with v_right:
            cmp_df = pd.DataFrame([
                {"Province": sel_prov, "Channel": "General Basket Impact", "Impact_pp": round(res_main["direct_acct_pp"], 2)},
                {"Province": sel_prov, "Channel": "Bottom-30% Extra Rice Burden", "Impact_pp": round(res_main["b30_extra_pp"], 2)},
                {"Province": cmp_prov, "Channel": "General Basket Impact", "Impact_pp": round(res_cmp["direct_acct_pp"], 2)},
                {"Province": cmp_prov, "Channel": "Bottom-30% Extra Rice Burden", "Impact_pp": round(res_cmp["b30_extra_pp"], 2)},
            ])
            fig_cmp = px.bar(
                cmp_df, x="Province", y="Impact_pp", color="Channel", barmode="group",
                color_discrete_sequence=["#1D4ED8", "#D97706"],
                title="Cross-Province Basket Sensitivity",
                labels={"Impact_pp": "Direct Price Shift (pp)", "Province": ""},
            )
            fig_cmp.update_layout(height=360, margin=dict(t=45, b=20), legend=dict(orientation="h", y=-0.18))
            st.plotly_chart(fig_cmp, width="stretch")

        if res_main["sim_prob"] >= 0.50 or res_main["sim_pred"] > 4.0:
            st.warning(
                f"**Policy Risk Advisory ({sel_prov}):** Projected next-month inflation reaches "
                f"**`{res_main['sim_pred']:.2f}%`** (`{res_main['sim_prob']:.1%}` breach probability), with Bottom-30% "
                f"households absorbing an additional **`{res_main['b30_extra_pp']:+.2f} pp`** rice impact "
                f"(`{res_main['b30_rice_w_pct']:.1f}%` low-income rice share vs `{res_main['w_rice_pct']:.1f}%` general). "
                "Priority response: local Price Coordinating Council monitoring, Kadiwa buffer-stock releases, and "
                "targeted food or transport relief."
            )
        else:
            st.success(
                f"**Stability Assessment ({sel_prov}):** Projected next-month inflation remains at "
                f"**`{res_main['sim_pred']:.2f}%`** (`{res_main['sim_prob']:.1%}` breach probability), holding "
                "within or below the `4.0%` BSP target ceiling."
            )

        render_section_footer(
            step_num=4,
            summary_line=(
                "Local basket weights govern how national supply shocks translate into provincial breach probability "
                "and low-income household strain."
            ),
            prev_idx=2,
            prev_label="← Forecasts & Breach Risk",
            next_idx=0,
            next_label="Back to National vs. Local Reality",
            next_primary=False,
        )

    # ------------------------------------------------------------------
    # DuckDB SQL Audit Expander
    # ------------------------------------------------------------------
    st.markdown("")
    with st.expander("DuckDB SQL Audit Trail (<1,000 Aggregated Rows per View)"):
        st.caption(
            "All charts execute parameterized SQL inside DuckDB over `2.2 MB` ZSTD Parquet marts so strictly fewer "
            "than `1,000` aggregated rows pass to Plotly."
        )
        for q_title, row_cnt, q_sql in audit_queries:
            st.markdown(f"**{q_title}** · `{row_cnt:,}` rows")
            st.code(q_sql.strip(), language="sql")


if __name__ == "__main__":
    main()
