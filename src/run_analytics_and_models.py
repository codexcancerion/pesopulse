"""
PesoPulse — MIT 118 Final Group Project (Team 3):
Exploratory Diagnostics, Statistical Hypothesis Tests, Time-Series CV ML & SARIMAX,
Exact TreeSHAP Explainability, and 12 Publication-Grade Figures (300 DPI).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "navy": "#0F172A",
    "blue": "#1D4ED8",
    "teal": "#0D9488",
    "amber": "#D97706",
    "red": "#DC2626",
    "emerald": "#059669",
    "purple": "#7C3AED",
    "slate": "#64748B",
    "sky": "#0284C7",
    "orange": "#EA580C",
}

sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 0.9,
    "figure.facecolor": "white",
})


def _draw_bezier_ribbon(ax, x0, x1, y0_bot, y0_top, y1_bot, y1_top, color, alpha=0.45):
    dx = (x1 - x0) * 0.45
    verts = [
        (x0, y0_bot),
        (x0 + dx, y0_bot), (x1 - dx, y1_bot), (x1, y1_bot),
        (x1, y1_top),
        (x1 - dx, y1_top), (x0 + dx, y0_top), (x0, y0_top),
        (x0, y0_bot),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", alpha=alpha))


def run_all() -> dict:
    panel_path = DATA_DIR / "cpi_provincial_ml_panel.parquet"
    div_path = DATA_DIR / "cpi_division_monthly.parquet"
    reg_path = DATA_DIR / "cpi_regional_monthly.csv"

    con = duckdb.connect(":memory:")
    panel = con.execute(f"SELECT * FROM read_parquet('{str(panel_path).replace(chr(92), '/')}')").df()
    panel["obs_date"] = pd.to_datetime(panel["obs_date"])

    div_df = con.execute(f"SELECT * FROM read_parquet('{str(div_path).replace(chr(92), '/')}')").df()
    div_df["obs_date"] = pd.to_datetime(div_df["obs_date"])

    reg_df = pd.read_csv(reg_path)
    reg_df["obs_date"] = pd.to_datetime(reg_df["obs_date"])

    # Filter to Jan 2019 - Oct 2026 where YoY inflation exists
    yoy_panel = panel.dropna(subset=["headline_yoy_pct"]).copy()

    # ------------------------------------------------------------------
    # 1. EDA & Statistical Hypothesis Tests
    # ------------------------------------------------------------------
    h_vals = yoy_panel["headline_yoy_pct"].dropna()
    q1, q3 = np.percentile(h_vals, [25, 75])
    iqr = q3 - q1
    tukey_lo, tukey_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr

    # H1: Island Group & Region ANOVA
    isl_groups = [g["headline_yoy_pct"].dropna().values for _, g in yoy_panel[yoy_panel["island_group"].isin(["Luzon", "Visayas", "Mindanao"])].groupby("island_group")]
    f_isl, p_isl = stats.f_oneway(*isl_groups)
    kw_isl, kp_isl = stats.kruskal(*isl_groups)

    reg_groups = [g["headline_yoy_pct"].dropna().values for _, g in yoy_panel.groupby("parent_region") if len(g) > 10]
    f_reg, p_reg = stats.f_oneway(*reg_groups)

    # H2: Paired t-test & Wilcoxon on Bottom-30% vs All-Income Inflation
    b30_pair = yoy_panel.dropna(subset=["headline_yoy_pct", "b30_headline_yoy_pct"])
    t_b30, p_b30 = stats.ttest_rel(b30_pair["b30_headline_yoy_pct"], b30_pair["headline_yoy_pct"])
    w_b30, wp_b30 = stats.wilcoxon(b30_pair["b30_headline_yoy_pct"], b30_pair["headline_yoy_pct"])

    high_food = b30_pair[b30_pair["food_bev_yoy_pct"] > 5.0]
    t_hf, p_hf = stats.ttest_rel(high_food["b30_headline_yoy_pct"], high_food["headline_yoy_pct"])

    # H3: Upstream cost-push correlations
    corr_gwpi_df = yoy_panel.dropna(subset=["gwpi_all_yoy_lag1", "target_yoy_t1"])
    r_gwpi, p_gwpi = stats.pearsonr(corr_gwpi_df["gwpi_all_yoy_lag1"], corr_gwpi_df["target_yoy_t1"])

    corr_rice_df = yoy_panel.dropna(subset=["retail_rice_wm_yoy_pct", "headline_yoy_pct"])
    r_rice, p_rice = stats.pearsonr(corr_rice_df["retail_rice_wm_yoy_pct"], corr_rice_df["headline_yoy_pct"])

    # ------------------------------------------------------------------
    # 2. Chronological ML & Time-Series Benchmarking
    # ------------------------------------------------------------------
    ml_df = yoy_panel.dropna(subset=["target_yoy_t1", "yoy_lag1", "yoy_lag3", "yoy_lag12"]).sort_values(["obs_date", "clean_geo"]).reset_index(drop=True)

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

    X = ml_df[num_feats + cat_feats].copy()
    y_reg = ml_df["target_yoy_t1"].values
    y_clf = ml_df["target_breach_4pct_t1"].astype(int).values

    # Chronological Split: Train = 2020-01 to 2024-12 | Test = 2025-01 to 2026-09 (Out-of-Time!)
    tr_mask = ml_df["year"] <= 2024
    te_mask = ml_df["year"] >= 2025
    X_tr, X_te = X[tr_mask], X[te_mask]
    y_tr, y_te = y_reg[tr_mask], y_reg[te_mask]
    yc_tr, yc_te = y_clf[tr_mask], y_clf[te_mask]

    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
    ])

    # Baseline Persistence: target_yoy_t1 = current headline_yoy_pct
    pred_persist = X_te["headline_yoy_pct"].values

    models_reg = {
        "Ridge Regression": Pipeline([("pre", pre), ("m", Ridge(alpha=10.0))]),
        "Random Forest": Pipeline([("pre", pre), ("m", RandomForestRegressor(n_estimators=220, max_depth=12, min_samples_leaf=4, random_state=42, n_jobs=-1))]),
        "XGBoost Regressor": Pipeline([("pre", pre), ("m", xgb.XGBRegressor(n_estimators=280, max_depth=5, learning_rate=0.05, subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1))]),
    }

    reg_results = {
        "Persistence Baseline": {
            "MAE": round(float(mean_absolute_error(y_te, pred_persist)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_te, pred_persist))), 4),
            "R2": round(float(r2_score(y_te, pred_persist)), 4),
        }
    }

    # Expanding-window TimeSeriesSplit CV on training years
    tscv = TimeSeriesSplit(n_splits=5)
    preds_te_map = {"Persistence Baseline": pred_persist}
    for name, pipe in models_reg.items():
        cv_maes = []
        for tr_i, va_i in tscv.split(X_tr):
            pipe.fit(X_tr.iloc[tr_i], y_tr[tr_i])
            p_va = pipe.predict(X_tr.iloc[va_i])
            cv_maes.append(mean_absolute_error(y_tr[va_i], p_va))
        pipe.fit(X_tr, y_tr)
        p_te = pipe.predict(X_te)
        preds_te_map[name] = p_te
        reg_results[name] = {
            "CV_MAE": round(float(np.mean(cv_maes)), 4),
            "MAE": round(float(mean_absolute_error(y_te, p_te)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_te, p_te))), 4),
            "R2": round(float(r2_score(y_te, p_te)), 4),
        }

    # Quantile Gradient Boosting for 80% (q=0.10, 0.90) & 95% (q=0.025, 0.975) Prediction Intervals
    X_tr_enc = pre.fit_transform(X_tr)
    X_te_enc = pre.transform(X_te)
    feat_names = num_feats + list(pre.named_transformers_["cat"].get_feature_names_out(cat_feats))

    q_preds = {}
    for q in [0.025, 0.10, 0.50, 0.90, 0.975]:
        gbr_q = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=180, max_depth=4, learning_rate=0.05, random_state=42)
        gbr_q.fit(X_tr_enc, y_tr)
        q_preds[q] = gbr_q.predict(X_te_enc)
    cov_80 = float(np.mean((y_te >= q_preds[0.10]) & (y_te <= q_preds[0.90])) * 100.0)
    cov_95 = float(np.mean((y_te >= q_preds[0.025]) & (y_te <= q_preds[0.975])) * 100.0)

    # Exact TreeSHAP values via native XGBoost pred_contribs=True
    xgb_fitted = models_reg["XGBoost Regressor"].named_steps["m"]
    contribs = xgb_fitted.get_booster().predict(xgb.DMatrix(X_te_enc, feature_names=feat_names), pred_contribs=True)
    shap_vals = contribs[:, :-1]  # last column is bias
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)

    # Classification benchmark for >4.0% BSP target breach
    models_clf = {
        "Logistic Regression": Pipeline([("pre", pre), ("m", LogisticRegression(max_iter=1000, random_state=42))]),
        "Random Forest Clf": Pipeline([("pre", pre), ("m", RandomForestClassifier(n_estimators=220, max_depth=10, random_state=42, n_jobs=-1))]),
        "XGBoost Classifier": Pipeline([("pre", pre), ("m", xgb.XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, random_state=42, eval_metric="logloss", n_jobs=-1))]),
    }
    clf_results = {}
    roc_curves = {}
    for name, pipe in models_clf.items():
        pipe.fit(X_tr, yc_tr)
        prob_te = pipe.predict_proba(X_te)[:, 1]
        pred_cls = (prob_te >= 0.5).astype(int)
        clf_results[name] = {
            "ROC_AUC": round(float(roc_auc_score(yc_te, prob_te)), 4),
            "PR_AUC": round(float(average_precision_score(yc_te, prob_te)), 4),
            "Accuracy": round(float(accuracy_score(yc_te, pred_cls)), 4),
            "F1": round(float(f1_score(yc_te, pred_cls)), 4),
            "Precision": round(float(precision_score(yc_te, pred_cls, zero_division=0)), 4),
            "Recall": round(float(recall_score(yc_te, pred_cls, zero_division=0)), 4),
        }
        fpr, tpr, _ = roc_curve(yc_te, prob_te)
        roc_curves[name] = (fpr, tpr, clf_results[name]["ROC_AUC"])

    # National SARIMAX Benchmark on Philippines Headline YoY
    natl = reg_df[reg_df["region"] == "PHILIPPINES"].dropna(subset=["headline_yoy_pct"]).sort_values("obs_date").set_index("obs_date")
    natl_tr = natl.loc[:"2024-12-01", "headline_yoy_pct"]
    natl_te = natl.loc["2025-01-01":, "headline_yoy_pct"]
    sarimax_fit = SARIMAX(natl_tr, order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sarimax_fc = sarimax_fit.get_forecast(steps=len(natl_te))
    sarimax_mae = float(mean_absolute_error(natl_te, sarimax_fc.predicted_mean))

    # Full-data SARIMAX 6-month forward forecast (Nov 2026 - Apr 2027)
    sarimax_full = SARIMAX(natl["headline_yoy_pct"], order=(1, 0, 1), seasonal_order=(1, 0, 1, 12), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fwd_fc = sarimax_full.get_forecast(steps=6)
    fwd_ci_80 = fwd_fc.conf_int(alpha=0.20)
    fwd_ci_95 = fwd_fc.conf_int(alpha=0.05)

    # Save out-of-time test predictions & SHAP summary for app / inspection
    te_export = ml_df.loc[te_mask, ["clean_geo", "parent_region", "island_group", "obs_date", "headline_yoy_pct", "target_yoy_t1", "target_breach_4pct_t1"]].copy()
    te_export["pred_xgb"] = preds_te_map["XGBoost Regressor"]
    te_export["pred_q10"] = q_preds[0.10]
    te_export["pred_q90"] = q_preds[0.90]
    te_export["pred_q025"] = q_preds[0.025]
    te_export["pred_q975"] = q_preds[0.975]
    te_export["prob_breach_4pct"] = models_clf["XGBoost Classifier"].predict_proba(X_te)[:, 1]
    te_export.to_csv(DATA_DIR / "out_of_time_predictions_2025_2026.csv", index=False)
    shap_df.to_csv(DATA_DIR / "treeshap_feature_importance.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Generate 12 Publication-Grade Figures (300 DPI)
    # ------------------------------------------------------------------
    # Fig 01: National Headline vs Core vs Bottom-30% Inflation
    natl_full = div_df[(div_df["clean_geo"] == "PHILIPPINES") & (div_df["coicop_code"] == "0") & (div_df["year"] >= 2019)].sort_values("obs_date")
    core_natl = yoy_panel.groupby("obs_date")["natl_core_yoy_pct"].first().reset_index()
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=300)
    ax.axhspan(2.0, 4.0, color=PALETTE["emerald"], alpha=0.10, label="BSP Target Band (2.0% - 4.0%)")
    ax.plot(natl_full["obs_date"], natl_full["official_yoy_pct"], color=PALETTE["navy"], lw=2.3, label="Headline Inflation (All Income)")
    ax.plot(natl_full["obs_date"], natl_full["b30_yoy_pct"], color=PALETTE["red"], lw=2.0, ls="--", label="Bottom 30% Income Households")
    ax.plot(core_natl["obs_date"], core_natl["natl_core_yoy_pct"], color=PALETTE["blue"], lw=1.9, ls="-.", label="Core Inflation (Excl. Volatile Food/Energy)")
    ax.set_title("Philippine Monthly Inflation Dynamics: Headline vs. Core vs. Bottom 30% (Jan 2019 - Oct 2026)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Year-on-Year Inflation (%)")
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_national_headline_vs_core_vs_b30.png", dpi=300)
    plt.close(fig)

    # Fig 02: Distribution + Skewness/Kurtosis + Tukey Fences
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=300)
    sns.histplot(h_vals, bins=45, kde=True, color=PALETTE["blue"], ax=ax1)
    ax1.axvline(h_vals.mean(), color=PALETTE["red"], ls="--", lw=1.8, label=f"Mean = {h_vals.mean():.2f}%")
    ax1.axvline(h_vals.median(), color=PALETTE["emerald"], ls="-", lw=1.8, label=f"Median = {h_vals.median():.2f}%")
    ax1.axvline(tukey_hi, color=PALETTE["amber"], ls=":", lw=1.8, label=f"Tukey Upper Fence = {tukey_hi:.2f}%")
    ax1.set_title(f"Provincial Headline YoY Distribution\nSkew={stats.skew(h_vals):.2f}, Kurtosis={stats.kurtosis(h_vals, fisher=False):.2f}", fontweight="bold")
    ax1.set_xlabel("YoY Inflation (%)")
    ax1.legend(fontsize=8)
    sns.boxplot(data=yoy_panel[yoy_panel["island_group"].isin(["Luzon", "Visayas", "Mindanao"])], x="island_group", y="headline_yoy_pct", palette=[PALETTE["blue"], PALETTE["teal"], PALETTE["amber"]], ax=ax2)
    ax2.axhline(4.0, color=PALETTE["red"], ls="--", lw=1.2, label="BSP 4.0% Ceiling")
    ax2.set_title(f"Island-Group Inflation Dispersion\nOne-Way ANOVA F={f_isl:.2f} (p={p_isl:.2e})", fontweight="bold")
    ax2.set_xlabel("Island Cluster")
    ax2.set_ylabel("YoY Inflation (%)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_inflation_distribution_tukey_outliers.png", dpi=300)
    plt.close(fig)

    # Fig 03: Regional Boxplots ranked by median inflation
    reg_order = yoy_panel.groupby("parent_region")["headline_yoy_pct"].median().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=300)
    sns.boxplot(data=yoy_panel, y="parent_region", x="headline_yoy_pct", order=reg_order, color=PALETTE["sky"], fliersize=2, ax=ax)
    ax.axvline(4.0, color=PALETTE["red"], ls="--", lw=1.6, label="BSP 4.0% Upper Target")
    ax.axvline(2.0, color=PALETTE["emerald"], ls="--", lw=1.6, label="BSP 2.0% Lower Target")
    ax.set_title(f"Regional Inflation Distribution Across 18 Philippine Regions (2019-2026) | ANOVA F={f_reg:.2f}", fontweight="bold")
    ax.set_xlabel("Monthly Provincial Headline YoY Inflation (%)")
    ax.set_ylabel("")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_regional_anova_boxplot.png", dpi=300)
    plt.close(fig)

    # Fig 04: Scatter & Regression — Rice YoY vs Bottom-30% Inflation Spread
    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    sc_df = yoy_panel.dropna(subset=["rice_cpi_yoy_pct", "b30_inflation_spread_pp", "food_bev_weight"]).sample(min(3000, len(yoy_panel)), random_state=42)
    sns.scatterplot(data=sc_df, x="rice_cpi_yoy_pct", y="b30_inflation_spread_pp", hue="island_group", alpha=0.55, s=28, ax=ax)
    sns.regplot(data=sc_df, x="rice_cpi_yoy_pct", y="b30_inflation_spread_pp", scatter=False, color=PALETTE["navy"], line_kws={"lw": 2.0}, ax=ax)
    ax.axhline(0.0, color=PALETTE["slate"], ls="--", lw=1.0)
    ax.set_title("Inflation Inequality: How Provincial Rice Inflation Widens the Bottom-30% Spread", fontweight="bold")
    ax.set_xlabel("Provincial Rice CPI YoY Inflation (%)")
    ax.set_ylabel("Bottom 30% Spread (B30 YoY - All Income YoY, pp)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_rice_vs_headline_scatter_regression.png", dpi=300)
    plt.close(fig)

    # Fig 05: Multivariate Correlation Matrix Heatmap
    corr_cols = ["headline_yoy_pct", "b30_headline_yoy_pct", "food_bev_yoy_pct", "rice_cpi_yoy_pct", "elec_gas_fuel_yoy_pct", "transport_yoy_pct", "gwpi_all_yoy_lag1", "retail_rice_wm_yoy_pct", "natl_core_yoy_pct", "target_yoy_t1"]
    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=300)
    sns.heatmap(yoy_panel[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Multivariate Correlation Matrix: CPI Components, Wholesale/Retail Rice & Next-Month Target", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig05_correlation_heatmap_drivers.png", dpi=300)
    plt.close(fig)

    # Fig 06: Advanced 1 — Geospatial Provincial Inflation & Bottom-30% Spread Map (101 Units)
    geo_agg = yoy_panel.groupby(["clean_geo", "latitude", "longitude", "island_group"]).agg(
        mean_yoy=("headline_yoy_pct", "mean"),
        latest_yoy=("headline_yoy_pct", "last"),
        breach_rate=("is_breach_bsp_4pct", "mean"),
    ).reset_index().dropna(subset=["latitude", "longitude"])
    fig, ax = plt.subplots(figsize=(8.2, 9.5), dpi=300)
    sc = ax.scatter(
        geo_agg["longitude"], geo_agg["latitude"],
        c=geo_agg["mean_yoy"], s=65 + geo_agg["breach_rate"] * 220,
        cmap="viridis", edgecolors="#1E293B", linewidth=0.6, alpha=0.88,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.72)
    cbar.set_label("Mean YoY Inflation 2019-2026 (%)")
    top_hot = geo_agg.nlargest(6, "mean_yoy").reset_index(drop=True)
    offsets = {
        "Occidental Mindoro": (-88, -6),
        "Oriental Mindoro": (7, -6),
        "Dinagat Islands": (-72, 6),
        "Negros Occidental": (-82, -10),
    }
    for _, row in top_hot.iterrows():
        ox, oy = offsets.get(row["clean_geo"], (6, 3))
        ax.annotate(f"{row['clean_geo']} ({row['mean_yoy']:.1f}%)", (row["longitude"], row["latitude"]), xytext=(ox, oy), textcoords="offset points", fontsize=7.5, fontweight="bold")
    ax.set_title("Geospatial Inflation Intensity Across 101 Philippine Provinces & HUCs\n(Color = Mean YoY %, Bubble Size = >4% BSP Breach Frequency)", fontweight="bold")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_advanced_provincial_choropleth_geo.png", dpi=300)
    plt.close(fig)

    # Fig 07: Advanced 2 — Sankey Diagram: Cost-Push Drivers -> COICOP Basket -> Island Clusters
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.axis("off")
    # Compute true percentage-point contributions across all 13 COICOP divisions (2022-2026)
    sankey_df = div_df[(div_df["is_subnational_leaf"]) & (div_df["year"] >= 2022) & (div_df["coicop_code"].isin(["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13"]))].copy()
    div_pp = sankey_df.groupby("coicop_code")["weighted_pp_contrib"].mean().clip(lower=0.0)
    food_pp = float(div_pp.get("01", 0.0))
    energy_pp = float(div_pp.get("04", 0.0) + div_pp.get("07", 0.0))
    core_pp = float(div_pp.sum() - food_pp - energy_pp)
    tot_pp = max(food_pp + energy_pp + core_pp, 1e-6)
    scale = 88.0 / tot_pp
    left_vals = {
        f"Agri / Food Supply ({food_pp:.2f} pp)": food_pp * scale,
        f"Utilities & Transport ({energy_pp:.2f} pp)": energy_pp * scale,
        f"Services & Other Core ({core_pp:.2f} pp)": core_pp * scale,
    }
    mid_vals = {
        f"01 Food & Non-Alc. ({food_pp / tot_pp:.0%})": food_pp * scale,
        f"04+07 Housing/Util & Transp ({energy_pp / tot_pp:.0%})": energy_pp * scale,
        f"11+Other 10 Divisions ({core_pp / tot_pp:.0%})": core_pp * scale,
    }
    isl_totals = yoy_panel[yoy_panel["year"] >= 2022].groupby("island_group")["headline_yoy_pct"].mean()
    isl_shares = (isl_totals / isl_totals.sum()) * 88.0
    right_vals = {f"{k} (Avg {isl_totals[k]:.1f}%)": float(isl_shares[k]) for k in ["Luzon", "Visayas", "Mindanao"] if k in isl_shares}

    def _draw_col(x, items, colors):
        y = 5.0
        pos = {}
        for i, (lbl, val) in enumerate(items.items()):
            ax.add_patch(mpatches.Rectangle((x, y), 0.35, val, color=colors[i % len(colors)], ec="white"))
            ax.text(x - 0.12 if x < 2 else (x + 0.45 if x > 7 else x + 0.17), y + val / 2, lbl,
                    ha="right" if x < 2 else ("left" if x > 7 else "center"), va="center", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8) if 2 <= x <= 7 else None)
            pos[lbl] = (y, y + val)
            y += val + 2.0
        return pos

    p_l = _draw_col(2.4, left_vals, [PALETTE["amber"], PALETTE["red"], PALETTE["blue"]])
    p_m = _draw_col(5.0, mid_vals, [PALETTE["amber"], PALETTE["red"], PALETTE["blue"]])
    p_r = _draw_col(7.6, right_vals, [PALETTE["sky"], PALETTE["teal"], PALETTE["purple"]])
    for (lk, (lb, lt)), (mk, (mb, mt)) in zip(p_l.items(), p_m.items()):
        _draw_bezier_ribbon(ax, 2.75, 5.0, lb, lt, mb, mt, PALETTE["amber"] if "Food" in mk else (PALETTE["red"] if "Util" in mk else PALETTE["blue"]), 0.42)
    # Distribute mid to right proportionally
    r_cursors = {rk: rb for rk, (rb, _) in p_r.items()}
    r_tot = sum(right_vals.values())
    for mk, (mb, mt) in p_m.items():
        mh = mt - mb
        m_cur = mb
        c = PALETTE["amber"] if "Food" in mk else (PALETTE["red"] if "Util" in mk else PALETTE["blue"])
        for rk, rv in right_vals.items():
            seg = mh * (rv / r_tot)
            _draw_bezier_ribbon(ax, 5.35, 7.6, m_cur, m_cur + seg, r_cursors[rk], r_cursors[rk] + seg, c, 0.36)
            m_cur += seg
            r_cursors[rk] += seg
    ax.set_title("Sankey Flow: Inflation Transmission from Supply Channels → COICOP Basket → Island Clusters (2022–2026)", fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_advanced_sankey_transmission.png", dpi=300)
    plt.close(fig)

    # Fig 08: Advanced 3 — Parallel Coordinates Profiling Across 101 Provinces/HUCs
    p_cols = ["headline_yoy_pct", "food_bev_yoy_pct", "rice_cpi_yoy_pct", "elec_gas_fuel_yoy_pct", "transport_yoy_pct", "b30_inflation_spread_pp"]
    p_prof = yoy_panel[yoy_panel["year"] >= 2023].groupby(["clean_geo", "island_group"])[p_cols].mean().dropna().reset_index()
    p_norm = p_prof.copy()
    for c in p_cols:
        mn, mx = p_norm[c].min(), p_norm[c].max()
        p_norm[c] = (p_norm[c] - mn) / (mx - mn + 1e-9)
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    isl_colors = {"Luzon": PALETTE["blue"], "Visayas": PALETTE["teal"], "Mindanao": PALETTE["amber"]}
    xs = np.arange(len(p_cols))
    for _, row in p_norm.iterrows():
        ax.plot(xs, [row[c] for c in p_cols], color=isl_colors.get(row["island_group"], PALETTE["slate"]), alpha=0.35, lw=1.1)
    for k, c in isl_colors.items():
        ax.plot([], [], color=c, lw=2.5, label=k)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Headline YoY", "Food & Bev YoY", "Rice YoY", "Electricity/Fuel YoY", "Transport YoY", "Bottom-30% Spread"], fontsize=9, fontweight="bold")
    ax.set_ylabel("Min-Max Normalized Intensity (2023-2026 Mean)")
    ax.set_title("Parallel Coordinates: Multivariate Inflation Profiles Across 101 Provinces & HUCs", fontweight="bold")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_advanced_parallel_coordinates.png", dpi=300)
    plt.close(fig)

    # Fig 09: Advanced 4 — Hierarchical Contribution Bar-Treemap (COICOP Divisions by Region)
    reg_contrib = div_df[(div_df["geo_level"] == "Region") & (div_df["year"] >= 2023) & (div_df["coicop_code"].isin(["01", "04", "07", "11", "02", "06", "10"]))].groupby(["clean_geo", "coicop_code"])["weighted_pp_contrib"].mean().unstack(fill_value=0)
    reg_contrib = reg_contrib.loc[reg_contrib.sum(axis=1).sort_values(ascending=True).index]
    fig, ax = plt.subplots(figsize=(11.5, 6.2), dpi=300)
    reg_contrib.plot(kind="barh", stacked=True, colormap="viridis", ax=ax)
    ax.set_title("CPI Basket Weight × YoY Inflation Decomposition Across Philippine Regions (2023–2026 Mean)", fontweight="bold")
    ax.set_xlabel("Percentage-Point Contribution to Regional Headline Inflation (pp)")
    ax.set_ylabel("")
    ax.legend(title="COICOP Division", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig09_advanced_sunburst_treemap_weights.png", dpi=300)
    plt.close(fig)

    # Fig 10: Advanced 5 — Streamgraph of Percentage-Point Contributions (National 2019-2026)
    stream_df = div_df[(div_df["clean_geo"] == "PHILIPPINES") & (div_df["year"] >= 2019) & (div_df["coicop_code"].isin(["01", "04", "07", "11"]))].pivot_table(index="obs_date", columns="coicop_name", values="weighted_pp_contrib", aggfunc="mean").fillna(0)
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    ax.stackplot(stream_df.index, *[stream_df[c].values for c in stream_df.columns], labels=stream_df.columns, baseline="wiggle", colors=[PALETTE["amber"], PALETTE["red"], PALETTE["blue"], PALETTE["teal"]], alpha=0.85)
    ax.set_title("Streamgraph: Shifting Commodity Drivers of Philippine Headline Inflation (Jan 2019 – Oct 2026)", fontweight="bold")
    ax.set_ylabel("Weighted Contribution Flow (pp)")
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig10_advanced_streamgraph_contributions.png", dpi=300)
    plt.close(fig)

    # Fig 11: Predictive Fan Chart (Out-of-Time 2025-2026 + 6-Month Forward Forecast with 80%/95% CI)
    oot_agg = te_export.groupby("obs_date").agg(
        actual=("target_yoy_t1", "mean"),
        pred=("pred_xgb", "mean"),
        q10=("pred_q10", "mean"),
        q90=("pred_q90", "mean"),
        q025=("pred_q025", "mean"),
        q975=("pred_q975", "mean"),
    ).reset_index()
    fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=300)
    hist_tail = natl.loc["2022-01-01":]
    ax.plot(hist_tail.index, hist_tail["headline_yoy_pct"], color=PALETTE["navy"], lw=2.1, label="Observed National CPI YoY (%)")
    ax.fill_between(oot_agg["obs_date"], oot_agg["q025"], oot_agg["q975"], color=PALETTE["sky"], alpha=0.18, label="XGB Quantile 95% Interval")
    ax.fill_between(oot_agg["obs_date"], oot_agg["q10"], oot_agg["q90"], color=PALETTE["blue"], alpha=0.28, label="XGB Quantile 80% Interval")
    ax.plot(oot_agg["obs_date"], oot_agg["pred"], color=PALETTE["blue"], lw=2.0, ls="--", label="Out-of-Time XGBoost Provincial Mean Forecast")
    fwd_dates = pd.date_range(natl.index.max() + pd.DateOffset(months=1), periods=6, freq="MS")
    ax.fill_between(fwd_dates, fwd_ci_95.iloc[:, 0], fwd_ci_95.iloc[:, 1], color=PALETTE["amber"], alpha=0.18, label="SARIMAX 6-Mo Forward 95% CI")
    ax.fill_between(fwd_dates, fwd_ci_80.iloc[:, 0], fwd_ci_80.iloc[:, 1], color=PALETTE["amber"], alpha=0.35, label="SARIMAX 6-Mo Forward 80% CI")
    ax.plot(fwd_dates, fwd_fc.predicted_mean.values, color=PALETTE["orange"], lw=2.2, marker="o", label="SARIMAX Forward Point Forecast (Nov 2026 - Apr 2027)")
    ax.axhspan(2.0, 4.0, color=PALETTE["emerald"], alpha=0.08, label="BSP 2%-4% Target Band")
    ax.set_title("Time-Series Fan Chart: Out-of-Time Provincial ML Intervals & 6-Month SARIMAX Forecast", fontweight="bold")
    ax.set_ylabel("YoY Inflation (%)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig11_predictive_fan_chart_sarimax_xgb.png", dpi=300)
    plt.close(fig)

    # Fig 12: XAI TreeSHAP Feature Importance + ROC Curves for >4.0% BSP Breach
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=300)
    top_shap = shap_df.head(12).iloc[::-1]
    ax1.barh(top_shap["feature"], top_shap["mean_abs_shap"], color=PALETTE["teal"])
    ax1.set_title("Exact TreeSHAP Feature Importance (XGBoost Next-Month YoY)", fontweight="bold")
    ax1.set_xlabel("Mean |SHAP Contribution| (percentage points)")
    for name, (fpr, tpr, auc_v) in roc_curves.items():
        ax2.plot(fpr, tpr, lw=2.0, label=f"{name} (AUC = {auc_v:.3f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1.0)
    ax2.set_title("ROC Curves: Predicting Provincial >4.0% BSP Target Breach (2025–2026 Holdout)", fontweight="bold")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig12_xai_treeshap_and_roc_breach.png", dpi=300)
    plt.close(fig)

    summary = {
        "project": "PesoPulse (Team 3): Forecasting Philippine Provincial Inflation Using Monthly CPI & Upstream Price Drivers",
        "eda": {
            "n_subnational_yoy_obs": int(len(h_vals)),
            "mean_yoy_pct": round(float(h_vals.mean()), 3),
            "median_yoy_pct": round(float(h_vals.median()), 3),
            "sd_yoy_pct": round(float(h_vals.std()), 3),
            "iqr_yoy_pct": round(float(iqr), 3),
            "skewness": round(float(stats.skew(h_vals)), 3),
            "kurtosis_pearson": round(float(stats.kurtosis(h_vals, fisher=False)), 3),
            "tukey_upper_fence": round(float(tukey_hi), 3),
            "tukey_outlier_pct": round(float(((h_vals < tukey_lo) | (h_vals > tukey_hi)).mean() * 100), 2),
        },
        "hypothesis_tests": {
            "H1_island_anova_F": round(float(f_isl), 3),
            "H1_island_anova_p": float(p_isl),
            "H1_island_kruskal_H": round(float(kw_isl), 3),
            "H1_regional_anova_F": round(float(f_reg), 3),
            "H1_regional_anova_p": float(p_reg),
            "H2_b30_vs_all_mean_diff_pp": round(float((b30_pair["b30_headline_yoy_pct"] - b30_pair["headline_yoy_pct"]).mean()), 3),
            "H2_b30_paired_t": round(float(t_b30), 3),
            "H2_b30_paired_p": float(p_b30),
            "H2_high_food_b30_diff_pp": round(float((high_food["b30_headline_yoy_pct"] - high_food["headline_yoy_pct"]).mean()), 3),
            "H2_high_food_paired_t": round(float(t_hf), 3),
            "H3_gwpi_lag1_pearson_r": round(float(r_gwpi), 4),
            "H3_rice_yoy_pearson_r": round(float(r_rice), 4),
        },
        "regression_oot_2025_2026": reg_results,
        "uncertainty_coverage": {"empirical_80pct_cov": round(cov_80, 2), "empirical_95pct_cov": round(cov_95, 2)},
        "classification_breach_4pct_oot": clf_results,
        "national_sarimax_oot_mae": round(sarimax_mae, 4),
        "top_10_treeshap_features": shap_df.head(10).to_dict(orient="records"),
    }
    with open(DATA_DIR / "analytics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    run_all()
