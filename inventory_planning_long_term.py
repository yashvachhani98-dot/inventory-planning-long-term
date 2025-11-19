"""
Long-Term Inventory Planning – Retail Store (Kaggle)

This script reproduces the end-to-end pipeline from the notebook:

1. Load the Kaggle retail inventory dataset.
2. Aggregate daily sales to monthly SKU–store demand.
3. Engineer lag + calendar features for time-series modeling.
4. Fit baseline models (Naive, MA(3)) and ML models (Linear Regression, XGBoost).
5. Generate a 12-month ahead forecast for every SKU–store.
6. Perform ABC–XYZ segmentation on SKUs.
7. Compute safety stock and target inventory for a hero AX SKU.

Run from terminal (if CSV is in same folder):
    python inventory_planning_long_term.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from dateutil.relativedelta import relativedelta

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Data loading & aggregation
# ---------------------------------------------------------------------------

def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load raw Kaggle CSV and normalize column names."""
    df_raw = pd.read_csv(csv_path)
    df = df_raw.rename(
        columns={
            "Store ID": "store",
            "Product ID": "sku",
            "Units Sold": "demand_units",
            "Inventory Level": "inventory_level",
            "Units Ordered": "units_ordered",
            "Demand Forecast": "demand_forecast",
            "Weather Condition": "weather",
            "Holiday/Promotion": "promotion",
            "Competitor Pricing": "competitor_price",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def aggregate_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily demand to monthly demand per sku + store."""
    df = df.copy()
    df["month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    monthly = (
        df.groupby(["sku", "store", "month"], as_index=False)["demand_units"]
        .sum()
    )

    # simple sanity checks
    print("Monthly table shape:", monthly.shape)
    print("\nDate range in monthly data:")
    print("  From:", monthly["month"].min())
    print("  To  :", monthly["month"].max())
    print("\nNumber of unique SKUs:", monthly["sku"].nunique())
    print("Number of unique stores:", monthly["store"].nunique())
    print("\nMissing demand values:", monthly["demand_units"].isna().sum())
    print("Any negative demand?:", (monthly["demand_units"] < 0).any())

    return monthly.sort_values(["sku", "store", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hero SKU selection
# ---------------------------------------------------------------------------

def select_hero_sku_store(monthly: pd.DataFrame) -> Tuple[str, str]:
    """Pick highest-volume SKU, then its highest-volume store."""
    sku_volume = (
        monthly.groupby("sku", as_index=False)["demand_units"]
        .sum()
        .rename(columns={"demand_units": "total_units"})
        .sort_values("total_units", ascending=False)
    )
    hero_sku = sku_volume["sku"].iloc[0]

    hero_sku_store_volume = (
        monthly[monthly["sku"] == hero_sku]
        .groupby("store", as_index=False)["demand_units"]
        .sum()
        .rename(columns={"demand_units": "total_units"})
        .sort_values("total_units", ascending=False)
    )
    hero_store = hero_sku_store_volume["store"].iloc[0]

    print(f"Hero SKU: {hero_sku}, hero store: {hero_store}")
    return hero_sku, hero_store


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_lags_for_group(group: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """For one SKU+store group, add lag_1, lag_2, lag_3 based on demand_units."""
    group = group.sort_values("month").copy()
    for l in lags:
        group[f"lag_{l}"] = group["demand_units"].shift(l)
    return group


def build_feature_table(monthly: pd.DataFrame) -> pd.DataFrame:
    """Create modeling table with lag features and month_of_year."""
    features = (
        monthly
        .groupby(["sku", "store"], group_keys=False)
        .apply(add_lags_for_group)
    )
    features["month_of_year"] = features["month"].dt.month
    features_model = features.dropna(subset=["lag_1", "lag_2", "lag_3"]).reset_index(drop=True)

    print("Original monthly rows:", monthly.shape)
    print("Rows after adding lags & dropping NA lags:", features_model.shape)

    return features_model


# ---------------------------------------------------------------------------
# Baselines & ML models
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    model_name: str
    rmse: float
    mae: float
    wape: float


def train_test_split_time(features_model: pd.DataFrame, test_months: int = 6):
    """Time-based split by month: early months → train, last N months → test."""
    all_months = np.sort(features_model["month"].unique())
    test_period = all_months[-test_months:]
    train_period = all_months[:-test_months]

    train_df = features_model[features_model["month"].isin(train_period)].copy()
    test_df = features_model[features_model["month"].isin(test_period)].copy()

    print("Train period: from", train_period[0], "to", train_period[-1])
    print("Test period : from", test_period[0], "to", test_period[-1])
    print("Train rows:", train_df.shape[0])
    print("Test rows :", test_df.shape[0])

    target_col = "demand_units"
    feature_cols = ["lag_1", "lag_2", "lag_3", "month_of_year"]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    return train_df, test_df, X_train, X_test, y_train, y_test, feature_cols


def compute_wape(y_true, y_pred) -> float:
    return (np.abs(y_pred - y_true).sum() / y_true.sum()) * 100


def evaluate_baselines(test_df: pd.DataFrame, y_test: pd.Series) -> Tuple[ModelMetrics, ModelMetrics]:
    """Naive last-month and 3-month moving average baselines."""
    # Naive
    y_pred_naive = test_df["lag_1"].values
    rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    mae_naive = mean_absolute_error(y_test, y_pred_naive)
    wape_naive = compute_wape(y_test, y_pred_naive)

    # MA(3)
    y_pred_ma3 = test_df[["lag_1", "lag_2", "lag_3"]].mean(axis=1).values
    rmse_ma3 = np.sqrt(mean_squared_error(y_test, y_pred_ma3))
    mae_ma3 = mean_absolute_error(y_test, y_pred_ma3)
    wape_ma3 = compute_wape(y_test, y_pred_ma3)

    print("\nBaseline models on test period:")
    print(f"Naive last-month  – RMSE {rmse_naive:.2f}, MAE {mae_naive:.2f}, WAPE {wape_naive:.2f}%")
    print(f"MA(3)             – RMSE {rmse_ma3:.2f}, MAE {mae_ma3:.2f}, WAPE {wape_ma3:.2f}%")

    return (
        ModelMetrics("Naive (last month)", rmse_naive, mae_naive, wape_naive),
        ModelMetrics("Moving Average (3 months)", rmse_ma3, mae_ma3, wape_ma3),
    )


def train_linear_regression(X_train, y_train, X_test, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    wape = compute_wape(y_test, y_pred)

    print("\nLinear Regression on test period:")
    print(f"RMSE {rmse:.2f}, MAE {mae:.2f}, WAPE {wape:.2f}%")

    return lr, ModelMetrics("Linear Regression", rmse, mae, wape)


def train_xgboost(X_train, y_train, X_test, y_test):
    if not HAS_XGBOOST:
        print("xgboost is not installed; skipping XGBoost model.")
        return ModelMetrics("XGBoost Regressor (not run)", np.nan, np.nan, np.nan), None

    xgb_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    y_pred = xgb_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    wape = compute_wape(y_test, y_pred)

    print("\nXGBoost on test period:")
    print(f"RMSE {rmse:.2f}, MAE {mae:.2f}, WAPE {wape:.2f}%")

    return ModelMetrics("XGBoost Regressor", rmse, mae, wape), xgb_model


# ---------------------------------------------------------------------------
# Forecasting & hero-level evaluation
# ---------------------------------------------------------------------------

def evaluate_hero(test_df, hero_sku, hero_store, feature_cols, lr_model):
    hero_test = test_df[(test_df["sku"] == hero_sku) & (test_df["store"] == hero_store)].copy()
    if hero_test.empty:
        print("\nNo test rows for hero SKU/store.")
        return

    y_test_hero = hero_test["demand_units"]
    y_pred_naive = hero_test["lag_1"]
    y_pred_ma3 = hero_test[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
    X_test_hero = hero_test[feature_cols]
    y_pred_lr_hero = lr_model.predict(X_test_hero)

    print(f"\nHero SKU {hero_sku} | Store {hero_store} – WAPE on test period:")
    print(f"  Naive  : {compute_wape(y_test_hero, y_pred_naive):.2f}%")
    print(f"  MA(3)  : {compute_wape(y_test_hero, y_pred_ma3):.2f}%")
    print(f"  LinReg : {compute_wape(y_test_hero, y_pred_lr_hero):.2f}%")


def build_12_month_forecast(monthly, lr_model):
    """Roll forward 12-month forecast for all sku+store combinations."""
    FORECAST_HORIZON = 12
    future_forecasts: List[Dict] = []

    last_actual_month = monthly["month"].max()
    print("\nLast actual month in history:", last_actual_month)

    for (sku, store), group in monthly.groupby(["sku", "store"]):
        group_sorted = group.sort_values("month")
        last3 = list(group_sorted["demand_units"].tail(3).values)
        current_month = last_actual_month + relativedelta(months=1)

        for _ in range(FORECAST_HORIZON):
            lag_1, lag_2, lag_3 = last3[-1], last3[-2], last3[-3]
            month_of_year = current_month.month
            X_future = [[lag_1, lag_2, lag_3, month_of_year]]
            pred = lr_model.predict(X_future)[0]

            future_forecasts.append(
                {
                    "sku": sku,
                    "store": store,
                    "month": current_month,
                    "forecast_units": float(pred),
                }
            )

            last3.append(pred)
            current_month = current_month + relativedelta(months=1)

    future_forecast_df = pd.DataFrame(future_forecasts)
    print("Future forecast shape:", future_forecast_df.shape)
    return future_forecast_df


# ---------------------------------------------------------------------------
# ABC–XYZ and inventory policy
# ---------------------------------------------------------------------------

def abc_xyz_segmentation(monthly: pd.DataFrame) -> pd.DataFrame:
    # ABC by total volume
    sku_totals = (
        monthly.groupby("sku", as_index=False)["demand_units"]
        .sum()
        .rename(columns={"demand_units": "total_units"})
        .sort_values("total_units", ascending=False)
    )
    total_all = sku_totals["total_units"].sum()
    sku_totals["cum_share"] = sku_totals["total_units"].cumsum() / total_all * 100

    def abc_class(row):
        if row["cum_share"] <= 80:
            return "A"
        elif row["cum_share"] <= 95:
            return "B"
        return "C"

    sku_totals["ABC"] = sku_totals.apply(abc_class, axis=1)

    # XYZ by CV of monthly demand (summed over stores)
    sku_monthly = (
        monthly.groupby(["sku", "month"], as_index=False)["demand_units"]
        .sum()
        .rename(columns={"demand_units": "monthly_units"})
    )
    sku_stats = (
        sku_monthly.groupby("sku")["monthly_units"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "mean_monthly", "std": "std_monthly"})
    )
    sku_stats["CV"] = sku_stats["std_monthly"] / sku_stats["mean_monthly"]

    def xyz_class(cv):
        if cv <= 0.3:
            return "X"
        if cv <= 0.7:
            return "Y"
        return "Z"

    sku_stats["XYZ"] = sku_stats["CV"].apply(xyz_class)

    sku_seg = sku_totals[["sku", "ABC"]].merge(
        sku_stats[["sku", "CV", "XYZ"]], on="sku", how="left"
    )
    sku_seg["segment"] = sku_seg["ABC"] + sku_seg["XYZ"]
    return sku_seg


def hero_inventory_policy(monthly, future_forecast_df, hero_sku, hero_store, sku_seg):
    hero_hist = (
        monthly[(monthly["sku"] == hero_sku) & (monthly["store"] == hero_store)]
        .sort_values("month")
        .copy()
    )
    hero_future = (
        future_forecast_df[(future_forecast_df["sku"] == hero_sku) & (future_forecast_df["store"] == hero_store)]
        .sort_values("month")
        .copy()
    )

    mean_demand_hero = hero_hist["demand_units"].mean()
    std_demand_hero = hero_hist["demand_units"].std()

    hero_segment = sku_seg[sku_seg["sku"] == hero_sku].iloc[0]
    hero_seg_code = hero_segment["segment"]

    service_level_map = {
        "AX": 0.98,
        "AY": 0.95,
        "AZ": 0.93,
        "BX": 0.95,
        "BY": 0.93,
        "BZ": 0.90,
        "CX": 0.93,
        "CY": 0.90,
        "CZ": 0.85,
    }
    z_map = {0.98: 2.05, 0.95: 1.65, 0.93: 1.48, 0.90: 1.28, 0.85: 1.04}

    hero_service_level = service_level_map.get(hero_seg_code, 0.95)
    hero_z = z_map[hero_service_level]
    LEAD_TIME_MONTHS = 1.0

    safety_stock_hero = hero_z * std_demand_hero
    cycle_stock_hero = mean_demand_hero * LEAD_TIME_MONTHS
    target_inventory_hero = safety_stock_hero + cycle_stock_hero

    print(f"\nHero segment: {hero_seg_code}")
    print(f"Target service level: {hero_service_level:.0%} (z = {hero_z})")
    print(f"Mean monthly demand : {mean_demand_hero:.1f}")
    print(f"Std dev monthly     : {std_demand_hero:.1f}")
    print(f"Safety stock (units): {round(safety_stock_hero)}")
    print(f"Cycle stock (units) : {round(cycle_stock_hero)}")
    print(f"Target inventory    : {round(target_inventory_hero)}")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(hero_hist["month"], hero_hist["demand_units"], marker="o", label="Historical demand")
    plt.plot(hero_future["month"], hero_future["forecast_units"], marker="o", linestyle="--", label="Forecast (12m)")
    all_months_for_plot = list(hero_hist["month"]) + list(hero_future["month"])
    plt.hlines(
        target_inventory_hero,
        xmin=min(all_months_for_plot),
        xmax=max(all_months_for_plot),
        linestyles="dotted",
        label=f"Target inventory (~{round(target_inventory_hero)} units)",
    )
    plt.title(f"Hero SKU {hero_sku} | Store {hero_store} – Forecast & Target Inventory")
    plt.xlabel("Month")
    plt.ylabel("Units")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str | Path = "retail_store_inventory.csv"):
    df = load_data(csv_path)
    monthly = aggregate_monthly(df)
    hero_sku, hero_store = select_hero_sku_store(monthly)
    features_model = build_feature_table(monthly)

    train_df, test_df, X_train, X_test, y_train, y_test, feature_cols = train_test_split_time(features_model)

    # Baselines + ML
    naive_metrics, ma3_metrics = evaluate_baselines(test_df, y_test)
    lr_model, lr_metrics = train_linear_regression(X_train, y_train, X_test, y_test)
    xgb_metrics, _ = train_xgboost(X_train, y_train, X_test, y_test)

    # Compare
    results_df = pd.DataFrame(
        [
            vars(naive_metrics),
            vars(ma3_metrics),
            vars(lr_metrics),
            vars(xgb_metrics),
        ]
    )
    print("\nModel comparison:")
    print(results_df)

    # Hero evaluation
    evaluate_hero(test_df, hero_sku, hero_store, feature_cols, lr_model)

    # Forecast + segmentation + inventory policy
    future_forecast_df = build_12_month_forecast(monthly, lr_model)
    sku_seg = abc_xyz_segmentation(monthly)
    hero_inventory_policy(monthly, future_forecast_df, hero_sku, hero_store, sku_seg)


if __name__ == "__main__":
    run_pipeline()
