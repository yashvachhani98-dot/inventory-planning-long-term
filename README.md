# Long-Term Inventory Planning – Retail Store (Kaggle)

This project builds a simple long-term inventory planning engine in Python using the
**Retail Store Inventory Forecasting** dataset from Kaggle.

It shows how to go from raw daily sales to a 12-month SKU–store plan, with
forecast accuracy checks, ABC–XYZ segmentation, and an inventory policy
(safety stock + target inventory) for a hero SKU.

---

## Business question

> If I’m responsible for long-term planning for a retail portfolio (like L’Oréal / Home Depot),
> how can I:
> - Forecast demand 12–24 months out by SKU–store?
> - Focus on the most important SKUs (ABC–XYZ)?
> - Set safety stock and target inventory levels for hero products?

---

## Pipeline

1. **Load & clean data**
   - Kaggle CSV: daily records for multiple stores and products.
   - Normalize column names, convert `Date` to datetime.

2. **Monthly aggregation**
   - Aggregate daily sales into **monthly demand per SKU–store**.
   - Table: `sku`, `store`, `month`, `demand_units`.

3. **Feature engineering**
   - For each SKU–store time series, create:
     - `lag_1`, `lag_2`, `lag_3` (last 3 months demand)
     - `month_of_year` (1–12 for seasonality)
   - Drop early rows where lags aren’t available.

4. **Train/test split**
   - Time-based split (not random):
     - Train on early months
     - Test on the **last 6 months** to mimic a real planning cycle.

5. **Baseline models**
   - **Naive**: forecast = last month (`lag_1`).
   - **Moving Average (3 months)**: mean of `lag_1`, `lag_2`, `lag_3`.
   - Metrics: RMSE, MAE, **WAPE** (Weighted Absolute Percentage Error).

6. **ML models**
   - **Linear Regression** on `[lag_1, lag_2, lag_3, month_of_year]`.
   - **XGBoost Regressor** on the same features.
   - Compare models on the last 6 months using WAPE:

   | Model                     | WAPE (approx) |
   |---------------------------|---------------|
   | Naive (last month)       | ~36%          |
   | Moving Average (3 months)| ~33%          |
   | **Linear Regression**    | **~31%**      |
   | XGBoost Regressor        | ~32%          |

   Linear Regression gives the best balance of accuracy and simplicity.

7. **12-month forecast**
   - For each SKU–store:
     - Take last 3 actual months as starting lags.
     - Roll forward 12 steps:
       - Predict next month using the model.
       - Feed the prediction back as the newest lag.
   - Result: `future_forecast_df` with 12-month forecast for every SKU–store.
   - Plot example: history + 12-month forecast for a **hero SKU & store**.

8. **ABC–XYZ segmentation**
   - **ABC** by total volume:
     - A = top ~80% of volume, B = next ~15%, C = remaining tail.
   - **XYZ** by coefficient of variation (CV) of monthly demand:
     - X = stable (CV ≤ 0.3), Y = medium, Z = very variable.
   - Combine into segments like **AX, BY, CZ**.
   - Hero SKU in this dataset is classified as **AX** (high volume, stable).

9. **Hero SKU inventory policy**
   - For the hero AX SKU at its top store:
     - Assume **1-month lead time**.
     - Map AX → **98% service level** (z ≈ 2.05).
     - Compute:
       - Mean monthly demand and standard deviation.
       - **Safety stock** = z × σ.
       - **Cycle stock** = mean demand × lead time.
       - **Target inventory** = safety stock + cycle stock.
   - Plot history, 12-month forecast, and the target inventory line.

---

## Files

- `inventory_planning_long_term.ipynb`  
  Interactive Colab notebook with full analysis, plots, and commentary.

- `inventory_planning_long_term.py`  
  Standalone Python script that runs the full pipeline end-to-end
  (load data → modeling → 12-month forecast → ABC–XYZ → hero inventory policy).

---

## How to run (script)

```bash
# Clone repo
git clone https://github.com/yashvachhani98-dot/inventory-planning-long-term.git
cd inventory-planning-long-term

# Install dependencies (example)
pip install -r requirements.txt  # if you add one, or install manually:
pip install pandas numpy matplotlib scikit-learn xgboost python-dateutil

# Make sure retail_store_inventory.csv is in this folder, then:
python inventory_planning_long_term.py
