# Rossmann Store Sales — Sales Prediction & Staffing Decision Support
**MSE 433 Final Project | Yusur Araim | University of Waterloo**

---

## Overview

Retail managers face a daily challenge — they have to decide how many staff to schedule without knowing exactly how busy the store will be. Schedule too many and you waste money on wages. Schedule too few and customers wait too long, which hurts the store's reputation. Most stores either rely on gut feeling or use the same fixed number every day, neither of which accounts for the natural variation in demand caused by promotions, holidays, day of the week, and seasonal patterns.

This project tackles that problem using machine learning. Using three years of real daily sales data from Rossmann stores across Germany (2013–2015), I built a system that predicts how much a store will sell each day and then translates that prediction directly into a staffing recommendation. What makes this different from a standard forecasting project is the decision-making step at the end — the predictions are not just numbers, they are turned into an actionable output that a store manager could actually use.

---

## Problem Statement

The goal is not just to predict sales accurately, but to use those predictions to answer a practical question: **how many staff should be scheduled today?** This involves two layers of decision making:

1. Predicting the expected sales level for a given store on a given day
2. Accounting for uncertainty — on days where the model is less confident, an extra buffer staff member is added to avoid being caught understaffed on an unexpectedly busy day

---

## Dataset

The project uses the [Rossmann Store Sales dataset from Kaggle](https://www.kaggle.com/competitions/rossmann-store-sales), which contains:

- Daily sales records for **1,115 stores** over roughly 2.5 years (2013–2015)
- Over **1 million rows** of data after merging store and sales files
- Key variables including promotions, school holidays, state holidays, store type, assortment type, and competition distance
- A separate store metadata file with store-level characteristics

The data was filtered to only include days when stores were open and had real sales, leaving around **844,000 rows**. After adding lag and rolling features (which require historical data to calculate), the final modelling dataset had around **404,000 rows**.

---

## Approach

### Data Cleaning

Missing values in numeric columns were filled with the column median. Categorical variables like store type, assortment, and state holiday were label-encoded so the models could use them as numeric inputs.

### Feature Engineering

Beyond the raw columns, several additional features were created to help the model pick up on temporal patterns:

| Feature | Description |
|---|---|
| `Lag1` | Yesterday's sales |
| `Lag7` | Sales 7 days ago (same day last week) |
| `Lag14` | Sales 14 days ago |
| `Roll7` | 7-day rolling mean of past sales |
| `Roll30` | 30-day rolling mean of past sales |
| `Month`, `WeekOfYear`, `Quarter` | Calendar position features |
| `IsWeekend`, `IsMonthStart`, `IsMonthEnd` | Binary calendar flags |

These features are critical because retail demand is heavily driven by recent history and calendar patterns. All lag and rolling features are computed using past data only (`.shift(1)` before the rolling window) to prevent any lookahead leakage.

### Train / Test Split

The data was split **temporally** — the first 80% of dates were used for training and the most recent 20% for testing. The split date was **10 February 2015**, giving 323,169 training rows and 80,747 test rows. This mirrors how the model would actually work in production: always training on the past and predicting the future.

### Models

Two models were built and compared:

**Linear Regression (baseline)**
A simple model that fits a straight line through the features. Used as a benchmark to check whether a more complex model is actually worth the added complexity.

**Quantile Gradient Boosting (main model)**
Gradient Boosting builds an ensemble of decision trees where each tree corrects the errors of the previous one. The quantile version was trained three times:
- At the **10th percentile** — an optimistic lower bound
- At the **50th percentile** — the point prediction (median)
- At the **90th percentile** — a pessimistic upper bound

The gap between the 10th and 90th percentile forms an **80% prediction interval** that captures how uncertain the model is on any given day.

### Staffing Decision Rules

The staffing recommendation works as follows:

1. Divide the predicted sales (50th percentile) by the sales-per-staff threshold to get a base staff count
2. Clip the result between a minimum of 3 and a maximum of 12 staff
3. If the **relative uncertainty** (interval width ÷ median prediction) exceeds 0.6, add one buffer staff member
4. Flag an inventory action based on the predicted demand tier (standard / monitor / restock)

The sales-per-staff threshold was derived from the dataset itself using the `Customers` column, rather than being an assumed number. The median sales per customer (€9.25) multiplied by 60 customers per staff member (a standard retail industry benchmark) gives a threshold of approximately **€555 per staff member per day**.

### Cost Analysis

To evaluate whether the model's staffing decisions are actually better in practice, a simple cost model was built:

- **Overstaffing cost** = extra staff × €120 per day (estimated daily wage)
- **Understaffing cost** = missing staff × €555 × 20% (estimated lost sales from being short-staffed)

The model's staffing schedule was compared against a **fixed baseline** that always schedules the data-derived average of 10 staff per day, regardless of forecast.

---

## Results

### Model Accuracy

| Metric | Linear Regression | Quantile GBR |
|---|---|---|
| MAE | €957 | **€769** |
| RMSE | €1,341 | **€1,156** |
| R² | 0.7497 | **0.8137** |
| MAPE | 15.01% | **12.08%** |

Quantile GBR outperformed Linear Regression on every metric. Cross-validation confirmed the results are consistent across different time windows and not specific to this particular split.

### Feature Importance

The two most important features by far were `Lag1` (yesterday's sales, **37.6%**) and `Roll30` (30-day rolling average, **30.2%**), together accounting for nearly **68%** of the model's predictive power. This confirms that recent sales history is the strongest signal for tomorrow's demand. Promotions (13.9%) and day of week (9.3%) were the next most important. Many features such as `StateHoliday`, `Lag7`, and `Store` contributed almost nothing individually.

### Validation

| Check | Result |
|---|---|
| 5-fold time-series cross-validation | LR mean MAE €1,017 ± €44 / QGBR mean MAE €930 ± €60 — consistent across all folds |
| Calibration coverage | 82.3% of actuals fell inside the 80% interval — well-calibrated |
| Residual analysis | Errors centred near zero; slight over-prediction on Mondays, under-prediction on Sundays |
| Sensitivity heatmap | Worst performance on Sunday non-promo days (MAE €2,376) and certain store/assortment combinations |

### Cost Savings

| | Cost |
|---|---|
| Fixed schedule (10 staff/day) | €17,290,263 |
| Model-based schedule | €5,811,687 |
| **Total savings** | **€11,478,576 (66.4%)** |
| Average daily saving | €142.15 |

---

## Files

| File | Description |
|---|---|
| `Final Project Yusur MSE 433.ipynb` | Full notebook with all code, markdown explanations, charts, and outputs |
| `Final Project Yusur MSE 433.html` | Same notebook rendered as HTML — viewable in any browser without Python |
| `train.csv` | Daily sales data (download from Kaggle — not included due to file size) |
| `store.csv` | Store metadata (download from Kaggle — not included due to file size) |

---

## How to Run

1. Download `train.csv` and `store.csv` from the [Rossmann Kaggle page](https://www.kaggle.com/competitions/rossmann-store-sales/data) — these are not included in the repo due to file size limits
2. Place both CSV files in the same folder as the notebook
3. Install dependencies:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
4. Open `Final Project Yusur MSE 433.ipynb` and run all cells top to bottom in order

Alternatively, open `Final Project Yusur MSE 433.html` in any browser to view all outputs and charts without running anything.

---

## Dependencies

| Package | Purpose |
|---|---|
| Python 3.8+ | |
| `pandas` | Data loading, cleaning, and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Linear Regression, Gradient Boosting, cross-validation, metrics |
| `matplotlib` | All charts |
| `seaborn` | Heatmaps and plot styling |
