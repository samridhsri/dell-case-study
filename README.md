# Dell Technologies - Data Scientist Case Study
**Samridh Srivastava | March 2026**

---

## Overview

This repository contains my solution to the Dell Global Operations manufacturing analytics case study. The objective is to **predict kit end date** and **kit cycle time** (the interval in minutes between kit assembly start and completion) for manufacturing orders, evaluated primarily on MAE, with a secondary lens on on-time delivery classification.

The work is structured as a four-iteration, cumulative Jupyter notebook that progressively builds from data cleaning through to business-ready recommendations.

---

## Repository Structure

```
├── SamridhSrivastava_DS_Case_Notebook_2026.ipynb   # Main analysis notebook
├── SamridhSrivastava_DS_Case_Slides_2026.pdf      # Executive summary slides pdf
└── README.md
```

> **Note:** Raw data files (`train.parquet`, `calendar.parquet`, `capacity.parquet`, `inference.parquet`) are **not included** in this repository per submission guidelines.

---

## Notebook Structure

| Iteration | Focus | Key Output |
|-----------|-------|------------|
| **1. Data Cleaning & Baseline** | Negative durations, sentinel dates, outlier removal, Linear Regression baseline | Clean dataset, baseline MAE |
| **2. Calendar Feature Engineering** | Fiscal week, EOQ flag, weekend indicator, day-of-week | Best holdout MAE (~0.88 days); LR + Calendar outperforms tree models on holiday-dense test window |
| **3. Capacity Feature Engineering** | OT utilisation, site congestion (weighted by remaining `expected_kit_minutes`), lag/rolling features | Time-series CV with reduced variance vs. Iteration 2 |
| **4. Insights & Classification** | SHAP attribution, OT/congestion what-if simulations, on-time classification (Logistic Regression + Random Forest), operational recommendations | ROC/PR curves, 5 strategic recommendations |

---

## Key Modelling Decisions

- **Time-based validation throughout**: `TimeSeriesSplit` and a chronological holdout set — no random splits that would leak future information.
- **Calendar features over model complexity**: Fiscal calendar flags (`is_eoq`, `fiscal_qtr_weeknum`, `is_weekend`) outperformed tree-based ensembles on the holiday-dense test window, reinforcing that domain context can matter more than algorithmic sophistication.
- **EOQ defined by Dell's fiscal quarters**, not calendar quarters.
- **Site congestion weighted** by remaining `expected_kit_minutes` rather than raw concurrent order count.
- **Causal caveats on simulations**: What-if OT analyses are framed as model-based simulations, not causal claims.

---

---

---

## Performance Summary

**Best Test MAE: 0.70 days** (XGBoost + Calendar + Capacity Features)

The model achieved a **24% improvement** over the baseline (0.92 days → 0.70 days) through progressive feature engineering. Calendar features alone contributed a 18% gain (0.92 → 0.75 days), while capacity features added the final 7% (0.75 → 0.70 days). Critically, hyperparameter tuning via TimeSeriesSplit grid search produced <0.005 day improvement, confirming that **feature engineering, not model complexity, drove the gains**.

**Progression across iterations:**
- Baseline (Order + Expected Minutes): 0.92 days (22 hrs)
- + Calendar Features: 0.75 days (18 hrs)
- + Capacity Features: 0.70 days (17 hrs)

The model is stress-tested on the Dec 2025 – Jan 2026 holdout period (EOQ + holidays), making the 0.70-day MAE especially robust for operational deployment.



---

## Strategic Recommendations (Summary)

1. **Weekend staffing adjustments** based on observed cycle time differentials
2. **Congestion-based scheduling** to smooth order intake during high-utilisation periods
3. **OT utilisation as a leading indicator** for proactive intervention
4. **pre-EOQ preparedness planning** aligned to Dell's fiscal calendar
5. **Shipping promise calibration** informed by predicted cycle time distributions

---

## Tools & Libraries

`Python` · `pandas` · `scikit-learn` · `XGBoost`  · `SHAP` · `matplotlib` · `seaborn`