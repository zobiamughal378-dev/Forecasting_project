# Task 4 - Time-Series Forecasting & Interpretability Dashboard

**Dataset:** Mauna Loa atmospheric CO2 (climate data, built into `statsmodels` - no download).
**Models:** SARIMA (AIC grid search, 95% CI) + Gradient Boosting explained with SHAP.

## Run
```bash
pip install -r requirements.txt
python project.py
```
Output folder `outputs/`: figures (fig1..fig10, dashboard.png), CSV tables, `future_forecast.csv`,
`results.json` and **`whitepaper.pdf`** (auto-generated, 5-6 pages, numbers filled from your run).
