"""
TASK 4 - Time-Series Forecasting Model & Advanced Model Interpretability Dashboard
-----------------------------------------------------------------------------------
Dataset : Mauna Loa atmospheric CO2 (climate data, built into statsmodels - no download needed)
Pipeline: load -> EDA/STL -> stationarity (ADF+KPSS) -> seasonal differencing
          -> SARIMA (AIC grid search) -> forecast with 95% CI -> baselines
          -> SHAP interpretability on a lag-feature gradient-boosting model
          -> dashboard figure -> auto-generated 5-6 page whitepaper (PDF)

Run:   pip install -r requirements.txt
       python project.py
Outputs go to ./outputs  (figures, CSV tables, results.json, whitepaper.pdf)
Works cell-by-cell in Jupyter / VS Code too (# %% markers).
"""
# %% imports
import os, json, warnings, itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.ensemble import GradientBoostingRegressor
import shap

from whitepaper import build_whitepaper

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)
H = 24            # hold-out horizon (months)
H_FUTURE = 36     # true future forecast (months)
SEED = 42
np.random.seed(SEED)
def P(name): return os.path.join(OUT, name)

def metrics(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return dict(MAE=float(np.mean(np.abs(e))), RMSE=float(np.sqrt(np.mean(e**2))),
                MAPE=float(np.mean(np.abs(e / np.asarray(y_true))) * 100))

# %% 1. load data
raw = sm.datasets.co2.load_pandas().data["co2"]            # weekly, 1958-03 .. 2001-12, has gaps
y = raw.resample("MS").mean().interpolate("time")          # monthly mean, gaps interpolated
y.name = "CO2 (ppm)"
n_missing = int(raw.isna().sum())
R = dict(n_obs=len(y), start=str(y.index[0].date()), end=str(y.index[-1].date()),
         n_missing_weekly=n_missing, y_min=float(y.min()), y_max=float(y.max()),
         y_mean=float(y.mean()), H=H, H_future=H_FUTURE)
y.to_csv(P("monthly_series.csv"))

# %% 2. EDA + STL decomposition
fig, ax = plt.subplots(2, 1, figsize=(10, 6.2))
ax[0].plot(y, lw=1.1, color="#1f4e79"); ax[0].set_title("Monthly mean atmospheric CO$_2$ (Mauna Loa)"); ax[0].set_ylabel("ppm")
yy = y.loc["1990":"1994"]
ax[1].plot(yy, marker="o", ms=3, lw=1, color="#c0392b"); ax[1].set_title("Zoom 1990-1994: yearly seasonal cycle"); ax[1].set_ylabel("ppm")
plt.tight_layout(); fig.savefig(P("fig1_eda.png")); plt.close(fig)

stl = STL(y, period=12, robust=True).fit()
fig = stl.plot(); fig.set_size_inches(10, 6.5); plt.tight_layout(); fig.savefig(P("fig2_stl.png")); plt.close(fig)
R["trend_gain_ppm"] = float(stl.trend.iloc[-1] - stl.trend.iloc[0])
R["seasonal_amp_ppm"] = float(stl.seasonal.max() - stl.seasonal.min())
R["seasonal_strength"] = float(max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal)))
R["trend_strength"] = float(max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.trend)))

# %% 3. stationarity checks + differencing pipeline
def stationarity(s, name):
    s = s.dropna()
    a = adfuller(s, autolag="AIC")
    k = kpss(s, regression="c", nlags="auto")
    stat = (a[1] < .05) and (k[1] > .05)
    return dict(Series=name, ADF_stat=a[0], ADF_p=a[1], KPSS_stat=k[0], KPSS_p=k[1],
                Verdict="Stationary" if stat else "Non-stationary")

variants = {
    "Raw series": y,
    "First difference (d=1)": y.diff(),
    "Seasonal difference (D=1, s=12)": y.diff(12),
    "d=1 and D=1 (s=12)": y.diff().diff(12),
}
stat_tbl = pd.DataFrame([stationarity(v, k) for k, v in variants.items()])
stat_tbl.to_csv(P("table_stationarity.csv"), index=False)
R["stationarity"] = stat_tbl.round(4).to_dict("records")

fig, ax = plt.subplots(2, 2, figsize=(10, 6))
for a_, (k, v) in zip(ax.ravel(), variants.items()):
    a_.plot(v, lw=.8, color="#1f4e79"); a_.set_title(k, fontsize=10)
plt.tight_layout(); fig.savefig(P("fig3_differencing.png")); plt.close(fig)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
plot_acf(y.diff().diff(12).dropna(), lags=36, ax=ax[0]); plot_pacf(y.diff().diff(12).dropna(), lags=36, ax=ax[1], method="ywm")
ax[0].set_title("ACF of d=1, D=1 series"); ax[1].set_title("PACF of d=1, D=1 series")
plt.tight_layout(); fig.savefig(P("fig4_acf_pacf.png")); plt.close(fig)

# %% 4. SARIMA - train/test split + AIC grid search
y_tr, y_te = y.iloc[:-H], y.iloc[-H:]
rows = []
for p, q, Pp, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
    try:
        m = SARIMAX(y_tr, order=(p, 1, q), seasonal_order=(Pp, 1, Q, 12),
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=200)
        rows.append(dict(order=(p, 1, q), seasonal=(Pp, 1, Q, 12), AIC=m.aic, BIC=m.bic))
    except Exception:
        pass
grid = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
grid.head(10).to_csv(P("table_sarima_grid_top10.csv"), index=False)
best_o, best_s = grid.loc[0, "order"], grid.loc[0, "seasonal"]
R["grid_size"] = len(grid); R["best_order"] = list(best_o); R["best_seasonal"] = list(best_s)
R["best_aic"] = float(grid.loc[0, "AIC"])
R["grid_top5"] = [dict(spec=f"SARIMA{r.order}x{r.seasonal}", AIC=float(r.AIC), BIC=float(r.BIC)) for r in grid.head(5).itertuples()]

best = SARIMAX(y_tr, order=best_o, seasonal_order=best_s,
               enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)
fc = best.get_forecast(H)
fc_mean = fc.predicted_mean; ci = fc.conf_int(alpha=.05)
lo, hi = ci.iloc[:, 0], ci.iloc[:, 1]
R["sarima_holdout"] = metrics(y_te, fc_mean)
R["ci95_coverage"] = float(np.mean((y_te.values >= lo.values) & (y_te.values <= hi.values)))
R["ci95_mean_width"] = float(np.mean(hi - lo))

# baselines
naive = pd.Series(np.tile(y_tr.iloc[-12:].values, H // 12 + 1)[:H], index=y_te.index)   # seasonal naive
R["seasonal_naive_holdout"] = metrics(y_te, naive)

# residual diagnostics
resid = best.resid.iloc[13:]
lb = acorr_ljungbox(resid, lags=[24], return_df=True)
jb = jarque_bera(resid)
R["ljung_box_p"] = float(lb["lb_pvalue"].iloc[0]); R["jarque_bera_p"] = float(jb[1])
R["resid_std"] = float(resid.std())
fig = best.plot_diagnostics(figsize=(10, 7), lags=30); plt.tight_layout(); fig.savefig(P("fig6_diagnostics.png")); plt.close(fig)

# SARIMA native interpretability: coefficients
coef = pd.DataFrame({"coef": best.params, "std_err": best.bse, "p_value": best.pvalues}).round(4)
coef.to_csv(P("table_sarima_coefficients.csv"))
R["sarima_coefs"] = [dict(term=i, coef=float(r.coef), se=float(r.std_err), p=float(r.p_value)) for i, r in coef.iterrows()]

# hold-out forecast figure
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(y.loc["1996":], color="#7f8c8d", lw=1.2, label="Observed")
ax.plot(fc_mean, color="#c0392b", lw=1.8, label="SARIMA forecast")
ax.fill_between(fc_mean.index, lo, hi, color="#c0392b", alpha=.2, label="95% prediction interval")
ax.plot(naive, color="#2980b9", ls="--", lw=1, label="Seasonal naive")
ax.axvline(y_te.index[0], color="k", ls=":", lw=1); ax.set_ylabel("CO$_2$ (ppm)")
ax.set_title(f"Hold-out forecast (last {H} months) - SARIMA{best_o}x{best_s}"); ax.legend(loc="upper left")
plt.tight_layout(); fig.savefig(P("fig5_holdout_forecast.png")); plt.close(fig)

# %% 5. final refit on all data + future forecast with confidence intervals
final = SARIMAX(y, order=best_o, seasonal_order=best_s,
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=300)
ff = final.get_forecast(H_FUTURE)
ff_mean = ff.predicted_mean; ffci = ff.conf_int(alpha=.05)
ff_df = pd.DataFrame({"forecast": ff_mean, "lower_95": ffci.iloc[:, 0], "upper_95": ffci.iloc[:, 1]})
ff_df.index.name = "date"; ff_df.to_csv(P("future_forecast.csv"))
R["future_first"] = float(ff_mean.iloc[0]); R["future_last"] = float(ff_mean.iloc[-1])
R["future_last_lo"] = float(ffci.iloc[-1, 0]); R["future_last_hi"] = float(ffci.iloc[-1, 1])
R["future_last_date"] = str(ff_mean.index[-1].date())
R["future_first_date"] = str(ff_mean.index[0].date())

fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(y.loc["1992":], color="#1f4e79", lw=1.2, label="Observed")
ax.plot(ff_mean, color="#c0392b", lw=1.8, label=f"Forecast (+{H_FUTURE} months)")
ax.fill_between(ff_mean.index, ffci.iloc[:, 0], ffci.iloc[:, 1], color="#c0392b", alpha=.2, label="95% confidence interval")
ax.set_ylabel("CO$_2$ (ppm)"); ax.set_title("Future forecast with 95% confidence interval"); ax.legend(loc="upper left")
plt.tight_layout(); fig.savefig(P("fig7_future_forecast.png")); plt.close(fig)

# %% 6. interpretability layer: lag-feature gradient boosting + SHAP
def make_features(s):
    d = s.diff(12)                                   # seasonal difference = target
    X = pd.DataFrame(index=s.index)
    X["sdiff_lag1"] = d.shift(1)
    X["sdiff_lag2"] = d.shift(2)
    X["sdiff_lag3"] = d.shift(3)
    X["sdiff_lag12"] = d.shift(12)
    X["sdiff_roll6"] = d.shift(1).rolling(6).mean()
    X["mom_change_lag1"] = s.diff().shift(1)         # last month-on-month change
    X["month_sin"] = np.sin(2 * np.pi * s.index.month / 12)
    X["month_cos"] = np.cos(2 * np.pi * s.index.month / 12)
    return X, d
X_all, d_all = make_features(y)
ok = X_all.dropna().index.intersection(d_all.dropna().index)
X_all, d_all = X_all.loc[ok], d_all.loc[ok]
tr_idx = X_all.index < y_te.index[0]
gbm = GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=.03,
                                subsample=.8, random_state=SEED).fit(X_all[tr_idx], d_all[tr_idx])
te_idx = ~tr_idx
y_gbm = y.shift(12).loc[X_all.index[te_idx]] + gbm.predict(X_all[te_idx])   # one-step-ahead level forecast
R["gbm_holdout_1step"] = metrics(y_te, y_gbm)

# fair comparison: SARIMA one-step-ahead with parameters frozen from training
one_step = best.apply(y).get_prediction(start=len(y_tr)).predicted_mean
R["sarima_holdout_1step"] = metrics(y_te, one_step)
R["seasonal_naive_1step"] = metrics(y_te, y.shift(12).loc[y_te.index])

explainer = shap.TreeExplainer(gbm)
sv = explainer(X_all)
imp = pd.DataFrame({"feature": X_all.columns, "mean_abs_shap": np.abs(sv.values).mean(0)}).sort_values("mean_abs_shap", ascending=False)
imp["share_pct"] = 100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
imp.to_csv(P("table_shap_importance.csv"), index=False)
R["shap_importance"] = imp.round(5).to_dict("records")
R["gbm_gini_top"] = pd.Series(gbm.feature_importances_, index=X_all.columns).sort_values(ascending=False).round(4).head(3).to_dict()

plt.figure(); shap.summary_plot(sv.values, X_all, show=False, plot_size=(9, 4.6))
plt.tight_layout(); plt.savefig(P("fig8_shap_beeswarm.png")); plt.close("all")
plt.figure(); shap.plots.bar(sv, show=False); plt.tight_layout(); plt.savefig(P("fig9_shap_bar.png")); plt.close("all")
last = len(X_all) - 1
plt.figure(); shap.plots.waterfall(sv[last], show=False, max_display=8); plt.tight_layout()
plt.savefig(P("fig10_shap_waterfall.png")); plt.close("all")
R["waterfall_date"] = str(X_all.index[last].date())
R["waterfall_base"] = float(sv.base_values[last]); R["waterfall_pred"] = float(sv.base_values[last] + sv.values[last].sum())

# %% 7. dashboard (one-page summary figure)
fig = plt.figure(figsize=(14, 8.5)); gs = fig.add_gridspec(2, 3, hspace=.38, wspace=.32)
a = fig.add_subplot(gs[0, :2]); a.plot(y.loc["1992":], color="#1f4e79", lw=1); a.plot(ff_mean, color="#c0392b", lw=1.6)
a.fill_between(ff_mean.index, ffci.iloc[:, 0], ffci.iloc[:, 1], color="#c0392b", alpha=.2); a.set_title("Forecast + 95% CI")
b = fig.add_subplot(gs[0, 2]); names = ["Seasonal naive", "SARIMA", "GBM (lags)"]
vals = [R["seasonal_naive_1step"]["RMSE"], R["sarima_holdout_1step"]["RMSE"], R["gbm_holdout_1step"]["RMSE"]]
b.bar(names, vals, color=["#95a5a6", "#c0392b", "#27ae60"]); b.set_title("Hold-out RMSE, 1-step-ahead (ppm)"); b.tick_params(axis="x", labelrotation=15)
c = fig.add_subplot(gs[1, 0]); top = imp.head(8).iloc[::-1]; c.barh(top.feature, top.mean_abs_shap, color="#8e44ad"); c.set_title("SHAP: mean |impact| (ppm)")
d_ = fig.add_subplot(gs[1, 1]); plot_acf(resid, lags=30, ax=d_); d_.set_title("SARIMA residual ACF")
e = fig.add_subplot(gs[1, 2]); e.plot(stl.seasonal.loc["1995":"1999"], color="#16a085"); e.set_title("Seasonal component (STL)")
fig.suptitle(f"CO$_2$ Forecasting & Interpretability Dashboard - SARIMA{best_o}x{best_s}", fontsize=14, y=.97)
fig.savefig(P("dashboard.png"), bbox_inches="tight"); plt.close(fig)

# %% 8. save results + whitepaper
with open(P("results.json"), "w") as f:
    json.dump(R, f, indent=2, default=float)
pages = build_whitepaper(R, OUT, P("whitepaper.pdf"))
print(json.dumps({k: R[k] for k in ["best_order", "best_seasonal", "sarima_holdout", "seasonal_naive_holdout",
                                    "sarima_holdout_1step", "gbm_holdout_1step", "ci95_coverage", "ljung_box_p"]}, indent=2))
print(f"\nDone. Whitepaper pages: {pages} (target 5-6). Files are in ./{OUT}/")
