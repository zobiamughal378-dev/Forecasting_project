"""Builds the 5-6 page data-science whitepaper (PDF) from results.json values + saved figures."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether)
from PIL import Image as PILImage

def _build(R, outdir, path, scale=1.0):
    ss = getSampleStyleSheet()
    body = ParagraphStyle("b", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.2, leading=12.2, alignment=TA_JUSTIFY, spaceAfter=4)
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13.5, textColor=colors.HexColor("#1f4e79"), spaceBefore=9, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.8, textColor=colors.HexColor("#2c3e50"), spaceBefore=5, spaceAfter=3)
    cap = ParagraphStyle("c", parent=body, fontSize=8.3, leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=7)
    title = ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#1f4e79"), spaceAfter=4)
    sub = ParagraphStyle("s", parent=body, alignment=TA_CENTER, fontSize=10.5, textColor=colors.HexColor("#444444"))
    small = ParagraphStyle("sm", parent=body, fontSize=8.6, leading=11, alignment=0, spaceAfter=2)

    def fig(name, caption, width=15.6 * cm):
        p = os.path.join(outdir, name)
        w, h = PILImage.open(p).size
        width = width * scale
        im = Image(p, width=width, height=width * h / w)
        return KeepTogether([im, Paragraph(caption, cap)])

    def table(data, widths, hdr=True, fs=8.3):
        t = Table(data, colWidths=widths, repeatRows=1 if hdr else 0)
        st = [("FONTSIZE", (0, 0), (-1, -1), fs), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
              ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
              ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#bbbbbb")),
              ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f6fa")]),
              ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
              ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        t.setStyle(TableStyle(st)); return t

    f2 = lambda v: f"{v:.2f}"
    f3 = lambda v: f"{v:.3f}"
    bo, bs = R["best_order"], R["best_seasonal"]
    spec = f"SARIMA({bo[0]},{bo[1]},{bo[2]})x({bs[0]},{bs[1]},{bs[2]},{bs[3]})"
    sa, nv = R["sarima_holdout"], R["seasonal_naive_holdout"]
    sa1, nv1, gb1 = R["sarima_holdout_1step"], R["seasonal_naive_1step"], R["gbm_holdout_1step"]
    imp = R["shap_importance"]; top3 = imp[:3]
    best_name, best_rmse = min([("the seasonal-naive benchmark", nv1["RMSE"]), (spec, sa1["RMSE"]), ("the SHAP-explained gradient-boosting model", gb1["RMSE"])], key=lambda t: t[1])
    resid_ok = R["ljung_box_p"] > .05
    adf_raw = R["stationarity"][0]; adf_dd = R["stationarity"][3]

    S = []
    S += [Paragraph("Forecasting Atmospheric CO<sub>2</sub> with SARIMA and Explaining the Drivers with SHAP", title),
          Paragraph("A data science whitepaper on seasonal time-series modelling, uncertainty quantification and model interpretability", sub),
          Paragraph("Task 4 - Mini Project: Time-Series Forecasting Model &amp; Advanced Model Interpretability Dashboard", sub), Spacer(1, 8)]

    S += [Paragraph("1. Executive Summary", h1),
          Paragraph(f"This paper fits a seasonal ARIMA model to {R['n_obs']} monthly observations of atmospheric CO<sub>2</sub> measured at Mauna Loa "
                    f"({R['start']} to {R['end']}), tests the series for stationarity, removes trend and seasonality through differencing, and produces "
                    f"forecasts with 95% confidence intervals. The selected specification, <b>{spec}</b>, was chosen by an AIC search over {R['grid_size']} candidate models. "
                    f"On a {R['H']}-month hold-out the model reached an RMSE of <b>{f3(sa['RMSE'])} ppm</b> (MAE {f3(sa['MAE'])}, MAPE {f3(sa['MAPE'])}%) for the multi-step forecast, and its 95% "
                    f"intervals covered <b>{100*R['ci95_coverage']:.0f}%</b> of the realised values. A second, machine-learning model built on lagged seasonal differences was explained with SHAP "
                    f"to quantify exactly how much each input contributes to a forecast. The most influential feature was <b>{top3[0]['feature']}</b> "
                    f"({top3[0]['share_pct']:.1f}% of total attribution), followed by {top3[1]['feature']} and {top3[2]['feature']}.", body)]

    S += [Paragraph("2. Introduction and Objectives", h1),
          Paragraph("Accurate forecasts are only useful when decision makers can trust them. Classical statistical models such as SARIMA give calibrated uncertainty "
                    "and interpretable coefficients, whereas flexible machine-learning models often reach good accuracy but behave as black boxes. This project combines both "
                    "views. The objectives are: (i) load a historical sequential climate dataset; (ii) diagnose non-stationarity and apply a seasonal differencing pipeline; "
                    "(iii) train a SARIMA forecaster that predicts future values with confidence intervals; (iv) wrap a model in an interpretability layer (SHAP) that "
                    "measures exact feature-importance weights; and (v) report the findings in this whitepaper and a one-page dashboard.", body)]

    S += [Paragraph("3. Data", h1),
          Paragraph(f"The data are the Keeling/Scripps Mauna Loa weekly atmospheric CO<sub>2</sub> record distributed with the <i>statsmodels</i> library. The weekly series contains {R['n_missing_weekly']} missing values; "
                    f"it was aggregated to monthly means and remaining gaps were filled by time-based linear interpolation. Values range from {f2(R['y_min'])} to {f2(R['y_max'])} ppm "
                    f"(mean {f2(R['y_mean'])} ppm). Figure 1 shows the long-run upward trend together with a regular within-year cycle caused by seasonal vegetation uptake in the northern hemisphere.", body),
          fig("fig1_eda.png", "Figure 1 - Monthly CO<sub>2</sub> series (top) and a five-year zoom showing the annual cycle (bottom).", 12.0 * cm),
          Paragraph(f"An STL decomposition (Figure 2) separates the series into trend, seasonal and remainder components. The trend rises by about {f2(R['trend_gain_ppm'])} ppm over the sample, "
                    f"the seasonal cycle has a peak-to-trough amplitude of about {f2(R['seasonal_amp_ppm'])} ppm, and the strength of trend and seasonality are "
                    f"{f3(R['trend_strength'])} and {f3(R['seasonal_strength'])} respectively (1 = perfectly deterministic). Both structures must be handled by the model.", body),
          fig("fig2_stl.png", "Figure 2 - STL decomposition (period = 12) into trend, seasonal and remainder.", 10.6 * cm)]

    st = [["Series", "ADF stat", "ADF p", "KPSS stat", "KPSS p", "Verdict"]]
    for r in R["stationarity"]:
        st.append([r["Series"], f3(r["ADF_stat"]), f3(r["ADF_p"]), f3(r["KPSS_stat"]), f3(r["KPSS_p"]), r["Verdict"]])
    S += [Paragraph("4. Stationarity Analysis and Seasonal Differencing", h1),
          Paragraph("SARIMA assumes that, after differencing, the series has a constant mean and variance structure. Two complementary tests were used: the Augmented Dickey-Fuller (ADF) test, whose "
                    "null hypothesis is a unit root (non-stationarity), and the KPSS test, whose null hypothesis is stationarity. A series is declared stationary only when ADF rejects (p &lt; 0.05) "
                    "<i>and</i> KPSS does not reject (p &gt; 0.05). Using both tests protects against the low power of either test alone.", body),
          table(st, [5.2 * cm, 1.9 * cm, 1.7 * cm, 2.0 * cm, 1.7 * cm, 3.0 * cm]), Spacer(1, 3),
          Paragraph(f"Table 1 - Stationarity tests after each transformation. The raw series is {adf_raw['Verdict'].lower()} (ADF p = {f3(adf_raw['ADF_p'])}, KPSS p = {f3(adf_raw['KPSS_p'])}); "
                    f"after one regular and one seasonal (lag-12) difference the verdict is <b>{adf_dd['Verdict'].lower()}</b> (ADF p = {f3(adf_dd['ADF_p'])}, KPSS p = {f3(adf_dd['KPSS_p'])}). "
                    "KPSS p-values are truncated by the test to the interval [0.01, 0.10].", cap),
          Paragraph("The first difference removes the stochastic trend, and the seasonal difference <i>y<sub>t</sub> - y<sub>t-12</sub></i> removes the annual cycle (Figure 3). Their combination "
                    "motivates d = 1 and D = 1 with seasonal period s = 12 in the SARIMA specification. The ACF/PACF of the doubly differenced series (saved as fig4_acf_pacf.png) show short-lag and seasonal-lag "
                    "structure that guides the candidate range for the autoregressive (p, P) and moving-average (q, Q) orders.", body),
          fig("fig3_differencing.png", "Figure 3 - Effect of regular and seasonal differencing.", 9.6 * cm),
          ]

    top5 = [["Rank", "Candidate model", "AIC", "BIC"]] + [[str(i + 1), g["spec"], f2(g["AIC"]), f2(g["BIC"])] for i, g in enumerate(R["grid_top5"])]
    S += [Paragraph("5. SARIMA Model and Forecast", h1),
          Paragraph(f"A SARIMA(p,d,q)x(P,D,Q)<sub>s</sub> model applies the differencing above and models the result with autoregressive and moving-average terms at both the regular and seasonal lags. "
                    f"Parameters are estimated by maximum likelihood on a Kalman-filter state-space form. Holding out the final {R['H']} months, a grid search over p, q in {{0,1,2}} and P, Q in {{0,1}} "
                    f"(d = D = 1, s = 12) was run on the training window and ranked by the Akaike information criterion (Table 2).", body),
          table(top5, [1.5 * cm, 9.2 * cm, 2.6 * cm, 2.6 * cm]), Spacer(1, 3),
          Paragraph("Table 2 - Five best candidates by AIC on the training window.", cap),
          Paragraph(f"The winner, <b>{spec}</b>, was used to forecast the {R['H']}-month hold-out (Figure 4). Prediction intervals come from the model's forecast error variance, which grows with the horizon. "
                    f"The 95% interval was on average {f2(R['ci95_mean_width'])} ppm wide and contained {100*R['ci95_coverage']:.0f}% of the actual values.", body),
          fig("fig5_holdout_forecast.png", "Figure 4 - Hold-out forecast with 95% prediction interval against a seasonal-naive benchmark.", 12.0 * cm)]

    cmp_ = [["Model", "Horizon", "MAE", "RMSE", "MAPE %"],
            ["SARIMA", f"{R['H']}-step", f3(sa["MAE"]), f3(sa["RMSE"]), f3(sa["MAPE"])],
            ["Seasonal naive", f"{R['H']}-step", f3(nv["MAE"]), f3(nv["RMSE"]), f3(nv["MAPE"])],
            ["SARIMA (params frozen)", "1-step", f3(sa1["MAE"]), f3(sa1["RMSE"]), f3(sa1["MAPE"])],
            ["Seasonal naive", "1-step", f3(nv1["MAE"]), f3(nv1["RMSE"]), f3(nv1["MAPE"])],
            ["Gradient boosting + lags", "1-step", f3(gb1["MAE"]), f3(gb1["RMSE"]), f3(gb1["MAPE"])]]
    S += [Paragraph("5.1 Accuracy and benchmarks", h2),
          table(cmp_, [5.2 * cm, 2.4 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]), Spacer(1, 3),
          Paragraph("Table 3 - Hold-out accuracy in ppm. Multi-step rows forecast all months from the end of training; one-step rows predict each month given all earlier observations.", cap),
          Paragraph(f"For the {R['H']}-step forecast SARIMA {'improved on' if sa['RMSE'] < nv['RMSE'] else 'did not improve on'} the seasonal-naive benchmark (RMSE {f3(sa['RMSE'])} versus {f3(nv['RMSE'])} ppm). "
                    f"Among one-step-ahead predictors the lowest RMSE ({f3(best_rmse)} ppm) was achieved by {best_name}. Because the benchmark simply repeats last year's value, "
                    "it cannot capture the persistent upward trend, which is why differencing-based models are the natural reference for this series.", body),
          Paragraph("5.2 Residual diagnostics", h2),
          Paragraph(f"A well-specified model leaves residuals that resemble white noise. The Ljung-Box test at 24 lags gave p = {f3(R['ljung_box_p'])}, "
                    f"{'so there is no significant remaining autocorrelation' if resid_ok else 'indicating some remaining autocorrelation that a richer specification or exogenous regressors could address'}. "
                    f"The Jarque-Bera normality test gave p = {f3(R['jarque_bera_p'])} and the residual standard deviation is {f3(R['resid_std'])} ppm (Figure 5).", body),
          fig("fig6_diagnostics.png", "Figure 5 - Standardised residuals, histogram/density, Q-Q plot and correlogram.", 10.2 * cm)]

    S += [Paragraph("5.3 Future forecast", h2),
          Paragraph(f"After selecting the specification, the model was refit on the complete history and projected {R['H_future']} months ahead (Figure 6). The point forecast moves from "
                    f"{f2(R['future_first'])} ppm in {R['future_first_date'][:7]} to {f2(R['future_last'])} ppm in {R['future_last_date'][:7]}, where the 95% confidence interval spans "
                    f"{f2(R['future_last_lo'])} to {f2(R['future_last_hi'])} ppm. The widening band reflects the accumulation of uncertainty from the integrated (differenced) components.", body),
          fig("fig7_future_forecast.png", "Figure 6 - Future forecast with 95% confidence interval.", 12.0 * cm)]

    co = [["Term", "Coefficient", "Std. error", "p-value"]] + [[c["term"], f3(c["coef"]), f3(c["se"]), f3(c["p"])] for c in R["sarima_coefs"]]
    S += [Paragraph("6. Model Interpretability", h1),
          Paragraph("Interpretability is addressed at two levels. SARIMA is <i>intrinsically</i> interpretable: each fitted coefficient has a direct meaning (Table 4). Because SARIMA is linear, however, "
                    "SHAP and LIME - which target flexible models - are applied to a complementary tree-ensemble forecaster whose predictions can be decomposed exactly.", body),
          table(co, [5.0 * cm, 3.0 * cm, 3.0 * cm, 3.0 * cm]), Spacer(1, 3),
          Paragraph("Table 4 - Fitted SARIMA coefficients with standard errors. Moving-average terms close to -1 indicate that shocks are largely absorbed by the differencing, a typical pattern for series that were possibly over-differenced or strongly seasonal.", cap),
          Paragraph("6.1 SHAP on a lag-feature model", h2),
          Paragraph("A gradient-boosting regressor (400 depth-3 trees, learning rate 0.03) predicts the seasonal difference <i>y<sub>t</sub> - y<sub>t-12</sub></i> from: three short lags of that "
                    "difference, its 12-month lag, a 6-month rolling mean, last month's month-on-month change, and sine/cosine encodings of the calendar month. Predicting the difference rather than "
                    "the level avoids the well-known inability of trees to extrapolate a trend. The forecast of the level is recovered by adding back <i>y<sub>t-12</sub></i>. "
                    "SHAP values (Shapley values from cooperative game theory, computed exactly for trees by TreeSHAP) distribute each prediction among the features so that the contributions "
                    "plus the base value sum to the model output.", body),
          fig("fig8_shap_beeswarm.png", "Figure 7 - SHAP beeswarm: each dot is one month; colour shows the feature value and horizontal position its effect on the prediction (ppm).", 10.2 * cm)]

    ti = [["Rank", "Feature", "Mean |SHAP| (ppm)", "Share of total %"]] + [[str(i + 1), r["feature"], f"{r['mean_abs_shap']:.4f}", f"{r['share_pct']:.1f}"] for i, r in enumerate(imp)]
    S += [Paragraph(f"Table 5 gives the exact global importance weights. The top three features - {top3[0]['feature']}, {top3[1]['feature']} and {top3[2]['feature']} - jointly account for "
                    f"{sum(t['share_pct'] for t in top3):.1f}% of total attribution. The ranking is consistent with the tree model's own impurity-based importances, whose top entries were "
                    f"{', '.join(R['gbm_gini_top'].keys())}; SHAP is preferred because it is additive, locally exact and does not favour high-cardinality features.", body),
          table(ti, [1.5 * cm, 5.2 * cm, 4.2 * cm, 4.2 * cm]), Spacer(1, 3),
          Paragraph("Table 5 - Global feature-importance weights from SHAP (mean absolute value over all modelled months).", cap),
          Paragraph(f"6.2 Local explanation", h2),
          Paragraph(f"Figure 8 explains a single forecast, the month {R['waterfall_date'][:7]}. Starting from the average model output of {R['waterfall_base']:.3f} ppm, each feature pushes the predicted "
                    f"year-on-year change up or down until it reaches the final prediction of {R['waterfall_pred']:.3f} ppm. Such a breakdown lets an analyst audit any individual forecast.", body),
          fig("fig10_shap_waterfall.png", "Figure 8 - SHAP waterfall plot for the most recent month.", 9.2 * cm)]

    S += [Paragraph("6.3 Dashboard", h2),
          Paragraph("Figure 9 consolidates the forecast, model comparison, SHAP ranking, residual ACF and seasonal component into one view for stakeholders.", body),
          fig("dashboard.png", "Figure 9 - Forecasting and interpretability dashboard.", 13.4 * cm)]

    S += [Paragraph("7. Discussion and Limitations", h1),
          Paragraph("<b>Uncertainty.</b> SARIMA intervals assume Gaussian, homoscedastic errors; the residual tests above show how well this holds. Intervals widen with horizon, so long-range forecasts should be read as scenarios, not guarantees. "
                    "<b>Structural change.</b> The fitted model extrapolates the historical growth pattern and cannot anticipate policy-driven or physical regime changes. "
                    "<b>Interpretation scope.</b> SHAP explains the <i>model</i>, not the physical world: a large attribution to a lag means the model relies on persistence, not that the lag causes CO<sub>2</sub> changes. "
                    "<b>Data.</b> Series ends in 2001; applying the pipeline to newer observations is straightforward since the code is data-source agnostic once a monthly Series is provided. "
                    "<b>Future work</b> includes exogenous regressors (e.g. ENSO index), Prophet as a comparator, rolling-origin cross-validation and conformal prediction intervals.", body),
          Paragraph("8. Conclusion", h1),
          Paragraph(f"A seasonal ARIMA model selected by information criteria, preceded by a rigorous stationarity and differencing workflow, produces accurate monthly CO<sub>2</sub> forecasts with interval estimates "
                    f"(hold-out RMSE {f3(sa['RMSE'])} ppm over {R['H']} months). Pairing it with a SHAP-explained lag model provides exact, auditable feature weights, so stakeholders can see both "
                    f"<i>what</i> is forecast and <i>why</i>. All code, tables and figures are reproducible with <font face='Courier'>python project.py</font>.", body),
          Paragraph("References", h2)]
    for ref in ["Box, G., Jenkins, G., Reinsel, G. (2015). <i>Time Series Analysis: Forecasting and Control</i>, 5th ed. Wiley.",
                "Hyndman, R.J., Athanasopoulos, G. (2021). <i>Forecasting: Principles and Practice</i>, 3rd ed. OTexts.",
                "Dickey, D., Fuller, W. (1979). Distribution of the estimators for autoregressive time series with a unit root. <i>JASA</i> 74.",
                "Kwiatkowski, D., Phillips, P., Schmidt, P., Shin, Y. (1992). Testing the null hypothesis of stationarity. <i>J. Econometrics</i> 54.",
                "Cleveland, R. et al. (1990). STL: A seasonal-trend decomposition procedure based on loess. <i>J. Official Statistics</i> 6.",
                "Lundberg, S., Lee, S.-I. (2017). A unified approach to interpreting model predictions. <i>NeurIPS</i> 30.",
                "Keeling, C.D., Whorf, T.P. Atmospheric CO<sub>2</sub> records from Mauna Loa, Scripps Institution of Oceanography."]:
        S.append(Paragraph(ref, small))

    def footer(c, d):
        c.saveState(); c.setFont("Helvetica", 8); c.setFillColor(colors.grey)
        c.drawString(2 * cm, 1.1 * cm, "CO2 Forecasting & Interpretability - Task 4 Whitepaper"); c.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {d.page}")
        c.restoreState()
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                            title="Time-Series Forecasting & Interpretability Whitepaper")
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    return doc.page


def build_whitepaper(R, outdir, path, max_pages=6):
    """Shrinks figures automatically until the paper fits in max_pages (target: 5-6 pages)."""
    scale = 1.0
    while True:
        pages = _build(R, outdir, path, scale)
        if pages <= max_pages or scale < .6:
            return pages
        scale -= .08
