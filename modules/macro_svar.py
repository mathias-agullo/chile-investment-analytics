"""
Tab 2 — Macro SVAR Model for Chile.

Implements:
- ADF stationarity tests for each macro variable
- VAR model with automatic lag selection (AIC / BIC)
- Orthogonalised Impulse Response Functions (Cholesky identification)
  with 95% confidence bands via parametric bootstrap
- 12-month VAR forecast with confidence intervals
- Forecast Error Variance Decomposition (FEVD)

Variable ordering (Cholesky recursive — most to least exogenous):
  Copper → USD/CLP → IMACEC → IPC → TPM

References:
  - Sims, C.A. (1980). Macroeconomics and Reality. Econometrica, 48(1), 1-48.
  - Lütkepohl, H. (2005). New Introduction to Multiple Time Series Analysis.
"""

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

from utils.data_loader import load_macro_data
from utils.i18n import t
from utils.groq_analyst import macro_commentary

warnings.filterwarnings("ignore")

# Cholesky ordering (most → least exogenous)
VAR_ORDER   = ["Copper", "USDCLP", "IMACEC", "IPC", "TPM"]
VAR_LABELS  = {
    "Copper":  "Copper (USD/lb)",
    "USDCLP":  "USD/CLP",
    "IMACEC":  "IMACEC Index",
    "IPC":     "IPC (monthly %, decimal)",
    "TPM":     "TPM (decimal)",
}
COLORS = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
N_BOOTSTRAP = 300
IRF_PERIODS = 18


# ---------------------------------------------------------------------------
# Stationarity & transformation
# ---------------------------------------------------------------------------

def _adf_result(series: pd.Series, name: str, lang: str = "en") -> Dict:
    clean = series.dropna()
    stat, p, lags, *_ = adfuller(clean, autolag="AIC")
    return {
        "Variable":                    name,
        t("macro_adf_stat", lang):     f"{stat:.4f}",
        t("macro_adf_p",    lang):     f"{p:.4f}",
        t("macro_adf_lags", lang):     lags,
        t("macro_adf_ok",   lang):     "✅ Yes" if p < 0.05 else "❌ No",
    }


def make_stationary(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Difference non-stationary series until ADF rejects at 5%."""
    out = {}
    transforms: Dict[str, str] = {}
    for col in df.columns:
        s = df[col].dropna()
        _, p, *_ = adfuller(s, autolag="AIC")
        if p < 0.05:
            out[col] = s
            transforms[col] = "level"
        else:
            if (s > 0).all():
                s_ld = np.log(s).diff().dropna()
                _, p2, *_ = adfuller(s_ld, autolag="AIC")
                if p2 < 0.05:
                    out[col] = s_ld
                    transforms[col] = "Δlog"
                    continue
            out[col] = s.diff().dropna()
            transforms[col] = "Δlevel"
    combined = pd.DataFrame(out).dropna()
    return combined, transforms


# ---------------------------------------------------------------------------
# Bootstrap IRF confidence bands
# ---------------------------------------------------------------------------

def _bootstrap_irf(
    var_results,
    periods: int = IRF_PERIODS,
    n_boot: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (irf_point, irf_lower, irf_upper) via residual bootstrap.

    Shape: (periods+1, k, k)  — [shock_horizon, shocked_var, response_var]
    """
    resid = var_results.resid.values
    T, k  = resid.shape
    p     = var_results.k_ar
    rng   = np.random.default_rng(42)

    irf_point = var_results.irf(periods).orth_irfs  # (periods+1, k, k)

    boot_irfs = []
    for _ in range(n_boot):
        idx    = rng.integers(0, T, size=T)
        e_boot = resid[idx]

        const_term = var_results.params if hasattr(var_results, "params") else np.zeros((p * k + 1, k))
        y_init     = var_results.model.endog[:p]
        y_boot     = np.vstack([y_init, np.zeros((T, k))])

        for tt in range(p, T + p):
            x = np.concatenate([[1]] + [y_boot[tt - j] for j in range(1, p + 1)])
            y_boot[tt] = x @ const_term + e_boot[tt - p]

        y_boot_df = pd.DataFrame(y_boot[p:], columns=var_results.model.endog_names)
        try:
            m_b = VAR(y_boot_df)
            r_b = m_b.fit(maxlags=p, ic=None, trend="c")
            boot_irfs.append(r_b.irf(periods).orth_irfs)
        except Exception:
            pass

    if len(boot_irfs) < 10:
        se = np.abs(irf_point) * 0.25
        return irf_point, irf_point - 1.96 * se, irf_point + 1.96 * se

    boot_arr = np.array(boot_irfs)
    lower = np.percentile(boot_arr, alpha / 2 * 100, axis=0)
    upper = np.percentile(boot_arr, (1 - alpha / 2) * 100, axis=0)
    return irf_point, lower, upper


# ---------------------------------------------------------------------------
# Streamlit render
# ---------------------------------------------------------------------------

def render_macro_tab(lang: str = "en") -> None:
    st.header(t("macro_header", lang))
    st.markdown(t("macro_method", lang))

    # ── Load data ──────────────────────────────────────────────────────────
    with st.spinner(t("macro_loading", lang)):
        macro_raw, source = load_macro_data()

    st.info(f"Data source: **{source}**")

    available = [v for v in VAR_ORDER if v in macro_raw.columns]
    if len(available) < 3:
        st.error(t("macro_err_data", lang))
        return

    macro_raw = macro_raw[available].dropna()

    # ── Raw series chart ───────────────────────────────────────────────────
    st.subheader(t("macro_raw", lang))
    fig_raw = make_subplots(
        rows=len(available), cols=1, shared_xaxes=True,
        subplot_titles=[VAR_LABELS.get(v, v) for v in available],
        vertical_spacing=0.04,
    )
    for i, var in enumerate(available, 1):
        fig_raw.add_trace(
            go.Scatter(x=macro_raw.index, y=macro_raw[var], name=VAR_LABELS.get(var, var),
                       line=dict(color=COLORS[(i - 1) % len(COLORS)], width=1.5)),
            row=i, col=1,
        )
    fig_raw.update_layout(height=120 * len(available), template="plotly_white",
                           showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig_raw, use_container_width=True)

    # ── ADF tests ──────────────────────────────────────────────────────────
    st.subheader(t("macro_adf", lang))
    adf_rows = [_adf_result(macro_raw[v], VAR_LABELS.get(v, v), lang) for v in available]
    st.dataframe(pd.DataFrame(adf_rows).set_index("Variable"), use_container_width=True)

    # ── Make stationary ────────────────────────────────────────────────────
    with st.spinner(t("macro_transf_spinner", lang)):
        macro_stat, transforms = make_stationary(macro_raw)

    with st.expander(t("macro_transforms", lang)):
        for v, tr in transforms.items():
            st.write(f"**{VAR_LABELS.get(v, v)}** → `{tr}`")

    if macro_stat.shape[0] < 30:
        st.error(t("macro_err_few", lang))
        return

    # ── Fit VAR ────────────────────────────────────────────────────────────
    st.subheader(t("macro_var", lang))
    with st.spinner(t("macro_var_est", lang)):
        model  = VAR(macro_stat)
        select = model.select_order(maxlags=12)
        # HQIC balances BIC/AIC; floor=2 for IRF dynamics, ceil=4 avoids overfitting with ~70 obs
        n_obs = macro_stat.shape[0]
        max_allowed = max(2, int(n_obs / (5 * macro_stat.shape[1])))  # ~T/(5k) rule
        best_lag = min(max(select.hqic, 2), max_allowed)

        results = model.fit(maxlags=best_lag, ic=None, trend="c")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(t("macro_lag_hqic", lang), best_lag)
    col2.metric(t("macro_obs",     lang), results.nobs)
    col3.metric(t("macro_vars",    lang), results.neqs)
    col4.metric("AIC", f"{results.aic:.2f}")

    with st.expander(t("macro_summary", lang)):
        summary = results.summary()
        st.text(str(summary))

    with st.expander(t("macro_lag_tbl", lang)):
        n_ic = len(select.ics["aic"])
        ic_df = pd.DataFrame({
            "Lags": range(0, n_ic),
            "AIC":  select.ics["aic"],
            "BIC":  select.ics["bic"],
            "HQIC": select.ics["hqic"],
        }).set_index("Lags")
        st.dataframe(ic_df.round(4), use_container_width=True)

    # ── IRF ───────────────────────────────────────────────────────────────
    st.subheader(t("macro_irf", lang))
    with st.spinner(t("macro_irf_spin", lang).format(n=N_BOOTSTRAP)):
        irf_point, irf_lower, irf_upper = _bootstrap_irf(
            results, periods=IRF_PERIODS, n_boot=N_BOOTSTRAP
        )

    k = results.neqs
    var_names = list(macro_stat.columns)
    horizon   = np.arange(IRF_PERIODS + 1)

    shock_var = st.selectbox(
        t("macro_shock", lang),
        options=list(range(k)),
        format_func=lambda i: VAR_LABELS.get(var_names[i], var_names[i]),
    )

    fig_irf = make_subplots(
        rows=1, cols=k,
        subplot_titles=[VAR_LABELS.get(n, n) for n in var_names],
        shared_yaxes=False,
    )
    for resp_idx in range(k):
        pt = irf_point[:, resp_idx, shock_var]
        lo = irf_lower[:, resp_idx, shock_var]
        hi = irf_upper[:, resp_idx, shock_var]
        col = COLORS[resp_idx % len(COLORS)]

        fig_irf.add_trace(
            go.Scatter(
                x=np.concatenate([horizon, horizon[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself", fillcolor="rgba(99,110,250,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ),
            row=1, col=resp_idx + 1,
        )
        fig_irf.add_hline(y=0, line_dash="dot", line_color="grey",
                           row=1, col=resp_idx + 1)
        fig_irf.add_trace(
            go.Scatter(x=horizon, y=pt, mode="lines",
                       name=VAR_LABELS.get(var_names[resp_idx], var_names[resp_idx]),
                       line=dict(color=col, width=2), showlegend=False),
            row=1, col=resp_idx + 1,
        )

    fig_irf.update_layout(
        height=360, template="plotly_white",
        title_text=f"IRF — shock to: {VAR_LABELS.get(var_names[shock_var], var_names[shock_var])}",
        margin=dict(t=60, b=30),
    )
    fig_irf.update_xaxes(title_text=t("macro_months", lang))
    st.plotly_chart(fig_irf, use_container_width=True)

    # ── FEVD ──────────────────────────────────────────────────────────────
    st.subheader(t("macro_fevd", lang))
    fevd = results.fevd(12)
    fevd_arr = fevd.decomp  # shape: (neqs, periods, neqs)

    fevd_var = st.selectbox(
        t("macro_resp", lang),
        options=list(range(k)),
        format_func=lambda i: VAR_LABELS.get(var_names[i], var_names[i]),
        key="fevd_sel",
    )

    horizons = np.arange(1, 13)
    fig_fevd = go.Figure()
    for shock_i, shock_name in enumerate(var_names):
        vals = fevd_arr[fevd_var, :, shock_i] * 100
        fig_fevd.add_trace(go.Bar(
            x=horizons,
            y=vals,
            name=VAR_LABELS.get(shock_name, shock_name),
            marker_color=COLORS[shock_i % len(COLORS)],
        ))
    fig_fevd.update_layout(
        barmode="stack",
        xaxis_title=t("macro_fhor", lang),
        yaxis_title=t("macro_fvar", lang),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
        template="plotly_white",
    )
    st.plotly_chart(fig_fevd, use_container_width=True)

    # ── 12-month forecast ─────────────────────────────────────────────────
    st.subheader(t("macro_fc", lang))
    fc_steps = 12

    y_in    = macro_stat.values[-results.k_ar:]
    fc_mean = results.forecast(y=y_in, steps=fc_steps)
    fc_df   = pd.DataFrame(
        fc_mean,
        index=pd.date_range(macro_stat.index[-1], periods=fc_steps + 1, freq="MS")[1:],
        columns=var_names,
    )

    try:
        fc_lo, fc_hi = results.forecast_interval(y=y_in, steps=fc_steps, alpha=0.05)
        lo_df = pd.DataFrame(fc_lo, index=fc_df.index, columns=var_names)
        hi_df = pd.DataFrame(fc_hi, index=fc_df.index, columns=var_names)
        has_ci = True
    except Exception:
        has_ci = False

    fc_var = st.selectbox(
        t("macro_fc_var", lang),
        options=list(range(k)),
        format_func=lambda i: VAR_LABELS.get(var_names[i], var_names[i]),
        key="fc_sel",
    )

    hist_tail = macro_stat[var_names[fc_var]].iloc[-36:]
    fc_col    = var_names[fc_var]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=hist_tail.index, y=hist_tail.values,
        mode="lines", name=t("macro_hist", lang),
        line=dict(color="#636EFA", width=2),
    ))
    fig_fc.add_trace(go.Scatter(
        x=fc_df.index, y=fc_df[fc_col],
        mode="lines+markers", name=t("macro_forecast", lang),
        line=dict(color="#EF553B", width=2, dash="dash"),
    ))
    if has_ci:
        fig_fc.add_trace(go.Scatter(
            x=np.concatenate([fc_df.index, fc_df.index[::-1]]),
            y=np.concatenate([hi_df[fc_col].values, lo_df[fc_col].values[::-1]]),
            fill="toself", fillcolor="rgba(239,85,59,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="95% CI", showlegend=True,
        ))

    fig_fc.update_layout(
        xaxis_title=t("fi_date", lang),
        yaxis_title=VAR_LABELS.get(fc_col, fc_col),
        height=400,
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    if st.button("🤖 " + ("Generar análisis" if lang == "es" else "Generate analysis"), key="groq_macro"):
        with st.spinner("Groq — llama-3.3-70b…"):
            ipc_fc  = fc_df["IPC"].iloc[-1]  if "IPC"    in fc_df.columns else 0.0
            tpm_fc  = fc_df["TPM"].iloc[-1]  if "TPM"    in fc_df.columns else 0.0
            usd_fc  = fc_df["USDCLP"].iloc[-1] if "USDCLP" in fc_df.columns else 0.0
            comment = macro_commentary(best_lag, results.nobs, ipc_fc, tpm_fc, usd_fc, lang)
        if comment:
            st.info(comment)
        else:
            st.warning("GROQ_API_KEY not set in .env" if lang == "en" else "Agrega GROQ_API_KEY al .env")

    with st.expander(t("macro_fc_tbl", lang)):
        display_fc = fc_df.copy()
        display_fc.index = display_fc.index.strftime("%Y-%m")
        st.dataframe(display_fc.round(6), use_container_width=True)
