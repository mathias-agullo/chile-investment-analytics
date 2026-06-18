"""
Tab 1 — Markowitz Portfolio Optimizer.

Implements:
- Efficient frontier via Monte Carlo simulation (5 000 portfolios)
- Maximum Sharpe Ratio portfolio (scipy.optimize)
- Minimum Variance portfolio
- Maximum Return portfolio
- VaR (95%), Sharpe, drawdown metrics
- Backtesting vs. IPSA benchmark (last 12 months)
- Risk-free rate sourced from BCCh TPM
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import minimize, OptimizeResult

from utils.data_loader import load_stock_data, get_risk_free_rate, TICKER_LABELS
from utils.helpers import (
    portfolio_performance,
    historical_var,
    max_drawdown,
    fmt_pct,
)
from utils.i18n import t
from utils.groq_analyst import portfolio_commentary

warnings.filterwarnings("ignore")

N_SIMULATIONS = 5_000
PALETTE = px.colors.qualitative.Plotly

# ---------------------------------------------------------------------------
# Optimisation helpers
# ---------------------------------------------------------------------------

def _neg_sharpe(w, mean_ret, cov, rf, td=252):
    ret, vol, _ = portfolio_performance(w, mean_ret, cov, rf, td)
    return -((ret - rf) / vol) if vol > 0 else 1e6


def _portfolio_vol(w, cov, td=252):
    return np.sqrt(w @ cov @ w) * np.sqrt(td)


def _portfolio_ret(w, mean_ret, td=252):
    return np.dot(w, mean_ret) * td


def _optimise(objective, n_assets, constraints, bounds, args, n_starts=5):
    best: Optional[OptimizeResult] = None
    rng = np.random.default_rng(0)
    for _ in range(n_starts):
        w0 = rng.dirichlet(np.ones(n_assets))
        res = minimize(objective, w0, args=args, method="SLSQP",
                       bounds=bounds, constraints=constraints,
                       options={"ftol": 1e-12, "maxiter": 1000})
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        best = minimize(objective, np.ones(n_assets) / n_assets, args=args,
                        method="SLSQP", bounds=bounds, constraints=constraints)
    return best.x / best.x.sum()


def optimise_portfolios(
    mean_ret: np.ndarray, cov: np.ndarray, rf: float, n_assets: int
) -> Dict[str, np.ndarray]:
    td = 252
    bounds = [(0.0, 1.0)] * n_assets
    eq_con = {"type": "eq", "fun": lambda w: w.sum() - 1}

    w_sharpe = _optimise(_neg_sharpe, n_assets, [eq_con], bounds,
                          args=(mean_ret, cov, rf, td))
    w_minvar = _optimise(_portfolio_vol, n_assets, [eq_con], bounds,
                          args=(cov, td))
    max_con = [eq_con, {"type": "ineq", "fun": lambda w: 0.40 - w.max()}]
    w_maxret = _optimise(lambda w, m, t: -_portfolio_ret(w, m, t),
                          n_assets, max_con, bounds, args=(mean_ret, td))

    return {"Max Sharpe": w_sharpe, "Min Variance": w_minvar, "Max Return": w_maxret}


# ---------------------------------------------------------------------------
# Monte Carlo efficient frontier
# ---------------------------------------------------------------------------

def monte_carlo_frontier(
    mean_ret: np.ndarray, cov: np.ndarray, rf: float, n_sim: int = N_SIMULATIONS
) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = len(mean_ret)
    results = []
    for _ in range(n_sim):
        w = rng.dirichlet(np.ones(n))
        r, v, s = portfolio_performance(w, mean_ret, cov, rf)
        results.append({"Return": r, "Volatility": v, "Sharpe": s})
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------

def backtest(
    weights: np.ndarray,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    months: int = 12,
) -> Tuple[pd.Series, pd.Series]:
    start = prices.index[-1] - pd.DateOffset(months=months)
    p = prices[prices.index >= start]
    b = benchmark[benchmark.index >= start]
    port_ret = (p / p.iloc[0]).mul(weights, axis=1).sum(axis=1)
    bench_ret = b / b.iloc[0]
    return port_ret, bench_ret


# ---------------------------------------------------------------------------
# Streamlit render
# ---------------------------------------------------------------------------

def render_portfolio_tab(lang: str = "en") -> None:
    st.header(t("port_header", lang))
    st.markdown(t("port_method", lang))

    # ── Data loading ──────────────────────────────────────────────────────
    with st.spinner(t("port_loading", lang)):
        prices, returns, benchmark = load_stock_data(period="3y")
        rf = get_risk_free_rate()

    tickers = list(prices.columns)
    labels  = [TICKER_LABELS.get(tk, tk) for tk in tickers]

    st.success(t("port_loaded", lang).format(n=len(tickers), days=len(prices), rf=rf * 100))

    # ── Sidebar controls ──────────────────────────────────────────────────
    with st.sidebar:
        st.subheader(t("port_settings", lang))
        selected = st.multiselect(
            t("port_assets", lang), tickers,
            default=tickers,
            format_func=lambda tk: TICKER_LABELS.get(tk, tk),
        )
        backtest_months = st.slider(t("port_backtest_slider", lang), 6, 24, 12)

    if len(selected) < 3:
        st.warning(t("port_warn", lang))
        return

    p  = prices[selected].copy()
    r  = returns[selected].copy()
    lb = [TICKER_LABELS.get(tk, tk) for tk in selected]

    mean_ret = r.mean().values
    cov      = r.cov().values

    # ── Run optimisation ──────────────────────────────────────────────────
    with st.spinner(t("port_running", lang)):
        opt_weights = optimise_portfolios(mean_ret, cov, rf, len(selected))
        frontier_df = monte_carlo_frontier(mean_ret, cov, rf)

    # ── Metrics table ─────────────────────────────────────────────────────
    st.subheader(t("port_metrics", lang))
    metric_rows = []
    for name, w in opt_weights.items():
        ret, vol, sr = portfolio_performance(w, mean_ret, cov, rf)
        daily_port_ret = r.values @ w
        var95 = historical_var(pd.Series(daily_port_ret))
        port_value = (p / p.iloc[0]).mul(w, axis=1).sum(axis=1)
        mdd = max_drawdown(port_value)
        metric_rows.append({
            t("port_col_port", lang): name,
            t("port_col_ret",  lang): fmt_pct(ret),
            t("port_col_vol",  lang): fmt_pct(vol),
            t("port_col_sr",   lang): f"{sr:.3f}",
            t("port_col_var",  lang): fmt_pct(var95),
            t("port_col_mdd",  lang): fmt_pct(mdd),
        })
    st.dataframe(
        pd.DataFrame(metric_rows).set_index(t("port_col_port", lang)),
        use_container_width=True,
    )

    if st.button("🤖 " + ("Generar análisis" if lang == "es" else "Generate analysis"), key="groq_port"):
        with st.spinner("Groq — llama-3.3-70b…"):
            w_s = opt_weights["Max Sharpe"]
            w_v = opt_weights["Min Variance"]
            r_s, v_s, sr_s = portfolio_performance(w_s, mean_ret, cov, rf)
            r_v, v_v, _    = portfolio_performance(w_v, mean_ret, cov, rf)
            comment = portfolio_commentary(sr_s, r_s*100, v_s*100, r_v*100, v_v*100, rf, lang)
        if comment:
            st.info(comment)
        else:
            st.warning("GROQ_API_KEY not set in .env" if lang == "en" else "Agrega GROQ_API_KEY al .env")

    # ── Efficient frontier chart ──────────────────────────────────────────
    st.subheader(t("port_frontier", lang))
    fig_ef = go.Figure()

    fig_ef.add_trace(go.Scatter(
        x=frontier_df["Volatility"] * 100,
        y=frontier_df["Return"]     * 100,
        mode="markers",
        marker=dict(
            color=frontier_df["Sharpe"],
            colorscale="Viridis",
            size=4,
            opacity=0.6,
            colorbar=dict(title="Sharpe"),
        ),
        name=t("port_sim", lang),
        hovertemplate="Vol: %{x:.2f}%<br>Ret: %{y:.2f}%<extra></extra>",
    ))

    colors_opt  = {"Max Sharpe": "#EF553B", "Min Variance": "#00CC96", "Max Return": "#AB63FA"}
    symbols_opt = {"Max Sharpe": "star",    "Min Variance": "diamond",  "Max Return": "triangle-up"}
    for name, w in opt_weights.items():
        ret, vol, _ = portfolio_performance(w, mean_ret, cov, rf)
        fig_ef.add_trace(go.Scatter(
            x=[vol * 100], y=[ret * 100],
            mode="markers+text",
            marker=dict(color=colors_opt[name], size=16, symbol=symbols_opt[name],
                        line=dict(color="white", width=1.5)),
            text=[name], textposition="top center",
            name=name,
        ))

    fig_ef.update_layout(
        xaxis_title=t("port_xvol", lang),
        yaxis_title=t("port_yret", lang),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=480,
        template="plotly_white",
    )
    st.plotly_chart(fig_ef, use_container_width=True)

    # ── Weights breakdown ─────────────────────────────────────────────────
    st.subheader(t("port_weights", lang))
    cols = st.columns(3)
    for idx, (name, w) in enumerate(opt_weights.items()):
        mask = w > 0.005
        fig_pie = go.Figure(go.Pie(
            labels=[lb[i] for i, m in enumerate(mask) if m],
            values=[w[i]  for i, m in enumerate(mask) if m],
            hole=0.35,
            textinfo="label+percent",
        ))
        fig_pie.update_layout(title=name, height=320, showlegend=False,
                               margin=dict(t=40, b=10, l=10, r=10))
        cols[idx].plotly_chart(fig_pie, use_container_width=True)

    weights_df = pd.DataFrame(
        {name: {lb[i]: fmt_pct(ww) for i, ww in enumerate(w)}
         for name, w in opt_weights.items()}
    )
    with st.expander(t("port_weights_full", lang)):
        st.dataframe(weights_df, use_container_width=True)

    # ── Backtesting ───────────────────────────────────────────────────────
    st.subheader(t("port_backtest", lang).format(m=backtest_months))
    fig_bt = go.Figure()

    for name, w in opt_weights.items():
        port_cum, bench_cum = backtest(w, p, benchmark, backtest_months)
        if name == "Max Sharpe":
            fig_bt.add_trace(go.Scatter(
                x=bench_cum.index, y=(bench_cum - 1) * 100,
                mode="lines", name=t("port_bench", lang),
                line=dict(color="grey", dash="dash", width=2),
            ))
        fig_bt.add_trace(go.Scatter(
            x=port_cum.index, y=(port_cum - 1) * 100,
            mode="lines", name=name,
            line=dict(color=colors_opt[name], width=2),
        ))

    fig_bt.update_layout(
        xaxis_title=t("fi_date", lang),
        yaxis_title=t("port_cumret", lang),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        height=400,
        template="plotly_white",
        hovermode="x unified",
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────
    st.subheader(t("port_corr", lang))
    corr_mat = r.corr()
    corr_mat.columns = lb
    corr_mat.index   = lb
    fig_hm = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=lb, y=lb,
        colorscale="RdBu_r",
        zmin=-1, zmax=1,
        text=np.round(corr_mat.values, 2),
        texttemplate="%{text}",
        showscale=True,
    ))
    fig_hm.update_layout(height=420, template="plotly_white",
                          margin=dict(t=20, b=20, l=20, r=20))
    st.plotly_chart(fig_hm, use_container_width=True)
