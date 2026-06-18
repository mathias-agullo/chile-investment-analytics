"""
Tab 3 — Fixed Income Analyzer for Chilean sovereign bonds.

Implements:
- Real BCP (Bonos del Banco Central en pesos) and BCU (en UF) yield curves
- Nelson-Siegel (1987) yield curve fitting via scipy.curve_fit
- Macaulay duration, Modified duration, Convexity, DV01
- BCP–BCU breakeven inflation spread analysis
- Interactive bond pricing calculator
- Historical yield evolution and correlation heatmap

References:
  - Nelson, C.R. & Siegel, A.F. (1987). Parsimonious Modeling of Yield Curves.
    Journal of Business, 60(4), 473-489.
  - Fabozzi, F.J. (2012). Fixed Income Mathematics, Analysis and Valuation, 4th ed.
"""

import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from scipy.optimize import curve_fit

from utils.data_loader import load_yield_curve_data
from utils.helpers import (
    nelson_siegel,
    bond_price,
    macaulay_duration,
    modified_duration,
    bond_convexity,
    dv01,
    fmt_pct,
    fmt_num,
)
from utils.i18n import t
from utils.groq_analyst import fixed_income_commentary

warnings.filterwarnings("ignore")

# BCP20 is not published in BCCh secondary market — only BCP 2/5/10 available
BCP_COLS  = ["BCP2", "BCP5", "BCP10"]
BCU_COLS  = ["BCU2", "BCU5", "BCU10", "BCU20"]
MATURITIES_BCP = np.array([2.0, 5.0, 10.0])
MATURITIES_BCU = np.array([2.0, 5.0, 10.0, 20.0])
FINE_TAU   = np.linspace(0.25, 21, 200)

BCP_COLOR    = "#636EFA"
BCU_COLOR    = "#EF553B"
SPREAD_COLOR = "#00CC96"


# ---------------------------------------------------------------------------
# Nelson-Siegel fitting
# ---------------------------------------------------------------------------

def fit_nelson_siegel(maturities: np.ndarray, yields: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    p0 = [yields.mean(), yields[0] - yields[-1], 0.0, 1.5]
    try:
        popt, _ = curve_fit(
            nelson_siegel, maturities, yields,
            p0=p0, maxfev=10_000,
            bounds=([-0.2, -0.3, -0.3, 0.1], [0.25, 0.3, 0.3, 10.0]),
        )
    except Exception:
        popt = p0
    fitted = nelson_siegel(FINE_TAU, *popt)
    return popt, fitted


# ---------------------------------------------------------------------------
# Duration / convexity for the yield curve nodes
# ---------------------------------------------------------------------------

def curve_analytics(
    ytm_dict: Dict[str, Tuple[float, float]],
    face: float = 1_000_000,
    freq: int = 2,
    lang: str = "en",
) -> pd.DataFrame:
    rows = []
    for label, (years, ytm) in ytm_dict.items():
        coupon_rate = ytm
        mac  = macaulay_duration(face, coupon_rate, years, ytm, freq)
        mod  = modified_duration(mac, ytm, freq)
        conv = bond_convexity(face, coupon_rate, years, ytm, freq)
        dv   = dv01(face, coupon_rate, years, ytm, freq)
        p    = bond_price(face, coupon_rate, years, ytm, freq)
        rows.append({
            "Instrument":              label,
            t("fi_mat_yrs", lang):     years,
            "YTM":                     fmt_pct(ytm),
            t("fi_clean",   lang):     f"${p:,.0f}",
            t("fi_mac",     lang):     f"{mac:.3f} yrs",
            t("fi_mod",     lang):     f"{mod:.3f} yrs",
            t("fi_conv",    lang):     f"{conv:.4f}",
            "DV01 (CLP/bp)":           f"${dv:,.0f}",
        })
    return pd.DataFrame(rows).set_index("Instrument")


# ---------------------------------------------------------------------------
# Streamlit render
# ---------------------------------------------------------------------------

def render_fixed_income_tab(lang: str = "en") -> None:
    st.header(t("fi_header", lang))
    st.markdown(t("fi_method", lang))

    # ── Load data ──────────────────────────────────────────────────────────
    with st.spinner(t("fi_loading", lang)):
        yield_df, source = load_yield_curve_data()

    st.info(f"Data source: **{source}**")

    bcp_available = [c for c in BCP_COLS if c in yield_df.columns]
    bcu_available = [c for c in BCU_COLS if c in yield_df.columns]

    if len(bcp_available) < 2 or len(bcu_available) < 2:
        st.error(
            f"Insufficient yield curve data: "
            f"{len(bcp_available)} BCP series and {len(bcu_available)} BCU series available "
            f"(need at least 2 of each). Check data source."
        )
        return

    _mat_map = {"BCP2": 2.0, "BCP5": 5.0, "BCP10": 10.0,
                "BCU2": 2.0, "BCU5": 5.0, "BCU10": 10.0, "BCU20": 20.0}
    latest = yield_df.iloc[-1]
    bcp_yields = np.array([latest.get(c, np.nan) for c in bcp_available])
    bcu_yields = np.array([latest.get(c, np.nan) for c in bcu_available])
    bcp_mats   = np.array([_mat_map[c] for c in bcp_available])
    bcu_mats   = np.array([_mat_map[c] for c in bcu_available])

    mask_bcp = ~np.isnan(bcp_yields)
    mask_bcu = ~np.isnan(bcu_yields)
    bcp_yields = bcp_yields[mask_bcp]; bcp_mats = bcp_mats[mask_bcp]
    bcu_yields = bcu_yields[mask_bcu]; bcu_mats = bcu_mats[mask_bcu]

    # ── Nelson-Siegel fit ─────────────────────────────────────────────────
    st.subheader(t("fi_ns", lang))
    tab_cur, tab_hist = st.tabs([t("fi_cur", lang), t("fi_hist", lang)])

    with tab_cur:
        fig_ns = go.Figure()

        if len(bcp_yields) >= 3:
            p_bcp, fit_bcp = fit_nelson_siegel(bcp_mats, bcp_yields)
            fig_ns.add_trace(go.Scatter(
                x=bcp_mats, y=bcp_yields * 100,
                mode="markers", name="BCP (market)",
                marker=dict(color=BCP_COLOR, size=12, symbol="circle"),
            ))
            fig_ns.add_trace(go.Scatter(
                x=FINE_TAU, y=fit_bcp * 100,
                mode="lines", name="BCP Nelson-Siegel fit",
                line=dict(color=BCP_COLOR, width=2),
            ))
            with st.expander(t("fi_bcp_params", lang)):
                st.markdown(
                    f"β₀={p_bcp[0]*100:.3f}%  β₁={p_bcp[1]*100:.3f}%  "
                    f"β₂={p_bcp[2]*100:.3f}%  λ={p_bcp[3]:.3f}"
                )

        if len(bcu_yields) >= 3:
            p_bcu, fit_bcu = fit_nelson_siegel(bcu_mats, bcu_yields)
            fig_ns.add_trace(go.Scatter(
                x=bcu_mats, y=bcu_yields * 100,
                mode="markers", name="BCU (market)",
                marker=dict(color=BCU_COLOR, size=12, symbol="diamond"),
            ))
            fig_ns.add_trace(go.Scatter(
                x=FINE_TAU, y=fit_bcu * 100,
                mode="lines", name="BCU Nelson-Siegel fit",
                line=dict(color=BCU_COLOR, width=2, dash="dash"),
            ))
            if len(bcp_yields) >= 3:
                breakeven = fit_bcp - fit_bcu
                fig_ns.add_trace(go.Scatter(
                    x=FINE_TAU, y=breakeven * 100,
                    mode="lines", name="Breakeven Inflation (BCP−BCU)",
                    line=dict(color=SPREAD_COLOR, width=2, dash="dot"),
                ))

        fig_ns.update_layout(
            xaxis_title=t("fi_mat_yrs",  lang),
            yaxis_title=t("fi_yield_sp", lang),
            height=440,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig_ns, use_container_width=True)

    with tab_hist:
        st.markdown(f"**{t('fi_hist', lang)} — BCP / BCU**")
        cols_avail = bcp_available + bcu_available
        selected_hist = st.multiselect(
            t("fi_hist_sel", lang), cols_avail,
            default=["BCP5", "BCP10", "BCU5", "BCU10"]
                    if all(c in cols_avail for c in ["BCP5", "BCP10", "BCU5", "BCU10"])
                    else cols_avail[:4],
            format_func=lambda c: f"{c[:3]} {c[3:]}Y",
        )
        if selected_hist:
            fig_hist = go.Figure()
            color_seq = px.colors.qualitative.Plotly
            for i, col in enumerate(selected_hist):
                fig_hist.add_trace(go.Scatter(
                    x=yield_df.index,
                    y=yield_df[col] * 100,
                    mode="lines",
                    name=f"{col[:3]} {col[3:]}Y",
                    line=dict(color=color_seq[i % len(color_seq)], width=1.8),
                    connectgaps=True,
                ))
            fig_hist.update_layout(
                xaxis_title=t("fi_date",    lang),
                yaxis_title=t("fi_yld_pct", lang),
                height=400,
                template="plotly_white",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    # ── Spread / breakeven ─────────────────────────────────────────────────
    st.subheader(t("fi_spread", lang))
    spread_pairs = [
        (b, u) for b, u in zip(bcp_available, bcu_available)
        if b in yield_df.columns and u in yield_df.columns
    ]
    if spread_pairs:
        fig_spread = go.Figure()
        mat_labels = {
            ("BCP2",  "BCU2"):  "2Y",
            ("BCP5",  "BCU5"):  "5Y",
            ("BCP10", "BCU10"): "10Y",
            ("BCP20", "BCU20"): "20Y",
        }
        cpal = px.colors.qualitative.Safe
        for i, (b, u) in enumerate(spread_pairs):
            spread_s = (yield_df[b] - yield_df[u]) * 100
            label = mat_labels.get((b, u), b + "-" + u)
            fig_spread.add_trace(go.Scatter(
                x=spread_s.index, y=spread_s,
                mode="lines", name=f"BEI {label}",
                line=dict(color=cpal[i % len(cpal)], width=1.8),
                connectgaps=True,
            ))
        fig_spread.add_hline(y=3.0, line_dash="dot", line_color="grey",
                              annotation_text=t("fi_target", lang))
        fig_spread.update_layout(
            xaxis_title=t("fi_date", lang),
            yaxis_title=t("fi_bei",  lang),
            height=380,
            template="plotly_white",
            hovermode="x unified",
        )
        st.plotly_chart(fig_spread, use_container_width=True)

    if st.button("🤖 " + ("Generar análisis" if lang == "es" else "Generate analysis"), key="groq_fi"):
        with st.spinner("Groq — llama-3.3-70b…"):
            bei_vals = {}
            for b, u in zip(bcp_available, bcu_available):
                if b in yield_df.columns and u in yield_df.columns:
                    mat = _mat_map.get(b, "")
                    bei_vals[f"{int(mat)}Y"] = float((yield_df[b].iloc[-1] - yield_df[u].iloc[-1]) * 100)
            bcp10 = float(latest.get("BCP10")) if "BCP10" in latest.index else None
            bcu10 = float(latest.get("BCU10")) if "BCU10" in latest.index else None
            tpm_val = None
            try:
                from utils.data_loader import get_risk_free_rate
                tpm_val = get_risk_free_rate()
            except Exception:
                pass
            comment = fixed_income_commentary(bei_vals, bcp10, bcu10, tpm_val, lang)
        if comment:
            st.info(comment)
        else:
            st.warning("GROQ_API_KEY not set in .env" if lang == "en" else "Agrega GROQ_API_KEY al .env")

    # ── Duration / Convexity table ─────────────────────────────────────────
    st.subheader(t("fi_dur", lang))
    if len(bcp_yields) > 0 or len(bcu_yields) > 0:
        ytm_inputs = {}
        for m, y in zip(bcp_mats, bcp_yields):
            ytm_inputs[f"BCP {int(m)}Y"] = (m, float(y))
        for m, y in zip(bcu_mats, bcu_yields):
            ytm_inputs[f"BCU {int(m)}Y"] = (m, float(y))
        analytics_df = curve_analytics(ytm_inputs, lang=lang)
        st.dataframe(analytics_df, use_container_width=True)

    # ── Correlation heatmap ───────────────────────────────────────────────
    st.subheader(t("fi_corr", lang))
    all_cols = [c for c in bcp_available + bcu_available if c in yield_df.columns]
    if len(all_cols) >= 2:
        corr_yld = yield_df[all_cols].dropna().corr()
        nice_labs = [f"{c[:3]} {c[3:]}Y" for c in all_cols]
        fig_hm = go.Figure(go.Heatmap(
            z=corr_yld.values,
            x=nice_labs, y=nice_labs,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr_yld.values, 3),
            texttemplate="%{text}",
            showscale=True,
        ))
        fig_hm.update_layout(
            height=380, template="plotly_white",
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_hm, use_container_width=True)

    # ── Bond pricing calculator ────────────────────────────────────────────
    st.subheader(t("fi_calc", lang))
    st.markdown(t("fi_calc_sub", lang))

    freq_labels = {
        1: t("fi_annual",    lang),
        2: t("fi_semi",      lang),
        4: t("fi_quarterly", lang),
    }

    with st.form("bond_calc"):
        c1, c2, c3, c4 = st.columns(4)
        face_val   = c1.number_input(t("fi_face",   lang), value=1_000_000, step=100_000)
        coupon_ann = c2.number_input(t("fi_coupon", lang), value=6.00, step=0.25) / 100
        mat_years  = c3.number_input(t("fi_mat",    lang), value=10, min_value=1, max_value=30)
        ytm_inp    = c4.number_input(t("fi_ytm",    lang), value=6.50, step=0.05) / 100
        freq_sel   = st.radio(
            t("fi_freq", lang), [1, 2, 4], index=1,
            format_func=lambda f: freq_labels[f],
            horizontal=True,
        )
        submitted = st.form_submit_button(t("fi_calc_btn", lang))

    if submitted:
        p   = bond_price(face_val, coupon_ann, mat_years, ytm_inp, freq_sel)
        mac = macaulay_duration(face_val, coupon_ann, mat_years, ytm_inp, freq_sel)
        mod = modified_duration(mac, ytm_inp, freq_sel)
        cx  = bond_convexity(face_val, coupon_ann, mat_years, ytm_inp, freq_sel)
        dv  = dv01(face_val, coupon_ann, mat_years, ytm_inp, freq_sel)

        clean_price = p

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(t("fi_clean", lang), f"${clean_price:,.0f}")
        m2.metric(t("fi_mac",   lang), f"{mac:.4f} yrs")
        m3.metric(t("fi_mod",   lang), f"{mod:.4f} yrs")
        m4.metric(t("fi_conv",  lang), f"{cx:.5f}")
        m5.metric(t("fi_dv01",  lang), f"${dv:,.2f}")

        prem = ((clean_price / face_val) - 1) * 100
        direction = t("fi_premium", lang) if prem > 0 else t("fi_discount", lang)
        st.markdown(
            f"Bond trades at **{direction}** "
            f"({prem:+.3f}% of par). "
            f"A 1 bp increase in YTM → price change ≈ **${dv:,.2f}** "
            f"(−{mod * 100 * 0.0001 * 100:.4f}%)"
        )

        ytm_range    = np.linspace(max(0.001, ytm_inp - 0.04), ytm_inp + 0.04, 200)
        prices_range = [bond_price(face_val, coupon_ann, mat_years, y, freq_sel) for y in ytm_range]
        fig_py = go.Figure()
        fig_py.add_trace(go.Scatter(
            x=ytm_range * 100, y=prices_range,
            mode="lines", name="Price",
            line=dict(color=BCP_COLOR, width=2),
        ))
        fig_py.add_trace(go.Scatter(
            x=[ytm_inp * 100], y=[p],
            mode="markers", name="Current",
            marker=dict(color="red", size=12, symbol="cross"),
        ))
        fig_py.update_layout(
            xaxis_title=t("fi_ytm",        lang),
            yaxis_title="Clean Price (CLP)",
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig_py, use_container_width=True)
