"""
Chile Investment Analytics — Professional Streamlit Dashboard.

Three-module quantitative finance platform oriented toward the Chilean market.

Modules:
  1. Portfolio Optimizer   — Markowitz mean–variance optimisation (IPSA stocks)
  2. Macro SVAR Model      — Structural VAR for Chilean macro variables
  3. Fixed Income Analyzer — Nelson-Siegel yield curve + bond analytics

Run:
    streamlit run app.py
"""

import streamlit as st

# Must be the first Streamlit call
st.set_page_config(
    page_title="Chile Investment Analytics",
    page_icon="🇨🇱",
    layout="wide",
    initial_sidebar_state="expanded",
)

from utils.i18n import t  # noqa: E402 — must come after set_page_config

# ── Language selector (sidebar, very first widget) ─────────────────────────
if "lang" not in st.session_state:
    st.session_state["lang"] = "en"

with st.sidebar:
    st.markdown("## 🇨🇱 Chile Investment Analytics")
    lang = st.selectbox(
        t("lang_label", st.session_state["lang"]),
        options=["en", "es"],
        index=0 if st.session_state["lang"] == "en" else 1,
        format_func=lambda x: "🇺🇸 English" if x == "en" else "🇨🇱 Español",
        key="_lang_sel",
    )
    st.session_state["lang"] = lang

LANG = st.session_state["lang"]

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Header gradient */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: #ffffff; margin: 0; font-size: 2rem; }
    .main-header p  { color: #a0aec0; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.5rem 1.2rem;
        border-radius: 8px 8px 0 0;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 0.8rem;
        border-radius: 8px;
    }

    /* Sidebar */
    .css-1d391kg { background-color: #f0f4f8; }

    /* ── Custom spinner ─────────────────────────────────────────────────── */
    /* Replace Streamlit's default thin running-bar with a branded overlay  */
    [data-testid="stStatusWidget"] { display: none !important; }

    /* Keyframes */
    @keyframes cia-spin   { to { transform: rotate(360deg); } }
    @keyframes cia-pulse  { 0%,100%{opacity:1} 50%{opacity:.4} }
    @keyframes cia-fadein { from{opacity:0;transform:translateY(4px)} to{opacity:1;transform:translateY(0)} }

    /* Overlay shown while Streamlit is running */
    .cia-loader-overlay {
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, #0f3460, #636EFA, #EF553B, #0f3460);
        background-size: 200% 100%;
        animation: cia-pulse 1.4s ease-in-out infinite;
        z-index: 9999;
        pointer-events: none;
    }

    /* Circular spinner floating bottom-right */
    .cia-spinner-wrap {
        position: fixed;
        bottom: 1.4rem;
        right: 1.4rem;
        z-index: 9998;
        display: flex;
        align-items: center;
        gap: 0.55rem;
        background: rgba(15,52,96,0.92);
        padding: 0.5rem 0.85rem 0.5rem 0.6rem;
        border-radius: 2rem;
        box-shadow: 0 4px 18px rgba(0,0,0,0.3);
        animation: cia-fadein 0.25s ease both;
        pointer-events: none;
    }
    .cia-circle {
        width: 22px; height: 22px;
        border: 3px solid rgba(255,255,255,0.2);
        border-top-color: #636EFA;
        border-right-color: #EF553B;
        border-radius: 50%;
        animation: cia-spin 0.75s linear infinite;
        flex-shrink: 0;
    }
    .cia-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #e2e8f0;
        letter-spacing: 0.03em;
        white-space: nowrap;
    }
    </style>

    <!-- Running-state overlay injected by JS when Streamlit marks itself busy -->
    <div id="cia-loader-bar"    class="cia-loader-overlay"  style="display:none"></div>
    <div id="cia-spinner-badge" class="cia-spinner-wrap"    style="display:none">
        <div class="cia-circle"></div>
        <span class="cia-label" id="cia-label-text">Loading…</span>
    </div>

    <script>
    (function() {
        const bar   = document.getElementById('cia-loader-bar');
        const badge = document.getElementById('cia-spinner-badge');
        const lbl   = document.getElementById('cia-label-text');

        // Streamlit sets [data-testid="stStatusWidget"] text to "Running…" while busy.
        // We piggy-back on the MutationObserver trick to detect running state.
        function setVisible(v) {
            bar.style.display   = v ? 'block' : 'none';
            badge.style.display = v ? 'flex'  : 'none';
        }

        // Observe <body> for the class Streamlit adds when running
        const obs = new MutationObserver(function() {
            const el = document.querySelector('[data-testid="stApp"]');
            if (!el) return;
            // Streamlit adds data-loading-state="loading" while running
            const isRunning = el.getAttribute('data-loading-state') === 'loading';
            setVisible(isRunning);
        });
        obs.observe(document.body, { attributes: true, subtree: true, attributeFilter: ['data-loading-state'] });
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <div class="main-header">
        <h1>🇨🇱 Chile Investment Analytics</h1>
        <p>{t("subtitle", LANG)}</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar (rest) ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.markdown(
        f"""
        {t("sidebar_data", LANG)}
        - {t("sidebar_eq", LANG)}
        - {t("sidebar_macro", LANG)}

        {t("sidebar_cred", LANG)}
        {t("sidebar_cred_body", LANG)}

        {t("sidebar_models", LANG)}
        - {t("sidebar_m1", LANG)}
        - {t("sidebar_m2", LANG)}
        - {t("sidebar_m3", LANG)}
        """
    )
    st.markdown("---")
    st.caption(t("sidebar_built", LANG))

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    [
        t("tab_portfolio", LANG),
        t("tab_macro", LANG),
        t("tab_fixed", LANG),
    ]
)

with tab1:
    try:
        from modules.portfolio import render_portfolio_tab
        render_portfolio_tab(lang=LANG)
    except Exception as exc:
        st.error(f"Portfolio module error: {exc}")
        st.exception(exc)

with tab2:
    try:
        from modules.macro_svar import render_macro_tab
        render_macro_tab(lang=LANG)
    except Exception as exc:
        st.error(f"Macro SVAR module error: {exc}")
        st.exception(exc)

with tab3:
    try:
        from modules.fixed_income import render_fixed_income_tab
        render_fixed_income_tab(lang=LANG)
    except Exception as exc:
        st.error(f"Fixed income module error: {exc}")
        st.exception(exc)
