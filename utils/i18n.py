"""Internationalisation — English / Spanish translations."""

TEXTS: dict = {
    "en": {
        # ── app.py ──────────────────────────────────────────────────────────
        "subtitle": (
            "Quantitative Finance Dashboard · IPSA Portfolio Optimization · "
            "Macro SVAR · Fixed Income (BCP/BCU)"
        ),
        "tab_portfolio": "📈  Portfolio Optimizer",
        "tab_macro":     "📊  Macro SVAR",
        "tab_fixed":     "💰  Fixed Income",
        "sidebar_data":  "**Data sources**",
        "sidebar_eq":    "Equities: Yahoo Finance (`yfinance`)",
        "sidebar_macro": "Macro / Rates: BCCh API (`bcchapi`)",
        "sidebar_cred":  "**Credentials**",
        "sidebar_cred_body": (
            "Add the following to a `.env` file:\n\n"
            "`BCCH_USER` / `BCCH_PASS` — live Banco Central de Chile data "
            "(without them the app uses synthetic series).\n\n"
            "`GROQ_API_KEY` — enables the 🤖 AI analyst commentary in each tab "
            "(free at console.groq.com)."
        ),
        "sidebar_models": "**Models**",
        "sidebar_m1":    "Portfolio: Markowitz (1952)",
        "sidebar_m2":    "Macro: Sims (1980) SVAR",
        "sidebar_m3":    "Fixed income: Nelson-Siegel (1987)",
        "sidebar_built": "Built with Streamlit · Plotly · statsmodels · scipy",
        "lang_label":    "Language / Idioma",

        # ── portfolio.py ─────────────────────────────────────────────────────
        "port_header":   "Portfolio Optimizer — IPSA Constituents",
        "port_method": (
            "**Methodology:** Markowitz (1952) mean–variance optimisation. "
            "Efficient frontier estimated via Monte Carlo simulation (5 000 portfolios); "
            "optimal portfolios solved with `scipy.optimize.minimize` (SLSQP). "
            "Risk-free rate: BCCh Tasa de Política Monetaria (TPM)."
        ),
        "port_loading":  "Loading market data from Yahoo Finance…",
        "port_loaded":   "Loaded {n} stocks · {days} trading days · Risk-free rate (TPM): {rf:.2f}%",
        "port_settings": "Portfolio Settings",
        "port_assets":   "Select assets",
        "port_backtest_slider": "Backtest window (months)",
        "port_warn":     "Select at least 3 assets.",
        "port_running":  "Running optimisation…",
        "port_metrics":  "Optimal Portfolios — Key Metrics",
        "port_col_port": "Portfolio",
        "port_col_ret":  "Ann. Return",
        "port_col_vol":  "Ann. Volatility",
        "port_col_sr":   "Sharpe Ratio",
        "port_col_var":  "VaR 95% (daily)",
        "port_col_mdd":  "Max Drawdown",
        "port_frontier": "Efficient Frontier",
        "port_sim":      "Simulated portfolios",
        "port_weights":  "Portfolio Weights",
        "port_weights_full": "Full weights table",
        "port_backtest": "Backtesting — last {m} months vs. IPSA",
        "port_bench":    "IPSA benchmark",
        "port_corr":     "Return Correlation Matrix",
        "port_xvol":     "Annual Volatility (%)",
        "port_yret":     "Annual Return (%)",
        "port_cumret":   "Cumulative Return (%)",

        # ── macro_svar.py ────────────────────────────────────────────────────
        "macro_header":  "Macro SVAR Model — Chile",
        "macro_method": (
            "**Methodology:** Structural VAR (Sims 1980) with Cholesky identification. "
            "Lag length selected by HQIC with cap T/(5k) to avoid overfitting. "
            "IRF confidence bands via parametric bootstrap "
            "(300 replications, 95% CI). Cholesky ordering: "
            "*Copper → USD/CLP → IMACEC → IPC → TPM*."
        ),
        "macro_loading": "Loading macro data…",
        "macro_raw":     "Raw Macro Series",
        "macro_adf":     "Stationarity Tests (Augmented Dickey-Fuller)",
        "macro_adf_stat":"ADF Statistic",
        "macro_adf_p":   "p-value",
        "macro_adf_lags":"Lags",
        "macro_adf_ok":  "Stationary (5%)",
        "macro_transforms": "Applied transformations",
        "macro_transf_spinner": "Transforming series to stationarity…",
        "macro_var":     "VAR Model",
        "macro_var_est": "Estimating VAR…",
        "macro_lag_aic": "Optimal Lags (AIC)",
        "macro_lag_bic":  "Optimal Lags (BIC)",
        "macro_lag_hqic": "Optimal Lags (HQIC)",
        "macro_obs":      "Observations",
        "macro_vars":    "Variables",
        "macro_summary": "VAR Model Summary",
        "macro_lag_tbl": "Lag order selection criteria",
        "macro_irf":     "Orthogonalised Impulse Response Functions (95% CI)",
        "macro_irf_spin":"Computing IRF bootstrap ({n} replications)…",
        "macro_shock":   "Shock variable (column = response variable)",
        "macro_fevd":    "Forecast Error Variance Decomposition (FEVD) — 12 months",
        "macro_resp":    "Response variable",
        "macro_fhor":    "Forecast Horizon (months)",
        "macro_fvar":    "Variance Explained (%)",
        "macro_fc":      "VAR Forecast — 12 months ahead",
        "macro_fc_var":  "Forecast variable",
        "macro_hist":    "Historical",
        "macro_forecast":"Forecast",
        "macro_fc_tbl":  "Forecast table (all variables)",
        "macro_months":  "Months",
        "macro_err_few": "Too few observations for VAR estimation after differencing.",
        "macro_err_data":"Not enough macro variables to estimate VAR. Check data source.",

        # ── fixed_income.py ──────────────────────────────────────────────────
        "fi_header":     "Fixed Income Analyzer — Chilean Sovereign Bonds",
        "fi_method": (
            "**Methodology:** Yield curve fitted with the Nelson-Siegel (1987) model. "
            "Duration and convexity computed using discounted cash-flow formulae (Fabozzi 2012). "
            "BCP = nominal peso bonds; BCU = UF-indexed bonds. "
            "BCP − BCU spread ≈ breakeven inflation."
        ),
        "fi_loading":    "Loading yield curve data…",
        "fi_ns":         "Yield Curve — Nelson-Siegel Fit",
        "fi_cur":        "Current Curve",
        "fi_hist":       "Historical Evolution",
        "fi_bcp_params": "BCP Nelson-Siegel parameters",
        "fi_spread":     "Breakeven Inflation Spread (BCP − BCU)",
        "fi_target":     "BCCh 3% inflation target",
        "fi_dur":        "Duration, Convexity & DV01 — Current Curve",
        "fi_corr":       "Yield Correlation Heatmap",
        "fi_calc":       "Bond Pricing Calculator",
        "fi_calc_sub":   "Price a bond given its cash-flow characteristics and yield to maturity.",
        "fi_face":       "Face Value (CLP)",
        "fi_coupon":     "Annual Coupon Rate (%)",
        "fi_mat":        "Years to Maturity",
        "fi_ytm":        "YTM (%)",
        "fi_freq":       "Coupon frequency",
        "fi_annual":     "Annual",
        "fi_semi":       "Semi-annual",
        "fi_quarterly":  "Quarterly",
        "fi_calc_btn":   "Calculate",
        "fi_clean":      "Clean Price",
        "fi_mac":        "Macaulay Duration",
        "fi_mod":        "Modified Duration",
        "fi_conv":       "Convexity",
        "fi_dv01":       "DV01",
        "fi_premium":    "premium",
        "fi_discount":   "discount",
        "fi_price_yield":"Price / YTM Sensitivity",
        "fi_hist_sel":   "Instruments",
        "fi_mat_yrs":    "Maturity (years)",
        "fi_yield_sp":   "Yield / Spread (%)",
        "fi_bei":        "Breakeven Inflation (%)",
        "fi_yld_pct":    "Yield (%)",
        "fi_date":       "Date",
    },

    "es": {
        # ── app.py ──────────────────────────────────────────────────────────
        "subtitle": (
            "Dashboard de Finanzas Cuantitativas · Optimización IPSA · "
            "Macro SVAR · Renta Fija (BCP/BCU)"
        ),
        "tab_portfolio": "📈  Optimizador de Portafolios",
        "tab_macro":     "📊  Macro SVAR",
        "tab_fixed":     "💰  Renta Fija",
        "sidebar_data":  "**Fuentes de datos**",
        "sidebar_eq":    "Acciones: Yahoo Finance (`yfinance`)",
        "sidebar_macro": "Macro / Tasas: API BCCh (`bcchapi`)",
        "sidebar_cred":  "**Credenciales**",
        "sidebar_cred_body": (
            "Agrega lo siguiente a un archivo `.env`:\n\n"
            "`BCCH_USER` / `BCCH_PASS` — datos en vivo del Banco Central de Chile "
            "(sin ellos la app usa series sintéticas).\n\n"
            "`GROQ_API_KEY` — habilita el comentario 🤖 analista IA en cada tab "
            "(gratis en console.groq.com)."
        ),
        "sidebar_models": "**Modelos**",
        "sidebar_m1":    "Portafolio: Markowitz (1952)",
        "sidebar_m2":    "Macro: Sims (1980) SVAR",
        "sidebar_m3":    "Renta fija: Nelson-Siegel (1987)",
        "sidebar_built": "Construido con Streamlit · Plotly · statsmodels · scipy",
        "lang_label":    "Language / Idioma",

        # ── portfolio.py ─────────────────────────────────────────────────────
        "port_header":   "Optimizador de Portafolios — Constituyentes del IPSA",
        "port_method": (
            "**Metodología:** Optimización media-varianza de Markowitz (1952). "
            "Frontera eficiente estimada vía simulación Monte Carlo (5 000 portafolios); "
            "portafolios óptimos resueltos con `scipy.optimize.minimize` (SLSQP). "
            "Tasa libre de riesgo: Tasa de Política Monetaria del BCCh (TPM)."
        ),
        "port_loading":  "Cargando datos de mercado desde Yahoo Finance…",
        "port_loaded":   "Cargadas {n} acciones · {days} días de trading · Tasa libre de riesgo (TPM): {rf:.2f}%",
        "port_settings": "Configuración del Portafolio",
        "port_assets":   "Seleccionar activos",
        "port_backtest_slider": "Ventana de backtesting (meses)",
        "port_warn":     "Selecciona al menos 3 activos.",
        "port_running":  "Ejecutando optimización…",
        "port_metrics":  "Portafolios Óptimos — Métricas Clave",
        "port_col_port": "Portafolio",
        "port_col_ret":  "Retorno Anual",
        "port_col_vol":  "Volatilidad Anual",
        "port_col_sr":   "Ratio de Sharpe",
        "port_col_var":  "VaR 95% (diario)",
        "port_col_mdd":  "Máx. Caída",
        "port_frontier": "Frontera Eficiente",
        "port_sim":      "Portafolios simulados",
        "port_weights":  "Pesos del Portafolio",
        "port_weights_full": "Tabla completa de pesos",
        "port_backtest": "Backtesting — últimos {m} meses vs. IPSA",
        "port_bench":    "Benchmark IPSA",
        "port_corr":     "Matriz de Correlación de Retornos",
        "port_xvol":     "Volatilidad Anual (%)",
        "port_yret":     "Retorno Anual (%)",
        "port_cumret":   "Retorno Acumulado (%)",

        # ── macro_svar.py ────────────────────────────────────────────────────
        "macro_header":  "Modelo Macro SVAR — Chile",
        "macro_method": (
            "**Metodología:** VAR Estructural (Sims 1980) con identificación de Cholesky. "
            "Orden de rezagos seleccionado por HQIC con techo T/(5k) para evitar sobreparametrización. "
            "Bandas de confianza de la IRF vía bootstrap paramétrico (300 réplicas, IC 95%). "
            "Orden de Cholesky: *Cobre → USD/CLP → IMACEC → IPC → TPM*."
        ),
        "macro_loading": "Cargando datos macroeconómicos…",
        "macro_raw":     "Series Macroeconómicas",
        "macro_adf":     "Tests de Estacionariedad (Dickey-Fuller Aumentado)",
        "macro_adf_stat":"Estadístico ADF",
        "macro_adf_p":   "p-valor",
        "macro_adf_lags":"Rezagos",
        "macro_adf_ok":  "Estacionaria (5%)",
        "macro_transforms": "Transformaciones aplicadas",
        "macro_transf_spinner": "Transformando series a estacionariedad…",
        "macro_var":     "Modelo VAR",
        "macro_var_est": "Estimando VAR…",
        "macro_lag_aic": "Rezagos Óptimos (AIC)",
        "macro_lag_bic":  "Rezagos Óptimos (BIC)",
        "macro_lag_hqic": "Rezagos Óptimos (HQIC)",
        "macro_obs":      "Observaciones",
        "macro_vars":    "Variables",
        "macro_summary": "Resumen del Modelo VAR",
        "macro_lag_tbl": "Criterios de selección de orden de rezagos",
        "macro_irf":     "Funciones Impulso-Respuesta Ortogonalizadas (IC 95%)",
        "macro_irf_spin":"Calculando bootstrap IRF ({n} réplicas)…",
        "macro_shock":   "Variable de choque (columna = variable de respuesta)",
        "macro_fevd":    "Descomposición del Error de Pronóstico (FEVD) — 12 meses",
        "macro_resp":    "Variable de respuesta",
        "macro_fhor":    "Horizonte de Pronóstico (meses)",
        "macro_fvar":    "Varianza Explicada (%)",
        "macro_fc":      "Pronóstico VAR — 12 meses adelante",
        "macro_fc_var":  "Variable a pronosticar",
        "macro_hist":    "Histórico",
        "macro_forecast":"Pronóstico",
        "macro_fc_tbl":  "Tabla de pronósticos (todas las variables)",
        "macro_months":  "Meses",
        "macro_err_few": "Muy pocas observaciones para estimar VAR tras diferenciar.",
        "macro_err_data":"Variables macro insuficientes para estimar VAR. Revisa la fuente de datos.",

        # ── fixed_income.py ──────────────────────────────────────────────────
        "fi_header":     "Analizador de Renta Fija — Bonos Soberanos de Chile",
        "fi_method": (
            "**Metodología:** Curva de rendimiento ajustada con el modelo Nelson-Siegel (1987). "
            "Duración y convexidad calculadas con flujos de caja descontados (Fabozzi 2012). "
            "BCP = bonos en pesos nominales; BCU = bonos en UF (indexados). "
            "Spread BCP − BCU ≈ inflación de equilibrio (breakeven)."
        ),
        "fi_loading":    "Cargando datos de curva de rendimiento…",
        "fi_ns":         "Curva de Rendimiento — Ajuste Nelson-Siegel",
        "fi_cur":        "Curva Actual",
        "fi_hist":       "Evolución Histórica",
        "fi_bcp_params": "Parámetros Nelson-Siegel BCP",
        "fi_spread":     "Spread de Inflación de Equilibrio (BCP − BCU)",
        "fi_target":     "Meta de inflación BCCh 3%",
        "fi_dur":        "Duración, Convexidad y DV01 — Curva Actual",
        "fi_corr":       "Mapa de Calor de Correlación de Tasas",
        "fi_calc":       "Calculadora de Bonos",
        "fi_calc_sub":   "Valoriza un bono dado sus características de flujo y tasa de rendimiento.",
        "fi_face":       "Valor Nominal (CLP)",
        "fi_coupon":     "Tasa de Cupón Anual (%)",
        "fi_mat":        "Años al Vencimiento",
        "fi_ytm":        "TIR (%)",
        "fi_freq":       "Frecuencia de cupones",
        "fi_annual":     "Anual",
        "fi_semi":       "Semestral",
        "fi_quarterly":  "Trimestral",
        "fi_calc_btn":   "Calcular",
        "fi_clean":      "Precio Limpio",
        "fi_mac":        "Duración de Macaulay",
        "fi_mod":        "Duración Modificada",
        "fi_conv":       "Convexidad",
        "fi_dv01":       "DV01",
        "fi_premium":    "sobre la par",
        "fi_discount":   "bajo la par",
        "fi_price_yield":"Sensibilidad Precio / TIR",
        "fi_hist_sel":   "Instrumentos",
        "fi_mat_yrs":    "Madurez (años)",
        "fi_yield_sp":   "Tasa / Spread (%)",
        "fi_bei":        "Inflación de Equilibrio (%)",
        "fi_yld_pct":    "Tasa (%)",
        "fi_date":       "Fecha",
    },
}


def t(key: str, lang: str = "en") -> str:
    """Return translated string for key in given language, falling back to English."""
    return TEXTS.get(lang, TEXTS["en"]).get(key, TEXTS["en"].get(key, key))
