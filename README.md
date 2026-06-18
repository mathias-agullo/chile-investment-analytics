# Chile Investment Analytics

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Data](https://img.shields.io/badge/Data-BCCh%20%7C%20Yahoo%20Finance-orange)

A professional quantitative finance dashboard built with Python and Streamlit,
oriented toward the **Chilean capital market**. Designed to demonstrate
skills in quantitative investment analysis, financial econometrics, and
fixed income analytics — relevant for roles at institutions such as
LarrainVial Asset Management, AFP Cuprum, Banco Central de Chile, and CMF.

---

## Modules

### 1 — Portfolio Optimizer (Markowitz 1952)

Implements the classic mean–variance framework on **10 IPSA constituents**
(SQM-B, Falabella, Copec, Santander Chile, BCI, Banco de Chile,
Enel Américas, CMPC, CCU, Colbún) with three years of daily data from
Yahoo Finance.

**Key features:**
- Efficient frontier via Monte Carlo simulation (5 000 portfolios)
- Three optimal portfolios solved with `scipy.optimize` (SLSQP):
  - Maximum Sharpe Ratio
  - Minimum Variance
  - Maximum Return (diversification constraint ≤ 40% per asset)
- Risk metrics: annualised return, volatility, Sharpe ratio, historical
  VaR (95%), maximum drawdown
- Backtesting vs. IPSA benchmark (configurable 6–24 month window)
- Risk-free rate: BCCh Tasa de Política Monetaria (TPM)
- Interactive Plotly scatter, pie charts, and correlation heatmap

**Reference:** Markowitz, H. (1952). Portfolio Selection.
*Journal of Finance*, 7(1), 77–91.

---

### 2 — Macro SVAR Model (Sims 1980)

Estimates a **Structural Vector Autoregression** for five Chilean macro
variables loaded from the BCCh Statistical Database (`bcchapi`):

| Variable | Description |
|---|---|
| Copper | International copper price (USD/lb) |
| USD/CLP | Observed exchange rate |
| IMACEC | Monthly economic activity index |
| IPC | Monthly CPI variation |
| TPM | Central bank policy rate |

**Key features:**
- Augmented Dickey-Fuller stationarity tests with automatic differencing
- Lag selection via AIC / BIC (up to 12 lags)
- Orthogonalised IRF with Cholesky identification (recursive ordering:
  Copper → USD/CLP → IMACEC → IPC → TPM)
- 95% confidence bands via parametric residual bootstrap (300 replications)
- 12-month VAR forecast with confidence intervals (`forecast_interval`)
- Forecast Error Variance Decomposition (FEVD) — stacked bar chart
- Interactive variable selector for shock and response variables

**Reference:** Sims, C.A. (1980). Macroeconomics and Reality.
*Econometrica*, 48(1), 1–48.

---

### 3 — Fixed Income Analyzer

Analyses the **Chilean sovereign yield curve** (BCP and BCU bonds at
2, 5, 10, and 20-year maturities) sourced from the BCCh API.

| Instrument | Description |
|---|---|
| BCP | *Bonos del Banco Central en Pesos* (nominal) |
| BCU | *Bonos del Banco Central en UF* (real, inflation-indexed) |

**Key features:**
- Nelson-Siegel (1987) yield curve fitting via `scipy.curve_fit`
  (parameters: β₀ long-term, β₁ short-term, β₂ hump, λ decay)
- Per-node analytics: Macaulay duration, modified duration,
  convexity, DV01
- BCP − BCU breakeven inflation spread vs. BCCh 3% target
- Historical yield evolution chart (configurable instrument selection)
- Full yield correlation heatmap
- **Interactive bond pricing calculator**: given face value, coupon,
  maturity, and YTM → clean price, durations, convexity, DV01,
  and price/yield sensitivity curve

**Reference:** Nelson, C.R. & Siegel, A.F. (1987). Parsimonious Modeling
of Yield Curves. *Journal of Business*, 60(4), 473–489.

---

## Installation

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USER/chile-investment-analytics.git
cd chile-investment-analytics

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure API credentials
cp .env.example .env
# Edit .env with your credentials:
#
#   BCCh SI3 — live macro/rates data (register at https://si3.bcentral.cl):
#   BCCH_USER=your_email@example.com
#   BCCH_PASS=your_password
#
#   Groq — AI analyst commentary in each tab (free at https://console.groq.com):
#   GROQ_API_KEY=your_groq_api_key

# 5. Run the dashboard
streamlit run app.py
```

> **Without BCCh credentials** the app automatically falls back to
> realistic synthetic series that mirror actual Chilean macro dynamics
> (TPM trajectory, IPC peaks, IMACEC, FX, copper). All models run
> identically on real or synthetic data.
>
> **Without a Groq API key** the 🤖 AI analyst commentary buttons are
> simply hidden — all three analytical modules remain fully functional.

---

## Project Structure

```
chile-investment-analytics/
├── app.py                  # Main Streamlit app (3 tabs)
├── modules/
│   ├── portfolio.py        # Tab 1 — Markowitz Portfolio Optimizer
│   ├── macro_svar.py       # Tab 2 — Macro SVAR Model
│   └── fixed_income.py     # Tab 3 — Fixed Income Analyzer
├── utils/
│   ├── data_loader.py      # yfinance + bcchapi data layer (with fallback)
│   └── helpers.py          # Financial calculation utilities
├── requirements.txt
├── .env.example
└── README.md
```

---

## Data Sources

| Source | Access | Variables |
|---|---|---|
| **Yahoo Finance** | Public (via `yfinance`) | IPSA stock prices, IPSA index |
| **BCCh SI3** | Free registration required | TPM, IPC, IMACEC, USD/CLP, BCP/BCU rates |

BCCh API documentation: <https://si3.bcentral.cl/estadisticas/Principal1>

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.58.0 | Dashboard framework |
| plotly | 5.22.0 | Interactive charts |
| pandas | 2.2.2 | Data manipulation |
| numpy | 2.2.6 | Numerical computing |
| scipy | 1.13.1 | Optimisation, curve fitting |
| statsmodels | 0.14.6 | VAR, ADF tests |
| yfinance | 1.4.1 | Yahoo Finance data |
| bcchapi | 1.1.2 | BCCh statistical API |
| python-dotenv | 1.0.1 | Environment variable management |
| groq | ≥0.9.0 | AI analyst commentary (Llama 3.3 70B) |

---

## License

MIT License — see [LICENSE](LICENSE).

---

*Built as a quantitative finance portfolio project. Models are for
educational and demonstration purposes only and do not constitute
investment advice.*
