"""
Data loading layer: yfinance for equities, bcchapi for macro/rates.
Falls back to realistic synthetic data when external APIs are unavailable.
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Optional imports handled gracefully
try:
    import yfinance as yf
    _YFINANCE = True
except ImportError:
    _YFINANCE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from bcchapi import Siete as _Siete
    _BCCHAPI = True
except ImportError:
    _BCCHAPI = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IPSA_TICKERS: list[str] = [
    "SQM-B.SN", "FALABELLA.SN", "COPEC.SN", "BSANTANDER.SN", "BCI.SN",
    "CHILE.SN", "ENELAM.SN", "CMPC.SN", "CCU.SN", "COLBUN.SN",
]

TICKER_LABELS: Dict[str, str] = {
    "SQM-B.SN":      "SQM-B",
    "FALABELLA.SN":  "Falabella",
    "COPEC.SN":      "Copec",
    "BSANTANDER.SN": "Santander Chile",
    "BCI.SN":        "BCI",
    "CHILE.SN":      "Banco de Chile",
    "ENELAM.SN":     "Enel Américas",
    "CMPC.SN":       "CMPC",
    "CCU.SN":        "CCU",
    "COLBUN.SN":     "Colbún",
}

# BCCh SI3 series identifiers — verified against bcchapi 1.1.2 (June 2026)
_BCCH_SERIES = {
    "USDCLP":  "F073.TCO.PRE.Z.D",               # Tipo de cambio observado (diario)
    "IPC":     "F074.IPC.VAR.Z.Z.C.M",            # IPC variación mensual (en %, ÷100 → decimal)
    "IMACEC":  "F032.IMC.IND.Z.Z.EP18.Z.Z.0.M",  # IMACEC índice, base 2018=100
    "TPM":     "F022.TPM.TIN.D001.NO.Z.D",        # Tasa de política monetaria (en %)
    # BCP — mercado secundario (solo hasta 10 años disponible)
    "BCP2":    "F022.BCLP.TIS.AN02.NO.Z.D",
    "BCP5":    "F022.BCLP.TIS.AN05.NO.Z.D",
    "BCP10":   "F022.BCLP.TIS.AN10.NO.Z.D",
    # BCU — mercado secundario (bonos en UF)
    "BCU2":    "F022.BUF.TIS.AN02.UF.Z.D",
    "BCU5":    "F022.BUF.TIS.AN05.UF.Z.D",
    "BCU10":   "F022.BUF.TIS.AN10.UF.Z.D",
    "BCU20":   "F022.BUF.TIS.AN20.UF.Z.D",
}


# ---------------------------------------------------------------------------
# BCCh client helper
# ---------------------------------------------------------------------------

def _bcch_client() -> Optional[object]:
    if not _BCCHAPI:
        return None
    user = os.getenv("BCCH_USER", "")
    pwd  = os.getenv("BCCH_PASS", "")
    if not user or not pwd:
        return None
    try:
        return _Siete(user, pwd)
    except Exception:
        return None


def _bcch_get(client, key: str, start: str, end: str) -> Optional[pd.Series]:
    # bcchapi 1.x uses cuadro(series=[...], desde=..., hasta=...) instead of get()
    try:
        df = client.cuadro([_BCCH_SERIES[key]], desde=start, hasta=end)
        if df is None or df.empty:
            return None
        s = df.iloc[:, 0].copy()
        s.index = pd.to_datetime(s.index)
        s = pd.to_numeric(s, errors="coerce").dropna()
        return s
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Equity data
# ---------------------------------------------------------------------------

def load_stock_data(period: str = "3y") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Download IPSA constituent prices from Yahoo Finance.

    Returns:
        prices   : DataFrame of adjusted closing prices (business-day index)
        returns  : DataFrame of daily log-returns
        benchmark: ^IPSA closing price series
    """
    if _YFINANCE:
        try:
            return _yf_stock_data(period)
        except Exception:
            pass
    return _synthetic_stock_data()


def _yf_stock_data(period: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    # Use default group_by="column": result["Close"] is a DataFrame with tickers as columns.
    # This is more robust than group_by="ticker" which creates a nested MultiIndex.
    raw = yf.download(IPSA_TICKERS, period=period, auto_adjust=True, progress=False)

    if raw.empty:
        return _synthetic_stock_data()

    # Extract Close prices — handle both flat and MultiIndex column layouts
    if isinstance(raw.columns, pd.MultiIndex):
        # Shape: (field, ticker) — default yfinance layout for multi-ticker downloads
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            return _synthetic_stock_data()
    else:
        # Single ticker fell through somehow
        prices = raw[["Close"]].copy()

    prices = prices.dropna(how="all").ffill().bfill()
    # Keep only tickers with <10% missing data
    prices = prices.loc[:, prices.isna().mean() < 0.10].dropna()

    if prices.shape[1] < 5:
        return _synthetic_stock_data()

    returns = np.log(prices / prices.shift(1)).dropna()

    try:
        bm_raw = yf.download("^IPSA", period=period, auto_adjust=True, progress=False)
        # ^IPSA download returns a flat DataFrame (single ticker)
        bm_close = bm_raw["Close"] if "Close" in bm_raw.columns else bm_raw.iloc[:, 0]
        benchmark = bm_close.dropna()
    except Exception:
        benchmark = prices.mean(axis=1)

    return prices, returns, benchmark


def _synthetic_stock_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Realistic synthetic IPSA data when yfinance is unavailable."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(
        start=(datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d"),
        end=datetime.today().strftime("%Y-%m-%d"),
    )
    n = len(dates)
    tickers = IPSA_TICKERS

    # Annualised parameters (approx IPSA constituents 2021-2024)
    mu_ann  = [0.14, 0.06, 0.11, 0.08, 0.09, 0.08, 0.05, 0.10, 0.07, 0.12]
    sig_ann = [0.34, 0.23, 0.26, 0.21, 0.23, 0.20, 0.25, 0.24, 0.19, 0.27]
    mu_d  = np.array(mu_ann)  / 252
    sig_d = np.array(sig_ann) / np.sqrt(252)

    # Correlation matrix (banking cluster + rest)
    corr = np.array([
        [1.00, 0.35, 0.40, 0.30, 0.32, 0.28, 0.36, 0.42, 0.33, 0.37],
        [0.35, 1.00, 0.44, 0.43, 0.47, 0.44, 0.35, 0.39, 0.50, 0.34],
        [0.40, 0.44, 1.00, 0.37, 0.39, 0.35, 0.41, 0.38, 0.36, 0.46],
        [0.30, 0.43, 0.37, 1.00, 0.75, 0.70, 0.31, 0.34, 0.44, 0.30],
        [0.32, 0.47, 0.39, 0.75, 1.00, 0.72, 0.32, 0.36, 0.46, 0.32],
        [0.28, 0.44, 0.35, 0.70, 0.72, 1.00, 0.29, 0.33, 0.43, 0.28],
        [0.36, 0.35, 0.41, 0.31, 0.32, 0.29, 1.00, 0.37, 0.35, 0.40],
        [0.42, 0.39, 0.38, 0.34, 0.36, 0.33, 0.37, 1.00, 0.34, 0.43],
        [0.33, 0.50, 0.36, 0.44, 0.46, 0.43, 0.35, 0.34, 1.00, 0.32],
        [0.37, 0.34, 0.46, 0.30, 0.32, 0.28, 0.40, 0.43, 0.32, 1.00],
    ])
    cov = np.diag(sig_d) @ corr @ np.diag(sig_d)
    L = np.linalg.cholesky(cov + np.eye(len(tickers)) * 1e-10)
    z = rng.standard_normal((n, len(tickers)))
    daily_ret = mu_d + (L @ z.T).T

    init = [5200, 920, 7600, 36, 28000, 82, 62, 1850, 9100, 145]
    prices = {}
    for i, t in enumerate(tickers):
        prices[t] = init[i] * np.cumprod(1 + daily_ret[:, i])

    prices_df = pd.DataFrame(prices, index=dates)
    returns_df = pd.DataFrame(daily_ret, index=dates, columns=tickers)
    benchmark  = pd.Series(25000 * np.cumprod(1 + daily_ret.mean(axis=1)), index=dates, name="^IPSA")
    return prices_df, returns_df, benchmark


# ---------------------------------------------------------------------------
# Macro data
# ---------------------------------------------------------------------------

def load_macro_data() -> Tuple[pd.DataFrame, str]:
    """Load monthly macro variables (TPM, IPC, IMACEC, USD/CLP, Copper).

    Returns:
        df    : DataFrame with columns [TPM, IPC, IMACEC, USDCLP, Copper]
        source: Description of data origin
    """
    client = _bcch_client()
    if client is not None:
        try:
            return _bcch_macro_data(client)
        except Exception:
            pass
    return _synthetic_macro_data(), "Synthetic data (BCCh API unavailable — add BCCH_USER/BCCH_PASS to .env)"


def _bcch_macro_data(client) -> Tuple[pd.DataFrame, str]:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 7)).strftime("%Y-%m-%d")

    data: Dict[str, pd.Series] = {}

    # USD/CLP: daily → monthly last
    usdclp = _bcch_get(client, "USDCLP", start, end)
    if usdclp is not None:
        data["USDCLP"] = usdclp.resample("MS").last()

    # IPC: monthly variation already in %; BCCh returns e.g. 0.7 meaning 0.7% → ÷100
    ipc = _bcch_get(client, "IPC", start, end)
    if ipc is not None:
        data["IPC"] = ipc / 100

    # IMACEC: monthly index (base 2018=100)
    imacec = _bcch_get(client, "IMACEC", start, end)
    if imacec is not None:
        data["IMACEC"] = imacec

    # TPM: daily in %; → monthly last → decimal
    tpm = _bcch_get(client, "TPM", start, end)
    if tpm is not None:
        data["TPM"] = (tpm / 100).resample("MS").last()

    # Copper (USD/lb): BCCh doesn't publish this — use yfinance HG=F
    if _YFINANCE:
        try:
            cu_raw = yf.download("HG=F", start=start, end=end, auto_adjust=True, progress=False)
            if not cu_raw.empty:
                cu_close = cu_raw["Close"] if "Close" in cu_raw.columns else cu_raw.iloc[:, 0]
                data["Copper"] = cu_close.squeeze().resample("MS").last().dropna()
        except Exception:
            pass

    if len(data) < 2:
        raise ValueError("Insufficient data from BCCh API")

    df = pd.DataFrame(data).dropna(how="all").sort_index()
    return df, "BCCh API (TPM/IPC/IMACEC/USD·CLP) + Yahoo Finance (Cobre)"


def _synthetic_macro_data() -> pd.DataFrame:
    """Realistic synthetic Chilean macro data (2018-present)."""
    rng = np.random.default_rng(123)
    dates = pd.date_range("2018-01-01", datetime.today().strftime("%Y-%m-%d"), freq="MS")
    n = len(dates)

    # TPM (decimal): mirrors actual BCCh trajectory
    tpm = np.zeros(n)
    for i, d in enumerate(dates):
        y, m = d.year, d.month
        if y <= 2019:
            tpm[i] = 0.025 + rng.normal(0, 0.002)
        elif y == 2020 and m < 4:
            tpm[i] = 0.015
        elif y == 2020:
            tpm[i] = 0.005
        elif y == 2021:
            tpm[i] = 0.005 + (m / 12) * 0.020
        elif y == 2022:
            tpm[i] = 0.025 + ((m - 1) / 11) * 0.0875
        elif y == 2023:
            tpm[i] = 0.1125 - ((m - 1) / 11) * 0.040
        else:
            tpm[i] = 0.055 + rng.normal(0, 0.003)
    tpm = np.clip(tpm, 0.001, 0.13)

    # IPC monthly variation (decimal)
    ipc_base = {2018: 0.003, 2019: 0.003, 2020: 0.003, 2021: 0.005, 2022: 0.009, 2023: 0.005, 2024: 0.003, 2025: 0.003}
    ipc = np.array([
        rng.normal(ipc_base.get(d.year, 0.003), 0.0015)
        for d in dates
    ])
    ipc = np.clip(ipc, -0.005, 0.02)

    # IMACEC index (monthly, base 100 = 2013)
    imacec_g = np.zeros(n)
    for i, d in enumerate(dates):
        y, m = d.year, d.month
        if y == 2020 and 4 <= m <= 9:
            imacec_g[i] = rng.normal(-0.006, 0.01)
        elif y == 2021:
            imacec_g[i] = rng.normal(0.012, 0.005)
        else:
            imacec_g[i] = rng.normal(0.002, 0.004)
    imacec = 115.0 * np.cumprod(1 + imacec_g)

    # USD/CLP (level, random walk with regime)
    usdclp = np.zeros(n)
    usdclp[0] = 665.0
    for i in range(1, n):
        trend = 1.5 if dates[i].year == 2022 else 0
        usdclp[i] = usdclp[i - 1] + rng.normal(trend, 9.0)
    usdclp = np.clip(usdclp, 580, 1080)

    # Copper price (USD/lb)
    copper = np.zeros(n)
    copper[0] = 2.75
    for i in range(1, n):
        copper[i] = copper[i - 1] + rng.normal(0.010, 0.055)
    copper = np.clip(copper, 1.9, 5.2)

    return pd.DataFrame(
        {"TPM": tpm, "IPC": ipc, "IMACEC": imacec, "USDCLP": usdclp, "Copper": copper},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Yield curve data
# ---------------------------------------------------------------------------

def load_yield_curve_data() -> Tuple[pd.DataFrame, str]:
    """Load BCP and BCU yield curve rates from BCCh or synthetic fallback.

    Returns:
        df    : Weekly DataFrame with columns BCP2/5/10/20, BCU2/5/10/20 (decimal)
        source: Description of data origin
    """
    client = _bcch_client()
    if client is not None:
        try:
            return _bcch_yield_data(client)
        except Exception:
            pass
    return _synthetic_yield_data(), "Synthetic data (BCCh API unavailable)"


def _bcch_yield_data(client) -> Tuple[pd.DataFrame, str]:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 6)).strftime("%Y-%m-%d")

    # BCP20 is not published in BCCh secondary market — use BCP 2/5/10 + BCU 2/5/10/20
    keys = ["BCP2", "BCP5", "BCP10", "BCU2", "BCU5", "BCU10", "BCU20"]
    data: Dict[str, pd.Series] = {}
    for k in keys:
        s = _bcch_get(client, k, start, end)
        if s is not None:
            data[k] = s / 100  # % → decimal

    if len(data) < 4:
        raise ValueError("Insufficient yield data from BCCh API")

    df = pd.DataFrame(data).dropna(how="all").sort_index()
    df = df.resample("W-FRI").last().dropna(how="all")
    return df, "BCCh API — Mercado secundario BCP/BCU"


def _synthetic_yield_data() -> pd.DataFrame:
    """Realistic synthetic Chilean yield curve data."""
    rng = np.random.default_rng(456)
    dates = pd.date_range("2019-01-04", datetime.today().strftime("%Y-%m-%d"), freq="W-FRI")
    n = len(dates)

    # Rate cycle correlated with TPM trajectory
    cycle = np.zeros(n)
    for i, d in enumerate(dates):
        if d.year == 2022:
            cycle[i] = 0.025
        elif d.year == 2023:
            cycle[i] = 0.015
        elif d.year == 2024:
            cycle[i] = 0.005

    def rw_series(base: float, vol: float, sensitivity: float) -> np.ndarray:
        s = np.zeros(n)
        s[0] = base
        for i in range(1, n):
            s[i] = s[i - 1] + rng.normal(0, vol) + 0.05 * (base - s[i - 1])
        return np.clip(s + cycle * sensitivity, 0.005, 0.18)

    bcp2  = rw_series(0.055, 0.006, 1.00)
    bcp5  = rw_series(0.060, 0.005, 0.90)
    bcp10 = rw_series(0.062, 0.004, 0.80)
    bcp20 = rw_series(0.063, 0.004, 0.70)
    bcu2  = rw_series(0.015, 0.004, 0.30)
    bcu5  = rw_series(0.020, 0.003, 0.35)
    bcu10 = rw_series(0.025, 0.003, 0.40)
    bcu20 = rw_series(0.027, 0.003, 0.38)

    return pd.DataFrame(
        {"BCP2": bcp2, "BCP5": bcp5, "BCP10": bcp10, "BCP20": bcp20,
         "BCU2": bcu2, "BCU5": bcu5, "BCU10": bcu10, "BCU20": bcu20},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Risk-free rate
# ---------------------------------------------------------------------------

def get_risk_free_rate() -> float:
    """Return current Chilean risk-free rate (TPM) as a decimal."""
    client = _bcch_client()
    if client is not None:
        end   = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=45)).strftime("%Y-%m-%d")
        s = _bcch_get(client, "TPM", start, end)
        if s is not None and not s.empty:
            return float(s.dropna().iloc[-1] / 100)
    return 0.050  # BCCh TPM as of mid-2026
