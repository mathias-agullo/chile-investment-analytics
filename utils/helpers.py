"""
Financial calculation utilities shared across all modules.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple


# ---------------------------------------------------------------------------
# Return / risk metrics
# ---------------------------------------------------------------------------

def annualize_returns(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return float(daily_returns.mean() * trading_days)


def annualize_volatility(daily_returns: pd.Series, trading_days: int = 252) -> float:
    return float(daily_returns.std() * np.sqrt(trading_days))


def sharpe_ratio(ann_return: float, ann_vol: float, risk_free: float) -> float:
    if ann_vol == 0:
        return 0.0
    return (ann_return - risk_free) / ann_vol


def historical_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR (returned as positive loss)."""
    return float(-np.percentile(returns.dropna(), (1 - confidence) * 100))


def parametric_var(returns: pd.Series, confidence: float = 0.95) -> float:
    """Gaussian VaR (returned as positive loss)."""
    mu, sigma = returns.mean(), returns.std()
    return float(-(mu + stats.norm.ppf(1 - confidence) * sigma))


def max_drawdown(prices: pd.Series) -> float:
    rolling_max = prices.cummax()
    dd = (prices - rolling_max) / rolling_max
    return float(dd.min())


# ---------------------------------------------------------------------------
# Portfolio metrics
# ---------------------------------------------------------------------------

def portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free: float = 0.055,
    trading_days: int = 252,
) -> Tuple[float, float, float]:
    """Return (annual_return, annual_volatility, sharpe_ratio)."""
    ann_ret = float(np.dot(weights, mean_returns) * trading_days)
    ann_vol = float(np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(trading_days))
    sr = sharpe_ratio(ann_ret, ann_vol, risk_free)
    return ann_ret, ann_vol, sr


# ---------------------------------------------------------------------------
# Fixed income
# ---------------------------------------------------------------------------

def nelson_siegel(tau: np.ndarray, beta0: float, beta1: float, beta2: float, lam: float) -> np.ndarray:
    """Nelson-Siegel yield curve model (1987).

    y(τ) = β₀ + β₁·f(τ,λ) + β₂·[f(τ,λ) − e^(−τ/λ)]
    where f(τ,λ) = (1 − e^(−τ/λ)) / (τ/λ)
    """
    tau = np.asarray(tau, dtype=float)
    lam = max(lam, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = tau / lam
        factor = np.where(x < 1e-8, 1.0, (1.0 - np.exp(-x)) / x)
    return beta0 + beta1 * factor + beta2 * (factor - np.exp(-tau / lam))


def bond_price(face: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """Clean bond price via discounted cash flows.

    Args:
        face: Face (par) value
        coupon_rate: Annual coupon rate (decimal)
        years: Years to maturity
        ytm: Annual yield to maturity (decimal)
        freq: Coupon payments per year (2 = semi-annual)
    """
    n = int(round(years * freq))
    c = face * coupon_rate / freq
    r = ytm / freq
    if r == 0:
        return c * n + face
    t = np.arange(1, n + 1)
    return float(np.sum(c / (1 + r) ** t) + face / (1 + r) ** n)


def macaulay_duration(face: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """Macaulay duration in years."""
    n = int(round(years * freq))
    c = face * coupon_rate / freq
    r = ytm / freq
    t = np.arange(1, n + 1)
    cf = np.full(n, c)
    cf[-1] += face
    pv = cf / (1 + r) ** t
    price = pv.sum()
    if price == 0:
        return 0.0
    return float(np.dot(t / freq, pv) / price)


def modified_duration(mac_dur: float, ytm: float, freq: int = 2) -> float:
    return mac_dur / (1 + ytm / freq)


def bond_convexity(face: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """Full convexity (years²)."""
    n = int(round(years * freq))
    c = face * coupon_rate / freq
    r = ytm / freq
    t = np.arange(1, n + 1)
    cf = np.full(n, c, dtype=float)
    cf[-1] += face
    price = bond_price(face, coupon_rate, years, ytm, freq)
    if price == 0:
        return 0.0
    conv = np.sum(t * (t + 1) * cf / (1 + r) ** (t + 2))
    return float(conv / (price * freq**2))


def dv01(face: float, coupon_rate: float, years: float, ytm: float, freq: int = 2) -> float:
    """Dollar value of one basis point."""
    p_up = bond_price(face, coupon_rate, years, ytm + 0.0001, freq)
    p_dn = bond_price(face, coupon_rate, years, ytm - 0.0001, freq)
    return float((p_dn - p_up) / 2)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt_pct(v: float, decimals: int = 2) -> str:
    return f"{v * 100:.{decimals}f}%"


def fmt_num(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"
