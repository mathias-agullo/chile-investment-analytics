"""Groq-powered commentary for each dashboard tab."""

import os
from typing import Optional

_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                return None
            _client = Groq(api_key=api_key)
        except Exception:
            return None
    return _client


def _ask(prompt: str, lang: str = "en") -> Optional[str]:
    client = _get_client()
    if client is None:
        return None
    sys_en = (
        "You are a senior quantitative analyst specializing in Chilean financial markets "
        "(IPSA, BCCh monetary policy, BCP/BCU bonds). "
        "Reply in 2-3 concise sentences. Be direct and analytical, no filler phrases."
    )
    sys_es = (
        "Eres un analista cuantitativo senior especializado en mercados financieros chilenos "
        "(IPSA, política monetaria BCCh, bonos BCP/BCU). "
        "Responde en 2-3 oraciones concisas. Sé directo y analítico, sin frases de relleno."
    )
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=220,
            temperature=0.4,
            messages=[
                {"role": "system", "content": sys_es if lang == "es" else sys_en},
                {"role": "user",   "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


# ── Per-tab commentary helpers ──────────────────────────────────────────────

def portfolio_commentary(
    sharpe_max: float,
    ret_max: float,
    vol_max: float,
    ret_minvar: float,
    vol_minvar: float,
    rf: float,
    lang: str = "en",
) -> Optional[str]:
    if lang == "es":
        prompt = (
            f"Portafolio Max Sharpe: retorno anual {ret_max:.1f}%, volatilidad {vol_max:.1f}%, "
            f"Sharpe {sharpe_max:.3f}. Portafolio Min Varianza: retorno {ret_minvar:.1f}%, "
            f"volatilidad {vol_minvar:.1f}%. Tasa libre de riesgo (TPM): {rf*100:.2f}%. "
            "Interpreta estos resultados en el contexto del mercado accionario chileno (IPSA)."
        )
    else:
        prompt = (
            f"Max Sharpe portfolio: annual return {ret_max:.1f}%, volatility {vol_max:.1f}%, "
            f"Sharpe {sharpe_max:.3f}. Min Variance portfolio: return {ret_minvar:.1f}%, "
            f"volatility {vol_minvar:.1f}%. Risk-free rate (TPM): {rf*100:.2f}%. "
            "Interpret these results in the context of the Chilean equity market (IPSA)."
        )
    return _ask(prompt, lang)


def macro_commentary(
    best_lag: int,
    n_obs: int,
    ipc_forecast: float,
    tpm_forecast: float,
    usdclp_forecast: float,
    lang: str = "en",
) -> Optional[str]:
    if lang == "es":
        prompt = (
            f"Modelo VAR({best_lag}) con {n_obs} observaciones mensuales. "
            f"Pronóstico a 12 meses: IPC mensual {ipc_forecast*100:.3f}%, "
            f"TPM {tpm_forecast*100:.2f}%, USD/CLP Δlog {usdclp_forecast:.4f}. "
            "¿Qué sugiere este pronóstico sobre el ciclo macro chileno y la política monetaria del BCCh?"
        )
    else:
        prompt = (
            f"VAR({best_lag}) model with {n_obs} monthly observations. "
            f"12-month forecast: monthly IPC {ipc_forecast*100:.3f}%, "
            f"TPM {tpm_forecast*100:.2f}%, USD/CLP Δlog {usdclp_forecast:.4f}. "
            "What does this forecast imply for Chile's macro cycle and BCCh monetary policy?"
        )
    return _ask(prompt, lang)


def fixed_income_commentary(
    bei_vals: dict,
    bcp_10y: Optional[float],
    bcu_10y: Optional[float],
    tpm: Optional[float],
    lang: str = "en",
) -> Optional[str]:
    bei_str = ", ".join(f"BEI {k}: {v:.2f}%" for k, v in bei_vals.items())
    bcp_str = f"BCP 10Y: {bcp_10y*100:.2f}%. " if bcp_10y is not None else ""
    bcu_str = f"BCU 10Y: {bcu_10y*100:.2f}%. " if bcu_10y is not None else ""
    tpm_str = f"TPM actual: {tpm*100:.2f}%. "  if tpm    is not None else ""

    if lang == "es":
        prompt = (
            f"{bei_str}. Meta de inflación BCCh: 3%. "
            f"{bcp_str}{bcu_str}{tpm_str}"
            "Interpreta la curva de inflación de equilibrio y su posición respecto a la meta del BCCh."
        )
    else:
        prompt = (
            f"{bei_str}. BCCh inflation target: 3%. "
            f"{bcp_str}{bcu_str}{tpm_str}"
            "Interpret the breakeven inflation curve and its positioning relative to the BCCh target."
        )
    return _ask(prompt, lang)
