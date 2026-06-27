"""Risk-adjusted performance metrics computed from an equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2 or equity.iloc[0] == 0:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1.0)


def cagr(equity: pd.Series, periods_per_year: int) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return -1.0 if (len(equity) >= 2 and equity.iloc[-1] <= 0) else 0.0
    years = len(equity) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0)


def annualized_volatility(equity: pd.Series, periods_per_year: int) -> float:
    r = _returns(equity)
    if r.empty:
        return 0.0
    return float(r.std(ddof=0) * np.sqrt(periods_per_year))


def sharpe_ratio(equity: pd.Series, periods_per_year: int, risk_free: float = 0.0) -> float:
    r = _returns(equity)
    if r.empty or r.std(ddof=0) == 0:
        return 0.0
    excess = r - risk_free / periods_per_year
    return float(excess.mean() / r.std(ddof=0) * np.sqrt(periods_per_year))


def sortino_ratio(equity: pd.Series, periods_per_year: int, risk_free: float = 0.0) -> float:
    r = _returns(equity)
    if r.empty:
        return 0.0
    excess = r - risk_free / periods_per_year
    downside = excess[excess < 0]
    dd = downside.std(ddof=0)
    if dd == 0 or np.isnan(dd):
        return 0.0
    return float(excess.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction (e.g. -0.62)."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(equity: pd.Series, periods_per_year: int) -> float:
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / abs(mdd))


def win_rate(equity: pd.Series) -> float:
    """Fraction of periods with a positive return."""
    r = _returns(equity)
    if r.empty:
        return 0.0
    return float((r > 0).mean())


def profit_factor(equity: pd.Series) -> float:
    """Gross gains / gross losses across periods."""
    r = _returns(equity)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def compute_all(equity: pd.Series, periods_per_year: int, n_trades: int = 0,
                exposure: float = 0.0, liquidated: bool = False) -> dict:
    """Bundle every metric into a single dict for reporting."""
    return {
        "total_return": total_return(equity),
        "cagr": cagr(equity, periods_per_year),
        "annualized_volatility": annualized_volatility(equity, periods_per_year),
        "sharpe": sharpe_ratio(equity, periods_per_year),
        "sortino": sortino_ratio(equity, periods_per_year),
        "max_drawdown": max_drawdown(equity),
        "calmar": calmar_ratio(equity, periods_per_year),
        "win_rate": win_rate(equity),
        "profit_factor": profit_factor(equity),
        "n_trades": int(n_trades),
        "exposure": float(exposure),
        "liquidated": bool(liquidated),
        "final_equity": float(equity.iloc[-1]) if not equity.empty else 0.0,
    }


def format_metrics(metrics: dict) -> str:
    """Human-readable, terminal-friendly metrics table."""
    pct = lambda x: f"{x * 100:,.2f}%"
    lines = [
        f"  Total return    : {pct(metrics['total_return'])}",
        f"  CAGR            : {pct(metrics['cagr'])}",
        f"  Ann. volatility : {pct(metrics['annualized_volatility'])}",
        f"  Sharpe          : {metrics['sharpe']:.2f}",
        f"  Sortino         : {metrics['sortino']:.2f}",
        f"  Max drawdown    : {pct(metrics['max_drawdown'])}",
        f"  Calmar          : {metrics['calmar']:.2f}",
        f"  Win rate        : {pct(metrics['win_rate'])}",
        f"  Profit factor   : {metrics['profit_factor']:.2f}",
        f"  # trades        : {metrics['n_trades']}",
        f"  Exposure        : {pct(metrics['exposure'])}",
        f"  Final equity    : ${metrics['final_equity']:,.2f}",
    ]
    if metrics.get("liquidated"):
        lines.append("  ** LIQUIDATED ** position was wiped out by leverage.")
    return "\n".join(lines)
