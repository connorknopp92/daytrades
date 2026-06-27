"""Accumulation / entry-timing analysis: *when* to buy for a long-term hold.

This answers the honest version of "when should I buy?" — not by predicting the
bottom (nobody can), but by testing systematic accumulation rules against simply
buying on a fixed schedule. It also reports a plain-language signal for *today*
based on those same rules (current conditions, NOT a forecast).

Strategies compared (all contribute a fixed weekly budget):
- ``dca``           : invest the weekly budget every week, no matter the price.
- ``lump_sum``      : invest the whole budget (all weeks at once) on day one.
- ``buy_the_dip``   : stash the weekly budget as cash; deploy it all only when
                      price is >= ``dip_pct`` below its rolling ~1y high.
- ``below_ma``      : stash cash; deploy only when price is below its long-term
                      moving average (default 200 days).

Waiting strategies can end with idle cash — that "cash drag" is itself a lesson.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weekly_frame(df: pd.DataFrame, years: int, ma_days: int, high_window_weeks: int = 52) -> pd.DataFrame:
    """Resample daily OHLCV to weekly closes with MA and rolling-high columns."""
    daily_close = df["close"]
    ma = daily_close.rolling(ma_days, min_periods=ma_days).mean()

    wk = pd.DataFrame({
        "close": daily_close.resample("W").last(),
        "ma": ma.resample("W").last(),
    }).dropna(subset=["close"])
    wk["high"] = wk["close"].rolling(high_window_weeks, min_periods=1).max()

    if years:
        cutoff = wk.index[-1] - pd.Timedelta(days=int(years * 365))
        wk = wk[wk.index >= cutoff]
    return wk


def _simulate(wk: pd.DataFrame, weekly_budget: float, dip_pct: float):
    """Run all accumulation strategies over the weekly frame.

    Returns (per_strategy_results, value_timeseries_dataframe).
    """
    n = len(wk)
    prices = wk["close"].to_numpy(dtype=float)
    ma = wk["ma"].to_numpy(dtype=float)
    high = wk["high"].to_numpy(dtype=float)
    total_budget = weekly_budget * n

    # coins held + idle cash for each strategy as we step week by week.
    state = {
        "dca": {"coins": 0.0, "cash": 0.0},
        "lump_sum": {"coins": 0.0, "cash": 0.0},
        "buy_the_dip": {"coins": 0.0, "cash": 0.0},
        "below_ma": {"coins": 0.0, "cash": 0.0},
    }
    spent = {k: 0.0 for k in state}  # cash actually converted into coins
    value_rows = {k: np.empty(n) for k in state}
    contributed = np.empty(n)

    for t in range(n):
        price = prices[t]

        # DCA: invest this week's budget immediately.
        state["dca"]["coins"] += weekly_budget / price
        spent["dca"] += weekly_budget

        # Lump sum: deploy the entire budget on the first week.
        if t == 0:
            state["lump_sum"]["coins"] += total_budget / price
            spent["lump_sum"] += total_budget

        # Buy-the-dip: accumulate cash, deploy all when price is in a dip.
        d = state["buy_the_dip"]
        d["cash"] += weekly_budget
        if price <= high[t] * (1.0 - dip_pct):
            spent["buy_the_dip"] += d["cash"]
            d["coins"] += d["cash"] / price
            d["cash"] = 0.0

        # Below-MA: accumulate cash, deploy all when price < long-term MA.
        m = state["below_ma"]
        m["cash"] += weekly_budget
        if not np.isnan(ma[t]) and price < ma[t]:
            spent["below_ma"] += m["cash"]
            m["coins"] += m["cash"] / price
            m["cash"] = 0.0

        contributed[t] = weekly_budget * (t + 1)
        for k in state:
            value_rows[k][t] = state[k]["coins"] * price + state[k]["cash"]

    last_price = prices[-1]
    results = {}
    for k, s in state.items():
        final_value = s["coins"] * last_price + s["cash"]
        results[k] = {
            "contributed": float(total_budget),
            "invested": float(spent[k]),
            "idle_cash": float(s["cash"]),
            "coins": float(s["coins"]),
            "final_value": float(final_value),
            "roi": float(final_value / total_budget - 1.0) if total_budget else 0.0,
            "avg_cost": float(spent[k] / s["coins"]) if s["coins"] > 0 else float("nan"),
            "pct_uninvested": float(s["cash"] / total_budget) if total_budget else 0.0,
        }

    ts = pd.DataFrame(value_rows, index=wk.index)
    ts["contributed"] = contributed
    return results, ts


def accumulation_backtest(cfg: dict, symbol: str, df: pd.DataFrame,
                          years: int = 3, weekly_budget: float = 50.0,
                          dip_pct: float = 0.20, ma_days: int = 200) -> dict:
    """Backtest weekly accumulation strategies on ``df`` over the last ``years``."""
    wk = _weekly_frame(df, years, ma_days)
    if len(wk) < 2:
        raise ValueError("Not enough history for the requested window.")
    results, ts = _simulate(wk, weekly_budget, dip_pct)
    return {
        "results": results,
        "value_ts": ts,
        "weeks": len(wk),
        "weekly_budget": weekly_budget,
        "years": years,
        "dip_pct": dip_pct,
        "ma_days": ma_days,
        "start": wk.index[0],
        "end": wk.index[-1],
        "synthetic": bool(df.attrs.get("synthetic", False)),
    }


def current_signal(df: pd.DataFrame, dip_pct: float = 0.20, ma_days: int = 200,
                   high_window_days: int = 365) -> dict:
    """Plain-language 'is now a good time to accumulate?' read on TODAY.

    Reflects current conditions versus the same rules we backtest — it is a
    description of the present, not a prediction of the future.
    """
    close = df["close"]
    price = float(close.iloc[-1])
    ma_series = close.rolling(ma_days, min_periods=ma_days).mean()
    ma = ma_series.iloc[-1]
    ma = float(ma) if not pd.isna(ma) else None

    window = close[close.index >= close.index[-1] - pd.Timedelta(days=high_window_days)]
    recent_high = float(window.max())
    drawdown_from_high = price / recent_high - 1.0
    below_ma = ma is not None and price < ma
    in_dip = drawdown_from_high <= -dip_pct

    if below_ma and in_dip:
        verdict = "FAVORABLE"
        note = "Price is both below its long-term trend and in a dip — historically the kind of window the rules accumulate into."
    elif below_ma or in_dip:
        verdict = "LEANING FAVORABLE"
        note = ("Below the long-term average." if below_ma else
                f"In a dip ({drawdown_from_high*100:.0f}% off the recent high).")
    else:
        verdict = "NEUTRAL"
        note = "Price is above its long-term trend and not in a meaningful dip — the rules would hold cash and wait."

    return {
        "price": price,
        "ma": ma,
        "ma_days": ma_days,
        "pct_vs_ma": (price / ma - 1.0) if ma else None,
        "recent_high": recent_high,
        "drawdown_from_high": drawdown_from_high,
        "below_ma": below_ma,
        "in_dip": in_dip,
        "verdict": verdict,
        "note": note,
        "disclaimer": "This describes current conditions vs. tested rules. It is NOT a "
                      "prediction and not financial advice. No one can reliably time the bottom.",
    }


# Human-friendly labels for the UI / reports.
STRATEGY_LABELS = {
    "dca": "Buy weekly (DCA)",
    "lump_sum": "Lump sum (all at start)",
    "buy_the_dip": "Buy the dip",
    "below_ma": "Buy below 200-day avg",
}
