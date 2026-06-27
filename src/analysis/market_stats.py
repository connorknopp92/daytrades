"""Descriptive market statistics for the 5-year analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_returns(df: pd.DataFrame) -> pd.Series:
    return df["close"].pct_change().dropna()


def drawdown_series(df: pd.DataFrame) -> pd.Series:
    """Drawdown of price from its running peak (negative fractions)."""
    price = df["close"]
    return price / price.cummax() - 1.0


def summary_stats(df: pd.DataFrame, periods_per_year: int = 365) -> dict:
    """Headline numbers describing the market over the sample."""
    r = daily_returns(df)
    price = df["close"]
    dd = drawdown_series(df)

    if len(price) >= 2 and price.iloc[0] > 0:
        total_ret = float(price.iloc[-1] / price.iloc[0] - 1.0)
        years = len(price) / periods_per_year
        cagr = float((price.iloc[-1] / price.iloc[0]) ** (1 / years) - 1.0) if years > 0 else 0.0
    else:
        total_ret = cagr = 0.0

    return {
        "start": df.index[0],
        "end": df.index[-1],
        "n_bars": len(df),
        "start_price": float(price.iloc[0]),
        "end_price": float(price.iloc[-1]),
        "total_return": total_ret,
        "cagr": cagr,
        "ann_volatility": float(r.std(ddof=0) * np.sqrt(periods_per_year)) if not r.empty else 0.0,
        "best_day": float(r.max()) if not r.empty else 0.0,
        "worst_day": float(r.min()) if not r.empty else 0.0,
        "max_drawdown": float(dd.min()) if not dd.empty else 0.0,
        "pct_up_days": float((r > 0).mean()) if not r.empty else 0.0,
    }


def rolling_volatility(df: pd.DataFrame, window: int = 30, periods_per_year: int = 365) -> pd.Series:
    r = daily_returns(df)
    return r.rolling(window).std(ddof=0) * np.sqrt(periods_per_year)


def rolling_correlation(df_a: pd.DataFrame, df_b: pd.DataFrame, window: int = 30) -> pd.Series:
    """Rolling correlation of daily returns between two assets (aligned index)."""
    ra = daily_returns(df_a)
    rb = daily_returns(df_b)
    joined = pd.concat([ra, rb], axis=1, join="inner")
    joined.columns = ["a", "b"]
    return joined["a"].rolling(window).corr(joined["b"])
