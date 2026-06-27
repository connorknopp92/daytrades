"""Keltner Channel breakout: long on a close above the upper ATR band."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def average_true_range(df: pd.DataFrame, window: int) -> pd.Series:
    """Wilder-style ATR via a simple rolling mean of the true range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()


class KeltnerBreakout(Strategy):
    name = "keltner_breakout"
    family = "breakout"
    classic = False
    blurb = "Long on a close above EMA+2·ATR, short below EMA−2·ATR; holds until the opposite band breaks. Volatility-aware breakout."

    def __init__(self, window: int = 20, atr_mult: float = 2.0):
        self.window = window
        self.atr_mult = atr_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        ema = df["close"].ewm(span=self.window, adjust=False).mean()
        atr = average_true_range(df, self.window)
        # Use the PRIOR bar's channel so the current bar's own range can't leak in.
        upper = (ema + self.atr_mult * atr).shift(1)
        lower = (ema - self.atr_mult * atr).shift(1)
        close = df["close"]

        raw = pd.Series(np.nan, index=df.index)
        raw[close >= upper] = 1.0
        raw[close <= lower] = -1.0
        return raw.ffill().fillna(0.0).rename("signal")
