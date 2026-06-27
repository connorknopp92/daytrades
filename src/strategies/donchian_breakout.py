"""Donchian channel breakout: long on new highs, short on new lows, else hold."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class DonchianBreakout(Strategy):
    name = "donchian_breakout"

    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        # Channel from the PRIOR window only (shift(1)) so the current bar's
        # own high/low can't leak into its own breakout test.
        upper = df["high"].rolling(self.window, min_periods=self.window).max().shift(1)
        lower = df["low"].rolling(self.window, min_periods=self.window).min().shift(1)
        close = df["close"]

        raw = pd.Series(np.nan, index=df.index)
        raw[close >= upper] = 1.0
        raw[close <= lower] = -1.0
        # Hold the last breakout direction until the opposite breakout fires.
        signal = raw.ffill().fillna(0.0)
        return signal.rename("signal")
