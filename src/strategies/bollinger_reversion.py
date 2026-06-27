"""Bollinger Band mean-reversion: buy below the lower band, short above the upper."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class BollingerReversion(Strategy):
    name = "bollinger_reversion"
    family = "mean_reversion"
    classic = False
    blurb = "Long when price closes below the lower band (mean−2σ), short above the upper band. Works in ranges, fights trends."

    def __init__(self, window: int = 20, num_std: float = 2.0):
        self.window = window
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        mid = close.rolling(self.window, min_periods=self.window).mean()
        std = close.rolling(self.window, min_periods=self.window).std(ddof=0)
        upper = mid + self.num_std * std
        lower = mid - self.num_std * std

        out = pd.Series(0.0, index=df.index)
        out[close < lower] = 1.0    # oversold -> long
        out[close > upper] = -1.0   # overbought -> short
        out[mid.isna()] = 0.0
        return out.rename("signal")
