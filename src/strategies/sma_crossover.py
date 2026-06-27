"""SMA crossover trend-follower: long when fast SMA > slow SMA, else short."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class SmaCrossover(Strategy):
    name = "sma_crossover"
    family = "trend"
    classic = True
    blurb = "Long when the fast SMA is above the slow SMA, else short. Works in trends, whipsaws in choppy markets."

    def __init__(self, fast: int = 20, slow: int = 50):
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.fast = fast
        self.slow = slow

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        fast = close.rolling(self.fast, min_periods=self.fast).mean()
        slow = close.rolling(self.slow, min_periods=self.slow).mean()
        signal = pd.Series(np.where(fast > slow, 1.0, -1.0), index=df.index)
        signal[slow.isna()] = 0.0  # flat until both averages are defined
        return signal.rename("signal")
