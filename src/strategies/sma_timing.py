"""Faber-style trend timing: long when price is above its long SMA, else flat.

Meb Faber's 'A Quantitative Approach to Tactical Asset Allocation' (the 10-month
SMA rule): hold the asset while price > SMA, move to cash when price < SMA. One
of the most-cited, evidence-backed timing rules — its documented benefit is
drawdown reduction (it sidesteps sustained bear markets), not higher returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class SmaTiming(Strategy):
    name = "sma_timing"
    family = "trend"
    classic = True
    blurb = "Faber timing: long while price is above its 10-period SMA, else flat (cash). Cuts deep drawdowns; whipsaws when flat."

    def __init__(self, window: int = 10):
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        sma = df["close"].rolling(self.window, min_periods=self.window).mean()
        out = pd.Series(np.where(df["close"] > sma, 1.0, 0.0), index=df.index)
        out[sma.isna()] = 0.0  # flat until the SMA is defined
        return out.rename("signal")
