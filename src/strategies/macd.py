"""MACD trend strategy: long when the MACD line is above its signal line."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class Macd(Strategy):
    name = "macd"
    family = "trend"
    classic = True
    blurb = "Long when MACD (EMA12−EMA26) is above its 9-period signal line, else short. A classic trend/momentum oscillator."

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        if fast >= slow:
            raise ValueError("fast must be shorter than slow")
        self.fast, self.slow, self.signal = fast, slow, signal

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        macd = close.ewm(span=self.fast, adjust=False).mean() - \
            close.ewm(span=self.slow, adjust=False).mean()
        sig_line = macd.ewm(span=self.signal, adjust=False).mean()
        out = pd.Series(np.where(macd > sig_line, 1.0, -1.0), index=df.index)
        # Flat until the slow EMA has enough data to be meaningful.
        out.iloc[: self.slow] = 0.0
        return out.rename("signal")
