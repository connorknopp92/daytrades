"""RSI mean-reversion: go long when oversold, short when overbought, else flat."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[avg_loss == 0.0] = 100.0  # no losses -> maximally overbought
    return out


class RsiReversion(Strategy):
    name = "rsi_reversion"

    def __init__(self, period: int = 14, oversold: float = 30.0, overbought: float = 70.0):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        r = rsi(df["close"], self.period)
        signal = pd.Series(0.0, index=df.index)
        signal[r < self.oversold] = 1.0     # oversold -> buy the dip
        signal[r > self.overbought] = -1.0  # overbought -> short
        return signal.rename("signal")
