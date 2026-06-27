"""Z-score mean-reversion: fade moves far from a rolling mean."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class ZScoreReversion(Strategy):
    name = "zscore_reversion"
    family = "mean_reversion"
    classic = False
    blurb = "Long when price is >1σ below its rolling mean, short when >1σ above. A simple statistical fade; works in ranges."

    def __init__(self, window: int = 20, threshold: float = 1.0):
        self.window = window
        self.threshold = threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        close = df["close"]
        mean = close.rolling(self.window, min_periods=self.window).mean()
        std = close.rolling(self.window, min_periods=self.window).std(ddof=0)
        z = (close - mean) / std.replace(0.0, pd.NA)

        out = pd.Series(0.0, index=df.index)
        out[z < -self.threshold] = 1.0
        out[z > self.threshold] = -1.0
        out[z.isna()] = 0.0
        return out.rename("signal")
