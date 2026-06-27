"""Rate-of-change momentum: long when ROC is positive, short when negative."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class MomentumRoc(Strategy):
    name = "momentum_roc"
    family = "momentum"
    classic = False
    blurb = "Long when the N-bar rate-of-change is positive, short when negative. A simple two-sided momentum tilt."

    def __init__(self, window: int = 20):
        self.window = window

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        roc = df["close"].pct_change(self.window)
        out = pd.Series(np.where(roc > 0, 1.0, -1.0), index=df.index)
        out[roc.isna()] = 0.0
        return out.rename("signal")
