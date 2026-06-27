"""Time-series / absolute momentum: long if the trailing return is positive.

Moskowitz-Ooi-Pedersen (2012) / Antonacci absolute momentum: if an asset's own
trailing-period return is positive, hold it long; otherwise move to cash. The
strongest-evidenced 'classic' rule in the research, though benefits are mostly
risk/drawdown reduction and are contested out-of-sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Strategy


class TsMomentum(Strategy):
    name = "ts_momentum"
    family = "momentum"
    classic = True
    blurb = "Long while the trailing-N-bar return is positive, else flat (cash). Rides persistent moves; exits downtrends."

    def __init__(self, lookback: int = 100):
        self.lookback = lookback

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        trailing = df["close"].pct_change(self.lookback)
        out = pd.Series(np.where(trailing > 0, 1.0, 0.0), index=df.index)
        out[trailing.isna()] = 0.0  # flat until the lookback window is available
        return out.rename("signal")
