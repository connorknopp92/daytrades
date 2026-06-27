"""Buy-and-hold benchmark: always long. The bar every strategy must beat."""

from __future__ import annotations

import pandas as pd

from .base import Strategy


class BuyAndHold(Strategy):
    name = "buy_and_hold"
    family = "benchmark"
    classic = False
    blurb = "Always long — the yardstick every strategy must beat (it usually wins net of costs)."

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=df.index, name="signal")
