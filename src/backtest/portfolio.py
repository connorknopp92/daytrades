"""Result container for a backtest run."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class BacktestResult:
    """Everything produced by a single backtest, ready for metrics/plots."""

    equity_curve: pd.Series          # account value per bar (USD), DatetimeIndex
    trades: list = field(default_factory=list)  # list of trade dicts
    liquidated: bool = False         # did a leveraged position get wiped out?
    exposure: float = 0.0            # fraction of bars holding a position
    params: dict = field(default_factory=dict)  # backtest parameters used

    @property
    def n_trades(self) -> int:
        return len(self.trades)
