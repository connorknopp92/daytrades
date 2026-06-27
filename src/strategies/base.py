"""Strategy interface.

A strategy maps an OHLCV DataFrame to a target-position series in {-1, 0, +1}
(short / flat / long). Keeping the output to a simple position series makes every
strategy directly comparable and easy to feed into the backtest engine.

Strategies must not peek into the future: any indicator that uses a window
should rely only on data up to and including the current bar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a position series (index-aligned to df) with values in {-1,0,1}."""
        raise NotImplementedError


def get_strategy(name: str, **params) -> Strategy:
    """Factory: look up a strategy by name (lazy imports avoid cycles)."""
    from .buy_and_hold import BuyAndHold
    from .sma_crossover import SmaCrossover
    from .rsi_reversion import RsiReversion
    from .donchian_breakout import DonchianBreakout

    registry = {
        "buy_and_hold": BuyAndHold,
        "sma_crossover": SmaCrossover,
        "rsi_reversion": RsiReversion,
        "donchian_breakout": DonchianBreakout,
    }
    if name not in registry:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {', '.join(sorted(registry))}"
        )
    return registry[name](**params)


def available_strategies() -> list[str]:
    return ["buy_and_hold", "sma_crossover", "rsi_reversion", "donchian_breakout"]
