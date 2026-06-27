"""Strategy interface.

A strategy maps an OHLCV DataFrame to a target-position series in {-1, 0, +1}
(short / flat / long). Keeping the output to a simple position series makes every
strategy directly comparable and easy to feed into the backtest engine.

Strategies must not peek into the future: any indicator that uses a window
should rely only on data up to and including the current bar.

Each strategy also carries light metadata for the UI/digest/docs:
- ``family``  : trend / momentum / mean_reversion / breakout / benchmark
- ``classic`` : True for the well-known, evidence-backed rules (⭐). Per the
                deep-research review, the strongest evidence is for trend/momentum
                (Faber SMA timing, time-series/absolute momentum); other families
                are widely used but their edge is weaker/mixed.
- ``blurb``   : one-line rule + the regime where it tends to work or fail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    name: str = "base"
    family: str = "other"
    classic: bool = False
    blurb: str = ""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Return a position series (index-aligned to df) with values in {-1,0,1}."""
        raise NotImplementedError


def _registry() -> dict:
    """Map of strategy name -> class (lazy imports avoid import cycles)."""
    from .buy_and_hold import BuyAndHold
    from .sma_crossover import SmaCrossover
    from .rsi_reversion import RsiReversion
    from .donchian_breakout import DonchianBreakout
    from .macd import Macd
    from .sma_timing import SmaTiming
    from .ts_momentum import TsMomentum
    from .momentum_roc import MomentumRoc
    from .bollinger_reversion import BollingerReversion
    from .zscore_reversion import ZScoreReversion
    from .keltner_breakout import KeltnerBreakout

    return {
        "buy_and_hold": BuyAndHold,
        "sma_crossover": SmaCrossover,
        "sma_timing": SmaTiming,
        "macd": Macd,
        "ts_momentum": TsMomentum,
        "momentum_roc": MomentumRoc,
        "donchian_breakout": DonchianBreakout,
        "keltner_breakout": KeltnerBreakout,
        "rsi_reversion": RsiReversion,
        "bollinger_reversion": BollingerReversion,
        "zscore_reversion": ZScoreReversion,
    }


def get_strategy(name: str, **params) -> Strategy:
    """Factory: look up a strategy by name."""
    registry = _registry()
    if name not in registry:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {', '.join(sorted(registry))}"
        )
    return registry[name](**params)


def available_strategies() -> list[str]:
    """All strategy names, in a sensible display order."""
    return list(_registry().keys())


def strategy_meta(name: str) -> dict:
    """Metadata for one strategy (family, classic flag, blurb)."""
    cls = _registry()[name]
    return {"name": name, "family": cls.family, "classic": cls.classic, "blurb": cls.blurb}
