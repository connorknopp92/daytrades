"""Strategy signal shape and value checks on synthetic data."""

import numpy as np
import pandas as pd

from src.strategies.base import get_strategy, available_strategies
from tests.conftest import make_ohlcv


def test_all_strategies_return_valid_signals(uptrend):
    for name in available_strategies():
        strat = get_strategy(name)
        sig = strat.generate_signals(uptrend)
        assert isinstance(sig, pd.Series)
        assert sig.index.equals(uptrend.index)
        assert set(np.unique(sig.dropna())).issubset({-1.0, 0.0, 1.0})


def test_buy_and_hold_always_long(uptrend):
    sig = get_strategy("buy_and_hold").generate_signals(uptrend)
    assert (sig == 1.0).all()


def test_sma_crossover_goes_long_in_uptrend(uptrend):
    sig = get_strategy("sma_crossover").generate_signals(uptrend)
    # Once both SMAs are defined in a clean uptrend, position should be long.
    assert sig.iloc[-1] == 1.0


def test_sma_crossover_goes_short_in_downtrend():
    close = 100.0 * (0.99 ** np.arange(80))
    df = make_ohlcv(close)
    sig = get_strategy("sma_crossover").generate_signals(df)
    assert sig.iloc[-1] == -1.0


def test_donchian_no_lookahead_first_bars_flat():
    close = 100.0 * (1.01 ** np.arange(40))
    df = make_ohlcv(close)
    sig = get_strategy("donchian_breakout").generate_signals(df)
    # Before the channel window fills, the strategy must be flat (0).
    assert sig.iloc[0] == 0.0


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        get_strategy("definitely_not_real")
