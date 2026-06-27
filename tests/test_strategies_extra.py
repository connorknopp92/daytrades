"""Tests for the expanded strategy library + metadata."""

import numpy as np
import pandas as pd

from src.strategies.base import available_strategies, get_strategy, strategy_meta
from tests.conftest import make_ohlcv


def test_eleven_strategies_registered():
    names = available_strategies()
    assert len(names) == 11
    for expected in ["macd", "sma_timing", "ts_momentum", "momentum_roc",
                     "bollinger_reversion", "zscore_reversion", "keltner_breakout"]:
        assert expected in names


def test_classic_flags_cover_trend_momentum():
    classics = {n for n in available_strategies() if strategy_meta(n)["classic"]}
    # The evidence-backed trend/momentum set is flagged classic.
    assert {"sma_timing", "ts_momentum", "macd"} <= classics
    # Mean-reversion is not flagged classic.
    assert not strategy_meta("bollinger_reversion")["classic"]


def test_every_strategy_has_family_and_blurb():
    for n in available_strategies():
        m = strategy_meta(n)
        assert m["family"]
        assert m["blurb"]


def test_sma_timing_goes_to_cash_in_downtrend():
    close = 100.0 * (0.99 ** np.arange(60))
    sig = get_strategy("sma_timing").generate_signals(make_ohlcv(close))
    assert sig.iloc[-1] == 0.0          # below its SMA -> flat (cash)
    assert set(np.unique(sig)) <= {0.0, 1.0}   # long/flat only, never short


def test_ts_momentum_long_in_uptrend_flat_in_downtrend():
    up = 100.0 * (1.01 ** np.arange(200))
    down = 100.0 * (0.99 ** np.arange(200))
    assert get_strategy("ts_momentum").generate_signals(make_ohlcv(up)).iloc[-1] == 1.0
    assert get_strategy("ts_momentum").generate_signals(make_ohlcv(down)).iloc[-1] == 0.0


def test_macd_two_sided_in_uptrend():
    close = 100.0 * (1.01 ** np.arange(80))
    sig = get_strategy("macd").generate_signals(make_ohlcv(close))
    assert sig.iloc[-1] == 1.0          # rising -> MACD above signal -> long
    assert sig.iloc[0] == 0.0           # flat until slow EMA is meaningful


def test_bollinger_long_below_lower_band():
    # Flat series, then a sharp one-bar drop -> close below the lower band -> long.
    close = [100.0] * 25 + [80.0]
    sig = get_strategy("bollinger_reversion").generate_signals(make_ohlcv(close))
    assert sig.iloc[-1] == 1.0


def test_keltner_no_lookahead_first_bar_flat():
    close = 100.0 * (1.01 ** np.arange(60))
    sig = get_strategy("keltner_breakout").generate_signals(make_ohlcv(close))
    assert sig.iloc[0] == 0.0
