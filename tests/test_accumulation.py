"""Tests for the accumulation / entry-timing analysis (offline, synthetic data)."""

import numpy as np
import pandas as pd

from src.config import load_config
from src import accumulation as accum
from src.data.fetch import synthetic_ohlcv
from tests.conftest import make_ohlcv


def _cfg_df():
    return load_config(), synthetic_ohlcv("BTC/USD", "1d", years=4, seed=5)


def test_backtest_has_all_strategies():
    cfg, df = _cfg_df()
    bt = accum.accumulation_backtest(cfg, "BTC/USD", df, years=3, weekly_budget=50.0)
    assert set(bt["results"]) == {"dca", "lump_sum", "buy_the_dip", "below_ma"}
    assert bt["weeks"] > 100  # ~3 years of weekly buys
    # value time series aligns and includes a contributions column.
    assert "contributed" in bt["value_ts"].columns
    assert len(bt["value_ts"]) == bt["weeks"]


def test_contributions_match_budget_times_weeks():
    cfg, df = _cfg_df()
    budget = 25.0
    bt = accum.accumulation_backtest(cfg, "BTC/USD", df, years=2, weekly_budget=budget)
    expected = budget * bt["weeks"]
    for r in bt["results"].values():
        assert r["contributed"] == expected


def test_dca_fully_invests_no_idle_cash():
    cfg, df = _cfg_df()
    bt = accum.accumulation_backtest(cfg, "BTC/USD", df, years=3, weekly_budget=50.0)
    assert bt["results"]["dca"]["idle_cash"] == 0.0
    assert bt["results"]["dca"]["coins"] > 0


def test_rising_market_lump_sum_beats_dca():
    """In a steady uptrend, investing everything early should beat spreading it out."""
    cfg = load_config()
    close = 100.0 * (1.003 ** np.arange(1500))  # ~4 years of daily gains
    df = make_ohlcv(close)
    bt = accum.accumulation_backtest(cfg, "BTC/USD", df, years=3, weekly_budget=50.0)
    assert bt["results"]["lump_sum"]["final_value"] > bt["results"]["dca"]["final_value"]


def test_current_signal_below_ma_is_favorable():
    # A long decline ends below its moving average and well off the high.
    close = 100.0 * (0.999 ** np.arange(400))
    df = make_ohlcv(close)
    sig = accum.current_signal(df, dip_pct=0.20, ma_days=200)
    assert sig["below_ma"] is True
    assert sig["verdict"] in {"FAVORABLE", "LEANING FAVORABLE"}
    assert "price" in sig and "drawdown_from_high" in sig


def test_current_signal_at_highs_is_neutral():
    close = 100.0 * (1.002 ** np.arange(400))  # steadily making new highs
    df = make_ohlcv(close)
    sig = accum.current_signal(df)
    assert sig["below_ma"] is False
    assert sig["in_dip"] is False
    assert sig["verdict"] == "NEUTRAL"
