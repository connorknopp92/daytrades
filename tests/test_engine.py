"""Backtest engine behavior on deterministic price series."""

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_backtest
from tests.conftest import make_ohlcv


def _long_signal(df):
    return pd.Series(1.0, index=df.index)


def test_no_cost_long_tracks_price(uptrend):
    """With zero costs and leverage 1, a long should match the price return."""
    sig = _long_signal(uptrend)
    res = run_backtest(uptrend, sig, initial_capital=1000, leverage=1.0,
                       fee=0.0, slippage=0.0, funding_rate=0.0)
    price_return = uptrend["close"].iloc[-1] / uptrend["close"].iloc[0] - 1
    strat_return = res.equity_curve.iloc[-1] / 1000 - 1
    assert strat_return == pytest.approx(price_return, rel=1e-6)
    assert not res.liquidated
    assert res.exposure == 1.0


def test_fees_erode_equity_in_flat_market(flat_market):
    """A flat market with fees can only lose money to costs."""
    sig = _long_signal(flat_market)
    res = run_backtest(flat_market, sig, initial_capital=1000, leverage=1.0,
                       fee=0.01, slippage=0.0, funding_rate=0.0)
    assert res.equity_curve.iloc[-1] < 1000
    assert not res.liquidated


def test_high_leverage_liquidates_on_drawdown():
    """10x long into a ~11% drop should be liquidated and wiped out."""
    close = [100.0, 100.0, 89.0, 95.0]
    df = make_ohlcv(close, high=close, low=close)
    sig = _long_signal(df)
    res = run_backtest(df, sig, initial_capital=1000, leverage=10.0,
                       fee=0.0, slippage=0.0, funding_rate=0.0,
                       maintenance_margin=0.005)
    assert res.liquidated
    assert res.equity_curve.iloc[-1] == 0.0


def test_low_leverage_survives_same_drawdown():
    """The same ~11% drop at 1x must NOT liquidate."""
    close = [100.0, 100.0, 89.0, 95.0]
    df = make_ohlcv(close, high=close, low=close)
    sig = _long_signal(df)
    res = run_backtest(df, sig, initial_capital=1000, leverage=1.0,
                       fee=0.0, slippage=0.0, funding_rate=0.0)
    assert not res.liquidated
    assert res.equity_curve.iloc[-1] > 0.0


def test_flat_signal_stays_in_cash(flat_market):
    """A zero signal never trades and equity is unchanged."""
    sig = pd.Series(0.0, index=flat_market.index)
    res = run_backtest(flat_market, sig, initial_capital=1000, leverage=1.0,
                       fee=0.01, slippage=0.01)
    assert res.equity_curve.iloc[-1] == pytest.approx(1000)
    assert res.n_trades == 0
    assert res.exposure == 0.0
