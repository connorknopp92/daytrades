"""Metric correctness on known inputs."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import metrics


def _equity(values):
    idx = pd.date_range("2020-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.Series(values, index=idx, dtype=float)


def test_total_return():
    eq = _equity([100, 110, 120])
    assert metrics.total_return(eq) == pytest.approx(0.2)


def test_max_drawdown():
    # Peak 120, trough 60 -> -50%.
    eq = _equity([100, 120, 60, 90])
    assert metrics.max_drawdown(eq) == -0.5


def test_max_drawdown_monotonic_up_is_zero():
    eq = _equity([100, 110, 120, 130])
    assert metrics.max_drawdown(eq) == 0.0


def test_cagr_doubling_in_one_year():
    # 365 daily bars, equity doubles -> ~100% CAGR.
    vals = np.linspace(100, 200, 365)
    eq = _equity(vals)
    assert metrics.cagr(eq, periods_per_year=365) == pytest.approx(1.0, rel=0.05)


def test_win_rate():
    eq = _equity([100, 110, 105, 120])  # up, down, up -> 2/3
    assert metrics.win_rate(eq) == pytest.approx(2 / 3)


def test_profit_factor_no_losses_is_inf():
    eq = _equity([100, 110, 120])
    assert metrics.profit_factor(eq) == float("inf")


def test_sharpe_zero_variance_is_zero():
    eq = _equity([100, 100, 100, 100])
    assert metrics.sharpe_ratio(eq, 365) == 0.0
