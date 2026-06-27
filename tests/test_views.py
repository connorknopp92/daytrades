"""Tests for daily/weekly/monthly resampling and view-aware annualization."""

import numpy as np
import pandas as pd

from src.config import load_config
from src import service


def _daily(n=40, start="2020-01-01"):
    idx = pd.date_range(start, periods=n, freq="D", tz="UTC")
    close = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1,
         "close": close, "volume": np.ones(n)},
        index=idx,
    )


def test_daily_view_is_noop():
    df = _daily(10)
    assert service.resample_ohlcv(df, "daily") is df


def test_resample_preserves_endpoints_and_volume():
    df = _daily(40)
    for view in ["weekly", "monthly"]:
        r = service.resample_ohlcv(df, view)
        assert len(r) < len(df)                      # fewer, larger candles
        assert r["open"].iloc[0] == df["open"].iloc[0]
        assert r["close"].iloc[-1] == df["close"].iloc[-1]
        assert r["volume"].sum() == df["volume"].sum()   # volume conserved
        assert (r["high"] >= r["low"]).all()


def test_periods_per_year_by_view():
    cfg = load_config()
    assert service.periods_per_year(cfg, "BTC/USD", "weekly") == 52
    assert service.periods_per_year(cfg, "AAPL", "monthly") == 12
    assert service.periods_per_year(cfg, "AAPL", "daily") == 252
    assert service.periods_per_year(cfg, "BTC/USD", "daily") == cfg["backtest"]["periods_per_year"]


def test_slice_years():
    idx = pd.date_range("2010-01-01", periods=4000, freq="D", tz="UTC")
    df = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
                       "volume": 1.0}, index=idx)
    one = service.slice_years(df, 1)
    assert len(one) < len(df)
    assert one.index[0] >= df.index[-1] - pd.Timedelta(days=366)
    assert service.slice_years(df, None) is df       # None = keep everything


def test_market_summary_weekly_runs():
    cfg = load_config()
    df = _daily(400)
    out = service.market_summary(cfg, "BTC/USD", df=service.resample_ohlcv(df, "weekly"),
                                 view="weekly")
    assert "stats" in out and out["stats"]["n_bars"] < 400
