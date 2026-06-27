"""Service-layer smoke tests (offline, using synthetic data)."""

import pandas as pd

from src.config import load_config
from src import service
from src.data.fetch import synthetic_ohlcv


def _cfg_and_df():
    cfg = load_config()
    df = synthetic_ohlcv("BTC/USD", "1d", years=2, seed=11)
    return cfg, df


def test_market_summary_shapes():
    cfg, df = _cfg_and_df()
    out = service.market_summary(cfg, "BTC/USD", df=df)
    assert set(out) >= {"stats", "price", "drawdown", "rolling_vol", "synthetic"}
    assert out["synthetic"] is True
    assert len(out["price"]) == len(df)
    for key in ["total_return", "cagr", "ann_volatility", "max_drawdown"]:
        assert key in out["stats"]


def test_backtest_summary_returns_both_curves():
    cfg, df = _cfg_and_df()
    out = service.backtest_summary(cfg, "BTC/USD", "sma_crossover",
                                   leverage=1.0, fee=0.001, slippage=0.0005,
                                   funding=0.0, df=df)
    assert isinstance(out["strat_equity"], pd.Series)
    assert isinstance(out["bench_equity"], pd.Series)
    for key in ["total_return", "max_drawdown", "sharpe", "final_equity"]:
        assert key in out["strat_metrics"]
        assert key in out["bench_metrics"]
    assert out["params"]["leverage"] == 1.0


def test_backtest_summary_defaults_from_config():
    cfg, df = _cfg_and_df()
    out = service.backtest_summary(cfg, "BTC/USD", "buy_and_hold", df=df)
    assert out["params"]["fee"] == cfg["backtest"]["fee"]
    assert out["params"]["leverage"] == cfg["backtest"]["leverage"]


def test_high_leverage_liquidates():
    from tests.conftest import make_ohlcv
    # A deterministic ~30% crash on day 3 must wipe out a 25x long.
    close = [100.0, 100.0, 70.0, 75.0, 80.0]
    df = make_ohlcv(close, high=close, low=close)
    out = service.backtest_summary(cfg=load_config(), symbol="BTC/USD",
                                   strategy_name="buy_and_hold", leverage=25.0,
                                   fee=0.001, slippage=0.001, funding=0.0, df=df)
    assert out["strat_metrics"]["liquidated"] is True
