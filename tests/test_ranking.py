"""Tests for the walk-forward strategy ranking engine."""

import numpy as np

from src.config import load_config
from src import ranking
from src.strategies.base import available_strategies
from tests.conftest import make_ohlcv


def _trending_df(n=500, seed=3):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0008, 0.02, n)))
    return make_ohlcv(close, high=close * 1.01, low=close * 0.99)


def test_rank_strategies_structure():
    cfg = load_config()
    df = _trending_df()
    r = ranking.rank_strategies(cfg, "BTC/USD", df, view="daily")
    assert r["n_strategies_tried"] == len(available_strategies())
    assert set(r["best_by"]) == {"sharpe", "total_return", "calmar"}
    assert r["best_by"]["sharpe"] in [row["name"] for row in r["rows"]]
    assert r["train_period"] is not None and r["holdout_period"] is not None
    for row in r["rows"]:
        for key in ("sharpe", "total_return", "calmar"):
            assert key in row["train"]
        assert row["holdout"] is not None  # 500 bars -> holdout exists


def test_rank_is_deterministic():
    cfg = load_config()
    df = _trending_df()
    a = ranking.rank_strategies(cfg, "BTC/USD", df)
    b = ranking.rank_strategies(cfg, "BTC/USD", df)
    assert a["best_by"] == b["best_by"]


def test_best_for_symbol_summary():
    cfg = load_config()
    df = _trending_df(n=900)
    out = ranking.best_for_symbol(cfg, "BTC/USD", df, years=None)
    assert out["winner"] in available_strategies()
    assert out["metric"] == "sharpe"
    assert out["n_strategies_tried"] == len(available_strategies())
    # hold-out returns are present (numbers) for winner and benchmark
    assert out["winner_holdout_return"] is not None
    assert out["benchmark_holdout_return"] is not None


def test_short_history_has_no_holdout():
    cfg = load_config()
    df = _trending_df(n=8)
    r = ranking.rank_strategies(cfg, "BTC/USD", df)
    assert r["holdout_period"] is None
