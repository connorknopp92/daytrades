"""Cache round-trip and synthetic data generation (no network)."""

import numpy as np
import pandas as pd

from src.data import cache
from src.data.fetch import synthetic_ohlcv
from tests.conftest import make_ohlcv


def test_cache_roundtrip(tmp_path):
    df = make_ohlcv(100.0 * (1.01 ** np.arange(30)))
    path = cache.save(df, str(tmp_path), "coinbase", "BTC/USD", "1d")
    assert path.endswith(".parquet")

    loaded = cache.load(str(tmp_path), "coinbase", "BTC/USD", "1d")
    assert loaded is not None
    assert list(loaded.columns) == list(df.columns)
    assert len(loaded) == len(df)
    np.testing.assert_allclose(loaded["close"].values, df["close"].values)


def test_cache_miss_returns_none(tmp_path):
    assert cache.load(str(tmp_path), "coinbase", "NOPE/USD", "1d") is None


def test_synthetic_is_deterministic_and_labelled():
    a = synthetic_ohlcv("BTC/USD", "1d", years=1, seed=7)
    b = synthetic_ohlcv("BTC/USD", "1d", years=1, seed=7)
    assert a.attrs.get("synthetic") is True
    np.testing.assert_allclose(a["close"].values, b["close"].values)
    # All OHLCV columns present and positive.
    for col in ["open", "high", "low", "close", "volume"]:
        assert col in a.columns
        assert (a[col] > 0).all()
