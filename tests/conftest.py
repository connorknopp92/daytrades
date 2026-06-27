"""Shared fixtures and path setup for the test suite."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make the project root importable so `import src...` works under pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_ohlcv(close, high=None, low=None):
    """Build a minimal OHLCV frame from a close-price array."""
    close = np.asarray(close, dtype=float)
    n = len(close)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    high = close if high is None else np.asarray(high, dtype=float)
    low = close if low is None else np.asarray(low, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close,
         "volume": np.ones(n)},
        index=idx,
    )


@pytest.fixture
def flat_market():
    """Constant price — no strategy should make or lose money (minus fees)."""
    return make_ohlcv([100.0] * 50)


@pytest.fixture
def uptrend():
    """Steadily rising price."""
    return make_ohlcv(100.0 * (1.01 ** np.arange(60)))
