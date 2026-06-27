"""Tests for stock support: routing, Yahoo parsing, and annualization."""

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src import service
from src.data import fetch


def test_is_stock_routing():
    assert fetch.is_stock("AAPL") is True
    assert fetch.is_stock("SPY") is True
    assert fetch.is_stock("BTC/USD") is False
    assert fetch.is_stock("ETH/USD") is False


def test_periods_per_year_is_asset_aware():
    cfg = load_config()
    assert service.periods_per_year(cfg, "AAPL") == 252
    assert service.periods_per_year(cfg, "BTC/USD") == cfg["backtest"]["periods_per_year"]


def _yahoo_payload():
    return {
        "chart": {
            "error": None,
            "result": [{
                "timestamp": [1577836800, 1577923200, 1578009600],
                "indicators": {"quote": [{
                    "open": [100.0, 101.0, 102.0],
                    "high": [101.0, 102.0, 103.0],
                    "low": [99.0, 100.0, 101.0],
                    "close": [100.5, 101.5, 102.5],
                    "volume": [1000, 1100, 1200],
                }]},
            }],
        }
    }


def test_parse_yahoo_json_shapes():
    df = fetch._parse_yahoo_json(_yahoo_payload())
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 3
    assert df.attrs.get("synthetic") is False
    np.testing.assert_allclose(df["close"].values, [100.5, 101.5, 102.5])
    assert df.index.tz is not None  # UTC-aware index


def test_parse_yahoo_json_drops_nan_rows():
    payload = _yahoo_payload()
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][1] = None
    df = fetch._parse_yahoo_json(payload)
    assert len(df) == 2  # the NaN-close row is dropped


def test_parse_yahoo_json_error_raises():
    with pytest.raises(RuntimeError):
        fetch._parse_yahoo_json({"chart": {"error": "Not Found", "result": None}})
