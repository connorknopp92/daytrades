"""Parquet-backed cache for OHLCV data, keyed by exchange/symbol/timeframe."""

from __future__ import annotations

import os

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _safe_name(symbol: str) -> str:
    """Turn 'BTC/USD' into a filesystem-safe token like 'BTC-USD'."""
    return symbol.replace("/", "-").replace(":", "-")


def cache_path(cache_dir: str, exchange: str, symbol: str, timeframe: str) -> str:
    fname = f"{exchange}_{_safe_name(symbol)}_{timeframe}.parquet"
    return os.path.join(cache_dir, fname)


def save(df: pd.DataFrame, cache_dir: str, exchange: str, symbol: str, timeframe: str) -> str:
    """Persist an OHLCV DataFrame (DatetimeIndex) to parquet. Returns the path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, exchange, symbol, timeframe)
    df.to_parquet(path)
    return path


def load(cache_dir: str, exchange: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load a cached OHLCV DataFrame, or None if not present."""
    path = cache_path(cache_dir, exchange, symbol, timeframe)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    return df
