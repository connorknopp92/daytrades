"""Download OHLCV history via ccxt (Coinbase) with a local parquet cache.

Coinbase serves *spot* price history. We use it as the underlying for our
simulated-futures backtester. Network access is optional: if the exchange is
unreachable, a clearly-labelled deterministic synthetic series is returned so
the rest of the tool remains usable offline.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from . import cache
from .cache import OHLCV_COLUMNS

# ccxt timeframe -> milliseconds, for pagination math.
_TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def timeframe_ms(timeframe: str) -> int:
    if timeframe not in _TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return _TIMEFRAME_MS[timeframe]


def _now_ms() -> int:
    return int(time.time() * 1000)


def _configure_proxy(exchange) -> None:
    """Make ccxt honor the environment proxy + CA bundle.

    ccxt creates its requests Session with ``trust_env = False``, so it ignores
    the ``HTTPS_PROXY`` and ``REQUESTS_CA_BUNDLE`` variables that plain
    ``requests`` (and curl) pick up automatically. In this remote environment
    outbound HTTPS goes through an intercepting proxy, so without this the very
    first market-loading call fails TLS verification. Re-enabling trust_env and
    pointing the session at the proxy CA bundle restores normal access.
    """
    session = getattr(exchange, "session", None)
    if session is None:
        return
    session.trust_env = True
    ca = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE")
    if ca and os.path.exists(ca):
        session.verify = ca
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}


def _fetch_from_exchange(exchange_id: str, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Page through an exchange's OHLCV endpoint from ``since_ms`` to now."""
    import ccxt  # imported lazily so offline use / tests don't require it

    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    _configure_proxy(exchange)
    step = timeframe_ms(timeframe)
    limit = 300  # Coinbase caps candles per request
    all_rows: list[list] = []
    cursor = since_ms
    now = _now_ms()

    while cursor < now:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        next_cursor = last_ts + step
        if next_cursor <= cursor:  # no forward progress -> stop
            break
        cursor = next_cursor
        if len(batch) < limit:
            # Reached the most recent candle.
            if cursor >= now:
                break

    if not all_rows:
        raise RuntimeError(f"No data returned for {symbol} from {exchange_id}")

    df = pd.DataFrame(all_rows, columns=["ts", *OHLCV_COLUMNS])
    df = df.drop_duplicates(subset="ts").sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop(columns=["ts"])
    df.attrs["synthetic"] = False
    return df


def synthetic_ohlcv(symbol: str, timeframe: str, years: int, seed: int = 42) -> pd.DataFrame:
    """Deterministic geometric-random-walk series for offline use / tests.

    Clearly labelled via ``df.attrs['synthetic'] = True``. This is NOT real
    market data and must never be presented as such.
    """
    step = timeframe_ms(timeframe)
    n = max(int(years * 365 * 24 * 60 * 60 * 1000 / step), 50)
    rng = np.random.default_rng(seed + (abs(hash(symbol)) % 1000))

    # Mild upward drift with crypto-like volatility.
    daily_vol = 0.04
    drift = 0.0005
    rets = rng.normal(drift, daily_vol, size=n)
    start_price = 20000.0 if symbol.upper().startswith("BTC") else 1500.0
    close = start_price * np.exp(np.cumsum(rets))

    high = close * (1 + np.abs(rng.normal(0, daily_vol / 2, size=n)))
    low = close * (1 - np.abs(rng.normal(0, daily_vol / 2, size=n)))
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    volume = np.abs(rng.normal(1000, 300, size=n))

    end = pd.Timestamp.now(tz="UTC").floor("D")
    index = pd.date_range(end=end, periods=n, freq=pd.Timedelta(milliseconds=step), tz="UTC")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )
    df.attrs["synthetic"] = True
    return df


def is_stock(symbol: str) -> bool:
    """Crypto pairs look like 'BTC/USD'; bare tickers like 'AAPL' are stocks."""
    return "/" not in symbol


_YAHOO_INTERVALS = {"1d": "1d", "1wk": "1wk", "1mo": "1mo"}


def _parse_yahoo_json(data: dict) -> pd.DataFrame:
    """Turn a Yahoo chart-API response into an OHLCV DataFrame."""
    chart = data.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(f"Yahoo error: {chart['error']}")
    result = (chart.get("result") or [None])[0]
    if not result or "timestamp" not in result:
        raise RuntimeError("Yahoo returned no data for this ticker")

    ts = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    df = pd.DataFrame(
        {
            "open": quote["open"],
            "high": quote["high"],
            "low": quote["low"],
            "close": quote["close"],
            "volume": quote["volume"],
        },
        index=pd.to_datetime(ts, unit="s", utc=True),
    ).dropna(subset=["close"])
    df.attrs["synthetic"] = False
    return df


def _fetch_from_yahoo(ticker: str, timeframe: str, years: int) -> pd.DataFrame:
    """Fetch daily stock/ETF history from the keyless Yahoo Finance chart API.

    Plain ``requests`` honors the env proxy + CA bundle automatically (unlike
    ccxt), so no special session setup is needed here.
    """
    import requests  # lazy import so offline use / tests don't require it

    interval = _YAHOO_INTERVALS.get(timeframe, "1d")
    period2 = _now_ms() // 1000
    period1 = period2 - int(years * 365.25 * 86400)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; daytrades-sandbox/1.0)"}
    resp = requests.get(
        url, params={"period1": period1, "period2": period2, "interval": interval},
        headers=headers, timeout=30,
    )
    resp.raise_for_status()
    df = _parse_yahoo_json(resp.json())
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker} from Yahoo")
    return df


def get_ohlcv(
    cfg: dict,
    symbol: str,
    timeframe: str | None = None,
    years: int | None = None,
    use_cache: bool = True,
    allow_synthetic: bool = True,
) -> pd.DataFrame:
    """Return OHLCV history for ``symbol``: cache -> source -> synthetic.

    Routes crypto pairs (e.g. ``BTC/USD``) to the exchange via ccxt and bare
    stock tickers (e.g. ``AAPL``) to Yahoo Finance. Always returns a DataFrame
    indexed by UTC timestamp with columns open/high/low/close/volume.
    """
    data_cfg = cfg["data"]
    timeframe = timeframe or data_cfg["timeframe"]
    years = years or data_cfg["years"]
    cache_dir = data_cfg["cache_dir"]
    stock = is_stock(symbol)
    source = "yahoo" if stock else data_cfg["exchange"]

    if use_cache:
        cached = cache.load(cache_dir, source, symbol, timeframe)
        if cached is not None and not cached.empty:
            return cached

    try:
        if stock:
            df = _fetch_from_yahoo(symbol, timeframe, years)
        else:
            since_ms = _now_ms() - int(years * 365.25 * timeframe_ms("1d"))
            df = _fetch_from_exchange(source, symbol, timeframe, since_ms)
        cache.save(df, cache_dir, source, symbol, timeframe)
        return df
    except Exception as exc:  # network blocked, geo-restricted, bad ticker, etc.
        if not allow_synthetic:
            raise
        print(
            f"[warn] Could not fetch real data for {symbol} ({exc}). "
            f"Falling back to SYNTHETIC data (not real market history)."
        )
        return synthetic_ohlcv(symbol, timeframe, years)
