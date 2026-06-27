"""UI-agnostic service layer shared by the CLI and the web dashboard.

Returns plain Python/pandas data (no printing, no plotting, no Streamlit) so it
can be reused by `src/cli.py`, `streamlit_app.py`, and tests alike.
"""

from __future__ import annotations

import pandas as pd

from .analysis import market_stats
from .backtest import metrics as metrics_mod
from .backtest.engine import run_backtest
from .data import fetch as data_fetch
from .strategies.base import get_strategy

# Stocks trade ~252 sessions/year; crypto trades every day. Using the right
# figure keeps annualized vol, Sharpe/Sortino and CAGR honest per asset class.
STOCK_PERIODS_PER_YEAR = 252

# "View" = candle aggregation for display/backtest. Coinbase only serves daily
# candles, so weekly/monthly are produced by resampling daily data — works
# uniformly for crypto and stocks.
_RESAMPLE_RULE = {"daily": None, "weekly": "W", "monthly": "ME"}
_VIEW_PPY = {"weekly": 52, "monthly": 12}
_VIEW_WINDOW = {"daily": 30, "weekly": 26, "monthly": 12}


def periods_per_year(cfg: dict, symbol: str, view: str = "daily") -> int:
    """Annualization factor for ``symbol`` at a given view.

    Weekly→52, monthly→12; daily→252 for stocks, config value (365) for crypto.
    """
    if view in _VIEW_PPY:
        return _VIEW_PPY[view]
    if data_fetch.is_stock(symbol):
        return STOCK_PERIODS_PER_YEAR
    return cfg["backtest"]["periods_per_year"]


def resample_ohlcv(df: pd.DataFrame, view: str = "daily") -> pd.DataFrame:
    """Aggregate a daily OHLCV frame to weekly/monthly candles (daily = no-op)."""
    rule = _RESAMPLE_RULE.get(view)
    if rule is None:
        return df
    agg = (df.resample(rule)
             .agg({"open": "first", "high": "max", "low": "min",
                   "close": "last", "volume": "sum"})
             .dropna(subset=["close"]))
    agg.attrs["synthetic"] = bool(df.attrs.get("synthetic", False))
    return agg


def slice_years(df: pd.DataFrame, years: float | None) -> pd.DataFrame:
    """Keep only the most recent ``years`` of rows (None/0 = keep all)."""
    if not years:
        return df
    cutoff = df.index[-1] - pd.Timedelta(days=int(365.25 * years))
    out = df[df.index >= cutoff]
    out.attrs["synthetic"] = bool(df.attrs.get("synthetic", False))
    return out


def load_symbol(cfg: dict, symbol: str, timeframe: str | None = None,
                years: int | None = None, use_cache: bool = True):
    """Fetch OHLCV for a symbol. Returns (df, synthetic_flag)."""
    df = data_fetch.get_ohlcv(cfg, symbol, timeframe=timeframe, years=years,
                              use_cache=use_cache)
    return df, bool(df.attrs.get("synthetic", False))


def market_summary(cfg: dict, symbol: str, df: pd.DataFrame | None = None,
                   view: str = "daily", window: int | None = None) -> dict:
    """Headline stats and chart-ready series for a market at a given view.

    Pass an already-prepared ``df`` (resampled/sliced) to avoid refetching;
    otherwise daily data is loaded and resampled to ``view``.
    """
    ppy = periods_per_year(cfg, symbol, view)
    if window is None:
        window = _VIEW_WINDOW.get(view, cfg["analysis"]["rolling_window"])
    synthetic = False
    if df is None:
        df, synthetic = load_symbol(cfg, symbol)
        df = resample_ohlcv(df, view)
    else:
        synthetic = bool(df.attrs.get("synthetic", False))

    return {
        "stats": market_stats.summary_stats(df, ppy),
        "price": df["close"],
        "drawdown": market_stats.drawdown_series(df),
        "rolling_vol": market_stats.rolling_volatility(df, window, ppy),
        "synthetic": synthetic,
    }


def run_strategy(cfg: dict, df: pd.DataFrame, strategy_name: str,
                 leverage: float, fee: float, slippage: float, funding: float,
                 ppy: int | None = None):
    """Run a single strategy over ``df``. Returns (BacktestResult, metrics dict)."""
    bt = cfg["backtest"]
    ppy = bt["periods_per_year"] if ppy is None else ppy
    strat = get_strategy(strategy_name)
    signals = strat.generate_signals(df)
    result = run_backtest(
        df, signals,
        initial_capital=bt["initial_capital"], leverage=leverage,
        fee=fee, slippage=slippage, funding_rate=funding,
        maintenance_margin=bt["maintenance_margin"],
    )
    m = metrics_mod.compute_all(
        result.equity_curve, ppy, n_trades=result.n_trades,
        exposure=result.exposure, liquidated=result.liquidated,
    )
    return result, m


def backtest_summary(cfg: dict, symbol: str, strategy_name: str,
                     leverage: float | None = None, fee: float | None = None,
                     slippage: float | None = None, funding: float | None = None,
                     df: pd.DataFrame | None = None, view: str = "daily") -> dict:
    """Backtest ``strategy_name`` and the buy-and-hold benchmark on one symbol.

    Any of leverage/fee/slippage/funding left as None falls back to config.
    ``view`` selects daily/weekly/monthly candles for the backtest.
    Returns metrics + equity curves for both, plus the resolved parameters.
    """
    bt = cfg["backtest"]
    leverage = bt["leverage"] if leverage is None else leverage
    fee = bt["fee"] if fee is None else fee
    slippage = bt["slippage"] if slippage is None else slippage
    funding = bt["funding_rate"] if funding is None else funding

    synthetic = False
    if df is None:
        df, synthetic = load_symbol(cfg, symbol)
        df = resample_ohlcv(df, view)
    else:
        synthetic = bool(df.attrs.get("synthetic", False))

    ppy = periods_per_year(cfg, symbol, view)
    strat_result, strat_m = run_strategy(cfg, df, strategy_name, leverage, fee, slippage, funding, ppy)
    bench_result, bench_m = run_strategy(cfg, df, "buy_and_hold", leverage, fee, slippage, funding, ppy)

    return {
        "strat_metrics": strat_m,
        "bench_metrics": bench_m,
        "strat_equity": strat_result.equity_curve,
        "bench_equity": bench_result.equity_curve,
        "synthetic": synthetic,
        "params": {"leverage": leverage, "fee": fee, "slippage": slippage, "funding": funding},
    }
