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


def load_symbol(cfg: dict, symbol: str, timeframe: str | None = None,
                years: int | None = None, use_cache: bool = True):
    """Fetch OHLCV for a symbol. Returns (df, synthetic_flag)."""
    df = data_fetch.get_ohlcv(cfg, symbol, timeframe=timeframe, years=years,
                              use_cache=use_cache)
    return df, bool(df.attrs.get("synthetic", False))


def market_summary(cfg: dict, symbol: str, df: pd.DataFrame | None = None) -> dict:
    """Headline stats and chart-ready series for a market.

    Pass an already-loaded ``df`` to avoid refetching; otherwise it is loaded.
    """
    ppy = cfg["backtest"]["periods_per_year"]
    window = cfg["analysis"]["rolling_window"]
    synthetic = False
    if df is None:
        df, synthetic = load_symbol(cfg, symbol)
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
                 leverage: float, fee: float, slippage: float, funding: float):
    """Run a single strategy over ``df``. Returns (BacktestResult, metrics dict)."""
    bt = cfg["backtest"]
    ppy = bt["periods_per_year"]
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
                     df: pd.DataFrame | None = None) -> dict:
    """Backtest ``strategy_name`` and the buy-and-hold benchmark on one symbol.

    Any of leverage/fee/slippage/funding left as None falls back to config.
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
    else:
        synthetic = bool(df.attrs.get("synthetic", False))

    strat_result, strat_m = run_strategy(cfg, df, strategy_name, leverage, fee, slippage, funding)
    bench_result, bench_m = run_strategy(cfg, df, "buy_and_hold", leverage, fee, slippage, funding)

    return {
        "strat_metrics": strat_m,
        "bench_metrics": bench_m,
        "strat_equity": strat_result.equity_curve,
        "bench_equity": bench_result.equity_curve,
        "synthetic": synthetic,
        "params": {"leverage": leverage, "fee": fee, "slippage": slippage, "funding": funding},
    }
