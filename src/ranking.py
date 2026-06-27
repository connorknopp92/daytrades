"""Walk-forward strategy ranking — the honest 'best strategy per asset'.

Naively crowning whichever strategy has the best backtest is overfitting: with
enough strategies tried, a winner is guaranteed by chance and rarely repeats
(Sullivan-Timmermann-White 1999; Bailey & López de Prado 2014). So we rank
strategies on a TRAIN window and then report how each actually did on a later
HOLD-OUT window — surfacing the in-sample-vs-out-of-sample gap rather than hiding
it. Results are historical and after-cost; they are not predictions.
"""

from __future__ import annotations

import pandas as pd

from . import service
from .strategies.base import available_strategies, strategy_meta

METRICS = ("sharpe", "total_return", "calmar")


def _metrics_on(cfg, df, name, ppy, leverage, fee, slippage):
    """Backtest one strategy on a slice; return its metrics dict (or None if too short)."""
    if df is None or len(df) < 5:
        return None
    _, m = service.run_strategy(cfg, df, name, leverage=leverage, fee=fee,
                                slippage=slippage, funding=0.0, ppy=ppy)
    return m


def rank_strategies(cfg: dict, symbol: str, df: pd.DataFrame, view: str = "daily",
                    train_frac: float = 0.7, leverage: float = 1.0) -> dict:
    """Rank all strategies walk-forward on one prepared (resampled) OHLCV frame.

    Returns per-strategy train+holdout metrics, the trial count, and the
    train-window winner for each of Sharpe / total return / Calmar.
    """
    bt = cfg["backtest"]
    fee, slippage = bt["fee"], bt["slippage"]
    ppy = service.periods_per_year(cfg, symbol, view)
    synthetic = bool(df.attrs.get("synthetic", False))

    n = len(df)
    split = int(n * train_frac)
    train_df = df.iloc[:split]
    holdout_df = df.iloc[split:] if n - split >= 5 else None

    rows = []
    for name in available_strategies():
        meta = strategy_meta(name)
        train_m = _metrics_on(cfg, train_df, name, ppy, leverage, fee, slippage)
        hold_m = _metrics_on(cfg, holdout_df, name, ppy, leverage, fee, slippage)
        if train_m is None:
            continue
        rows.append({**meta, "train": train_m, "holdout": hold_m})

    # Winner per metric, ranked on the TRAIN window (holdout shown alongside).
    best_by = {}
    for metric in METRICS:
        ranked = sorted(rows, key=lambda r: r["train"].get(metric, float("-inf")),
                        reverse=True)
        best_by[metric] = ranked[0]["name"] if ranked else None

    return {
        "symbol": symbol,
        "view": view,
        "rows": rows,
        "best_by": best_by,
        "n_strategies_tried": len(rows),
        "train_period": (train_df.index[0], train_df.index[-1]) if len(train_df) else None,
        "holdout_period": (holdout_df.index[0], holdout_df.index[-1]) if holdout_df is not None else None,
        "synthetic": synthetic,
    }


def best_for_symbol(cfg: dict, symbol: str, df: pd.DataFrame, view: str = "daily",
                    years: int | None = 3, metric: str = "sharpe") -> dict:
    """Compact 'best historical strategy (walk-forward)' summary for the digest.

    Picks the train-window leader by ``metric`` and reports its hold-out result
    next to buy-and-hold's hold-out — so the email shows whether the winner
    actually held up out-of-sample.
    """
    prepared = service.slice_years(service.resample_ohlcv(df, view), years)
    ranking = rank_strategies(cfg, symbol, prepared, view=view)
    rows = {r["name"]: r for r in ranking["rows"]}
    winner_name = ranking["best_by"].get(metric)
    winner = rows.get(winner_name)
    bench = rows.get("buy_and_hold")

    def holdout_return(row):
        if row and row.get("holdout"):
            return row["holdout"].get("total_return")
        return None

    return {
        "symbol": symbol,
        "winner": winner_name,
        "winner_classic": winner["classic"] if winner else False,
        "train_metric": winner["train"].get(metric) if winner else None,
        "metric": metric,
        "winner_holdout_return": holdout_return(winner),
        "benchmark_holdout_return": holdout_return(bench),
        "n_strategies_tried": ranking["n_strategies_tried"],
        "synthetic": ranking["synthetic"],
    }
