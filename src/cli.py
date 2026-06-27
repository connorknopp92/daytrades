"""Command-line interface: fetch / analyze / backtest.

Run with:  python -m src.cli <command> [options]
Everything here is simulation only — no real money is ever traded.
"""

from __future__ import annotations

import argparse
import sys

from . import service
from .config import load_config, project_path
from .data import fetch as data_fetch
from .analysis import report as report_mod
from .backtest import metrics as metrics_mod
from .strategies.base import available_strategies


def _load_symbol(cfg, symbol, timeframe=None, years=None, use_cache=True):
    df = data_fetch.get_ohlcv(cfg, symbol, timeframe=timeframe, years=years, use_cache=use_cache)
    return df, bool(df.attrs.get("synthetic", False))


def cmd_fetch(args, cfg) -> int:
    symbols = args.symbols or cfg["data"]["symbols"]
    timeframe = args.timeframe or cfg["data"]["timeframe"]
    years = args.years or cfg["data"]["years"]
    for symbol in symbols:
        print(f"Fetching {symbol} ({timeframe}, ~{years}y)...")
        df, synthetic = _load_symbol(cfg, symbol, timeframe, years, use_cache=not args.refresh)
        tag = " [SYNTHETIC]" if synthetic else ""
        print(f"  -> {len(df)} bars, {df.index[0].date()} → {df.index[-1].date()}{tag}")
    return 0


def cmd_analyze(args, cfg) -> int:
    out_dir = project_path(cfg, cfg["output_dir"])
    ppy = cfg["backtest"]["periods_per_year"]
    window = cfg["analysis"]["rolling_window"]
    df, synthetic = _load_symbol(cfg, args.symbol)
    path = report_mod.market_report(
        df, args.symbol, out_dir, rolling_window=window,
        periods_per_year=ppy, synthetic=synthetic,
    )
    print(f"Wrote market analysis report: {path}")
    if synthetic:
        print("  (used SYNTHETIC data — exchange was unreachable)")
    return 0


def cmd_backtest(args, cfg) -> int:
    out_dir = project_path(cfg, cfg["output_dir"])
    res = service.backtest_summary(
        cfg, args.symbol, args.strategy,
        leverage=args.leverage, fee=args.fee,
        slippage=args.slippage, funding=args.funding,
    )
    strat_m, bench_m = res["strat_metrics"], res["bench_metrics"]
    p = res["params"]
    strat_m.update({"_leverage": p["leverage"], "_fee": p["fee"], "_slippage": p["slippage"]})

    print(f"\n=== {args.strategy} on {args.symbol} "
          f"(leverage {p['leverage']}x, fee {p['fee']}, slippage {p['slippage']}) ===")
    if res["synthetic"]:
        print("  (SYNTHETIC data — exchange was unreachable)")
    print(metrics_mod.format_metrics(strat_m))
    print(f"\n--- benchmark: buy_and_hold ---")
    print(metrics_mod.format_metrics(bench_m))

    path = report_mod.backtest_report(
        args.symbol, args.strategy, strat_m, bench_m,
        res["strat_equity"], res["bench_equity"],
        out_dir, synthetic=res["synthetic"],
    )
    print(f"\nWrote backtest report: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="daytrades",
        description="Crypto research & backtesting sandbox (simulation only).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pf = sub.add_parser("fetch", help="download & cache OHLCV history")
    pf.add_argument("--symbols", nargs="+", help="e.g. BTC/USD ETH/USD")
    pf.add_argument("--timeframe", help="candle size, e.g. 1d, 1h")
    pf.add_argument("--years", type=int, help="years of history")
    pf.add_argument("--refresh", action="store_true", help="ignore cache, refetch")
    pf.set_defaults(func=cmd_fetch)

    pa = sub.add_parser("analyze", help="write a market-analysis report")
    pa.add_argument("--symbol", required=True, help="e.g. BTC/USD")
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("backtest", help="backtest a strategy vs buy-and-hold")
    pb.add_argument("--strategy", required=True, choices=available_strategies())
    pb.add_argument("--symbol", required=True, help="e.g. BTC/USD")
    pb.add_argument("--leverage", type=float, help="position notional / equity")
    pb.add_argument("--fee", type=float, help="taker fee per side (fraction)")
    pb.add_argument("--slippage", type=float, help="slippage per trade (fraction)")
    pb.add_argument("--funding", type=float, help="per-bar perp funding on held notional")
    pb.set_defaults(func=cmd_backtest)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config()
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
