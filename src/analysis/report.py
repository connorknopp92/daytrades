"""Generate markdown + PNG reports for market analysis and backtests."""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
import pandas as pd

from . import market_stats


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def market_report(df: pd.DataFrame, symbol: str, output_dir: str,
                  rolling_window: int = 30, periods_per_year: int = 365,
                  synthetic: bool = False) -> str:
    """Write a market-analysis report (markdown + plots). Returns the md path."""
    _ensure_dir(output_dir)
    safe = symbol.replace("/", "-")
    stats = market_stats.summary_stats(df, periods_per_year)

    # Price + drawdown plot.
    dd = market_stats.drawdown_series(df)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot(df.index, df["close"], color="tab:blue")
    ax1.set_title(f"{symbol} price")
    ax1.set_ylabel("Price (USD)")
    ax1.grid(alpha=0.3)
    ax2.fill_between(dd.index, dd.values * 100, 0, color="tab:red", alpha=0.4)
    ax2.set_ylabel("Drawdown %")
    ax2.grid(alpha=0.3)
    price_png = os.path.join(output_dir, f"{safe}_price_drawdown.png")
    fig.tight_layout()
    fig.savefig(price_png, dpi=110)
    plt.close(fig)

    # Rolling volatility plot.
    rvol = market_stats.rolling_volatility(df, rolling_window, periods_per_year)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(rvol.index, rvol.values * 100, color="tab:purple")
    ax.set_title(f"{symbol} rolling annualized volatility ({rolling_window}-bar)")
    ax.set_ylabel("Volatility %")
    ax.grid(alpha=0.3)
    vol_png = os.path.join(output_dir, f"{safe}_volatility.png")
    fig.tight_layout()
    fig.savefig(vol_png, dpi=110)
    plt.close(fig)

    md_path = os.path.join(output_dir, f"{safe}_analysis.md")
    warn = ("\n> ⚠️ **SYNTHETIC DATA** — the exchange was unreachable, so this "
            "report uses a simulated price series, NOT real market history.\n"
            if synthetic else "")
    lines = [
        f"# Market Analysis — {symbol}",
        warn,
        f"- Period: **{stats['start'].date()} → {stats['end'].date()}** "
        f"({stats['n_bars']} bars)",
        f"- Price: **${stats['start_price']:,.2f} → ${stats['end_price']:,.2f}**",
        f"- Total return: **{_pct(stats['total_return'])}**",
        f"- CAGR: **{_pct(stats['cagr'])}**",
        f"- Annualized volatility: **{_pct(stats['ann_volatility'])}**",
        f"- Best / worst day: **{_pct(stats['best_day'])} / {_pct(stats['worst_day'])}**",
        f"- Max drawdown: **{_pct(stats['max_drawdown'])}**",
        f"- Up days: **{_pct(stats['pct_up_days'])}**",
        "",
        "## What this tells you",
        f"This market historically swung ~{_pct(stats['ann_volatility'])} per year and "
        f"suffered a peak-to-trough drop of {_pct(stats['max_drawdown'])}. A drawdown of "
        "that size would liquidate any meaningfully leveraged position. High volatility is "
        "opportunity and danger in equal measure — sizing and leverage decide which.",
        "",
        "## Charts",
        f"![price and drawdown]({os.path.basename(price_png)})",
        f"![rolling volatility]({os.path.basename(vol_png)})",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return md_path


def backtest_report(symbol: str, strategy_name: str, strat_metrics: dict,
                    bench_metrics: dict, strat_equity: pd.Series,
                    bench_equity: pd.Series, output_dir: str,
                    synthetic: bool = False) -> str:
    """Write a backtest report comparing strategy vs buy-and-hold."""
    _ensure_dir(output_dir)
    safe = f"{symbol.replace('/', '-')}_{strategy_name}"

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(strat_equity.index, strat_equity.values, label=strategy_name, color="tab:blue")
    ax.plot(bench_equity.index, bench_equity.values, label="buy_and_hold",
            color="tab:gray", linestyle="--")
    ax.set_title(f"{symbol}: {strategy_name} vs buy-and-hold")
    ax.set_ylabel("Equity (USD)")
    ax.legend()
    ax.grid(alpha=0.3)
    eq_png = os.path.join(output_dir, f"{safe}_equity.png")
    fig.tight_layout()
    fig.savefig(eq_png, dpi=110)
    plt.close(fig)

    def row(label, key, pct=True):
        a, b = strat_metrics[key], bench_metrics[key]
        fmt = (lambda x: f"{x * 100:,.2f}%") if pct else (lambda x: f"{x:.2f}")
        return f"| {label} | {fmt(a)} | {fmt(b)} |"

    md_path = os.path.join(output_dir, f"{safe}_backtest.md")
    warn = ("\n> ⚠️ **SYNTHETIC DATA** — results are from a simulated price "
            "series, NOT real market history.\n" if synthetic else "")
    lines = [
        f"# Backtest — {strategy_name} on {symbol}",
        warn,
        f"Leverage **{strat_metrics.get('_leverage', '?')}x**, "
        f"fee **{strat_metrics.get('_fee', '?')}**, "
        f"slippage **{strat_metrics.get('_slippage', '?')}**.",
        "",
        "| Metric | Strategy | Buy & Hold |",
        "| --- | --- | --- |",
        row("Total return", "total_return"),
        row("CAGR", "cagr"),
        row("Ann. volatility", "annualized_volatility"),
        row("Sharpe", "sharpe", pct=False),
        row("Sortino", "sortino", pct=False),
        row("Max drawdown", "max_drawdown"),
        row("Calmar", "calmar", pct=False),
        row("Win rate", "win_rate"),
        row("Profit factor", "profit_factor", pct=False),
        f"| # trades | {strat_metrics['n_trades']} | {bench_metrics['n_trades']} |",
        f"| Final equity | ${strat_metrics['final_equity']:,.2f} | "
        f"${bench_metrics['final_equity']:,.2f} |",
        "",
    ]
    if strat_metrics.get("liquidated"):
        lines.append("> 💥 **The strategy was LIQUIDATED** — leverage wiped the account. "
                     "This is the central risk of trading futures.\n")
    beat = strat_metrics["total_return"] > bench_metrics["total_return"]
    lines.append(
        f"**Verdict:** the strategy {'beat' if beat else 'did NOT beat'} simply "
        "buying and holding over this period. Most don't, once fees, slippage and "
        "funding are paid — which is the whole lesson.")
    lines += ["", "## Equity curve", f"![equity curve]({os.path.basename(eq_png)})", ""]

    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return md_path
