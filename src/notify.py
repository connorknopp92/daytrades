"""Daily market digest notifier.

Each day, computes a current-conditions signal for every tracked market and
emails a ranked digest: the top stocks/ETFs and top crypto that are currently
cheapest versus their OWN long-term trend (most below their 200-day average /
deepest in a dip) — i.e. historically sensible *accumulation* candidates.

This is explicitly NOT a prediction of which investments will go up, and not
financial advice. It ranks by present conditions vs. tested rules — a rule for
*where to look*, not what will win. Run via ``python -m src.notify``.

Safety: synthetic/fallback data (exchange unreachable) is ignored entirely, so a
data outage can't produce a misleading digest.
"""

from __future__ import annotations

import os

from . import accumulation as accum
from . import service
from .config import load_config
from .data import fetch as data_fetch

ACTIONABLE = {"FAVORABLE", "LEANING FAVORABLE"}
_EMOJI = {"FAVORABLE": "🟢", "LEANING FAVORABLE": "🟡", "NEUTRAL": "⚪"}
TOP_N = 3


def evaluate_symbol(cfg: dict, symbol: str) -> dict:
    """Load fresh data for ``symbol`` and compute its current signal."""
    df, synthetic = service.load_symbol(cfg, symbol, use_cache=False)
    sig = accum.current_signal(df)
    sig["symbol"] = symbol
    sig["synthetic"] = synthetic
    sig["is_stock"] = data_fetch.is_stock(symbol)
    return sig


def _score(r: dict) -> float:
    """Cheapness vs. the market's own trend: below-MA depth + dip depth.

    Higher = more 'on sale' relative to its history. NOT a return forecast.
    """
    score = 0.0
    if r.get("pct_vs_ma") is not None:
        score += -r["pct_vs_ma"]                       # below the 200-day avg
    score += 0.5 * (-r.get("drawdown_from_high", 0.0))  # off the recent high
    return score


def _line(r: dict) -> str:
    emoji = _EMOJI.get(r["verdict"], "")
    parts = [f"${r['price']:,.2f}" if r["price"] < 100 else f"${r['price']:,.0f}"]
    if r.get("pct_vs_ma") is not None:
        parts.append(f"{r['pct_vs_ma']*100:+.0f}% vs {r['ma_days']}-day avg")
    parts.append(f"{r['drawdown_from_high']*100:.0f}% off recent high")
    return f"{emoji} {r['symbol']} ({r['verdict']}) — " + ", ".join(parts)


def build_payload(results: list[dict], top_n: int = TOP_N):
    """Pure logic: build the daily digest (should_send, subject, body).

    Sends every day there is real data; ignores synthetic rows. Returns
    (should_send, subject, body).
    """
    real = [r for r in results if not r.get("synthetic")]
    should_send = len(real) > 0
    if not real:
        return False, "Daily market digest — no data", (
            "Could not reach the data sources for fresh prices today, so no "
            "digest was generated.")

    stocks = sorted((r for r in real if r.get("is_stock")), key=_score, reverse=True)
    crypto = sorted((r for r in real if not r.get("is_stock")), key=_score, reverse=True)
    favorable = [r for r in real if r["verdict"] in ACTIONABLE]

    if favorable:
        subject = f"📈 Daily picks — {len(favorable)} market(s) look favorable to accumulate"
    else:
        subject = "📈 Daily market digest — nothing notably cheap today"

    lines = [
        "Daily market digest",
        "Ranked by how cheap each market is vs. its OWN long-term trend",
        "(most below its 200-day average / deepest dip first).",
        "",
        "📊 TOP STOCKS / ETFs:",
    ]
    lines += [f"  {i}. {_line(r)}" for i, r in enumerate(stocks[:top_n], 1)] or ["  (none)"]
    lines += ["", "🪙 TOP CRYPTO:"]
    lines += [f"  {i}. {_line(r)}" for i, r in enumerate(crypto[:top_n], 1)] or ["  (none)"]
    lines += [
        "",
        "─" * 40,
        "⚠️ This is NOT a prediction of which investments will go up, and NOT "
        "financial advice. It ranks markets only by current price vs. their own "
        "history — a rule for where to look, not what will win. No one can predict "
        "the best stock to buy. Most active trading loses to patient, diversified "
        "holding. Simulation / education only.",
    ]
    return should_send, subject, "\n".join(lines)


def _emit_github_output(should_send: bool, subject: str, body: str) -> None:
    """Write step outputs for the workflow (or print if running locally)."""
    out_path = os.environ.get("GITHUB_OUTPUT")
    if not out_path:
        print(f"should_send={should_send}\nsubject={subject}\n\n{body}")
        return
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write(f"should_send={'true' if should_send else 'false'}\n")
        fh.write(f"subject={subject}\n")
        fh.write("body<<__BODY_EOF__\n")
        fh.write(body + "\n")
        fh.write("__BODY_EOF__\n")


def main() -> int:
    cfg = load_config()
    symbols = cfg["data"]["symbols"] + cfg["data"].get("stocks", [])
    results = []
    for symbol in symbols:
        try:
            results.append(evaluate_symbol(cfg, symbol))
        except Exception as exc:  # don't let one symbol break the run
            print(f"[warn] {symbol}: {exc}")
    should_send, subject, body = build_payload(results)
    _emit_github_output(should_send, subject, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
