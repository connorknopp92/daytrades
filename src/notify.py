"""Daily accumulation-signal notifier.

Computes today's buy signal for each configured market and, when one is in an
actionable window (FAVORABLE / LEANING FAVORABLE), emits an email subject + body
for the GitHub Actions workflow to send. Run via ``python -m src.notify``.

Safety: never alerts on synthetic/fallback data (if the exchange was
unreachable), so a data outage can't masquerade as a "buy now" signal.
"""

from __future__ import annotations

import os

from . import accumulation as accum
from . import service
from .config import load_config

ACTIONABLE = {"FAVORABLE", "LEANING FAVORABLE"}
_EMOJI = {"FAVORABLE": "🟢", "LEANING FAVORABLE": "🟡", "NEUTRAL": "⚪"}


def evaluate_symbol(cfg: dict, symbol: str) -> dict:
    """Load live data for ``symbol`` and compute its current signal."""
    df, synthetic = service.load_symbol(cfg, symbol, use_cache=False)
    sig = accum.current_signal(df)
    sig["symbol"] = symbol
    sig["synthetic"] = synthetic
    return sig


def _line(r: dict) -> str:
    emoji = _EMOJI.get(r["verdict"], "")
    parts = [f"${r['price']:,.0f}"]
    if r.get("pct_vs_ma") is not None:
        parts.append(f"{r['pct_vs_ma']*100:+.0f}% vs {r['ma_days']}-day avg")
    parts.append(f"{r['drawdown_from_high']*100:.0f}% off recent high")
    return f"{emoji} {r['symbol']}: {r['verdict']} — " + ", ".join(parts)


def build_payload(results: list[dict], actionable: set[str] = ACTIONABLE):
    """Pure logic: decide whether to send and build (subject, body).

    Synthetic results are ignored entirely so a data outage never triggers a
    misleading alert. Returns (should_send, subject, body).
    """
    real = [r for r in results if not r.get("synthetic")]
    hits = [r for r in real if r["verdict"] in actionable]
    should_send = len(hits) > 0

    if hits:
        names = ", ".join(h["symbol"].split("/")[0] for h in hits)
        subject = f"🟢 Crypto buy signal — {names} favorable to accumulate"
    else:
        subject = "Crypto check — no actionable buy signal today"

    body_lines = ["Daily accumulation check", ""]
    body_lines += [_line(r) for r in real] or ["(no live market data available)"]
    if not real:
        body_lines.append("Could not reach the exchange for fresh prices — no signal sent.")
    body_lines += [
        "",
        "What this means: a FAVORABLE/LEANING reading is when price is below its "
        "long-term trend and/or in a dip — historically a steadier window to add to "
        "a long-term hold.",
        "",
        "⚠️ This describes current conditions vs. tested rules. It is NOT a prediction "
        "and not financial advice. No one can reliably time the bottom. Simulation/education only.",
    ]
    return should_send, subject, "\n".join(body_lines)


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
