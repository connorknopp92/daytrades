"""Paper-trading sandbox — a fake-money forward test that persists over time.

Each run advances a simulated portfolio one step on the latest prices. For every
symbol it keeps two sub-accounts:

- **agent**: follows the *walk-forward-best* strategy's current signal (the
  "agent learning from the backtests"), rebalanced only when the signal flips;
- **hold**: a buy-and-hold shadow bought once at inception.

Comparing the two going forward is the only honest test of whether the adaptive
agent actually beats just holding — true out-of-sample, no hindsight. It is
**simulation only**: no real money, no exchange, no orders. State is a JSON file
committed back to the repo so paper P&L accumulates across daily runs.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

from . import ranking, service
from .config import load_config, project_path
from .strategies.base import get_strategy


def load_state(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_state(path: str, state: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, sort_keys=True)


def _target_signal(cfg: dict, symbol: str, df, metric: str):
    """Pick the walk-forward-best strategy and return (name, today's signal)."""
    best = ranking.best_for_symbol(cfg, symbol, df, metric=metric)
    name = best["winner"] or "buy_and_hold"
    sig = float(get_strategy(name).generate_signals(df).iloc[-1])
    return name, sig


def _advance_symbol(acct, price, name, sig, subcap, fee):
    """Advance one symbol's agent + hold sub-accounts to the latest price."""
    if acct is None:  # first time we see this symbol — open both books
        a_units = (sig * subcap) / price
        acct = {
            "agent": {"units": a_units, "cash": subcap - a_units * price,
                      "sig": sig, "strategy": name},
            "hold": {"units": subcap / price, "cash": 0.0},
        }
    else:
        a = acct["agent"]
        equity = a["cash"] + a["units"] * price
        if sig != a["sig"]:  # signal flipped -> rebalance (pay a fee on the change)
            target_units = (sig * equity) / price
            delta = target_units - a["units"]
            a["cash"] -= delta * price + fee * abs(delta) * price
            a["units"] = target_units
            a["sig"] = sig
        a["strategy"] = name

    a, h = acct["agent"], acct["hold"]
    acct["price"] = price
    acct["agent_equity"] = a["cash"] + a["units"] * price
    acct["hold_equity"] = h["cash"] + h["units"] * price
    return acct


def resolve_symbols(cfg: dict) -> list[str]:
    """Resolve the paper watchlist; 'all' = every crypto + stock."""
    sym = cfg["paper"]["symbols"]
    if sym == "all":
        return list(cfg["data"]["symbols"]) + list(cfg["data"].get("stocks", []))
    return list(sym)


def step(cfg: dict, df_provider=None, today: str | None = None) -> dict:
    """Advance the paper portfolio one step on the latest data; return new state."""
    pcfg = cfg["paper"]
    symbols = resolve_symbols(cfg)
    fee = cfg["backtest"]["fee"]
    metric = pcfg.get("select_metric", "sharpe")
    path = project_path(cfg, pcfg["state_file"])
    today = today or _dt.date.today().isoformat()

    # Per-market stake ($50 each) if set; else split a fixed initial capital.
    stake = pcfg.get("stake")
    initial = float(stake) * len(symbols) if stake else float(pcfg["initial_capital"])

    state = load_state(path)
    if state is None:
        state = {"created": today, "initial_capital": initial,
                 "symbols": symbols, "accounts": {}, "history": []}
    subcap = state["initial_capital"] / len(symbols)

    for symbol in symbols:
        try:
            if df_provider is not None:
                df = df_provider(symbol)
            else:
                df, _ = service.load_symbol(cfg, symbol, use_cache=False)
            price = float(df["close"].iloc[-1])
            name, sig = _target_signal(cfg, symbol, df, metric)
            state["accounts"][symbol] = _advance_symbol(
                state["accounts"].get(symbol), price, name, sig, subcap, fee)
        except Exception as exc:  # one bad symbol shouldn't break the sandbox
            print(f"[warn] paper {symbol}: {exc}")

    agent_total = sum(a["agent_equity"] for a in state["accounts"].values())
    hold_total = sum(a["hold_equity"] for a in state["accounts"].values())
    state["history"].append({"date": today, "agent_total": agent_total,
                             "hold_total": hold_total})
    state["last_run"] = today
    save_state(path, state)
    return state


def summary_lines(state: dict) -> list[str]:
    """Plain-text paper-sandbox section for the daily email."""
    if not state or not state.get("accounts"):
        return []
    init = state["initial_capital"]
    agent_total = sum(a["agent_equity"] for a in state["accounts"].values())
    hold_total = sum(a["hold_equity"] for a in state["accounts"].values())

    def pct(v):
        base = init / len(state["accounts"])
        return f"{(v / base - 1) * 100:+.1f}%"

    stake = init / len(state["accounts"])
    lines = [
        f"🧪 PAPER SANDBOX (fake money, ${stake:,.0f} per market, started {state['created']}):",
        f"   Agent ${agent_total:,.0f} ({(agent_total/init-1)*100:+.1f}%) "
        f"vs Buy & Hold ${hold_total:,.0f} ({(hold_total/init-1)*100:+.1f}%)",
        "   Leaderboard — $50 in each, best -> worst (buy & hold):",
    ]
    ranked = sorted(state["accounts"].items(),
                    key=lambda kv: kv[1]["hold_equity"], reverse=True)
    for sym, a in ranked:
        lines.append(f"     {sym}: ${a['hold_equity']:,.2f} ({pct(a['hold_equity'])})")
    return lines


def main() -> int:
    cfg = load_config()
    state = step(cfg)
    print("\n".join(summary_lines(state)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
