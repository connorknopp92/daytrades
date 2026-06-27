"""Bar-by-bar backtest engine with simulated futures mechanics.

The model is deliberately explicit rather than clever, so the costs that matter
are visible: trading fees, slippage, leverage, funding, and liquidation. It
tracks ``cash`` and signed ``units`` of the asset; account equity is always
``cash + units * price``. Leverage simply lets the position notional exceed
equity (cash goes negative, i.e. borrowed), which is exactly what makes
liquidation possible.

Convention (no lookahead): the target position for bar ``t`` is derived from the
signal at bar ``t`` and executed at that bar's close; profit/loss on it accrues
over the following bar, where funding and liquidation are also assessed.

A position is opened/closed only when the signal *changes*. While the signal
holds steady the position simply rides (its effective leverage drifts with
price, exactly as a real held futures position does). This keeps fees realistic
-- buy-and-hold makes a single trade rather than churning every bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .portfolio import BacktestResult

_EPS = 1e-12


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10000.0,
    leverage: float = 1.0,
    fee: float = 0.001,
    slippage: float = 0.0005,
    funding_rate: float = 0.0,
    maintenance_margin: float = 0.005,
) -> BacktestResult:
    """Simulate a strategy over ``df`` given target positions ``signals``.

    Parameters
    ----------
    df : OHLCV DataFrame indexed by timestamp (needs high/low/close).
    signals : target position per bar in {-1, 0, +1} (short/flat/long).
    leverage : position notional as a multiple of equity.
    fee, slippage, maintenance_margin : cost model (fractions).
    funding_rate : per-bar perp funding on held notional (0 disables it).
    """
    sig = signals.reindex(df.index).fillna(0.0).clip(-1, 1).values
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    n = len(df)

    cash = float(initial_capital)
    units = 0.0
    active_sig = 0.0  # the signal value our current position was opened under
    equity_curve = np.empty(n, dtype=float)
    trades: list[dict] = []
    liquidated = False
    held_bars = 0

    for t in range(n):
        price = close[t]

        # --- costs on the position carried into this bar ---
        if units != 0.0:
            if funding_rate:
                cash -= funding_rate * abs(units) * price

            # Liquidation: check the worst intrabar price for the held side.
            worst = low[t] if units > 0 else high[t]
            equity_worst = cash + units * worst
            if equity_worst <= maintenance_margin * abs(units) * worst:
                # Forced close at the worst price (with fee). Equity is wiped.
                cash += units * worst
                cash -= fee * abs(units) * worst
                trades.append({
                    "timestamp": df.index[t], "side": "liquidation",
                    "price": worst, "units": -units,
                })
                units = 0.0
                liquidated = True
                active_sig = sig[t]  # wait for a fresh signal change to re-enter

        equity = cash + units * price
        if equity <= 0.0:
            # Account busted; nothing left to trade.
            equity_curve[t:] = max(equity, 0.0)
            equity = 0.0
            break

        # --- (re)open a position only when the signal changes ---
        if sig[t] != active_sig:
            target_units = sig[t] * equity * leverage / price
            delta = target_units - units
            if abs(delta) * price > _EPS:
                exec_price = price * (1.0 + slippage * np.sign(delta))
                cash -= delta * exec_price
                cash -= fee * abs(delta) * exec_price
                units = target_units
                trades.append({
                    "timestamp": df.index[t],
                    "side": "buy" if delta > 0 else "sell",
                    "price": exec_price, "units": delta,
                })
            active_sig = sig[t]

        if units != 0.0:
            held_bars += 1
        equity_curve[t] = cash + units * price

    exposure = held_bars / n if n else 0.0
    curve = pd.Series(equity_curve, index=df.index, name="equity")

    return BacktestResult(
        equity_curve=curve,
        trades=trades,
        liquidated=liquidated,
        exposure=exposure,
        params={
            "initial_capital": initial_capital,
            "leverage": leverage,
            "fee": fee,
            "slippage": slippage,
            "funding_rate": funding_rate,
            "maintenance_margin": maintenance_margin,
        },
    )
