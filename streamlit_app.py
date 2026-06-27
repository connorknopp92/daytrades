"""Crypto research & backtesting dashboard (Streamlit web app).

A touch-friendly browser UI over the same engine the CLI uses. Simulation only:
no real money, no exchange keys, no order execution. Run locally with
``streamlit run streamlit_app.py`` or deploy free on Streamlit Community Cloud.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.config import load_config
from src import service
from src import accumulation as accum
from src import ranking as ranking_mod
from src.strategies.base import available_strategies

st.set_page_config(page_title="Crypto Backtest Sandbox", page_icon="📊", layout="wide")

cfg = load_config()


@st.cache_data(show_spinner="Loading market data…", ttl=3600)
def _load(symbol: str):
    """Cache OHLCV so the UI stays snappy. Returns (df, synthetic)."""
    return service.load_symbol(cfg, symbol)


def _pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


# --- header + honest expectations ------------------------------------------
st.title("📊 Crypto Backtest Sandbox")
st.warning(
    "**Simulation only — no real money, no live trading.** This tool can't predict "
    "prices or guarantee profit. It exists to show what strategies *would* have done "
    "on real history after fees, slippage and liquidation — and how leverage wipes "
    "accounts out. Most active strategies lose to simply holding.",
    icon="⚠️",
)

# --- shared control: which market --------------------------------------------
with st.sidebar:
    st.header("Market")
    asset = st.radio("Asset type", ["Crypto", "Stocks / ETFs"], horizontal=True)
    if asset == "Crypto":
        symbol = st.selectbox("Coin", cfg["data"]["symbols"], index=0)
    else:
        symbol = st.selectbox("Stock / ETF", cfg["data"]["stocks"], index=0)
    st.caption("This choice applies to both tabs.")
    st.caption("ℹ️ Stablecoins (USDC/USDT) are excluded — they sit at ~$1, "
               "so there's nothing to backtest.")

df, synthetic = _load(symbol)
if synthetic:
    st.error("Live exchange unreachable — showing **SYNTHETIC** data (not real market history).")

tab_bt, tab_buy, tab_lab = st.tabs(
    ["📉 Strategy backtest", "🛒 When to buy (accumulation)", "🧪 Strategy Lab"])

# ===========================================================================
# TAB 1 — strategy backtest vs buy & hold
# ===========================================================================
with tab_bt:
    # Timeframe (candle size) + history window controls.
    tc1, tc2 = st.columns(2)
    view = tc1.radio("Timeframe", ["Daily", "Weekly", "Monthly"],
                     horizontal=True, key="view").lower()
    hist_label = tc2.selectbox("History", ["1 year", "3 years", "5 years",
                                           "10 years", "Max"], index=2)
    hist_years = {"1 year": 1, "3 years": 3, "5 years": 5,
                  "10 years": 10, "Max": None}[hist_label]
    prepared = service.slice_years(service.resample_ohlcv(df, view), hist_years)

    st.subheader(f"Market — {symbol} ({view}, {hist_label.lower()})")
    if len(prepared) < 2:
        st.warning("Not enough data for this timeframe/history. Try a longer window.")
        st.stop()
    mkt = service.market_summary(cfg, symbol, df=prepared, view=view)
    s = mkt["stats"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", _pct(s["total_return"]))
    c2.metric("CAGR", _pct(s["cagr"]))
    c3.metric("Annualized vol", _pct(s["ann_volatility"]))
    c4.metric("Max drawdown", _pct(s["max_drawdown"]))

    st.caption("Price")
    st.line_chart(mkt["price"], height=260)
    left, right = st.columns(2)
    with left:
        st.caption("Drawdown")
        st.area_chart(mkt["drawdown"], height=220, color="#d62728")
    with right:
        st.caption("Rolling annualized volatility")
        st.line_chart(mkt["rolling_vol"].dropna(), height=220, color="#9467bd")

    st.divider()
    st.subheader("Backtest a trading strategy")
    cc = st.columns(5)
    strategy = cc[0].selectbox(
        "Strategy", [s for s in available_strategies() if s != "buy_and_hold"], index=0)
    leverage = cc[1].slider("Leverage", 1, 25, 1, 1,
                            help="1 = spot-like. Crank it up and watch liquidation.")
    fee = cc[2].number_input("Fee/side", value=float(cfg["backtest"]["fee"]),
                             min_value=0.0, max_value=0.02, step=0.0005, format="%.4f")
    slippage = cc[3].number_input("Slippage", value=float(cfg["backtest"]["slippage"]),
                                  min_value=0.0, max_value=0.02, step=0.0005, format="%.4f")
    funding = cc[4].number_input("Funding/bar", value=float(cfg["backtest"]["funding_rate"]),
                                 min_value=0.0, max_value=0.005, step=0.0001, format="%.4f",
                                 help="Perp funding on held notional. Try 0.0003.")
    run = st.button("Run backtest", type="primary")

    st.subheader(f"{strategy} vs buy & hold")
    if not run:
        st.info("Set your controls above, then tap **Run backtest**.")
    else:
        res = service.backtest_summary(
            cfg, symbol, strategy, leverage=float(leverage),
            fee=fee, slippage=slippage, funding=funding, df=prepared, view=view,
        )
        sm, bm = res["strat_metrics"], res["bench_metrics"]

        if sm.get("liquidated"):
            st.error("💥 **LIQUIDATED** — leverage wiped the account out. This is the "
                     "central risk of trading futures.", icon="💥")

        beat = sm["total_return"] > bm["total_return"]
        (st.success if beat else st.info)(
            f"The strategy **{'beat' if beat else 'did NOT beat'}** simply buying and "
            f"holding over this period. Most don't, once costs are paid.")

        equity = pd.DataFrame({strategy: res["strat_equity"], "buy_and_hold": res["bench_equity"]})
        st.caption("Equity curve (USD)")
        st.line_chart(equity, height=300)

        def fmt(metric, key, pct=True):
            f = _pct if pct else (lambda x: f"{x:.2f}")
            return {"Metric": metric, strategy: f(sm[key]), "Buy & Hold": f(bm[key])}

        table = [
            fmt("Total return", "total_return"),
            fmt("CAGR", "cagr"),
            fmt("Annualized vol", "annualized_volatility"),
            fmt("Sharpe", "sharpe", pct=False),
            fmt("Sortino", "sortino", pct=False),
            fmt("Max drawdown", "max_drawdown"),
            fmt("Calmar", "calmar", pct=False),
            fmt("Win rate", "win_rate"),
            fmt("Profit factor", "profit_factor", pct=False),
            {"Metric": "# trades", strategy: sm["n_trades"], "Buy & Hold": bm["n_trades"]},
            {"Metric": "Final equity",
             strategy: f"${sm['final_equity']:,.0f}",
             "Buy & Hold": f"${bm['final_equity']:,.0f}"},
        ]
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2 — when to buy (accumulation for a long-term hold)
# ===========================================================================
with tab_buy:
    st.subheader("When is a good time to buy?")
    st.write("The honest answer isn't *predicting the bottom* — it's picking a sensible "
             "rule for **when to put money in** and sticking to it. Below: a live read on "
             "today's conditions, then a backtest of weekly accumulation rules.")

    # --- live signal ---------------------------------------------------------
    sig = accum.current_signal(df)
    color = {"FAVORABLE": "🟢", "LEANING FAVORABLE": "🟡", "NEUTRAL": "⚪"}.get(sig["verdict"], "⚪")
    box = st.success if sig["verdict"] == "FAVORABLE" else (
        st.warning if sig["verdict"] == "LEANING FAVORABLE" else st.info)
    box(f"{color} **Signal today: {sig['verdict']}** — {sig['note']}")

    g1, g2, g3 = st.columns(3)
    g1.metric("Price", f"${sig['price']:,.0f}")
    if sig["pct_vs_ma"] is not None:
        g2.metric(f"vs {sig['ma_days']}-day avg", _pct(sig["pct_vs_ma"]),
                  help="Negative = below the long-term trend (historically a better entry).")
    g3.metric("Off recent high", _pct(sig["drawdown_from_high"]))
    st.caption("⚠️ " + sig["disclaimer"])

    st.divider()

    # --- accumulation backtest ----------------------------------------------
    st.markdown("#### Backtest: weekly buying over the last few years")
    a1, a2, a3 = st.columns(3)
    weekly_budget = a1.number_input("Weekly budget ($)", value=50.0, min_value=1.0, step=10.0)
    years = a2.slider("Years to test", 1, 5, 3, 1)
    dip_pct = a3.slider("Dip threshold (%)", 5, 50, 20, 5,
                        help="'Buy the dip' only buys when price is this far below its ~1y high.") / 100.0

    try:
        bt = accum.accumulation_backtest(
            cfg, symbol, df, years=years, weekly_budget=weekly_budget, dip_pct=dip_pct)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    st.caption(f"{bt['weeks']} weekly buys of ${weekly_budget:,.0f} "
               f"({bt['start'].date()} → {bt['end'].date()}). "
               f"Total contributed: ${bt['results']['dca']['contributed']:,.0f}.")

    # Comparison table.
    rows = []
    for key in ["dca", "lump_sum", "buy_the_dip", "below_ma"]:
        r = bt["results"][key]
        rows.append({
            "Strategy": accum.STRATEGY_LABELS[key],
            "Final value": f"${r['final_value']:,.0f}",
            "ROI": _pct(r["roi"]),
            "Avg cost": f"${r['avg_cost']:,.0f}" if r["avg_cost"] == r["avg_cost"] else "—",
            "Idle cash left": f"${r['idle_cash']:,.0f} ({_pct(r['pct_uninvested'])})",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Best by final value.
    best = max(bt["results"].items(), key=lambda kv: kv[1]["final_value"])
    st.success(f"🏆 Best over this window: **{accum.STRATEGY_LABELS[best[0]]}** "
               f"(${best[1]['final_value']:,.0f} from ${best[1]['contributed']:,.0f} in).")

    st.caption("Portfolio value over time (vs total money contributed)")
    chart_df = bt["value_ts"].rename(columns={**accum.STRATEGY_LABELS, "contributed": "Contributed"})
    st.line_chart(chart_df, height=320)

    st.info("Things to notice: **lump sum usually wins in a rising market** (money's in "
            "longer), but it also means buying right before any crash — more risk and more "
            "regret. **Buying weekly (DCA)** smooths that out and is the easiest to actually "
            "stick to. **Waiting for dips** can backfire — you may sit in cash while price "
            "runs away. There's no free lunch in timing; consistency is the realistic edge.")

# ===========================================================================
# TAB 3 — Strategy Lab: walk-forward ranking of every strategy
# ===========================================================================
with tab_lab:
    st.subheader(f"Strategy Lab — {symbol}")
    st.write("Ranks **every** strategy on a training window, then shows how each "
             "actually did on a later **hold-out** window. The gap between the two "
             "is the whole point: a strategy that looks great in training but falls "
             "apart out-of-sample was just overfit.")

    lc1, lc2 = st.columns(2)
    lab_view = lc1.radio("Timeframe", ["Daily", "Weekly", "Monthly"],
                         horizontal=True, key="lab_view").lower()
    lab_hist = lc2.selectbox("History", ["3 years", "5 years", "10 years", "Max"],
                             index=1, key="lab_hist")
    lab_years = {"3 years": 3, "5 years": 5, "10 years": 10, "Max": None}[lab_hist]
    lab_df = service.slice_years(service.resample_ohlcv(df, lab_view), lab_years)

    if len(lab_df) < 30:
        st.warning("Not enough data for this timeframe/history. Try a longer window.")
    else:
        rk = ranking_mod.rank_strategies(cfg, symbol, lab_df, view=lab_view)
        tp, hp = rk["train_period"], rk["holdout_period"]
        st.caption(
            f"Tested **{rk['n_strategies_tried']}** strategies · "
            f"train {tp[0].date()}→{tp[1].date()} · "
            f"hold-out {hp[0].date()}→{hp[1].date()}" if hp else
            f"Tested {rk['n_strategies_tried']} strategies (history too short for a hold-out).")

        st.error("⚠️ Rankings are **historical and after-cost — not a prediction.** "
                 "Picking the past winner is the easiest way to fool yourself; that's "
                 "why the hold-out column is here. Most strategies still lose to buy & hold.",
                 icon="⚠️")

        def cell(m, key, pct=True):
            if not m:
                return "—"
            v = m.get(key)
            return f"{v*100:,.1f}%" if pct else f"{v:,.2f}"

        table = []
        for r in sorted(rk["rows"], key=lambda x: x["train"].get("sharpe", -9), reverse=True):
            table.append({
                "Strategy": ("⭐ " if r["classic"] else "") + r["name"],
                "Family": r["family"],
                "Train Sharpe": cell(r["train"], "sharpe", pct=False),
                "Train return": cell(r["train"], "total_return"),
                "Hold-out return": cell(r["holdout"], "total_return"),
                "Hold-out Sharpe": cell(r["holdout"], "sharpe", pct=False),
                "Train Calmar": cell(r["train"], "calmar", pct=False),
            })
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

        b = rk["best_by"]
        st.info(f"**Train-window leaders** — by Sharpe: `{b['sharpe']}` · "
                f"by total return: `{b['total_return']}` · by Calmar: `{b['calmar']}`. "
                "Check their **hold-out** columns above before trusting any of them. "
                "⭐ = well-known, evidence-backed classic (trend/momentum).")

        # Equity overlay for a chosen strategy vs buy & hold over the full window.
        names = [r["name"] for r in rk["rows"]]
        default = b["sharpe"] if b["sharpe"] in names else names[0]
        pick = st.selectbox("Equity curve (full window) — compare a strategy vs buy & hold",
                            names, index=names.index(default))
        res = service.backtest_summary(cfg, symbol, pick, leverage=1.0, df=lab_df, view=lab_view)
        eq = pd.DataFrame({pick: res["strat_equity"], "buy_and_hold": res["bench_equity"]})
        st.line_chart(eq, height=300)

st.divider()
st.caption("Educational sandbox. Not financial advice. Past performance — even real "
           "and after costs — does not predict the future.")
