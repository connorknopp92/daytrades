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

# --- sidebar controls -------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    symbol = st.selectbox("Market", cfg["data"]["symbols"], index=0)
    strategy = st.selectbox(
        "Strategy", [s for s in available_strategies() if s != "buy_and_hold"], index=0
    )
    leverage = st.slider("Leverage", min_value=1, max_value=25, value=1, step=1,
                         help="1 = spot-like. Crank it up and watch liquidation.")
    fee = st.number_input("Fee per side", value=float(cfg["backtest"]["fee"]),
                          min_value=0.0, max_value=0.02, step=0.0005, format="%.4f")
    slippage = st.number_input("Slippage", value=float(cfg["backtest"]["slippage"]),
                               min_value=0.0, max_value=0.02, step=0.0005, format="%.4f")
    funding = st.number_input("Funding / bar", value=float(cfg["backtest"]["funding_rate"]),
                              min_value=0.0, max_value=0.005, step=0.0001, format="%.4f",
                              help="Perp funding on held notional. Try 0.0003 to see the bleed.")
    run = st.button("Run backtest", type="primary", use_container_width=True)

# --- load data --------------------------------------------------------------
df, synthetic = _load(symbol)
if synthetic:
    st.error("Live exchange unreachable — showing **SYNTHETIC** data (not real market history).")

# --- market analysis --------------------------------------------------------
st.subheader(f"Market — {symbol}")
mkt = service.market_summary(cfg, symbol, df=df)
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

# --- backtest ---------------------------------------------------------------
st.subheader(f"Backtest — {strategy} vs buy & hold")
if not run:
    st.info("Set your controls in the sidebar, then tap **Run backtest**.")
else:
    res = service.backtest_summary(
        cfg, symbol, strategy, leverage=float(leverage),
        fee=fee, slippage=slippage, funding=funding, df=df,
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

st.divider()
st.caption("Educational sandbox. Not financial advice. Past performance — even real "
           "and after costs — does not predict the future.")
