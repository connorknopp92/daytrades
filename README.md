# daytrades — Crypto Research & Backtesting Sandbox

A Python tool for **learning how crypto trading actually works** by analyzing real
market history and backtesting strategies in a realistic simulation. It is a
*sandbox for building skill*, not a money machine.

## Read this first (honest expectations)

- **This does not trade real money.** There is no live trading, no exchange API
  keys, and no order execution anywhere in this project. It simulates.
- **It cannot predict the future or guarantee profit.** Nobody — no human, no AI,
  no algorithm — can reliably turn a small stake into a fortune by day-trading
  leveraged crypto futures. Most retail futures traders *lose* money.
- **Leverage cuts both ways.** With 10x leverage, a ~9–10% move against you wipes
  your entire position. This tool lets you *watch* that happen on real historical
  data (`--leverage 10`) so you learn the risk safely, with fake money.
- The real value here is education: see what strategies would actually have done
  after fees, slippage, funding, and liquidation — and how rarely they beat
  simply buying and holding.

## What it does

1. **Fetch** ~5 years of real daily price history, cached locally:
   - **Crypto** (BTC, ETH, SOL, XRP, DOGE, ADA, LTC, AVAX, LINK) from Coinbase.
   - **Stocks / ETFs** (AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, SPY, QQQ) from
     Yahoo Finance. Bare tickers (e.g. `AAPL`) route to stocks; `BTC/USD`-style
     pairs route to crypto. Annualized metrics use 252 trading days for stocks,
     365 for crypto.
   - Stablecoins (USDC/USDT/DAI) are intentionally excluded — pegged to ~$1, so
     there's nothing to chart or backtest.
2. **Analyze** the markets: volatility, trends, drawdowns, BTC/ETH correlation —
   written up as a report with charts.
3. **Backtest** classic strategies against that history with realistic simulated
   futures mechanics (fees, slippage, leverage, funding, liquidation), always
   compared against a buy-and-hold benchmark.

## Setup

```bash
pip install -r requirements.txt
```

## Web app (the clickable version)

A browser dashboard — pick a coin and strategy, drag the leverage slider, tap
**Run**, and see charts and stats. Works great on a tablet/phone browser. Two tabs:

- **Strategy backtest** — market analysis + a strategy vs. buy-and-hold, with
  **Daily / Weekly / Monthly** timeframes and a selectable history window
  (1/3/5/10 years or Max). Up to ~12 years of daily data is fetched and
  resampled; metrics annualize correctly per timeframe (52 weeks / 12 months).
- **When to buy (accumulation)** — a live "is now a good time to accumulate?"
  signal plus a backtest of weekly buying rules (DCA, lump sum, buy-the-dip,
  buy-below-200-day) over the last few years. Answers *when to enter a hold*
  without pretending to predict the bottom.
- **Strategy Lab** — ranks all 11 strategies **walk-forward** (trained on older
  data, then scored on a recent hold-out) by Sharpe / total return / Calmar, with
  ⭐ marking the evidence-backed classics. The hold-out column exposes overfitting:
  past winners routinely fail out-of-sample. See `docs/strategy-research.md` for
  the fact-checked basis (Faber, Moskowitz-Ooi-Pedersen, Sullivan-Timmermann-White,
  Bailey-López de Prado).

Strategies: `buy_and_hold`, `sma_crossover`⭐, `sma_timing`⭐, `macd`⭐,
`ts_momentum`⭐, `momentum_roc`, `donchian_breakout`⭐, `keltner_breakout`,
`rsi_reversion`, `bollinger_reversion`, `zscore_reversion`.

## Paper-trading sandbox (forward test)

`src/paper.py` runs a **fake-money** portfolio forward in time. By default it puts
**$50 in every market** (all coins + stocks): each day an "agent" follows the
walk-forward-best strategy per symbol, tracked against a buy-and-hold shadow, and
a **leaderboard** ranks which $50 stake is worth the most. State persists
(`paper_state.json`) so the record accumulates across the daily GitHub Action —
the honest, true out-of-sample test of whether picking strategies (or picking
"the best market") actually beats holding. Usually it doesn't, and last period's
winner rarely repeats. **Simulation only: no real money, no exchange, no orders.**
The daily email shows the running leaderboard.

Run it locally:

```bash
streamlit run streamlit_app.py
```

Then open the printed `http://localhost:8501` URL.

### Deploy a public link (free, no install for viewers)

Streamlit Community Cloud serves the app straight from this GitHub repo:

1. Go to **https://share.streamlit.io** and sign in with GitHub, then authorize it.
2. **Create app → Deploy a public app from GitHub** → pick the repo
   `connorknopp92/daytrades`, choose the branch, set the main file to
   `streamlit_app.py`.
3. **Deploy** and wait ~2 minutes. You'll get a URL like
   `https://<name>.streamlit.app`.
4. Open that URL in any browser. On Android Chrome, use **⋮ → Add to Home screen**
   for an app-like icon.

The deploy step is tied to your own GitHub login, so it's the one part you click
yourself — everything it needs is already in the repo.

## Usage (command line)

```bash
# 1. Download ~5 years of history (cached to data/cache/)
python -m src.cli fetch --symbols BTC/USD ETH/USD --years 5 --timeframe 1d

# 2. Produce a market analysis report (writes to outputs/)
python -m src.cli analyze --symbol BTC/USD

# 3. Backtest a strategy (writes metrics + equity-curve plot to outputs/)
python -m src.cli backtest --strategy sma_crossover --symbol BTC/USD --leverage 1

# 4. The lesson: same strategy, 10x leverage + higher fees
python -m src.cli backtest --strategy sma_crossover --symbol BTC/USD --leverage 10 --fee 0.002
```

Available strategies: `buy_and_hold`, `sma_crossover`, `rsi_reversion`,
`donchian_breakout`.

If the data source is unreachable, the tool falls back to a deterministic
synthetic price series so you can still explore the analysis and backtester
offline. Such data is clearly labelled and must not be mistaken for real history.

## Tests

```bash
pytest -q
```

## Project layout

```
src/
  cli.py              # fetch / analyze / backtest commands
  config.py           # load config.yaml
  data/               # ccxt Coinbase downloader + parquet cache
  strategies/         # strategy interface + 4 strategies + benchmark
  backtest/           # bar-by-bar engine + metrics + portfolio
  analysis/           # market statistics + report generation
tests/                # unit tests (run without network)
```

## A note on getting rich quick

If a strategy in here looks like it prints money, the first suspect is a bug
(lookahead bias, ignored fees, unmodeled liquidation) — not genius. That
skepticism is the most valuable thing this project can teach you.
