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

1. **Fetch** ~5 years of real daily price history (BTC/USD, ETH/USD, ...) from
   Coinbase, cached locally.
2. **Analyze** the markets: volatility, trends, drawdowns, BTC/ETH correlation —
   written up as a report with charts.
3. **Backtest** classic strategies against that history with realistic simulated
   futures mechanics (fees, slippage, leverage, funding, liquidation), always
   compared against a buy-and-hold benchmark.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

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
