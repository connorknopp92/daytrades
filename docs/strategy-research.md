# Trading-strategy research (fact-checked)

This summarizes a multi-source, adversarially-verified review of systematic
trading strategies, used to decide which strategies this app implements and —
more importantly — **how** it presents "the best strategy." It is educational
context, not financial advice.

## The classic, evidence-backed strategies (⭐ in the app)

The strategies the academic and practitioner literature most consistently cite
are **trend-following and momentum**:

- **Moving-average trend timing (Faber).** Hold while price is above its long
  SMA (the canonical rule uses the 10-*month* SMA), move to cash below it.
  Source: Faber, *A Quantitative Approach to Tactical Asset Allocation*.
- **Time-series / absolute momentum.** Go long if the asset's own trailing
  ~12-month return is positive, else exit. Sources: Moskowitz, Ooi & Pedersen
  (2012, *JFE*); Antonacci (2013, dual momentum).
- Widely-used trend/breakout cousins: **MACD**, **SMA crossover**,
  **Donchian/"turtle" breakout**.

Mean-reversion, channel, and oscillator families (**RSI**, **Bollinger**,
**z-score**, **Keltner**) are implemented for comparison but are **not** flagged
classic — their edge is weaker, mixed, or uncited in the verified evidence.

## What the evidence actually says (the honest part)

- **The benefit is risk reduction, not higher returns.** Faber's timing model
  beat the S&P 500 on *compound* return (10.18% vs 9.32%, 1901–2012) mainly by
  cutting volatility and drawdown (early-1930s drawdown 83.66% → 42.24%); average
  returns were essentially identical. Trend rules shine in sustained
  trends/crashes and **whipsaw in choppy markets**.
- **Most headline numbers are gross-of-cost and in-sample.** TSMOM's Sharpe >1 is
  before realistic fees/slippage; later work (Huang et al. 2020) finds weak
  out-of-sample evidence and post-2008 decay.
- **Mixed track record overall.** Park & Irwin (2007) surveyed 95 studies: 56
  positive, 20 negative, 19 mixed — and profitability largely predates the early
  1990s, confounded by data snooping.
- **In-sample success routinely reverses out-of-sample.** Sullivan, Timmermann &
  White (1999): the best DJIA rule was robust in-sample 1897–1986 but its edge
  *completely reversed* out-of-sample 1987–1996.
- **Picking a strategy by its best backtest is overfitting.** Bailey & López de
  Prado (2014): with enough strategies tried, a profitable-looking one is
  *guaranteed* by chance; a backtest that ignores how many were tried is
  "worthless" without out-of-sample/walk-forward validation.
- **Crypto-specific evidence is thin** — the verified sources test equities, FX,
  commodities, and bonds, not crypto. Applying any of this to BTC/ETH is an
  extrapolation.

## How this app responds

1. **Walk-forward ranking** (`src/ranking.py`): strategies are ranked on a
   *training* window, then their *hold-out* performance is shown next to it, so
   the in-sample-vs-out-of-sample gap is visible instead of hidden.
2. **Buy-and-hold is always the benchmark** and is itself one of the ranked
   strategies — it frequently wins, which is the point.
3. **Three metrics, side by side** (Sharpe / total return / Calmar) so no single
   number is cherry-picked.
4. **Every surface carries the caveat** that rankings are historical, after-cost,
   and *not predictive*, and that selecting by past performance is overfitting.

## Key sources

- Faber (2007/2013), *A Quantitative Approach to Tactical Asset Allocation*.
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, JFE.
- Antonacci (2013), *Absolute Momentum* (NAAIM; practitioner).
- Park & Irwin (2007), *What Do We Know About the Profitability of Technical
  Analysis?*, Journal of Economic Surveys.
- Sullivan, Timmermann & White (1999), *Data-Snooping, Technical Trading Rule
  Performance, and the Bootstrap*, Journal of Finance.
- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, JPM; Carr & López
  de Prado (2014), *Determining Optimal Trading Rules without Backtesting*.

*Caveat on sources:* the Faber and Antonacci backtests are single in-sample tests
by the strategies' own promoters with modest cost assumptions; treat their
outperformance as illustrative, not proof.
