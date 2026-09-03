# Quant Factor Lab

A multi-factor alpha mining and backtesting framework for cross-sectional equity
research: factor construction → statistical hypothesis testing → multiple-testing
control → factor mining → composite alpha construction → out-of-sample
long-short backtest.

[中文说明 → README.zh-CN.md](README.zh-CN.md)

---

## Overview

The pipeline consumes a daily panel of **prices / volume / market capitalization /
earnings yield** per stock and:

1. builds a curated library of cross-sectional factors (momentum, reversal,
   volatility, idiosyncratic volatility, size, value, liquidity);
2. tests each factor with IC / Rank-IC / ICIR / t-stat and exact p-values
   (implemented from first principles, no scipy dependency);
3. controls the false-discovery rate across the full candidate universe with
   **BHY-FDR** (valid under arbitrary dependence among tests);
4. mines additional candidate factors and runs them through the same tests;
5. constructs a **rank-average composite** from the FDR survivors;
6. evaluates the composite in an **out-of-sample long-short backtest** with
   transaction costs and turnover tracking.

A synthetic data generator with a *planted* alpha structure and two planted
noise features ships with the repository, so the full statistical machinery can
be validated against known ground truth. A **real-data adapter** (Yahoo Finance)
runs the identical pipeline on real market data.

## Features

- **Factor library (9 curated)**: momentum (12-1, 6-1), short-term reversal,
  realized volatility, idiosyncratic volatility, log market cap, earnings yield,
  Amihud illiquidity, turnover.
- **Statistical tests**: mean IC, Rank-IC, ICIR, t-stat, two-sided p-values;
  cross-sectional winsorization and z-scoring on every rebalance date.
- **Multiple-testing control**: BHY-FDR at a configurable q level; planted
  noise factors are correctly rejected.
- **Factor mining**: 13 candidates (momentum horizons, volatility windows,
  size/value/liquidity variants) screened in-sample with the same FDR gate.
- **Composite**: rank-average of survivors (equal weight), refreshed each
  rebalance; no look-ahead (factors use only data up to the rebalance date).
- **Backtest**: long-short quintile portfolio, both legs costed at the
  configured one-way bps, turnover reported per rebalance; drawdown / Sharpe /
  Calmar / hit-rate metrics.
- **Synthetic panel** (`synthetic_data.py`): 120 stocks × 1008 days, planted
  alpha (momentum + value − volatility − illiquidity − size) plus noise.
- **Real data** (`data_adapter.py`): yfinance download for arbitrary tickers /
  date ranges, per-field on-disk caching, offline re-runs from cache.

## Results — synthetic validation (seed=42, 120 × 1008 days, 20d rebalance)

### Curated factor tests (in-sample, 26 rebalance dates; FDR at q=0.05)

| Factor | Mean IC | ICIR | t-stat | Survives FDR |
|---|---|---|---|---|
| momentum_12_1 | 0.229 | 2.44 | 12.45 | yes |
| momentum_6_1 | 0.196 | 1.74 | 8.86 | yes |
| reversal_1m | 0.066 | 0.56 | 2.84 | yes |
| volatility_20d | -0.140 | -1.22 | -6.23 | yes |
| idiosyncratic_vol | -0.148 | -1.30 | -6.61 | yes |
| earnings_yield | 0.116 | 1.48 | 7.55 | yes |
| turnover_20d | 0.113 | 1.61 | 8.22 | yes |
| log_market_cap | 0.024 | 0.21 | 1.09 | no |
| amihud_illiquidity | -0.046 | -0.33 | -1.68 | no |

7 of 9 curated factors survive; the two planted noise factors in the mining
universe are rejected (`noise_a` p=0.79, `noise_b` p=0.06).

### Out-of-sample backtest (9 rebalance dates, 10 bps one-way cost)

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| Long-Short (7 survivors) | 50.3% | 6.9% | 7.29 | -1.8% | 28.0 |
| Long-Only | 4.8% | 15.5% | 0.31 | -11.0% | 0.44 |
| Benchmark (equal-weight) | -19.0% | 14.7% | -1.29 | -22.0% | -0.87 |

- Composite out-of-sample mean IC = 0.18 (t=12.8).
- Quintile spread for `momentum_12_1` is strictly monotone:
  -1.9% / +0.7% / +1.3% / +2.1% / +3.6%.

## Real-data mode

```bash
pip install -r requirements-data.txt        # yfinance
python scripts/run_pipeline.py --real                       # default 20 tickers, 2019–2024
python scripts/run_pipeline.py --real --tickers AAPL,MSFT,NVDA --start 2019-01-01 --end 2024-12-31
python scripts/run_pipeline.py --real --refresh             # force re-download
```

The first run downloads and caches the panel (default cache `data/yahoo_cache`);
subsequent runs load from cache. Outputs are written with a `real_` prefix so
synthetic and real runs do not overwrite each other.

Reference result on the default 20 mega-cap tickers (2019-01-01 → 2024-12-31,
1509 trading days, 60 rebalances / 45 train / 15 test):

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD |
|---|---|---|---|---|
| Long-Short (FDR survivors) | 23.2% | 28.1% | 0.83 | -25.8% |
| Long-Only | 58.5% | 29.0% | 2.02 | -18.5% |
| Benchmark (equal-weight) | 40.4% | 13.4% | 3.02 | -8.2% |

Only `momentum_6_1` survives FDR in this universe: a 20-name mega-cap pool has
low cross-sectional dispersion, which the framework surfaces honestly instead
of forcing significance.

## Project structure

```
quant-factor-lab/
├── quant_factor_lab/
│   ├── config.py          # universe, rebalance, costs, FDR q, seeds
│   ├── synthetic_data.py  # panel generator (planted alpha + noise)
│   ├── data_adapter.py    # real-data adapter (yfinance, cached)
│   ├── factors.py         # curated library + mining candidates + composite
│   ├── factor_tests.py    # IC/Rank-IC/t-stat, quantiles, BHY-FDR
│   ├── metrics.py         # Sharpe, drawdown, Calmar, hit rate
│   ├── backtest.py        # long-short engine with costs & turnover
│   └── pipeline.py        # end-to-end orchestration + charts
├── scripts/run_pipeline.py
├── tests/test_smoke.py
├── requirements.txt
├── requirements-data.txt
└── output/                # generated CSVs + PNG charts
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py        # synthetic validation run
python tests/test_smoke.py            # sanity checks
```

Charts and CSVs are written to `output/`.

## Methodology

- **No look-ahead**: every factor uses only data up to its rebalance date;
  forward returns are measured from the day after.
- **Winsorization + cross-sectional z-score** on every rebalance date keeps
  factor values robust and comparable across dates.
- **BHY-FDR** controls the false-discovery rate under arbitrary dependence
  among the tests; the factor count used by the BH correction is the number of
  tests in the candidate set (curated or mined).
- **Transaction costs** are charged on total two-leg turnover each rebalance;
  turnover is reported so capacity can be assessed.
- **Synthetic ground truth**: the alpha structure is planted, so every
  downstream statistic (FDR survival, IC, backtest PnL) can be checked against
  what the generator put in.

## Configuration

All knobs live in `quant_factor_lab/config.py`: universe size / horizon,
rebalance period, top-quantile width, one-way cost in bps, FDR q-level,
seeds, and the synthetic alpha/volatility structure.

## Limitations

- The synthetic universe has strong planted effects by design; real-market
  factor signal is weaker and the framework reports it as-is.
- Equal-weight quintile portfolios; no risk model, sector neutrality, or
  position sizing.
- Short out-of-sample window (9 synthetic rebalances / 15 real rebalances) —
  Sharpe values are indicative, not estimates.

---

*Copyright © 2025 EMT Physics. All Rights Reserved.*
