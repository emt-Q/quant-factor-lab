# Quant Factor Lab — Multi-Factor Alpha Mining & Backtesting Framework

> A complete, research-grade factor research pipeline for the **Quantitative
> Research (QR)** interview: synthetic data with a *planted* alpha structure →
> factor construction → hypothesis testing → multiple-testing correction →
> factor mining → composite construction → out-of-sample long-short backtest.

[中文说明 → README.zh-CN.md](README.zh-CN.md)

---

## Why this project matters for QR interviews

| What desks look for | What this repo demonstrates |
|---|---|
| Statistical rigor | IC / Rank-IC / ICIR / t-stat hypothesis tests with a real p-value implementation (no scipy shortcuts) |
| The "factor zoo" problem | **BHY-FDR** multiple-testing correction across the full candidate universe — planted noise factors are rejected |
| Out-of-sample discipline | All factor *selection* happens in-sample; the composite is only *evaluated* out-of-sample |
| Portfolio construction | Rank-average composite, long/short quintiles, transaction costs on both legs, turnover tracking |
| Ability to explain results | Reproducible synthetic data with known ground truth + quantile monotonicity diagnostics |

## Results (seed=42, 120 stocks × 1008 days, monthly rebalance)

### 1. Curated factor tests (in-sample, 26 rebalance dates)

| Factor | Mean IC | ICIR | t-stat | Survives FDR |
|---|---|---|---|---|
| momentum_12_1 | 0.229 | 2.44 | 12.45 | ✅ |
| momentum_6_1 | 0.196 | 1.74 | 8.86 | ✅ |
| reversal_1m | 0.066 | 0.56 | 2.84 | ✅ |
| volatility_20d | -0.140 | -1.22 | -6.23 | ✅ |
| idiosyncratic_vol | -0.148 | -1.30 | -6.61 | ✅ |
| earnings_yield | 0.116 | 1.48 | 7.55 | ✅ |
| turnover_20d | 0.113 | 1.61 | 8.22 | ✅ |
| log_market_cap | 0.024 | 0.21 | 1.09 | ❌ |
| amihud_illiquidity | -0.046 | -0.33 | -1.68 | ❌ |

7 of 9 curated factors survive BHY-FDR at q=0.05.

### 2. Factor mining (13 candidates incl. 2 planted noise walks)

- Real signals recovered: momentum across horizons, volatility, value (EP), liquidity.
- **Both planted noise factors rejected** (`noise_a` p=0.79, `noise_b` p=0.06) — exactly what FDR control is for.

### 3. Out-of-sample backtest (9 rebalance dates, 10 bps one-way cost)

| Strategy | Ann. return | Ann. vol | Sharpe | Max DD | Calmar |
|---|---|---|---|---|---|
| **Long-Short (7 survivors)** | **50.3%** | 6.9% | **7.29** | -1.8% | **28.0** |
| Long-Only | 4.8% | 15.5% | 0.31 | -11.0% | 0.44 |
| Benchmark (equal-weight) | -19.0% | 14.7% | -1.29 | -22.0% | -0.87 |

- Composite **out-of-sample mean IC = 0.18 (t=12.8)** — signal survives in the test window.
- Quintile spread for `momentum_12_1` is strictly monotone: -1.9%, +0.7%, +1.3%, +2.1%, +3.6%.

## Project structure

```
quant-factor-lab/
├── quant_factor_lab/
│   ├── config.py          # all knobs: universe, costs, FDR target
│   ├── synthetic_data.py  # panel with planted alpha + noise features
│   ├── factors.py         # curated library + mining candidates + composite
│   ├── factor_tests.py    # IC/Rank-IC/t-stat, quantiles, BHY-FDR
│   ├── metrics.py         # Sharpe, drawdown, Calmar, hit rate
│   ├── backtest.py        # long-short engine with costs & turnover
│   └── pipeline.py        # end-to-end orchestration + charts
├── scripts/run_pipeline.py
├── tests/test_smoke.py
└── output/                # generated CSVs + PNG charts
```

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py        # full pipeline
python tests/test_smoke.py            # sanity checks
```

Charts and CSVs are written to `output/`.

## Methodology notes (the interview answers)

1. **No look-ahead**: every factor uses only data up to its rebalance date; forward returns are measured after.
2. **Winsorize + cross-sectional z-score**: robust to outliers, comparable across dates.
3. **BHY-FDR** controls the false-discovery rate under *arbitrary dependence* among tests — the safe choice for correlated factor tests (plain BH is too optimistic).
4. **Transaction costs** are applied on total two-leg turnover each rebalance; turnover itself is reported so you can discuss capacity.
5. **Synthetic data**: the alpha structure is *planted* (momentum + value − volatility + illiquidity − size), so the pipeline's output can be checked against ground truth. Swap in real data via the `prices/volume/market_cap/ep` DataFrames — the pipeline is data-agnostic.

## Limitations

- Synthetic universe (planted effects are strong by design); real-market factors are noisier.
- Equal-weight quintiles; no risk model / sector neutrality / position sizing.
- 9 out-of-sample rebalances is a short test window — treat Sharpe as a demo, not an estimate.

## Extension ideas

- Add a Barra-style risk model and mean-variance factor weights.
- Time-series cross-validation (walk-forward) for factor selection.
- Sector/industry neutralization.
- Real data adapter (yfinance / local CSV) + data quality checks.

---
*Copyright © 2025 EMT Physics. All Rights Reserved.*
