"""
End-to-end research pipeline (the 'paper' of the project).

  Data (synthetic, planted alpha)
    -> Factor construction (curated library + mining candidates)
    -> In-sample hypothesis tests (IC / Rank-IC / t / quantile / LS)
    -> BHY multiple-testing correction  (kills the noise factors)
    -> Composite construction (rank-average of survivors)
    -> Out-of-sample long-short backtest with costs
    -> Performance metrics + charts + CSV exports
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")  # headless backend: save PNGs, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["svg.fonttype"] = "none"  # keep SVG text as text (small files)

from . import config as C
from .synthetic_data import generate_panel
from .factors import (rebalance_grid, grid_positions, build_curated_factors,
                      build_mining_candidates, rank_average)
from .factor_tests import (forward_returns, run_all_tests, bhy_correction,
                           quantile_spreads)
from .metrics import perf_summary, drawdown_series
from .backtest import run_backtest, benchmark_equal_weight

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")


def _fmt(x, nd: int = 3) -> str:
    return "nan" if x is None or (isinstance(x, float) and x != x) else f"{x:.{nd}f}"


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 78)
    print("  EMT QUANT FACTOR LAB - Multi-Factor Alpha Mining & Backtest")
    print(f"  universe={C.N_STOCKS} stocks | {C.N_DAYS} days | "
          f"rebalance={C.REBAL_PERIOD}d | cost={C.COST_BPS}bps | seed={C.SEED}")
    print("=" * 78)

    # ---- 1. Data ----
    panel = generate_panel()
    prices = panel["prices"]
    grid = rebalance_grid(prices)
    positions = grid_positions(prices, grid)
    grid = list(positions.keys())
    n_train = int(len(grid) * C.TRAIN_SPLIT)
    train_grid, test_grid = grid[:n_train], grid[n_train:]
    fwd = forward_returns(prices, grid, C.REBAL_PERIOD)
    print(f"\n[1/6] Panel generated: {prices.shape[0]} dates x {prices.shape[1]} stocks | "
          f"rebalance dates: {len(grid)} (train {len(train_grid)}, test {len(test_grid)})")

    # ---- 2. Factors ----
    curated = build_curated_factors(panel, positions)
    mined = build_mining_candidates(panel, positions)
    print(f"[2/6] Curated factors: {len(curated)} | mining candidates: {len(mined)}")

    # ---- 3. In-sample hypothesis tests ----
    fwd_train = fwd.loc[train_grid]
    res_curated = run_all_tests(curated, fwd_train)
    res_mined = run_all_tests(mined, fwd_train)

    sig_curated = bhy_correction(res_curated["p_value"], C.FDR_Q)
    sig_mined = bhy_correction(res_mined["p_value"], C.FDR_Q)

    print("\n[3/6] Curated factor tests (in-sample)")
    show = res_curated.copy()
    show["sig@FDR"] = ["YES" if s else "no" for s in sig_curated.values]
    print(show[["mean_ic", "icir", "t_stat", "mean_rank_ic", "ls_mean", "p_value", "sig@FDR"]]
          .round(3).to_string())

    print("\n      Factor-mining candidates (in-sample, incl. planted noise)")
    show_m = res_mined.copy()
    show_m["sig@FDR"] = ["YES" if s else "no" for s in sig_mined.values]
    print(show_m[["mean_ic", "t_stat", "p_value", "sig@FDR"]].round(3).to_string())

    # ---- 4. Composite: rank-average of surviving curated factors ----
    selected = [k for k in res_curated.index if sig_curated[k]]
    if not selected:
        selected = [res_curated["t_stat"].abs().idxmax()]
        print("\n      WARNING: no curated factor survived FDR; using best t-stat.")
    print(f"\n[4/6] Survivors after BHY-FDR: {selected}")

    composite = rank_average(curated, selected)
    # OOS sanity: IC of composite on the test window
    fwd_test = fwd.loc[test_grid]
    oos_ics = [np.corrcoef(composite.loc[d].dropna(), fwd_test.loc[d]
                           .reindex(composite.columns).dropna())[0, 1]
               for d in test_grid if d in composite.index and d in fwd_test.index
               and len(composite.loc[d].dropna()) > 10]
    oos_ics = [x for x in oos_ics if not np.isnan(x)]
    if oos_ics:
        print(f"      Composite out-of-sample mean IC: {np.mean(oos_ics):.4f} "
              f"(t={np.mean(oos_ics) / (np.std(oos_ics, ddof=1) / np.sqrt(len(oos_ics))):.2f})")

    # ---- 5. Out-of-sample backtest ----
    print("\n[5/6] Out-of-sample backtest (test window only)")
    ls_ret, ls_turn = run_backtest(composite, prices, test_grid, long_only=False)
    lo_ret, lo_turn = run_backtest(composite, prices, test_grid, long_only=True)
    bench = benchmark_equal_weight(prices)

    def _window(s: pd.Series) -> pd.Series:
        if ls_ret.empty:
            return s
        return s.loc[ls_ret.index[0]:ls_ret.index[-1]]

    summary = pd.DataFrame([
        perf_summary(ls_ret, name="long-short (selected factors)"),
        perf_summary(lo_ret, name="long-only (selected factors)"),
        perf_summary(_window(bench), name="benchmark (equal-weight)"),
    ])
    print(summary.round(3).to_string(index=False))
    if ls_turn:
        print(f"      avg one-way turnover per rebalance: {np.mean(ls_turn):.3f} "
              f"(cost {C.COST_BPS}bps applied on both legs)")

    # ---- 6. Quantile diagnostics for the strongest survivor ----
    best = max(selected, key=lambda k: abs(res_curated.loc[k, "t_stat"]))
    qs = quantile_spreads(curated[best], fwd_train)
    print(f"\n[6/6] Quintile mean forward returns for '{best}' (in-sample):")
    print(qs.mean().round(4).to_string())

    # ---- Save outputs ----
    res_curated.to_csv(os.path.join(OUT_DIR, "factor_test_results.csv"))
    res_mined.to_csv(os.path.join(OUT_DIR, "mining_results.csv"))
    summary.to_csv(os.path.join(OUT_DIR, "backtest_summary.csv"), index=False)
    ls_ret.rename("long_short").to_csv(os.path.join(OUT_DIR, "portfolio_returns.csv"))
    _plot_factors(res_curated, sig_curated)
    _plot_quantiles(qs, best)
    _plot_equity(ls_ret, lo_ret, _window(bench))
    print(f"\nSaved outputs -> {OUT_DIR}/  (CSV + PNG)")


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------

def _plot_factors(res: pd.DataFrame, sig: pd.Series):
    names = res.index.tolist()
    ics = res["mean_ic"].values
    ts = res["t_stat"].values
    colors = ["#3b82f6" if sig[n] else "#cbd5e1" for n in names]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    bars = ax.bar(names, ics, color=colors)
    for b, t in zip(bars, ts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{t:.1f}", ha="center", va="bottom" if t >= 0 else "top",
                fontsize=8, color="#1d4ed8")
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_title("Factor IC with t-statistics (in-sample) - blue = survives BHY-FDR")
    ax.set_ylabel("Mean IC")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "factor_ic.png"), dpi=150)
    fig.savefig(os.path.join(OUT_DIR, "factor_ic.svg"))
    plt.close(fig)


def _plot_quantiles(qs: pd.DataFrame, best: str):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    means = qs.mean()
    ax.bar([f"Q{i+1}" for i in range(len(means))], means.values,
           color=["#93c5fd", "#bfdbfe", "#dbeafe", "#fca5a5", "#ef4444"])
    for i, v in enumerate(means.values):
        ax.text(i, v, f"{v*100:.2f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.axhline(0, color="#94a3b8", lw=0.8)
    ax.set_title(f"Mean forward return by factor quintile: {best}")
    ax.set_ylabel("Forward 20d return")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "quantile_returns.png"), dpi=150)
    fig.savefig(os.path.join(OUT_DIR, "quantile_returns.svg"))
    plt.close(fig)


def _plot_equity(ls: pd.Series, lo: pd.Series, bench: pd.Series):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    ax1.plot((1 + ls).cumprod(), label="Long-Short (selected factors)", color="#3b82f6", lw=1.6)
    ax1.plot((1 + lo).cumprod(), label="Long-Only (selected factors)", color="#a855f7", lw=1.4)
    ax1.plot((1 + bench).cumprod(), label="Benchmark (equal-weight)", color="#94a3b8", lw=1.2, ls="--")
    ax1.set_title("Out-of-sample cumulative returns")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.25)
    ax2.fill_between(drawdown_series(ls).index, drawdown_series(ls).values * 100,
                     0, color="#ef4444", alpha=0.35)
    ax2.set_title("Long-Short drawdown (%)")
    ax2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "equity_curve.png"), dpi=150)
    fig.savefig(os.path.join(OUT_DIR, "equity_curve.svg"))
    plt.close(fig)
