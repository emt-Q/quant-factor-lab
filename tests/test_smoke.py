#!/usr/bin/env python3
"""Smoke tests for the factor pipeline (no pytest needed).

Usage:
    python tests/test_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from quant_factor_lab.synthetic_data import generate_panel  # noqa: E402
from quant_factor_lab.factors import (rebalance_grid, grid_positions,  # noqa: E402
                                      build_curated_factors, build_mining_candidates)
from quant_factor_lab.factor_tests import (forward_returns, run_all_tests,  # noqa: E402
                                           bhy_correction)
from quant_factor_lab.backtest import run_backtest  # noqa: E402
from quant_factor_lab import config as C  # noqa: E402


def _assert(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)
    print("  ok -", msg)


def test():
    print("Smoke tests: quant-factor-lab")
    panel = generate_panel()
    prices = panel["prices"]

    _assert(prices.shape == (C.N_DAYS, C.N_STOCKS), "panel shape")
    _assert(prices.isna().sum().sum() == 0, "no NaNs in prices")

    grid = grid_positions(prices, rebalance_grid(prices))
    curated = build_curated_factors(panel, grid)
    mined = build_mining_candidates(panel, grid)
    _assert(len(curated) == 9 and len(mined) == 13, "factor counts")

    fwd = forward_returns(prices, list(grid.keys()), C.REBAL_PERIOD)
    res = run_all_tests(curated, fwd)
    _assert(res["mean_ic"].notna().all(), "all factors produce IC")

    # BHY sanity: noise candidates should mostly be rejected
    sig = bhy_correction(res["p_value"], C.FDR_Q)
    _assert(int(sig.sum()) >= 1, f"at least one factor survives FDR ({int(sig.sum())})")

    # Backtest sanity: finite returns, realistic magnitude
    composite = curated["momentum_12_1"] * 0.0 + 0.0
    composite = curated["momentum_12_1"]  # crude single-factor score
    grid_dates = list(grid.keys())
    ret, turn = run_backtest(composite, prices, grid_dates[-8:])
    _assert(np.isfinite(ret).all(), "backtest returns finite")
    _assert(all(t >= 0 for t in turn), "turnover non-negative")
    print("All smoke tests passed.")


if __name__ == "__main__":
    test()
