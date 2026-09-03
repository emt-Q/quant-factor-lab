"""
Long-short portfolio backtest with transaction costs.

A clean, deterministic engine: on every rebalance date the strategy sorts the
universe by a composite score, goes long the top quintile and short the bottom
quintile (equal weight), and applies one-way transaction costs on total
turnover (both legs). Daily portfolio returns are the weighted sum of stock
returns; the benchmark is an equal-weight long-only portfolio of the universe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REBAL_PERIOD, TOP_QUANTILE, COST_BPS


def run_backtest(score: pd.DataFrame,
                 prices: pd.DataFrame,
                 grid: list[pd.Timestamp],
                 top_q: float = TOP_QUANTILE,
                 cost_bps: float = COST_BPS,
                 long_only: bool = False) -> tuple[pd.Series, list[float]]:
    """Backtest a score panel. Returns (daily_returns, turnover_per_rebalance)."""
    daily = prices.pct_change().fillna(0.0)
    port_ret = pd.Series(0.0, index=prices.index)
    holdings: pd.Series | None = None
    turnover: list[float] = []

    for d in grid:
        if d not in score.index:
            continue
        s = score.loc[d].dropna()
        if len(s) < 20:
            continue
        k = max(1, int(round(top_q * len(s))))
        ranked = s.sort_values(ascending=False)
        long_idx = ranked.index[:k]
        short_idx = ranked.index[-k:] if not long_only else []

        w = pd.Series(0.0, index=s.index)
        w[long_idx] = 1.0 / k
        if len(short_idx) > 0:
            w[short_idx] = -1.0 / k

        if holdings is None:
            turn = float(w.abs().sum())
        else:
            turn = float((w - holdings.reindex(w.index).fillna(0.0)).abs().sum())
        turnover.append(turn)
        holdings = w

        cost = turn * (cost_bps / 1e4)
        i = prices.index.get_loc(d)
        window = daily.iloc[i:i + REBAL_PERIOD]
        for j, (dt, row) in enumerate(window.iterrows()):
            r = float((w * row).sum())
            if j == 0:
                r -= cost
            port_ret.loc[dt] = r

    # keep only the live window (first rebalance .. end of last holding period)
    if grid and port_ret.abs().sum() > 0:
        start, end = grid[0], grid[-1]
        end_pos = prices.index.get_loc(end) + REBAL_PERIOD
        port_ret = port_ret.loc[start:prices.index[min(end_pos, len(prices) - 1)]]
    return port_ret, turnover


def benchmark_equal_weight(prices: pd.DataFrame) -> pd.Series:
    """Equal-weight long-only return of the full universe."""
    return prices.pct_change().fillna(0.0).mean(axis=1)
