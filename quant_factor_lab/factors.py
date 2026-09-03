"""
Factor construction from raw panels.

Every factor is computed on a *rebalance grid* (every REBAL_PERIOD days)
using only information available up to that date (no look-ahead), then
cross-sectionally winsorized (3 sigma) and z-scored. The sign convention is
"higher factor value = more exposed to the characteristic"; the hypothesis
tests report significance regardless of sign.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REBAL_PERIOD, MIN_HISTORY


def rebalance_grid(prices: pd.DataFrame) -> list[pd.Timestamp]:
    """Dates (aligned to price index) on which factors are measured."""
    n = len(prices)
    positions = range(MIN_HISTORY, n - REBAL_PERIOD, REBAL_PERIOD)
    return [prices.index[i] for i in positions]


def grid_positions(prices: pd.DataFrame, grid: list[pd.Timestamp]) -> dict:
    """Map each grid date to its integer position in the price index."""
    idx = prices.index
    pos = {d: idx.get_loc(d) for d in grid}
    # keep only dates with enough trailing history and leading forward window
    pos = {d: i for d, i in pos.items()
           if i >= MIN_HISTORY and i + REBAL_PERIOD < len(prices)}
    return pos


# --------------------------------------------------------------------------
# Cross-sectional helpers
# --------------------------------------------------------------------------

def _winsorize(s: pd.Series, n: float = 3.0) -> pd.Series:
    med, std = s.median(), s.std(ddof=0)
    if std is None or (isinstance(std, float) and (std != std)) or std == 0:
        return s
    return s.clip(med - n * std, med + n * std)


def _zscore_cs(s: pd.Series) -> pd.Series:
    s = _winsorize(s)
    m, sd = s.mean(), s.std(ddof=0)
    if sd is None or (isinstance(sd, float) and (sd != sd)) or sd < 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - m) / sd


def _snapshot(func, positions: dict) -> pd.DataFrame:
    """Evaluate func(i) -> Series(stock) for every grid position i."""
    out = {}
    for d, i in positions.items():
        out[d] = func(i)
    return pd.DataFrame(out).T  # dates x stocks


# --------------------------------------------------------------------------
# Curated factor library
# --------------------------------------------------------------------------

def build_curated_factors(panel: dict, positions: dict) -> dict[str, pd.DataFrame]:
    """Nine textbook factors, each a dates x stocks z-score panel."""
    prices, rets = panel["prices"], panel["returns"]
    cap, ep, vol_panel = panel["market_cap"], panel["ep"], panel["volume"]

    def _mom(i: int, lookback: int, skip: int) -> pd.Series:
        p0 = prices.iloc[i - lookback - skip]
        p1 = prices.iloc[i - skip]
        return p1 / p0 - 1.0

    def _realized_vol(i: int, w: int = 20) -> pd.Series:
        return rets.iloc[i - w:i].std()

    def _idio_vol(i: int, w: int = 20) -> pd.Series:
        r = rets.iloc[i - w:i]
        demeaned = r.sub(r.mean(axis=1), axis=0)   # remove market/level effect
        return demeaned.std()

    def _amihud(i: int, w: int = 20) -> pd.Series:
        return (rets.iloc[i - w:i].abs() / vol_panel.iloc[i - w:i]).mean()

    def _turnover(i: int, w: int = 20) -> pd.Series:
        shares = cap.iloc[i] / prices.iloc[i]       # shares outstanding
        return (vol_panel.iloc[i - w:i] / shares).mean()

    factors = {
        "momentum_12_1":      _snapshot(lambda i: _zscore_cs(_mom(i, 240, 20)), positions),
        "momentum_6_1":       _snapshot(lambda i: _zscore_cs(_mom(i, 120, 20)), positions),
        "reversal_1m":        _snapshot(lambda i: _zscore_cs(_mom(i, 20, 0)), positions),
        "volatility_20d":     _snapshot(lambda i: _zscore_cs(_realized_vol(i)), positions),
        "idiosyncratic_vol":  _snapshot(lambda i: _zscore_cs(_idio_vol(i)), positions),
        "log_market_cap":     _snapshot(lambda i: _zscore_cs(np.log(cap.iloc[i])), positions),
        "earnings_yield":     _snapshot(lambda i: _zscore_cs(ep.iloc[i]), positions),
        "amihud_illiquidity": _snapshot(lambda i: _zscore_cs(_amihud(i)), positions),
        "turnover_20d":       _snapshot(lambda i: _zscore_cs(_turnover(i)), positions),
    }
    return factors


# --------------------------------------------------------------------------
# Factor mining: a candidate factory over raw features
# --------------------------------------------------------------------------

def build_mining_candidates(panel: dict, positions: dict) -> dict[str, pd.DataFrame]:
    """
    A larger candidate universe built from raw features plus planted noise.
    The BHY multiple-testing step is exactly what separates the real signals
    from the noise candidates (the 'factor zoo' problem).
    """
    prices, rets = panel["prices"], panel["returns"]
    cap, ep, vol_panel = panel["market_cap"], panel["ep"], panel["volume"]

    def _mom(i, lookback, skip):
        return prices.iloc[i - lookback - skip] / prices.iloc[i - skip] - 1.0

    def _vol(i, w):
        return rets.iloc[i - w:i].std()

    def _amihud(i, w):
        return (rets.iloc[i - w:i].abs() / vol_panel.iloc[i - w:i]).mean()

    cand: dict[str, pd.DataFrame] = {}
    # momentum across horizons
    for lb, skip in [(5, 0), (20, 0), (60, 20), (120, 20)]:
        cand[f"mom_{lb}d"] = _snapshot(lambda i, lb=lb, skip=skip: _zscore_cs(_mom(i, lb, skip)), positions)
    cand["neg_mom_5d"] = -cand["mom_5d"]                       # sign-agnostic demo
    # volatility across horizons
    cand["vol_20d"] = _snapshot(lambda i: _zscore_cs(_vol(i, 20)), positions)
    cand["vol_60d"] = _snapshot(lambda i: _zscore_cs(_vol(i, 60)), positions)
    # fundamentals / liquidity
    cand["log_mktcap"] = _snapshot(lambda i: _zscore_cs(np.log(cap.iloc[i])), positions)
    cand["ep"] = _snapshot(lambda i: _zscore_cs(ep.iloc[i]), positions)
    cand["amihud_20d"] = _snapshot(lambda i: _zscore_cs(_amihud(i, 20)), positions)
    cand["log_volume_20d"] = _snapshot(
        lambda i: _zscore_cs(np.log(vol_panel.iloc[i - 20:i].mean())), positions)
    # planted noise (only present in the synthetic panel - should be rejected)
    if "noise_a" in panel and "noise_b" in panel:
        cand["noise_a"] = _snapshot(lambda i: _zscore_cs(panel["noise_a"].iloc[i]), positions)
        cand["noise_b"] = _snapshot(lambda i: _zscore_cs(panel["noise_b"].iloc[i]), positions)
    return cand


# --------------------------------------------------------------------------
# Composite construction
# --------------------------------------------------------------------------

def rank_average(factors: dict[str, pd.DataFrame],
                 selected: list[str]) -> pd.DataFrame:
    """Equal-weight composite of selected factors after cross-sectional
    rank-normalization (robust to outliers and factor scale)."""
    frames = []
    for name in selected:
        f = factors[name]
        frames.append(f.rank(axis=1, pct=True) - 0.5)
    composite = sum(frames) / len(frames)
    return composite
