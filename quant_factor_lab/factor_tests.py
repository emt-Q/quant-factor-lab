"""
Factor hypothesis testing & multiple-testing correction.

Implements the statistical machinery that QR desks actually use to vet an
alpha idea before it ever reaches a portfolio:

  - IC (Pearson) and Rank-IC (Spearman) per rebalance date
  - ICIR and Student t-statistic (H0: IC = 0)
  - quantile-portfolio spread (Q5 - Q1) with t-statistic
  - BHY (Benjamini-Hochberg-Yekutieli) FDR control across the whole
    candidate universe - the standard answer to the "factor zoo" problem.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Forward returns
# --------------------------------------------------------------------------

def forward_returns(prices: pd.DataFrame,
                    grid: list[pd.Timestamp],
                    horizon: int = 20) -> pd.DataFrame:
    """h-day ahead total return for every stock, on the grid dates."""
    idx = prices.index
    out = {}
    for d in grid:
        i = idx.get_loc(d)
        if i + horizon >= len(prices):
            continue
        out[d] = prices.iloc[i + horizon] / prices.iloc[i] - 1.0
    return pd.DataFrame(out).T


# --------------------------------------------------------------------------
# Per-factor tests
# --------------------------------------------------------------------------

def _spearman(x: pd.Series, y: pd.Series) -> float:
    return x.rank().corr(y.rank())


def test_factor(factor: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    """Run IC / Rank-IC / quintile long-short tests for one factor."""
    ics, rics, ls_rets = [], [], []
    for d in factor.index:
        if d not in fwd.index:
            continue
        x, y = factor.loc[d], fwd.loc[d]
        m = x.notna() & y.notna()
        if int(m.sum()) < 10:
            continue
        x, y = x[m], y[m]
        ics.append(np.corrcoef(x, y)[0, 1])
        rics.append(_spearman(x, y))
        q = x.rank(pct=True)
        top, bot = y[q >= 0.80].mean(), y[q <= 0.20].mean()
        ls_rets.append(top - bot)

    ic = np.asarray(ics, dtype=float)
    ls = np.asarray(ls_rets, dtype=float)
    n = len(ic)
    if n < 3:
        return {"n_obs": n, "mean_ic": float("nan"), "icir": float("nan"),
                "t_stat": float("nan"), "mean_rank_ic": float("nan"),
                "ls_mean": float("nan"), "ls_t": float("nan"),
                "p_value": float("nan")}

    mean_ic = ic.mean()
    sd_ic = ic.std(ddof=1)
    t_stat = mean_ic / (sd_ic / math.sqrt(n)) if sd_ic > 0 else float("nan")
    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2)))) \
        if not math.isnan(t_stat) else float("nan")
    return {
        "n_obs": n,
        "mean_ic": mean_ic,
        "icir": mean_ic / sd_ic if sd_ic > 0 else float("nan"),
        "t_stat": t_stat,
        "mean_rank_ic": float(np.mean(rics)),
        "ls_mean": float(ls.mean()),
        "ls_t": float(ls.mean() / (ls.std(ddof=1) / math.sqrt(n))) if ls.std(ddof=1) > 0 else float("nan"),
        "p_value": p_value,
    }


def run_all_tests(factors: dict[str, pd.DataFrame],
                  fwd: pd.DataFrame) -> pd.DataFrame:
    """Test every factor and return a summary table (rows = factors)."""
    rows = {}
    for name, f in factors.items():
        rows[name] = test_factor(f, fwd)
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------
# Multiple-testing correction: BHY-FDR
# --------------------------------------------------------------------------

def bhy_correction(p_values: pd.Series, q: float = 0.05) -> pd.Series:
    """
    Benjamini-Hochberg-Yekutieli step-up procedure. Controls the False
    Discovery Rate under arbitrary dependence among tests (the safe choice
    for correlated factor tests, where classic BH is too optimistic).
    Returns a boolean Series: True = factor survives multiple-testing.
    """
    p = p_values.astype(float)
    n = int(p.shape[0])
    order = np.argsort(p.values)
    sorted_p = p.values[order]
    c_m = float(np.sum(1.0 / np.arange(1, n + 1)))   # harmonic number
    cutoff = np.arange(1, n + 1) / n * q / c_m

    k = 0
    for i, pv in enumerate(sorted_p):
        if not np.isnan(pv) and pv <= cutoff[i]:
            k = i + 1
    selected = np.zeros(n, dtype=bool)
    selected[order[:k]] = True
    return pd.Series(selected, index=p.index)


def quantile_spreads(factor: pd.DataFrame, fwd: pd.DataFrame,
                     n_groups: int = 5) -> pd.DataFrame:
    """Mean forward return per factor quintile (for the quantile chart)."""
    rows = {}
    for d in factor.index:
        if d not in fwd.index:
            continue
        x, y = factor.loc[d], fwd.loc[d]
        m = x.notna() & y.notna()
        x, y = x[m], y[m]
        q = pd.qcut(x.rank(method="first"), n_groups, labels=False)
        rows[d] = y.groupby(q).mean()
    return pd.DataFrame(rows).T
