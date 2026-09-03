"""Performance & risk metrics for portfolio return series."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

PPY = 252  # trading days per year


def perf_summary(returns: pd.Series,
                 periods_per_year: int = PPY,
                 name: str = "strategy") -> dict:
    """Annualized return / vol / Sharpe / max drawdown / Calmar / hit rate."""
    r = returns.dropna()
    if len(r) == 0:
        return {"name": name, "ann_return": float("nan"), "ann_vol": float("nan"),
                "sharpe": float("nan"), "max_drawdown": float("nan"),
                "calmar": float("nan"), "hit_rate": float("nan")}
    ann_ret = float(r.mean()) * periods_per_year
    ann_vol = float(r.std(ddof=1)) * math.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    mdd = float(dd.min())
    calmar = ann_ret / abs(mdd) if (mdd < 0 and not math.isnan(mdd)) else float("nan")
    hit = float((r > 0).mean())

    return {"name": name, "ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": sharpe, "max_drawdown": mdd, "calmar": calmar,
            "hit_rate": hit}


def drawdown_series(returns: pd.Series) -> pd.Series:
    eq = (1.0 + returns.dropna()).cumprod()
    return eq / eq.cummax() - 1.0
