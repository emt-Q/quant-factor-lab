"""
Synthetic cross-sectional panel with a *known, planted* alpha structure.

The generator plants five true characteristics (momentum, size, value,
volatility, liquidity) and maps them into a per-stock expected return
(alpha). Daily prices / volumes / market caps are then simulated from that
alpha plus a common market factor and idiosyncratic risk.

Because the true characteristics are recoverable from price / volume data,
a properly built factor pipeline *should* rediscover most of them and reject
the planted noise features. This makes the dataset an excellent end-to-end
test bed for factor mining, hypothesis testing and backtesting, with a
ground-truth answer to check against.

NOTE: signs follow textbook factor economics:
  - high past momentum   -> higher expected return  (+)
  - high earnings yield  -> higher expected return  (+)
  - high volatility      -> lower  expected return  (-)
  - high illiquidity     -> higher expected return  (+)
  - large market cap     -> lower  expected return  (-)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import N_STOCKS, N_DAYS, SEED


def generate_panel(n_stocks: int = N_STOCKS,
                   n_days: int = N_DAYS,
                   seed: int = SEED) -> dict[str, pd.DataFrame]:
    """Generate a complete price/volume/cap/EP panel with planted alpha."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    columns = [f"STK{i:03d}" for i in range(n_stocks)]

    # ---- Planted characteristics (cross-sectional z-scores) ----
    mom   = rng.normal(0, 1, n_stocks)   # momentum characteristic
    size  = rng.normal(0, 1, n_stocks)   # log market-cap characteristic
    ep    = rng.normal(0, 1, n_stocks)   # earnings-yield characteristic
    vol   = rng.normal(0, 1, n_stocks)   # volatility characteristic
    illiq = rng.normal(0, 1, n_stocks)   # illiquidity characteristic

    # ---- Planted monthly alpha (~2.5% cross-sectional std / month) ----
    z = (0.30 * mom + 0.25 * ep - 0.20 * vol + 0.15 * illiq - 0.20 * size)
    z = (z - z.mean()) / z.std()
    alpha_monthly = 0.025 * z                    # ~ 30% annualized long-short spread
    alpha_daily = alpha_monthly / 21.0

    # ---- Common market factor & idiosyncratic risk ----
    mkt = rng.normal(0.0004, 0.009, n_days)
    # idiosyncratic vol is itself a function of the vol characteristic so that
    # realized-volatility factors carry signal (low-vol anomaly).
    idio_vol = 0.010 * np.clip(1.3 + vol, 0.5, None)
    idio = rng.normal(0.0, 1.0, (n_days, n_stocks)) * idio_vol[None, :]

    daily_ret = alpha_daily[None, :] + mkt[:, None] + idio

    # ---- Price panel ----
    price = 100.0 * np.exp(np.cumsum(daily_ret, axis=0))
    price = pd.DataFrame(price, index=dates, columns=columns)

    # ---- Volume panel (higher illiquidity  ->  lower volume) ----
    base_vol = 1e6 * np.exp(-0.4 * illiq)
    volume = pd.DataFrame(
        np.abs(rng.normal(1.0, 0.3, (n_days, n_stocks))) * base_vol[None, :],
        index=dates, columns=columns,
    )

    # ---- Market-cap panel (size characteristic) ----
    base_cap = 1e9 * np.exp(1.0 * size)
    market_cap = pd.DataFrame(
        base_cap[None, :] * (price / price.iloc[0]).values,
        index=dates, columns=columns,
    )

    # ---- Earnings-yield panel (value characteristic + noise) ----
    ep_panel = pd.DataFrame(
        (0.05 + 0.02 * ep)[None, :] + rng.normal(0, 0.005, (n_days, n_stocks)),
        index=dates, columns=columns,
    ).clip(lower=0.001)

    # ---- Planted noise features for the factor-mining stage ----
    # Random-walk cross-sectional scores: they look like "real" features but
    # carry zero predictive power, so the BHY filter should reject them.
    noise_a = pd.DataFrame(
        np.cumsum(rng.normal(0, 1, (n_days, n_stocks)), axis=0),
        index=dates, columns=columns,
    )
    noise_b = pd.DataFrame(
        np.cumsum(rng.normal(0, 1, (n_days, n_stocks)), axis=0),
        index=dates, columns=columns,
    )

    returns = price.pct_change()

    return {
        "prices": price,
        "returns": returns,
        "volume": volume,
        "market_cap": market_cap,
        "ep": ep_panel,
        "noise_a": noise_a,
        "noise_b": noise_b,
        "true_characteristics": pd.DataFrame(
            {"momentum": mom, "size": size, "value_ep": ep,
             "volatility": vol, "illiquidity": illiq},
            index=columns,
        ),
    }
