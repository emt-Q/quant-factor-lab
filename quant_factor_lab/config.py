"""Central configuration for the factor research pipeline.

All knobs are kept in one place so the pipeline can be re-run with different
universe sizes, horizons, cost assumptions, or FDR targets.
"""
from __future__ import annotations

# ---- Universe & sample ----
N_STOCKS = 120               # number of stocks in the synthetic universe
N_DAYS = 1008                # ~4 years of daily bars

# ---- Research protocol ----
TRAIN_SPLIT = 0.75           # fraction of rebalance dates used in-sample
REBAL_PERIOD = 20            # rebalance every 20 trading days (~monthly)
MIN_HISTORY = 300            # min days of history required for 12-1 momentum
TOP_QUANTILE = 0.20          # long/short quintile portfolios

# ---- Costs & statistics ----
COST_BPS = 10.0              # one-way transaction cost (bps)
FDR_Q = 0.05                 # target false-discovery rate for BHY correction

# ---- Reproducibility ----
SEED = 42
