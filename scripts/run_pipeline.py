#!/usr/bin/env python3
"""Run the factor research pipeline end-to-end.

Synthetic panel (default):
    python scripts/run_pipeline.py

Real market data (Yahoo Finance via yfinance, cached after first fetch):
    python scripts/run_pipeline.py --real
    python scripts/run_pipeline.py --real --tickers AAPL,MSFT,NVDA \
        --start 2019-01-01 --end 2024-12-31 --cache data/yahoo_cache
    python scripts/run_pipeline.py --real --refresh     # force re-download
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_factor_lab.pipeline import run  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Quant Factor Lab pipeline")
    p.add_argument("--real", action="store_true",
                   help="run on real market data instead of the synthetic panel")
    p.add_argument("--tickers",
                   default="AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,AVGO,JPM,"
                           "BRK-B,V,MA,UNH,LLY,XOM,ORCL,HD,PG,COST,KO",
                   help="comma-separated ticker list (with --real)")
    p.add_argument("--start", default="2019-01-01", help="start date (with --real)")
    p.add_argument("--end", default="2024-12-31", help="end date (with --real)")
    p.add_argument("--cache", default="data/yahoo_cache",
                   help="cache directory for the real panel (with --real)")
    p.add_argument("--refresh", action="store_true",
                   help="ignore cache and re-download (with --real)")
    args = p.parse_args()

    if args.real:
        from quant_factor_lab.data_adapter import get_real_panel
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        panel = get_real_panel(cache_dir=args.cache, tickers=tickers,
                               start=args.start, end=args.end,
                               refresh=args.refresh)
        run(panel=panel, out_prefix="real_")
    else:
        run()


if __name__ == "__main__":
    main()
