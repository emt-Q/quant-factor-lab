"""
Real-market data adapter (Yahoo Finance via yfinance).

Produces the same panel schema the pipeline consumes (prices / returns /
volume / market_cap / ep), so the whole factor research flow - factor
construction, IC & FDR tests, composite, out-of-sample backtest - runs
unchanged on real data.

Field construction:
  prices       = adjusted close          (yfinance bulk download)
  volume       = raw volume              (yfinance bulk download)
  market_cap   = adjusted close x shares outstanding (static share count;
                 time variation enters only through price - documented)
  ep           = 1 / trailing P/E        (static per ticker; NaN if missing)
  returns      = pct_change of adjusted close

Every fetch is cached to disk (per-field CSVs under a cache dir), so
re-runs are offline, fast and reproducible after the first download.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd


DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO",
    "JPM", "BRK-B", "V", "MA", "UNH", "LLY", "XOM", "ORCL", "HD", "PG",
    "COST", "KO",
]
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2024-12-31"


# --------------------------------------------------------------------------
# Yahoo download & per-ticker fundamentals
# --------------------------------------------------------------------------

def _yf():
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "yfinance is required for the real-data adapter. "
            "Install it with: pip install -r requirements-data.txt"
        ) from exc
    return yf


def _extract_series(panel: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
    """Pull one field out of a yfinance download (handles MultiIndex)."""
    if isinstance(panel.columns, pd.MultiIndex):
        cols = [(t, field) for t in tickers if (t, field) in panel.columns]
        if not cols:
            return pd.DataFrame(index=panel.index)
        out = panel[cols]
        out.columns = [t for t, _ in cols]
    else:  # single ticker collapses the index
        out = panel[[field]].rename(columns={field: tickers[0]})
    return out


def _fundamentals(tickers: list[str]) -> tuple[pd.Series, pd.Series]:
    """Shares outstanding and trailing-PE per ticker (NaN when unavailable)."""
    yf = _yf()
    shares, eps = {}, {}
    for t in tickers:
        sh, pe = np.nan, np.nan
        try:
            info = yf.Ticker(t).info
            sh = info.get("sharesOutstanding") or info.get("shares")
            pe = info.get("trailingPE")
        except Exception:
            try:  # lighter fallback for share count
                sh = yf.Ticker(t).fast_info.get("shares")
            except Exception:
                pass
        shares[t] = sh
        eps[t] = 1.0 / pe if pe and pe > 0 else np.nan
        time.sleep(0.15)  # be gentle with Yahoo rate limits
    return pd.Series(shares), pd.Series(eps)


# --------------------------------------------------------------------------
# Main entry points
# --------------------------------------------------------------------------

def fetch_yahoo_panel(tickers: list[str] | None = None,
                      start: str = DEFAULT_START,
                      end: str = DEFAULT_END,
                      cache_dir: str | None = None) -> dict:
    """Download a real panel from Yahoo Finance and (optionally) cache it."""
    tickers = list(tickers or DEFAULT_TICKERS)
    yf = _yf()

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True,
                      group_by="ticker", threads=True, progress=False)
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no data for the requested "
                           "tickers/range - check tickers and network access.")

    prices = _extract_series(raw, tickers, "Close").sort_index()
    volume = _extract_series(raw, tickers, "Volume").sort_index()
    if prices.isna().all().all():
        raise RuntimeError("No adjusted-close data was returned.")

    shares, ep_static = _fundamentals(tickers)
    market_cap = prices.mul(shares, axis=1)
    ep = pd.DataFrame(np.tile(ep_static.values, (len(prices), 1)),
                      index=prices.index, columns=prices.columns)

    panel = {
        "prices": prices,
        "returns": prices.pct_change(),
        "volume": volume,
        "market_cap": market_cap,
        "ep": ep,
        "shares": shares,
        "source": "yahoo",
        "tickers": tickers,
    }

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        prices.to_csv(os.path.join(cache_dir, "prices.csv"))
        volume.to_csv(os.path.join(cache_dir, "volume.csv"))
        pd.DataFrame({"shares": shares, "ep": ep_static}).to_csv(
            os.path.join(cache_dir, "fundamentals.csv"))
    return panel


def load_yahoo_panel(cache_dir: str) -> dict:
    """Load a previously cached real panel (offline, fast)."""
    prices = pd.read_csv(os.path.join(cache_dir, "prices.csv"), index_col=0,
                         parse_dates=True)
    volume = pd.read_csv(os.path.join(cache_dir, "volume.csv"), index_col=0,
                         parse_dates=True)
    fund = pd.read_csv(os.path.join(cache_dir, "fundamentals.csv"), index_col=0)
    shares = fund["shares"]
    ep_static = fund["ep"]
    market_cap = prices.mul(shares, axis=1)
    ep = pd.DataFrame(np.tile(ep_static.values, (len(prices), 1)),
                      index=prices.index, columns=prices.columns)
    return {
        "prices": prices,
        "returns": prices.pct_change(),
        "volume": volume,
        "market_cap": market_cap,
        "ep": ep,
        "shares": shares,
        "source": "yahoo-cache",
        "tickers": list(prices.columns),
    }


def get_real_panel(cache_dir: str | None = None,
                   tickers: list[str] | None = None,
                   start: str = DEFAULT_START,
                   end: str = DEFAULT_END,
                   refresh: bool = False) -> dict:
    """Use cache when present, otherwise download; `refresh` forces a fetch."""
    if cache_dir and not refresh and all(
            os.path.exists(os.path.join(cache_dir, f))
            for f in ("prices.csv", "volume.csv", "fundamentals.csv")):
        print(f"[data] loading cached real panel from {cache_dir}")
        return load_yahoo_panel(cache_dir)
    print(f"[data] downloading {len(tickers or DEFAULT_TICKERS)} tickers "
          f"{start}..{end} from Yahoo Finance")
    return fetch_yahoo_panel(tickers=tickers, start=start, end=end,
                             cache_dir=cache_dir)
