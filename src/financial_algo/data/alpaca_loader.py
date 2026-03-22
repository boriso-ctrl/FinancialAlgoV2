"""Alpaca Markets intraday data loader with CSV caching.

Fetches historical 1-minute OHLCV bars from the Alpaca Data API (free
tier / paid SIP feed) and caches them as gzip-compressed CSV files under
~/.financial_algo_cache/alpaca/.

API credentials are read exclusively from environment variables so they
are never hard-coded in source:

    ALPACA_API_KEY   -- your Alpaca API key ID
    ALPACA_SECRET    -- your Alpaca secret key

Set these in a .env file or your shell profile before running any script
that uses this module.

Usage
-----
    from financial_algo.data.alpaca_loader import load_intraday, load_daily_alpaca

    # 1-minute bars for the last 5 years
    bars = load_intraday(["SPY", "QQQ"], start="2020-01-01")

    # Daily bars (useful for cross-validation with yfinance)
    daily = load_daily_alpaca(["SPY"], start="2020-01-01")
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import pandas as pd

_CACHE_ROOT = Path.home() / ".financial_algo_cache" / "alpaca"
_MINUTE_SUBDIR = _CACHE_ROOT / "1min"
_DAILY_SUBDIR = _CACHE_ROOT / "daily"

# Alpaca free-tier keeps ~5 years of 1-min history (IEX source).
# SIP feed (paid) has full depth. Both use the same API.
_DEFAULT_FEED: Literal["iex", "sip"] = "iex"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_intraday(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str | None = None,
    feed: str = _DEFAULT_FEED,
    cache_dir: Path | str | None = _MINUTE_SUBDIR,
    force_refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load 1-minute OHLCV bars for each ticker.

    Parameters
    ----------
    tickers:
        List of equity ticker symbols (Alpaca format, e.g. "SPY").
        Note: crypto and ^VIX are not supported by this loader.
    start:
        ISO date string for the earliest bar to fetch.
    end:
        ISO date string for the last bar (defaults to yesterday).
    feed:
        ``"iex"`` (free, ~5yr, US equities) or ``"sip"`` (paid, full SIP).
    cache_dir:
        Directory to store per-ticker gzip CSV caches.  ``None`` disables
        caching.
    force_refresh:
        If ``True``, ignore existing cache and re-download.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping ticker -> DataFrame with columns [open, high, low, close,
        volume, vwap, trade_count] indexed by UTC timestamp.
    """
    client = _get_client()
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    result: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = _load_ticker_bars(
            client, ticker, start, end, "1Min", feed, cache_dir, force_refresh
        )
        if df is not None and not df.empty:
            result[ticker] = df

    return result


def load_daily_alpaca(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    feed: str = _DEFAULT_FEED,
    cache_dir: Path | str | None = _DAILY_SUBDIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load daily OHLCV bars and return a Close-price DataFrame.

    Useful for cross-validation against yfinance daily data.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with one column per ticker (Close prices).
    """
    client = _get_client()
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    frames: dict[str, pd.Series] = {}
    for ticker in tickers:
        df = _load_ticker_bars(
            client, ticker, start, end, "1Day", feed, cache_dir, force_refresh
        )
        if df is not None and not df.empty:
            frames[ticker] = df["close"].rename(ticker)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames.values(), axis=1).sort_index()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client():
    """Construct StockHistoricalDataClient from environment variables."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
    except ImportError as exc:
        raise ImportError(
            "alpaca-py is not installed. Run: uv pip install alpaca-py>=0.30"
        ) from exc

    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET", "")

    if not api_key or not secret:
        warnings.warn(
            "ALPACA_API_KEY and/or ALPACA_SECRET not set. "
            "Set them as environment variables before using alpaca_loader. "
            "Attempting unauthenticated access (limited rate).",
            stacklevel=3,
        )
        return StockHistoricalDataClient(api_key=None, secret_key=None)

    return StockHistoricalDataClient(api_key=api_key, secret_key=secret)


def _load_ticker_bars(
    client,
    ticker: str,
    start: str,
    end: str,
    timeframe_str: str,
    feed: str,
    cache_dir: Path | str | None,
    force_refresh: bool,
) -> pd.DataFrame | None:
    """Load bars for a single ticker, using cache if available."""
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{ticker}_{start}_{end}_{timeframe_str}.csv.gz"
        cache_path = cache_dir / fname
        if cache_path.exists() and not force_refresh:
            try:
                return pd.read_csv(
                    cache_path, index_col=0, parse_dates=True, compression="gzip"
                )
            except Exception:
                cache_path.unlink(missing_ok=True)
    else:
        cache_path = None

    df = _fetch_bars(client, ticker, start, end, timeframe_str, feed)

    if df is not None and cache_path is not None:
        df.to_csv(cache_path, compression="gzip")

    return df


def _fetch_bars(
    client,
    ticker: str,
    start: str,
    end: str,
    timeframe_str: str,
    feed: str,
) -> pd.DataFrame | None:
    """Download bars from Alpaca API and return a clean DataFrame."""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError as exc:
        raise ImportError("alpaca-py not installed") from exc

    _TF_MAP = {
        "1Min": TimeFrame(1, TimeFrameUnit.Minute),
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
        "1Day": TimeFrame(1, TimeFrameUnit.Day),
    }
    if timeframe_str not in _TF_MAP:
        raise ValueError(f"Unsupported timeframe: {timeframe_str}")

    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=_TF_MAP[timeframe_str],
        start=start,
        end=end,
        feed=feed,
        adjustment="all",
    )

    try:
        bars = client.get_stock_bars(request)
        df = bars.df
    except Exception as exc:
        warnings.warn(f"Failed to fetch {ticker} {timeframe_str}: {exc}", stacklevel=4)
        return None

    if df is None or df.empty:
        return None

    # alpaca-py returns MultiIndex (symbol, timestamp) — drop symbol level
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "timestamp"
    df.columns = df.columns.str.lower()

    # Ensure expected columns present, fill missing with NaN
    for col in ("open", "high", "low", "close", "volume", "vwap", "trade_count"):
        if col not in df.columns:
            df[col] = float("nan")

    return df[["open", "high", "low", "close", "volume", "vwap", "trade_count"]]


# ---------------------------------------------------------------------------
# Utility: check available history length without downloading everything
# ---------------------------------------------------------------------------

def get_available_history(
    ticker: str = "SPY",
    feed: str = _DEFAULT_FEED,
) -> tuple[str | None, str | None]:
    """Return (earliest_date, latest_date) of available 1-min history.

    Quick probe — fetches only the first and last day to check data bounds.
    """
    client = _get_client()
    df = _fetch_bars(client, ticker, "2015-01-01", "2015-01-05", "1Min", feed)
    if df is None or df.empty:
        df = _fetch_bars(client, ticker, "2019-01-01", "2019-01-05", "1Min", feed)

    earliest = df.index.min().date().isoformat() if df is not None and not df.empty else None

    df_recent = _fetch_bars(
        client, ticker,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "1Min", feed,
    )
    latest = (
        df_recent.index.max().date().isoformat()
        if df_recent is not None and not df_recent.empty else None
    )

    return earliest, latest
