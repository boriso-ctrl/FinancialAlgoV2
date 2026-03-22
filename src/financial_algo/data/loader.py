"""Price data loader using yfinance with local CSV caching."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yfinance as yf


_DEFAULT_CACHE_DIR = Path.home() / ".financial_algo_cache"


def load_prices(
    tickers: list[str],
    start: str = "2010-01-01",
    end: str | None = None,
    field: str = "Close",
    cache_dir: Path | str | None = _DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Download adjusted daily prices via yfinance, with optional CSV caching.

    Parameters
    ----------
    tickers:
        List of yfinance-compatible ticker symbols.
    start:
        Start date string (ISO format, e.g. ``"2010-01-01"``).
    end:
        End date string. ``None`` means today.
    field:
        OHLCV column to extract (default ``"Close"``).
    cache_dir:
        Directory to store cached CSV files. ``None`` disables caching.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ``Date`` with one column per ticker.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / _cache_key(tickers, start, end, field)
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            # Verify all requested tickers are present
            if set(tickers).issubset(set(df.columns)):
                return df[tickers]
    else:
        cache_path = None

    df = _download(tickers, start, end, field)

    if cache_path is not None:
        df.to_csv(cache_path)

    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _download(
    tickers: list[str],
    start: str,
    end: str | None,
    field: str,
) -> pd.DataFrame:
    """Download from yfinance and return a clean DataFrame."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(f"No data returned for tickers: {tickers}")

    # yf.download returns MultiIndex columns for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw[field].copy()
    else:
        # Single ticker — columns are just OHLCV strings
        df = raw[[field]].copy()
        df.columns = tickers[:1]

    df.index.name = "Date"
    # Forward-fill then back-fill small gaps (weekends/holidays already aligned)
    df = df.ffill().bfill()
    return df


def _cache_key(
    tickers: list[str],
    start: str,
    end: str | None,
    field: str,
) -> str:
    """Deterministic filename for the cache entry."""
    token = f"{sorted(tickers)}|{start}|{end}|{field}"
    h = hashlib.sha256(token.encode()).hexdigest()[:16]
    return f"prices_{h}.csv"
