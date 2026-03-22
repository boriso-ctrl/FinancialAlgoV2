"""Intraday feature extractor and daily feature store.

Takes 1-minute OHLCV bars (from Alpaca) and extracts a set of
microstructure-derived signals that are aggregated to end-of-day values.
The result is a multi-ticker feature DataFrame that can be fed into
any ML strategy alongside the existing daily price features.

Feature definitions
-------------------
1.  realized_vol_1min  -- 1-min realized volatility (annualised, 21-day roll)
2.  vwap_deviation     -- (Close - VWAP) / VWAP, end-of-day z-score vs 60 days
3.  opening_gap        -- (Open_t - Close_{t-1}) / Close_{t-1}
4.  overnight_ret      -- same as opening_gap (alias used by some lit)
5.  intraday_range     -- (High - Low) / Open, normalised daily range
6.  vol_of_vol         -- 5-day rolling std of realized_vol_1min
7.  volume_surprise    -- Volume_t / rolling_60_mean_volume - 1
8.  close_to_high      -- (High - Close) / (High - Low), how close to daily high

All features are NaN-safe, purely look-back, and stored as a daily
DataFrame indexed by date with per-ticker column pairs:
    "{ticker}_{feature_name}"

Storage format
--------------
~/.financial_algo_cache/alpaca/features/
    {ticker}_{start}_{end}.csv.gz   -- per-ticker feature CSVs

Usage
-----
    from financial_algo.data.feature_store import (
        FeatureStore, load_feature_store,
    )

    # First run — downloads and caches everything (~5 min for 42 tickers)
    store = FeatureStore(tickers=["SPY", "QQQ"], start="2020-01-01")
    store.build()         # fetches 1-min bars, extracts, saves to cache
    df = store.load()     # returns wide daily DataFrame

    # Subsequent runs — loads from cache instantly
    df = load_feature_store(tickers=["SPY", "QQQ"], start="2020-01-01")
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from financial_algo.data.alpaca_loader import load_intraday

_FEATURES_DIR = Path.home() / ".financial_algo_cache" / "alpaca" / "features"

# Feature names produced per ticker
FEATURE_NAMES = [
    "realized_vol_1min",
    "vwap_deviation",
    "opening_gap",
    "intraday_range",
    "vol_of_vol",
    "volume_surprise",
    "close_to_high",
]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def load_feature_store(
    tickers: list[str],
    start: str = "2020-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load the daily feature store, downloading from Alpaca if needed.

    Returns a date-indexed DataFrame with columns:
        "{ticker}_{feature}" for each ticker and each feature in
        FEATURE_NAMES.

    Tickers for which data is unavailable (e.g. crypto, VIX) are silently
    skipped — their feature columns will be absent from the result.
    """
    store = FeatureStore(tickers=tickers, start=start, end=end)
    return store.load(force_refresh=force_refresh)


class FeatureStore:
    """Build and cache per-ticker intraday feature DataFrames."""

    def __init__(
        self,
        tickers: list[str],
        start: str = "2020-01-01",
        end: str | None = None,
        cache_dir: Path | str = _FEATURES_DIR,
    ) -> None:
        self.tickers = tickers
        self.start = start
        self.end = end
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build(self, force_refresh: bool = False) -> pd.DataFrame:
        """Download 1-min bars and compute features. Caches the result."""
        # Filter to equity tickers only (Alpaca does not support VIX/crypto)
        equity_tickers = _equity_only(self.tickers)
        if not equity_tickers:
            warnings.warn("No equity tickers found — feature store is empty.", stacklevel=2)
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []

        for ticker in equity_tickers:
            cache_path = self._cache_path(ticker)
            if cache_path.exists() and not force_refresh:
                try:
                    df_feat = pd.read_csv(
                        cache_path, index_col=0, parse_dates=True, compression="gzip"
                    )
                    frames.append(df_feat)
                    continue
                except Exception:
                    cache_path.unlink(missing_ok=True)

            print(f"  [FeatureStore] Downloading 1-min bars: {ticker} ...", flush=True)
            bars_dict = load_intraday(
                [ticker], start=self.start, end=self.end, cache_dir=None
            )
            if ticker not in bars_dict or bars_dict[ticker].empty:
                warnings.warn(f"No 1-min data for {ticker} — skipping.", stacklevel=2)
                continue

            bars = bars_dict[ticker]
            df_feat = _extract_features(bars, ticker)
            if df_feat.empty:
                continue

            df_feat.to_csv(cache_path, compression="gzip")
            frames.append(df_feat)

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, axis=1).sort_index()
        return combined

    # ------------------------------------------------------------------
    def load(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load features from cache, building if necessary."""
        equity_tickers = _equity_only(self.tickers)
        missing = [t for t in equity_tickers if not self._cache_path(t).exists()]

        if missing or force_refresh:
            return self.build(force_refresh=force_refresh)

        frames: list[pd.DataFrame] = []
        for ticker in equity_tickers:
            try:
                df = pd.read_csv(
                    self._cache_path(ticker),
                    index_col=0,
                    parse_dates=True,
                    compression="gzip",
                )
                frames.append(df)
            except Exception as exc:
                warnings.warn(f"Could not load cache for {ticker}: {exc}", stacklevel=2)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).sort_index()

    # ------------------------------------------------------------------
    def _cache_path(self, ticker: str) -> Path:
        end_tag = self.end or "now"
        return self.cache_dir / f"{ticker}_{self.start}_{end_tag}.csv.gz"


# ---------------------------------------------------------------------------
# Feature extraction (pure pandas/numpy, vectorized, NaN-safe)
# ---------------------------------------------------------------------------

def _extract_features(bars: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Compute daily microstructure features from 1-min OHLCV bars.

    Parameters
    ----------
    bars:
        DataFrame with columns [open, high, low, close, volume, vwap,
        trade_count] indexed by UTC timestamp (1-min frequency).
    ticker:
        Ticker symbol — used to prefix column names in the output.

    Returns
    -------
    pd.DataFrame
        Date-indexed (local date, not UTC timestamp) DataFrame with
        columns "{ticker}_{feature}" for each feature in FEATURE_NAMES.
    """
    if bars.empty:
        return pd.DataFrame()

    # Localise to US/Eastern then extract date
    bars = bars.copy()
    bars.index = bars.index.tz_convert("America/New_York")
    bars["_date"] = bars.index.normalize()  # midnight in NYC

    # Keep only regular session: 09:30 to 16:00 Eastern
    time_mask = (
        (bars.index.time >= pd.Timestamp("09:30").time())
        & (bars.index.time <= pd.Timestamp("16:00").time())
    )
    session = bars.loc[time_mask].copy()
    if session.empty:
        return pd.DataFrame()

    # --- Daily aggregates from 1-min data ---
    daily_close = session.groupby("_date")["close"].last()
    daily_open = session.groupby("_date")["open"].first()
    daily_high = session.groupby("_date")["high"].max()
    daily_low = session.groupby("_date")["low"].min()
    daily_volume = session.groupby("_date")["volume"].sum().replace(0, np.nan)
    daily_vwap = _compute_daily_vwap(session)

    # --- 1. realized_vol_1min (annualised) ---
    session["_1min_ret"] = (
        session["close"].pct_change().fillna(0.0)
    )
    daily_rv = (
        session.groupby("_date")["_1min_ret"]
        .apply(lambda x: np.sqrt((x ** 2).sum()) * np.sqrt(252 * 390))
    )
    rv_21 = daily_rv.rolling(21, min_periods=5).mean()

    # --- 2. vwap_deviation (z-scored over 60 days) ---
    raw_dev = (daily_close - daily_vwap) / daily_vwap.replace(0, np.nan)
    raw_dev = raw_dev.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    roll_mean = raw_dev.rolling(60, min_periods=10).mean()
    roll_std = raw_dev.rolling(60, min_periods=10).std().replace(0, np.nan)
    vwap_dev_z = ((raw_dev - roll_mean) / roll_std).fillna(0.0)

    # --- 3. opening_gap ---
    prev_close = daily_close.shift(1)
    opening_gap = (daily_open - prev_close) / prev_close.replace(0, np.nan)
    opening_gap = opening_gap.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- 4. intraday_range ---
    intraday_range = (daily_high - daily_low) / daily_open.replace(0, np.nan)
    intraday_range = intraday_range.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- 5. vol_of_vol (5-day rolling std of realized_vol) ---
    vol_of_vol = daily_rv.rolling(5, min_periods=2).std().fillna(0.0)

    # --- 6. volume_surprise ---
    vol_roll_mean = daily_volume.rolling(60, min_periods=10).mean().replace(0, np.nan)
    volume_surprise = (daily_volume / vol_roll_mean) - 1.0
    volume_surprise = volume_surprise.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # --- 7. close_to_high ---
    hl_range = (daily_high - daily_low).replace(0, np.nan)
    close_to_high = (daily_high - daily_close) / hl_range
    close_to_high = close_to_high.replace([np.inf, -np.inf], np.nan).fillna(0.5)

    out = pd.DataFrame(
        {
            f"{ticker}_realized_vol_1min": rv_21,
            f"{ticker}_vwap_deviation": vwap_dev_z,
            f"{ticker}_opening_gap": opening_gap,
            f"{ticker}_intraday_range": intraday_range,
            f"{ticker}_vol_of_vol": vol_of_vol,
            f"{ticker}_volume_surprise": volume_surprise,
            f"{ticker}_close_to_high": close_to_high,
        }
    )
    # Convert index to plain date for merge compatibility with daily prices
    out.index = pd.DatetimeIndex(out.index).normalize().tz_localize(None)
    out.index.name = "Date"
    return out.sort_index()


def _compute_daily_vwap(session: pd.DataFrame) -> pd.Series:
    """Compute dollar-VWAP per day from 1-min bars.

    Falls back to raw VWAP column if present (Alpaca provides it).
    If the column is missing or all-NaN, computes from close * volume.
    """
    # Prefer Alpaca's built-in vwap if populated
    if "vwap" in session.columns:
        vwap_last = session.groupby("_date")["vwap"].last()
        if vwap_last.notna().mean() > 0.8:
            return vwap_last

    # Fallback: dollar-weighted average
    session = session.copy()
    session["_dolvol"] = session["close"] * session["volume"].replace(0, np.nan)
    dolvol_sum = session.groupby("_date")["_dolvol"].sum()
    vol_sum = session.groupby("_date")["volume"].sum().replace(0, np.nan)
    vwap = (dolvol_sum / vol_sum).replace([np.inf, -np.inf], np.nan).fillna(
        session.groupby("_date")["close"].last()
    )
    return vwap


# ---------------------------------------------------------------------------
# Tickers that Alpaca supports (exclude VIX, ETH, BTC)
# ---------------------------------------------------------------------------

_NON_EQUITY = {"^VIX", "BTC-USD", "ETH-USD"}


def _equity_only(tickers: list[str]) -> list[str]:
    return [t for t in tickers if t not in _NON_EQUITY and not t.startswith("^")]
