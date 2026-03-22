"""News and sentiment data feeds — free, no-API-key sources.

Data sources (all free, no auth required):
    - GDELT Project: Global event database with tone/sentiment scores
    - Federal Reserve FRED: Economic indicators (fear proxies)
    - Yahoo Finance news: Headline scraping for ticker-specific sentiment
    - RSS feeds: Major financial news outlets

For backtesting, uses GDELT's historical archive (2015+) and cached data.
For live trading, fetches real-time feeds.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from urllib.error import URLError

import pandas as pd


_DEFAULT_CACHE_DIR = Path.home() / ".financial_algo_cache" / "sentiment"

# GDELT GKG (Global Knowledge Graph) — free, no API key
_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# FRED — free, no API key for basic series
_FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    """A single news article with metadata."""

    title: str
    source: str
    published: datetime
    url: str
    tone: float = 0.0          # sentiment score (-10 to +10)
    relevance: float = 1.0     # relevance to query (0 to 1)
    themes: list[str] = field(default_factory=list)


@dataclass
class SentimentSnapshot:
    """Aggregated sentiment for a single day."""

    date: pd.Timestamp
    score: float               # mean tone (-10 to +10)
    volume: int                # number of articles
    positive_pct: float        # % of articles with positive tone
    negative_pct: float        # % of articles with negative tone
    max_negative: float        # most negative tone (severity)
    themes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# GDELT fetcher
# ---------------------------------------------------------------------------

def fetch_gdelt_sentiment(
    query: str,
    start: str | None = None,
    end: str | None = None,
    max_records: int = 250,
    cache_dir: Path | str | None = _DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch news sentiment from GDELT's DOC API.

    Parameters
    ----------
    query:
        Search query (e.g. "oil crisis OPEC", "war Ukraine Russia").
    start, end:
        Date range as ISO strings. GDELT has data from ~2015+.
    max_records:
        Maximum number of articles to fetch per request.
    cache_dir:
        Where to cache results. None disables caching.

    Returns
    -------
    pd.DataFrame
        Daily sentiment with columns: date, score, volume,
        positive_pct, negative_pct, max_negative.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(f"gdelt_{query}_{start}_{end}_{max_records}")
        cache_path = cache_dir / f"{key}.csv"
        if cache_path.exists():
            return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    params = {
        "query": quote_plus(query),
        "mode": "timelinetone",
        "format": "csv",
        "maxrecords": str(max_records),
    }
    if start:
        params["startdatetime"] = start.replace("-", "") + "000000"
    if end:
        params["enddatetime"] = end.replace("-", "") + "000000"

    url = _GDELT_DOC_API + "?" + "&".join(f"{k}={v}" for k, v in params.items())

    try:
        df = _safe_fetch_csv(url)
    except Exception:
        # GDELT down or blocked — return empty
        return _empty_sentiment_df()

    if df.empty:
        return _empty_sentiment_df()

    # GDELT timeline returns: date, tone (average)
    df.columns = ["date", "score"]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["volume"] = 1  # timeline mode doesn't give volume per row
    df["positive_pct"] = (df["score"] > 0).astype(float)
    df["negative_pct"] = (df["score"] < 0).astype(float)
    df["max_negative"] = df["score"].clip(upper=0)

    # Resample to daily
    daily = df.resample("D").agg({
        "score": "mean",
        "volume": "sum",
        "positive_pct": "mean",
        "negative_pct": "mean",
        "max_negative": "min",
    }).dropna(subset=["score"])

    if cache_path is not None:
        daily.to_csv(cache_path)

    return daily


# ---------------------------------------------------------------------------
# FRED economic indicators (free, no API key)
# ---------------------------------------------------------------------------

def fetch_fred_series(
    series_id: str,
    start: str = "2010-01-01",
    end: str | None = None,
    cache_dir: Path | str | None = _DEFAULT_CACHE_DIR,
) -> pd.Series:
    """Fetch an economic series from FRED.

    Useful series for crisis detection:
        - VIXCLS: VIX close
        - TEDRATE: TED spread (interbank stress)
        - T10Y2Y: 10Y-2Y Treasury spread (recession signal)
        - BAMLH0A0HYM2: High-yield spread (credit stress)
        - UMCSENT: U Michigan consumer sentiment
        - DTWEXBGS: Trade-weighted USD (flight to safety)
        - DCOILWTICO: WTI crude oil spot

    Parameters
    ----------
    series_id:
        FRED series identifier.
    start, end:
        Date range.
    cache_dir:
        Cache directory.

    Returns
    -------
    pd.Series
        Daily values, forward-filled for weekends/holidays.
    """
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(f"fred_{series_id}_{start}_{end}")
        cache_path = cache_dir / f"{key}.csv"
        if cache_path.exists():
            s = pd.read_csv(cache_path, index_col=0, parse_dates=True).squeeze("columns")
            return s

    end_str = end or datetime.now().strftime("%Y-%m-%d")
    url = (
        f"{_FRED_BASE}?id={series_id}"
        f"&cosd={start}&coed={end_str}&fmt=csv"
    )

    try:
        df = _safe_fetch_csv(url)
    except Exception:
        return pd.Series(dtype=float, name=series_id)

    if df.empty or len(df.columns) < 2:
        return pd.Series(dtype=float, name=series_id)

    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().set_index("date")
    s = df["value"].rename(series_id)

    if cache_path is not None:
        s.to_csv(cache_path)

    return s


# ---------------------------------------------------------------------------
# Multi-query sentiment aggregator
# ---------------------------------------------------------------------------

# Pre-defined query sets for crisis detection
CRISIS_QUERIES: dict[str, list[str]] = {
    "oil_crisis": [
        "oil crisis OPEC supply shock",
        "oil price crash energy",
        "crude oil embargo sanctions",
    ],
    "war_crisis": [
        "war military conflict escalation",
        "geopolitical tension invaded attack",
        "sanctions weapons defense escalation",
    ],
    "financial_crisis": [
        "financial crisis bank collapse recession",
        "credit crunch liquidity default",
        "stock market crash panic sell",
    ],
    "pandemic": [
        "pandemic outbreak virus lockdown",
        "COVID health emergency quarantine",
    ],
    "general_fear": [
        "economic recession fear uncertainty",
        "market volatility panic selling",
    ],
}


def fetch_crisis_sentiment(
    category: str = "general_fear",
    start: str | None = None,
    end: str | None = None,
    cache_dir: Path | str | None = _DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Fetch and combine sentiment for a crisis category.

    Parameters
    ----------
    category:
        Key from ``CRISIS_QUERIES``.
    start, end:
        Date range.
    cache_dir:
        Cache directory.

    Returns
    -------
    pd.DataFrame
        Combined daily sentiment across all queries in the category.
    """
    queries = CRISIS_QUERIES.get(category, CRISIS_QUERIES["general_fear"])

    dfs = []
    for q in queries:
        df = fetch_gdelt_sentiment(q, start=start, end=end, cache_dir=cache_dir)
        if not df.empty:
            dfs.append(df)
        time.sleep(0.5)  # rate limiting

    if not dfs:
        return _empty_sentiment_df()

    combined = pd.concat(dfs)
    daily = combined.resample("D").agg({
        "score": "mean",
        "volume": "sum",
        "positive_pct": "mean",
        "negative_pct": "mean",
        "max_negative": "min",
    }).dropna(subset=["score"])

    return daily


# ---------------------------------------------------------------------------
# Synthetic sentiment for backtesting (no API dependency)
# ---------------------------------------------------------------------------

def build_synthetic_sentiment(
    prices: pd.DataFrame,
    vix: pd.Series | None = None,
    lookback: int = 20,
) -> pd.DataFrame:
    """Build proxy sentiment from price/vol data for backtest use.

    Combines VIX level, realized vol, put-call-like proxy, credit spreads,
    and momentum breadth into a synthetic sentiment score. This does NOT
    use future data — all signals are trailing.

    Parameters
    ----------
    prices:
        Price DataFrame (must include SPY; optionally HYG, LQD, GLD, TLT).
    vix:
        VIX series. If None, approximated from SPY realized vol.
    lookback:
        Rolling window for sentiment computation.

    Returns
    -------
    pd.DataFrame
        Columns: sentiment_score, fear_index, greed_index, news_velocity_proxy.
    """
    from financial_algo.indicators import realized_vol, zscore

    idx = prices.index
    result = pd.DataFrame(index=idx)

    # --- VIX / vol component (fear) ---
    if vix is not None:
        vix_aligned = vix.reindex(idx).ffill()
    else:
        vix_aligned = realized_vol(prices["SPY"], lookback) * 100

    vix_z = zscore(vix_aligned, lookback * 3)
    result["vix_fear"] = vix_z.clip(-3, 3)

    # --- Credit stress component ---
    if "HYG" in prices.columns and "LQD" in prices.columns:
        credit_ratio = prices["HYG"] / prices["LQD"]
        credit_z = zscore(credit_ratio, lookback * 3)
        result["credit_stress"] = (-credit_z).clip(-3, 3)  # inverted: low ratio = high stress
    else:
        result["credit_stress"] = 0.0

    # --- Safe-haven demand (GLD + TLT relative to SPY) ---
    haven_score = pd.Series(0.0, index=idx)
    if "GLD" in prices.columns:
        gld_mom = prices["GLD"].pct_change(lookback)
        spy_mom = prices["SPY"].pct_change(lookback)
        haven_score += (gld_mom - spy_mom).clip(-0.3, 0.3) * 10
    if "TLT" in prices.columns:
        tlt_mom = prices["TLT"].pct_change(lookback)
        spy_mom = prices["SPY"].pct_change(lookback)
        haven_score += (tlt_mom - spy_mom).clip(-0.3, 0.3) * 10
    result["haven_demand"] = haven_score.clip(-3, 3)

    # --- Momentum breadth (market confidence) ---
    sector_cols = [c for c in prices.columns if c.startswith("XL")]
    if len(sector_cols) >= 3:
        sma = prices[sector_cols].rolling(lookback * 2).mean()
        above_sma = (prices[sector_cols] > sma).sum(axis=1) / len(sector_cols)
        result["breadth_confidence"] = (above_sma * 6 - 3).clip(-3, 3)  # scale to [-3, 3]
    else:
        result["breadth_confidence"] = 0.0

    # --- Price velocity / news proxy (large moves = high news activity) ---
    spy_ret = prices["SPY"].pct_change().abs()
    vol_of_vol = spy_ret.rolling(lookback).std()
    result["news_velocity_proxy"] = zscore(vol_of_vol, lookback * 3).clip(-3, 3)

    # --- Composite scores ---
    fear_components = result[["vix_fear", "credit_stress", "haven_demand"]].mean(axis=1)
    greed_components = -fear_components  # inverse

    result["fear_index"] = fear_components.clip(-3, 3)
    result["greed_index"] = greed_components.clip(-3, 3)

    # Sentiment score: negative = bearish/fear, positive = bullish/greed
    # Weighted: VIX (40%), credit (25%), haven (20%), breadth (15%)
    result["sentiment_score"] = (
        -0.40 * result["vix_fear"]
        + -0.25 * result["credit_stress"]
        + -0.20 * result["haven_demand"]
        + 0.15 * result["breadth_confidence"]
    ).clip(-5, 5)

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_fetch_csv(url: str, timeout: int = 15) -> pd.DataFrame:
    """Fetch a CSV from a URL with timeout and error handling."""
    req = Request(url, headers={"User-Agent": "FinancialAlgoV2/2.0"})
    with urlopen(req, timeout=timeout) as resp:
        import io
        data = resp.read().decode("utf-8", errors="replace")
        return pd.read_csv(io.StringIO(data))


def _cache_key(identifier: str) -> str:
    """Generate a stable hash key for caching."""
    return hashlib.md5(identifier.encode()).hexdigest()[:16]


def _empty_sentiment_df() -> pd.DataFrame:
    """Return an empty sentiment DataFrame with correct columns."""
    return pd.DataFrame(
        columns=["score", "volume", "positive_pct", "negative_pct", "max_negative"]
    )
