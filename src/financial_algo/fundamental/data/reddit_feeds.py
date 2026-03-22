"""Reddit sentiment data pipeline via PRAW.

Scrapes daily mention counts and sentiment per ticker from Reddit
subreddits (r/wallstreetbets, r/stocks, r/options).  Caches results
to ``~/.financial_algo_cache/reddit/`` for backtest reuse.

Data flow:
    PRAW API -> raw posts/comments -> per-ticker aggregation
    -> daily cache files -> feature extraction -> signal DataFrame

For backtesting without live Reddit access, provides
:func:`build_synthetic_reddit_sentiment` which uses price/vol proxies
to simulate reddit-like mention velocity and sentiment patterns.

Dependencies (optional — gracefully degrade if missing):
    - praw: Reddit API wrapper (pip install praw)
    - vaderSentiment: Lexicon-based sentiment (pip install vaderSentiment)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_DEFAULT_CACHE_DIR = Path.home() / ".financial_algo_cache" / "reddit"

# Subreddits ranked by signal quality for retail sentiment
SUBREDDITS = ("wallstreetbets", "stocks", "options")

# Tickers we track in these subreddits (subset of our universe that
# retail traders actually discuss — no bond ETFs or obscure tickers)
REDDIT_TICKERS = (
    "SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA", "AMD", "AMZN",
    "MSFT", "META", "GOOG", "NFLX", "GLD", "XLE", "EEM",
    "BTC", "ETH",  # crypto aliases used on reddit
)

# Regex pattern to match ticker mentions (uppercase 2-5 chars, word boundary)
_TICKER_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in REDDIT_TICKERS) + r")\b"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RedditMention:
    """A single ticker mention extracted from a Reddit post/comment."""

    ticker: str
    subreddit: str
    timestamp: datetime
    sentiment: float  # -1 to +1
    score: int  # upvotes - downvotes
    is_post: bool  # True = submission, False = comment


@dataclass
class DailyRedditSentiment:
    """Aggregated daily Reddit sentiment for one ticker."""

    date: pd.Timestamp
    ticker: str
    mention_count: int
    mean_sentiment: float
    bullish_pct: float  # fraction of mentions with sentiment > 0.05
    bearish_pct: float  # fraction with sentiment < -0.05
    weighted_sentiment: float  # upvote-weighted sentiment
    total_score: int  # sum of upvotes across mentions


# ---------------------------------------------------------------------------
# PRAW scraper (requires praw + credentials)
# ---------------------------------------------------------------------------

def _get_praw_instance(
    client_id: str,
    client_secret: str,
    user_agent: str = "financial_algo_research:v1.0",
) -> Any:
    """Create a read-only PRAW Reddit instance.  Returns ``praw.Reddit``.

    Credentials should come from environment variables, NOT hardcoded:
        REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
    """
    try:
        import praw  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise ImportError(
            "praw is required for Reddit scraping. "
            "Install with: pip install praw"
        ) from exc

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )


def _score_text_vader(text: str) -> float:
    """Score text sentiment using VADER lexicon.

    Returns compound score in [-1, +1].
    Falls back to 0.0 if vaderSentiment is not installed.
    """
    try:
        from vaderSentiment.vaderSentiment import (  # type: ignore[import-untyped]
            SentimentIntensityAnalyzer,
        )
    except ImportError:
        return 0.0

    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)["compound"]


def scrape_subreddit_mentions(
    subreddit_name: str,
    client_id: str,
    client_secret: str,
    limit: int = 500,
    time_filter: str = "day",
) -> list[RedditMention]:
    """Scrape ticker mentions from a subreddit.

    Parameters
    ----------
    subreddit_name:
        Name without r/ prefix (e.g. "wallstreetbets").
    client_id, client_secret:
        Reddit API credentials from env vars.
    limit:
        Max posts to scan.
    time_filter:
        One of "hour", "day", "week", "month".

    Returns
    -------
    list[RedditMention]
        Extracted ticker mentions with sentiment scores.
    """
    reddit = _get_praw_instance(client_id, client_secret)
    subreddit = reddit.subreddit(subreddit_name)
    mentions: list[RedditMention] = []

    for submission in subreddit.hot(limit=limit):
        text = f"{submission.title} {submission.selftext}"
        tickers_found = set(_TICKER_PATTERN.findall(text))
        if not tickers_found:
            continue

        sentiment = _score_text_vader(text)
        ts = datetime.utcfromtimestamp(submission.created_utc)

        for ticker in tickers_found:
            mentions.append(RedditMention(
                ticker=ticker,
                subreddit=subreddit_name,
                timestamp=ts,
                sentiment=sentiment,
                score=max(submission.score, 0),
                is_post=True,
            ))

        # Sample top-level comments for richer signal
        submission.comments.replace_more(limit=0)
        for comment in submission.comments[:20]:
            c_tickers = set(_TICKER_PATTERN.findall(comment.body))
            if not c_tickers:
                continue
            c_sentiment = _score_text_vader(comment.body)
            c_ts = datetime.utcfromtimestamp(comment.created_utc)
            for ticker in c_tickers:
                mentions.append(RedditMention(
                    ticker=ticker,
                    subreddit=subreddit_name,
                    timestamp=c_ts,
                    sentiment=c_sentiment,
                    score=max(comment.score, 0),
                    is_post=False,
                ))

    return mentions


def scrape_all_subreddits(
    client_id: str,
    client_secret: str,
    subreddits: tuple[str, ...] = SUBREDDITS,
    limit: int = 500,
    cache_dir: Path | str | None = _DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Scrape all tracked subreddits and aggregate to daily per-ticker.

    Results are cached to ``cache_dir/YYYY-MM-DD.parquet``.

    Returns
    -------
    pd.DataFrame
        Columns: ticker, mention_count, mean_sentiment, bullish_pct,
        bearish_pct, weighted_sentiment, total_score.
        Index: date.
    """
    today = pd.Timestamp.now().normalize()

    # Check cache first
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{today.strftime('%Y-%m-%d')}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

    all_mentions: list[RedditMention] = []
    for sub in subreddits:
        mentions = scrape_subreddit_mentions(
            sub, client_id, client_secret, limit=limit,
        )
        all_mentions.extend(mentions)
        time.sleep(1.0)  # rate limit between subreddits

    if not all_mentions:
        return _empty_reddit_df()

    df = _aggregate_mentions(all_mentions, today)

    if cache_dir is not None:
        df.to_parquet(cache_path)

    return df


def _aggregate_mentions(
    mentions: list[RedditMention],
    date: pd.Timestamp,
) -> pd.DataFrame:
    """Aggregate raw mentions into daily per-ticker summary."""
    rows = []
    by_ticker: dict[str, list[RedditMention]] = {}
    for m in mentions:
        by_ticker.setdefault(m.ticker, []).append(m)

    for ticker, ticker_mentions in by_ticker.items():
        sentiments = [m.sentiment for m in ticker_mentions]
        scores = [m.score for m in ticker_mentions]
        total_score = sum(scores)
        n = len(sentiments)

        # Upvote-weighted sentiment: higher-voted posts count more
        if total_score > 0:
            w_sent = sum(
                m.sentiment * max(m.score, 1) for m in ticker_mentions
            ) / total_score
        else:
            w_sent = float(np.mean(sentiments)) if sentiments else 0.0

        bullish = sum(1 for s in sentiments if s > 0.05) / max(n, 1)
        bearish = sum(1 for s in sentiments if s < -0.05) / max(n, 1)

        rows.append({
            "date": date,
            "ticker": ticker,
            "mention_count": n,
            "mean_sentiment": float(np.mean(sentiments)),
            "bullish_pct": bullish,
            "bearish_pct": bearish,
            "weighted_sentiment": w_sent,
            "total_score": total_score,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_reddit_df()
    return df.set_index("date")


def _empty_reddit_df() -> pd.DataFrame:
    """Return empty DataFrame with correct schema."""
    return pd.DataFrame(
        columns=[
            "ticker", "mention_count", "mean_sentiment",
            "bullish_pct", "bearish_pct", "weighted_sentiment", "total_score",
        ],
        index=pd.DatetimeIndex([], name="date"),
    )


# ---------------------------------------------------------------------------
# Feature extraction: raw daily data -> strategy-ready signals
# ---------------------------------------------------------------------------

def extract_reddit_features(
    daily_data: pd.DataFrame,
    tickers: list[str] | None = None,
    mention_velocity_window: int = 30,
    sentiment_zscore_window: int = 30,
) -> pd.DataFrame:
    """Extract strategy-ready features from cached daily Reddit data.

    Parameters
    ----------
    daily_data:
        Multi-day DataFrame with columns: ticker, mention_count,
        mean_sentiment, etc.  Index = date.
    tickers:
        Subset of tickers to extract.  None = all.
    mention_velocity_window:
        Rolling window for mention-velocity z-score.
    sentiment_zscore_window:
        Rolling window for sentiment z-score.

    Returns
    -------
    pd.DataFrame
        Date-indexed, columns like ``{ticker}_mention_vel``,
        ``{ticker}_sent_zscore``, ``{ticker}_bullish_pct``.
    """
    if daily_data.empty:
        return pd.DataFrame()

    if tickers is None:
        tickers = list(daily_data["ticker"].unique())

    features = pd.DataFrame(index=daily_data.index.unique().sort_values())

    for ticker in tickers:
        mask = daily_data["ticker"] == ticker
        t_data = daily_data.loc[mask].sort_index()

        if t_data.empty:
            continue

        # Reindex to full date range and fill missing days with 0
        t_data = t_data.reindex(features.index)
        mentions = t_data["mention_count"].fillna(0.0)
        sentiment = t_data["mean_sentiment"].fillna(0.0)
        bullish = t_data["bullish_pct"].fillna(0.5)

        # Mention velocity: z-score of daily mentions vs rolling mean
        mu = mentions.rolling(mention_velocity_window, min_periods=5).mean()
        sigma = mentions.rolling(
            mention_velocity_window, min_periods=5,
        ).std().replace(0, np.nan)
        vel = ((mentions - mu) / sigma).fillna(0.0)
        features[f"{ticker}_mention_vel"] = vel.clip(-5, 5)

        # Sentiment z-score
        s_mu = sentiment.rolling(
            sentiment_zscore_window, min_periods=5,
        ).mean()
        s_sigma = sentiment.rolling(
            sentiment_zscore_window, min_periods=5,
        ).std().replace(0, np.nan)
        s_z = ((sentiment - s_mu) / s_sigma).fillna(0.0)
        features[f"{ticker}_sent_zscore"] = s_z.clip(-5, 5)

        # Raw bullish percentage
        features[f"{ticker}_bullish_pct"] = bullish

    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# ---------------------------------------------------------------------------
# Synthetic Reddit sentiment for backtesting
# ---------------------------------------------------------------------------

def build_synthetic_reddit_sentiment(
    prices: pd.DataFrame,
    vix: pd.Series | None = None,
    tickers: list[str] | None = None,
    lookback: int = 20,
    noise_seed: int = 42,
) -> pd.DataFrame:
    """Build synthetic Reddit-like sentiment from price/vol proxies.

    Simulates the key patterns observed in Reddit sentiment data:

    1. **Mention velocity** correlates with absolute returns (big moves
       = more discussion) and realized vol.
    2. **Sentiment** is trend-following with lag (retail chases momentum)
       but mean-reverts at extremes (contrarian opportunity).
    3. **WSB effect**: Extreme bullish sentiment on high-vol names
       tends to precede reversals (crowd is wrong at extremes).

    Parameters
    ----------
    prices:
        Adjusted close prices (Date x Ticker).
    vix:
        VIX series.  If None, approximated from SPY realized vol.
    tickers:
        Which tickers to generate sentiment for.  None = all in prices.
    lookback:
        Base rolling window for sentiment construction.
    noise_seed:
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Date-indexed with columns: ``{ticker}_mention_vel``,
        ``{ticker}_sent_zscore``, ``{ticker}_bullish_pct`` for each ticker.
    """
    from financial_algo.indicators import realized_vol as calc_rv, zscore

    rng = np.random.RandomState(noise_seed)
    idx = prices.index

    if tickers is None:
        tickers = list(prices.columns)

    result = pd.DataFrame(index=idx)

    # VIX proxy for overall market fear
    if vix is not None:
        vix_aligned = vix.reindex(idx).ffill().fillna(20.0)
    elif "SPY" in prices.columns:
        vix_aligned = calc_rv(prices["SPY"], lookback) * 100
        vix_aligned = vix_aligned.fillna(15.0)
    else:
        vix_aligned = pd.Series(15.0, index=idx)

    vix_z = zscore(vix_aligned, lookback * 3).fillna(0.0)

    for ticker in tickers:
        if ticker not in prices.columns:
            continue

        price = prices[ticker]
        ret = price.pct_change().fillna(0.0)
        abs_ret = ret.abs()

        # 1. Mention velocity: driven by absolute returns + vol + noise
        # Big moves generate reddit buzz; vol clustering means mentions
        # are also clustered
        rv = calc_rv(price, lookback).fillna(0.15)
        mention_driver = (
            zscore(abs_ret, lookback * 2).fillna(0.0) * 0.5
            + zscore(rv, lookback * 2).fillna(0.0) * 0.3
            + rng.normal(0, 0.3, len(idx)) * 0.2  # noise
        )
        result[f"{ticker}_mention_vel"] = pd.Series(
            mention_driver, index=idx,
        ).clip(-5, 5)

        # 2. Sentiment z-score: lagged momentum-chasing with mean reversion
        # Retail sentiment follows the trend with a lag, but reverts
        # at extremes (which is the contrarian alpha source)
        mom_5 = price.pct_change(5).fillna(0.0)
        mom_20 = price.pct_change(lookback).fillna(0.0)
        raw_sentiment = (
            zscore(mom_5, lookback).fillna(0.0) * 0.4
            + zscore(mom_20, lookback * 2).fillna(0.0) * 0.4
            + rng.normal(0, 0.4, len(idx)) * 0.2
        )
        # Smooth to simulate the lag in retail crowd shifting
        smoothed = pd.Series(raw_sentiment, index=idx).ewm(span=5).mean()
        result[f"{ticker}_sent_zscore"] = smoothed.clip(-5, 5)

        # 3. Bullish percentage: sigmoid transform of sentiment
        # When sentiment is very positive, nearly everyone is bullish
        # (that's the contrarian signal)
        bull_pct = 1.0 / (1.0 + np.exp(-smoothed * 1.5))
        # Add noise to make it realistic (not perfectly correlated)
        bull_pct = bull_pct + rng.normal(0, 0.05, len(idx))
        result[f"{ticker}_bullish_pct"] = pd.Series(
            bull_pct, index=idx,
        ).clip(0.0, 1.0)

    # Add aggregate market-level features
    # Cross-ticker mention velocity (market-wide buzz)
    vel_cols = [c for c in result.columns if c.endswith("_mention_vel")]
    if vel_cols:
        result["market_mention_vel"] = result[vel_cols].mean(axis=1)

    # Cross-ticker sentiment (market-wide mood)
    sent_cols = [c for c in result.columns if c.endswith("_sent_zscore")]
    if sent_cols:
        result["market_sent_zscore"] = result[sent_cols].mean(axis=1)

    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
