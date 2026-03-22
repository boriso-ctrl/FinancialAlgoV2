"""Fundamental data pipeline — news feeds and economic data."""

from financial_algo.fundamental.data.news_feeds import (
    CRISIS_QUERIES,
    NewsItem,
    SentimentSnapshot,
    build_synthetic_sentiment,
    fetch_crisis_sentiment,
    fetch_fred_series,
    fetch_gdelt_sentiment,
)
from financial_algo.fundamental.data.reddit_feeds import (
    build_synthetic_reddit_sentiment,
    extract_reddit_features,
    scrape_all_subreddits,
)

__all__ = [
    "fetch_gdelt_sentiment",
    "fetch_fred_series",
    "fetch_crisis_sentiment",
    "build_synthetic_sentiment",
    "build_synthetic_reddit_sentiment",
    "extract_reddit_features",
    "scrape_all_subreddits",
    "CRISIS_QUERIES",
    "NewsItem",
    "SentimentSnapshot",
]
