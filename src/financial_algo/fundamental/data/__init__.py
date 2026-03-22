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

__all__ = [
    "fetch_gdelt_sentiment",
    "fetch_fred_series",
    "fetch_crisis_sentiment",
    "build_synthetic_sentiment",
    "CRISIS_QUERIES",
    "NewsItem",
    "SentimentSnapshot",
]
