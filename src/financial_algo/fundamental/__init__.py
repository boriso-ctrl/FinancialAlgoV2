"""Fundamental analysis package — sentiment, news, and macro indicators.

Separated from the technical analysis layer. While ``financial_algo.indicators``
and ``financial_algo.strategies`` operate purely on price/volume data, this
package analyses market *context*: news sentiment, fear/greed, macro indicators,
and cross-asset flows.

Subpackages
-----------
data
    News feeds (GDELT, FRED) and sentiment data pipeline.
indicators
    Sentiment-based indicators (fear/greed, news velocity, divergence).
signals
    Actionable trading signals from fundamental data.
strategies
    Trading strategies that combine fundamental + technical analysis.
"""

from financial_algo.fundamental.data.news_feeds import (
    build_synthetic_sentiment,
    fetch_crisis_sentiment,
    fetch_fred_series,
    fetch_gdelt_sentiment,
)
from financial_algo.fundamental.indicators import (
    fear_greed_composite,
    fear_spike_detector,
    greed_spike_detector,
    news_velocity,
    sentiment_momentum,
    sentiment_price_divergence,
    sentiment_zscore,
)
from financial_algo.fundamental.signals import (
    crisis_onset_signal,
    divergence_signal,
    fear_greed_signal,
    recovery_signal,
    sentiment_trend_signal,
)
from financial_algo.fundamental.strategies import (
    FearGreedContrarian,
    SentimentCrisisAlpha,
    SentimentDivergence,
    SentimentEnhancedRegime,
)

__all__ = [
    # Data
    "fetch_gdelt_sentiment",
    "fetch_fred_series",
    "fetch_crisis_sentiment",
    "build_synthetic_sentiment",
    # Indicators
    "sentiment_zscore",
    "sentiment_momentum",
    "fear_spike_detector",
    "greed_spike_detector",
    "news_velocity",
    "fear_greed_composite",
    "sentiment_price_divergence",
    # Signals
    "fear_greed_signal",
    "sentiment_trend_signal",
    "divergence_signal",
    "crisis_onset_signal",
    "recovery_signal",
    # Strategies
    "SentimentCrisisAlpha",
    "FearGreedContrarian",
    "SentimentDivergence",
    "SentimentEnhancedRegime",
]
