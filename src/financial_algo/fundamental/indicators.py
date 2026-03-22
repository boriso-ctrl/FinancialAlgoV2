"""Fundamental / sentiment indicators.

These complement the technical indicators in ``financial_algo.indicators``
by quantifying market *sentiment* and *macro context* rather than
price/volume patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sentiment momentum & mean-reversion
# ---------------------------------------------------------------------------

def sentiment_zscore(
    sentiment: pd.Series,
    window: int = 60,
) -> pd.Series:
    """Rolling z-score of a sentiment series.

    Positive z-score = unusually optimistic.
    Negative z-score = unusually fearful.
    """
    mu = sentiment.rolling(window, min_periods=max(1, window // 2)).mean()
    sigma = sentiment.rolling(window, min_periods=max(1, window // 2)).std()
    return ((sentiment - mu) / sigma.replace(0, np.nan)).fillna(0.0)


def sentiment_momentum(
    sentiment: pd.Series,
    fast: int = 5,
    slow: int = 20,
) -> pd.Series:
    """Sentiment momentum — fast EMA minus slow EMA of sentiment score.

    Positive = sentiment improving (less fear / more greed).
    Negative = sentiment deteriorating.
    """
    fast_ema = sentiment.ewm(span=fast, min_periods=fast).mean()
    slow_ema = sentiment.ewm(span=slow, min_periods=slow).mean()
    return fast_ema - slow_ema


def fear_spike_detector(
    fear_index: pd.Series,
    threshold: float = 2.0,
    lookback: int = 60,
) -> pd.Series:
    """Detect sudden fear spikes (z-score > threshold).

    Returns a boolean Series — True on days when fear is abnormally high.
    """
    z = sentiment_zscore(fear_index, lookback)
    return z >= threshold


def greed_spike_detector(
    greed_index: pd.Series,
    threshold: float = 2.0,
    lookback: int = 60,
) -> pd.Series:
    """Detect greed spikes (z-score > threshold) — potential reversal."""
    z = sentiment_zscore(greed_index, lookback)
    return z >= threshold


# ---------------------------------------------------------------------------
# News velocity (information flow rate)
# ---------------------------------------------------------------------------

def news_velocity(
    volume_or_proxy: pd.Series,
    fast: int = 5,
    slow: int = 20,
) -> pd.Series:
    """Rate of change in news/information flow.

    High velocity + negative sentiment = crisis developing.
    High velocity + positive sentiment = recovery rally news.
    """
    fast_avg = volume_or_proxy.rolling(fast, min_periods=1).mean()
    slow_avg = volume_or_proxy.rolling(slow, min_periods=1).mean()
    return (fast_avg / slow_avg.replace(0, np.nan)).fillna(1.0)


# ---------------------------------------------------------------------------
# Composite fear/greed indicator (inspired by CNN Fear & Greed)
# ---------------------------------------------------------------------------

def fear_greed_composite(
    sentiment_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Composite Fear & Greed index from sentiment DataFrame.

    Parameters
    ----------
    sentiment_df:
        DataFrame from ``build_synthetic_sentiment()`` with columns like
        vix_fear, credit_stress, haven_demand, breadth_confidence.
    weights:
        Component weights. Defaults to equal weighting of available columns.

    Returns
    -------
    pd.Series
        Score from -100 (Extreme Fear) to +100 (Extreme Greed).
    """
    components = [
        "vix_fear", "credit_stress", "haven_demand",
        "breadth_confidence", "news_velocity_proxy",
    ]
    available = [c for c in components if c in sentiment_df.columns]

    if not available:
        return pd.Series(0.0, index=sentiment_df.index, name="fear_greed")

    if weights is None:
        w = {c: 1.0 / len(available) for c in available}
    else:
        w = weights

    # Normalize each component to [-1, 1] range, then scale to [-100, 100]
    composite = pd.Series(0.0, index=sentiment_df.index)
    for col in available:
        col_norm = sentiment_df[col] / 3.0  # components are clipped to [-3, 3]
        col_weight = w.get(col, 1.0 / len(available))
        # Invert fear components so positive = greed
        if col in ("vix_fear", "credit_stress", "haven_demand"):
            composite -= col_norm * col_weight
        else:
            composite += col_norm * col_weight

    return (composite * 100).clip(-100, 100).rename("fear_greed")


# ---------------------------------------------------------------------------
# Sentiment divergence (sentiment vs price)
# ---------------------------------------------------------------------------

def sentiment_price_divergence(
    sentiment: pd.Series,
    prices: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """Detect divergence between sentiment and price action.

    Positive divergence: sentiment improving while price falling (buy signal).
    Negative divergence: sentiment deteriorating while price rising (sell signal).
    """
    sent_change = sentiment.rolling(lookback, min_periods=lookback // 2).mean().diff(lookback)
    price_change = prices.pct_change(lookback)

    # Normalize to comparable scale
    sent_norm = sent_change / sent_change.rolling(lookback * 3).std().replace(0, np.nan)
    price_norm = price_change / price_change.rolling(lookback * 3).std().replace(0, np.nan)

    return (sent_norm - price_norm).fillna(0.0)
