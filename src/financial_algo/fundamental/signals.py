"""Sentiment-based trading signals.

Converts sentiment indicators into actionable trading signals
(+1 = long, -1 = short, 0 = flat).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_algo.fundamental.indicators import (
    fear_greed_composite,
    fear_spike_detector,
    news_velocity,
    sentiment_momentum,
    sentiment_price_divergence,
    sentiment_zscore,
)


def fear_greed_signal(
    sentiment_df: pd.DataFrame,
    fear_threshold: float = -40.0,
    greed_threshold: float = 40.0,
) -> pd.Series:
    """Generate long/short signals from Fear & Greed composite.

    - Extreme Fear (< fear_threshold): contrarian long signal (+1)
    - Extreme Greed (> greed_threshold): contrarian short signal (-1)
    - Otherwise: neutral (0)
    """
    fg = fear_greed_composite(sentiment_df)
    signal = pd.Series(0, index=fg.index, dtype=int)
    signal[fg <= fear_threshold] = 1    # buy fear
    signal[fg >= greed_threshold] = -1  # sell greed
    return signal


def sentiment_trend_signal(
    sentiment_score: pd.Series,
    fast: int = 5,
    slow: int = 20,
    threshold: float = 0.3,
) -> pd.Series:
    """Trend-following signal on sentiment momentum.

    Goes long when sentiment is improving (momentum > threshold),
    short when sentiment deteriorating (momentum < -threshold).
    """
    mom = sentiment_momentum(sentiment_score, fast, slow)
    signal = pd.Series(0, index=mom.index, dtype=int)
    signal[mom > threshold] = 1
    signal[mom < -threshold] = -1
    return signal


def divergence_signal(
    sentiment_score: pd.Series,
    prices: pd.Series,
    lookback: int = 20,
    threshold: float = 1.5,
) -> pd.Series:
    """Trade sentiment-price divergences.

    Positive divergence (sentiment up, price down) = long.
    Negative divergence (sentiment down, price up) = short.
    """
    div = sentiment_price_divergence(sentiment_score, prices, lookback)
    signal = pd.Series(0, index=div.index, dtype=int)
    signal[div > threshold] = 1
    signal[div < -threshold] = -1
    return signal


def crisis_onset_signal(
    sentiment_df: pd.DataFrame,
    fear_z_threshold: float = 1.5,
    velocity_threshold: float = 1.5,
    lookback: int = 60,
) -> pd.Series:
    """Detect crisis onset: simultaneous fear spike + news acceleration.

    This is the key signal — it fires EARLY in a crisis when:
    1. Fear is spiking (z-score above threshold)
    2. News velocity is elevated (information accelerating)

    Returns +1 on crisis onset days (signal to activate hedges / go short).
    """
    fear = sentiment_df.get("fear_index", pd.Series(0.0, index=sentiment_df.index))
    velocity_proxy = sentiment_df.get(
        "news_velocity_proxy",
        pd.Series(0.0, index=sentiment_df.index),
    )

    fear_z = sentiment_zscore(fear, lookback)
    vel = news_velocity(velocity_proxy.abs(), fast=5, slow=20)

    crisis_detected = (fear_z >= fear_z_threshold) & (vel >= velocity_threshold)

    signal = pd.Series(0, index=sentiment_df.index, dtype=int)
    signal[crisis_detected] = 1
    return signal


def recovery_signal(
    sentiment_df: pd.DataFrame,
    fear_recovery_threshold: float = -0.5,
    momentum_threshold: float = 0.5,
) -> pd.Series:
    """Detect crisis-to-recovery transition.

    Fires when:
    1. Fear index was recently elevated but is now improving
    2. Sentiment momentum turns positive

    Returns +1 on recovery days (signal to go leveraged long).
    """
    fear = sentiment_df.get("fear_index", pd.Series(0.0, index=sentiment_df.index))
    sentiment = sentiment_df.get("sentiment_score", pd.Series(0.0, index=sentiment_df.index))

    # Was fearful recently (within last 10 days)
    recent_high_fear = fear.rolling(10).max().shift(1) >= 1.0

    # Fear now subsiding
    fear_subsiding = fear <= fear_recovery_threshold

    # Sentiment momentum turning positive
    mom = sentiment_momentum(sentiment, fast=5, slow=20)
    positive_momentum = mom >= momentum_threshold

    recovery = recent_high_fear & fear_subsiding & positive_momentum

    signal = pd.Series(0, index=sentiment_df.index, dtype=int)
    signal[recovery] = 1
    return signal
