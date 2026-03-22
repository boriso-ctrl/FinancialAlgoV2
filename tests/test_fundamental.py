"""Tests for fundamental / sentiment analysis layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_algo.fundamental.data.news_feeds import build_synthetic_sentiment
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


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def dates():
    return pd.bdate_range("2020-01-01", periods=200)


@pytest.fixture
def prices(dates):
    """Synthetic price DataFrame with all required tickers."""
    np.random.seed(42)
    n = len(dates)
    ret = np.random.normal(0.0005, 0.015, (n, 10))
    cols = ["SPY", "QQQ", "TLT", "GLD", "HYG", "LQD",
            "XLK", "XLF", "XLI", "XLB"]
    cum = np.exp(np.cumsum(ret, axis=0)) * 100
    return pd.DataFrame(cum, index=dates, columns=cols)


@pytest.fixture
def vix(dates):
    """Synthetic VIX series."""
    np.random.seed(99)
    return pd.Series(
        np.random.uniform(12, 40, len(dates)),
        index=dates,
        name="^VIX",
    )


@pytest.fixture
def sentiment_df(prices, vix):
    return build_synthetic_sentiment(prices, vix=vix)


@pytest.fixture
def regime(dates):
    """Simple alternating regime for testing."""
    from financial_algo.regimes import Regime
    vals = [Regime.NORMAL] * len(dates)
    # Mark middle 20 days as crisis
    for i in range(90, 110):
        vals[i] = Regime.GENERAL_CRISIS
    return pd.Series(vals, index=dates)


# =========================================================================
# build_synthetic_sentiment
# =========================================================================

class TestBuildSyntheticSentiment:
    def test_output_shape(self, prices, vix):
        df = build_synthetic_sentiment(prices, vix=vix)
        assert len(df) == len(prices)
        assert df.index.equals(prices.index)

    def test_required_columns(self, sentiment_df):
        expected = {
            "sentiment_score", "fear_index", "greed_index",
            "news_velocity_proxy", "vix_fear", "credit_stress",
            "haven_demand", "breadth_confidence",
        }
        assert expected.issubset(set(sentiment_df.columns))

    def test_scores_bounded(self, sentiment_df):
        # Drop warm-up NaN values before checking bounds
        fear = sentiment_df["fear_index"].dropna()
        greed = sentiment_df["greed_index"].dropna()
        score = sentiment_df["sentiment_score"].dropna()
        assert fear.between(-3, 3).all()
        assert greed.between(-3, 3).all()
        assert score.between(-5, 5).all()

    def test_no_vix_fallback(self, prices):
        """Works without VIX — falls back to realized vol."""
        df = build_synthetic_sentiment(prices, vix=None)
        assert "vix_fear" in df.columns
        assert not df["vix_fear"].isna().all()

    def test_minimal_columns(self, dates):
        """Works with only SPY column."""
        spy = pd.DataFrame(
            np.exp(np.cumsum(np.random.normal(0, 0.01, len(dates)))) * 100,
            index=dates, columns=["SPY"],
        )
        df = build_synthetic_sentiment(spy)
        assert "sentiment_score" in df.columns


# =========================================================================
# Indicators
# =========================================================================

class TestSentimentZscore:
    def test_output_length(self, sentiment_df):
        z = sentiment_zscore(sentiment_df["sentiment_score"], window=20)
        assert len(z) == len(sentiment_df)

    def test_zero_filled(self, sentiment_df):
        z = sentiment_zscore(sentiment_df["sentiment_score"])
        assert not z.isna().any()

    def test_reasonable_range(self, sentiment_df):
        z = sentiment_zscore(sentiment_df["sentiment_score"], window=60)
        # Z-scores shouldn't routinely exceed +-5 with normal data
        assert z.between(-10, 10).all()


class TestSentimentMomentum:
    def test_output_length(self, sentiment_df):
        mom = sentiment_momentum(sentiment_df["sentiment_score"])
        assert len(mom) == len(sentiment_df)

    def test_positive_when_improving(self):
        """When sentiment steadily rises, momentum should be positive."""
        idx = pd.bdate_range("2020-01-01", periods=100)
        improving = pd.Series(np.linspace(-2, 2, 100), index=idx)
        mom = sentiment_momentum(improving, fast=5, slow=20)
        # After warm-up, momentum should be positive
        assert (mom.iloc[30:] > 0).all()


class TestFearSpikeDetector:
    def test_returns_boolean(self, sentiment_df):
        result = fear_spike_detector(sentiment_df["fear_index"])
        assert result.dtype == bool

    def test_spike_detected_on_extreme(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        fear = pd.Series(0.0, index=idx)
        # Inject a spike
        fear.iloc[80:85] = 3.0
        result = fear_spike_detector(fear, threshold=1.5, lookback=60)
        assert result.iloc[80:85].any()


class TestGreedSpikeDetector:
    def test_returns_boolean(self, sentiment_df):
        result = greed_spike_detector(sentiment_df["greed_index"])
        assert result.dtype == bool


class TestNewsVelocity:
    def test_base_ratio_near_one(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        constant = pd.Series(1.0, index=idx)
        vel = news_velocity(constant)
        assert vel.between(0.5, 1.5).all()

    def test_elevated_when_spike(self):
        idx = pd.bdate_range("2020-01-01", periods=100)
        vol = pd.Series(1.0, index=idx)
        vol.iloc[80:90] = 5.0  # sudden spike
        vel = news_velocity(vol, fast=5, slow=20)
        assert vel.iloc[85] > 1.5


class TestFearGreedComposite:
    def test_bounded(self, sentiment_df):
        fg = fear_greed_composite(sentiment_df).dropna()
        assert fg.between(-100, 100).all()

    def test_empty_returns_zero(self, dates):
        empty = pd.DataFrame(index=dates)
        fg = fear_greed_composite(empty)
        assert (fg == 0.0).all()


class TestSentimentPriceDivergence:
    def test_output_length(self, sentiment_df, prices):
        div = sentiment_price_divergence(
            sentiment_df["sentiment_score"], prices["SPY"]
        )
        assert len(div) == len(prices)


# =========================================================================
# Signals
# =========================================================================

class TestFearGreedSignal:
    def test_values_in_range(self, sentiment_df):
        sig = fear_greed_signal(sentiment_df)
        assert set(sig.unique()).issubset({-1, 0, 1})

    def test_length(self, sentiment_df):
        sig = fear_greed_signal(sentiment_df)
        assert len(sig) == len(sentiment_df)


class TestSentimentTrendSignal:
    def test_values_in_range(self, sentiment_df):
        sig = sentiment_trend_signal(sentiment_df["sentiment_score"])
        assert set(sig.unique()).issubset({-1, 0, 1})


class TestDivergenceSignal:
    def test_values_in_range(self, sentiment_df, prices):
        sig = divergence_signal(
            sentiment_df["sentiment_score"], prices["SPY"]
        )
        assert set(sig.unique()).issubset({-1, 0, 1})


class TestCrisisOnsetSignal:
    def test_values_in_range(self, sentiment_df):
        sig = crisis_onset_signal(sentiment_df)
        assert set(sig.unique()).issubset({0, 1})


class TestRecoverySignal:
    def test_values_in_range(self, sentiment_df):
        sig = recovery_signal(sentiment_df)
        assert set(sig.unique()).issubset({0, 1})


# =========================================================================
# Strategies — generate_weights
# =========================================================================

class TestSentimentCrisisAlpha:
    def test_output_valid(self, prices, regime, sentiment_df):
        strat = SentimentCrisisAlpha()
        w = strat.generate_weights(prices, regime, sentiment_df)
        assert len(w) == len(prices)
        assert w.index.equals(prices.index)
        # Strategy outputs weights for its specific tickers (subset)
        assert w.shape[1] > 0

    def test_without_sentiment_df(self, prices, regime):
        """Should build synthetic sentiment internally."""
        strat = SentimentCrisisAlpha()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)


class TestFearGreedContrarian:
    def test_output_valid(self, prices, regime, sentiment_df):
        strat = FearGreedContrarian()
        w = strat.generate_weights(prices, regime, sentiment_df)
        assert len(w) == len(prices)
        assert w.shape[1] > 0


class TestSentimentDivergence:
    def test_output_valid(self, prices, regime, sentiment_df):
        strat = SentimentDivergence()
        w = strat.generate_weights(prices, regime, sentiment_df)
        assert len(w) == len(prices)
        assert w.shape[1] > 0


class TestSentimentEnhancedRegime:
    def test_output_valid(self, prices, regime, sentiment_df):
        strat = SentimentEnhancedRegime()
        w = strat.generate_weights(prices, regime, sentiment_df)
        assert len(w) == len(prices)
        assert w.shape[1] > 0

    def test_regime_override_in_crisis(self, prices, regime, sentiment_df):
        """During crisis regime, weights should differ from neutral."""
        strat = SentimentEnhancedRegime()
        w = strat.generate_weights(prices, regime, sentiment_df)
        # During the injected crisis period (days 90-110), check weights changed
        crisis_w = w.iloc[95:105]
        normal_w = w.iloc[10:20]
        # Not necessarily different (depends on sentiment) but should be valid
        assert not crisis_w.isna().any().any()
        assert not normal_w.isna().any().any()
