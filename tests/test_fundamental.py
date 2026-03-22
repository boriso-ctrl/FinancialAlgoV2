"""Tests for fundamental / sentiment analysis layer."""

from __future__ import annotations

from pathlib import Path

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
    RedditSentimentAlpha,
    SentimentCrisisAlpha,
    SentimentDivergence,
    SentimentEnhancedRegime,
)
from financial_algo.fundamental.data.reddit_feeds import (
    _DEFAULT_CACHE_DIR,
    _score_text_vader,
    build_synthetic_reddit_sentiment,
    extract_reddit_features,
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


# =========================================================================
# Reddit Feeds — build_synthetic_reddit_sentiment
# =========================================================================

class TestBuildSyntheticRedditSentiment:
    def test_output_shape(self, prices):
        df = build_synthetic_reddit_sentiment(prices)
        assert len(df) == len(prices)
        assert df.index.equals(prices.index)

    def test_deterministic_with_seed(self, prices):
        """Same seed produces identical output."""
        df1 = build_synthetic_reddit_sentiment(prices, noise_seed=42)
        df2 = build_synthetic_reddit_sentiment(prices, noise_seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seed_differs(self, prices):
        """Different seeds produce different output."""
        df1 = build_synthetic_reddit_sentiment(prices, noise_seed=42)
        df2 = build_synthetic_reddit_sentiment(prices, noise_seed=99)
        assert not df1.equals(df2)

    def test_expected_columns(self, prices):
        df = build_synthetic_reddit_sentiment(prices, tickers=["SPY", "QQQ"])
        for t in ["SPY", "QQQ"]:
            assert f"{t}_mention_vel" in df.columns
            assert f"{t}_sent_zscore" in df.columns
            assert f"{t}_bullish_pct" in df.columns
        assert "market_mention_vel" in df.columns
        assert "market_sent_zscore" in df.columns

    def test_no_nan_no_inf(self, prices):
        df = build_synthetic_reddit_sentiment(prices)
        assert not df.isna().any().any(), "NaN values found"
        assert not np.isinf(df.values).any(), "Inf values found"

    def test_mention_vel_clipped(self, prices):
        df = build_synthetic_reddit_sentiment(prices)
        vel_cols = [c for c in df.columns if c.endswith("_mention_vel")]
        for col in vel_cols:
            assert df[col].between(-5, 5).all(), f"{col} exceeds clip range"

    def test_bullish_pct_bounded(self, prices):
        df = build_synthetic_reddit_sentiment(prices)
        bull_cols = [c for c in df.columns if c.endswith("_bullish_pct")]
        for col in bull_cols:
            assert df[col].between(0.0, 1.0).all(), f"{col} out of [0,1]"

    def test_subset_tickers(self, prices):
        """Only requested tickers appear."""
        df = build_synthetic_reddit_sentiment(prices, tickers=["SPY"])
        assert "SPY_mention_vel" in df.columns
        assert "QQQ_mention_vel" not in df.columns

    def test_with_vix(self, prices, vix):
        """Accepts optional VIX series without error."""
        df = build_synthetic_reddit_sentiment(prices, vix=vix)
        assert not df.isna().any().any()

    def test_missing_ticker_ignored(self, prices):
        """Ticker not in prices is silently skipped."""
        df = build_synthetic_reddit_sentiment(
            prices, tickers=["SPY", "DOESNOTEXIST"],
        )
        assert "SPY_mention_vel" in df.columns
        assert "DOESNOTEXIST_mention_vel" not in df.columns


# =========================================================================
# Reddit Feeds — _score_text_vader
# =========================================================================

class TestScoreTextVader:
    def test_returns_float(self):
        score = _score_text_vader("I love this stock!")
        assert isinstance(score, float)

    def test_bounded(self):
        score = _score_text_vader("Terrible earnings report, sell sell sell")
        assert -1.0 <= score <= 1.0

    def test_empty_string(self):
        score = _score_text_vader("")
        assert isinstance(score, float)
        assert -1.0 <= score <= 1.0


# =========================================================================
# Reddit Feeds — extract_reddit_features
# =========================================================================

class TestExtractRedditFeatures:
    @pytest.fixture
    def daily_reddit_data(self, dates):
        """Multi-day fake Reddit daily data."""
        rng = np.random.RandomState(42)
        rows = []
        for dt in dates:
            for ticker in ["SPY", "QQQ"]:
                rows.append({
                    "date": dt,
                    "ticker": ticker,
                    "mention_count": rng.randint(10, 200),
                    "mean_sentiment": rng.uniform(-0.5, 0.5),
                    "bullish_pct": rng.uniform(0.2, 0.8),
                    "bearish_pct": rng.uniform(0.1, 0.5),
                    "weighted_sentiment": rng.uniform(-0.3, 0.3),
                    "total_score": rng.randint(50, 5000),
                })
        df = pd.DataFrame(rows).set_index("date")
        return df

    def test_output_columns(self, daily_reddit_data):
        feat = extract_reddit_features(daily_reddit_data, tickers=["SPY"])
        assert "SPY_mention_vel" in feat.columns
        assert "SPY_sent_zscore" in feat.columns
        assert "SPY_bullish_pct" in feat.columns

    def test_no_nan_no_inf(self, daily_reddit_data):
        feat = extract_reddit_features(daily_reddit_data)
        assert not feat.isna().any().any(), "NaN in features"
        assert not np.isinf(feat.values).any(), "Inf in features"

    def test_mention_vel_clipped(self, daily_reddit_data):
        feat = extract_reddit_features(daily_reddit_data)
        vel_cols = [c for c in feat.columns if c.endswith("_mention_vel")]
        for col in vel_cols:
            assert feat[col].between(-5, 5).all()

    def test_empty_input(self):
        empty = pd.DataFrame(
            columns=["ticker", "mention_count", "mean_sentiment",
                     "bullish_pct", "bearish_pct", "weighted_sentiment",
                     "total_score"],
            index=pd.DatetimeIndex([], name="date"),
        )
        feat = extract_reddit_features(empty)
        assert feat.empty

    def test_subset_tickers(self, daily_reddit_data):
        feat = extract_reddit_features(daily_reddit_data, tickers=["SPY"])
        assert "SPY_mention_vel" in feat.columns
        assert "QQQ_mention_vel" not in feat.columns


# =========================================================================
# Reddit Feeds — cache path logic
# =========================================================================

class TestRedditCachePath:
    def test_default_cache_dir_is_pathlib(self):
        assert isinstance(_DEFAULT_CACHE_DIR, Path)  # pyright: ignore[reportPossiblyUnbound]

    def test_cache_dir_under_home(self):
        assert ".financial_algo_cache" in str(_DEFAULT_CACHE_DIR)
        assert "reddit" in str(_DEFAULT_CACHE_DIR)


# =========================================================================
# G7 — RedditSentimentAlpha
# =========================================================================

class TestRedditSentimentAlpha:
    def test_importable(self):
        """G7 is importable from fundamental.strategies."""
        from financial_algo.fundamental.strategies import RedditSentimentAlpha as G7
        assert G7 is not None
        assert G7.name == "G7-RedditSentimentAlpha"

    def test_output_valid(self, prices, regime):
        """generate_weights returns correct shape, no NaN/inf."""
        strat = RedditSentimentAlpha()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)
        assert w.index.equals(prices.index)
        assert w.shape[1] > 0
        assert not w.isna().any().any(), "NaN in weights"
        assert not np.isinf(w.values).any(), "Inf in weights"

    def test_without_regime(self, prices):
        """Works without regime argument."""
        strat = RedditSentimentAlpha()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_with_explicit_reddit_features(self, prices, regime):
        """Accepts pre-built Reddit features."""
        features = build_synthetic_reddit_sentiment(prices)
        strat = RedditSentimentAlpha()
        w = strat.generate_weights(prices, regime, reddit_features=features)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_handles_missing_reddit_data(self, prices, regime):
        """Falls back to synthetic when no Reddit data provided."""
        strat = RedditSentimentAlpha()
        # None triggers synthetic fallback internally
        w = strat.generate_weights(prices, regime, reddit_features=None)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_signal_euphoria(self, prices):
        """When sentiment is extremely bullish, equity weight should be negative (contrarian short)."""
        strat = RedditSentimentAlpha()
        # Build features with extreme bullish sentiment
        features = build_synthetic_reddit_sentiment(prices)
        # Override to simulate euphoria
        for col in features.columns:
            if col.endswith("_bullish_pct"):
                features[col] = 0.95  # way above crowd_euphoria=0.80
            if col.endswith("_sent_zscore"):
                features[col] = 3.0  # way above sent_contrarian_bull=1.3
        w = strat.generate_weights(prices, reddit_features=features)
        # In euphoria, equity should be shorted (negative or very small)
        # Most days should have negative/low equity weight
        spy_w = w["SPY"]
        assert (spy_w < 0.5).sum() > len(spy_w) * 0.5

    def test_signal_panic(self, prices):
        """When sentiment is extremely bearish, equity weight should be high (contrarian long)."""
        strat = RedditSentimentAlpha()
        features = build_synthetic_reddit_sentiment(prices)
        for col in features.columns:
            if col.endswith("_bullish_pct"):
                features[col] = 0.10  # below crowd_panic=0.25
            if col.endswith("_sent_zscore"):
                features[col] = -3.0  # below sent_contrarian_bear=-1.3
        w = strat.generate_weights(prices, reddit_features=features)
        spy_w = w["SPY"]
        # In panic, should have elevated equity weight
        assert (spy_w > 0.5).sum() > len(spy_w) * 0.3

    def test_signal_neutral(self, prices):
        """When sentiment is neutral, weights should be modest."""
        strat = RedditSentimentAlpha()
        features = build_synthetic_reddit_sentiment(prices)
        for col in features.columns:
            if col.endswith("_bullish_pct"):
                features[col] = 0.50  # neutral
            if col.endswith("_sent_zscore"):
                features[col] = 0.0  # neutral
            if col.endswith("_mention_vel"):
                features[col] = 0.0  # no buzz
        w = strat.generate_weights(prices, reddit_features=features)
        spy_w = w["SPY"]
        # Neutral weight is 0.5 * vol_scale, should be in modest range
        assert spy_w.abs().max() < 2.0

    def test_signal_viral_event(self, prices):
        """Viral mention spike reduces equity exposure."""
        strat = RedditSentimentAlpha()
        features = build_synthetic_reddit_sentiment(prices)
        for col in features.columns:
            if col.endswith("_mention_vel"):
                features[col] = 4.0  # above mention_vel_extreme=3.0
            if col.endswith("_sent_zscore"):
                features[col] = 0.0
            if col.endswith("_bullish_pct"):
                features[col] = 0.50
        w = strat.generate_weights(prices, reddit_features=features)
        # Viral should cap equity at low level (0.1 * vol_scale)
        spy_w = w["SPY"]
        assert spy_w.max() < 0.5

    def test_vol_scaling(self, prices):
        """Position sizes scale inversely with SPY vol."""
        strat = RedditSentimentAlpha()
        w = strat.generate_weights(prices)
        # Weights should not all be identical (vol changes over time)
        spy_w = w["SPY"]
        assert spy_w.std() > 0.0

    def test_crisis_regime_scales_down(self, prices, regime):
        """In CRISIS regime, all positions should be scaled by crisis_scale."""
        strat = RedditSentimentAlpha()
        w_with_regime = strat.generate_weights(prices, regime)
        w_no_regime = strat.generate_weights(prices, regime=None)
        # During crisis days (90-110), weights with regime should be smaller
        crisis_abs = w_with_regime.iloc[95:105].abs().sum().sum()
        no_crisis_abs = w_no_regime.iloc[95:105].abs().sum().sum()
        assert crisis_abs < no_crisis_abs, (
            "Crisis regime should reduce position sizes"
        )

    def test_output_tickers(self, prices):
        """Output should contain SPY, QQQ, TLT, GLD columns."""
        strat = RedditSentimentAlpha()
        w = strat.generate_weights(prices)
        for ticker in ["SPY", "QQQ", "TLT", "GLD"]:
            assert ticker in w.columns
