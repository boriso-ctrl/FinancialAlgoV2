"""Tests for financial_algo.indicators."""

import math
import numpy as np
import pandas as pd
import pytest

from financial_algo.indicators import (
    bollinger_bands,
    bollinger_bandwidth,
    bollinger_pctb,
    close_open_ratio,
    daily_price_range,
    dv2,
    hurst_exponent,
    kst,
    lower_shadow_pct,
    macd,
    moving_average,
    pgo,
    price_volume_corr,
    relative_volume,
    rmi,
    rolling_kurt,
    rolling_quantile_ratio,
    rolling_skew,
    rsi,
    rsi_change_rate,
    tsi,
    upper_shadow_pct,
    volume_std,
    williams_r,
)


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_basic(self):
        prices = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = moving_average(prices, 3)
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(2.0)
        assert result[3] == pytest.approx(3.0)
        assert result[4] == pytest.approx(4.0)

    def test_window_1_returns_prices(self):
        prices = [10.0, 20.0, 30.0]
        assert moving_average(prices, 1) == pytest.approx(prices)

    def test_window_equals_length(self):
        prices = [2.0, 4.0, 6.0]
        result = moving_average(prices, 3)
        assert math.isnan(result[0])
        assert math.isnan(result[1])
        assert result[2] == pytest.approx(4.0)

    def test_single_element(self):
        result = moving_average([5.0], 1)
        assert result == [5.0]

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            moving_average([1.0, 2.0], 0)


# ---------------------------------------------------------------------------
# rsi
# ---------------------------------------------------------------------------

class TestRsi:
    def _rising_prices(self, n: int = 20) -> list[float]:
        return [float(i) for i in range(1, n + 1)]

    def test_output_length_matches_input(self):
        prices = self._rising_prices(20)
        result = rsi(prices, 14)
        assert len(result) == len(prices)

    def test_first_period_values_are_nan(self):
        prices = self._rising_prices(20)
        result = rsi(prices, 14)
        for i in range(14):
            assert math.isnan(result[i])

    def test_rsi_100_on_all_gains(self):
        prices = self._rising_prices(20)
        result = rsi(prices, 14)
        assert result[14] == pytest.approx(100.0)

    def test_rsi_range(self):
        import random
        random.seed(42)
        prices = [100.0]
        for _ in range(29):
            prices.append(prices[-1] * (1 + random.uniform(-0.02, 0.02)))
        result = rsi(prices, 14)
        valid = [v for v in result if not math.isnan(v)]
        assert all(0.0 <= v <= 100.0 for v in valid)

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError):
            rsi([1.0, 2.0, 3.0], period=0)

    def test_too_few_prices_raises(self):
        with pytest.raises(ValueError):
            rsi([42.0], period=14)


# ---------------------------------------------------------------------------
# bollinger_bands
# ---------------------------------------------------------------------------

class TestBollingerBands:
    def _prices(self, n: int = 25) -> list[float]:
        return [100.0 + i * 0.5 for i in range(n)]

    def test_output_lengths(self):
        prices = self._prices(25)
        upper, middle, lower = bollinger_bands(prices, window=20)
        assert len(upper) == len(middle) == len(lower) == 25

    def test_upper_greater_than_lower(self):
        prices = self._prices(25)
        upper, _, lower = bollinger_bands(prices, window=20)
        for u, l in zip(upper[19:], lower[19:]):
            assert u >= l

    def test_middle_is_sma(self):
        prices = self._prices(25)
        _, middle, _ = bollinger_bands(prices, window=20)
        sma = moving_average(prices, 20)
        for m, s in zip(middle, sma):
            if not math.isnan(s):
                assert m == pytest.approx(s)

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            bollinger_bands([1.0, 2.0, 3.0], window=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ohlc(n: int = 300, seed: int = 42) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (high, low, close) pd.Series of length *n*."""
    rng = np.random.default_rng(seed)
    close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.5, n)))
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    return high, low, close


def _volume(n: int = 300, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(np.abs(rng.normal(1_000_000, 200_000, n)))


# ---------------------------------------------------------------------------
# williams_r
# ---------------------------------------------------------------------------

class TestWilliamsR:
    def test_range(self):
        high, low, close = _ohlc()
        result = williams_r(high, low, close, period=14)
        valid = result.dropna()
        assert (valid >= -100).all() and (valid <= 0).all()

    def test_nan_for_initial_bars(self):
        high, low, close = _ohlc(20)
        result = williams_r(high, low, close, period=14)
        assert result.iloc[:13].isna().all()

    def test_overbought_when_close_equals_highest_high(self):
        # Monotonically rising close == high → rolling max always = close → WILLR = 0
        s = pd.Series(np.linspace(10.0, 20.0, 30))
        result = williams_r(s, s - 2.0, s, period=5)
        valid = result.dropna()
        assert valid.abs().max() < 1e-10


# ---------------------------------------------------------------------------
# dv2
# ---------------------------------------------------------------------------

class TestDV2:
    def test_output_length(self):
        high, low, close = _ohlc(300)
        result = dv2(high, low, close)
        assert len(result) == 300

    def test_range_after_warmup(self):
        high, low, close = _ohlc(300)
        result = dv2(high, low, close)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_high_dv2_after_strong_rally(self):
        # Strongly rising close relative to midprice → high DV2
        n = 260
        mid = pd.Series(np.ones(n) * 100.0)
        close = pd.Series(100.0 + np.linspace(0, 10, n))  # rising close
        result = dv2(mid, mid - 1, close)
        assert result.dropna().iloc[-1] > 60

    def test_low_dv2_after_strong_decline(self):
        n = 260
        close = pd.Series(100.0 - np.linspace(0, 10, n))  # falling close
        mid = pd.Series(np.ones(n) * 100.0)
        result = dv2(mid, mid - 1, close)
        assert result.dropna().iloc[-1] < 40


# ---------------------------------------------------------------------------
# kst
# ---------------------------------------------------------------------------

class TestKST:
    def test_columns(self):
        _, _, close = _ohlc(300)
        result = kst(close)
        assert "kst" in result.columns and "kst_signal" in result.columns

    def test_signal_is_smoothed_kst(self):
        _, _, close = _ohlc(300)
        result = kst(close)
        # signal must be rolling(9) mean of kst
        expected_signal = result["kst"].rolling(9).mean()
        valid = result["kst_signal"].dropna()
        pd.testing.assert_series_equal(
            valid, expected_signal.dropna(), check_names=False
        )

    def test_rising_market_positive_kst(self):
        close = pd.Series(np.linspace(100, 200, 200))
        result = kst(close)
        assert result["kst"].dropna().iloc[-1] > 0


# ---------------------------------------------------------------------------
# tsi
# ---------------------------------------------------------------------------

class TestTSI:
    def test_range(self):
        _, _, close = _ohlc(300)
        result = tsi(close)
        valid = result.dropna()
        assert (valid >= -100).all() and (valid <= 100).all()

    def test_rising_prices_positive(self):
        close = pd.Series(np.linspace(100, 200, 200))
        result = tsi(close)
        # Sustained rally → positive TSI
        assert result.iloc[-1] > 0

    def test_falling_prices_negative(self):
        close = pd.Series(np.linspace(200, 100, 200))
        result = tsi(close)
        assert result.iloc[-1] < 0

    def test_output_length(self):
        _, _, close = _ohlc(100)
        result = tsi(close)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# pgo
# ---------------------------------------------------------------------------

class TestPGO:
    def test_output_length(self):
        high, low, close = _ohlc(100)
        result = pgo(high, low, close)
        assert len(result) == 100

    def test_breakout_gives_positive_value(self):
        # Monotonically rising close stays above its SMA → positive PGO
        close = pd.Series(np.linspace(90.0, 140.0, 50))
        high = close + 1.0
        low = close - 1.0
        result = pgo(high, low, close, period=14)
        assert result.dropna().iloc[-1] > 0

    def test_nan_for_initial_bars(self):
        high, low, close = _ohlc(30)
        result = pgo(high, low, close, period=14)
        assert result.iloc[:13].isna().all()


# ---------------------------------------------------------------------------
# rmi
# ---------------------------------------------------------------------------

class TestRMI:
    def test_range(self):
        _, _, close = _ohlc(300)
        result = rmi(close)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_flat_prices_near_50(self):
        close = pd.Series(np.ones(100) * 100.0)
        result = rmi(close)
        valid = result.iloc[20:].dropna()
        # Flat: no ups or downs → neutral ~50
        assert (valid - 50.0).abs().max() < 1e-5

    def test_rising_prices_above_50(self):
        close = pd.Series(np.linspace(100, 200, 100))
        result = rmi(close)
        assert result.iloc[-1] > 50


# ---------------------------------------------------------------------------
# rolling_skew
# ---------------------------------------------------------------------------

class TestRollingSkew:
    def test_output_length(self):
        _, _, close = _ohlc(100)
        returns = close.pct_change().fillna(0.0)
        result = rolling_skew(returns, window=20)
        assert len(result) == 100

    def test_symmetric_distribution_near_zero(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 200))
        result = rolling_skew(returns, window=100)
        # Last value should be close-ish to 0 for normal distribution
        assert abs(result.iloc[-1]) < 1.0

    def test_nan_before_warmup(self):
        returns = pd.Series(np.linspace(0, 0.01, 30))
        result = rolling_skew(returns, window=20)
        assert result.iloc[:19].isna().all()


# ---------------------------------------------------------------------------
# rolling_kurt
# ---------------------------------------------------------------------------

class TestRollingKurt:
    def test_output_length(self):
        _, _, close = _ohlc(100)
        returns = close.pct_change().fillna(0.0)
        result = rolling_kurt(returns, window=20)
        assert len(result) == 100

    def test_normal_distribution_near_zero(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0, 0.01, 500))
        result = rolling_kurt(returns, window=200)
        # Fisher excess kurtosis of normal distribution ~= 0
        assert abs(result.iloc[-1]) < 1.5

    def test_nan_before_warmup(self):
        returns = pd.Series(np.linspace(0, 0.01, 25))
        result = rolling_kurt(returns, window=20)
        assert result.iloc[:19].isna().all()


# ---------------------------------------------------------------------------
# hurst_exponent
# ---------------------------------------------------------------------------

class TestHurstExponent:
    def test_trending_series_high_hurst(self):
        # AR(1) with strong positive autocorrelation in increments → H >> 0.5
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 1.0, 300)
        dx = np.zeros(300)
        for i in range(1, 300):
            dx[i] = 0.8 * dx[i - 1] + noise[i]
        close = pd.Series(100.0 + np.cumsum(dx))
        result = hurst_exponent(close, window=200, min_lag=2, max_lag=10)
        valid = result.dropna()
        assert valid.iloc[-1] > 0.6

    def test_output_clipped(self):
        _, _, close = _ohlc(300)
        result = hurst_exponent(close, window=200, min_lag=2, max_lag=10)
        valid = result.dropna()
        assert (valid >= 0.05).all() and (valid <= 0.95).all()

    def test_nan_before_warmup(self):
        _, _, close = _ohlc(300)
        result = hurst_exponent(close, window=200, min_lag=2, max_lag=10)
        assert result.iloc[:199].isna().all()


# ---------------------------------------------------------------------------
# relative_volume
# ---------------------------------------------------------------------------

class TestRelativeVolume:
    def test_average_volume_gives_one(self):
        # Constant volume → relative volume = 1.0 after warmup
        vol = pd.Series(np.ones(50) * 1_000_000.0)
        result = relative_volume(vol, window=20)
        assert result.iloc[20:].to_numpy() == pytest.approx(np.ones(30))

    def test_spike_above_one(self):
        vol = pd.Series([1_000_000.0] * 50 + [5_000_000.0])
        result = relative_volume(vol, window=20)
        assert result.iloc[-1] > 1.0

    def test_output_length(self):
        vol = _volume(100)
        result = relative_volume(vol, window=20)
        assert len(result) == 100


# ---------------------------------------------------------------------------
# rsi_change_rate
# ---------------------------------------------------------------------------

class TestRsiChangeRate:
    def test_output_length(self):
        _, _, close = _ohlc(100)
        result = rsi_change_rate(close)
        assert len(result) == 100

    def test_accelerating_rally_positive(self):
        # Exponentially rising prices → RSI accelerating upward → positive rate
        close = pd.Series(100.0 * np.exp(np.linspace(0, 0.5, 100)))
        result = rsi_change_rate(close)
        # After warmup, most of the early acceleration period should be positive
        assert result.iloc[30:].mean() >= 0

    def test_flat_prices_near_zero(self):
        close = pd.Series(np.ones(100) * 100.0)
        result = rsi_change_rate(close)
        # Flat: RSI stays constant → rate of change = 0
        assert result.iloc[20:].abs().max() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Alpha158 gap-fill indicators
# ---------------------------------------------------------------------------


def _open(n: int = 300, seed: int = 42) -> pd.Series:
    """Return an open-price series aligned with _ohlc."""
    high, low, close = _ohlc(n, seed)
    return (high + low) / 2.0  # rough proxy


# ---------------------------------------------------------------------------
# macd
# ---------------------------------------------------------------------------

class TestMACD:
    def test_columns(self):
        _, _, close = _ohlc(100)
        result = macd(close)
        assert list(result.columns) == ["macd", "signal", "histogram"]
        assert len(result) == 100

    def test_histogram_identity(self):
        _, _, close = _ohlc(200)
        result = macd(close)
        expected = result["macd"] - result["signal"]
        pd.testing.assert_series_equal(result["histogram"], expected, check_names=False)

    def test_rising_market_positive_macd(self):
        close = pd.Series(np.linspace(100, 200, 200))
        result = macd(close)
        assert result["macd"].iloc[-1] > 0


# ---------------------------------------------------------------------------
# bollinger_pctb
# ---------------------------------------------------------------------------

class TestBollingerPctB:
    def test_output_length(self):
        _, _, close = _ohlc(100)
        result = bollinger_pctb(close, window=20)
        assert len(result) == 100

    def test_range_after_warmup(self):
        _, _, close = _ohlc(200)
        result = bollinger_pctb(close, window=20)
        valid = result.iloc[20:]
        # %B can exceed [0,1] but should be finite
        assert np.isfinite(valid).all()

    def test_zero_std_returns_default(self):
        close = pd.Series(np.ones(30) * 100.0)
        result = bollinger_pctb(close, window=20)
        # Zero std → bands collapse → fillna(0.5)
        assert result.iloc[19:].eq(0.5).all()


# ---------------------------------------------------------------------------
# bollinger_bandwidth
# ---------------------------------------------------------------------------

class TestBollingerBandwidth:
    def test_output_length(self):
        _, _, close = _ohlc(100)
        result = bollinger_bandwidth(close, window=20)
        assert len(result) == 100

    def test_positive_after_warmup(self):
        _, _, close = _ohlc(200)
        result = bollinger_bandwidth(close, window=20)
        valid = result.iloc[20:]
        assert (valid >= 0).all()

    def test_zero_std_returns_zero(self):
        close = pd.Series(np.ones(30) * 100.0)
        result = bollinger_bandwidth(close, window=20)
        assert result.iloc[19:].eq(0.0).all()


# ---------------------------------------------------------------------------
# daily_price_range
# ---------------------------------------------------------------------------

class TestDailyPriceRange:
    def test_output_length(self):
        high, low, close = _ohlc(100)
        result = daily_price_range(high, low, close)
        assert len(result) == 100

    def test_positive(self):
        high, low, close = _ohlc(100)
        result = daily_price_range(high, low, close)
        assert (result >= 0).all()

    def test_zero_close_safe(self):
        high = pd.Series([10.0, 10.0])
        low = pd.Series([5.0, 5.0])
        close = pd.Series([0.0, 100.0])
        result = daily_price_range(high, low, close)
        assert np.isfinite(result).all()
        assert result.iloc[0] == 0.0  # fillna


# ---------------------------------------------------------------------------
# volume_std
# ---------------------------------------------------------------------------

class TestVolumeStd:
    def test_output_length(self):
        vol = _volume(100)
        result = volume_std(vol, window=20)
        assert len(result) == 100

    def test_fillna_early(self):
        vol = _volume(100)
        result = volume_std(vol, window=20)
        # First 19 values should be 0 (fillna)
        assert result.iloc[:19].eq(0.0).all()

    def test_constant_volume_zero_std(self):
        vol = pd.Series(np.ones(30) * 1_000_000.0)
        result = volume_std(vol, window=20)
        assert result.iloc[19:].eq(0.0).all()


# ---------------------------------------------------------------------------
# price_volume_corr
# ---------------------------------------------------------------------------

class TestPriceVolumeCorr:
    def test_output_length(self):
        _, _, close = _ohlc(100)
        vol = _volume(100)
        result = price_volume_corr(close, vol, window=10)
        assert len(result) == 100

    def test_range(self):
        _, _, close = _ohlc(200)
        vol = _volume(200)
        result = price_volume_corr(close, vol, window=10)
        valid = result.iloc[10:]
        assert (valid >= -1.001).all() and (valid <= 1.001).all()

    def test_fillna_early(self):
        _, _, close = _ohlc(100)
        vol = _volume(100)
        result = price_volume_corr(close, vol, window=10)
        # NaN-safe: no NaN in output
        assert not result.isna().any()


# ---------------------------------------------------------------------------
# close_open_ratio
# ---------------------------------------------------------------------------

class TestCloseOpenRatio:
    def test_output_length(self):
        _, _, close = _ohlc(100)
        op = _open(100)
        result = close_open_ratio(close, op)
        assert len(result) == 100

    def test_zero_open_safe(self):
        close = pd.Series([100.0, 200.0])
        op = pd.Series([0.0, 100.0])
        result = close_open_ratio(close, op)
        assert np.isfinite(result).all()
        assert result.iloc[0] == 1.0  # fillna

    def test_same_close_open_gives_one(self):
        s = pd.Series([100.0, 200.0, 300.0])
        result = close_open_ratio(s, s)
        pd.testing.assert_series_equal(result, pd.Series([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# upper_shadow_pct
# ---------------------------------------------------------------------------

class TestUpperShadowPct:
    def test_output_length(self):
        high, low, close = _ohlc(100)
        op = _open(100)
        result = upper_shadow_pct(high, op, close)
        assert len(result) == 100

    def test_non_negative(self):
        high, low, close = _ohlc(100)
        op = _open(100)
        result = upper_shadow_pct(high, op, close)
        assert (result >= -1e-10).all()

    def test_zero_close_safe(self):
        high = pd.Series([10.0])
        op = pd.Series([5.0])
        close = pd.Series([0.0])
        result = upper_shadow_pct(high, op, close)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# lower_shadow_pct
# ---------------------------------------------------------------------------

class TestLowerShadowPct:
    def test_output_length(self):
        high, low, close = _ohlc(100)
        op = _open(100)
        result = lower_shadow_pct(low, op, close)
        assert len(result) == 100

    def test_non_negative(self):
        high, low, close = _ohlc(100)
        op = _open(100)
        result = lower_shadow_pct(low, op, close)
        assert (result >= -1e-10).all()

    def test_zero_close_safe(self):
        low = pd.Series([1.0])
        op = pd.Series([5.0])
        close = pd.Series([0.0])
        result = lower_shadow_pct(low, op, close)
        assert np.isfinite(result).all()


# ---------------------------------------------------------------------------
# rolling_quantile_ratio
# ---------------------------------------------------------------------------

class TestRollingQuantileRatio:
    def test_output_length(self):
        _, _, close = _ohlc(100)
        result = rolling_quantile_ratio(close, window=20)
        assert len(result) == 100

    def test_default_fill(self):
        _, _, close = _ohlc(100)
        result = rolling_quantile_ratio(close, window=20)
        # No NaN in output
        assert not result.isna().any()

    def test_constant_price_returns_default(self):
        close = pd.Series(np.ones(30) * 100.0)
        result = rolling_quantile_ratio(close, window=20)
        # Constant price → q_hi == q_lo → fillna(0.5)
        assert result.iloc[19:].eq(0.5).all()
