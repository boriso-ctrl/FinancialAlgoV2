"""Tests for financial_algo.indicators."""

import math
import pytest

from financial_algo.indicators import bollinger_bands, moving_average, rsi


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
