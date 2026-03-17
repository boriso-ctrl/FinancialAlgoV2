"""Tests for financial_algo.signals."""

import math
import pytest

from financial_algo.signals import crossover_signal


class TestCrossoverSignal:
    def test_buy_signal_on_golden_cross(self):
        # fast crosses from below to above slow
        fast = [1.0, 2.0, 4.0]
        slow = [3.0, 3.0, 3.0]
        signals = crossover_signal(fast, slow)
        assert signals[2] == 1

    def test_sell_signal_on_death_cross(self):
        fast = [4.0, 3.0, 1.0]
        slow = [2.0, 2.0, 2.0]
        signals = crossover_signal(fast, slow)
        assert signals[2] == -1

    def test_no_signal_when_no_crossover(self):
        fast = [5.0, 5.0, 5.0]
        slow = [3.0, 3.0, 3.0]
        signals = crossover_signal(fast, slow)
        assert all(s == 0 for s in signals[1:])

    def test_nan_values_produce_zero_signal(self):
        fast = [float("nan"), 2.0, 4.0]
        slow = [float("nan"), 3.0, 3.0]
        signals = crossover_signal(fast, slow)
        assert signals[1] == 0

    def test_output_length_matches_input(self):
        fast = [1.0, 2.0, 3.0, 4.0, 5.0]
        slow = [2.0, 2.0, 2.0, 2.0, 2.0]
        assert len(crossover_signal(fast, slow)) == len(fast)

    def test_first_element_always_zero(self):
        fast = [1.0, 2.0]
        slow = [2.0, 1.0]
        assert crossover_signal(fast, slow)[0] == 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            crossover_signal([1.0, 2.0], [1.0])

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            crossover_signal([1.0], [1.0])
