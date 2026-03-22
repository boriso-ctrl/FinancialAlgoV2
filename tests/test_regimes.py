"""Tests for regime detection module."""

import numpy as np
import pandas as pd
import pytest

from financial_algo.regimes import Regime, RegimeConfig, detect_regime, is_crisis


def _make_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic price DataFrame for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["SPY", "USO", "ITA", "HYG", "LQD", "XLE", "XLK", "XLF"]
    data = {}
    for t in tickers:
        returns = rng.normal(0.0005, 0.015, n)
        data[t] = 100.0 * np.exp(np.cumsum(returns))
    return pd.DataFrame(data, index=dates)


class TestRegimeDetection:
    def test_returns_series_of_regimes(self):
        prices = _make_prices()
        result = detect_regime(prices)
        assert isinstance(result, pd.Series)
        assert len(result) == len(prices)
        # All values should be valid Regime members
        for val in result.unique():
            assert isinstance(val, Regime)

    def test_default_config(self):
        cfg = RegimeConfig()
        assert cfg.vix_elevated == 20.0
        assert cfg.vix_crisis == 30.0

    def test_custom_config(self):
        cfg = RegimeConfig(vix_elevated=15.0, vix_crisis=25.0)
        prices = _make_prices()
        result = detect_regime(prices, config=cfg)
        assert len(result) == len(prices)

    def test_with_vix(self):
        prices = _make_prices()
        # Create a VIX series that spikes mid-way
        vix = pd.Series(15.0, index=prices.index)
        vix.iloc[150:200] = 35.0  # crisis-level VIX
        result = detect_regime(prices, vix=vix)
        assert len(result) == len(prices)

    def test_is_crisis_helper(self):
        assert is_crisis(Regime.OIL_CRISIS)
        assert is_crisis(Regime.WAR_CRISIS)
        assert is_crisis(Regime.GENERAL_CRISIS)
        assert not is_crisis(Regime.NORMAL)
        assert not is_crisis(Regime.ELEVATED)
        assert not is_crisis(Regime.RECOVERY)
