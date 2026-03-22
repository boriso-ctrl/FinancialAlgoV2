"""Tests for backtest engine and performance metrics."""

import numpy as np
import pandas as pd
import pytest

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics


def _make_prices(n: int = 252, seed: int = 42) -> pd.DataFrame:
    """Synthetic prices — two assets."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    spy = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n)))
    tlt = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.006, n)))
    return pd.DataFrame({"SPY": spy, "TLT": tlt}, index=dates)


def _make_weights(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple buy-and-hold SPY weights (already shifted +1)."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w["SPY"] = 1.0
    return w


class TestBacktest:
    def test_returns_dict_with_required_keys(self):
        prices = _make_prices()
        weights = _make_weights(prices)
        result = backtest(prices, weights)
        assert "equity" in result
        assert "returns" in result
        assert "weights" in result
        assert "metrics" in result

    def test_equity_shape(self):
        prices = _make_prices()
        weights = _make_weights(prices)
        result = backtest(prices, weights)
        assert len(result["equity"]) == len(prices)

    def test_equity_starts_at_initial_capital(self):
        prices = _make_prices()
        weights = _make_weights(prices)
        cfg = BacktestConfig(initial_capital=500_000.0)
        result = backtest(prices, weights, config=cfg)
        assert result["equity"].iloc[0] == pytest.approx(500_000.0, rel=1e-2)

    def test_leveraged_weights(self):
        prices = _make_prices()
        weights = _make_weights(prices)
        weights["SPY"] = 2.0  # 2x leverage
        result = backtest(prices, weights)
        assert result["metrics"]["annual_vol"] > 0

    def test_short_weights(self):
        prices = _make_prices()
        weights = _make_weights(prices)
        weights["SPY"] = -1.0  # short
        result = backtest(prices, weights)
        assert len(result["equity"]) == len(prices)

    def test_vol_target_overlay(self):
        prices = _make_prices()
        weights = _make_weights(prices)
        weights["SPY"] = 2.0
        cfg = BacktestConfig(vol_target=0.10)
        result = backtest(prices, weights, config=cfg)
        assert result["metrics"]["annual_vol"] > 0

    def test_drawdown_control(self):
        prices = _make_prices()
        weights = _make_weights(prices)
        cfg = BacktestConfig(max_drawdown_trigger=-0.05)
        result = backtest(prices, weights, config=cfg)
        assert len(result["equity"]) == len(prices)


class TestComputeMetrics:
    def test_basic_metrics(self):
        rng = np.random.RandomState(42)
        returns = pd.Series(rng.normal(0.0005, 0.01, 252))
        metrics = compute_metrics(returns)
        assert "cagr" in metrics
        assert "sharpe" in metrics
        assert "sortino" in metrics
        assert "max_drawdown" in metrics
        assert "calmar" in metrics
        assert "win_rate" in metrics
        assert "annual_vol" in metrics
        assert "total_return" in metrics

    def test_positive_sharpe_for_positive_returns(self):
        rng = np.random.RandomState(99)
        returns = pd.Series(rng.normal(0.01, 0.005, 252))  # positive mean, small std
        metrics = compute_metrics(returns)
        assert metrics["sharpe"] > 0

    def test_empty_returns(self):
        returns = pd.Series([0.0])
        metrics = compute_metrics(returns)
        assert metrics == {}
