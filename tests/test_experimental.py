"""Tests for experimental strategies -- valid weights, no NaN, proper shapes."""

import numpy as np
import pandas as pd
import pytest


def _make_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic prices with all tickers needed by experimental strategies."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = [
        "SPY", "QQQ", "IWM", "TLT", "GLD", "IEF", "UUP",
        "XLE", "USO", "XOP", "XLK", "XLF", "XLI", "XLB",
        "XLP", "XLU", "XLY", "XLV",
        "ITA", "LMT", "RTX",
        "EFA", "EEM",
        "HYG", "LQD",
        "BTC-USD",
    ]
    data = {}
    for t in tickers:
        ret = rng.normal(0.0003, 0.015, n)
        data[t] = 100 * np.exp(np.cumsum(ret))
    return pd.DataFrame(data, index=dates)


def _make_regime(prices: pd.DataFrame) -> pd.Series:
    from financial_algo.regimes import Regime
    n = len(prices)
    cycle = (
        [Regime.NORMAL] * 60
        + [Regime.ELEVATED] * 40
        + [Regime.OIL_CRISIS] * 30
        + [Regime.WAR_CRISIS] * 30
        + [Regime.GENERAL_CRISIS] * 30
        + [Regime.RECOVERY] * 40
    )
    regimes = (cycle * ((n // len(cycle)) + 1))[:n]
    return pd.Series(regimes, index=prices.index)


# =========================================================================
# Category X -- Cross-Asset Divergence
# =========================================================================

class TestCrossAssetStrategies:
    def test_copper_gold_growth(self):
        from experimental.strategies.cross_asset import CopperGoldGrowth
        prices = _make_prices()
        strat = CopperGoldGrowth()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"
        assert w.columns.tolist() == prices.columns.tolist()

    def test_credit_equity_divergence(self):
        from experimental.strategies.cross_asset import CreditEquityDivergence
        prices = _make_prices()
        strat = CreditEquityDivergence()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_dollar_wrecking_ball(self):
        from experimental.strategies.cross_asset import DollarWreckingBall
        prices = _make_prices()
        strat = DollarWreckingBall()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_copper_gold_missing_tickers(self):
        """Should return zeros if required tickers are missing."""
        from experimental.strategies.cross_asset import CopperGoldGrowth
        prices = _make_prices()[["SPY", "QQQ"]]
        strat = CopperGoldGrowth()
        w = strat.generate_weights(prices)
        assert (w == 0.0).all().all()

    def test_strategies_have_names(self):
        from experimental.strategies.cross_asset import (
            CopperGoldGrowth, CreditEquityDivergence, DollarWreckingBall,
        )
        assert CopperGoldGrowth().name == "X1-CopperGoldGrowth"
        assert CreditEquityDivergence().name == "X2-CreditEquityDivergence"
        assert DollarWreckingBall().name == "X3-DollarWreckingBall"


# =========================================================================
# Category Y -- Behavioral Anomaly
# =========================================================================

class TestBehavioralStrategies:
    def test_disposition_reversal(self):
        from experimental.strategies.behavioral import DispositionEffectReversal
        prices = _make_prices()
        strat = DispositionEffectReversal()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_attention_overreaction(self):
        from experimental.strategies.behavioral import AttentionOverreaction
        prices = _make_prices()
        strat = AttentionOverreaction()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_lunar_cycle_alpha(self):
        from experimental.strategies.behavioral import LunarCycleAlpha
        prices = _make_prices()
        strat = LunarCycleAlpha()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_strategies_have_names(self):
        from experimental.strategies.behavioral import (
            DispositionEffectReversal, AttentionOverreaction, LunarCycleAlpha,
        )
        assert DispositionEffectReversal().name == "Y1-DispositionReversal"
        assert AttentionOverreaction().name == "Y2-AttentionOverreaction"
        assert LunarCycleAlpha().name == "Y3-LunarCycleAlpha"


# =========================================================================
# Category Z -- Structural Alpha
# =========================================================================

class TestStructuralStrategies:
    def test_rebalancing_flow(self):
        from experimental.strategies.structural import RebalancingFlow
        prices = _make_prices()
        strat = RebalancingFlow()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_gamma_pin(self):
        from experimental.strategies.structural import GammaPin
        prices = _make_prices()
        strat = GammaPin()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_sector_dispersion(self):
        from experimental.strategies.structural import SectorDispersion
        prices = _make_prices()
        strat = SectorDispersion()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_strategies_have_names(self):
        from experimental.strategies.structural import (
            RebalancingFlow, GammaPin, SectorDispersion,
        )
        assert RebalancingFlow().name == "Z1-RebalancingFlow"
        assert GammaPin().name == "Z2-GammaPin"
        assert SectorDispersion().name == "Z3-SectorDispersion"


# =========================================================================
# Category W -- Alternative Signals
# =========================================================================

class TestAltSignalStrategies:
    def test_breadth_divergence(self):
        from experimental.strategies.alt_signals import BreadthDivergence
        prices = _make_prices()
        strat = BreadthDivergence()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_correlation_regime_break(self):
        from experimental.strategies.alt_signals import CorrelationRegimeBreak
        prices = _make_prices()
        strat = CorrelationRegimeBreak()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_liquidity_vacuum(self):
        from experimental.strategies.alt_signals import LiquidityVacuum
        prices = _make_prices()
        strat = LiquidityVacuum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "Weights contain NaN"

    def test_strategies_have_names(self):
        from experimental.strategies.alt_signals import (
            BreadthDivergence, CorrelationRegimeBreak, LiquidityVacuum,
        )
        assert BreadthDivergence().name == "W1-BreadthDivergence"
        assert CorrelationRegimeBreak().name == "W2-CorrelationRegimeBreak"
        assert LiquidityVacuum().name == "W3-LiquidityVacuum"


# =========================================================================
# Package-level imports
# =========================================================================

class TestExperimentalPackage:
    def test_all_experimental_list(self):
        from experimental.strategies import ALL_EXPERIMENTAL
        assert len(ALL_EXPERIMENTAL) == 12
        names = [s.name for s in ALL_EXPERIMENTAL]
        assert "X1-CopperGoldGrowth" in names
        assert "Y1-DispositionReversal" in names  # class is DispositionEffectReversal
        assert "Z1-RebalancingFlow" in names
        assert "W1-BreadthDivergence" in names

    def test_all_strategies_are_strategy_instances(self):
        from financial_algo.strategies.base import Strategy
        from experimental.strategies import ALL_EXPERIMENTAL
        for s in ALL_EXPERIMENTAL:
            assert isinstance(s, Strategy), f"{s.name} is not a Strategy"

    def test_all_strategies_have_unique_names(self):
        from experimental.strategies import ALL_EXPERIMENTAL
        names = [s.name for s in ALL_EXPERIMENTAL]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"

    def test_all_strategies_produce_valid_weights(self):
        """Smoke test: every strategy produces valid weights on synthetic data."""
        from experimental.strategies import ALL_EXPERIMENTAL
        prices = _make_prices()
        for strat in ALL_EXPERIMENTAL:
            w = strat.generate_weights(prices)
            assert isinstance(w, pd.DataFrame), f"{strat.name} did not return DataFrame"
            assert len(w) == len(prices), f"{strat.name} wrong length"
            assert not w.isna().any().any(), f"{strat.name} has NaN in weights"
