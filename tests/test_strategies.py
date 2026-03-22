"""Tests for strategy bots — valid weights, no NaN, proper shapes."""

import numpy as np
import pandas as pd
import pytest

from financial_algo.regimes import Regime


def _make_prices(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Synthetic prices with all tickers needed by strategies."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = [
        "SPY", "QQQ", "IWM", "TLT", "GLD", "IEF", "UUP",
        "XLE", "USO", "XOP", "OIH", "XLK", "XLF", "XLI", "XLB",
        "XLP", "XLU", "XLY", "XLV", "XLRE", "XLC",
        "ITA", "LMT", "RTX", "NOC", "GD",
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
    """Synthetic regime series cycling through regimes."""
    n = len(prices)
    cycle = [Regime.NORMAL] * 60 + [Regime.ELEVATED] * 40 + \
            [Regime.OIL_CRISIS] * 30 + [Regime.WAR_CRISIS] * 30 + \
            [Regime.GENERAL_CRISIS] * 30 + [Regime.RECOVERY] * 40
    regimes = (cycle * ((n // len(cycle)) + 1))[:n]
    return pd.Series(regimes, index=prices.index)


class TestCrashHedgeStrategies:
    def test_four_state_tactical(self):
        from financial_algo.strategies.crash_hedge import FourStateTactical
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = FourStateTactical()
        w = strat.generate_weights(prices, regime)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_crash_hedge_qqq(self):
        from financial_algo.strategies.crash_hedge import CrashHedgeQQQ
        prices = _make_prices()
        strat = CrashHedgeQQQ()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)

    def test_vol_carry(self):
        from financial_algo.strategies.crash_hedge import VolCarry
        prices = _make_prices()
        strat = VolCarry()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)


class TestOilCrisisStrategies:
    def test_oil_momentum_surge(self):
        from financial_algo.strategies.oil_crisis import OilMomentumSurge
        prices = _make_prices()
        strat = OilMomentumSurge()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_oil_shock_hedge(self):
        from financial_algo.strategies.oil_crisis import OilShockHedge
        prices = _make_prices()
        strat = OilShockHedge()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)

    def test_oil_mean_reversion(self):
        from financial_algo.strategies.oil_crisis import OilMeanReversion
        prices = _make_prices()
        strat = OilMeanReversion()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)

    def test_energy_pairs(self):
        from financial_algo.strategies.oil_crisis import EnergyPairs
        prices = _make_prices()
        strat = EnergyPairs()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)


class TestWarCrisisStrategies:
    def test_defense_rotation(self):
        from financial_algo.strategies.war_crisis import DefenseRotation
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = DefenseRotation()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_safe_haven_flight(self):
        from financial_algo.strategies.war_crisis import SafeHavenFlight
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = SafeHavenFlight()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)

    def test_post_war_recovery(self):
        from financial_algo.strategies.war_crisis import PostWarRecovery
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = PostWarRecovery()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)

    def test_arms_race_momentum(self):
        from financial_algo.strategies.war_crisis import ArmsRaceMomentum
        prices = _make_prices()
        strat = ArmsRaceMomentum()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)


class TestPairsStrategies:
    def test_single_pair(self):
        from financial_algo.strategies.pairs import PairSpec, PairTrade
        prices = _make_prices()
        pair = PairSpec("XLE", "XLK")
        strat = PairTrade(pair)
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)
        assert "XLE" in w.columns
        assert "XLK" in w.columns

    def test_multi_pair_portfolio(self):
        from financial_algo.strategies.pairs import MultiPairPortfolio
        prices = _make_prices()
        strat = MultiPairPortfolio()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)


class TestCryptoCrisisStrategies:
    def test_crypto_flight_to_quality(self):
        from financial_algo.strategies.crypto_crisis import CryptoFlightToQuality
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = CryptoFlightToQuality()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)

    def test_crypto_recovery_surge(self):
        from financial_algo.strategies.crypto_crisis import CryptoRecoverySurge
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = CryptoRecoverySurge()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)

    def test_crypto_gold_divergence(self):
        from financial_algo.strategies.crypto_crisis import CryptoGoldDivergence
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = CryptoGoldDivergence()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)


class TestEnsembleStrategy:
    def test_ensemble_combines_strategies(self):
        from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
        from financial_algo.strategies.ensemble import EnsembleStrategy
        prices = _make_prices()
        strats = [CrashHedgeQQQ(), VolCarry()]
        ensemble = EnsembleStrategy(strats)
        w = ensemble.generate_weights(prices)
        assert len(w) == len(prices)

    def test_backtest_weights_shift(self):
        from financial_algo.strategies.crash_hedge import VolCarry
        prices = _make_prices()
        strat = VolCarry()
        raw = strat.generate_weights(prices)
        shifted = strat.backtest_weights(prices)
        # Shifted first row should be 0 (due to shift+1)
        assert (shifted.iloc[0] == 0.0).all()


class TestCrisisSpikeStrategies:
    """Cat H: Crisis Spike (Upward) strategies."""

    def test_commodity_shock_rider(self):
        from financial_algo.strategies.crisis_spike import CommodityShockRider
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = CommodityShockRider()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)
        assert not w.isna().any().any()
        # Should only have positive weights (pure long)
        assert (w >= 0).all().all()

    def test_commodity_shock_rider_no_regime(self):
        from financial_algo.strategies.crisis_spike import CommodityShockRider
        prices = _make_prices()
        strat = CommodityShockRider()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)

    def test_gold_fear_rally(self):
        from financial_algo.strategies.crisis_spike import GoldFearRally
        prices = _make_prices()
        strat = GoldFearRally()
        w = strat.generate_weights(prices)
        assert len(w) == len(prices)
        assert not w.isna().any().any()
        assert (w >= 0).all().all()

    def test_defense_spike_breakout(self):
        from financial_algo.strategies.crisis_spike import DefenseSpikeBreakout
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = DefenseSpikeBreakout()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)
        assert not w.isna().any().any()
        assert (w >= 0).all().all()

    def test_multi_asset_crisis_long(self):
        from financial_algo.strategies.crisis_spike import MultiAssetCrisisLong
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = MultiAssetCrisisLong()
        w = strat.generate_weights(prices, regime)
        assert len(w) == len(prices)
        assert not w.isna().any().any()
        # Should only allocate positive weights (pure long)
        assert (w >= 0).all().all()

    def test_multi_asset_adapts_to_strongest(self):
        """Verify strategy allocates to highest-momentum assets."""
        from financial_algo.strategies.crisis_spike import MultiAssetCrisisLong
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = MultiAssetCrisisLong()
        w = strat.generate_weights(prices, regime)
        # On active days, at most top_n assets should have weight
        active_days = (w > 0).any(axis=1)
        if active_days.any():
            max_active = (w.loc[active_days] > 0).sum(axis=1).max()
            assert max_active <= strat.cfg.top_n

    def test_commodity_shock_fast_zscore_trigger(self):
        """H1 should fire on fast 20-day z-score even without 60-day data."""
        from financial_algo.strategies.crisis_spike import (
            CommodityShockRider,
            CommodityShockConfig,
        )
        # Short window (40 days) — 60-day z-score will be NaN throughout
        np.random.seed(99)
        n = 40
        idx = pd.bdate_range("2025-05-01", periods=n)
        # Build a price series where USO spikes sharply mid-window
        uso_prices = np.concatenate([
            np.full(15, 30.0),              # flat
            np.linspace(30.0, 42.0, 15),    # +40% surge over 15 days
            np.full(10, 42.0),              # hold level
        ])
        prices = pd.DataFrame({
            "USO": uso_prices, "XLE": 80.0, "GLD": 180.0,
        }, index=idx)
        cfg = CommodityShockConfig(
            fast_zscore_window=10,      # narrow lookback
            fast_spike_z=1.5,           # lower bar for test
            fast_momentum_threshold=0.03,
        )
        strat = CommodityShockRider(cfg)
        w = strat.generate_weights(prices)
        # The spike should trigger at least some positive weights
        assert (w["XLE"] > 0).any(), "Fast z-score trigger should fire on short window"

    def test_multi_asset_extreme_zscore_bypasses_regime(self):
        """H4 should trade when z-score is extreme even if regime is CALM."""
        from financial_algo.strategies.crisis_spike import MultiAssetCrisisLong
        # Build prices where XLE has a massive spike
        np.random.seed(77)
        n = 200
        idx = pd.bdate_range("2024-01-01", periods=n)
        xle = np.concatenate([np.full(150, 80.0), np.linspace(80.0, 120.0, 50)])
        prices = pd.DataFrame({
            "XLE": xle, "GLD": 180.0, "ITA": 100.0,
            "TLT": 90.0, "UUP": 25.0, "SPY": 450.0,
        }, index=idx)
        # All NORMAL regime — normally H4 would sit out
        from financial_algo.regimes import Regime
        regime = pd.Series(Regime.NORMAL, index=idx)
        strat = MultiAssetCrisisLong()
        w = strat.generate_weights(prices, regime)
        # Extreme z-score bypass should allow some trades
        assert (w["XLE"] > 0).any(), "Extreme z-score should bypass CALM regime gate"


# =========================================================================
# Cat I: Momentum Strategies
# =========================================================================

class TestMomentumStrategies:
    def test_time_series_momentum(self):
        from financial_algo.strategies.momentum import TimeSeriesMomentum
        prices = _make_prices(n=400)
        strat = TimeSeriesMomentum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_cross_sectional_momentum(self):
        from financial_algo.strategies.momentum import CrossSectionalMomentum
        prices = _make_prices(n=300)
        strat = CrossSectionalMomentum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_dual_momentum(self):
        from financial_algo.strategies.momentum import DualMomentum
        prices = _make_prices(n=400)
        strat = DualMomentum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)


# =========================================================================
# Cat J: Mean Reversion Strategies
# =========================================================================

class TestMeanReversionStrategies:
    def test_sector_mean_reversion(self):
        from financial_algo.strategies.mean_reversion import SectorMeanReversion
        prices = _make_prices(n=300)
        strat = SectorMeanReversion()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)

    def test_rsi_mean_reversion(self):
        from financial_algo.strategies.mean_reversion import RSIMeanReversion
        prices = _make_prices(n=300)
        strat = RSIMeanReversion()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)


# =========================================================================
# Cat H-FI: Fixed Income Strategies
# =========================================================================

class TestFixedIncomeStrategies:
    def test_yield_curve_trade(self):
        from financial_algo.strategies.fixed_income import YieldCurveTrade
        prices = _make_prices()
        strat = YieldCurveTrade()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()
        # Long-only: no negative weights
        assert (w >= 0).all().all()

    def test_yield_curve_always_invested(self):
        """H1 should always be invested in either TLT or IEF."""
        from financial_algo.strategies.fixed_income import YieldCurveTrade
        prices = _make_prices(n=400)
        strat = YieldCurveTrade()
        w = strat.generate_weights(prices)
        # After warm-up, total weight should be > 0 on every day
        total_weight = w.sum(axis=1)
        assert (total_weight.iloc[200:] > 0).all()

    def test_credit_spread_mean_rev(self):
        from financial_algo.strategies.fixed_income import CreditSpreadMeanRev
        prices = _make_prices()
        strat = CreditSpreadMeanRev()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()
        # Long-only: no negative weights
        assert (w >= 0).all().all()

    def test_credit_spread_no_inf(self):
        """H2 should handle division safely."""
        from financial_algo.strategies.fixed_income import CreditSpreadMeanRev
        prices = _make_prices()
        strat = CreditSpreadMeanRev()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()


# =========================================================================
# Cat K: Factor Strategies
# =========================================================================

class TestFactorStrategies:
    def test_low_vol_factor(self):
        from financial_algo.strategies.factor import LowVolFactor
        prices = _make_prices()
        strat = LowVolFactor()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        # Low-vol should be long-only
        assert (w >= 0).all().all()

    def test_multi_factor_composite(self):
        from financial_algo.strategies.factor import MultiFactorComposite
        prices = _make_prices()
        strat = MultiFactorComposite()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)


# =========================================================================
# Cat L: Volatility Strategies
# =========================================================================

class TestVolatilityStrategies:
    def test_vol_risk_premium(self):
        from financial_algo.strategies.volatility_strats import VolRiskPremium
        prices = _make_prices()
        strat = VolRiskPremium()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


# =========================================================================
# Cat M: Macro Strategies
# =========================================================================

class TestMacroStrategies:
    def test_dollar_carry(self):
        from financial_algo.strategies.macro import DollarCarry
        prices = _make_prices()
        strat = DollarCarry()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_gold_dollar_inverse(self):
        from financial_algo.strategies.macro import GoldDollarInverse
        prices = _make_prices()
        strat = GoldDollarInverse()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


# =========================================================================
# Cat N: Seasonal Strategies
# =========================================================================

class TestSeasonalStrategies:
    def test_seasonal_strategy(self):
        from financial_algo.strategies.seasonal import SeasonalStrategy
        prices = _make_prices()
        strat = SeasonalStrategy()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_turn_of_month(self):
        from financial_algo.strategies.seasonal import TurnOfMonth
        prices = _make_prices()
        strat = TurnOfMonth()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_pre_holiday_drift(self):
        from financial_algo.strategies.seasonal import PreHolidayDrift
        prices = _make_prices()
        strat = PreHolidayDrift()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


# =========================================================================
# New Strategy Tests — J2, J4, L2, L4, I4, M3, M4, K3, K4, H3-FI, O1-O4, P1
# =========================================================================

class TestOvernightGapFade:
    def test_basic(self):
        from financial_algo.strategies.mean_reversion import OvernightGapFade
        prices = _make_prices(n=300)
        strat = OvernightGapFade()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestCointegrationPairs:
    def test_basic(self):
        from financial_algo.strategies.mean_reversion import CointegrationPairs
        prices = _make_prices(n=300)
        strat = CointegrationPairs()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)


class TestVolSpreadHarvest:
    def test_basic(self):
        from financial_algo.strategies.volatility_strats import VolSpreadHarvest
        prices = _make_prices()
        strat = VolSpreadHarvest()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestVolOfVolRegime:
    def test_basic(self):
        from financial_algo.strategies.volatility_strats import VolOfVolRegime
        prices = _make_prices()
        strat = VolOfVolRegime()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestMomentumVolScaled:
    def test_basic(self):
        from financial_algo.strategies.momentum import MomentumVolScaled
        prices = _make_prices(n=300)
        strat = MomentumVolScaled()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestEMRiskPremium:
    def test_basic(self):
        from financial_algo.strategies.macro import EMRiskPremium
        prices = _make_prices()
        strat = EMRiskPremium()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestCommodityMomentum:
    def test_basic(self):
        from financial_algo.strategies.macro import CommodityMomentum
        prices = _make_prices()
        strat = CommodityMomentum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestSizeFactor:
    def test_basic(self):
        from financial_algo.strategies.factor import SizeFactor
        prices = _make_prices()
        strat = SizeFactor()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestValueFactor:
    def test_basic(self):
        from financial_algo.strategies.factor import ValueFactor
        prices = _make_prices()
        strat = ValueFactor()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)


class TestDurationTiming:
    def test_basic(self):
        from financial_algo.strategies.fixed_income import DurationTiming
        prices = _make_prices()
        strat = DurationTiming()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestTailRiskParity:
    def test_basic(self):
        from financial_algo.strategies.tail_risk import TailRiskParity
        prices = _make_prices()
        strat = TailRiskParity()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        # Should be long-only (no shorts)
        assert (w >= 0).all().all()


class TestCrisisAlphaMomentum:
    def test_basic(self):
        from financial_algo.strategies.tail_risk import CrisisAlphaMomentum
        prices = _make_prices()
        strat = CrisisAlphaMomentum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestBlackSwanInsurance:
    def test_basic(self):
        from financial_algo.strategies.tail_risk import BlackSwanInsurance
        prices = _make_prices()
        strat = BlackSwanInsurance()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestFeatureComboSignal:
    def test_basic(self):
        from financial_algo.strategies.signal_combo import FeatureComboSignal
        prices = _make_prices(n=300)
        strat = FeatureComboSignal()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestRatesRegimeTrade:
    def test_basic(self):
        from financial_algo.strategies.macro import RatesRegimeTrade
        prices = _make_prices(n=400)
        strat = RatesRegimeTrade()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()
        # Long-only: no negative weights
        assert (w >= 0).all().all()

    def test_always_invested(self):
        """M5 should always be invested in either SPY or TLT."""
        from financial_algo.strategies.macro import RatesRegimeTrade
        prices = _make_prices(n=400)
        strat = RatesRegimeTrade()
        w = strat.generate_weights(prices)
        total_weight = w.sum(axis=1)
        # After warm-up period, always invested
        assert (total_weight.iloc[200:] > 0).all()

    def test_no_inf(self):
        from financial_algo.strategies.macro import RatesRegimeTrade
        prices = _make_prices(n=400)
        strat = RatesRegimeTrade()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()


# =========================================================================
# Cat L: New Volatility Strategies (L3, L5)
# =========================================================================

class TestVolTermStructure:
    def test_basic(self):
        from financial_algo.strategies.volatility_strats import VolTermStructure
        prices = _make_prices()
        strat = VolTermStructure()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.volatility_strats import VolTermStructure
        prices = _make_prices()
        strat = VolTermStructure()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.volatility_strats import VolTermStructure
        prices = _make_prices()
        strat = VolTermStructure()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


class TestVolSpikeRecovery:
    def test_basic(self):
        from financial_algo.strategies.volatility_strats import VolSpikeRecovery
        prices = _make_prices()
        strat = VolSpikeRecovery()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.volatility_strats import VolSpikeRecovery
        prices = _make_prices()
        strat = VolSpikeRecovery()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.volatility_strats import VolSpikeRecovery
        prices = _make_prices()
        strat = VolSpikeRecovery()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()
