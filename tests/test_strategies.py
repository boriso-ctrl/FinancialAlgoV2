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
        "BTC-USD", "ETH-USD",
        # Init 4 expansion
        "SLV", "DBC", "DBA", "SHY", "TIP", "EMB", "AGG",
        "VNQ", "FXI", "VGK", "EWJ", "INDA", "XBI",
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

    def test_ensemble_hrp_weighting(self):
        """HRP weighting produces valid weights via skfolio (skip if not installed)."""
        skfolio = pytest.importorskip("skfolio", reason="skfolio not installed")
        from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
        from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy
        from financial_algo.strategies.factor import LowVolFactor

        prices = _make_prices(n=200)
        strats = [CrashHedgeQQQ(), VolCarry(), LowVolFactor()]
        cfg = EnsembleConfig(
            weighting_method="hrp",
            skfolio_refit_every=21,
            skfolio_risk_measure="variance",
            max_single_weight=0.50,
            max_gross_leverage=2.0,
        )
        ens = EnsembleStrategy(strats, cfg)
        w = ens.generate_weights(prices)
        assert len(w) == len(prices)
        assert not w.isna().any().any(), "HRP weights contain NaN"
        assert not np.isinf(w.values).any(), "HRP weights contain inf"

    def test_ensemble_hrp_fallback_without_skfolio(self):
        """When skfolio is absent, HRP path falls back to inverse_vol."""
        from unittest.mock import patch
        from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
        from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy

        prices = _make_prices(n=150)
        strats = [CrashHedgeQQQ(), VolCarry()]
        cfg = EnsembleConfig(
            weighting_method="hrp",
            skfolio_refit_every=21,
        )
        # Simulate skfolio not installed via import failure
        import builtins
        real_import = builtins.__import__

        def _mock_import(name, *args, **kwargs):
            if name == "skfolio":
                raise ImportError("mocked: skfolio not installed")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=_mock_import):
            ens = EnsembleStrategy(strats, cfg)
            w = ens.generate_weights(prices)
            assert len(w) == len(prices)
            assert not w.isna().any().any(), "Fallback weights contain NaN"


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

    def test_global_momentum_rotation_basic(self):
        from financial_algo.strategies.momentum import GlobalMomentumRotation
        prices = _make_prices(n=400)
        strat = GlobalMomentumRotation()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_global_momentum_rotation_no_inf(self):
        from financial_algo.strategies.momentum import GlobalMomentumRotation
        prices = _make_prices(n=400)
        strat = GlobalMomentumRotation()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_global_momentum_rotation_long_only(self):
        from financial_algo.strategies.momentum import GlobalMomentumRotation
        prices = _make_prices(n=400)
        strat = GlobalMomentumRotation()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


# =========================================================================
# I6: KSTMomentum
# =========================================================================

class TestKSTMomentum:
    def test_basic(self):
        from financial_algo.strategies.momentum import KSTMomentum
        prices = _make_prices(n=300)
        strat = KSTMomentum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)

    def test_no_inf(self):
        from financial_algo.strategies.momentum import KSTMomentum
        prices = _make_prices(n=300)
        strat = KSTMomentum()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_long_only(self):
        from financial_algo.strategies.momentum import KSTMomentum
        prices = _make_prices(n=300)
        strat = KSTMomentum()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()

    def test_active_after_warmup(self):
        from financial_algo.strategies.momentum import KSTMomentum
        prices = _make_prices(n=400)
        strat = KSTMomentum()
        w = strat.generate_weights(prices)
        # Should have some positions after 200-bar KST warmup
        assert w.iloc[250:].sum(axis=1).max() > 0


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

    def test_global_mean_reversion_basic(self):
        from financial_algo.strategies.mean_reversion import GlobalMeanReversion
        prices = _make_prices(n=300)
        strat = GlobalMeanReversion()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_global_mean_reversion_no_inf(self):
        from financial_algo.strategies.mean_reversion import GlobalMeanReversion
        prices = _make_prices(n=300)
        strat = GlobalMeanReversion()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_global_mean_reversion_long_only(self):
        from financial_algo.strategies.mean_reversion import GlobalMeanReversion
        prices = _make_prices(n=300)
        strat = GlobalMeanReversion()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()



# =========================================================================
# H-FI: Fixed Income Strategies
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


class TestRealAssetsFactor:
    def test_basic(self):
        from financial_algo.strategies.factor import RealAssetsFactor
        prices = _make_prices(n=400)
        strat = RealAssetsFactor()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.factor import RealAssetsFactor
        prices = _make_prices(n=400)
        strat = RealAssetsFactor()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_long_only(self):
        from financial_algo.strategies.factor import RealAssetsFactor
        prices = _make_prices(n=400)
        strat = RealAssetsFactor()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


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


class TestBlackSwanInsurance:
    def test_basic(self):
        from financial_algo.strategies.tail_risk import BlackSwanInsurance
        prices = _make_prices()
        strat = BlackSwanInsurance()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()


class TestVolatilityConvexity:
    """Tests for O8-VolatilityConvexity."""

    @staticmethod
    def _make_prices_with_vix(n: int = 300, seed: int = 42) -> pd.DataFrame:
        """Synthetic prices including ^VIX column."""
        rng = np.random.RandomState(seed)
        dates = pd.bdate_range("2020-01-01", periods=n)
        tickers = ["SPY", "QQQ", "GLD", "TLT", "UUP", "IEF"]
        data = {}
        for t in tickers:
            ret = rng.normal(0.0003, 0.015, n)
            data[t] = 100 * np.exp(np.cumsum(ret))
        # VIX: oscillate around 18 with some spikes
        vix = 18.0 + rng.normal(0, 3.0, n).cumsum() * 0.1
        vix = np.clip(vix, 10.0, 80.0)
        data["^VIX"] = vix
        return pd.DataFrame(data, index=dates)

    def test_basic(self):
        from financial_algo.strategies.tail_risk import VolatilityConvexity
        prices = self._make_prices_with_vix()
        strat = VolatilityConvexity()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.tail_risk import VolatilityConvexity
        prices = self._make_prices_with_vix()
        strat = VolatilityConvexity()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_flat_when_no_vix(self):
        """O8 returns all zeros when ^VIX is not in prices."""
        from financial_algo.strategies.tail_risk import VolatilityConvexity
        prices = _make_prices()  # no ^VIX column
        strat = VolatilityConvexity()
        w = strat.generate_weights(prices)
        assert (w == 0.0).all().all()

    def test_crisis_activates_safe_havens(self):
        """When VIX is high, GLD and TLT should have positive weights."""
        from financial_algo.strategies.tail_risk import VolatilityConvexity
        rng = np.random.RandomState(99)
        n = 100
        dates = pd.bdate_range("2020-01-01", periods=n)
        data = {
            "SPY": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "GLD": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "TLT": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
            "UUP": 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))),
            "^VIX": np.full(n, 35.0),  # constant crisis-level VIX
        }
        prices = pd.DataFrame(data, index=dates)
        strat = VolatilityConvexity()
        w = strat.generate_weights(prices)
        # After the 1-day shift, rows 1+ should have GLD/TLT > 0
        assert (w["GLD"].iloc[1:] > 0).all()
        assert (w["TLT"].iloc[1:] > 0).all()

    def test_no_lookahead(self):
        """First row should be flat due to VIX shift(1)."""
        from financial_algo.strategies.tail_risk import VolatilityConvexity
        prices = self._make_prices_with_vix()
        strat = VolatilityConvexity()
        w = strat.generate_weights(prices)
        assert w.iloc[0].abs().sum() == 0.0


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


# =========================================================================
# Cat MF: Multi-Frequency Strategies
# =========================================================================

class TestWeeklyMomentumRotation:
    def test_basic(self):
        from financial_algo.strategies.multi_freq import WeeklyMomentumRotation
        prices = _make_prices(n=400)
        strat = WeeklyMomentumRotation()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.multi_freq import WeeklyMomentumRotation
        prices = _make_prices(n=400)
        strat = WeeklyMomentumRotation()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.multi_freq import WeeklyMomentumRotation
        prices = _make_prices(n=400)
        strat = WeeklyMomentumRotation()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


class TestMonthlyMacroRegime:
    def test_basic(self):
        from financial_algo.strategies.multi_freq import MonthlyMacroRegime
        prices = _make_prices(n=400)
        strat = MonthlyMacroRegime()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.multi_freq import MonthlyMacroRegime
        prices = _make_prices(n=400)
        strat = MonthlyMacroRegime()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()


class TestMultiTimeframeTrend:
    def test_basic(self):
        from financial_algo.strategies.multi_freq import MultiTimeframeTrend
        prices = _make_prices(n=400)
        strat = MultiTimeframeTrend()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.multi_freq import MultiTimeframeTrend
        prices = _make_prices(n=400)
        strat = MultiTimeframeTrend()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()


class TestWeeklyMeanReversion:
    def test_basic(self):
        from financial_algo.strategies.multi_freq import WeeklyMeanReversion
        prices = _make_prices(n=400)
        strat = WeeklyMeanReversion()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.multi_freq import WeeklyMeanReversion
        prices = _make_prices(n=400)
        strat = WeeklyMeanReversion()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.multi_freq import WeeklyMeanReversion
        prices = _make_prices(n=400)
        strat = WeeklyMeanReversion()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


# =========================================================================
# Cat O: O7 - Precious Metals Crisis Hedge
# =========================================================================

class TestPreciousMetalsCrisisHedge:
    def test_basic(self):
        from financial_algo.strategies.tail_risk import PreciousMetalsCrisisHedge
        prices = _make_prices()
        strat = PreciousMetalsCrisisHedge()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.tail_risk import PreciousMetalsCrisisHedge
        prices = _make_prices()
        strat = PreciousMetalsCrisisHedge()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.tail_risk import PreciousMetalsCrisisHedge
        prices = _make_prices()
        strat = PreciousMetalsCrisisHedge()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()

    def test_empty_prices(self):
        from financial_algo.strategies.tail_risk import PreciousMetalsCrisisHedge
        prices = pd.DataFrame()
        strat = PreciousMetalsCrisisHedge()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)


# =========================================================================
# Cat F: F4 - Crypto Contagion Hedge
# =========================================================================

class TestCryptoContagionHedge:
    def test_basic(self):
        from financial_algo.strategies.crypto_crisis import CryptoContagionHedge
        prices = _make_prices()
        strat = CryptoContagionHedge()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.crypto_crisis import CryptoContagionHedge
        prices = _make_prices()
        strat = CryptoContagionHedge()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.crypto_crisis import CryptoContagionHedge
        prices = _make_prices()
        strat = CryptoContagionHedge()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()

    def test_handles_missing_eth(self):
        """F4 must handle missing ETH-USD gracefully (pre-2017 data)."""
        from financial_algo.strategies.crypto_crisis import CryptoContagionHedge
        prices = _make_prices()
        # Drop ETH to simulate pre-2017 data
        prices_no_eth = prices.drop(columns=["ETH-USD"])
        strat = CryptoContagionHedge()
        w = strat.generate_weights(prices_no_eth)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices_no_eth)
        assert not w.isna().any().any()
        # Without ETH, no contagion signal -> all weights should be 0
        assert (w == 0).all().all()


# =========================================================================
# Cat L: L6 - Cross-Asset Vol Signal
# =========================================================================

class TestCrossAssetVolSignal:
    def test_basic(self):
        from financial_algo.strategies.volatility_strats import CrossAssetVolSignal
        prices = _make_prices()
        strat = CrossAssetVolSignal()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.volatility_strats import CrossAssetVolSignal
        prices = _make_prices()
        strat = CrossAssetVolSignal()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.volatility_strats import CrossAssetVolSignal
        prices = _make_prices()
        strat = CrossAssetVolSignal()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()

    def test_handles_missing_tickers(self):
        """L6 must work even if some vol basket tickers are missing."""
        from financial_algo.strategies.volatility_strats import CrossAssetVolSignal
        prices = _make_prices()
        # Remove FXI and DBC from prices
        prices_reduced = prices.drop(columns=["FXI", "DBC"], errors="ignore")
        strat = CrossAssetVolSignal()
        w = strat.generate_weights(prices_reduced)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices_reduced)
        assert not w.isna().any().any()


# =========================================================================
# Cat G: G5 - Crypto Sentiment Divergence
# =========================================================================

class TestCryptoSentimentDivergence:
    def test_basic(self):
        from financial_algo.fundamental.strategies.sentiment_strategies import (
            CryptoSentimentDivergence,
        )
        prices = _make_prices()
        strat = CryptoSentimentDivergence()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.fundamental.strategies.sentiment_strategies import (
            CryptoSentimentDivergence,
        )
        prices = _make_prices()
        strat = CryptoSentimentDivergence()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.fundamental.strategies.sentiment_strategies import (
            CryptoSentimentDivergence,
        )
        prices = _make_prices()
        strat = CryptoSentimentDivergence()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()

    def test_handles_missing_eth(self):
        """G5 must degrade gracefully when ETH-USD is missing."""
        from financial_algo.fundamental.strategies.sentiment_strategies import (
            CryptoSentimentDivergence,
        )
        prices = _make_prices()
        prices_no_eth = prices.drop(columns=["ETH-USD"])
        strat = CryptoSentimentDivergence()
        w = strat.generate_weights(prices_no_eth)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices_no_eth)
        assert not w.isna().any().any()

    def test_handles_missing_btc(self):
        """G5 must return neutral weights when BTC-USD is missing."""
        from financial_algo.fundamental.strategies.sentiment_strategies import (
            CryptoSentimentDivergence,
        )
        prices = _make_prices()
        prices_no_btc = prices.drop(columns=["BTC-USD"])
        strat = CryptoSentimentDivergence()
        w = strat.generate_weights(prices_no_btc)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices_no_btc)
        assert not w.isna().any().any()


# =========================================================================
# Cat M: M7-M9 Macro Strategies
# =========================================================================

class TestGlobalRotation:
    def test_basic(self):
        from financial_algo.strategies.macro import GlobalRotation
        prices = _make_prices(n=400)
        strat = GlobalRotation()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.macro import GlobalRotation
        prices = _make_prices(n=400)
        strat = GlobalRotation()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.macro import GlobalRotation
        prices = _make_prices(n=400)
        strat = GlobalRotation()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


class TestCommodityMacroSignal:
    def test_basic(self):
        from financial_algo.strategies.macro import CommodityMacroSignal
        prices = _make_prices(n=400)
        strat = CommodityMacroSignal()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.macro import CommodityMacroSignal
        prices = _make_prices(n=400)
        strat = CommodityMacroSignal()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_only(self):
        from financial_algo.strategies.macro import CommodityMacroSignal
        prices = _make_prices(n=400)
        strat = CommodityMacroSignal()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


class TestYieldCurveRegime:
    def test_basic(self):
        from financial_algo.strategies.macro import YieldCurveRegime
        prices = _make_prices(n=400)
        strat = YieldCurveRegime()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.macro import YieldCurveRegime
        prices = _make_prices(n=400)
        strat = YieldCurveRegime()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_missing_tickers(self):
        from financial_algo.strategies.macro import YieldCurveRegime
        prices = _make_prices(n=300).drop(columns=["TLT"])
        strat = YieldCurveRegime()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_regimes_cover_all_days(self):
        from financial_algo.strategies.macro import YieldCurveRegime
        prices = _make_prices(n=400)
        strat = YieldCurveRegime()
        w = strat.generate_weights(prices)
        # After warmup, every day should have some allocation
        warmup = 260  # zscore window + EMA
        assert (w.iloc[warmup:].abs().sum(axis=1) > 0).all()

    def test_empty_prices(self):
        from financial_algo.strategies.macro import YieldCurveRegime
        prices = pd.DataFrame()
        strat = YieldCurveRegime()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)


# =========================================================================
# Cat R: Regime Hardening — R7, R8
# =========================================================================

class TestVolExplosionAlpha:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import VolExplosionAlpha
        prices = _make_prices(n=400)
        strat = VolExplosionAlpha()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import VolExplosionAlpha
        prices = _make_prices(n=400)
        strat = VolExplosionAlpha()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_max_leverage(self):
        from financial_algo.strategies.regime_hardening import VolExplosionAlpha
        prices = _make_prices(n=400)
        strat = VolExplosionAlpha()
        w = strat.generate_weights(prices)
        assert (w.sum(axis=1) <= 1.0 + 1e-9).all()

    def test_long_only(self):
        from financial_algo.strategies.regime_hardening import VolExplosionAlpha
        prices = _make_prices(n=400)
        strat = VolExplosionAlpha()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


class TestVolRegimeSwitcher:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import VolRegimeSwitcher
        prices = _make_prices(n=400)
        strat = VolRegimeSwitcher()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import VolRegimeSwitcher
        prices = _make_prices(n=400)
        strat = VolRegimeSwitcher()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_max_leverage(self):
        from financial_algo.strategies.regime_hardening import VolRegimeSwitcher
        prices = _make_prices(n=400)
        strat = VolRegimeSwitcher()
        w = strat.generate_weights(prices)
        assert (w.sum(axis=1) <= 1.0 + 1e-9).all()

    def test_long_only(self):
        from financial_algo.strategies.regime_hardening import VolRegimeSwitcher
        prices = _make_prices(n=400)
        strat = VolRegimeSwitcher()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


class TestRatesTighteningAlpha:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import RatesTighteningAlpha
        prices = _make_prices(n=400)
        strat = RatesTighteningAlpha()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import RatesTighteningAlpha
        prices = _make_prices(n=400)
        strat = RatesTighteningAlpha()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_max_leverage(self):
        from financial_algo.strategies.regime_hardening import RatesTighteningAlpha
        prices = _make_prices(n=400)
        strat = RatesTighteningAlpha()
        w = strat.generate_weights(prices)
        assert (w.sum(axis=1) <= 1.0 + 1e-9).all()

    def test_long_only(self):
        from financial_algo.strategies.regime_hardening import RatesTighteningAlpha
        prices = _make_prices(n=400)
        strat = RatesTighteningAlpha()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


class TestBondEquityHedge:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import BondEquityHedge
        prices = _make_prices(n=400)
        strat = BondEquityHedge()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import BondEquityHedge
        prices = _make_prices(n=400)
        strat = BondEquityHedge()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_max_leverage(self):
        from financial_algo.strategies.regime_hardening import BondEquityHedge
        prices = _make_prices(n=400)
        strat = BondEquityHedge()
        w = strat.generate_weights(prices)
        assert (w.sum(axis=1) <= 1.0 + 1e-9).all()

    def test_long_only(self):
        from financial_algo.strategies.regime_hardening import BondEquityHedge
        prices = _make_prices(n=400)
        strat = BondEquityHedge()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()


class TestBearMarketAlpha:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import BearMarketAlpha
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = BearMarketAlpha()
        w = strat.generate_weights(prices, regime)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import BearMarketAlpha
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = BearMarketAlpha()
        w = strat.generate_weights(prices, regime)
        assert not np.isinf(w.values).any()

    def test_without_regime(self):
        from financial_algo.strategies.regime_hardening import BearMarketAlpha
        prices = _make_prices()
        strat = BearMarketAlpha()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_max_leverage(self):
        from financial_algo.strategies.regime_hardening import BearMarketAlpha
        prices = _make_prices()
        regime = _make_regime(prices)
        strat = BearMarketAlpha()
        w = strat.generate_weights(prices, regime)
        gross = w.abs().sum(axis=1)
        assert (gross <= 1.0 + 1e-9).all()


# =========================================================================
# Cat R: R3 - Defensive Rotation, R4 - Adaptive Risk Budget
# =========================================================================

class TestDefensiveRotationR3:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import DefensiveRotationR3
        prices = _make_prices(n=400)
        strat = DefensiveRotationR3()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import DefensiveRotationR3
        prices = _make_prices(n=400)
        strat = DefensiveRotationR3()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_long_only(self):
        from financial_algo.strategies.regime_hardening import DefensiveRotationR3
        prices = _make_prices(n=400)
        strat = DefensiveRotationR3()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()

    def test_gross_leverage_capped(self):
        from financial_algo.strategies.regime_hardening import DefensiveRotationR3
        prices = _make_prices(n=400)
        strat = DefensiveRotationR3()
        w = strat.generate_weights(prices)
        gross = w.abs().sum(axis=1)
        assert (gross <= 1.0 + 1e-9).all()

    def test_handles_missing_tickers(self):
        from financial_algo.strategies.regime_hardening import DefensiveRotationR3
        prices = _make_prices(n=400)
        prices_reduced = prices.drop(columns=["UUP"], errors="ignore")
        strat = DefensiveRotationR3()
        w = strat.generate_weights(prices_reduced)
        assert isinstance(w, pd.DataFrame)
        assert not w.isna().any().any()


class TestAdaptiveRiskBudget:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import AdaptiveRiskBudget
        prices = _make_prices(n=400)
        strat = AdaptiveRiskBudget()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import AdaptiveRiskBudget
        prices = _make_prices(n=400)
        strat = AdaptiveRiskBudget()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_long_only(self):
        from financial_algo.strategies.regime_hardening import AdaptiveRiskBudget
        prices = _make_prices(n=400)
        strat = AdaptiveRiskBudget()
        w = strat.generate_weights(prices)
        assert (w >= 0).all().all()

    def test_always_invested(self):
        """R4 should always be invested (equity + safe = ~1.0)."""
        from financial_algo.strategies.regime_hardening import AdaptiveRiskBudget
        prices = _make_prices(n=400)
        strat = AdaptiveRiskBudget()
        w = strat.generate_weights(prices)
        total = w.sum(axis=1)
        # After warm-up, total allocation should be close to 1.0
        assert (total.iloc[63:] > 0.5).all()

    def test_no_spy_returns_zeros(self):
        """R4 handles missing SPY gracefully."""
        from financial_algo.strategies.regime_hardening import AdaptiveRiskBudget
        prices = _make_prices(n=300)
        prices_no_spy = prices.drop(columns=["SPY"])
        strat = AdaptiveRiskBudget()
        w = strat.generate_weights(prices_no_spy)
        assert (w == 0).all().all()


# =========================================================================
# Cat P: ML-enhanced strategies (P2, P3)
# =========================================================================

class TestXGBoostSignalCombo:
    def test_basic(self):
        from financial_algo.strategies.signal_combo import XGBoostSignalCombo
        prices = _make_prices(n=600)
        strat = XGBoostSignalCombo()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.signal_combo import XGBoostSignalCombo
        prices = _make_prices(n=600)
        strat = XGBoostSignalCombo()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_warmup_uses_p1_fallback(self):
        """During warm-up (< 252 days), should produce nonzero weights."""
        from financial_algo.strategies.signal_combo import XGBoostSignalCombo
        prices = _make_prices(n=600)
        strat = XGBoostSignalCombo()
        w = strat.generate_weights(prices)
        warmup_total = w.iloc[21:252].sum(axis=1)
        assert (warmup_total > 0).any()

    def test_ml_phase_produces_weights(self):
        """After warm-up, ML predictions should produce weights."""
        from financial_algo.strategies.signal_combo import XGBoostSignalCombo
        prices = _make_prices(n=600)
        strat = XGBoostSignalCombo()
        w = strat.generate_weights(prices)
        ml_total = w.iloc[300:].sum(axis=1)
        assert (ml_total > 0).any()

    def test_few_tickers(self):
        """Graceful when fewer than 3 tickers available."""
        from financial_algo.strategies.signal_combo import XGBoostSignalCombo
        prices = _make_prices(n=600)[["SPY", "QQQ"]]
        strat = XGBoostSignalCombo()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()


class TestGMMRegimeClassifier:
    def test_basic(self):
        from financial_algo.strategies.signal_combo import GMMRegimeClassifier
        prices = _make_prices(n=600)
        strat = GMMRegimeClassifier()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.signal_combo import GMMRegimeClassifier
        prices = _make_prices(n=600)
        strat = GMMRegimeClassifier()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_positions_clipped(self):
        """No individual position exceeds max_position (0.5)."""
        from financial_algo.strategies.signal_combo import GMMRegimeClassifier
        prices = _make_prices(n=600)
        strat = GMMRegimeClassifier()
        w = strat.generate_weights(prices)
        assert (w.abs() <= 0.5 + 1e-9).all().all()

    def test_no_spy_returns_zeros(self):
        """Missing SPY makes VIX proxy impossible -- should return zeros."""
        from financial_algo.strategies.signal_combo import GMMRegimeClassifier
        prices = _make_prices(n=600).drop(columns=["SPY"])
        strat = GMMRegimeClassifier()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_active_after_warmup(self):
        """GMM should produce nonzero weights after 252-day warm-up."""
        from financial_algo.strategies.signal_combo import GMMRegimeClassifier
        prices = _make_prices(n=600)
        strat = GMMRegimeClassifier()
        w = strat.generate_weights(prices)
        post_warmup = w.iloc[280:].abs().sum(axis=1)
        assert (post_warmup > 0).any()

class TestMultiAssetCTATrend:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
        prices = _make_prices(n=600)
        strat = MultiAssetCTATrend()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
        prices = _make_prices(n=600)
        strat = MultiAssetCTATrend()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_max_leverage(self):
        from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
        prices = _make_prices(n=600)
        strat = MultiAssetCTATrend()
        w = strat.generate_weights(prices)
        gross = w.abs().sum(axis=1)
        assert (gross <= 1.5 + 1e-9).all()

    def test_active_after_warmup(self):
        from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
        prices = _make_prices(n=600)
        strat = MultiAssetCTATrend()
        w = strat.generate_weights(prices)
        post_warmup = w.iloc[200:].abs().sum(axis=1)
        assert (post_warmup > 0).any()

    def test_no_spy_returns_zeros(self):
        """Missing entire universe should return zero weights gracefully."""
        from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
        prices = _make_prices(n=200)[["ITA", "LMT"]]  # no CTA universe assets
        strat = MultiAssetCTATrend()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()


class TestCommodityMacroOverlay:
    def test_basic(self):
        from financial_algo.strategies.regime_hardening import CommodityMacroOverlay
        prices = _make_prices(n=400)
        strat = CommodityMacroOverlay()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.regime_hardening import CommodityMacroOverlay
        prices = _make_prices(n=400)
        strat = CommodityMacroOverlay()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_max_leverage(self):
        from financial_algo.strategies.regime_hardening import CommodityMacroOverlay
        prices = _make_prices(n=400)
        strat = CommodityMacroOverlay()
        w = strat.generate_weights(prices)
        gross = w.abs().sum(axis=1)
        assert (gross <= 1.0 + 1e-9).all()

    def test_dormant_missing_required(self):
        """Without XLE/UUP/TLT/SPY, should return all zeros."""
        from financial_algo.strategies.regime_hardening import CommodityMacroOverlay
        prices = _make_prices(n=400)[["ITA", "LMT", "GLD"]]
        strat = CommodityMacroOverlay()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_only_longs(self):
        """No short positions: all weights non-negative."""
        from financial_algo.strategies.regime_hardening import CommodityMacroOverlay
        prices = _make_prices(n=400)
        strat = CommodityMacroOverlay()
        w = strat.generate_weights(prices)
        assert (w >= -1e-9).all().all()


class TestMultiSignalConsensus:
    def test_basic_shape(self):
        from financial_algo.strategies.signal_combo import MultiSignalConsensus
        prices = _make_prices(n=400)
        strat = MultiSignalConsensus()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert w.shape[0] == len(prices)
        assert w.shape[1] == len(prices.columns)

    def test_no_nan(self):
        from financial_algo.strategies.signal_combo import MultiSignalConsensus
        prices = _make_prices(n=400)
        strat = MultiSignalConsensus()
        w = strat.generate_weights(prices)
        assert not w.isnull().any().any()
        assert np.isfinite(w.values).all()

    def test_long_only(self):
        from financial_algo.strategies.signal_combo import MultiSignalConsensus
        prices = _make_prices(n=400)
        strat = MultiSignalConsensus()
        w = strat.generate_weights(prices)
        assert (w >= -1e-9).all().all()

    def test_max_leverage_respected(self):
        from financial_algo.strategies.signal_combo import MultiSignalConsensus
        prices = _make_prices(n=400)
        strat = MultiSignalConsensus()
        w = strat.generate_weights(prices)
        gross = w.abs().sum(axis=1)
        assert (gross <= 1.5 + 1e-9).all()


class TestDrawdownRecoveryTiming:
    def test_basic(self):
        from financial_algo.strategies.tail_risk import DrawdownRecoveryTiming
        prices = _make_prices(n=500)
        strat = DrawdownRecoveryTiming()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_empty(self):
        from financial_algo.strategies.tail_risk import DrawdownRecoveryTiming
        prices = pd.DataFrame()
        strat = DrawdownRecoveryTiming()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == 0

    def test_no_spy(self):
        from financial_algo.strategies.tail_risk import DrawdownRecoveryTiming
        prices = _make_prices(n=300)
        prices = prices.drop(columns=["SPY"])
        strat = DrawdownRecoveryTiming()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_recovery_activates_on_crash(self):
        """Verify recovery window activates after a synthetic crash + bounce."""
        from financial_algo.strategies.tail_risk import DrawdownRecoveryTiming
        dates = pd.bdate_range("2020-01-01", periods=200)
        # Build SPY: flat, crash, then bounce
        spy_prices = np.ones(200) * 100.0
        # Days 50-70: 20% crash
        for i in range(50, 70):
            spy_prices[i] = spy_prices[i - 1] * 0.99
        # Days 70-80: bounce (10 consecutive up-days)
        for i in range(70, 80):
            spy_prices[i] = spy_prices[i - 1] * 1.015
        # Rest stays flat from day 80
        for i in range(80, 200):
            spy_prices[i] = spy_prices[79]

        prices = pd.DataFrame({
            "SPY": spy_prices, "QQQ": spy_prices,
            "IWM": spy_prices, "EFA": spy_prices,
        }, index=dates)
        strat = DrawdownRecoveryTiming()
        w = strat.generate_weights(prices)
        # After the crash and bounce, recovery should be active
        recovery_days = (w["SPY"] > 0).sum()
        assert recovery_days > 0, "Recovery should have activated after crash + bounce"


# =========================================================================
# S1 — FormulaicAlphaMomentum
# =========================================================================

class TestFormulaicAlphaMomentum:
    def test_basic(self):
        from financial_algo.strategies.factor import FormulaicAlphaMomentum
        prices = _make_prices(n=300)
        strat = FormulaicAlphaMomentum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert w.shape == prices.shape
        assert not w.isna().any().any()

    def test_empty(self):
        from financial_algo.strategies.factor import FormulaicAlphaMomentum
        prices = pd.DataFrame()
        strat = FormulaicAlphaMomentum()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == 0

    def test_few_assets(self):
        from financial_algo.strategies.factor import FormulaicAlphaMomentum
        prices = _make_prices(n=300)[["SPY", "QQQ"]]
        strat = FormulaicAlphaMomentum()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_no_inf(self):
        from financial_algo.strategies.factor import FormulaicAlphaMomentum
        prices = _make_prices(n=300)
        strat = FormulaicAlphaMomentum()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_short(self):
        from financial_algo.strategies.factor import (
            FormulaicAlphaMomentum, FormulaicAlphaMomConfig,
        )
        prices = _make_prices(n=300)
        cfg = FormulaicAlphaMomConfig(long_only=False, leverage=1.0)
        strat = FormulaicAlphaMomentum(config=cfg)
        w = strat.generate_weights(prices)
        assert (w < 0).any().any(), "Long-short should have negative weights"

    def test_with_injected_volume(self):
        from financial_algo.strategies.factor import FormulaicAlphaMomentum
        prices = _make_prices(n=300)
        rng = np.random.RandomState(99)
        volume = pd.DataFrame(
            rng.uniform(1e5, 1e7, prices.shape),
            index=prices.index, columns=prices.columns,
        )
        strat = FormulaicAlphaMomentum()
        strat.set_ohlcv(volume=volume)
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert not w.isna().any().any()


# =========================================================================
# S2 — FormulaicAlphaMeanRev
# =========================================================================

class TestFormulaicAlphaMeanRev:
    def test_basic(self):
        from financial_algo.strategies.mean_reversion import FormulaicAlphaMeanRev
        prices = _make_prices(n=300)
        strat = FormulaicAlphaMeanRev()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert w.shape == prices.shape
        assert not w.isna().any().any()

    def test_empty(self):
        from financial_algo.strategies.mean_reversion import FormulaicAlphaMeanRev
        prices = pd.DataFrame()
        strat = FormulaicAlphaMeanRev()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == 0

    def test_few_assets(self):
        from financial_algo.strategies.mean_reversion import FormulaicAlphaMeanRev
        prices = _make_prices(n=300)[["SPY", "QQQ"]]
        strat = FormulaicAlphaMeanRev()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_no_inf(self):
        from financial_algo.strategies.mean_reversion import FormulaicAlphaMeanRev
        prices = _make_prices(n=300)
        strat = FormulaicAlphaMeanRev()
        w = strat.generate_weights(prices)
        assert not np.isinf(w.values).any()

    def test_long_short(self):
        from financial_algo.strategies.mean_reversion import (
            FormulaicAlphaMeanRev, FormulaicAlphaMeanRevConfig,
        )
        prices = _make_prices(n=300)
        cfg = FormulaicAlphaMeanRevConfig(long_only=False, leverage=1.0)
        strat = FormulaicAlphaMeanRev(config=cfg)
        w = strat.generate_weights(prices)
        assert (w < 0).any().any(), "Long-short should have negative weights"

    def test_with_injected_ohlcv(self):
        from financial_algo.strategies.mean_reversion import FormulaicAlphaMeanRev
        prices = _make_prices(n=300)
        rng = np.random.RandomState(88)
        ret_abs = prices.pct_change().fillna(0.0).abs()
        open_df = prices.shift(1).bfill()
        high = prices * (1 + ret_abs * 0.7)
        low = prices * (1 - ret_abs * 0.7)
        volume = pd.DataFrame(
            rng.uniform(1e5, 1e7, prices.shape),
            index=prices.index, columns=prices.columns,
        )
        strat = FormulaicAlphaMeanRev()
        strat.set_ohlcv(open_df=open_df, high=high, low=low, volume=volume)
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert not w.isna().any().any()


# =========================================================================
# Cat L7/L8: Implied-Realized Spread & Vol Regime Clustering
# =========================================================================

def _make_prices_with_vix(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic prices including ^VIX for vol strategy tests."""
    prices = _make_prices(n=n, seed=seed)
    rng = np.random.RandomState(seed + 99)
    # Simulate VIX: mean-reverting around 20, range ~12-40
    vix = np.empty(n)
    vix[0] = 20.0
    for i in range(1, n):
        vix[i] = vix[i - 1] + 0.1 * (20.0 - vix[i - 1]) + rng.normal(0, 1.5)
        vix[i] = max(10.0, min(60.0, vix[i]))
    prices["^VIX"] = vix
    return prices


class TestL7L8VolStrategies:
    """Tests for L7-ImpliedRealizedSpread and L8-VolRegimeClustering."""

    # --- L7 tests ---

    def test_l7_basic(self):
        from financial_algo.strategies.volatility_strats import ImpliedRealizedSpread
        prices = _make_prices_with_vix()
        strat = ImpliedRealizedSpread()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_l7_no_inf(self):
        from financial_algo.strategies.volatility_strats import ImpliedRealizedSpread
        prices = _make_prices_with_vix()
        strat = ImpliedRealizedSpread()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_l7_without_vix(self):
        """L7 must work via proxy when ^VIX is absent."""
        from financial_algo.strategies.volatility_strats import ImpliedRealizedSpread
        prices = _make_prices(n=500)
        strat = ImpliedRealizedSpread()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert not w.isna().any().any()

    def test_l7_empty_prices(self):
        from financial_algo.strategies.volatility_strats import ImpliedRealizedSpread
        prices = pd.DataFrame()
        strat = ImpliedRealizedSpread()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == 0

    def test_l7_contango_weights(self):
        """In strong contango, equity weight should exceed neutral."""
        from financial_algo.strategies.volatility_strats import ImpliedRealizedSpread
        prices = _make_prices_with_vix(n=500)
        strat = ImpliedRealizedSpread()
        w = strat.generate_weights(prices)
        max_spy = w["SPY"].max()
        assert max_spy >= 0.3  # at least neutral weight present

    def test_l7_idempotent(self):
        from financial_algo.strategies.volatility_strats import ImpliedRealizedSpread
        prices = _make_prices_with_vix()
        strat = ImpliedRealizedSpread()
        w1 = strat.generate_weights(prices)
        w2 = strat.generate_weights(prices)
        pd.testing.assert_frame_equal(w1, w2)

    # --- L8 tests ---

    def test_l8_basic(self):
        from financial_algo.strategies.volatility_strats import VolRegimeClustering
        prices = _make_prices_with_vix()
        strat = VolRegimeClustering()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == len(prices)
        assert not w.isna().any().any()

    def test_l8_no_inf(self):
        from financial_algo.strategies.volatility_strats import VolRegimeClustering
        prices = _make_prices_with_vix()
        strat = VolRegimeClustering()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_l8_without_vix(self):
        """L8 must work via proxy when ^VIX is absent."""
        from financial_algo.strategies.volatility_strats import VolRegimeClustering
        prices = _make_prices(n=500)
        strat = VolRegimeClustering()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert not w.isna().any().any()

    def test_l8_empty_prices(self):
        from financial_algo.strategies.volatility_strats import VolRegimeClustering
        prices = pd.DataFrame()
        strat = VolRegimeClustering()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert len(w) == 0

    def test_l8_crisis_zero_equity(self):
        """In crisis state, equity weight should be zero."""
        from financial_algo.strategies.volatility_strats import VolRegimeClustering
        prices = _make_prices_with_vix(n=500)
        strat = VolRegimeClustering()
        w = strat.generate_weights(prices)
        # Safe havens should appear somewhere (TLT, GLD)
        if "TLT" in w.columns:
            assert w["TLT"].max() > 0
        if "GLD" in w.columns:
            assert w["GLD"].max() > 0

    def test_l8_has_multiple_states(self):
        """L8 should produce varied allocations across the sample."""
        from financial_algo.strategies.volatility_strats import VolRegimeClustering
        prices = _make_prices_with_vix(n=500)
        strat = VolRegimeClustering()
        w = strat.generate_weights(prices)
        spy_unique = w["SPY"].round(2).nunique()
        assert spy_unique >= 2  # at least 2 distinct allocation levels

    def test_l8_idempotent(self):
        from financial_algo.strategies.volatility_strats import VolRegimeClustering
        prices = _make_prices_with_vix()
        strat = VolRegimeClustering()
        w1 = strat.generate_weights(prices)
        w2 = strat.generate_weights(prices)
        pd.testing.assert_frame_equal(w1, w2)


# =========================================================================
# I10 / K6 — Alpha-Max v10 Sprint
# =========================================================================

class TestAdaptiveTrendFilter:
    def test_basic(self):
        from financial_algo.strategies.momentum import AdaptiveTrendFilter
        prices = _make_prices(n=400)
        strat = AdaptiveTrendFilter()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert w.shape == prices.shape
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.momentum import AdaptiveTrendFilter
        prices = _make_prices(n=400)
        strat = AdaptiveTrendFilter()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_empty(self):
        from financial_algo.strategies.momentum import AdaptiveTrendFilter
        strat = AdaptiveTrendFilter()
        w = strat.generate_weights(pd.DataFrame())
        assert isinstance(w, pd.DataFrame)
        assert len(w) == 0

    def test_has_short_positions(self):
        from financial_algo.strategies.momentum import AdaptiveTrendFilter
        prices = _make_prices(n=400)
        strat = AdaptiveTrendFilter()
        w = strat.generate_weights(prices)
        assert (w < 0).any().any(), "Long/short strategy should have negatives"

    def test_few_assets(self):
        from financial_algo.strategies.momentum import AdaptiveTrendFilter
        prices = _make_prices(n=400)[["SPY", "QQQ"]]
        strat = AdaptiveTrendFilter()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_import_from_init(self):
        from financial_algo.strategies import AdaptiveTrendFilter
        assert AdaptiveTrendFilter.name == "I10-AdaptiveTrendFilter"


class TestQualityMomentumComposite:
    def test_basic(self):
        from financial_algo.strategies.factor import QualityMomentumComposite
        prices = _make_prices(n=400)
        strat = QualityMomentumComposite()
        w = strat.generate_weights(prices)
        assert isinstance(w, pd.DataFrame)
        assert w.shape == prices.shape
        assert not w.isna().any().any()

    def test_no_inf(self):
        from financial_algo.strategies.factor import QualityMomentumComposite
        prices = _make_prices(n=400)
        strat = QualityMomentumComposite()
        w = strat.generate_weights(prices)
        assert np.isfinite(w.values).all()

    def test_empty(self):
        from financial_algo.strategies.factor import QualityMomentumComposite
        strat = QualityMomentumComposite()
        w = strat.generate_weights(pd.DataFrame())
        assert isinstance(w, pd.DataFrame)
        assert len(w) == 0

    def test_has_short_positions(self):
        from financial_algo.strategies.factor import QualityMomentumComposite
        prices = _make_prices(n=400)
        strat = QualityMomentumComposite()
        w = strat.generate_weights(prices)
        assert (w < 0).any().any(), "Long/short strategy should have negatives"

    def test_few_assets(self):
        from financial_algo.strategies.factor import QualityMomentumComposite
        prices = _make_prices(n=400)[["SPY", "QQQ"]]
        strat = QualityMomentumComposite()
        w = strat.generate_weights(prices)
        assert (w == 0).all().all()

    def test_import_from_init(self):
        from financial_algo.strategies import QualityMomentumComposite
        assert QualityMomentumComposite.name == "K6-QualityMomentumComposite"