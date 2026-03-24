"""Profile the crisis backtest to identify time sinks."""
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.backtest import BacktestConfig, backtest

# Import all strategy categories
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, FourStateTactical, VolCarry, AdaptiveStopTrend
from financial_algo.strategies.oil_crisis import EnergyPairs, OilMeanReversion, OilMomentumSurge, OilShockHedge
from financial_algo.strategies.war_crisis import ArmsRaceMomentum, DefenseRotation, PostWarRecovery, SafeHavenFlight
from financial_algo.strategies.pairs import MultiPairPortfolio
from financial_algo.strategies.crypto_crisis import CryptoFlightToQuality, CryptoGoldDivergence, CryptoRecoverySurge, CryptoContagionHedge, CryptoRecoverySurgeATR
from financial_algo.strategies.crisis_spike import CommodityShockRider, DefenseSpikeBreakout, GoldFearRally, MultiAssetCrisisLong
from financial_algo.strategies.momentum import TimeSeriesMomentum, CrossSectionalMomentum, DualMomentum, MomentumVolScaled, GlobalMomentumRotation, AdaptiveTrendFilter, DriftRegimeMomentum, TimeSeriesMomentumDrift
from financial_algo.strategies.mean_reversion import SectorMeanReversion, RSIMeanReversion, OvernightGapFade, CointegrationPairs, GlobalMeanReversion, DriftReversalAlpha
from financial_algo.strategies.fixed_income import YieldCurveTrade, CreditSpreadMeanRev
from financial_algo.strategies.volatility_strats import VolRiskPremium, VolSpreadHarvest, VolOfVolRegime, VolTermStructure, VolSpikeRecovery, CrossAssetVolSignal, ImpliedRealizedSpread, VolRegimeClustering, VIXAdaptiveCarry, DynamicVolRegimeSwitch, VolRiskPremiumAdaptive
from financial_algo.strategies.macro import DollarCarry, GoldDollarInverse, EMRiskPremium, CommodityMomentum, RatesRegimeTrade, GlobalRotation, CommodityMacroSignal, YieldCurveRegime, MacroSignalScoreboard, AdaptiveMacroBlend, DollarCarryScoreboard
from financial_algo.strategies.seasonal import SeasonalStrategy, TurnOfMonth, PreHolidayDrift
from financial_algo.strategies.factor import LowVolFactor, MultiFactorComposite, SizeFactor, ValueFactor, RealAssetsFactor, QualityMomentumComposite
from financial_algo.strategies.tail_risk import TailRiskParity, BlackSwanInsurance, TailHedgeOverlay, PreciousMetalsCrisisHedge, VolatilityConvexity, ATRCrisisAlpha
from financial_algo.strategies.signal_combo import FeatureComboSignal, MultiSignalConsensus, XGBoostSignalCombo, GMMRegimeClassifier
from financial_algo.strategies.ml_strategies import AdaptiveThreshold, CrossSectionalRanker
from financial_algo.strategies.dl_strategies import TemporalCNNAlpha, LSTMRegimeDetector, AttentionCrossSectionalRanker
from financial_algo.strategies.quality_trend import QualityTrend, MultiAssetTrend, MomentumCrashFilter
from financial_algo.strategies.multi_freq import WeeklyMomentumRotation, MonthlyMacroRegime, MultiTimeframeTrend, WeeklyMeanReversion
from financial_algo.strategies.regime_hardening import BearMarketAlpha, DefensiveRotationR3, AdaptiveRiskBudget, RatesTighteningAlpha, BondEquityHedge, VolExplosionAlpha, VolRegimeSwitcher, MultiAssetCTATrend, CommodityMacroOverlay
from financial_algo.fundamental import build_synthetic_sentiment
from financial_algo.fundamental.strategies import FearGreedContrarian, SentimentCrisisAlpha, SentimentDivergence, SentimentEnhancedRegime, CryptoSentimentDivergence
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "SLV", "TLT", "IEF", "SHY", "UUP",
    "XLE", "USO", "XOP",
    "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "XBI", "XLC", "XLRE",
    "DBC", "DBA",
    "HYG", "LQD", "TIP", "AGG", "EMB",
    "FXI", "VGK", "EWJ", "INDA",
    "VNQ",
    "BTC-USD", "ETH-USD",
]))

BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
    initial_capital=1_000_000.0, vol_target=0.20, max_drawdown_trigger=-0.25,
    drawdown_recovery_rate=0.10,
)

def time_fn(label, fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    print(f"  {elapsed:8.2f}s  {label}")
    return result, elapsed

def main():
    print("=" * 70)
    print("BACKTEST PERFORMANCE PROFILER")
    print("=" * 70)

    # Phase 1: Data loading
    t0_total = time.perf_counter()
    print("\n[Phase 1] Data loading...")
    prices, t_load = time_fn("load_prices", load_prices, TICKERS, start="2009-01-01", end="2025-12-31")
    print(f"  Shape: {prices.shape}")

    try:
        vix_df, _ = time_fn("load VIX", load_prices, ["^VIX"], start="2009-01-01", end="2025-12-31")
        vix = vix_df["^VIX"]
    except Exception:
        vix = None

    # Phase 2: Regime detection
    print("\n[Phase 2] Regime detection...")
    regime, t_regime = time_fn("detect_regime", detect_regime, prices, vix=vix)

    # Phase 3: Individual strategy timing (Full Period only)
    print("\n[Phase 3] Strategy timing (Full Period 2010-2025)...")
    mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
    p = prices.loc[mask].copy()
    r = regime.loc[mask].copy()
    print(f"  Window: {p.index[0].date()} to {p.index[-1].date()}, {len(p)} days")

    # Build full strategy list with timing
    strategies = [
        ("B1-OilMomSurge", OilMomentumSurge(), True),
        ("B2-OilShockHedge", OilShockHedge(), True),
        ("B3-OilMeanRev", OilMeanReversion(), True),
        ("B4-EnergyPairs", EnergyPairs(), True),
        ("C1-DefenseRot", DefenseRotation(), True),
        ("C2-SafeHaven", SafeHavenFlight(), True),
        ("C3-PostWarRecov", PostWarRecovery(), True),
        ("C4-ArmsRace", ArmsRaceMomentum(), False),
        ("D1-FourState", FourStateTactical(), True),
        ("D2-CrashHedgeQQQ", CrashHedgeQQQ(), False),
        ("D3-VolCarry", VolCarry(), False),
        ("D4-AdaptiveStop", AdaptiveStopTrend(), False),
        ("E1-MultiPair", MultiPairPortfolio(), False),
        ("F1-CryptoFlight", CryptoFlightToQuality(), True),
        ("F2-CryptoRecov", CryptoRecoverySurge(), True),
        ("F3-CryptoGold", CryptoGoldDivergence(), True),
        ("F4-CryptoContagion", CryptoContagionHedge(), False),
        ("F2b-CryptoRecovATR", CryptoRecoverySurgeATR(), False),
        ("H1-CommShockRider", CommodityShockRider(), True),
        ("H2-GoldFearRally", GoldFearRally(), False),
        ("H3-DefSpikeBrk", DefenseSpikeBreakout(), True),
        ("H4-MultiAssetCrisis", MultiAssetCrisisLong(), True),
        ("H1-YieldCurve", YieldCurveTrade(), False),
        ("H2-CreditSpread", CreditSpreadMeanRev(), False),
        ("I1-TSMom", TimeSeriesMomentum(), False),
        ("I2-CrossSecMom", CrossSectionalMomentum(), False),
        ("I3-DualMom", DualMomentum(), False),
        ("I4-MomVolScaled", MomentumVolScaled(), False),
        ("I5-GlobMomRot", GlobalMomentumRotation(), False),
        ("I10-AdaptTrend", AdaptiveTrendFilter(), False),
        ("I7-DriftRegMom", DriftRegimeMomentum(), False),
        ("I8-TSMomDrift", TimeSeriesMomentumDrift(), False),
        ("J1-SectorMR", SectorMeanReversion(), False),
        ("J2-OvernightGap", OvernightGapFade(), False),
        ("J3-RSIMR", RSIMeanReversion(), False),
        ("J4-CointPairs", CointegrationPairs(), False),
        ("J5-GlobMR", GlobalMeanReversion(), False),
        ("J6-DriftRevAlpha", DriftReversalAlpha(), False),
        ("K1-LowVol", LowVolFactor(), False),
        ("K2-MultiFactor", MultiFactorComposite(), False),
        ("K3-SizeFactor", SizeFactor(), False),
        ("K4-ValueFactor", ValueFactor(), False),
        ("K5-RealAssets", RealAssetsFactor(), False),
        ("K6-QualMomComp", QualityMomentumComposite(), False),
        ("L1-VolRiskPrem", VolRiskPremium(), False),
        ("L2-VolSpread", VolSpreadHarvest(), False),
        ("L3-VolTermStr", VolTermStructure(), False),
        ("L4-VolOfVol", VolOfVolRegime(), False),
        ("L5-VolSpikeRecov", VolSpikeRecovery(), False),
        ("L6-CrossAssetVol", CrossAssetVolSignal(), False),
        ("L7-ImplRealSpread", ImpliedRealizedSpread(), False),
        ("L8-VolRegimeClust", VolRegimeClustering(), False),
        ("L9-VIXAdaptCarry", VIXAdaptiveCarry(), False),
        ("L10-DynVolRegime", DynamicVolRegimeSwitch(), False),
        ("L11-VolRiskPremAdapt", VolRiskPremiumAdaptive(), False),
        ("M1-DollarCarry", DollarCarry(), False),
        ("M2-GoldDollar", GoldDollarInverse(), False),
        ("M3-EMRisk", EMRiskPremium(), False),
        ("M4-CommMom", CommodityMomentum(), False),
        ("M5-RatesRegime", RatesRegimeTrade(), False),
        ("M7-GlobRot", GlobalRotation(), False),
        ("M8-CommMacro", CommodityMacroSignal(), False),
        ("M9-YieldCurveReg", YieldCurveRegime(), False),
        ("M10-MacroScore", MacroSignalScoreboard(), False),
        ("M11-AdaptMacro", AdaptiveMacroBlend(), False),
        ("M1b-DollarCarryScore", DollarCarryScoreboard(), False),
        ("N1-Seasonal", SeasonalStrategy(), False),
        ("N2-TurnOfMonth", TurnOfMonth(), False),
        ("N3-PreHoliday", PreHolidayDrift(), False),
        ("O1-TailRiskPar", TailRiskParity(), False),
        ("O4-BlackSwan", BlackSwanInsurance(), False),
        ("O6-TailHedge", TailHedgeOverlay(), True),
        ("O7-PrecMetals", PreciousMetalsCrisisHedge(), False),
        ("O8-VolConvexity", VolatilityConvexity(), False),
        ("O9-ATRCrisisAlpha", ATRCrisisAlpha(), False),
        ("P1-FeatureCombo", FeatureComboSignal(), False),
        ("P2-XGBoostCombo", XGBoostSignalCombo(), False),
        ("P3-GMMRegime", GMMRegimeClassifier(), False),
        ("P4-AdaptThresh", AdaptiveThreshold(), False),
        ("P5-CrossSecRanker", CrossSectionalRanker(), False),
        ("G6-MultiSigConsensus", MultiSignalConsensus(), False),
        ("DL1-TemporalCNN", TemporalCNNAlpha(), False),
        ("DL2-LSTMRegime", LSTMRegimeDetector(), False),
        ("DL3-AttentionRanker", AttentionCrossSectionalRanker(), False),
        ("Q1-QualityTrend", QualityTrend(), False),
        ("Q2-MultiAssetTrend", MultiAssetTrend(), False),
        ("Q3-MomCrashFilter", MomentumCrashFilter(), False),
        ("MF1-WeeklyMomRot", WeeklyMomentumRotation(), False),
        ("MF2-MonthlyMacro", MonthlyMacroRegime(), False),
        ("MF3-MultiTFTrend", MultiTimeframeTrend(), False),
        ("MF4-WeeklyMR", WeeklyMeanReversion(), False),
        ("R1-BearMarket", BearMarketAlpha(), True),
        ("R3-DefensiveRot", DefensiveRotationR3(), True),
        ("R4-AdaptRiskBudget", AdaptiveRiskBudget(), False),
        ("R5-RatesTightening", RatesTighteningAlpha(), False),
        ("R6-BondEquityHedge", BondEquityHedge(), False),
        ("R7-VolExplosion", VolExplosionAlpha(), False),
        ("R8-VolRegimeSwitcher", VolRegimeSwitcher(), False),
        ("R9-MultiAssetCTA", MultiAssetCTATrend(), False),
        ("R10-CommMacroOvly", CommodityMacroOverlay(), False),
    ]

    timings = []
    for name, strat, needs_regime in strategies:
        t0 = time.perf_counter()
        try:
            if needs_regime:
                w = strat.backtest_weights(p, r)
            else:
                w = strat.backtest_weights(p)
            result = backtest(p, w, BT_CONFIG)
            elapsed = time.perf_counter() - t0
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"  {elapsed:8.2f}s  {name} [ERROR: {e}]")
            timings.append((name, elapsed, "ERROR"))
            continue
        timings.append((name, elapsed, "OK"))
        print(f"  {elapsed:8.2f}s  {name}")

    # Phase 4: Ensemble timing
    print("\n[Phase 4] Ensemble timing...")
    ensemble_members = [
        MonthlyMacroRegime(), MultiAssetCTATrend(), BondEquityHedge(),
        AdaptiveRiskBudget(), CrossAssetVolSignal(), FeatureComboSignal(),
        CrashHedgeQQQ(), CryptoRecoverySurge(), VolRiskPremium(),
        VolOfVolRegime(), PreciousMetalsCrisisHedge(), TailRiskParity(),
        CryptoGoldDivergence(), FearGreedContrarian(), VolCarry(),
        DefensiveRotationR3(), VolTermStructure(), VolSpreadHarvest(),
        VolExplosionAlpha(), LowVolFactor(), QualityTrend(),
        VolSpikeRecovery(), MomentumCrashFilter(), ValueFactor(),
        SentimentCrisisAlpha(), SeasonalStrategy(), SentimentDivergence(),
        AdaptiveThreshold(), VolRegimeClustering(), YieldCurveRegime(),
    ]
    sharpe_scores = [
        1.24, 1.13, 1.07, 1.02, 1.01, 0.83, 0.94, 0.93, 0.92, 0.92,
        0.92, 0.90, 0.89, 0.90, 0.89, 0.86, 0.86, 0.85, 0.85,
        0.84, 0.84, 0.84, 0.83, 0.81, 0.79, 0.79, 0.78, 0.75, 0.70, 0.65,
    ]
    ensemble_cfg = EnsembleConfig(
        use_inverse_vol=False, max_gross_leverage=2.5, max_single_weight=0.15,
        dd_scale_start=-0.12, dd_scale_end=-0.22,
        prior_weights=[s**2 for s in sharpe_scores],
        correlation_hedge_enabled=True, correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True, vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30, leverage_elevated=1.8, leverage_crisis=1.2,
        vix_prior_scaling=True, vix_prior_k=0.05, vix_prior_base=20.0,
        vix_prior_lookback=20,
        drift_filter_enabled=True, drift_lookback=63, drift_threshold=0.58,
        drift_scale_weak=0.50,
    )
    ens = EnsembleStrategy(ensemble_members, ensemble_cfg)
    _, t_ens = time_fn("Ensemble generate_weights", ens.backtest_weights, p, r)

    # Phase 5: Multiplication factor from windows
    n_windows = 11
    print(f"\n[Phase 5] Window multiplication...")
    print(f"  Crisis windows: {n_windows}")
    print(f"  Each window runs ALL strategies + ensemble")

    # Summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)

    timings.sort(key=lambda x: x[1], reverse=True)
    total_strat = sum(t for _, t, _ in timings)

    print(f"\n  Data loading:          {t_load:8.2f}s")
    print(f"  Regime detection:      {t_regime:8.2f}s")
    print(f"  All strategies (1 win):{total_strat:8.2f}s")
    print(f"  Ensemble (1 window):   {t_ens:8.2f}s")
    print(f"  ---- Single window tot:{t_load + t_regime + total_strat + t_ens:8.2f}s")
    print(f"  x {n_windows} windows (approx):  {(total_strat + t_ens) * n_windows:8.1f}s")
    total_est = t_load + t_regime + (total_strat + t_ens) * n_windows
    print(f"  TOTAL ESTIMATED:       {total_est:8.1f}s ({total_est/60:.1f} min)")

    print(f"\n  TOP 20 SLOWEST STRATEGIES:")
    for i, (name, elapsed, status) in enumerate(timings[:20]):
        pct = elapsed / total_strat * 100
        print(f"    {i+1:2d}. {elapsed:7.2f}s ({pct:5.1f}%)  {name} [{status}]")

    print(f"\n  BOTTOM 10 (fastest):")
    for name, elapsed, status in timings[-10:]:
        print(f"      {elapsed:7.2f}s  {name}")

    t_total = time.perf_counter() - t0_total
    print(f"\n  Profiler wall time: {t_total:.1f}s ({t_total/60:.1f} min)")


if __name__ == "__main__":
    main()
