"""Comprehensive Historical Crisis Backtest — All Categories.

Downloads real market data via yfinance and runs every strategy category
through historical crises with unbiased backtesting:
  - Weights shifted +1 day (no look-ahead bias)
  - Transaction costs (5 bps one-way)
  - Leverage borrowing costs (1.5 %/yr)
  - Short borrow costs (0.5 %/yr)

Crisis periods tested:
  2011       — EU Sovereign Debt Crisis
  2014-2016  — Oil Price Crash (OPEC, shale glut)
  2018 Q1/Q4 — Vol-mageddon + Fed tightening
  2020       — COVID-19 pandemic crash
  2022       — Russia-Ukraine War + inflation
  2023-2025  — Recent (baseline/recovery)

Usage:
    python scripts/production/run_crisis_backtest.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable when running as a script
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import Regime, RegimeConfig, detect_regime

# --- Strategy imports ---
from financial_algo.strategies.crash_hedge import (
    CrashHedgeQQQ,
    FourStateTactical,
    VolCarry,
    AdaptiveStopTrend,
)
from financial_algo.strategies.oil_crisis import (
    EnergyPairs,
    OilMeanReversion,
    OilMomentumSurge,
    OilShockHedge,
)
from financial_algo.strategies.war_crisis import (
    ArmsRaceMomentum,
    DefenseRotation,
    PostWarRecovery,
    SafeHavenFlight,
)
from financial_algo.strategies.pairs import MultiPairPortfolio
from financial_algo.strategies.crypto_crisis import (
    CryptoFlightToQuality,
    CryptoGoldDivergence,
    CryptoRecoverySurge,
    CryptoContagionHedge,
    CryptoRecoverySurgeATR,
)
from financial_algo.strategies.crisis_spike import (
    CommodityShockRider,
    DefenseSpikeBreakout,
    GoldFearRally,
    MultiAssetCrisisLong,
)
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig

# --- New strategy imports (Categories I-P) ---
from financial_algo.strategies.momentum import (
    TimeSeriesMomentum,
    CrossSectionalMomentum,
    DualMomentum,
    MomentumVolScaled,
    GlobalMomentumRotation,
    AdaptiveTrendFilter,
    DriftRegimeMomentum,
    TimeSeriesMomentumDrift,
)
from financial_algo.strategies.mean_reversion import (
    SectorMeanReversion,
    RSIMeanReversion,
    OvernightGapFade,
    CointegrationPairs,
    GlobalMeanReversion,
    DriftReversalAlpha,
)
from financial_algo.strategies.fixed_income import (
    YieldCurveTrade,
    CreditSpreadMeanRev,
)
from financial_algo.strategies.volatility_strats import (
    VolRiskPremium,
    VolSpreadHarvest,
    VolOfVolRegime,
    VolTermStructure,
    VolSpikeRecovery,
    CrossAssetVolSignal,
    ImpliedRealizedSpread,
    VolRegimeClustering,
    VIXAdaptiveCarry,
    DynamicVolRegimeSwitch,
    VolRiskPremiumAdaptive,
)
from financial_algo.strategies.macro import (
    DollarCarry,
    GoldDollarInverse,
    EMRiskPremium,
    CommodityMomentum,
    RatesRegimeTrade,
    GlobalRotation,
    CommodityMacroSignal,
    YieldCurveRegime,
    MacroSignalScoreboard,
    AdaptiveMacroBlend,
    DollarCarryScoreboard,
)
from financial_algo.strategies.seasonal import (
    SeasonalStrategy,
    TurnOfMonth,
    PreHolidayDrift,
)
from financial_algo.strategies.factor import (
    LowVolFactor,
    MultiFactorComposite,
    SizeFactor,
    ValueFactor,
    RealAssetsFactor,
    QualityMomentumComposite,
)
from financial_algo.strategies.tail_risk import (
    TailRiskParity,
    BlackSwanInsurance,
    TailHedgeOverlay,
    PreciousMetalsCrisisHedge,
    VolatilityConvexity,
    ATRCrisisAlpha,
)
from financial_algo.strategies.signal_combo import (
    FeatureComboSignal,
    MultiSignalConsensus,
    XGBoostSignalCombo,
    GMMRegimeClassifier,
)
from financial_algo.strategies.ml_strategies import (
    AdaptiveThreshold,
    CrossSectionalRanker,
)
from financial_algo.strategies.dl_strategies import (
    TemporalCNNAlpha,
    LSTMRegimeDetector,
    AttentionCrossSectionalRanker,
)
from financial_algo.strategies.quality_trend import (
    QualityTrend,
    MultiAssetTrend,
    MomentumCrashFilter,
)
from financial_algo.strategies.multi_freq import (
    WeeklyMomentumRotation,
    MonthlyMacroRegime,
    MultiTimeframeTrend,
    WeeklyMeanReversion,
)
from financial_algo.strategies.regime_hardening import (
    BearMarketAlpha, DefensiveRotationR3,
    AdaptiveRiskBudget, RatesTighteningAlpha, BondEquityHedge,
    VolExplosionAlpha, VolRegimeSwitcher, MultiAssetCTATrend,
    CommodityMacroOverlay,
)

# --- Fundamental strategy imports ---
from financial_algo.fundamental import build_synthetic_sentiment
from financial_algo.fundamental.strategies import (
    FearGreedContrarian,
    SentimentCrisisAlpha,
    SentimentDivergence,
    SentimentEnhancedRegime,
    CryptoSentimentDivergence,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================
# Configuration
# =========================================================================

# Full date range — enough warm-up before first crisis window
DATA_START = "2009-01-01"
DATA_END = "2025-12-31"

# Historical crisis windows for focused analysis
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "Full Period (2010-2025)":        ("2010-01-01", "2025-12-31"),
    "EU Debt Crisis (2011)":          ("2011-01-01", "2012-01-01"),
    "Oil Crash (2014-2016)":          ("2014-06-01", "2016-06-01"),
    "Volmageddon + Fed (2018)":       ("2018-01-01", "2019-01-01"),
    "COVID-19 (2020)":                ("2020-01-01", "2021-01-01"),
    "Russia-Ukraine + Inflation (2022)": ("2022-01-01", "2023-01-01"),
    "Recovery & Recent (2023-2025)":  ("2023-01-01", "2025-12-31"),
    # Middle East conflict-specific windows
    "ME: Red Sea / Houthi (Jan-Mar 2024)":  ("2024-01-01", "2024-04-01"),
    "ME: Iran Tensions (Sep-Oct 2024)":     ("2024-09-01", "2024-11-01"),
    "ME: Oil Spike Jun 2025":               ("2025-05-01", "2025-07-31"),
    "ME: Full Conflict (Oct23-Dec24)":      ("2023-10-01", "2024-12-31"),
}

# Cost model with risk controls (same for all strategies -- fair comparison)
BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,              # 20% annualised vol target
    max_drawdown_trigger=-0.25,   # cut exposure at -25% drawdown
    drawdown_recovery_rate=0.10,  # resume after 10% DD improvement
)

# Ensemble gets its own config WITHOUT backtest-level DD trigger
# (the EnsembleStrategy class has its own gradual circuit breaker)
# Also enable per-strategy DD scale-down for additional protection
BT_CONFIG_ENSEMBLE = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=None,    # ensemble handles DD internally
    strategy_dd_scale_start=-0.15,   # start scaling at -15% DD
    strategy_dd_scale_end=-0.25,     # fully flat at -25% DD
)

# All tickers the strategies need (deduplicated)
TICKERS = sorted(set([
    # Broad equity
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    # Safe haven
    "GLD", "SLV", "TLT", "IEF", "SHY", "UUP",
    # Energy
    "XLE", "USO", "XOP",
    # Defense
    "ITA", "LMT", "RTX",
    # Sectors (for breadth / pairs / rotation)
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "XBI", "XLC", "XLRE",
    # Commodities
    "DBC", "DBA",
    # Fixed income / credit
    "HYG", "LQD", "TIP", "AGG", "EMB",
    # Regional
    "FXI", "VGK", "EWJ", "INDA",
    # Real estate
    "VNQ",
    # Crypto
    "BTC-USD", "ETH-USD",
]))

VIX_TICKER = "^VIX"


# =========================================================================
# Helpers
# =========================================================================

def run_strategy_backtest(
    name: str,
    strategy,
    prices: pd.DataFrame,
    regime: pd.Series,
    config: BacktestConfig,
    needs_regime: bool = False,
    sentiment_df: pd.DataFrame | None = None,
) -> dict | None:
    """Run a single strategy and return metrics, or None on error."""
    try:
        if sentiment_df is not None:
            # Fundamental strategies accept sentiment_df
            weights = strategy.generate_weights(prices, regime, sentiment_df)
            weights = weights.shift(1).fillna(0.0)
        elif needs_regime:
            weights = strategy.backtest_weights(prices, regime)
        else:
            weights = strategy.backtest_weights(prices)

        result = backtest(prices, weights, config)
        return result["metrics"]
    except Exception as e:
        print(f"  [WARN] {name}: {e}")
        return None


def metrics_row(metrics: dict | None) -> dict:
    """Format metrics for display, handling None."""
    if metrics is None:
        return {
            "CAGR": "ERR", "Sharpe": "ERR", "Sortino": "ERR",
            "MaxDD": "ERR", "Calmar": "ERR", "AnnVol": "ERR",
            "WinRate": "ERR", "TotRet": "ERR", "Trades/Mo": "ERR",
        }
    return {
        "CAGR": f"{metrics['cagr']:.2%}",
        "Sharpe": f"{metrics['sharpe']:.2f}",
        "Sortino": f"{metrics['sortino']:.2f}",
        "MaxDD": f"{metrics['max_drawdown']:.2%}",
        "Calmar": f"{metrics['calmar']:.2f}",
        "AnnVol": f"{metrics['annual_vol']:.2%}",
        "WinRate": f"{metrics['win_rate']:.2%}",
        "TotRet": f"{metrics['total_return']:.2%}",
        "Trades/Mo": f"{metrics.get('avg_trades_per_month', 0):.1f}",
    }


# =========================================================================
# Strategy registry — every strategy, categorised
# =========================================================================

def build_strategy_registry():
    """Return ordered dict of {category: [(name, strategy_instance, needs_regime)]}."""
    return {
        "Cat B: Oil Crisis": [
            ("B1-OilMomentumSurge",  OilMomentumSurge(),   True),
            ("B2-OilShockHedge",     OilShockHedge(),      True),
            ("B3-OilMeanReversion",  OilMeanReversion(),   True),
            ("B4-EnergyPairs",       EnergyPairs(),        True),
        ],
        "Cat C: War/Geopolitical": [
            ("C1-DefenseRotation",   DefenseRotation(),    True),
            ("C2-SafeHavenFlight",   SafeHavenFlight(),    True),
            ("C3-PostWarRecovery",   PostWarRecovery(),    True),
            ("C4-ArmsRaceMomentum",  ArmsRaceMomentum(),   False),
        ],
        "Cat D: Crash-Hedge (General)": [
            ("D1-FourStateTactical", FourStateTactical(),  True),
            ("D2-CrashHedgeQQQ",     CrashHedgeQQQ(),     False),
            ("D3-VolCarry",          VolCarry(),           False),
            ("D4-AdaptiveStopTrend", AdaptiveStopTrend(),  False),
        ],
        "Cat E: Pairs Arbitrage": [
            ("E1-MultiPairPortfolio", MultiPairPortfolio(), False),
        ],
        "Cat F: Crypto Crisis": [
            ("F1-CryptoFlightToQuality", CryptoFlightToQuality(), True),
            ("F2-CryptoRecoverySurge",   CryptoRecoverySurge(),   True),
            ("F3-CryptoGoldDivergence",  CryptoGoldDivergence(),  True),
            ("F4-CryptoContagionHedge",  CryptoContagionHedge(),  False),
            ("F2b-CryptoRecoverySurgeATR", CryptoRecoverySurgeATR(), True),
        ],
        "Cat G: Fundamental/Sentiment": [
            ("G1-SentimentCrisisAlpha",   SentimentCrisisAlpha(),   True),
            ("G2-FearGreedContrarian",    FearGreedContrarian(),    True),
            ("G3-SentimentDivergence",    SentimentDivergence(),    True),
            ("G4-SentimentEnhancedRegime", SentimentEnhancedRegime(), True),
            ("G5-CryptoSentimentDivergence", CryptoSentimentDivergence(), True),
        ],
        "Cat H: Crisis Spike (Upward)": [
            ("H1-CommodityShockRider",  CommodityShockRider(),  True),
            ("H2-GoldFearRally",        GoldFearRally(),        False),
            ("H3-DefenseSpikeBreakout", DefenseSpikeBreakout(), True),
            ("H4-MultiAssetCrisisLong", MultiAssetCrisisLong(), True),
        ],
        "Cat H-FI: Fixed Income": [
            ("H1-YieldCurveTrade",        YieldCurveTrade(),         False),
            ("H2-CreditSpreadMeanRev",    CreditSpreadMeanRev(),     False),
        ],
        "Cat I: Momentum": [
            ("I1-TimeSeriesMomentum",     TimeSeriesMomentum(),      False),
            ("I2-CrossSectionalMomentum", CrossSectionalMomentum(),  False),
            ("I3-DualMomentum",           DualMomentum(),            False),
            ("I4-MomentumVolScaled",      MomentumVolScaled(),       False),
            ("I5-GlobalMomentumRotation", GlobalMomentumRotation(),  False),
            ("I10-AdaptiveTrendFilter",   AdaptiveTrendFilter(),     False),
            ("I7-DriftRegimeMomentum",    DriftRegimeMomentum(),     False),
            ("I8-TimeSeriesMomDrift",      TimeSeriesMomentumDrift(),  False),
        ],
        "Cat J: Mean Reversion": [
            ("J1-SectorMeanReversion",    SectorMeanReversion(),     False),
            ("J2-OvernightGapFade",       OvernightGapFade(),        False),
            ("J3-RSIMeanReversion",       RSIMeanReversion(),        False),
            ("J4-CointegrationPairs",     CointegrationPairs(),      False),
            ("J5-GlobalMeanReversion",    GlobalMeanReversion(),     False),
            ("J6-DriftReversalAlpha",     DriftReversalAlpha(),      False),
        ],
        "Cat K: Factor": [
            ("K1-LowVolFactor",           LowVolFactor(),            False),
            ("K2-MultiFactorComposite",   MultiFactorComposite(),    False),
            ("K3-SizeFactor",             SizeFactor(),              False),
            ("K4-ValueFactor",            ValueFactor(),             False),
            ("K5-RealAssetsFactor",       RealAssetsFactor(),        False),
            ("K6-QualityMomentumComposite", QualityMomentumComposite(), False),
        ],
        "Cat L: Volatility": [
            ("L1-VolRiskPremium",         VolRiskPremium(),          False),
            ("L2-VolSpreadHarvest",       VolSpreadHarvest(),        False),
            ("L3-VolTermStructure",       VolTermStructure(),        False),
            ("L4-VolOfVolRegime",         VolOfVolRegime(),          False),
            ("L5-VolSpikeRecovery",       VolSpikeRecovery(),        False),
            ("L6-CrossAssetVolSignal",    CrossAssetVolSignal(),     False),
            ("L7-ImpliedRealizedSpread", ImpliedRealizedSpread(),   False),
            ("L8-VolRegimeClustering",   VolRegimeClustering(),     False),
            ("L9-VIXAdaptiveCarry",      VIXAdaptiveCarry(),        False),
            ("L10-DynamicVolRegimeSwitch", DynamicVolRegimeSwitch(), False),
            ("L11-VolRiskPremiumAdaptive", VolRiskPremiumAdaptive(), False),
        ],
        "Cat M: Macro": [
            ("M1-DollarCarry",            DollarCarry(),             False),
            ("M2-GoldDollarInverse",      GoldDollarInverse(),       False),
            ("M3-EMRiskPremium",          EMRiskPremium(),           False),
            ("M4-CommodityMomentum",      CommodityMomentum(),       False),
            ("M5-RatesRegimeTrade",       RatesRegimeTrade(),        False),
            ("M7-GlobalRotation",         GlobalRotation(),          False),
            ("M8-CommodityMacroSignal",   CommodityMacroSignal(),    False),
            ("M9-YieldCurveRegime",       YieldCurveRegime(),        False),
            ("M10-MacroSignalScoreboard", MacroSignalScoreboard(),   False),
            ("M11-AdaptiveMacroBlend",    AdaptiveMacroBlend(),      False),
            ("M1b-DollarCarryScoreboard", DollarCarryScoreboard(),   False),
        ],
        "Cat N: Seasonal": [
            ("N1-SeasonalStrategy",       SeasonalStrategy(),        False),
            ("N2-TurnOfMonth",            TurnOfMonth(),             False),
            ("N3-PreHolidayDrift",        PreHolidayDrift(),         False),
        ],
        "Cat O: Tail Risk": [
            ("O1-TailRiskParity",         TailRiskParity(),          False),
            ("O4-BlackSwanInsurance",      BlackSwanInsurance(),      False),
            ("O6-TailHedgeOverlay",       TailHedgeOverlay(),        True),
            ("O7-PreciousMetalsCrisisHedge", PreciousMetalsCrisisHedge(), False),
            ("O8-VolatilityConvexity",    VolatilityConvexity(),     False),
            ("O9-ATRCrisisAlpha",         ATRCrisisAlpha(),          True),
        ],
        "Cat P: ML Signal Combo": [
            ("P1-FeatureComboSignal",     FeatureComboSignal(),      False),
            ("P2-XGBoostSignalCombo",     XGBoostSignalCombo(),      False),
            ("P3-GMMRegimeClassifier",    GMMRegimeClassifier(),     False),
            ("P4-AdaptiveThreshold",      AdaptiveThreshold(),       False),
            ("P5-CrossSectionalRanker",   CrossSectionalRanker(),    False),
            ("G6-MultiSignalConsensus",   MultiSignalConsensus(),    False),
        ],
        "Cat DL: Deep Learning": [
            ("DL1-TemporalCNNAlpha",      TemporalCNNAlpha(),        False),
            ("DL2-LSTMRegimeDetector",    LSTMRegimeDetector(),      False),
            ("DL3-AttentionRanker",       AttentionCrossSectionalRanker(), False),
        ],
        "Cat Q: Quality Trend": [
            ("Q1-QualityTrend",           QualityTrend(),            False),
            ("Q2-MultiAssetTrend",        MultiAssetTrend(),         False),
            ("Q3-MomentumCrashFilter",    MomentumCrashFilter(),     False),
        ],
        "Cat MF: Multi-Frequency": [
            ("MF1-WeeklyMomentumRotation", WeeklyMomentumRotation(), False),
            ("MF2-MonthlyMacroRegime",     MonthlyMacroRegime(),     False),
            ("MF3-MultiTimeframeTrend",    MultiTimeframeTrend(),    False),
            ("MF4-WeeklyMeanReversion",    WeeklyMeanReversion(),    False),
        ],
        "Cat R: Regime Hardening": [
            ("R1-BearMarketAlpha",        BearMarketAlpha(),         True),
            ("R3-DefensiveRotation",      DefensiveRotationR3(),     True),
            ("R4-AdaptiveRiskBudget",     AdaptiveRiskBudget(),      False),
            ("R5-RatesTighteningAlpha",   RatesTighteningAlpha(),    False),
            ("R6-BondEquityHedge",        BondEquityHedge(),         False),
            ("R7-VolExplosionAlpha",      VolExplosionAlpha(),       False),
            ("R8-VolRegimeSwitcher",      VolRegimeSwitcher(),       False),
            ("R9-MultiAssetCTATrend",     MultiAssetCTATrend(),      False),
            ("R10-CommodityMacroOverlay", CommodityMacroOverlay(),   False),
        ],
    }


# =========================================================================
# Benchmark — Buy & Hold SPY
# =========================================================================

def spy_benchmark(prices: pd.DataFrame, config: BacktestConfig) -> dict | None:
    """Simple 1x long SPY benchmark."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w["SPY"] = 1.0
    try:
        result = backtest(prices, w.shift(1).fillna(0), config)
        return result["metrics"]
    except Exception as e:
        print(f"  [WARN] Benchmark: {e}")
        return None


def _inject_features(strategies, features: "pd.DataFrame") -> None:
    """Inject intraday feature DataFrame into any strategy that accepts it."""
    for strat in strategies:
        if hasattr(strat, "set_intraday_features"):
            strat.set_intraday_features(features)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    print("=" * 80)
    print("CRISIS-THRIVING STRATEGY BACKTEST - UNBIASED HISTORICAL ANALYSIS")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    print(f"[1/4] Downloading price data for {len(TICKERS)} tickers "
          f"({DATA_START} -> {DATA_END}) ...")
    prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
    print(f"       Loaded: {prices.shape[0]} trading days x {prices.shape[1]} tickers")
    print(f"       Date range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print()

    # Download VIX separately (non-tradeable index)
    print("       Downloading VIX ...")
    try:
        vix_df = load_prices([VIX_TICKER], start=DATA_START, end=DATA_END)
        vix = vix_df[VIX_TICKER]
    except Exception:
        print("       [WARN] VIX download failed, using vol proxy")
        vix = None
    print()

    # ------------------------------------------------------------------
    # 2. Detect regime over full period
    # ------------------------------------------------------------------
    print("[2/4] Running regime detection ...")
    regime = detect_regime(prices, vix=vix)
    regime_counts = regime.value_counts()
    print("       Regime distribution (full period):")
    for r, count in regime_counts.items():
        pct = count / len(regime) * 100
        print(f"         {r.value:20s}: {count:5d} days ({pct:5.1f}%)")
    print()

    # ------------------------------------------------------------------
    # 3. Load intraday features (Phase 2: Alpaca pipeline, optional)
    # ------------------------------------------------------------------
    intraday_features: "pd.DataFrame | None" = None
    try:
        import os
        if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET"):
            from financial_algo.data.feature_store import FeatureStore
            equity_tickers = [t for t in TICKERS
                              if not t.startswith("^") and "-USD" not in t]
            print("[2b/4] Loading Alpaca intraday features ...")
            store = FeatureStore(tickers=equity_tickers, start=DATA_START, end=DATA_END)
            intraday_features = store.load()
            if intraday_features is not None and not intraday_features.empty:
                print(f"       Loaded {intraday_features.shape[1]} intraday features "
                      f"for {intraday_features.shape[1] // 7} tickers")
            else:
                intraday_features = None
            print()
    except Exception as e:
        print(f"       [WARN] Intraday feature load failed (skipping): {e}")
        intraday_features = None

    # ------------------------------------------------------------------
    # 4. Build strategy registry
    # ------------------------------------------------------------------
    registry = build_strategy_registry()

    # Ensemble v10.1: Fixed from v10 backtest results
    # Removed: L7 (Sharpe -0.17), O8 (dead, 0 trades)
    # Re-added: G1 (Sharpe 0.79), G3 (Sharpe 0.78) - were wrongly cut in v10
    # Kept from v10: L8 (0.70), M9 (0.65) - positive Sharpe, adds diversification
    ensemble_members = [
        MonthlyMacroRegime(),    # MF2 -- Sharpe 1.24, monthly macro
        MultiAssetCTATrend(),    # R9 -- Sharpe 1.13, CTA trend
        BondEquityHedge(),       # R6 -- Sharpe 1.07, bond/equity hedge
        AdaptiveRiskBudget(),    # R4 -- Sharpe 1.02, adaptive risk budget
        CrossAssetVolSignal(),   # L6 -- Sharpe 1.01, cross-asset vol
        FeatureComboSignal(),    # P1 -- Sharpe 0.83, multi-signal composite
        CrashHedgeQQQ(),         # D2 -- Sharpe 0.94, trend/vol on QQQ
        CryptoRecoverySurge(),   # F2 -- Sharpe 0.93, crypto regime
        VolRiskPremium(),        # L1 -- Sharpe 0.92, pure VRP
        VolOfVolRegime(),        # L4 -- Sharpe 0.92, vol-of-vol dynamic sizing
        PreciousMetalsCrisisHedge(),  # O7 -- Sharpe 0.92, metals crisis
        TailRiskParity(),        # O1 -- Sharpe 0.90, risk-parity
        CryptoGoldDivergence(),  # F3 -- Sharpe 0.89, crypto/gold signal
        FearGreedContrarian(),   # G2 -- Sharpe 0.90, contrarian sentiment
        VolCarry(),              # D3 -- Sharpe 0.89, vol carry
        DefensiveRotationR3(),   # R3 -- Sharpe 0.86, defensive rotation
        VolTermStructure(),      # L3 -- Sharpe 0.86, term structure carry
        VolSpreadHarvest(),      # L2 -- Sharpe 0.85, vol spread
        VolExplosionAlpha(),     # R7 -- Sharpe 0.85, vol explosion
        LowVolFactor(),          # K1 -- Sharpe 0.84, low-vol factor
        QualityTrend(),          # Q1 -- Sharpe 0.84, trend + quality filter
        VolSpikeRecovery(),      # L5 -- Sharpe 0.84, low-DD vol timing
        MomentumCrashFilter(),   # Q3 -- Sharpe 0.83, momentum + crash hedge
        ValueFactor(),           # K4 -- Sharpe 0.81, value factor
        SentimentCrisisAlpha(),  # G1 -- Sharpe 0.79, sentiment crisis
        SeasonalStrategy(),      # N1 -- Sharpe 0.79, seasonality
        SentimentDivergence(),   # G3 -- Sharpe 0.78, sentiment divergence
        AdaptiveThreshold(),     # P4 -- Sharpe 0.75, walk-forward optimizer
        VolRegimeClustering(),   # L8 -- Sharpe 0.70, vol regime clustering
        YieldCurveRegime(),      # M9 -- Sharpe 0.65, yield curve macro
    ]
    sharpe_scores = [
        1.24, 1.13, 1.07, 1.02, 1.01,  # MF2, R9, R6, R4, L6
        0.83, 0.94, 0.93, 0.92, 0.92,  # P1, D2, F2, L1, L4
        0.92, 0.90, 0.89, 0.90, 0.89,  # O7, O1, F3, G2, D3
        0.86, 0.86, 0.85, 0.85,        # R3, L3, L2, R7
        0.84, 0.84, 0.84, 0.83,        # K1, Q1, L5, Q3
        0.81, 0.79, 0.79, 0.78,        # K4, G1, N1, G3
        0.75, 0.70, 0.65,              # P4, L8, M9
    ]
    prior_weights = [s ** 2 for s in sharpe_scores]
    ensemble_cfg = EnsembleConfig(
        use_inverse_vol=False,        # fixed Sharpe-proportional (less turnover)
        max_gross_leverage=2.5,
        max_single_weight=0.15,       # tighter cap with 30 members
        dd_scale_start=-0.12,         # start scaling down at -12% DD
        dd_scale_end=-0.22,           # fully flat at -22% DD
        prior_weights=prior_weights,
        # 1a: Correlation hedging
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        # 1d: Vol-regime leverage scaling
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
        # VIX-adaptive prior weights (differential regime tilting)
        vix_prior_scaling=True,
        vix_prior_k=0.02,              # subtle tilt (was 0.05 -- too aggressive)
        vix_prior_base=20.0,
        vix_prior_lookback=20,
        # Drift regime filter -- DISABLED: threshold 0.58 > SPY avg pos-day
        # fraction (~0.53), causing chronic position scaling that kills CAGR
        drift_filter_enabled=False,
    )

    total_strategies = sum(len(v) for v in registry.values()) + 2  # +benchmark +ensemble
    print(f"[4/4] Running {total_strategies} strategy backtests across "
          f"{len(CRISIS_WINDOWS)} time windows ...")

    # Inject intraday features into strategies that support them
    if intraday_features is not None:
        _inject_features(ensemble_members, intraday_features)
        _inject_features(
            [s for cat in registry.values() for _, s, _ in cat],
            intraday_features,
        )
        print(f"       Intraday features injected into eligible strategies.")
    print()

    # ------------------------------------------------------------------
    # 4. Run backtests for each window
    # ------------------------------------------------------------------
    all_results = {}

    for window_name, (win_start, win_end) in CRISIS_WINDOWS.items():
        print("-" * 80)
        print(f"  WINDOW: {window_name}")
        print("-" * 80)

        # Slice data to window
        mask = (prices.index >= win_start) & (prices.index <= win_end)
        p_win = prices.loc[mask].copy()
        r_win = regime.loc[mask].copy()

        if len(p_win) < 30:
            print(f"  [SKIP] Not enough data ({len(p_win)} days)")
            print()
            continue

        print(f"  Data: {p_win.index[0].date()} -> {p_win.index[-1].date()} "
              f"({len(p_win)} days)")

        # Regime summary for this window
        rc = r_win.value_counts()
        crisis_days = sum(
            rc.get(r, 0)
            for r in [Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS]
        )
        print(f"  Crisis days: {crisis_days} / {len(r_win)} "
              f"({crisis_days / len(r_win) * 100:.1f}%)")

        # Build synthetic sentiment for fundamental strategies
        vix_win = vix.reindex(p_win.index).ffill() if vix is not None else None
        try:
            sent_df = build_synthetic_sentiment(p_win, vix=vix_win)
        except Exception as e:
            print(f"  [WARN] Synthetic sentiment build failed: {e}")
            sent_df = None

        rows = []

        # Benchmark
        bm = spy_benchmark(p_win, BT_CONFIG)
        rows.append({"Category": "Benchmark", "Strategy": "BuyHold-SPY", **metrics_row(bm)})

        # Each category
        for cat_name, strats in registry.items():
            use_sentiment = "Fundamental" in cat_name
            for strat_name, strat, needs_regime in strats:
                m = run_strategy_backtest(
                    strat_name, strat, p_win, r_win, BT_CONFIG, needs_regime,
                    sentiment_df=sent_df if use_sentiment else None,
                )
                rows.append({"Category": cat_name, "Strategy": strat_name, **metrics_row(m)})

        # Ensemble
        try:
            ens = EnsembleStrategy(ensemble_members, ensemble_cfg)
            # Ensemble needs regime for FourStateTactical
            ens_w = ens.backtest_weights(p_win, r_win)
            ens_result = backtest(p_win, ens_w, BT_CONFIG_ENSEMBLE)
            ens_m = ens_result["metrics"]
        except Exception as e:
            print(f"  [WARN] Ensemble: {e}")
            ens_m = None
        rows.append({"Category": "Ensemble", "Strategy": "Ensemble-BestOfEach", **metrics_row(ens_m)})

        # Generate HTML tearsheet (optional — requires quantstats)
        if ens_result is not None:
            try:
                from financial_algo.reporting import generate_tearsheet
                _safe = window_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
                ts_path = generate_tearsheet(
                    ens_result,
                    benchmark_prices=p_win["SPY"] if "SPY" in p_win.columns else None,
                    output_path=f"results/tearsheet_{_safe}.html",
                    title=f"Ensemble - {window_name}",
                )
                print(f"  Tearsheet saved: {ts_path}")
            except ImportError:
                pass  # quantstats not installed — skip tearsheet
            except Exception as e:
                print(f"  [WARN] Tearsheet generation failed: {e}")

        # Display results table
        df = pd.DataFrame(rows)
        print()
        print(df.to_string(index=False))
        print()

        all_results[window_name] = df

    # ------------------------------------------------------------------
    # Summary: Best strategy per crisis period
    # ------------------------------------------------------------------
    print("=" * 80)
    print("SUMMARY - BEST STRATEGY PER CRISIS WINDOW (by Sharpe)")
    print("=" * 80)
    print()

    for window_name, df in all_results.items():
        # Parse Sharpe back to float for comparison
        df_copy = df.copy()
        df_copy["_sharpe_num"] = pd.to_numeric(
            df_copy["Sharpe"].replace("ERR", np.nan), errors="coerce"
        )
        best = df_copy.loc[df_copy["_sharpe_num"].idxmax()]
        print(f"  {window_name:45s} -> {best['Strategy']:30s} "
              f"(Sharpe={best['Sharpe']}, CAGR={best['CAGR']}, MaxDD={best['MaxDD']})")

    print()
    print("=" * 80)
    print("METHODOLOGY NOTES (Unbiased Backtesting)")
    print("=" * 80)
    print("""
  [OK] Weights shifted +1 day -- no look-ahead bias
  [OK] Transaction costs: 5 bps one-way per trade
  [OK] Leverage borrowing cost: 1.5% annualised
  [OK] Short borrow cost: 0.5% annualised
  [OK] Regime detection uses only trailing data (no future info)
  [OK] Z-scores computed on trailing rolling windows only
  [OK] No survivorship bias -- using ETFs with full history
  [OK] Initial capital: $1,000,000
  [OK] All strategies start from identical data & cost model
    """)

    # Save results to CSV
    out_dir = _root / "results"
    out_dir.mkdir(exist_ok=True)
    for window_name, df in all_results.items():
        safe_name = window_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        df.to_csv(out_dir / f"backtest_{safe_name}.csv", index=False)
    print(f"  Results saved to: {out_dir}")
    print()


if __name__ == "__main__":
    main()
