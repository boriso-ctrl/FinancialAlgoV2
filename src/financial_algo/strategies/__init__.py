"""Strategy bot package -- Categories B-P + ensemble."""

from financial_algo.strategies.base import Strategy

from financial_algo.strategies.crash_hedge import (
    CrashHedgeQQQ,
    FourStateTactical,
    VolCarry,
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
from financial_algo.strategies.pairs import MultiPairPortfolio, PairTrade
from financial_algo.strategies.crypto_crisis import (
    CryptoFlightToQuality,
    CryptoGoldDivergence,
    CryptoRecoverySurge,
    CryptoContagionHedge,
)
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig
from financial_algo.strategies.crisis_spike import (
    CommodityShockRider,
    DefenseSpikeBreakout,
    GoldFearRally,
    MultiAssetCrisisLong,
)

# New strategy categories
from financial_algo.strategies.momentum import (
    TimeSeriesMomentum,
    CrossSectionalMomentum,
    DualMomentum,
    MomentumVolScaled,
    GlobalMomentumRotation,
    KSTMomentum,
)
from financial_algo.strategies.mean_reversion import (
    SectorMeanReversion,
    RSIMeanReversion,
    OvernightGapFade,
    CointegrationPairs,
    GlobalMeanReversion,
    FormulaicAlphaMeanRev,
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
    FormulaicAlphaMomentum,
    QualityMomentumComposite,
)
from financial_algo.strategies.tail_risk import (
    TailRiskParity,
    BlackSwanInsurance,
    TailHedgeOverlay,
    PreciousMetalsCrisisHedge,
    DrawdownRecoveryTiming,
    VolatilityConvexity,
)
from financial_algo.strategies.signal_combo import (
    Alpha158Ranker,
    FeatureComboSignal,
    FundamentalMomentumSignal,
    GMMRegimeClassifier,
    MultiSignalConsensus,
    XGBoostSignalCombo,
)
from financial_algo.strategies.ml_strategies import (
    AdaptiveThreshold,
    CrossSectionalRanker,
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
    BearMarketAlpha,
    DefensiveRotationR3,
    AdaptiveRiskBudget,
    RatesTighteningAlpha,
    BondEquityHedge,
    VolExplosionAlpha,
    VolRegimeSwitcher,
    MultiAssetCTATrend,
    CommodityMacroOverlay,
)
from financial_algo.strategies.dl_strategies import (
    TemporalCNNAlpha,
    LSTMRegimeDetector,
    AttentionCrossSectionalRanker,
)

__all__ = [
    "Strategy",
    # D - Crash Hedge
    "CrashHedgeQQQ", "FourStateTactical", "VolCarry",
    # B - Oil Crisis
    "OilMomentumSurge", "OilShockHedge", "OilMeanReversion", "EnergyPairs",
    # C - War Crisis
    "DefenseRotation", "SafeHavenFlight", "PostWarRecovery", "ArmsRaceMomentum",
    # E - Pairs
    "MultiPairPortfolio", "PairTrade",
    # F - Crypto
    "CryptoFlightToQuality", "CryptoRecoverySurge", "CryptoGoldDivergence",
    "CryptoContagionHedge",
    # Ensemble
    "EnsembleStrategy", "EnsembleConfig",
    # Crisis Spike
    "CommodityShockRider", "DefenseSpikeBreakout", "GoldFearRally", "MultiAssetCrisisLong",
    # I - Momentum
    "TimeSeriesMomentum", "CrossSectionalMomentum", "DualMomentum", "MomentumVolScaled",
    "GlobalMomentumRotation", "KSTMomentum",
    # J - Mean Reversion
    "SectorMeanReversion", "RSIMeanReversion", "OvernightGapFade", "CointegrationPairs",
    "GlobalMeanReversion", "FormulaicAlphaMeanRev",
    # H - Fixed Income
    "YieldCurveTrade", "CreditSpreadMeanRev",
    # L - Volatility
    "VolRiskPremium", "VolSpreadHarvest", "VolOfVolRegime",
    "VolTermStructure", "VolSpikeRecovery", "CrossAssetVolSignal",
    "ImpliedRealizedSpread", "VolRegimeClustering",
    # M - Macro
    "DollarCarry", "GoldDollarInverse", "EMRiskPremium", "CommodityMomentum",
    "RatesRegimeTrade", "GlobalRotation",
    "CommodityMacroSignal", "YieldCurveRegime",
    # N - Seasonal
    "SeasonalStrategy", "TurnOfMonth", "PreHolidayDrift",
    # K - Factor
    "LowVolFactor", "MultiFactorComposite", "SizeFactor", "ValueFactor",
    "RealAssetsFactor", "FormulaicAlphaMomentum", "QualityMomentumComposite",
    # O - Tail Risk
    "TailRiskParity", "BlackSwanInsurance",
    "TailHedgeOverlay",
    "PreciousMetalsCrisisHedge",
    "DrawdownRecoveryTiming",
    "VolatilityConvexity",
    # P - Signal Combo / ML
    "Alpha158Ranker",
    "FeatureComboSignal",
    "FundamentalMomentumSignal",
    "XGBoostSignalCombo",
    "GMMRegimeClassifier",
    "MultiSignalConsensus",
    "AdaptiveThreshold",
    "CrossSectionalRanker",
    # Q - Quality Trend
    "QualityTrend", "MultiAssetTrend", "MomentumCrashFilter",
    # MF - Multi-Frequency
    "WeeklyMomentumRotation", "MonthlyMacroRegime",
    "MultiTimeframeTrend", "WeeklyMeanReversion",
    # R - Regime Hardening
    "BearMarketAlpha",
    "DefensiveRotationR3", "AdaptiveRiskBudget",
    "RatesTighteningAlpha", "BondEquityHedge",
    "VolExplosionAlpha", "VolRegimeSwitcher",
    "MultiAssetCTATrend", "CommodityMacroOverlay",
    # DL - Deep Learning
    "TemporalCNNAlpha", "LSTMRegimeDetector", "AttentionCrossSectionalRanker",
]
