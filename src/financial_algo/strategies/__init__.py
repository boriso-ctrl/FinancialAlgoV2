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
)
from financial_algo.strategies.mean_reversion import (
    SectorMeanReversion,
    RSIMeanReversion,
    OvernightGapFade,
    CointegrationPairs,
)
from financial_algo.strategies.fixed_income import (
    YieldCurveTrade,
    CreditSpreadMeanRev,
    DurationTiming,
)
from financial_algo.strategies.volatility_strats import (
    VolRiskPremium,
    VolSpreadHarvest,
    VolOfVolRegime,
    VolTermStructure,
    VolSpikeRecovery,
)
from financial_algo.strategies.macro import (
    DollarCarry,
    GoldDollarInverse,
    EMRiskPremium,
    CommodityMomentum,
    RatesRegimeTrade,
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
)
from financial_algo.strategies.tail_risk import (
    TailRiskParity,
    CrisisAlphaMomentum,
    BlackSwanInsurance,
    CrisisRotation,
    VIXSpikeRecovery,
    TailHedgeOverlay,
)
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.quality_trend import (
    QualityTrend,
    MultiAssetTrend,
    MomentumCrashFilter,
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
    # Ensemble
    "EnsembleStrategy", "EnsembleConfig",
    # Crisis Spike
    "CommodityShockRider", "DefenseSpikeBreakout", "GoldFearRally", "MultiAssetCrisisLong",
    # I - Momentum
    "TimeSeriesMomentum", "CrossSectionalMomentum", "DualMomentum", "MomentumVolScaled",
    # J - Mean Reversion
    "SectorMeanReversion", "RSIMeanReversion", "OvernightGapFade", "CointegrationPairs",
    # H - Fixed Income
    "YieldCurveTrade", "CreditSpreadMeanRev", "DurationTiming",
    # L - Volatility
    "VolRiskPremium", "VolSpreadHarvest", "VolOfVolRegime",
    "VolTermStructure", "VolSpikeRecovery",
    # M - Macro
    "DollarCarry", "GoldDollarInverse", "EMRiskPremium", "CommodityMomentum",
    "RatesRegimeTrade",
    # N - Seasonal
    "SeasonalStrategy", "TurnOfMonth", "PreHolidayDrift",
    # K - Factor
    "LowVolFactor", "MultiFactorComposite", "SizeFactor", "ValueFactor",
    # O - Tail Risk
    "TailRiskParity", "CrisisAlphaMomentum", "BlackSwanInsurance",
    "CrisisRotation", "VIXSpikeRecovery", "TailHedgeOverlay",
    # P - Signal Combo
    "FeatureComboSignal",
    # Q - Quality Trend
    "QualityTrend", "MultiAssetTrend", "MomentumCrashFilter",
]
