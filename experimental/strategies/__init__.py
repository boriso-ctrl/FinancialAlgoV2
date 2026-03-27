"""Experimental strategies package.

All experimental strategies, grouped by category:
  X -- Cross-Asset Divergence
  Y -- Behavioral Anomaly
  Z -- Structural Alpha
  W -- Alternative Signals
"""

from financial_algo.strategies.base import Strategy as _Strategy

from .cross_asset import (
    CopperGoldGrowth,
    CreditEquityDivergence,
    DollarWreckingBall,
)
from .behavioral import (
    DispositionEffectReversal,
    AttentionOverreaction,
    LunarCycleAlpha,
)
from .structural import (
    RebalancingFlow,
    GammaPin,
    SectorDispersion,
)
from .alt_signals import (
    BreadthDivergence,
    CorrelationRegimeBreak,
    LiquidityVacuum,
)
try:
    from .chronos_forecast import ChronosForecast
    _CHRONOS_AVAILABLE = True
except ImportError:  # chronos package not installed
    _CHRONOS_AVAILABLE = False
    ChronosForecast = None  # type: ignore[assignment,misc]

ALL_EXPERIMENTAL: list[_Strategy] = [
    CopperGoldGrowth(),
    CreditEquityDivergence(),
    DollarWreckingBall(),
    DispositionEffectReversal(),
    AttentionOverreaction(),
    LunarCycleAlpha(),
    RebalancingFlow(),
    GammaPin(),
    SectorDispersion(),
    BreadthDivergence(),
    CorrelationRegimeBreak(),
    LiquidityVacuum(),
]
if _CHRONOS_AVAILABLE:
    ALL_EXPERIMENTAL.append(ChronosForecast())

__all__ = [
    "CopperGoldGrowth",
    "CreditEquityDivergence",
    "DollarWreckingBall",
    "DispositionEffectReversal",
    "AttentionOverreaction",
    "LunarCycleAlpha",
    "RebalancingFlow",
    "GammaPin",
    "SectorDispersion",
    "BreadthDivergence",
    "CorrelationRegimeBreak",
    "LiquidityVacuum",
    "ChronosForecast",
    "ALL_EXPERIMENTAL",
]
