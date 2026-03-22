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
    "ALL_EXPERIMENTAL",
]
