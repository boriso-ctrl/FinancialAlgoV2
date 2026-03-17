"""Backtest engines, blending utilities, and metrics."""

from .blender import blend_signals
from .engine import run_backtest
from .hft_engine import HFTBacktestEngine
from .metrics import compute_metrics
from .strategy_registry import StrategyRegistry, registry

__all__ = [
    "run_backtest",
    "blend_signals",
    "compute_metrics",
    "StrategyRegistry",
    "registry",
    "HFTBacktestEngine",
]
