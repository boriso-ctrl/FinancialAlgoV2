"""Adaptive blending with importance weighting and optional exponential scaling."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


def _apply_exponential_scaling(
    signal: pd.DataFrame,
    base: float = 2.3,
    max_signal_abs: float = 5.0,
) -> pd.DataFrame:
    """Apply symmetric exponential scaling to normalized signal.
    
    Args:
        signal: DataFrame with values typically in [-max_signal_abs, max_signal_abs]
        base: Exponential curve base (1.5-3.0)
        max_signal_abs: Signal clipping bound
    
    Returns:
        Exponentially scaled signal in [-max_signal_abs, max_signal_abs]
    """
    # Normalize to [-1, 1]
    normalized = signal / max_signal_abs
    normalized = normalized.clip(-1, 1)
    
    # Apply symmetric exponential
    def exponential_scale(x):
        if x == 0:
            return 0
        abs_x = abs(x)
        exp_val = (base ** abs_x - 1) / (base - 1)
        return np.sign(x) * exp_val
    
    scaled = normalized.map(exponential_scale)
    
    # Scale back to original range
    return scaled * max_signal_abs


def blend_signals(
    signal_map: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    *,
    max_signal_abs: float = 5.0,
) -> pd.DataFrame:
    """Original simple blending (backward compatible).
    
    Args:
        signal_map: Dict[indicator_name] → DataFrame with values
        weights: Dict[indicator_name] → importance weight
        max_signal_abs: Signal clipping bound
    
    Returns:
        Blended signal DataFrame
    """
    if not signal_map:
        raise ValueError("signal_map is empty")

    indices = [df.index for df in signal_map.values()]
    cols = [df.columns for df in signal_map.values()]
    common_index = indices[0]
    common_cols = cols[0]
    for idx in indices[1:]:
        common_index = common_index.intersection(idx)
    for c in cols[1:]:
        common_cols = common_cols.intersection(c)

    if len(common_index) == 0 or len(common_cols) == 0:
        raise ValueError("No common dates or tickers across signals")

    aligned = {
        name: df.reindex(index=common_index, columns=common_cols).fillna(0)
        for name, df in signal_map.items()
    }

    total = pd.DataFrame(0.0, index=common_index, columns=common_cols)
    for name, df in aligned.items():
        w = weights.get(name, 1.0)
        total += df * w

    denom = sum(abs(w) for w in weights.values()) or 1.0
    blended = total / denom

    return blended.clip(lower=-max_signal_abs, upper=max_signal_abs)


def blend_signals_adaptive(
    signal_map: Dict[str, pd.DataFrame],
    importance_weights: Optional[Dict[str, float]] = None,
    exponential_bases: Optional[Dict[str, float]] = None,
    global_exponential_base: Optional[float] = None,
    use_exponential_scaling: bool = False,
    *,
    max_signal_abs: float = 5.0,
) -> pd.DataFrame:
    """Adaptive blending with importance weighting and optional exponential scaling.
    
    This extends blend_signals() with:
    1. Per-indicator importance weights (0.1-3.0 range)
    2. Optional exponential scaling per indicator (configurable base)
    3. Backward compatibility: if no special params, behaves like blend_signals()
    
    Args:
        signal_map: Dict[indicator_name] → DataFrame with signal values
        importance_weights: Dict[indicator_name] → importance multiplier (default 1.0)
        exponential_bases: Dict[indicator_name] → exponential base for that indicator
                          (if None and use_exponential_scaling, uses global_exponential_base)
        global_exponential_base: Default exponential base if not specified per indicator
        use_exponential_scaling: Whether to apply exponential curves
        max_signal_abs: Signal clipping bound
    
    Returns:
        Blended signal DataFrame in [-max_signal_abs, max_signal_abs]
    
    Examples:
        # Simple importance weights (no exponential scaling)
        blend_signals_adaptive(
            signal_map,
            importance_weights={'rsi': 2.0, 'macd': 1.0, 'cmf': 0.5}
        )
        
        # Adaptive with uniform exponential base
        blend_signals_adaptive(
            signal_map,
            importance_weights={'rsi': 2.0, 'macd': 1.0},
            global_exponential_base=2.3,
            use_exponential_scaling=True
        )
        
        # Adaptive with per-indicator bases
        blend_signals_adaptive(
            signal_map,
            importance_weights={'rsi': 2.0, 'macd': 1.0},
            exponential_bases={'rsi': 2.8, 'macd': 1.8},
            use_exponential_scaling=True
        )
    """
    if not signal_map:
        raise ValueError("signal_map is empty")
    
    if importance_weights is None:
        importance_weights = {name: 1.0 for name in signal_map.keys()}
    
    if exponential_bases is None:
        exponential_bases = {}
    
    if global_exponential_base is None:
        global_exponential_base = 2.3
    
    # Align all signals to common index/columns
    indices = [df.index for df in signal_map.values()]
    cols = [df.columns for df in signal_map.values()]
    common_index = indices[0]
    common_cols = cols[0]
    for idx in indices[1:]:
        common_index = common_index.intersection(idx)
    for c in cols[1:]:
        common_cols = common_cols.intersection(c)

    if len(common_index) == 0 or len(common_cols) == 0:
        raise ValueError("No common dates or tickers across signals")

    aligned = {
        name: df.reindex(index=common_index, columns=common_cols).fillna(0)
        for name, df in signal_map.items()
    }

    # Apply transformations and blend
    total = pd.DataFrame(0.0, index=common_index, columns=common_cols)
    
    for name, signal_df in aligned.items():
        # Get importance weight for this indicator
        imp_weight = importance_weights.get(name, 1.0)
        
        # Apply exponential scaling if enabled
        if use_exponential_scaling:
            base = exponential_bases.get(name, global_exponential_base)
            signal_df = _apply_exponential_scaling(signal_df, base=base, max_signal_abs=max_signal_abs)
        
        # Apply importance weight and accumulate
        total += signal_df * imp_weight

    # Normalize by total weight
    denom = sum(abs(w) for w in importance_weights.values()) or 1.0
    blended = total / denom

    return blended.clip(lower=-max_signal_abs, upper=max_signal_abs)
