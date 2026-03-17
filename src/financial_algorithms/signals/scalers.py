"""Adaptive exponential scalers for all 15 indicators with importance weighting.

Symmetric exponential curves normalize raw indicator values to [-1, 1] range
with configurable aggressiveness (base parameter).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Union, Literal


def _symmetric_exponential(
    normalized_value: float,
    base: float = 2.3,
) -> float:
    """Apply symmetric exponential scaling to normalized [-1, 1] input.
    
    Maps [-1, 1] → [-1, 1] with exponential curve controlled by base.
    Example: base=2.3, normalized=0.5 → ~0.47 (aggressive for strong signals)
    
    Args:
        normalized_value: Input in range [-1, 1]
        base: Exponential base (typically 1.5-3.0); higher = more aggressive
    
    Returns:
        Scaled value in [-1, 1]
    """
    if normalized_value == 0:
        return 0.0
    
    # Apply exponential to absolute value, preserve sign
    abs_val = abs(normalized_value)
    exp_val = (base ** abs_val - 1) / (base - 1)
    return np.sign(normalized_value) * exp_val


def scale_rsi_exponential(
    rsi_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Scale RSI [0, 100] using symmetric exponential curve.
    
    Center: 50 → 0
    Strong Buy: 20 → varies by base
    Strong Sell: 80 → varies by base
    """
    if isinstance(rsi_value, pd.Series):
        normalized = (rsi_value - 50) / 50
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (rsi_value - 50) / 50
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_macd_exponential(
    macd_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
    scale_factor: float = 50.0,
) -> Union[float, pd.Series]:
    """Scale MACD values using symmetric exponential curve.
    
    MACD typically ranges [-50, 50]; scale_factor normalizes to sensible input.
    """
    if isinstance(macd_value, pd.Series):
        normalized = (macd_value / scale_factor).clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (macd_value / scale_factor) if scale_factor else 0
        normalized = max(-1, min(1, normalized))
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_adx_exponential(
    adx_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Scale ADX [0, 100] using symmetric exponential curve.
    
    Center: 50 → 0
    Strong trend: 80 → varies by base
    Weak trend: 20 → varies by base
    """
    if isinstance(adx_value, pd.Series):
        normalized = (adx_value - 50) / 50
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (adx_value - 50) / 50
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_williams_r_exponential(
    williams_r_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Scale Williams %R [-100, 0] using symmetric exponential curve.
    
    Center: -50 → 0
    Extreme oversold: -90 → varies by base
    Extreme overbought: -10 → varies by base
    """
    if isinstance(williams_r_value, pd.Series):
        normalized = (williams_r_value + 50) / 50
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (williams_r_value + 50) / 50
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_cmf_exponential(
    cmf_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Scale Chaikin Money Flow [-1, 1] using symmetric exponential curve.
    
    Already normalized, just apply exponential weighting.
    """
    if isinstance(cmf_value, pd.Series):
        normalized = cmf_value.clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = max(-1, min(1, cmf_value))
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_volume_osc_exponential(
    volume_osc_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
    scale_factor: float = 100.0,
) -> Union[float, pd.Series]:
    """Scale Volume Oscillator using symmetric exponential curve.
    
    Typically ranges [-100, 100] (as percentage).
    """
    if isinstance(volume_osc_value, pd.Series):
        normalized = (volume_osc_value / scale_factor).clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (volume_osc_value / scale_factor) if scale_factor else 0
        normalized = max(-1, min(1, normalized))
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_cci_exponential(
    cci_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
    scale_factor: float = 200.0,
) -> Union[float, pd.Series]:
    """Scale CCI using symmetric exponential curve.
    
    CCI ranges typically [-200, 200]; scale_factor normalizes.
    """
    if isinstance(cci_value, pd.Series):
        normalized = (cci_value / scale_factor).clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (cci_value / scale_factor) if scale_factor else 0
        normalized = max(-1, min(1, normalized))
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_stochastic_exponential(
    stoch_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Scale Stochastic Oscillator [0, 100] using symmetric exponential curve.
    
    Center: 50 → 0
    Overbought: 80 → varies by base
    Oversold: 20 → varies by base
    """
    if isinstance(stoch_value, pd.Series):
        normalized = (stoch_value - 50) / 50
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (stoch_value - 50) / 50
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_atr_exponential(
    atr_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
    scale_factor: float = 2.0,
) -> Union[float, pd.Series]:
    """Scale Average True Range using symmetric exponential curve.
    
    ATR is volatility measure; normalized relative to typical range.
    """
    if isinstance(atr_value, pd.Series):
        normalized = (atr_value / scale_factor).clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (atr_value / scale_factor) if scale_factor else 0
        normalized = max(-1, min(1, normalized))
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_force_index_exponential(
    force_index_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
    scale_factor: float = 1e6,
) -> Union[float, pd.Series]:
    """Scale Force Index using symmetric exponential curve.
    
    Force Index can have wide range; normalize relative to scale_factor.
    """
    if isinstance(force_index_value, pd.Series):
        normalized = (force_index_value / scale_factor).clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (force_index_value / scale_factor) if scale_factor else 0
        normalized = max(-1, min(1, normalized))
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_bollinger_bands_exponential(
    bb_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Scale Bollinger Bands position [0, 1] using symmetric exponential curve.
    
    Position 0 = at lower band, 0.5 = middle (neutral), 1 = at upper band.
    Maps to [-1, 1] centered at 0.5.
    """
    if isinstance(bb_value, pd.Series):
        normalized = (bb_value - 0.5) * 2  # Scale [0, 1] → [-1, 1]
        normalized = normalized.clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = (bb_value - 0.5) * 2
        normalized = max(-1, min(1, normalized))
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_moving_average_cross_exponential(
    ma_cross_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Scale Moving Average Crossover signals.
    
    Typically outputs discrete signals; apply exponential for continuity.
    """
    if isinstance(ma_cross_value, pd.Series):
        normalized = ma_cross_value.clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = max(-1, min(1, ma_cross_value))
        return importance_weight * _symmetric_exponential(normalized, base)


def scale_sar_stochastic_exponential(
    sar_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Scale SAR/Stochastic combined signal using symmetric exponential curve.
    
    Maps signal range to [-1, 1] exponentially.
    """
    if isinstance(sar_value, pd.Series):
        normalized = sar_value.clip(-1, 1)
        return importance_weight * normalized.apply(lambda x: _symmetric_exponential(x, base))
    else:
        normalized = max(-1, min(1, sar_value))
        return importance_weight * _symmetric_exponential(normalized, base)


# Indicator type mapping to scaler function
SCALERS = {
    'rsi': scale_rsi_exponential,
    'macd': scale_macd_exponential,
    'adx': scale_adx_exponential,
    'williams_r': scale_williams_r_exponential,
    'cmf': scale_cmf_exponential,
    'volume_osc': scale_volume_osc_exponential,
    'cci_adx': scale_cci_exponential,  # CCI is primary component
    'stoch': scale_stochastic_exponential,
    'atr_trend': scale_atr_exponential,
    'force_index': scale_force_index_exponential,
    'bb_rsi': scale_bollinger_bands_exponential,  # Bollinger Bands is primary
    'rsi_obv_bb': scale_bollinger_bands_exponential,  # BB is primary here too
    'ma_cross': scale_moving_average_cross_exponential,
    'sar_stoch': scale_sar_stochastic_exponential,
    'stoch_macd': scale_stochastic_exponential,  # Stoch is primary for this blend
}


def scale_signal_adaptive(
    indicator_type: str,
    raw_value: Union[float, pd.Series],
    base: float = 2.3,
    importance_weight: float = 1.0,
) -> Union[float, pd.Series]:
    """Apply adaptive exponential scaling to indicator values.
    
    Master function that routes to appropriate scaler based on indicator type.
    
    Args:
        indicator_type: Indicator name (key in SCALERS dict)
        raw_value: Raw indicator value or Series
        base: Exponential curve base (1.5-3.0 range typical)
        importance_weight: Indicator importance multiplier
    
    Returns:
        Scaled value in [-1, 1] range (before importance weighting)
    
    Raises:
        ValueError: If indicator_type not recognized
    """
    if indicator_type not in SCALERS:
        raise ValueError(
            f"Unknown indicator type: {indicator_type}. "
            f"Supported: {list(SCALERS.keys())}"
        )
    
    scaler = SCALERS[indicator_type]
    return scaler(raw_value, base=base, importance_weight=importance_weight)


if __name__ == "__main__":
    # Quick selftest of scaler functions
    print("Testing RSI scaler:")
    print(f"  RSI=30 → {scale_rsi_exponential(30):.3f}")
    print(f"  RSI=50 → {scale_rsi_exponential(50):.3f}")
    print(f"  RSI=70 → {scale_rsi_exponential(70):.3f}")
    print(f"  RSI=80 → {scale_rsi_exponential(80):.3f}")
    
    print("\nTesting MACD scaler:")
    print(f"  MACD=-10 → {scale_macd_exponential(-10):.3f}")
    print(f"  MACD=0 → {scale_macd_exponential(0):.3f}")
    print(f"  MACD=10 → {scale_macd_exponential(10):.3f}")
    
    print("\nTesting with importance weight:")
    print(f"  RSI=80, weight=2.0 → {scale_rsi_exponential(80, importance_weight=2.0):.3f}")
    print(f"  RSI=80, weight=0.5 → {scale_rsi_exponential(80, importance_weight=0.5):.3f}")
    
    print("\nTesting with different bases:")
    print(f"  RSI=80, base=1.5 → {scale_rsi_exponential(80, base=1.5):.3f}")
    print(f"  RSI=80, base=2.0 → {scale_rsi_exponential(80, base=2.0):.3f}")
    print(f"  RSI=80, base=2.3 → {scale_rsi_exponential(80, base=2.3):.3f}")
    print(f"  RSI=80, base=3.0 → {scale_rsi_exponential(80, base=3.0):.3f}")
