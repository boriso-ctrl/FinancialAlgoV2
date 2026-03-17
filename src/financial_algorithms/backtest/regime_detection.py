"""Regime detection filters for adaptive trading restricted to profitable market conditions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional


def detect_market_regime(
    close_prices: pd.Series,
    rsi_period: int = 14,
    rsi_oversold: float = 30,
    rsi_overbought: float = 70,
    volume: Optional[pd.Series] = None,
    volume_ma_period: int = 20,
) -> pd.Series:
    """Detect market regime and filter for tradeable conditions.
    
    Returns:
        Binary signal: 1 = trade allowed, 0 = skip this bar
    
    Tradeable conditions:
    - RSI not in extreme (not <30 or >70) = avoids whipsaw
    - Volume above average = liquidity
    """
    regime = pd.Series(1, index=close_prices.index)
    
    # Calculate RSI
    delta = close_prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI extremes filter: skip when oversold or overbought
    extreme_rsi = (rsi < rsi_oversold) | (rsi > rsi_overbought)
    regime = regime.where(~extreme_rsi, 0)
    
    # Volume filter
    if volume is not None:
        vol_ma = volume.rolling(volume_ma_period).mean()
        low_volume = volume < vol_ma
        regime = regime.where(~low_volume, 0)
    
    return regime


def detect_trend_direction(
    close_prices: pd.Series,
    fast_ma_period: int = 10,
    slow_ma_period: int = 20,
) -> Tuple[pd.Series, pd.Series]:
    """Detect trend using moving average crossover.
    
    Returns:
        Tuple of (trend: 1=uptrend, -1=downtrend, 0=neutral, regime: 1=tradeable, 0=skip)
    """
    fast_ma = close_prices.ewm(span=fast_ma_period, adjust=False).mean()
    slow_ma = close_prices.ewm(span=slow_ma_period, adjust=False).mean()
    
    trend = pd.Series(0, index=close_prices.index)
    trend = trend.where(fast_ma <= slow_ma, 1)  # Uptrend: fast > slow
    trend = trend.where(fast_ma >= slow_ma, -1)  # Downtrend: fast < slow
    
    # Regime: trade only in clear trends, not neutral
    regime = (trend != 0).astype(int)
    
    return trend, regime


def detect_volatility_regime(
    close_prices: pd.Series,
    atr_period: int = 14,
    vol_low_threshold: float = 0.01,
    vol_high_threshold: float = 0.05,
) -> Tuple[pd.Series, pd.Series]:
    """Detect volatility level and adapt position sizing.
    
    Returns:
        Tuple of (volatility: float (0-1 scale), regime: 1=normal, 0=avoid)
    
    Strategy:
    - Normal volatility (1-5% ATR): trade full size
    - Low volatility (<1% ATR): trade reduced size (less signal)
    - High volatility (>5% ATR): trade reduced size (more risk)
    """
    high_low = close_prices.rolling(atr_period).max() - close_prices.rolling(atr_period).min()
    tr = high_low  # True Range approximation
    atr = tr.rolling(atr_period).mean()
    volatility_pct = atr / close_prices
    
    # Normalize to 0-1 scale where 0.03 (3%) = 0.5
    normalized_vol = volatility_pct / 0.03
    normalized_vol = normalized_vol.clip(0, 1)
    
    # Regime: prefer normal volatility, avoid extremes
    regime = ((volatility_pct >= vol_low_threshold) & (volatility_pct <= vol_high_threshold)).astype(int)
    
    return normalized_vol, regime


def combine_regime_filters(
    filters: dict[str, pd.Series],
    require_all: bool = False,
) -> pd.Series:
    """Combine multiple regime filters into single trade signal.
    
    Args:
        filters: Dict of filter_name → binary signal (1=OK, 0=skip)
        require_all: If True, need ALL filters green (AND). If False, need ANY (OR).
    
    Returns:
        Combined regime signal
    """
    if not filters:
        return pd.Series(1, index=list(filters.values())[0].index)
    
    combined = list(filters.values())[0].copy()
    
    if require_all:
        for filt in list(filters.values())[1:]:
            combined = combined & filt
    else:
        for filt in list(filters.values())[1:]:
            combined = combined | filt
    
    return combined.astype(int)


if __name__ == "__main__":
    # Quick test with simulated data
    print("Testing regime detection...")
    
    np.random.seed(42)
    dates = pd.date_range('2026-01-01', periods=1000, freq='1min')
    close = pd.Series(
        np.cumsum(np.random.normal(0, 0.1, 1000)) + 100,
        index=dates
    )
    volume = pd.Series(
        np.random.uniform(1000, 5000, 1000),
        index=dates
    )
    
    # Detect regimes
    market_regime = detect_market_regime(close, volume=volume)
    trend, trend_regime = detect_trend_direction(close)
    volatility, vol_regime = detect_volatility_regime(close)
    
    print(f"Market regime allowed: {market_regime.sum() / len(market_regime) * 100:.1f}% of bars")
    print(f"Trend regime allowed: {trend_regime.sum() / len(trend_regime) * 100:.1f}% of bars")
    print(f"Vol regime allowed: {vol_regime.sum() / len(vol_regime) * 100:.1f}% of bars")
    
    combined = combine_regime_filters({
        'market': market_regime,
        'trend': trend_regime,
        'volatility': vol_regime,
    }, require_all=False)
    
    print(f"Combined (ANY): {combined.sum() / len(combined) * 100:.1f}% of bars")
    
    print(f"\nSample regime data:")
    print(pd.DataFrame({
        'close': close.head(10),
        'market_regime': market_regime.head(10),
        'trend': trend.head(10),
        'volatility_pct': volatility.head(10),
    }))
