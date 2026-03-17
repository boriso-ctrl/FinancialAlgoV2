"""Dynamic position sizing for risk management and volatility adaptation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def kelly_criterion(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    kelly_fraction: float = 0.25,
) -> float:
    """Calculate position size using Kelly Criterion.
    
    Formula: f* = (bp - q) / b
      where b = avg_win / avg_loss (odds)
            p = win_rate
            q = 1 - win_rate
            f* = Kelly fraction
    
    Args:
        win_rate: [0, 1] Probability of profitable trade
        avg_win: Average profit per winning trade
        avg_loss: Average loss per losing trade (absolute)
        kelly_fraction: Conservative fraction of Kelly (0.1-0.5 typical)
    
    Returns:
        Position size fraction [0, 1]
    """
    if avg_loss <= 0 or avg_win <= 0:
        return kelly_fraction * 0.1
    
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate
    
    kelly_full = (b * p - q) / b
    kelly_safe = kelly_full * kelly_fraction
    
    # Clamp to [0, 1]
    return max(0, min(1, kelly_safe))


def volatility_adjusted_size(
    base_size: float,
    current_volatility: float,
    target_volatility: float = 0.03,
) -> float:
    """Adjust position size inverse to volatility.
    
    High vol = smaller positions
    Low vol = larger positions
    
    Args:
        base_size: Base position size (e.g., 0.5 or 50%)
        current_volatility: ATR% or similar volatility metric
        target_volatility: Reference volatility (3% default)
    
    Returns:
        Adjusted position size
    """
    if current_volatility <= 0:
        return base_size
    
    vol_ratio = target_volatility / current_volatility
    adjusted = base_size * vol_ratio
    
    # Clamp to 10% - 100% of base
    return max(base_size * 0.1, min(base_size, adjusted))


def risk_parity_size(
    prices: pd.Series,
    target_risk_pct: float = 0.02,
    stop_loss_pips: Optional[float] = None,
) -> float:
    """Calculate position size to limit risk to target % of portfolio.
    
    Args:
        prices: Current price series
        target_risk_pct: Max loss as % of portfolio (e.g., 0.02 = 2%)
        stop_loss_pips: Fixed stop loss in price points
    
    Returns:
        Position size as fraction of account
    """
    current_price = prices.iloc[-1]
    
    if stop_loss_pips is None:
        # Use ATR as dynamic stop
        atr = prices.diff().abs().rolling(14).mean().iloc[-1]
        stop_loss_pips = atr
    
    if stop_loss_pips <= 0 or current_price <= 0:
        return target_risk_pct
    
    risk_per_share = stop_loss_pips
    position_size = target_risk_pct * current_price / risk_per_share
    
    return min(1.0, position_size)


def adaptive_size_ensemble(
    base_size: float = 0.5,
    win_rate: Optional[float] = None,
    avg_win: Optional[float] = None,
    avg_loss: Optional[float] = None,
    volatility: Optional[float] = None,
    target_volatility: float = 0.03,
    confidence: Optional[float] = None,
    max_size: float = 1.0,
    min_size: float = 0.05,
) -> float:
    """Ensemble multiple position sizing methods.
    
    Combines Kelly, volatility adjustment, and confidence weighting.
    
    Args:
        base_size: Starting position size
        win_rate, avg_win, avg_loss: Kelly inputs
        volatility: Current ATR%
        target_volatility: Reference vol for adjustment
        confidence: Signal confidence [0, 1]
        max_size: Maximum allowed
        min_size: Minimum allowed
    
    Returns:
        Final position size
    """
    size = base_size
    
    # Method 1: Kelly Criterion
    if win_rate is not None and avg_win is not None and avg_loss is not None:
        kelly_size = kelly_criterion(win_rate, avg_win, avg_loss, kelly_fraction=0.25)
        size = size * (1 + kelly_size) / 2  # Blend with base
    
    # Method 2: Volatility adjustment
    if volatility is not None:
        vol_factor = volatility_adjusted_size(1.0, volatility, target_volatility)
        size = size * vol_factor
    
    # Method 3: Confidence weighting
    if confidence is not None:
        confidence = max(0, min(1, confidence))
        size = size * confidence
    
    # Apply bounds
    return max(min_size, min(max_size, size))


class PositionSizer:
    """Stateful position sizing manager with historical tracking."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trades = []
        self.win_rate = 0.5
        self.avg_win = 0.0
        self.avg_loss = 0.0
    
    def record_trade(self, entry_price: float, exit_price: float, size: float):
        """Record a closed trade for win rate calculation."""
        pnl = (exit_price - entry_price) * size
        self.trades.append({
            'entry': entry_price,
            'exit': exit_price,
            'size': size,
            'pnl': pnl,
            'win': pnl > 0,
        })
        
        self._update_stats()
    
    def _update_stats(self):
        """Recalculate win rate and average win/loss."""
        if not self.trades:
            return
        
        df = pd.DataFrame(self.trades)
        wins = df[df['win']]
        losses = df[~df['win']]
        
        if len(wins) > 0:
            self.win_rate = len(wins) / len(df)
            self.avg_win = wins['pnl'].mean()
        else:
            self.avg_win = 0
        
        if len(losses) > 0:
            self.avg_loss = abs(losses['pnl'].mean())
        else:
            self.avg_loss = 0
    
    def calculate_size(
        self,
        current_price: float,
        volatility: float,
        confidence: float = 0.5,
        target_risk_pct: float = 0.02,
    ) -> float:
        """Calculate position size using all available info."""
        return adaptive_size_ensemble(
            base_size=0.5,
            win_rate=self.win_rate,
            avg_win=max(0.1, self.avg_win),
            avg_loss=max(0.1, self.avg_loss),
            volatility=volatility,
            target_volatility=0.03,
            confidence=confidence,
            max_size=1.0,
            min_size=0.05,
        )


if __name__ == "__main__":
    print("Testing position sizing...")
    
    # Test Kelly Criterion
    kelly_size = kelly_criterion(win_rate=0.55, avg_win=100, avg_loss=90)
    print(f"Kelly size (55% win, 100/90 ratio): {kelly_size:.3f}")
    
    # Test volatility adjustment
    vol_adj = volatility_adjusted_size(0.5, current_volatility=0.05, target_volatility=0.03)
    print(f"Vol-adjusted size (5% vol, 3% target): {vol_adj:.3f}")
    
    # Test adaptive ensemble
    adaptive = adaptive_size_ensemble(
        base_size=0.5,
        win_rate=0.6,
        avg_win=100,
        avg_loss=80,
        volatility=0.02,
        confidence=0.8,
    )
    print(f"Adaptive ensemble (high confidence, low vol): {adaptive:.3f}")
    
    # Test PositionSizer
    sizer = PositionSizer()
    sizer.record_trade(100, 105, 0.5)  # Win
    sizer.record_trade(105, 100, 0.5)  # Loss
    sizer.record_trade(105, 108, 0.5)  # Win
    
    print(f"\nPositionSizer stats:")
    print(f"  Win rate: {sizer.win_rate:.1%}")
    print(f"  Avg win: ${sizer.avg_win:.2f}")
    print(f"  Avg loss: ${sizer.avg_loss:.2f}")
    
    size = sizer.calculate_size(current_price=108, volatility=0.03, confidence=0.7)
    print(f"  Calculated size: {size:.3f}")
