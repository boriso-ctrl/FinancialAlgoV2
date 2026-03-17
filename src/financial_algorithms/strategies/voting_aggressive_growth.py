"""
Aggressive Growth Voting Strategy with Stacked Entries
Mode: Maximum position sizing (10-15%), extended exits (5-7% TP2), multiple concurrent positions

Architecture:
- 5 core indicators: SMA, RSI, Volume, ADX, ATR
- Each indicator scores -2 to +2
- Total range: -10 to +10
- Entry: score >= +2.0 (allows multiple stacked entries)
- Exit: score <= -2.0 (exits ALL stacked positions)
- Position sizing: 10-15% dynamic based on signal strength (aggressive)
- Profit targets: TP1 at 2.5% (exits 50%), TP2 at 6.0% (exits remaining)
- Stacked positions: Up to 3 concurrent positions per strong signal
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AggressiveGrowthVotingStrategy:
    """
    Aggressive growth version with stacked entries and extended exits.
    Targets 15-20%+ annual returns by maximizing position sizing and holding winners.
    """
    
    def __init__(
        self,
        risk_pct: float = 2.0,
        tp1_pct: float = 2.5,
        tp2_pct: float = 6.0,
        min_buy_score: float = 2.0,
        max_sell_score: float = -2.0,
        trailing_stop_pct: float = 1.5,
        max_stacked_positions: int = 3,
    ):
        """
        Initialize aggressive strategy parameters.
        
        Args:
            risk_pct: Risk per trade (stop loss %)
            tp1_pct: First profit target (%) - exits 50% of position
            tp2_pct: Second profit target (%) - exits remaining
            min_buy_score: Minimum score to enter long
            max_sell_score: Maximum score to exit all positions
            trailing_stop_pct: Trailing stop distance (%)
            max_stacked_positions: Max concurrent positions per asset
        """
        self.risk_pct = risk_pct
        self.tp1_pct = tp1_pct
        self.tp2_pct = tp2_pct
        self.min_buy_score = min_buy_score
        self.max_sell_score = max_sell_score
        self.trailing_stop_pct = trailing_stop_pct
        self.max_stacked_positions = max_stacked_positions
    
    def calculate_sma_signal(
        self,
        close: np.ndarray,
        fast_period: int = 20,
        slow_period: int = 50,
    ) -> float:
        """SMA crossover signal (-2 to +2 scale)."""
        if len(close) < slow_period + 1:
            return 0.0
        
        fast_sma = np.mean(close[-fast_period:])
        slow_sma = np.mean(close[-slow_period:])
        
        if fast_sma == slow_sma:
            return 0.0
        
        distance_pct = abs(fast_sma - slow_sma) / slow_sma * 100
        
        if fast_sma > slow_sma:
            if distance_pct > 1.0:
                return 2.0
            else:
                return 1.0
        else:
            if distance_pct > 1.0:
                return -2.0
            else:
                return -1.0
    
    def calculate_rsi_signal(
        self,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """RSI momentum signal (-2 to +2 scale)."""
        if len(close) < period + 1:
            return 0.0
        
        deltas = np.diff(close[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        if avg_loss == 0:
            return 2.0 if avg_gain > 0 else 0.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        if rsi >= 70:
            return 2.0
        elif rsi >= 55:
            return 1.0
        elif rsi <= 30:
            return -2.0
        elif rsi <= 45:
            return -1.0
        else:
            return 0.0
    
    def calculate_volume_signal(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 20,
    ) -> float:
        """Volume confirmation signal (-2 to +2 scale)."""
        if len(volume) < period + 1 or len(close) < period + 1:
            return 0.0
        
        recent_volume = np.mean(volume[-period:])
        historical_volume = np.mean(volume[-period*2:-period])
        
        if historical_volume == 0:
            return 0.0
        
        volume_ratio = recent_volume / historical_volume
        price_trend = 1.0 if close[-1] > close[-5] else -1.0
        
        if volume_ratio > 1.5:
            return 2.0 * price_trend
        elif volume_ratio > 1.2:
            return 1.0 * price_trend
        elif volume_ratio < 0.7:
            return -1.0 * price_trend
        else:
            return 0.0
    
    def calculate_adx_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """ADX trend strength signal (-2 to +2 scale)."""
        if len(high) < period + 1:
            return 0.0
        
        # Calculate ADX (simplified)
        plus_dm = high[-1] - high[-2]
        minus_dm = low[-2] - low[-1]
        tr = max(high[-1] - low[-1], abs(high[-1] - close[-2]), abs(low[-1] - close[-2]))
        
        if tr == 0:
            return 0.0
        
        di_diff = plus_dm - minus_dm
        
        # Estimate trend direction
        if di_diff > 0:
            return 2.0 if di_diff > 2*tr else 1.0
        else:
            return -2.0 if di_diff < -2*tr else -1.0
    
    def calculate_atr_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """ATR volatility confirmation signal (-2 to +2 scale)."""
        if len(high) < period + 1:
            return 0.0
        
        tr_values = []
        for i in range(len(high)-period, len(high)):
            tr = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]) if i > 0 else high[i] - low[i],
                abs(low[i] - close[i-1]) if i > 0 else high[i] - low[i],
            )
            tr_values.append(tr)
        
        atr = np.mean(tr_values)
        current_price = close[-1]
        
        if current_price == 0:
            return 0.0
        
        atr_pct = (atr / current_price) * 100
        
        # Volatility supports strong moves
        if atr_pct > 2.0:
            return 2.0
        elif atr_pct > 1.0:
            return 1.0
        elif atr_pct < 0.5:
            return -1.0
        else:
            return 0.0
    
    def calculate_voting_score(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> float:
        """
        Calculate 5-indicator voting score.
        
        Range: -10 to +10
        - Each indicator: -2 to +2
        - Higher = stronger buy signal
        - Lower = stronger sell signal
        
        Returns:
            float: Total voting score
        """
        sma_sig = self.calculate_sma_signal(close)
        rsi_sig = self.calculate_rsi_signal(close)
        vol_sig = self.calculate_volume_signal(close, volume)
        adx_sig = self.calculate_adx_signal(high, low, close)
        atr_sig = self.calculate_atr_signal(high, low, close)
        
        total_score = sma_sig + rsi_sig + vol_sig + adx_sig + atr_sig
        return np.clip(total_score, -10, 10)
    
    def should_enter(self, score: float) -> bool:
        """Check if score triggers aggressive buy signal."""
        return score >= self.min_buy_score
    
    def should_exit_all(self, score: float) -> bool:
        """Check if score triggers emergency exit on all positions."""
        return score <= self.max_sell_score
    
    def calculate_position_size(self, score: float, account_equity: float) -> float:
        """
        Calculate aggressive position size based on signal strength.
        
        Aggressive sizing (10-15% instead of 4-8%):
        - score +2 to +3: 10% risk
        - score +3 to +3.5: 12% risk
        - score +3.5+: 15% risk (max conviction)
        
        Returns:
            float: Position size in dollars
        """
        if score < self.min_buy_score:
            return 0.0
        
        # Aggressive risk allocation - increased from 4-8%
        if score < 3.0:
            risk_pct = 10.0
        elif score < 3.5:
            risk_pct = 12.0
        else:
            risk_pct = 15.0
        
        return account_equity * risk_pct / 100
