"""
Enhanced weighted voting strategy with -2 to +2 per-indicator scoring.
Implements 5-core indicators with tiered exits and scenario-based stop loss logic.

Architecture:
- 5 core indicators: SMA, RSI, Volume, ADX, ATR
- Each indicator scores -2 to +2 (not -1/0/+1)
- Total range: -10 to +10
- Buy signal: score >= +5
- Sell signal: score <= -5
- Position sizing: 2-4% dynamic based on signal strength
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class EnhancedWeightedVotingStrategy:
    """
    5-indicator consensus voting with -2 to +2 weighting.
    Implements tiered R/R 1:3 with scenario-based exits.
    """
    
    def __init__(
        self,
        risk_pct: float = 2.0,
        tp1_pct: float = 1.5,
        tp2_pct: float = 3.0,
        min_buy_score: float = 5.0,
        max_sell_score: float = -5.0,
        trailing_stop_pct: float = 1.0,
    ):
        """
        Initialize strategy parameters.
        
        Args:
            risk_pct: Risk per trade (stop loss %)
            tp1_pct: Partial profit target 1 (%)
            tp2_pct: Full profit target 2 (%)
            min_buy_score: Minimum score to enter long
            max_sell_score: Maximum score to enter short
            trailing_stop_pct: Trailing stop distance (%)
        """
        self.risk_pct = risk_pct
        self.tp1_pct = tp1_pct
        self.tp2_pct = tp2_pct
        self.min_buy_score = min_buy_score
        self.max_sell_score = max_sell_score
        self.trailing_stop_pct = trailing_stop_pct
        
        # Position tracking for scenario-based decisions
        self.position_state = {}
        
    def calculate_sma_signal(
        self,
        close: np.ndarray,
        fast_period: int = 20,
        slow_period: int = 50,
    ) -> float:
        """
        SMA crossover signal (-2 to +2 scale).
        
        Scoring:
        - +2: Fast SMA > Slow SMA + distance > 1%
        - +1: Fast SMA > Slow SMA, distance 0.1-1%
        - 0: Flat/crossing
        - -1: Fast SMA < Slow SMA, distance 0.1-1%
        - -2: Fast SMA < Slow SMA + distance > 1%
        """
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
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> float:
        """
        RSI momentum signal (-2 to +2 scale).
        
        Scoring:
        - +2: RSI < 30 (oversold bullish)
        - +1: RSI 30-50 (bullish momentum)
        - 0: RSI 50 (neutral)
        - -1: RSI 50-70 (bearish momentum)
        - -2: RSI > 70 (overbought bearish)
        """
        if len(close) < period + 1:
            return 0.0
        
        delta = np.diff(close[-period-1:])
        gain = np.mean(np.maximum(delta, 0))
        loss = np.mean(np.abs(np.minimum(delta, 0)))
        
        if loss == 0:
            rsi = 100.0 if gain > 0 else 0.0
        else:
            rs = gain / loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        
        if rsi < oversold:
            return 2.0
        elif rsi < 50:
            return 1.0
        elif rsi == 50:
            return 0.0
        elif rsi < overbought:
            return -1.0
        else:
            return -2.0
    
    def calculate_volume_signal(
        self,
        close: np.ndarray,
        volume: np.ndarray,
        period: int = 20,
    ) -> float:
        """
        Volume confirmation signal (-2 to +2 scale).
        
        Scoring based on volume vs its MA:
        - +2: Current volume > MA + 50%
        - +1: Current volume > MA
        - 0: Current volume = MA
        - -1: Current volume < MA
        - -2: Current volume < MA - 30%
        """
        if len(volume) < period + 1:
            return 0.0
        
        vol_ma = np.mean(volume[-period:])
        current_vol = volume[-1]
        
        if vol_ma == 0:
            return 0.0
        
        vol_ratio = current_vol / vol_ma
        
        if vol_ratio > 1.5:
            return 2.0
        elif vol_ratio > 1.0:
            return 1.0
        elif vol_ratio == 1.0:
            return 0.0
        elif vol_ratio > 0.7:
            return -1.0
        else:
            return -2.0
    
    def calculate_adx_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """
        ADX trend strength signal (-2 to +2 scale).
        
        Simplified version - uses directional movement ratio as proxy.
        """
        if len(high) < period + 1:
            return 0.0
        
        try:
            # Convert to float arrays to avoid indexing issues
            high = np.asarray(high, dtype=float)
            low = np.asarray(low, dtype=float)
            close = np.asarray(close, dtype=float)
            
            # Simple trend strength: compare current price to MA
            recent_high = np.mean(high[-period:])
            recent_low = np.mean(low[-period:])
            current_price = close[-1]
            
            # Distance from moving averages
            dist_from_high = current_price - recent_high
            dist_from_low = current_price - recent_low
            
            # Determine trend strength based on position relative to range
            if recent_high > 0:
                bullish_strength = dist_from_high / (recent_high - recent_low + 1e-6)
            else:
                bullish_strength = 0
            
            # Score based on strength
            if bullish_strength > 0.7:
                return 2.0  # Strong uptrend
            elif bullish_strength > 0.4:
                return 1.0  # Moderate uptrend
            elif bullish_strength < -0.7:
                return -2.0  # Strong downtrend
            elif bullish_strength < -0.4:
                return -1.0  # Moderate downtrend
            else:
                return 0.0  # No trend
        except Exception:
            return 0.0
    
    def calculate_atr_signal(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14,
    ) -> float:
        """
        ATR volatility signal (-2 to +2 scale).
        
        Weighted by signal direction. High volatility with clear direction = strong signal.
        Low volatility with mixed signals = weak signal.
        """
        if len(high) < period + 1:
            return 0.0
        
        try:
            # Convert to float arrays
            high = np.asarray(high, dtype=float)
            low = np.asarray(low, dtype=float)
            close = np.asarray(close, dtype=float)
            
            # Calculate True Range
            tr_list = []
            for i in range(len(high)):
                if i == 0:
                    tr_val = float(high[i]) - float(low[i])
                else:
                    tr_val = max(
                        float(high[i]) - float(low[i]),
                        abs(float(high[i]) - float(close[i-1])),
                        abs(float(low[i]) - float(close[i-1]))
                    )
                tr_list.append(tr_val)
            
            atr = np.mean(tr_list[-period:])
            atr_pct = (atr / float(close[-1]) * 100) if float(close[-1]) != 0 else 0
            
            # Volatility scoring with full -2 to +2 range
            if atr_pct > 3.0:
                return 2.0  # Very high volatility
            elif atr_pct > 2.0:
                return 1.5
            elif atr_pct > 1.5:
                return 1.0
            elif atr_pct > 1.0:
                return 0.5
            elif atr_pct > 0.7:
                return 0.0
            elif atr_pct > 0.4:
                return -0.5
            elif atr_pct > 0.2:
                return -1.0
            else:
                return -2.0  # Very low volatility
        except Exception:
            return 0.0
    
    def calculate_voting_score(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> float:
        """
        Calculate total voting score from all 5 indicators.
        
        Returns:
            float: Score in range [-10, +10]
        """
        if len(close) < 50:  # Minimum history for indicators
            return 0.0
        
        sma_signal = self.calculate_sma_signal(close)
        rsi_signal = self.calculate_rsi_signal(close)
        volume_signal = self.calculate_volume_signal(close, volume)
        adx_signal = self.calculate_adx_signal(high, low, close)
        atr_signal = self.calculate_atr_signal(high, low, close)
        
        # Total score: each indicator -2 to +2, total -10 to +10
        total_score = (
            sma_signal + 
            rsi_signal + 
            volume_signal + 
            adx_signal + 
            atr_signal
        )
        
        return np.clip(total_score, -10, 10)
    
    def should_enter(self, score: float) -> bool:
        """Check if score triggers buy signal."""
        return score >= self.min_buy_score
    
    def should_exit_on_signal(self, score: float) -> bool:
        """Check if score triggers sell signal."""
        return score <= self.max_sell_score
    
    def calculate_position_size(self, score: float, account_equity: float) -> float:
        """
        Calculate position size based on signal strength.
        
        Aggressive sizing (4-8% instead of 2-4%):
        - score +2 to +3: 4% risk
        - score +3 to +3.5: 6% risk
        - score +3.5 to +4: 8% risk (max conviction)
        
        Returns:
            float: Position size in dollars
        """
        if score < self.min_buy_score:
            return 0.0
        
        # Increased risk allocation
        if score < 3.0:
            risk_pct = 4.0
        elif score < 3.5:
            risk_pct = 6.0
        else:
            risk_pct = 8.0
        
        return account_equity * risk_pct / 100
    
    def calculate_entry_exit_levels(
        self,
        entry_price: float,
    ) -> Dict[str, float]:
        """
        Calculate R/R 1:3 with tiered profit targets.
        
        Returns:
            Dict with keys: sl, tp1, tp2
        """
        sl = entry_price * (1 - self.risk_pct / 100)
        tp1 = entry_price * (1 + self.tp1_pct / 100)
        tp2 = entry_price * (1 + self.tp2_pct / 100)
        
        return {
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2,
            'risk': entry_price - sl,
            'profit_tp1': tp1 - entry_price,
            'profit_tp2': tp2 - entry_price,
        }
    
    def update_position_at_tp1(
        self,
        symbol: str,
        current_price: float,
        tp1_price: float,
        current_score: float,
    ) -> Dict[str, any]:
        """
        Scenario decision at TP1: evaluate signal strength to determine exit approach.
        
        Scenario A (Signals Weakening):
        - Exit 50% at TP1 immediately
        - SL remaining 50% at TP1 * 0.99 (protect gains)
        
        Scenario B (Signals Strong):
        - Keep all position
        - Use trailing SL at 1% below TP1
        - Allow to run to TP2
        
        Returns:
            Dict with scenario, exit_qty, new_sl, etc.
        """
        entry_price = self.position_state.get(symbol, {}).get('entry_price', tp1_price)
        initial_quantity = self.position_state.get(symbol, {}).get('quantity', 1)
        
        # Signal strength assessment
        is_strong_signal = current_score >= 7  # +7 to +10 is "strong"
        
        if is_strong_signal:
            # Scenario B: Keep all, trailing stop at TP1 * 0.99
            trailing_sl = tp1_price * (1 - self.trailing_stop_pct / 100)
            return {
                'scenario': 'B_STRONG',
                'exit_qty_tp1': 0,  # Keep all
                'new_sl': trailing_sl,
                'action': 'TRAIL_SL',
                'tp_target': entry_price * (1 + self.tp2_pct / 100),  # TP2
            }
        else:
            # Scenario A: Exit 50%, protect 50% with reduced stop
            half_quantity = initial_quantity * 0.5
            protective_sl = entry_price * (1 + self.tp1_pct * 0.5 / 100)  # TP1 * 0.5
            
            return {
                'scenario': 'A_WEAK',
                'exit_qty_tp1': half_quantity,
                'exit_price_tp1': tp1_price,
                'new_sl': protective_sl,
                'action': 'EXIT_HALF',
                'remaining_qty': half_quantity,
            }
    
    def get_system_summary(self) -> Dict:
        """Return system configuration summary."""
        return {
            'name': 'Enhanced Weighted Voting (5-Indicator)',
            'indicators': ['SMA', 'RSI', 'Volume', 'ADX', 'ATR'],
            'score_range': [-10, 10],
            'buy_threshold': self.min_buy_score,
            'sell_threshold': self.max_sell_score,
            'risk_per_trade': f"{self.risk_pct}%",
            'tp1_reward': f"{self.tp1_pct}%",
            'tp2_reward': f"{self.tp2_pct}%",
            'trailing_stop': f"{self.trailing_stop_pct}%",
            'position_scaling': '2-4% based on signal strength',
            'scenarios': {
                'A': 'Weak signals at TP1: Exit 50%, SL protective',
                'B': 'Strong signals at TP1: Trail to TP2'
            }
        }
