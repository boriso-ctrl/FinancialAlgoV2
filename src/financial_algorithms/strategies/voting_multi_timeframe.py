"""
Multi-Timeframe Confluence Strategy
Combines 15-minute entry signals with 1-hour confirmation for 2-5 daily trades.

Architecture:
- 15M: Entry signal (reversal + bullish momentum)
- 1H: Confirmation (divergence, trend alignment, or momentum confirmation)
- Position sizing: 1-2% per trade (smaller, more frequent)
- Strict confluence: ALL conditions must align for entry

Signal Generation:
1. 15M Reversal: Price bouncing off support with increasing volume
2. 15M Momentum: RSI oversold (<30) or momentum turning positive
3. 1H Divergence: Price lower low but RSI higher low (bullish divergence)
4. 1H Trend: SMA alignment  favoring upside OR ATR expanding with clear direction
5. Exit: First target at +1%, trail to +2.5% (tighter for daily frequency)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MultiTimeframeSignal:
    """Container for multi-timeframe signal components."""
    
    def __init__(self):
        self.m15_reversal = False  # Price reversal on 15M
        self.m15_momentum = False  # Momentum confirmation on 15M
        self.h1_divergence = False  # Divergence on 1H
        self.h1_trend = False  # Trend confirmation on 1H
        self.h1_momentum = False  # Momentum alignment on 1H
        
        self.confluence_score = 0.0  # 0-5 (# of aligned conditions)
        self.signal_strength = 0.0  # Composite strength
        self.entry_price = None
        self.timestamp = None


class MultiTimeframeVotingStrategy:
    """
    Multi-timeframe strategy combining 15M/1H analysis.
    Generates 2-5 trades per day with strict confluence requirements.
    """
    
    def __init__(
        self,
        risk_pct: float = 1.5,  # Lower risk per trade
        tp1_pct: float = 1.0,  # Quick first target
        tp2_pct: float = 2.5,  # Trail to second target
        min_confluence: int = 4,  # Require 4/5 conditions aligned (hard confluence)
    ):
        """
        Initialize multi-timeframe strategy.
        
        Args:
            risk_pct: Risk per trade (%)
            tp1_pct: First profit target (%)
            tp2_pct: Second profit target (%)
            min_confluence: Minimum aligned conditions to enter (4-5)
        """
        self.risk_pct = risk_pct
        self.tp1_pct = tp1_pct
        self.tp2_pct = tp2_pct
        self.min_confluence = min_confluence
        
        # Divergence tracking for 1H
        self.rsi_history_1h = []
        self.price_history_1h = []
        self.last_rsi_low_1h = None
        self.last_price_low_1h = None
        
        logger.info(
            f"Initialized multi-timeframe strategy: "
            f"Risk={risk_pct}%, TP1={tp1_pct}%, TP2={tp2_pct}%, "
            f"Min confluence={min_confluence}/5"
        )
    
    # ==================== 15M SIGNALS ====================
    
    def detect_m15_reversal(
        self,
        close_15m: np.ndarray,
        low_15m: np.ndarray,
        volume_15m: np.ndarray,
    ) -> bool:
        """
        Detect reversal on 15M: price bouncing off recent support with volume.
        
        Conditions:
        - Current low > recent low (bouncing up)
        - Current volume > 20-bar average (confirmation)
        - RSI not overbought (< 60)
        
        Returns:
            True if reversal detected
        """
        if len(close_15m) < 20:
            return False
        
        try:
            current_low = low_15m[-1]
            recent_low = np.min(low_15m[-10:-1])  # Last 9 bars' low
            
            # Bouncing off support?
            if current_low <= recent_low:
                return False
            
            # Volume confirmation
            vol_ma = np.mean(volume_15m[-20:])
            if volume_15m[-1] < vol_ma:
                return False
            
            # RSI not overbought
            rsi_15m = self._calculate_rsi(close_15m, period=9)
            if rsi_15m > 60:
                return False
            
            logger.debug(
                f"15M Reversal detected: "
                f"Low bounce ${recent_low:.2f}→${current_low:.2f}, "
                f"Vol={volume_15m[-1]:.0f} vs MA={vol_ma:.0f}"
            )
            return True
        
        except Exception as e:
            logger.warning(f"Error detecting 15M reversal: {e}")
            return False
    
    def detect_m15_momentum(self, close_15m: np.ndarray) -> bool:
        """
        Detect bullish momentum on 15M.
        
        Conditions (pick one):
        - RSI < 30 (oversold reversal)
        - RSI rising into 40-60 zone (momentum turn)
        - Price above 10-bar MA (uptrend starting)
        
        Returns:
            True if bullish momentum detected
        """
        if len(close_15m) < 14:
            return False
        
        try:
            rsi_15m = self._calculate_rsi(close_15m, period=9)
            current_price = close_15m[-1]
            sma_10 = np.mean(close_15m[-10:])
            
            # Oversold reversal (strong signal)
            if rsi_15m < 30:
                logger.debug(f"15M Momentum: RSI oversold {rsi_15m:.1f}")
                return True
            
            # RSI turning positive in momentum zone
            if len(close_15m) > 2:
                rsi_prev = self._calculate_rsi(close_15m[:-1], period=9)
                if 40 <= rsi_15m <= 60 and rsi_15m > rsi_prev + 2:
                    logger.debug(f"15M Momentum: RSI rising {rsi_prev:.1f}→{rsi_15m:.1f}")
                    return True
            
            # Price above SMA (uptrend initiation)
            if current_price > sma_10 * 1.002:  # 0.2% above
                logger.debug(f"15M Momentum: Price above SMA {current_price:.2f} > {sma_10:.2f}")
                return True
            
            return False
        
        except Exception as e:
            logger.warning(f"Error detecting 15M momentum: {e}")
            return False
    
    # ==================== 1H SIGNALS ====================
    
    def detect_h1_divergence(
        self,
        close_1h: np.ndarray,
        low_1h: np.ndarray,
    ) -> bool:
        """
        Detect bullish divergence on 1H: Lower price low but higher RSI low.
        
        Classic bullish divergence = price making lower lows but momentum strengthening.
        
        Returns:
            True if bullish divergence detected
        """
        if len(close_1h) < 30:
            return False
        
        try:
            current_price = close_1h[-1]
            current_rsi = self._calculate_rsi(close_1h, period=14)
            current_low = low_1h[-1]
            
            # Need to identify previous swing low
            # Look for RSI local minimum in last 20 bars
            rsi_recent = np.array([self._calculate_rsi(close_1h[:i+1], period=14) 
                                  for i in range(max(0, len(close_1h)-20), len(close_1h))])
            
            if len(rsi_recent) < 5:
                return False
            
            # Find previous RSI low (minimum)
            prev_rsi_low_idx = np.argmin(rsi_recent[:-5])
            prev_rsi_low = rsi_recent[prev_rsi_low_idx]
            
            # Corresponding price low
            idx = max(0, len(close_1h) - 20 + prev_rsi_low_idx)
            prev_price_low = low_1h[idx]
            
            # Divergence: lower price low but RSI higher
            if (current_low < prev_price_low and 
                current_rsi > prev_rsi_low + 5):  # 5-point RSI improvement
                logger.debug(
                    f"1H Divergence: Lower price ${prev_price_low:.2f}→${current_low:.2f}, "
                    f"Higher RSI {prev_rsi_low:.1f}→{current_rsi:.1f}"
                )
                return True
            
            return False
        
        except Exception as e:
            logger.warning(f"Error detecting 1H divergence: {e}")
            return False
    
    def detect_h1_trend(self, close_1h: np.ndarray, high_1h: np.ndarray) -> bool:
        """
        Detect bullish trend confirmation on 1H.
        
        Conditions (any one):
        - Fast SMA (15) > Slow SMA (35) by 0.5%+
        - Price in upper half of 1H Bollinger band
        - ATR expanding (current > 20-bar MA)
        
        Returns:
            True if bullish trend detected
        """
        if len(close_1h) < 35:
            return False
        
        try:
            current_price = close_1h[-1]
            
            # SMA alignment
            sma_fast = np.mean(close_1h[-15:])
            sma_slow = np.mean(close_1h[-35:])
            
            if sma_fast > sma_slow * 1.005:  # 0.5% separation
                logger.debug(f"1H Trend: SMA aligned {sma_fast:.2f} > {sma_slow:.2f}")
                return True
            
            # Bollinger band position
            bb_mid = np.mean(close_1h[-20:])
            bb_std = np.std(close_1h[-20:])
            bb_upper = bb_mid + 2 * bb_std
            bb_lower = bb_mid - 2 * bb_std
            
            if current_price > bb_mid + bb_std * 0.5:  # Upper half
                logger.debug(
                    f"1H Trend: Price in upper BB ${current_price:.2f} "
                    f"vs mid ${bb_mid:.2f}"
                )
                return True
            
            # ATR expansion
            tr_list = []
            for i in range(1, len(close_1h)):
                h = high_1h[i]
                l = close_1h[i-1]
                tr = max(h - l, abs(h - close_1h[i-1]), abs(l - close_1h[i-1]))
                tr_list.append(tr)
            
            atr_current = np.mean(tr_list[-1:]) if tr_list else 0
            atr_ma = np.mean(tr_list[-20:]) if len(tr_list) >= 20 else atr_current
            
            if atr_current > atr_ma * 1.1:  # 10% above MA
                logger.debug(f"1H Trend: ATR expanding {atr_current:.2f} > {atr_ma:.2f}")
                return True
            
            return False
        
        except Exception as e:
            logger.warning(f"Error detecting 1H trend: {e}")
            return False
    
    def detect_h1_momentum(self, close_1h: np.ndarray) -> bool:
        """
        Detect momentum alignment on 1H: RSI > 50 and rising.
        
        Returns:
            True if momentum is positive
        """
        if len(close_1h) < 15:
            return False
        
        try:
            rsi = self._calculate_rsi(close_1h, period=14)
            rsi_prev = self._calculate_rsi(close_1h[:-1], period=14) if len(close_1h) > 1 else rsi
            
            # RSI above 50 and rising
            if rsi > 50 and rsi > rsi_prev:
                logger.debug(f"1H Momentum: RSI positive {rsi_prev:.1f}→{rsi:.1f}")
                return True
            
            return False
        
        except Exception as e:
            logger.warning(f"Error detecting 1H momentum: {e}")
            return False
    
    # ==================== COMPOSITE SIGNAL ====================
    
    def generate_entry_signal(
        self,
        close_15m: np.ndarray,
        high_15m: np.ndarray,
        low_15m: np.ndarray,
        volume_15m: np.ndarray,
        close_1h: np.ndarray,
        high_1h: np.ndarray,
        low_1h: np.ndarray,
    ) -> Tuple[bool, MultiTimeframeSignal]:
        """
        Generate composite multi-timeframe entry signal.
        
        Requires STRICT CONFLUENCE:
        - 15M Reversal: YES
        - 15M Momentum: YES
        - 1H Divergence OR (1H Trend AND 1H Momentum): YES
        - Must have 4/5 conditions aligned
        
        Returns:
            (should_enter: bool, signal: MultiTimeframeSignal)
        """
        signal = MultiTimeframeSignal()
        
        # 15M Signals (must both be true)
        signal.m15_reversal = self.detect_m15_reversal(close_15m, low_15m, volume_15m)
        signal.m15_momentum = self.detect_m15_momentum(close_15m)
        
        # 1H Signals
        signal.h1_divergence = self.detect_h1_divergence(close_1h, low_1h)
        signal.h1_trend = self.detect_h1_trend(close_1h, high_1h)
        signal.h1_momentum = self.detect_h1_momentum(close_1h)
        
        # Count aligned conditions
        conditions = [
            signal.m15_reversal,
            signal.m15_momentum,
            signal.h1_divergence,
            signal.h1_trend,
            signal.h1_momentum,
        ]
        signal.confluence_score = sum(conditions)
        
        # Entry requirements: STRICT confluence
        # 15M must have BOTH reversal + momentum
        # 1H must have EITHER divergence OR (trend + momentum)
        m15_confirmed = signal.m15_reversal and signal.m15_momentum
        h1_confirmed = signal.h1_divergence or (signal.h1_trend and signal.h1_momentum)
        
        should_enter = m15_confirmed and h1_confirmed and signal.confluence_score >= self.min_confluence
        
        signal.signal_strength = signal.confluence_score / 5.0  # 0-1 scale
        signal.entry_price = close_15m[-1]
        signal.timestamp = datetime.now()
        
        if should_enter:
            logger.info(
                f"ENTRY SIGNAL: Confluence {signal.confluence_score}/5 | "
                f"15M: {signal.m15_reversal}/{signal.m15_momentum} | "
                f"1H: {signal.h1_divergence}/{signal.h1_trend}/{signal.h1_momentum}"
            )
        
        return should_enter, signal
    
    def calculate_position_size(
        self,
        signal_strength: float,
        account_equity: float,
    ) -> float:
        """
        Calculate position size based on signal strength (confluence quality).
        
        1-2% risk, scaled by signal strength.
        """
        if signal_strength <= 0:
            return 0.0
        
        # Map confluence score to risk allocation
        base_risk = self.risk_pct
        scaled_risk = base_risk + (signal_strength * 0.5)  # +0.5% for perfect confluence
        
        return account_equity * scaled_risk / 100
    
    # ==================== UTILITIES ====================
    
    @staticmethod
    def _calculate_rsi(close: np.ndarray, period: int = 14) -> float:
        """Calculate RSI for given close prices."""
        if len(close) < period + 1:
            return 50.0
        
        delta = np.diff(close[-period-1:])
        gain = np.mean(np.maximum(delta, 0))
        loss = np.mean(np.abs(np.minimum(delta, 0)))
        
        if loss == 0:
            return 100.0 if gain > 0 else 0.0
        
        rs = gain / loss
        return 100.0 - (100.0 / (1.0 + rs))
