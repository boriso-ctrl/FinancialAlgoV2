"""
Enhanced indicator calculations with divergence detection, momentum tracking,
and multi-timeframe confirmation support.

Features:
- Historic divergence detection (price vs momentum)
- Signal momentum tracking (improving/deteriorating)
- Confidence weighting per indicator
- Multi-timeframe agreement scoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class EnhancedIndicators:
    """Enhanced technical indicators with divergence and momentum tracking."""
    
    @staticmethod
    def detect_divergence(
        prices: np.ndarray,
        indicator_values: np.ndarray,
        window: int = 20,
        divergence_threshold: float = 0.02,
    ) -> Dict[str, any]:
        """
        Detect price vs indicator divergence.
        
        Bullish divergence: Price new low, but indicator higher than previous low
        Bearish divergence: Price new high, but indicator lower than previous high
        
        Args:
            prices: Price array
            indicator_values: Indicator (RSI, momentum, etc.) array
            window: Lookback window for divergence detection
            divergence_threshold: Min % difference for divergence signal
        
        Returns:
            Dict with divergence_type, strength, confidence
        """
        if len(prices) < window + 2 or len(indicator_values) < window + 2:
            return {
                'divergence_type': 'NONE',
                'strength': 0.0,
                'confidence': 0.0,
            }
        
        # Find local extrema
        recent_prices = prices[-window:]
        recent_indicators = indicator_values[-window:]
        
        price_low_idx = np.argmin(recent_prices)
        price_high_idx = np.argmax(recent_prices)
        
        indicator_low_idx = np.argmin(recent_indicators)
        indicator_high_idx = np.argmax(recent_indicators)
        
        current_price = prices[-1]
        current_indicator = indicator_values[-1]
        
        # Check for bullish divergence
        if (current_price < prices[price_low_idx] and 
            current_indicator > indicator_values[indicator_low_idx]):
            
            price_change = (current_price - prices[price_low_idx]) / prices[price_low_idx]
            indicator_change = (current_indicator - indicator_values[indicator_low_idx]) / (
                indicator_values[indicator_low_idx] + 1e-6
            )
            
            if abs(price_change) > divergence_threshold:
                return {
                    'divergence_type': 'BULLISH',
                    'strength': min(1.0, abs(indicator_change) / (abs(price_change) + 1e-6)),
                    'confidence': 0.8,
                }
        
        # Check for bearish divergence
        if (current_price > prices[price_high_idx] and 
            current_indicator < indicator_values[indicator_high_idx]):
            
            price_change = (current_price - prices[price_high_idx]) / prices[price_high_idx]
            indicator_change = (current_indicator - indicator_values[indicator_high_idx]) / (
                indicator_values[indicator_high_idx] + 1e-6
            )
            
            if abs(price_change) > divergence_threshold:
                return {
                    'divergence_type': 'BEARISH',
                    'strength': min(1.0, abs(indicator_change) / (abs(price_change) + 1e-6)),
                    'confidence': 0.8,
                }
        
        return {
            'divergence_type': 'NONE',
            'strength': 0.0,
            'confidence': 0.0,
        }
    
    @staticmethod
    def calculate_signal_momentum(
        signal_scores: np.ndarray,
        window: int = 5,
    ) -> float:
        """
        Calculate momentum of signal scores.
        
        Positive: Signals improving (momentum toward +10)
        Negative: Signals deteriorating (momentum toward -10)
        
        Args:
            signal_scores: Array of recent score values
            window: Period for momentum calculation
        
        Returns:
            float: Score momentum in range [-10, +10]
        """
        if len(signal_scores) < window + 1:
            return 0.0
        
        recent = signal_scores[-window:]
        
        # Simple momentum: current - MA of previous
        current_score = recent[-1]
        prev_avg = np.mean(recent[:-1])
        
        momentum = current_score - prev_avg
        
        return np.clip(momentum, -10, 10)
    
    @staticmethod
    def calculate_confidence_weight(
        signal_values: Dict[str, float],
        agreement_level: float,
    ) -> Dict[str, float]:
        """
        Weight indicator confidence based on directional agreement.
        
        If multiple indicators agree (all bullish or all bearish), each gets higher weight.
        If mixed signals, lower weights.
        
        Args:
            signal_values: Dict of indicator signals (sma, rsi, volume, etc.)
            agreement_level: % of indicators agreeing on direction
        
        Returns:
            Dict of confidence weights per indicator
        """
        weights = {}
        
        # Count agreement
        bullish_count = sum(1 for v in signal_values.values() if v > 0)
        bearish_count = sum(1 for v in signal_values.values() if v < 0)
        total_count = len(signal_values)
        
        max_agreement = max(bullish_count, bearish_count) / total_count if total_count > 0 else 0
        
        # Assign weights based on agreement
        for indicator, value in signal_values.items():
            if max_agreement > 0.8:
                # Strong consensus: all signals get 1.0 weight
                weights[indicator] = 1.0
            elif max_agreement > 0.6:
                # Moderate consensus: signals agreeing get 0.9, others 0.7
                if (bullish_count >= bearish_count and value > 0) or \
                   (bearish_count > bullish_count and value < 0):
                    weights[indicator] = 0.9
                else:
                    weights[indicator] = 0.7
            else:
                # Low consensus: all get reduced weight
                weights[indicator] = 0.5
        
        return weights
    
    @staticmethod
    def calculate_regime_context(
        vix: Optional[np.ndarray] = None,
        volatility_pct: Optional[np.ndarray] = None,
    ) -> Dict[str, any]:
        """
        Determine market regime context for signal adjustment.
        
        Args:
            vix: VIX values (if available)
            volatility_pct: Historical volatility %
        
        Returns:
            Dict with regime, risk_level, confidence_adjustment
        """
        if vix is not None and len(vix) > 0:
            current_vix = vix[-1]
            
            if current_vix < 15:
                regime = 'LOW_VIX_CALM'
                risk_level = 'LOW'
                confidence_adj = 0.9  # Reduce confidence in calm markets
            elif current_vix < 20:
                regime = 'NORMAL'
                risk_level = 'MEDIUM'
                confidence_adj = 1.0
            elif current_vix < 30:
                regime = 'ELEVATED'
                risk_level = 'HIGH'
                confidence_adj = 0.95
            else:
                regime = 'HIGH_VIX_CRISIS'
                risk_level = 'VERY_HIGH'
                confidence_adj = 0.8
        elif volatility_pct is not None and len(volatility_pct) > 0:
            vol = volatility_pct[-1]
            
            if vol < 0.5:
                regime = 'LOW_VOLATILITY'
                risk_level = 'LOW'
                confidence_adj = 0.9
            elif vol < 1.0:
                regime = 'NORMAL'
                risk_level = 'MEDIUM'
                confidence_adj = 1.0
            elif vol < 2.0:
                regime = 'HIGH_VOLATILITY'
                risk_level = 'HIGH'
                confidence_adj = 0.95
            else:
                regime = 'EXTREME_VOLATILITY'
                risk_level = 'VERY_HIGH'
                confidence_adj = 0.8
        else:
            regime = 'UNKNOWN'
            risk_level = 'MEDIUM'
            confidence_adj = 0.95
        
        return {
            'regime': regime,
            'risk_level': risk_level,
            'confidence_adjustment': confidence_adj,
        }
    
    @staticmethod
    def calculate_multiframe_agreement(
        signals_1m: Dict[str, float],
        signals_5m: Dict[str, float],
        signals_15m: Optional[Dict[str, float]] = None,
    ) -> Dict[str, any]:
        """
        Calculate multi-timeframe agreement strength.
        
        All timeframes bullish/bearish = strong confidence
        Mixed timeframes = weaker signal
        
        Args:
            signals_1m: 1-minute indicator signals
            signals_5m: 5-minute indicator signals
            signals_15m: 15-minute indicator signals (optional)
        
        Returns:
            Dict with agreement_score, strength, direction
        """
        frames = [signals_1m, signals_5m]
        if signals_15m is not None:
            frames.append(signals_15m)
        
        bullish_frames = sum(1 for f in frames if sum(f.values()) > 0)
        bearish_frames = sum(1 for f in frames if sum(f.values()) < 0)
        
        max_agreement = max(bullish_frames, bearish_frames) / len(frames)
        
        if bullish_frames > bearish_frames:
            direction = 'BULLISH'
        elif bearish_frames > bullish_frames:
            direction = 'BEARISH'
        else:
            direction = 'NEUTRAL'
        
        return {
            'agreement_score': max_agreement,
            'strength': 'STRONG' if max_agreement > 0.8 else 'MODERATE' if max_agreement > 0.6 else 'WEAK',
            'direction': direction,
            'frames_bullish': bullish_frames,
            'frames_total': len(frames),
        }
    
    @staticmethod
    def calculate_indicator_rank(
        sma_signal: float,
        rsi_signal: float,
        volume_signal: float,
        adx_signal: float,
        atr_signal: float,
    ) -> List[Tuple[str, float]]:
        """
        Rank indicators by absolute signal strength.
        
        Returns:
            List of (indicator_name, abs_strength) tuples, sorted by strength
        """
        signals = [
            ('SMA', abs(sma_signal)),
            ('RSI', abs(rsi_signal)),
            ('Volume', abs(volume_signal)),
            ('ADX', abs(adx_signal)),
            ('ATR', abs(atr_signal)),
        ]
        
        return sorted(signals, key=lambda x: x[1], reverse=True)
    
    @staticmethod
    def apply_divergence_bonus(
        base_score: float,
        divergence: Dict[str, any],
        max_bonus: float = 1.0,
    ) -> float:
        """
        Apply bonus to voting score if divergence detected.
        
        Bullish divergence adds up to +1.0 to score.
        Bearish divergence subtracts up to -1.0 from score.
        
        Args:
            base_score: Original voting score
            divergence: Divergence detection result
            max_bonus: Maximum bonus/penalty to apply
        
        Returns:
            float: Adjusted score
        """
        if divergence['divergence_type'] == 'BULLISH':
            bonus = divergence['strength'] * divergence['confidence'] * max_bonus
            return min(10.0, base_score + bonus)
        elif divergence['divergence_type'] == 'BEARISH':
            penalty = divergence['strength'] * divergence['confidence'] * max_bonus
            return max(-10.0, base_score - penalty)
        
        return base_score
    
    @staticmethod
    def get_signal_explanation(
        score: float,
        sma_sig: float,
        rsi_sig: float,
        vol_sig: float,
        adx_sig: float,
        atr_sig: float,
    ) -> str:
        """Generate human-readable explanation of signal."""
        explanation = []
        
        if score >= 5:
            explanation.append("STRONG BUY" if score >= 8 else "BUY")
        elif score <= -5:
            explanation.append("STRONG SELL" if score <= -8 else "SELL")
        else:
            explanation.append("NEUTRAL")
        
        explanation.append(f" | Score: {score:+.1f}/10")
        explanation.append(f" | SMA:{sma_sig:+.1f} RSI:{rsi_sig:+.1f} Vol:{vol_sig:+.1f} ADX:{adx_sig:+.1f} ATR:{atr_sig:+.1f}")
        
        return "".join(explanation)
