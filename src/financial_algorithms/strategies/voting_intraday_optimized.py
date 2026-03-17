#!/usr/bin/env python
"""
Intraday-Optimized Weighted Voting Strategy
Adjusts indicator periods for faster response on 1H, 4H, and shorter timeframes.

Key differences from daily strategy:
- Shorter SMA periods (10/25 vs 20/50) for faster trend detection
- Lower RSI/ADX periods (9 vs 14) for more sensitive reversals
- Higher volume thresholds (tighter band for confirmation)
- Lower entry thresholds (+1.5 vs +2.0) to capture more signals
"""

import numpy as np
from typing import Dict, Tuple, Optional
from financial_algorithms.strategies.voting_enhanced_weighted import EnhancedWeightedVotingStrategy
import logging

logger = logging.getLogger(__name__)


class IntradayOptimizedVotingStrategy(EnhancedWeightedVotingStrategy):
    """
    Intraday-optimized voting strategy with aggressive indicator tuning.
    """
    
    # Strategy parameters by timeframe
    TIMEFRAME_CONFIG = {
        '1h': {
            'sma_fast': 10,
            'sma_slow': 25,
            'rsi_period': 9,
            'volume_period': 10,
            'adx_period': 9,
            'atr_period': 9,
            'min_buy_score': 1.5,  # Lower threshold for more signals
            'max_sell_score': -1.5,
            'rsi_oversold': 35,  # Shifted from 30
            'rsi_overbought': 65,  # Shifted from 70
        },
        '4h': {
            'sma_fast': 15,
            'sma_slow': 35,
            'rsi_period': 11,
            'volume_period': 14,
            'adx_period': 11,
            'atr_period': 11,
            'min_buy_score': 1.8,
            'max_sell_score': -1.8,
            'rsi_oversold': 32,
            'rsi_overbought': 68,
        },
    }
    
    def __init__(
        self,
        timeframe: str = '1h',
        risk_pct: float = 2.0,
        tp1_pct: float = 1.5,
        tp2_pct: float = 3.0,
    ):
        """
        Initialize intraday-optimized strategy.
        
        Args:
            timeframe: '1h' or '4h' (determines indicator periods)
            risk_pct: Risk per trade (%)
            tp1_pct: Partial profit target (%)
            tp2_pct: Full profit target (%)
        """
        if timeframe not in self.TIMEFRAME_CONFIG:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Use 1h or 4h")
        
        self.timeframe = timeframe
        config = self.TIMEFRAME_CONFIG[timeframe]
        
        # Store config for later use
        self.sma_fast = config['sma_fast']
        self.sma_slow = config['sma_slow']
        self.rsi_period = config['rsi_period']
        self.volume_period = config['volume_period']
        self.adx_period = config['adx_period']
        self.atr_period = config['atr_period']
        self.rsi_oversold = config['rsi_oversold']
        self.rsi_overbought = config['rsi_overbought']
        
        # Initialize parent with intraday thresholds
        super().__init__(
            risk_pct=risk_pct,
            tp1_pct=tp1_pct,
            tp2_pct=tp2_pct,
            min_buy_score=config['min_buy_score'],
            max_sell_score=config['max_sell_score'],
        )
        
        logger.info(
            f"Initialized intraday strategy for {timeframe}: "
            f"SMA({self.sma_fast}/{self.sma_slow}), "
            f"RSI({self.rsi_period}/{self.rsi_oversold}/{self.rsi_overbought}), "
            f"Entry threshold: {config['min_buy_score']}"
        )
    
    def calculate_voting_score(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        volume: np.ndarray,
    ) -> float:
        """
        Calculate composite voting score with intraday-optimized periods.
        
        Returns:
            Composite score (-10 to +10)
        """
        # Calculate individual signals with intraday periods
        sma_signal = self.calculate_sma_signal(
            close,
            fast_period=self.sma_fast,
            slow_period=self.sma_slow,
        )
        
        rsi_signal = self.calculate_rsi_signal(
            close,
            period=self.rsi_period,
            overbought=self.rsi_overbought,
            oversold=self.rsi_oversold,
        )
        
        volume_signal = self.calculate_volume_signal(
            close,
            volume,
            period=self.volume_period,
        )
        
        adx_signal = self.calculate_adx_signal(
            high,
            low,
            close,
            period=self.adx_period,
        )
        
        atr_signal = self.calculate_atr_signal(
            high,
            low,
            close,
            period=self.atr_period,
        )
        
        # Weighted sum (-10 to +10 range)
        composite_score = (
            sma_signal * 2.0 +    # Weight: 2.0
            rsi_signal * 2.0 +    # Weight: 2.0
            volume_signal * 1.5 + # Weight: 1.5 (volume less critical)
            adx_signal * 2.0 +    # Weight: 2.0
            atr_signal * 1.5      # Weight: 1.5
        ) / 2.0  # Normalize by total weights / 2
        
        return composite_score
    
    def calculate_position_size(self, score: float, account_equity: float) -> float:
        """
        Calculate position size based on signal strength - intraday tuned.
        
        Intraday positions tend to close faster, so we can use tighter risk bands.
        """
        if score < self.min_buy_score:
            return 0.0
        
        # Tighter sizing for intraday volatility
        if score < 2.0:
            risk_pct = 3.0  # Weak signal: 3%
        elif score < 3.0:
            risk_pct = 4.0  # Moderate: 4%
        else:
            risk_pct = 6.0  # Strong: 6% (higher conviction)
        
        return account_equity * risk_pct / 100


class IntradayStrategyFactory:
    """Factory to create appropriate intraday strategies by timeframe."""
    
    @staticmethod
    def create(timeframe: str = '1h', **kwargs) -> IntradayOptimizedVotingStrategy:
        """
        Create intraday strategy for the specified timeframe.
        
        Args:
            timeframe: '1h' or '4h'
            **kwargs: Additional strategy parameters
        
        Returns:
            Configured IntradayOptimizedVotingStrategy
        """
        return IntradayOptimizedVotingStrategy(timeframe=timeframe, **kwargs)
