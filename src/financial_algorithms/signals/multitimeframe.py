"""Multi-timeframe ensemble for consensus trading signals."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple


def resample_ohlcv(
    df: pd.DataFrame,
    from_timeframe: str,
    to_timeframe: str,
) -> pd.DataFrame:
    """Resample OHLCV data to coarser timeframe.
    
    Args:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        from_timeframe: Source timeframe (e.g., '1min', '5min')
        to_timeframe: Target timeframe (e.g., '5min', '15min')
    
    Returns:
        Resampled OHLCV DataFrame
    """
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    resampled = pd.DataFrame()
    resampled['open'] = df['open'].resample(to_timeframe).first()
    resampled['high'] = df['high'].resample(to_timeframe).max()
    resampled['low'] = df['low'].resample(to_timeframe).min()
    resampled['close'] = df['close'].resample(to_timeframe).last()
    resampled['volume'] = df['volume'].resample(to_timeframe).sum()
    
    resampled = resampled.dropna(subset=['open'])
    return resampled.reset_index()


def generate_signals_for_timeframes(
    close_prices: Dict[str, pd.Series],
    signal_func,
    timeframes: list[str] = None,
) -> Dict[str, pd.Series]:
    """Generate signals at multiple timeframes.
    
    Args:
        close_prices: Dict mapping timeframe -> price series
        signal_func: Function that takes price series, returns signal
        timeframes: List of timeframes to process
    
    Returns:
        Dict mapping timeframe -> signal series
    """
    if timeframes is None:
        timeframes = sorted(close_prices.keys())
    
    signals = {}
    for tf in timeframes:
        if tf in close_prices:
            signals[tf] = signal_func(close_prices[tf])
    
    return signals


def align_signals_to_finest(
    signals: Dict[str, pd.Series],
    finest_timeframe: str = '1min',
) -> Dict[str, pd.Series]:
    """
    Align coarser timeframe signals to finest timeframe using forward-fill.
    
    Example:
        5min signal at 10:05 → fills all 1min bars from 10:05 to 10:09
    """
    aligned = {}
    finest_signal = signals[finest_timeframe].copy()
    aligned[finest_timeframe] = finest_signal
    
    for tf, sig in signals.items():
        if tf == finest_timeframe:
            continue
        
        # Reindex coarser signal to finest index, forward-fill
        reindexed = sig.reindex(finest_signal.index, method='ffill')
        aligned[tf] = reindexed
    
    return aligned


def consensus_signal(
    signals: Dict[str, pd.Series],
    vote_type: str = 'majority',
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Combine multi-timeframe signals via voting or weighted average.
    
    Args:
        signals: Dict mapping timeframe -> signal series (values in [-1, 1])
        vote_type: 'majority' (need >50%), 'unanimous' (all agree), 'weighted' (weighted avg)
        weights: Optional dict of timeframe -> weight for weighted voting
    
    Returns:
        Tuple of (consensus_signal, confidence_score)
    """
    if not signals:
        return pd.Series([0]), pd.Series([0])
    
    # Align all signals to finest timeframe
    finest_tf = min(signals.keys(), key=lambda x: int(x.split('min')[0]))
    aligned = align_signals_to_finest(signals, finest_tf)
    
    signal_df = pd.DataFrame(aligned)
    
    if vote_type == 'majority':
        # Signal where >50% of timeframes agree on direction
        consensus = signal_df.apply(lambda row: 1 if (row > 0).sum() > len(row) / 2 else (
            -1 if (row < 0).sum() > len(row) / 2 else 0
        ), axis=1)
        
        # Confidence: how many agree
        confidence = signal_df.apply(
            lambda row: max((row > 0).sum(), (row < 0).sum()) / len(row),
            axis=1
        )
    
    elif vote_type == 'unanimous':
        # Signal only when all timeframes agree
        consensus = pd.Series(0, index=signal_df.index)
        consensus = consensus.where(~((signal_df > 0).all(axis=1)), 1)
        consensus = consensus.where(~((signal_df < 0).all(axis=1)), -1)
        
        # Confidence: perfect (1.0) if unanimous, else 0
        confidence = ((signal_df > 0).all(axis=1) | (signal_df < 0).all(axis=1)).astype(float)
    
    elif vote_type == 'weighted':
        if weights is None:
            weights = {tf: 1.0 / len(signals) for tf in signals}
        
        # Normalize weights
        total_weight = sum(weights.values())
        weights = {tf: w / total_weight for tf, w in weights.items()}
        
        # Weighted average
        consensus = pd.Series(0.0, index=signal_df.index)
        confidence = pd.Series(0.0, index=signal_df.index)
        
        for tf, sig in aligned.items():
            w = weights.get(tf, 1.0 / len(signals))
            consensus += sig * w
            confidence += abs(sig) * w
        
        # Binarize to -1, 0, 1
        consensus = consensus.apply(lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0))
    
    else:
        raise ValueError(f"Unknown vote_type: {vote_type}")
    
    return consensus.astype(int), confidence


def ensemble_entry_conditions(
    signals_by_tf: Dict[str, pd.Series],
    voting_rule: str = 'majority',
    min_confidence: float = 0.5,
) -> Tuple[pd.Series, pd.Series]:
    """Generate ensemble entry signals with confidence filtering.
    
    Only generates entries when:
    1. Consensus signal exists (multiple timeframes agree)
    2. Confidence >= min_confidence
    
    Returns:
        Tuple of (ensemble_signal, raw_confidence)
    """
    consensus, confidence = consensus_signal(signals_by_tf, vote_type=voting_rule)
    
    # Filter by confidence threshold
    filtered = consensus.where(confidence >= min_confidence, 0)
    
    return filtered.astype(int), confidence


class MultiTimeframeEnsemble:
    """Manages multi-timeframe signal generation and consensus."""
    
    def __init__(
        self,
        timeframes: list[str] = None,
        voting: str = 'majority',
        min_confidence: float = 0.5,
    ):
        """
        Args:
            timeframes: List of timeframes ['1min', '5min', '15min']
            voting: Consensus type ('majority', 'unanimous', 'weighted')
            min_confidence: Min confidence [0, 1] to generate signal
        """
        self.timeframes = timeframes or ['1min', '5min', '15min']
        self.voting = voting
        self.min_confidence = min_confidence
        self.signals_cache = {}
    
    def prepare_data(
        self,
        df_1min: pd.DataFrame,
    ) -> Dict[str, pd.DataFrame]:
        """Prepare multi-timeframe OHLCV from 1-minute bars.
        
        Args:
            df_1min: 1-minute OHLCV DataFrame
        
        Returns:
            Dict of timeframe -> OHLCV DataFrame
        """
        data = {'1min': df_1min}
        
        for tf in self.timeframes:
            if tf == '1min':
                continue
            data[tf] = resample_ohlcv(df_1min, '1min', tf)
        
        return data
    
    def generate_consensus(
        self,
        signals: Dict[str, pd.Series],
    ) -> Tuple[pd.Series, pd.Series]:
        """Generate consensus signal from single-timeframe signals.
        
        Args:
            signals: Dict of timeframe -> signal series
        
        Returns:
            Tuple of (consensus_signal, confidence)
        """
        return ensemble_entry_conditions(
            signals,
            voting_rule=self.voting,
            min_confidence=self.min_confidence,
        )
    
    def analyze(
        self,
        df_1min: pd.DataFrame,
        signal_func,
    ) -> Dict:
        """Full pipeline: prepare data, generate signals, consensus.
        
        Args:
            df_1min: 1-minute OHLCV
            signal_func: Function to generate signal per timeframe
        
        Returns:
            Dict with keys: 'data', 'signals', 'consensus', 'confidence'
        """
        # Prepare multi-timeframe data
        data = self.prepare_data(df_1min)
        
        # Generate signals per timeframe
        signals = {}
        for tf, df in data.items():
            signals[tf] = signal_func(df['close'])
        
        # Generate consensus
        consensus, confidence = self.generate_consensus(signals)
        
        return {
            'data': data,
            'signals': signals,
            'consensus': consensus,
            'confidence': confidence,
        }


if __name__ == "__main__":
    print("Testing multi-timeframe ensemble...")
    
    # Create synthetic 1-min OHLCV
    np.random.seed(42)
    dates = pd.date_range('2026-01-01', periods=1440, freq='1min')  # 1 day
    prices = np.cumsum(np.random.normal(0, 0.1, 1440)) + 100
    
    df_1min = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices + np.abs(np.random.normal(0, 0.05, 1440)),
        'low': prices - np.abs(np.random.normal(0, 0.05, 1440)),
        'close': prices,
        'volume': np.random.uniform(1000, 5000, 1440),
    })
    
    # Resample to 5min and 15min
    df_5min = resample_ohlcv(df_1min, '1min', '5min')
    df_15min = resample_ohlcv(df_1min, '1min', '15min')
    
    print(f"1-min bars: {len(df_1min)}")
    print(f"5-min bars: {len(df_5min)}")
    print(f"15-min bars: {len(df_15min)}")
    
    # Simple signal: SMA crossover
    def sma_signal(close, fast=10, slow=20):
        fast_ma = close.ewm(span=fast).mean()
        slow_ma = close.ewm(span=slow).mean()
        sig = pd.Series(0, index=close.index)
        sig = sig.where(fast_ma <= slow_ma, 1)
        sig = sig.where(fast_ma >= slow_ma, -1)
        return sig
    
    signals = {
        '1min': sma_signal(df_1min['close']),
        '5min': sma_signal(df_5min['close']),
        '15min': sma_signal(df_15min['close']),
    }
    
    # Generate consensus
    consensus, confidence = ensemble_entry_conditions(
        signals,
        voting_rule='majority',
        min_confidence=0.5,
    )
    
    print(f"\nConsensus stats:")
    print(f"  Long signals: {(consensus == 1).sum()}")
    print(f"  Short signals: {(consensus == -1).sum()}")
    print(f"  Avg confidence: {confidence.mean():.3f}")
    
    print(f"\nSample output (last 20 bars):")
    output = pd.DataFrame({
        '1min_sig': signals['1min'].tail(20).values,
        '5min_sig': signals['5min'].tail(20).values,
        '15min_sig': signals['15min'].tail(20).values,
        'consensus': consensus.tail(20).values,
        'confidence': confidence.tail(20).values,
    })
    print(output)
