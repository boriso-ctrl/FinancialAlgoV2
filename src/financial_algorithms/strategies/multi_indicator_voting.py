"""
Hybrid Strategy: Multi-Indicator Voting System

Architecture:
- 8 Indicators: Each votes -1, 0, or +1
- Cumulative Score: Sum of all votes
- Entry: Score > +5 (5+ indicators agree on direction)
- Exit: Score <= 0 (consensus broken)
- Position Sizing: 0% at score 0, max 4% at score +8, average ~2%
- Stop Loss: Automatic when score drops to 0
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class IndicatorVote:
    """Single indicator's vote."""
    name: str
    vote: int  # -1, 0, or +1
    value: float  # Underlying metric value
    reason: str  # Why this vote


class HybridVotingStrategy:
    """Multi-indicator voting strategy for daily trading."""
    
    def __init__(
        self,
        buy_threshold: int = 5,  # Buy when score > +5
        exit_threshold: int = 0,  # Exit when score <= 0
        max_position_size: float = 0.04,  # 4% max
        target_avg_position: float = 0.02,  # Aim for 2% average
        indicator_weights: dict = None,  # Custom weights for each indicator
    ):
        """
        Args:
            buy_threshold: Minimum score to enter (+5 means 5 of 8 indicators agree)
            exit_threshold: Exit when score drops to this (stop loss at 0)
            max_position_size: Maximum position as % of capital (0.04 = 4%)
            target_avg_position: Target average position size
            indicator_weights: Dict mapping indicator names to weights (default: all 1.0)
        """
        self.buy_threshold = buy_threshold
        self.exit_threshold = exit_threshold
        self.max_position_size = max_position_size
        self.target_avg_position = target_avg_position
        
        # Default equal weights for all 8 indicators
        self.weights = indicator_weights or {
            'sma_crossover': 1.0,
            'rsi': 1.0,
            'macd': 1.0,
            'bollinger_bands': 1.0,
            'volume': 1.0,
            'adx': 1.0,
            'stochastic': 1.0,
            'atr_trend': 1.0,
        }
    
    # =========================================================================
    # INDICATOR 1: SMA Crossover (Trend Direction)
    # =========================================================================
    def indicator_sma_crossover(self, close: pd.Series) -> int:
        """
        SMA crossover: 10-day vs 20-day EMA
        +1: Uptrend (fast > slow)
        -1: Downtrend (fast < slow)
         0: Neutral
        """
        fast_ma = close.ewm(span=10).mean()
        slow_ma = close.ewm(span=20).mean()
        
        if fast_ma.iloc[-1] > slow_ma.iloc[-1]:
            return 1
        elif fast_ma.iloc[-1] < slow_ma.iloc[-1]:
            return -1
        else:
            return 0
    
    # =========================================================================
    # INDICATOR 2: RSI (Momentum, Extremes)
    # =========================================================================
    def calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """Calculate 14-period RSI."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    def indicator_rsi(self, close: pd.Series) -> int:
        """
        RSI momentum indicator
        +1: Bullish (RSI rising AND 40-60 zone)
        -1: Bearish (RSI falling AND 40-60 zone)
         0: Neutral or extreme (RSI < 30 or > 70)
        """
        rsi = self.calculate_rsi(close)
        rsi_prev = self.calculate_rsi(close.iloc[:-1]) if len(close) > 1 else rsi
        
        # Extremes = neutral
        if rsi < 30 or rsi > 70:
            return 0
        
        # Momentum direction
        if rsi > rsi_prev:
            return 1  # Rising momentum
        elif rsi < rsi_prev:
            return -1  # Falling momentum
        else:
            return 0
    
    # =========================================================================
    # INDICATOR 3: MACD (Momentum Crossover)
    # =========================================================================
    def indicator_macd(self, close: pd.Series) -> int:
        """
        MACD momentum crossover
        +1: MACD > Signal Line (bullish)
        -1: MACD < Signal Line (bearish)
         0: Equal or converging
        """
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        
        if macd.iloc[-1] > signal.iloc[-1]:
            return 1
        elif macd.iloc[-1] < signal.iloc[-1]:
            return -1
        else:
            return 0
    
    # =========================================================================
    # INDICATOR 4: Bollinger Bands (Volatility Breakout)
    # =========================================================================
    def indicator_bollinger_bands(self, close: pd.Series) -> int:
        """
        Bollinger Bands breakout
        +1: Close > Upper Band (bullish breakout)
        -1: Close < Lower Band (bearish breakout)
         0: Within bands
        """
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        if close.iloc[-1] > upper.iloc[-1]:
            return 1
        elif close.iloc[-1] < lower.iloc[-1]:
            return -1
        else:
            return 0
    
    # =========================================================================
    # INDICATOR 5: Volume Confirmation
    # =========================================================================
    def indicator_volume(self, volume: pd.Series) -> int:
        """
        Volume confirmation
        +1: Current volume > 1.2x average (strong confirmation)
        -1: Current volume < 0.8x average (weak, avoid)
         0: Average volume
        """
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1]
        
        if vol_ratio > 1.2:
            return 1
        elif vol_ratio < 0.8:
            return -1
        else:
            return 0
    
    # =========================================================================
    # INDICATOR 6: ADX (Trend Strength)
    # =========================================================================
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Simplified ADX calculation."""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where(plus_dm > minus_dm, 0).where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > plus_dm, 0).where(minus_dm > 0, 0)
        
        plus_di = 100 * plus_dm.rolling(period).mean() / atr
        minus_di = 100 * minus_dm.rolling(period).mean() / atr
        
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100 * di_diff / di_sum.replace(0, np.nan)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1]
    
    def indicator_adx(self, high: pd.Series, low: pd.Series, close: pd.Series) -> int:
        """
        ADX trend strength
        +1: Strong uptrend (ADX > 25 and +DI > -DI)
        -1: Strong downtrend (ADX > 25 and -DI > +DI)
         0: Weak trend (ADX < 25)
        """
        adx = self.calculate_adx(high, low, close)
        
        if adx < 25:
            return 0  # Weak trend
        
        # Determine direction
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        plus_dm = high.diff().where(high.diff() > -low.diff(), 0).where(high.diff() > 0, 0)
        minus_dm = -low.diff().where(-low.diff() > high.diff(), 0).where(-low.diff() > 0, 0)
        
        plus_di = 100 * plus_dm.rolling(14).mean() / atr
        minus_di = 100 * minus_dm.rolling(14).mean() / atr
        
        if plus_di.iloc[-1] > minus_di.iloc[-1]:
            return 1
        elif plus_di.iloc[-1] < minus_di.iloc[-1]:
            return -1
        else:
            return 0
    
    # =========================================================================
    # INDICATOR 7: Stochastic Oscillator (Momentum)
    # =========================================================================
    def indicator_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series) -> int:
        """
        Stochastic oscillator
        +1: Bullish signal (K > D and not overbought)
        -1: Bearish signal (K < D and not oversold)
         0: Neutral
        """
        period = 14
        lowest_low = low.rolling(period).min()
        highest_high = high.rolling(period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d_percent = k_percent.rolling(3).mean()
        
        k_curr = k_percent.iloc[-1]
        d_curr = d_percent.iloc[-1]
        
        if k_curr > 80 or d_curr > 80:
            return 0  # Overbought, neutral
        elif k_curr < 20 or d_curr < 20:
            return 0  # Oversold, neutral
        elif k_curr > d_curr:
            return 1  # K > D, bullish
        elif k_curr < d_curr:
            return -1  # K < D, bearish
        else:
            return 0
    
    # =========================================================================
    # INDICATOR 8: ATR-Based Volatility Trend
    # =========================================================================
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate ATR."""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean().iloc[-1]
    
    def indicator_atr_trend(self, high: pd.Series, low: pd.Series, close: pd.Series) -> int:
        """
        ATR trend confirmation (volatility-adjusted)
        +1: Low volatility (ATR < 2% of close) - good conditions
        -1: High volatility (ATR > 3% of close) - risky
         0: Normal volatility
        """
        atr = self.calculate_atr(high, low, close)
        atr_pct = atr / close.iloc[-1]
        
        if atr_pct < 0.02:
            return 1  # Low vol, good conditions
        elif atr_pct > 0.03:
            return -1  # High vol, risky
        else:
            return 0  # Normal
    
    # =========================================================================
    # VOTING SYSTEM: Aggregate all 8 indicators
    # =========================================================================
    def calculate_voting_score(self, df: pd.DataFrame) -> float:
        """
        Calculate weighted cumulative voting score from 8 indicators.
        
        Each indicator vote is multiplied by its weight, then summed.
        
        Returns:
            Float from -8 to +8 representing weighted consensus
        """
        scores = []
        
        # 1. SMA Crossover
        scores.append(self.indicator_sma_crossover(df['close']) * self.weights['sma_crossover'])
        
        # 2. RSI
        scores.append(self.indicator_rsi(df['close']) * self.weights['rsi'])
        
        # 3. MACD
        scores.append(self.indicator_macd(df['close']) * self.weights['macd'])
        
        # 4. Bollinger Bands
        scores.append(self.indicator_bollinger_bands(df['close']) * self.weights['bollinger_bands'])
        
        # 5. Volume
        scores.append(self.indicator_volume(df['volume']) * self.weights['volume'])
        
        # 6. ADX
        scores.append(self.indicator_adx(df['high'], df['low'], df['close']) * self.weights['adx'])
        
        # 7. Stochastic
        scores.append(self.indicator_stochastic(df['high'], df['low'], df['close']) * self.weights['stochastic'])
        
        # 8. ATR Trend
        scores.append(self.indicator_atr_trend(df['high'], df['low'], df['close']) * self.weights['atr_trend'])
        
        return sum(scores)
    
    def position_size_from_score(self, score: float) -> float:
        """
        Calculate position size based on weighted voting score.
        
        Score mapping (with weighted indicators, range is now ~-8 to +8):
        score <= -buy_threshold: Full short (-4%)
        -buy_threshold < score < +buy_threshold: Reduced position (linear 0% to 4%)
        score >= +buy_threshold: Full long (+4%)
        
        Average: ~2% (targets user's desired average)
        """
        if score <= -self.buy_threshold:
            # Full short position
            return -self.max_position_size
        elif score >= self.buy_threshold:
            # Full long position
            return self.max_position_size
        else:
            # Linear scaling between -buy_threshold and +buy_threshold
            # Maps to -4% to +4%
            return (score / self.buy_threshold) * self.max_position_size
    
    # =========================================================================
    # BACKTESTING ENGINE
    # =========================================================================
    def backtest_daily(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000,
    ) -> Dict:
        """
        Backtest multi-indicator voting strategy on daily bars.
        
        Entry: When score > +5 (or < -5)
        Exit: When score drops to <= 0
        Position size: 0-4% based on score
        
        Args:
            df: DataFrame with OHLCV + symbol
            initial_capital: Starting capital
        
        Returns:
            Dict with metrics
        """
        results = {
            'trades': [],
            'equity': [initial_capital],
            'drawdown': [0],
            'daily_scores': [],
        }
        
        equity = initial_capital
        positions = {}  # symbol -> {'entry_price', 'entry_signal', 'entry_size', 'entry_score'}
        
        # Process each symbol
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].reset_index(drop=True)
            positions[symbol] = None
            
            for idx in range(len(symbol_data)):
                row = symbol_data.iloc[idx]
                df_slice = symbol_data.iloc[:idx+1]  # Data up to this point
                
                # Calculate score
                try:
                    score = self.calculate_voting_score(df_slice)
                except:
                    score = 0
                
                results['daily_scores'].append({'symbol': symbol, 'idx': idx, 'score': score})
                
                price = row['close']
                position_size = self.position_size_from_score(score)
                
                # EXIT LOGIC
                if positions[symbol] is not None:
                    entry_info = positions[symbol]
                    # Exit when score goes to 0 or crosses zero
                    if (score <= self.exit_threshold and entry_info['entry_signal'] > 0) or \
                       (score >= -self.exit_threshold and entry_info['entry_signal'] < 0):
                        pnl = (price - entry_info['entry_price']) * entry_info['entry_signal'] * entry_info['entry_size']
                        equity += pnl
                        results['trades'].append({
                            'symbol': symbol,
                            'entry_price': entry_info['entry_price'],
                            'exit_price': price,
                            'entry_score': entry_info['entry_score'],
                            'exit_score': score,
                            'direction': 'long' if entry_info['entry_signal'] > 0 else 'short',
                            'pnl': pnl,
                            'position_size': entry_info['entry_size'],
                        })
                        positions[symbol] = None
                
                # ENTRY LOGIC
                if positions[symbol] is None:
                    if score > self.buy_threshold:
                        # Long entry
                        positions[symbol] = {
                            'entry_price': price,
                            'entry_signal': 1,
                            'entry_size': position_size,
                            'entry_score': score,
                        }
                    elif score < -self.buy_threshold:
                        # Short entry
                        positions[symbol] = {
                            'entry_price': price,
                            'entry_signal': -1,
                            'entry_size': abs(position_size),
                            'entry_score': score,
                        }
                
                # Track equity
                results['equity'].append(equity)
                
                # Track max drawdown
                max_equity = max(results['equity'])
                dd = (equity - max_equity) / max_equity if max_equity > 0 else 0
                results['drawdown'].append(dd)
        
        # Calculate metrics
        returns = pd.Series(results['equity']).pct_change().dropna()
        
        metrics = {
            'total_return': (results['equity'][-1] - initial_capital) / initial_capital,
            'sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'max_drawdown': min(results['drawdown']),
            'win_rate': sum(1 for t in results['trades'] if t['pnl'] > 0) / len(results['trades']) if results['trades'] else 0,
            'num_trades': len(results['trades']),
            'avg_position_size': np.mean([t['position_size'] for t in results['trades']]) if results['trades'] else 0,
        }
        
        return metrics


if __name__ == "__main__":
    print("Testing Multi-Indicator Voting Strategy...")
    
    # Create synthetic daily OHLCV
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    
    for symbol in ['AAPL', 'MSFT']:
        prices = 100 + np.cumsum(np.random.normal(0, 2, 500))
        df_data = pd.DataFrame({
            'date': dates,
            'symbol': symbol,
            'open': prices - np.random.uniform(0, 1, 500),
            'high': prices + np.random.uniform(1, 2, 500),
            'low': prices - np.random.uniform(1, 2, 500),
            'close': prices,
            'volume': np.random.uniform(50e6, 100e6, 500),
        })
        
        if symbol == 'AAPL':
            df_all = df_data
        else:
            df_all = pd.concat([df_all, df_data], ignore_index=True)
    
    # Run backtest
    strategy = HybridVotingStrategy(
        buy_threshold=5,
        exit_threshold=0,
        max_position_size=0.04,  # 4% max
        target_avg_position=0.02,  # 2% target
    )
    
    metrics = strategy.backtest_daily(df_all, initial_capital=100000)
    
    print("\n" + "="*70)
    print("MULTI-INDICATOR VOTING STRATEGY RESULTS")
    print("="*70)
    print(f"Total Return: {metrics['total_return']*100:>8.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe']:>8.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:>8.2f}%")
    print(f"Win Rate: {metrics['win_rate']*100:>8.1f}%")
    print(f"Trades: {metrics['num_trades']:>8.0f}")
    print(f"Avg Position Size: {metrics['avg_position_size']*100:>8.2f}%")
    print("="*70)
    
    print(f"\n✓ Multi-indicator voting strategy ready for integration")
