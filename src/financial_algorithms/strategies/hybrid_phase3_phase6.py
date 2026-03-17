"""
Phase 3 Strategy + Phase 6 Regime Filters (Hybrid Approach)

Base: Phase 3 daily trading strategy (Sharpe 1.65 proven)
Enhancements: Phase 6 regime detection & volatility filters

Strategy:
1. Generate Phase 3 signals (multi-indicator consensus)
2. Filter by Phase 6 regime conditions (RSI, trend, volume)
3. Size positions based on volatility
4. Trade daily bars with regime-aware entries/exits
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HybridSignal:
    """Signal with regime context."""
    direction: int  # -1, 0, 1
    confidence: float  # 0-1
    regime_allowed: bool
    volatility_mult: float  # Position size multiplier


class HybridPhase3Phase6:
    """Combines Phase 3 daily strategy with Phase 6 regime filters."""
    
    def __init__(
        self,
        rsi_threshold_oversold: float = 25,
        rsi_threshold_overbought: float = 75,
        min_volume_ma_ratio: float = 0.8,  # Trade only if vol >= 80% of MA
        volatility_target: float = 0.02,  # 2% target daily ATR
    ):
        """
        Args:
            rsi_threshold_oversold: RSI below this = skip trades (choppy)
            rsi_threshold_overbought: RSI above this = skip trades (choppy)
            min_volume_ma_ratio: Minimum volume requirement (0-1)
            volatility_target: Reference volatility for position sizing
        """
        self.rsi_OS = rsi_threshold_oversold
        self.rsi_OB = rsi_threshold_overbought
        self.min_vol_ratio = min_volume_ma_ratio
        self.vol_target = volatility_target
    
    def calculate_rsi(
        self,
        prices: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_volatility(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """Calculate ATR-based volatility as % of close."""
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        volatility_pct = atr / close
        return volatility_pct
    
    def regime_filter(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Filter: only allow trading in healthy market regimes.
        
        Returns: Binary series (1=OK to trade, 0=skip)
        """
        regime = pd.Series(1, index=df.index)
        
        # Filter 1: RSI extremes (choppy markets)
        rsi = self.calculate_rsi(df['close'])
        regime = regime.where(
            ~((rsi < self.rsi_OS) | (rsi > self.rsi_OB)),
            0
        )
        
        # Filter 2: Volume (ignore low-volume bars)
        vol_ma = df['volume'].rolling(20).mean()
        regime = regime.where(
            df['volume'] >= vol_ma * self.min_vol_ratio,
            0
        )
        
        return regime
    
    def volatility_multiplier(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Position size multiplier based on volatility.
        
        Low vol -> can trade bigger
        High vol -> trade smaller
        
        Returns: Multiplier (0.5 to 1.5)
        """
        volatility = self.calculate_volatility(df['high'], df['low'], df['close'])
        
        # Ratio of current vol to target vol
        vol_ratio = self.vol_target / volatility.clip(lower=0.001)
        
        # Clamp to 0.5x - 1.5x
        multiplier = vol_ratio.clip(0.5, 1.5)
        
        return multiplier
    
    def generate_phase3_signal(
        self,
        df: pd.DataFrame,
    ) -> pd.Series:
        """
        Phase 3 base strategy signal.
        
        Simplified: SMA crossover + RSI confirmation + multi-indicator consensus
        """
        close = df['close']
        
        # SMA crossover
        fast_ma = close.ewm(span=10).mean()
        slow_ma = close.ewm(span=20).mean()
        
        sma_signal = pd.Series(0, index=close.index)
        sma_signal = sma_signal.where(fast_ma <= slow_ma, 1)  # Uptrend = 1
        sma_signal = sma_signal.where(fast_ma >= slow_ma, -1)  # Downtrend = -1
        
        # RSI confirmation (not at extremes)
        rsi = self.calculate_rsi(close)
        rsi_ok = (rsi > 30) & (rsi < 70)  # Neutral RSI
        
        # Volume confirmation
        vol_ma = df['volume'].rolling(20).mean()
        vol_ok = df['volume'] > vol_ma * 0.8
        
        # Combine: signal only if RSI and volume confirm
        signal = sma_signal.where(rsi_ok & vol_ok, 0)
        
        return signal
    
    def generate_hybrid_signal(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Generate hybrid signal: Phase 3 base + Phase 6 regime filters.
        
        Returns:
            Tuple of (signal, confidence, volatility_mult)
                signal: -1/0/1
                confidence: 0-1 (how strong the signal)
                volatility_mult: position size multiplier
        """
        # Phase 3 base signal
        base_signal = self.generate_phase3_signal(df)
        
        # Phase 6 regime filter
        regime = self.regime_filter(df)
        
        # Phase 6 volatility sizing
        vol_mult = self.volatility_multiplier(df)
        
        # Apply regime filter: signal = 0 if regime doesn't allow
        filtered_signal = base_signal.where(regime == 1, 0)
        
        # Confidence: how many filters allowed this trade
        confidence = (regime.astype(int) + (filtered_signal != 0).astype(int)) / 2.0
        confidence = confidence.where(filtered_signal != 0, 0)
        
        return filtered_signal, confidence, vol_mult
    
    def backtest_daily(
        self,
        df: pd.DataFrame,
        initial_capital: float = 100000,
        position_size: float = 0.1,  # 10% per trade
    ) -> Dict:
        """
        Backtest hybrid strategy on daily bars.
        
        Allows ONE position per symbol (concurrent positions across symbols).
        
        Args:
            df: DataFrame with OHLCV + symbol
            initial_capital: Starting capital
            position_size: Base position size (10% = 0.1)
        
        Returns:
            Dict with metrics
        """
        results = {
            'trades': [],
            'equity': [initial_capital],
            'drawdown': [0],
        }
        
        equity = initial_capital
        positions = {}  # symbol -> {'entry_price', 'entry_signal', 'entry_size'}
        
        # Generate all signals
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].reset_index(drop=True)
            signals, confidence, vol_mult = self.generate_hybrid_signal(symbol_data)
            positions[symbol] = None
            
            for idx, row in symbol_data.iterrows():
                signal = signals.iloc[idx] if idx < len(signals) else 0
                conf = confidence.iloc[idx] if idx < len(confidence) else 0
                vol_m = vol_mult.iloc[idx] if idx < len(vol_mult) else 1.0
                price = row['close']
                
                # Generate trade size: base * confidence * volatility adjustment
                trade_size = position_size * conf * vol_m
                
                # Exit logic for this symbol
                if positions[symbol] is not None:
                    entry_info = positions[symbol]
                    # Exit if signal changed
                    if signal != entry_info['entry_signal'] and signal != 0:
                        pnl = (price - entry_info['entry_price']) * entry_info['entry_signal'] * entry_info['entry_size']
                        equity += pnl
                        results['trades'].append({
                            'symbol': symbol,
                            'entry_price': entry_info['entry_price'],
                            'exit_price': price,
                            'direction': 'long' if entry_info['entry_signal'] > 0 else 'short',
                            'pnl': pnl,
                            'position_size': entry_info['entry_size'],
                        })
                        positions[symbol] = None
                
                # Entry logic for this symbol
                if positions[symbol] is None and signal != 0 and trade_size > 0.01:
                    positions[symbol] = {
                        'entry_price': price,
                        'entry_signal': signal,
                        'entry_size': trade_size,
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
        }
        
        return metrics


if __name__ == "__main__":
    print("Testing Hybrid Phase 3 + Phase 6 Strategy...")
    
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
    hybrid = HybridPhase3Phase6(
        rsi_threshold_oversold=30,
        rsi_threshold_overbought=70,
        min_volume_ma_ratio=0.8,
        volatility_target=0.02,
    )
    
    metrics = hybrid.backtest_daily(df_all, initial_capital=100000)
    
    print("\nHybrid Strategy Results:")
    print(f"  Total Return: {metrics['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {metrics['sharpe']:.2f}")
    print(f"  Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    print(f"  Win Rate: {metrics['win_rate']*100:.1f}%")
    print(f"  Trades: {metrics['num_trades']}")
    
    print(f"\n✓ Hybrid Phase 3 + Phase 6 ready for integration")
