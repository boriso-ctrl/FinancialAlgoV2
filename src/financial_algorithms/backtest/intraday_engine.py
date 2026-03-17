"""Intraday backtest engine optimized for 1-minute to 15-minute bar trading."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import warnings


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class PositionMode(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    """Record of a completed trade."""
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    mode: PositionMode
    entry_signal: float  # -1, 0, 1
    exit_signal: float   # How we exited
    pnl: float = field(init=False)
    pnl_pct: float = field(init=False)
    bars_held: int = field(init=False)
    
    def __post_init__(self):
        if self.mode == PositionMode.LONG:
            self.pnl = (self.exit_price - self.entry_price) * self.size
            self.pnl_pct = (self.exit_price - self.entry_price) / self.entry_price
        elif self.mode == PositionMode.SHORT:
            self.pnl = (self.entry_price - self.exit_price) * self.size
            self.pnl_pct = (self.entry_price - self.exit_price) / self.entry_price
        else:
            self.pnl = 0
            self.pnl_pct = 0
        
        bar_diff = (self.exit_time - self.entry_time).total_seconds() // 60
        self.bars_held = max(1, int(bar_diff))


@dataclass
class Position:
    """Current open position."""
    symbol: str
    mode: PositionMode = PositionMode.FLAT
    entry_price: float = 0.0
    entry_time: pd.Timestamp = None
    size: float = 0.0
    entry_signal: float = 0.0
    
    def is_open(self) -> bool:
        return self.mode != PositionMode.FLAT


class IntradayBacktest:
    """
    Event-driven backtester for intraday trading on 1-minute to 15-minute bars.
    
    Features:
    - Per-bar position entry/exit
    - Realistic slippage (wider for intraday)
    - Multi-symbol concurrent positions
    - Win/loss tracking
    - Sharpe ratio calculation
    
    Usage:
        bt = IntradayBacktest(initial_capital=100000, slippage_bps=2)
        bt.backtest(df_bars, signal_func)
        stats = bt.calculate_metrics()
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        slippage_bps: float = 2.0,
        commission_pct: float = 0.01,
        max_positions: int = 10,
    ):
        """
        Args:
            initial_capital: Starting cash
            slippage_bps: Entry/exit slippage in basis points (2 bps = 0.02%)
            commission_pct: Trading commission as %
            max_positions: Max concurrent positions
        """
        self.initial_capital = initial_capital
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        self.commission_pct = commission_pct / 100
        self.max_positions = max_positions
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.bar_index = []
        
        # Stats
        self.total_pnl = 0
        self.num_trades = 0
        self.num_wins = 0
        self.num_losses = 0
    
    def _apply_slippage(self, price: float, direction: int) -> float:
        """Apply slippage to entry/exit price.
        
        Args:
            price: Reference price
            direction: 1 for buy, -1 for sell
        
        Returns:
            Price with slippage applied
        """
        slippage_amount = price * self.slippage_bps * direction
        return price + slippage_amount
    
    def _calculate_available_cash(self) -> float:
        """Calculate cash not tied up in positions."""
        position_value = sum(
            pos.size * pos.entry_price if pos.mode == PositionMode.LONG else -pos.size * pos.entry_price
            for pos in self.positions.values()
        )
        return self.cash - position_value
    
    def _try_entry(
        self,
        symbol: str,
        signal: float,
        price: float,
        size: float,
        timestamp: pd.Timestamp,
    ) -> bool:
        """Attempt to open a new position.
        
        Returns:
            True if entry successful, False otherwise
        """
        # Check if already in position
        if symbol in self.positions and self.positions[symbol].is_open():
            return False
        
        # Check max positions limit
        open_positions = sum(1 for p in self.positions.values() if p.is_open())
        if open_positions >= self.max_positions:
            return False
        
        # Check cash availability
        entry_cost = size * price * (1 + self.commission_pct)
        available = self._calculate_available_cash()
        if entry_cost > available:
            size = available / (price * (1 + self.commission_pct))
            if size < 0.001:  # Minimum 0.1% of capital
                return False
        
        # Apply slippage
        direction = 1 if signal > 0 else -1
        entry_price = self._apply_slippage(price, direction)
        
        # Create position
        mode = PositionMode.LONG if signal > 0 else PositionMode.SHORT
        self.positions[symbol] = Position(
            symbol=symbol,
            mode=mode,
            entry_price=entry_price,
            entry_time=timestamp,
            size=size,
            entry_signal=signal,
        )
        
        return True
    
    def _try_exit(
        self,
        symbol: str,
        signal: float,
        price: float,
        timestamp: pd.Timestamp,
    ) -> bool:
        """Attempt to close an open position.
        
        Returns:
            True if exit successful, False otherwise
        """
        if symbol not in self.positions:
            return False
        
        pos = self.positions[symbol]
        if not pos.is_open():
            return False
        
        # Exit if signal disagrees with position
        exit_trigger = (pos.mode == PositionMode.LONG and signal < 0) or \
                       (pos.mode == PositionMode.SHORT and signal > 0)
        
        if not exit_trigger and signal != 0:
            return False  # Wrong signal
        
        # Apply slippage
        direction = -1 if pos.mode == PositionMode.LONG else 1
        exit_price = self._apply_slippage(price, direction)
        
        # Close position and record trade
        trade = Trade(
            symbol=symbol,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size=pos.size,
            mode=pos.mode,
            entry_signal=pos.entry_signal,
            exit_signal=signal,
        )
        
        self.trades.append(trade)
        self.total_pnl += trade.pnl
        self.num_trades += 1
        if trade.pnl > 0:
            self.num_wins += 1
        else:
            self.num_losses += 1
        
        # Remove position
        del self.positions[symbol]
        
        return True
    
    def backtest(
        self,
        df_bars: pd.DataFrame,
        signal_func,
        close_on_bar_n: Optional[int] = None,
        min_history: int = 50,
    ) -> None:
        """Run backtest with WALK-FORWARD analysis (no lookahead bias).
        
        Args:
            df_bars: OHLCV bars with columns [timestamp, open, high, low, close, volume, symbol]
            signal_func: Function that takes historical df (only past bars), returns signal for CURRENT bar
            close_on_bar_n: Auto-close positions after N bars held
            min_history: Minimum bars required before generating signals (warm-up period)
        """
        # Sort by timestamp to ensure chronological order
        df_bars = df_bars.sort_values('timestamp').reset_index(drop=True)
        
        # Group by symbol
        symbols = sorted(df_bars['symbol'].unique())
        symbol_start_idx = {sym: df_bars[df_bars['symbol'] == sym].index[0] 
                           for sym in symbols}
        
        # Walk-forward: process bars one at a time
        for idx in range(len(df_bars)):
            row = df_bars.iloc[idx]
            timestamp = row['timestamp']
            price = row['close']
            symbol = row['symbol']
            
            # Count bars for this symbol only
            bars_so_far = idx - symbol_start_idx[symbol] + 1
            
            # Compute signal using ONLY historical data (up to current bar, exclusive)
            if bars_so_far >= min_history:
                # Create historical slice: bars for this symbol up to current bar
                hist_data = df_bars[(df_bars['symbol'] == symbol) & (df_bars.index <= idx)]
                
                try:
                    # signal_func processes only historical bars, returns signal for CURRENT bar
                    signal = signal_func(hist_data)
                except Exception as e:
                    signal = 0  # Skip signal if error
            else:
                signal = 0  # Warm-up period: no signals yet
            
            # Store equity before action
            self._update_equity(timestamp, price)
            
            # Position management
            if symbol in self.positions and self.positions[symbol].is_open():
                pos = self.positions[symbol]
                
                # Auto-exit after N bars
                if close_on_bar_n is not None:
                    bars_held = (timestamp - pos.entry_time).total_seconds() // 60
                    if bars_held >= close_on_bar_n:
                        self._try_exit(symbol, signal, price, timestamp)
                
                # Manual exit on opposite signal
                elif signal != pos.entry_signal:
                    self._try_exit(symbol, signal, price, timestamp)
            
            # Try entry on new signal
            if signal != 0:
                if symbol not in self.positions or not self.positions[symbol].is_open():
                    self._try_entry(symbol, signal, price, 0.1, timestamp)  # 10% position
    
    def _update_equity(self, timestamp: pd.Timestamp, price: float):
        """Update equity curve with current portfolio value."""
        position_value = 0
        for pos in self.positions.values():
            if pos.is_open():
                if pos.mode == PositionMode.LONG:
                    position_value += pos.size * price
                else:
                    position_value -= pos.size * price
        
        equity = self.cash + position_value + self.total_pnl
        self.equity_curve.append(equity)
        self.bar_index.append(timestamp)
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.equity_curve) == 0:
            return {
                'Total Return %': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown %': 0,
                'Win Rate %': 0,
                'Total Trades': 0,
            }
        
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        # Total return
        total_return = (equity_series.iloc[-1] - self.initial_capital) / self.initial_capital * 100
        
        # Sharpe ratio (annualized, assuming 252*60 bars per year)
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252 * 60)
        else:
            sharpe = 0
        
        # Max drawdown
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax
        max_dd = drawdown.min() * 100
        
        # Win rate
        win_rate = self.num_wins / max(1, self.num_trades) * 100 if self.num_trades > 0 else 0
        
        return {
            'Total Return %': round(total_return, 2),
            'Sharpe Ratio': round(sharpe, 2),
            'Max Drawdown %': round(max_dd, 2),
            'Win Rate %': round(win_rate, 1),
            'Total Trades': self.num_trades,
            'Winning Trades': self.num_wins,
            'Losing Trades': self.num_losses,
            'Avg Trade PnL': round(self.total_pnl / max(1, self.num_trades), 2),
        }
    
    def get_equity_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        return pd.DataFrame({
            'timestamp': self.bar_index,
            'equity': self.equity_curve,
        })
    
    def get_trades_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'symbol': trade.symbol,
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'size': trade.size,
                'mode': trade.mode.name,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct * 100,
                'bars_held': trade.bars_held,
            })
        
        return pd.DataFrame(data)


if __name__ == "__main__":
    print("Testing intraday backtest engine...")
    
    # Create synthetic 1-min OHLCV for 2 symbols
    np.random.seed(42)
    dates = pd.date_range('2026-03-11 09:30', periods=390, freq='1min')  # Trading day
    
    data = []
    for symbol in ['AAPL', 'MSFT']:
        prices = 100 + np.cumsum(np.random.normal(0, 0.2, 390))
        for i in range(len(dates)):
            data.append({
                'timestamp': dates[i],
                'symbol': symbol,
                'open': prices[i] - 0.5,
                'high': prices[i] + 1,
                'low': prices[i] - 1,
                'close': prices[i],
                'volume': np.random.uniform(10000, 100000),
            })
    
    df_bars = pd.DataFrame(data)
    
    # Simple SMA crossover signal
    def sma_signal(df):
        signals = pd.Series(0, index=df.index)
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            close = df.loc[mask, 'close'].reset_index(drop=True)
            fast_ma = close.ewm(span=10).mean()
            slow_ma = close.ewm(span=20).mean()
            sig = (fast_ma > slow_ma).astype(int) * 2 - 1  # Convert to 1/-1
            signals[mask] = sig.values
        return signals
    
    # Run backtest
    bt = IntradayBacktest(initial_capital=100000)
    bt.backtest(df_bars, sma_signal, close_on_bar_n=15)  # Close after 15 bars
    
    # Results
    metrics = bt.calculate_metrics()
    print("\nBacktest Results:")
    for key, val in metrics.items():
        print(f"  {key}: {val}")
    
    if bt.trades:
        print(f"\nFirst 5 trades:")
        print(bt.get_trades_df().head())
    
    print(f"\n✓ Intraday backtest engine ready")
