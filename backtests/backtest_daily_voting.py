#!/usr/bin/env python
"""
Phase 6: Parallel Single-Asset Backtest Runner
Tests enhanced weighted voting system with tiered exits on a single asset.

Usage:
    python scripts/phase6_weighted_parallel.py --asset AAPL --output results_aapl.json
    python scripts/phase6_weighted_parallel.py --asset SPY --output results_spy.json
    python scripts/phase6_weighted_parallel.py --asset QQQ --output results_qqq.json
    python scripts/phase6_weighted_parallel.py --asset TSLA --output results_tsla.json

Expected output: JSON with Sharpe, return %, trades count, win rate
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_enhanced_weighted import EnhancedWeightedVotingStrategy
from financial_algorithms.signals.enhanced_indicators import EnhancedIndicators
from financial_algorithms.backtest.tiered_exits import TieredExitManager, ExitScenario

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Phase6Backtest:
    """Single-asset backtest for enhanced weighted voting system."""
    
    def __init__(
        self,
        asset: str,
        start_date: str = '2023-01-01',
        end_date: str = '2025-12-31',
        initial_equity: float = 100000,
    ):
        self.asset = asset
        self.start_date = start_date
        self.end_date = end_date
        self.initial_equity = initial_equity
        
        self.strategy = EnhancedWeightedVotingStrategy(
            risk_pct=2.0,
            tp1_pct=1.5,
            tp2_pct=3.0,
            min_buy_score=2.0,  # Back to baseline
            max_sell_score=-2.0,  # Back to baseline
        )
        
        self.exit_manager = TieredExitManager(
            risk_pct=2.0,
            tp1_pct=1.5,
            tp2_pct=3.0,
            trailing_stop_pct=1.0,
        )
        
        self.trades_log = []
        self.equity_curve = []
        
        logger.info(f"Initialized Phase6 backtest for {asset}")
    
    def load_data(self) -> pd.DataFrame:
        """Load OHLCV data from yfinance."""
        logger.info(f"Loading {self.asset} data from {self.start_date} to {self.end_date}")
        
        df = yf.download(
            self.asset,
            start=self.start_date,
            end=self.end_date,
            interval='1d',
            progress=False,
        )
        
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
        })
        
        # Forward fill any NaNs
        df = df.fillna(method='ffill')
        
        logger.info(f"Loaded {len(df)} bars for {self.asset}")
        return df
    
    def backtest(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Run backtest on OHLCV data.
        
        Returns:
            Dict with metrics: sharpe, total_return, trades, win_rate, etc.
        """
        logger.info(f"Starting backtest on {len(df)} bars")
        
        equity = self.initial_equity
        position = None
        trades = []
        daily_returns = []
        score_history = []
        
        for idx in range(50, len(df)):  # Need 50 bars for indicators
            date = df.index[idx]
            row = df.iloc[idx]
            
            close_history = df['close'].iloc[:idx+1].values
            high_history = df['high'].iloc[:idx+1].values
            low_history = df['low'].iloc[:idx+1].values
            volume_history = df['volume'].iloc[:idx+1].values
            
            # Calculate signal
            score = self.strategy.calculate_voting_score(
                close=close_history,
                high=high_history,
                low=low_history,
                volume=volume_history,
            )
            score_history.append(score)
            
            daily_return = 0.0
            
            if position is None:
                # Check for entry
                if self.strategy.should_enter(score):
                    entry_price = float(row['close'])
                    position_size = self.strategy.calculate_position_size(score, equity)
                    quantity = position_size / entry_price if entry_price > 0 else 0
                    
                    if quantity > 0:
                        position = self.exit_manager.create_trade(
                            symbol=self.asset,
                            entry_price=entry_price,
                            entry_time=date,
                            quantity=quantity,
                        )
                        
                        trades.append({
                            'entry_date': date,
                            'entry_price': entry_price,
                            'entry_signal': score,
                            'quantity': quantity,
                            'position_value': position_size,
                        })
            
            else:
                # Update trailing stop if in Scenario B
                if position.scenario == ExitScenario.SCENARIO_B:
                    self.exit_manager.update_trailing_stop(position, float(row['close']))
                
                # Check for TP1
                if not position.tp1_hit:
                    self.exit_manager.evaluate_at_tp1(
                        position, float(row['close']), date, score
                    )
                
                # Check for exit
                exit_cond = self.exit_manager.check_exit_conditions(
                    position, float(row['close']), date, score
                )
                
                if exit_cond:
                    exit_price = exit_cond['exit_price']
                    pl = exit_cond['pl']
                    
                    equity += pl
                    daily_return = (exit_price - position.entry_price) / position.entry_price
                    
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': exit_price,
                        'exit_reason': exit_cond['exit_reason'],
                        'pl': pl,
                        'return_pct': daily_return * 100,
                        'scenario': position.scenario.value if position.scenario else 'UNKNOWN',
                    })
                    
                    logger.info(
                        f"{self.asset} | {date.date()} | "
                        f"Entry: {position.entry_price:.2f} → Exit: {exit_price:.2f} | "
                        f"PL: ${pl:.2f} | Return: {daily_return*100:.2f}% | "
                        f"Reason: {exit_cond['exit_reason']}"
                    )
                    
                    position = None
            
            if position is not None:
                # Mark-to-market for equity curve
                current_price = float(row['close'])
                mark_pl = (current_price - position.entry_price) * position.quantity
                current_equity = self.initial_equity - position.entry_price * position.quantity + current_price * position.quantity
            else:
                current_equity = equity
                mark_pl = 0
            
            daily_returns.append(daily_return)
            self.equity_curve.append(current_equity)
        
        # Metrics calculation
        logger.info(f"Trade count: {len(trades)}")
        
        # Debug: Score statistics
        if score_history:
            logger.info(f"Score stats - Min: {min(score_history):.2f}, Max: {max(score_history):.2f}, Mean: {np.mean(score_history):.2f}")
            above_5 = sum(1 for s in score_history if s >= 5)
            logger.info(f"Scores >= +5: {above_5}/{len(score_history)}")
        
        if len(trades) > 0:
            winning_trades = [t for t in trades if t.get('return_pct', 0) > 0]
            win_rate = len(winning_trades) / len(trades) * 100
            
            returns = np.array([t.get('return_pct', 0) / 100 for t in trades if 'return_pct' in t])
            avg_return = np.mean(returns) if len(returns) > 0 else 0
            sharpe = (avg_return / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            total_pl = sum(t.get('pl', 0) for t in trades)
            total_return = (total_pl / self.initial_equity) * 100
        else:
            win_rate = 0
            sharpe = 0
            total_return = 0
            total_pl = 0
        
        results = {
            'asset': self.asset,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_equity': self.initial_equity,
            'final_equity': equity,
            'total_pl': total_pl,
            'total_return_pct': total_return,
            'trade_count': len(trades),
            'win_rate_pct': win_rate,
            'avg_trade_return_pct': avg_return * 100 if len(trades) > 0 else 0,
            'sharpe_ratio': sharpe,
            'best_trade_pct': max([t.get('return_pct', 0) for t in trades], default=0),
            'worst_trade_pct': min([t.get('return_pct', 0) for t in trades], default=0),
            'trades': trades,
            'timestamp': datetime.now().isoformat(),
        }
        
        logger.info(f"Backtest complete: Sharpe={sharpe:.2f}, Return={total_return:.2f}%, Trades={len(trades)}, WR={win_rate:.1f}%")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Phase 6 Single-Asset Backtest')
    parser.add_argument('--asset', required=True, help='Asset ticker (AAPL, SPY, QQQ, TSLA)')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--start-date', default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2025-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--equity', type=float, default=100000, help='Initial equity')
    
    args = parser.parse_args()
    
    # Run backtest
    backtest = Phase6Backtest(
        asset=args.asset,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_equity=args.equity,
    )
    
    df = backtest.load_data()
    results = backtest.backtest(df)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print(f"BACKTEST SUMMARY: {args.asset}")
    print("="*70)
    print(f"  Return: {results['total_return_pct']:>8.2f}%")
    print(f"  Sharpe: {results['sharpe_ratio']:>8.2f}")
    print(f"  Trades: {results['trade_count']:>8}")
    print(f"  Win Rate: {results['win_rate_pct']:>6.1f}%")
    print(f"  Best Trade: {results['best_trade_pct']:>6.2f}%")
    print(f"  Worst Trade: {results['worst_trade_pct']:>6.2f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
