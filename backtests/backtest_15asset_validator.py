#!/usr/bin/env python
"""
Multi-Asset Strategy Validator
Tests daily voting strategy across 10+ diverse assets to validate signal quality.

Sectors tested:
- Tech: AAPL, MSFT, NVDA, GOOGL
- Finance: JPM, GS, BAC
- Energy: XOM, CVX
- Pharma: JNJ, PFE
- Retail: AMZN, WMT
- Broad Market: SPY, QQQ

Usage:
    python scripts/phase6_multi_asset_validator.py --output multi_asset_results.json
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import yfinance as yf
from financial_algorithms.strategies.voting_enhanced_weighted import EnhancedWeightedVotingStrategy
from financial_algorithms.backtest.tiered_exits import TieredExitManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SingleAssetBacktest:
    """Run backtest on a single asset."""
    
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
            min_buy_score=2.0,
            max_sell_score=-2.0,
        )
        
        self.exit_manager = TieredExitManager(
            risk_pct=2.0,
            tp1_pct=1.5,
            tp2_pct=3.0,
        )
        
        self.trades = []
    
    def load_data(self) -> pd.DataFrame:
        """Load daily OHLCV data."""
        try:
            df = yf.download(
                self.asset,
                start=self.start_date,
                end=self.end_date,
                interval='1d',
                progress=False,
            )
            
            if df.empty:
                logger.warning(f"{self.asset}: No data available")
                return pd.DataFrame()
            
            df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close',
            }, inplace=True)
            
            df.fillna(method='ffill', inplace=True)
            
            logger.info(f"{self.asset}: Loaded {len(df)} bars")
            return df
        
        except Exception as e:
            logger.error(f"{self.asset}: Failed to load data - {e}")
            return pd.DataFrame()
    
    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest on loaded data."""
        if df.empty or len(df) < 50:
            return {
                'asset': self.asset,
                'status': 'FAILED',
                'reason': 'Insufficient data',
                'start_date': self.start_date,
                'end_date': self.end_date,
            }
        
        equity = self.initial_equity
        position = None
        trades = []
        daily_returns = []
        
        for idx in range(50, len(df)):
            date = df.index[idx]
            row = df.iloc[idx]
            
            close_hist = df['close'].iloc[:idx+1].values
            high_hist = df['high'].iloc[:idx+1].values
            low_hist = df['low'].iloc[:idx+1].values
            volume_hist = df['volume'].iloc[:idx+1].values
            
            # Calculate signal
            score = self.strategy.calculate_voting_score(
                close=close_hist,
                high=high_hist,
                low=low_hist,
                volume=volume_hist,
            )
            
            daily_return = 0.0
            current_price = float(row['close'])
            
            if position is None:
                # Entry check
                if self.strategy.should_enter(score):
                    position_size = self.strategy.calculate_position_size(score, equity)
                    quantity = position_size / current_price if current_price > 0 else 0
                    
                    if quantity > 0:
                        position = self.exit_manager.create_trade(
                            symbol=self.asset,
                            entry_price=current_price,
                            entry_time=date,
                            quantity=quantity,
                        )
                        
                        trades.append({
                            'entry_date': date,
                            'entry_price': current_price,
                            'entry_signal': score,
                            'quantity': quantity,
                            'position_value': position_size,
                        })
            
            else:
                # Exit check
                if position.scenario == 'SCENARIO_B':
                    self.exit_manager.update_trailing_stop(position, current_price)
                
                if not position.tp1_hit:
                    self.exit_manager.evaluate_at_tp1(position, current_price, date, score)
                
                exit_cond = self.exit_manager.check_exit_conditions(
                    position, current_price, date, score
                )
                
                if exit_cond:
                    pl = exit_cond['pl']
                    exit_price = exit_cond['exit_price']
                    
                    equity += pl
                    daily_return = (exit_price - position.entry_price) / position.entry_price
                    
                    trades[-1].update({
                        'exit_date': date,
                        'exit_price': exit_price,
                        'exit_reason': exit_cond['exit_reason'],
                        'pl': pl,
                        'return_pct': daily_return * 100,
                    })
                    
                    position = None
            
            daily_returns.append(daily_return)
        
        # Calculate metrics
        if len(trades) > 0:
            winning = sum(1 for t in trades if t.get('return_pct', 0) > 0)
            win_rate = winning / len(trades) * 100
            
            returns = np.array([t.get('return_pct', 0) / 100 for t in trades if 'return_pct' in t])
            avg_return = np.mean(returns) if len(returns) > 0 else 0
            
            if np.std(returns) > 0:
                sharpe = (avg_return / np.std(returns)) * np.sqrt(252)
            else:
                sharpe = 0
            
            total_pl = sum(t.get('pl', 0) for t in trades)
            total_return = (total_pl / self.initial_equity) * 100
            best = max([t.get('return_pct', 0) for t in trades])
            worst = min([t.get('return_pct', 0) for t in trades])
            
            max_dd = self._calculate_max_drawdown(daily_returns)
        else:
            win_rate = 0
            sharpe = 0
            total_return = 0
            total_pl = 0
            avg_return = 0
            best = 0
            worst = 0
            max_dd = 0
        
        results = {
            'asset': self.asset,
            'status': 'SUCCESS',
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_pl': total_pl,
            'total_return_pct': total_return,
            'trade_count': len(trades),
            'win_rate_pct': win_rate,
            'avg_trade_return_pct': avg_return * 100,
            'sharpe_ratio': sharpe,
            'best_trade_pct': best,
            'worst_trade_pct': worst,
            'max_drawdown_pct': max_dd,
        }
        
        logger.info(
            f"{self.asset}: Return={total_return:.2f}% | Sharpe={sharpe:.2f} | "
            f"WR={win_rate:.1f}% | Trades={len(trades)}"
        )
        
        return results
    
    @staticmethod
    def _calculate_max_drawdown(returns: List[float]) -> float:
        """Calculate maximum drawdown from returns list."""
        if not returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return float(np.min(drawdown) * 100) if len(drawdown) > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description='Multi-asset daily strategy validator')
    parser.add_argument(
        '--assets',
        type=str,
        default='AAPL,MSFT,NVDA,GOOGL,JPM,GS,BAC,XOM,CVX,JNJ,PFE,AMZN,WMT,SPY,QQQ',
        help='Comma-separated asset list'
    )
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date')
    parser.add_argument('--end-date', type=str, default='2025-12-31', help='End date')
    parser.add_argument('--output', type=str, default='multi_asset_results.json', help='Output file')
    
    args = parser.parse_args()
    
    assets = [a.strip().upper() for a in args.assets.split(',')]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Multi-Asset Daily Strategy Validator")
    logger.info(f"Testing {len(assets)} assets from {args.start_date} to {args.end_date}")
    logger.info(f"{'='*60}\n")
    
    all_results = {
        'test_date': datetime.now().isoformat(),
        'start_date': args.start_date,
        'end_date': args.end_date,
        'assets_tested': len(assets),
        'results': [],
        'summary': {},
    }
    
    for asset in assets:
        backtest = SingleAssetBacktest(
            asset=asset,
            start_date=args.start_date,
            end_date=args.end_date,
        )
        
        df = backtest.load_data()
        result = backtest.run(df)
        all_results['results'].append(result)
    
    # Calculate summary
    successful = [r for r in all_results['results'] if r.get('status') == 'SUCCESS']
    
    if successful:
        avg_return = np.mean([r['total_return_pct'] for r in successful])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in successful])
        avg_wr = np.mean([r['win_rate_pct'] for r in successful])
        avg_trades = np.mean([r['trade_count'] for r in successful])
        
        all_results['summary'] = {
            'successful_assets': len(successful),
            'avg_return_pct': avg_return,
            'avg_sharpe': avg_sharpe,
            'avg_win_rate_pct': avg_wr,
            'avg_trades_per_asset': avg_trades,
            'best_asset': max(successful, key=lambda x: x['total_return_pct'])['asset'],
            'best_return_pct': max([r['total_return_pct'] for r in successful]),
        }
    
    # Save results
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Results Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Successful: {len(successful)}/{len(assets)}")
    if successful:
        logger.info(f"Average Return: {all_results['summary']['avg_return_pct']:.2f}%")
        logger.info(f"Average Sharpe: {all_results['summary']['avg_sharpe']:.2f}")
        logger.info(f"Average Win Rate: {all_results['summary']['avg_win_rate_pct']:.1f}%")
        logger.info(f"Best Asset: {all_results['summary']['best_asset']} ({all_results['summary']['best_return_pct']:.2f}%)")
    
    logger.info(f"\nResults saved to {args.output}\n")


if __name__ == '__main__':
    main()
