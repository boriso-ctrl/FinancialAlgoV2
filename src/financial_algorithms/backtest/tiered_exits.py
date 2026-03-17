"""
Tiered exit logic with scenario-based decision making and ratcheting stops.

Implements:
- TP1 scenario analysis (Scenario A: weak signals, Scenario B: strong signals)
- Ratcheting trailing stops (move up only, never down)
- R/R 1:3 with partial profit-taking at TP1 (1:1.5)
- Dynamic stop loss management based on signal strength
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ExitScenario(Enum):
    """Exit scenario types."""
    SCENARIO_A = "WEAK_AT_TP1"  # Exit 50%, protect with reduced SL
    SCENARIO_B = "STRONG_AT_TP1"  # Keep all, trail to TP2
    SIGNAL_EXIT = "SELL_SIGNAL"  # Score hit sell threshold
    TRAILING_SL_HIT = "TRAILING_SL_HIT"  # Trailing stop hit
    SL_HIT = "SL_HIT"  # Static stop loss hit
    TP2_HIT = "TP2_HIT"  # Full profit target hit


@dataclass
class Trade:
    """Represents an open trade with exit levels and scenario."""
    symbol: str
    entry_price: float
    entry_time: pd.Timestamp
    quantity: float
    
    # Exit levels
    sl: float
    tp1: float
    tp2: float
    
    # Scenario and current state
    scenario: Optional[ExitScenario] = None
    current_trailing_sl: Optional[float] = None
    tp1_hit: bool = False
    tp1_hit_time: Optional[pd.Timestamp] = None
    
    high_water_mark: float = None  # For ratcheting stop
    scenario_a_protective_sl: Optional[float] = None
    
    def __post_init__(self):
        if self.high_water_mark is None:
            self.high_water_mark = self.entry_price


class TieredExitManager:
    """
    Manages tiered exits with scenario-based stop loss adjustments.
    
    R/R Strategy:
    - Risk: 2% (SL at entry - 2%)
    - TP1: 1.5% (50% position) = 3:1 ratio to SL
    - TP2: 3.0% (remaining 50%) = 6:1 ratio to SL
    
    Scenario A (Signals Weakening at TP1):
    Entry: $100, SL: $98
    TP1: $101.50 (exit 50%)
    New SL for remaining: $100.75 (protect to breakeven + 0.75%)
    
    Scenario B (Signals Strong at TP1):
    Entry: $100, SL: $98
    TP1: $101.50 (hold all)
    New SL: $100.485 (TP1 * 0.99) - trailing at 1%
    TP2 target: $103
    """
    
    def __init__(
        self,
        risk_pct: float = 2.0,
        tp1_pct: float = 1.5,
        tp2_pct: float = 3.0,
        trailing_stop_pct: float = 1.0,
        breakout_level_pct: float = 20.0,  # At TP1 + 20%, upgrade to 1% trail
    ):
        self.risk_pct = risk_pct
        self.tp1_pct = tp1_pct
        self.tp2_pct = tp2_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.breakout_level_pct = breakout_level_pct
        
        self.trades: Dict[str, List[Trade]] = {}
    
    def create_trade(
        self,
        symbol: str,
        entry_price: float,
        entry_time: pd.Timestamp,
        quantity: float,
    ) -> Trade:
        """
        Create a new trade with initial exit levels.
        
        Returns:
            Trade object with SL, TP1, TP2 calculated
        """
        sl = entry_price * (1 - self.risk_pct / 100)
        tp1 = entry_price * (1 + self.tp1_pct / 100)
        tp2 = entry_price * (1 + self.tp2_pct / 100)
        
        trade = Trade(
            symbol=symbol,
            entry_price=entry_price,
            entry_time=entry_time,
            quantity=quantity,
            sl=sl,
            tp1=tp1,
            tp2=tp2,
            high_water_mark=entry_price,
        )
        
        if symbol not in self.trades:
            self.trades[symbol] = []
        
        self.trades[symbol].append(trade)
        
        logger.info(
            f"Created trade: {symbol} @ {entry_price:.2f} | "
            f"SL: {sl:.2f} | TP1: {tp1:.2f} | TP2: {tp2:.2f}"
        )
        
        return trade
    
    def evaluate_at_tp1(
        self,
        trade: Trade,
        current_price: float,
        current_time: pd.Timestamp,
        signal_score: float,
        signal_strength_threshold: float = 7.0,
    ) -> Dict[str, any]:
        """
        Evaluate trade at TP1 to determine scenario.
        
        Scenario A: Score < threshold → weak signals = exit 50%, protect 50%
        Scenario B: Score >= threshold → strong signals = keep all, trail
        
        Args:
            trade: Trade object
            current_price: Current market price
            current_time: Current timestamp
            signal_score: Current voting score (-10 to +10)
            signal_strength_threshold: Score level to consider "strong"
        
        Returns:
            Dict with scenario, actions, new_sl, etc.
        """
        if trade.tp1_hit:
            return {'status': 'ALREADY_AT_TP1'}
        
        if current_price < trade.tp1:
            return {'status': 'NOT_AT_TP1_YET'}
        
        # Mark TP1 as hit
        trade.tp1_hit = True
        trade.tp1_hit_time = current_time
        
        # Decide scenario based on signal strength
        if signal_score >= signal_strength_threshold:
            return self._apply_scenario_b(trade, current_price, current_time)
        else:
            return self._apply_scenario_a(trade, current_price)
    
    def _apply_scenario_a(
        self,
        trade: Trade,
        current_price: float,
    ) -> Dict[str, any]:
        """
        Scenario A: Signals weakening at TP1.
        
        Exit 50% immediately at TP1 price.
        Protect remaining 50% with reduced SL at TP1 * 0.5.
        
        For $100 entry, TP1 $101.50:
        - Exit 50% at $101.50 (lock in $0.75)
        - Remaining 50% SL at $100.75 (protect to breakeven + half profit)
        """
        exit_qty = trade.quantity * 0.5
        protective_sl = trade.entry_price * (1 + self.tp1_pct * 0.5 / 100)
        
        trade.scenario = ExitScenario.SCENARIO_A
        trade.scenario_a_protective_sl = protective_sl
        
        logger.info(
            f"Scenario A (Weak Signals): Exit 50% ({exit_qty:.0f}) @ {trade.tp1:.2f} | "
            f"Protect 50% with SL @ {protective_sl:.2f}"
        )
        
        return {
            'scenario': 'A_WEAK_SIGNALS',
            'exit_qty': exit_qty,
            'exit_price': trade.tp1,
            'exit_reason': 'TP1_REACHED',
            'remaining_qty': trade.quantity * 0.5,
            'new_sl': protective_sl,
            'action': 'EXIT_HALF_PROTECT_HALF',
            'profit_locked': exit_qty * (trade.tp1 - trade.entry_price),
        }
    
    def _apply_scenario_b(
        self,
        trade: Trade,
        current_price: float,
        current_time: pd.Timestamp,
    ) -> Dict[str, any]:
        """
        Scenario B: Signals strong at TP1.
        
        Keep all position. Start trailing SL at TP1 * 0.99.
        When price > TP1 * 1.20 (breakout), ratchet to 1% trailing.
        
        For $100 entry, TP1 $101.50:
        - Keep all at TP1 (don't exit)
        - Initial SL: $100.485 (TP1 * 0.99)
        - If price > $101.80 (TP1 * 1.20): upgrade to 1% trailing
        """
        initial_trailing_sl = trade.tp1 * (1 - self.trailing_stop_pct / 100)
        trade.current_trailing_sl = initial_trailing_sl
        trade.scenario = ExitScenario.SCENARIO_B
        
        logger.info(
            f"Scenario B (Strong Signals): Keep all @ TP1 | "
            f"Initial trailing SL @ {initial_trailing_sl:.2f} | "
            f"Breakout at {trade.tp1 * (1 + self.breakout_level_pct/100):.2f}"
        )
        
        return {
            'scenario': 'B_STRONG_SIGNALS',
            'action': 'KEEP_ALL_TRAIL',
            'exit_qty': 0,
            'trailing_sl': initial_trailing_sl,
            'tp_target': trade.tp2,
            'breakout_level': trade.tp1 * (1 + self.breakout_level_pct / 100),
            'notes': 'Ratchets UP only, never DOWN',
        }
    
    def update_trailing_stop(
        self,
        trade: Trade,
        current_price: float,
    ) -> Optional[Dict[str, any]]:
        """
        Update ratcheting trailing stop after TP1 in Scenario B.
        
        Rules:
        1. SL moves UP with price at 1% distance
        2. SL NEVER moves DOWN (ratchets up only)
        3. If price > TP1 + 20%, upgrade to 1% trail
        
        Returns:
            Dict if SL was updated, None otherwise
        """
        if trade.scenario != ExitScenario.SCENARIO_B or not trade.tp1_hit:
            return None
        
        if trade.current_trailing_sl is None:
            trade.current_trailing_sl = trade.tp1 * (1 - self.trailing_stop_pct / 100)
        
        # Calculate new SL at 1% below current price
        new_trailing_sl = current_price * (1 - self.trailing_stop_pct / 100)
        
        # Only update if new SL is HIGHER (ratchets UP only)
        if new_trailing_sl > trade.current_trailing_sl:
            old_sl = trade.current_trailing_sl
            trade.current_trailing_sl = new_trailing_sl
            trade.high_water_mark = current_price
            
            return {
                'status': 'UPDATED',
                'old_sl': old_sl,
                'new_sl': new_trailing_sl,
                'price': current_price,
                'moved_up_by': new_trailing_sl - old_sl,
            }
        
        return {
            'status': 'NO_CHANGE',
            'current_sl': trade.current_trailing_sl,
            'price': current_price,
        }
    
    def check_exit_conditions(
        self,
        trade: Trade,
        current_price: float,
        current_time: pd.Timestamp,
        current_signal_score: float,
    ) -> Optional[Dict[str, any]]:
        """
        Check all exit conditions for a trade.
        
        Returns:
            Dict with exit details if condition met, None otherwise
        """
        # Check static SL
        if trade.scenario == ExitScenario.SCENARIO_A and trade.scenario_a_protective_sl:
            if current_price <= trade.scenario_a_protective_sl:
                return {
                    'exit_scenario': ExitScenario.SL_HIT,
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'exit_reason': 'Protective SL hit (Scenario A)',
                    'pl': trade.quantity * (current_price - trade.entry_price),
                }
        
        # Check TP2
        if current_price >= trade.tp2:
            return {
                'exit_scenario': ExitScenario.TP2_HIT,
                'exit_price': current_price,
                'exit_time': current_time,
                'exit_reason': 'TP2 (3%) reached',
                'pl': trade.quantity * (current_price - trade.entry_price),
            }
        
        # Check trailing SL (Scenario B)
        if trade.scenario == ExitScenario.SCENARIO_B and trade.current_trailing_sl:
            if current_price <= trade.current_trailing_sl:
                return {
                    'exit_scenario': ExitScenario.TRAILING_SL_HIT,
                    'exit_price': current_price,
                    'exit_time': current_time,
                    'exit_reason': 'Trailing SL hit',
                    'pl': trade.quantity * (current_price - trade.entry_price),
                }
        
        # Check sell signal (score <= -5)
        if current_signal_score <= -5:
            return {
                'exit_scenario': ExitScenario.SIGNAL_EXIT,
                'exit_price': current_price,
                'exit_time': current_time,
                'exit_reason': f'Sell signal (score {current_signal_score:.1f})',
                'pl': trade.quantity * (current_price - trade.entry_price),
            }
        
        return None
    
    def simulate_daily_bar(
        self,
        trade: Trade,
        day_open: float,
        day_high: float,
        day_low: float,
        day_close: float,
        day_time: pd.Timestamp,
        signal_score: float,
    ) -> List[Dict[str, any]]:
        """
        Simulate a full trading day for a position.
        
        Returns:
            List of exit events if any occurred, empty otherwise
        """
        exits = []
        
        # Check if TP1 hit during day
        if day_high >= trade.tp1 and not trade.tp1_hit:
            scenario_result = self.evaluate_at_tp1(
                trade, day_close, day_time, signal_score
            )
            if scenario_result.get('status') != 'ALREADY_AT_TP1':
                logger.debug(f"TP1 hit: {scenario_result}")
        
        # Update trailing stop if in Scenario B
        if trade.scenario == ExitScenario.SCENARIO_B and day_high > trade.current_trailing_sl or 0:
            trail_update = self.update_trailing_stop(trade, day_close)
            if trail_update:
                logger.debug(f"Trailing stop update: {trail_update}")
        
        # Check exit conditions
        exit_cond = self.check_exit_conditions(
            trade, day_close, day_time, signal_score
        )
        
        if exit_cond:
            exits.append(exit_cond)
        
        return exits
    
    def get_all_open_trades(self) -> List[Trade]:
        """Return all open trades across all symbols."""
        all_trades = []
        for trades_list in self.trades.values():
            all_trades.extend(trades_list)
        return all_trades
    
    def close_trade_for_stats(self, trade: Trade, exit_price: float, exit_time: pd.Timestamp, pl: float):
        """Mark trade as closed for reporting."""
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.pl = pl
        trade.return_pct = (pl / (trade.entry_price * trade.quantity)) * 100 if trade.entry_price > 0 else 0
