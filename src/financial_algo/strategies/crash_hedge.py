"""Category D: General Crisis Alpha — Crash-Hedge strategies.

Ported and evolved from Financial-Algorithms ``strat_crash_hedged``
(vol-regime switching) and ``strategy_4state`` (4-state tactical allocation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import ema, realized_vol
from financial_algo.regimes import Regime
from financial_algo.strategies.base import Strategy


# =========================================================================
# D1 — 4-State Tactical Allocation
# =========================================================================

@dataclass
class FourStateConfig:
    """Thresholds for the 4-state tactical allocator."""

    leverage_optimal: float = 2.0
    leverage_normal: float = 1.0
    leverage_elevated: float = 0.5
    leverage_crisis: float = -0.3   # net short in crisis
    leverage_recovery: float = 1.5  # overweight on recovery

    equity_ticker: str = "QQQ"
    hedge_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    # Hedge allocation during crisis
    hedge_weight_crisis: float = 0.4
    gold_weight_crisis: float = 0.2


class FourStateTactical(Strategy):
    """4-State Tactical Allocation.

    Regime-driven allocation that goes leveraged-long in NORMAL/RECOVERY,
    hedges with TLT + GLD in crisis, and shorts equity in severe crises.
    """

    name = "FourStateTactical"

    def __init__(self, config: FourStateConfig | None = None) -> None:
        self.cfg = config or FourStateConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("FourStateTactical requires a regime Series")

        c = self.cfg
        idx = prices.index
        w = pd.DataFrame(0.0, index=idx, columns=[c.equity_ticker, c.hedge_ticker, c.gold_ticker])

        mapping = {
            Regime.NORMAL: c.leverage_normal,
            Regime.ELEVATED: c.leverage_elevated,
            Regime.OIL_CRISIS: c.leverage_crisis,
            Regime.WAR_CRISIS: c.leverage_crisis,
            Regime.GENERAL_CRISIS: c.leverage_crisis,
            Regime.RECOVERY: c.leverage_recovery,
        }

        for reg, lev in mapping.items():
            mask = regime.isin({reg})
            w.loc[mask, c.equity_ticker] = lev

            if lev < 0:
                # In crisis: allocate to hedges
                w.loc[mask, c.hedge_ticker] = c.hedge_weight_crisis
                w.loc[mask, c.gold_ticker] = c.gold_weight_crisis

        return w


# =========================================================================
# D2 — Crash-Hedged QQQ v2 (vol-regime switching)
# =========================================================================

@dataclass
class CrashHedgeConfig:
    """Parameters for the crash-hedge strategy."""

    equity_ticker: str = "QQQ"
    hedge_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    # Vol windows
    fast_vol_window: int = 20
    slow_vol_window: int = 60

    # Vol thresholds (annualised)
    vol_low: float = 0.12
    vol_high: float = 0.25

    # Leverage schedule
    leverage_low_vol: float = 2.5
    leverage_mid_vol: float = 1.0
    leverage_high_vol: float = 0.0  # flat in equity
    hedge_high_vol: float = 0.6     # TLT weight
    gold_high_vol: float = 0.3      # GLD weight

    # EMA smoothing for regime transitions
    ema_span: int = 5


class CrashHedgeQQQ(Strategy):
    """Crash-Hedged QQQ v2.

    Vol-regime switching strategy: leveraged long in calm markets,
    rotates into TLT + GLD during high-vol regimes.
    """

    name = "CrashHedgeQQQ"

    def __init__(self, config: CrashHedgeConfig | None = None) -> None:
        self.cfg = config or CrashHedgeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        equity = prices[c.equity_ticker]

        # Fast and slow realised vol
        fast_vol = realized_vol(equity, c.fast_vol_window)
        slow_vol = realized_vol(equity, c.slow_vol_window)
        # Blend: max of fast & slow (conservative)
        vol = pd.concat([fast_vol, slow_vol], axis=1).max(axis=1)
        vol = ema(vol, c.ema_span)  # smooth

        idx = prices.index
        w = pd.DataFrame(0.0, index=idx, columns=[c.equity_ticker, c.hedge_ticker, c.gold_ticker])

        low = vol <= c.vol_low
        mid = (vol > c.vol_low) & (vol <= c.vol_high)
        high = vol > c.vol_high

        w.loc[low, c.equity_ticker] = c.leverage_low_vol
        w.loc[mid, c.equity_ticker] = c.leverage_mid_vol
        w.loc[high, c.hedge_ticker] = c.hedge_high_vol
        w.loc[high, c.gold_ticker] = c.gold_high_vol

        return w


# =========================================================================
# D3 — Vol Carry
# =========================================================================

@dataclass
class VolCarryConfig:
    """Parameters for vol-carry strategy."""

    equity_ticker: str = "SPY"
    hedge_ticker: str = "TLT"

    vol_window: int = 20
    vol_calm: float = 0.12
    vol_storm: float = 0.22

    leverage_calm: float = 2.0
    leverage_transition: float = 0.5
    leverage_storm: float = 0.0
    hedge_storm: float = 0.5

    # Momentum confirmation
    momentum_window: int = 63  # ~3 months
    use_momentum_filter: bool = True


class VolCarry(Strategy):
    """Vol Carry — leveraged equity in calm, flatten in crisis."""

    name = "VolCarry"

    def __init__(self, config: VolCarryConfig | None = None) -> None:
        self.cfg = config or VolCarryConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        vol = realized_vol(prices[c.equity_ticker], c.vol_window)

        # Momentum filter: reduce leverage when trend is negative
        mom = prices[c.equity_ticker].pct_change(c.momentum_window)
        mom_up = mom > 0 if c.use_momentum_filter else pd.Series(True, index=prices.index)

        idx = prices.index
        w = pd.DataFrame(0.0, index=idx, columns=[c.equity_ticker, c.hedge_ticker])

        calm = vol <= c.vol_calm
        transition = (vol > c.vol_calm) & (vol <= c.vol_storm)
        storm = vol > c.vol_storm

        w.loc[calm & mom_up, c.equity_ticker] = c.leverage_calm
        w.loc[calm & ~mom_up, c.equity_ticker] = c.leverage_transition  # reduced
        w.loc[transition, c.equity_ticker] = c.leverage_transition
        w.loc[storm, c.hedge_ticker] = c.hedge_storm

        return w


# =========================================================================
# D4 — Adaptive Stop Trend (ATR-based trailing stop on QQQ)
# =========================================================================


def _close_atr_d4(close: pd.Series, period: int = 14) -> pd.Series:
    """Close-only ATR proxy: rolling mean of absolute daily changes."""
    abs_change = (close - close.shift(1)).abs()
    return abs_change.rolling(period, min_periods=1).mean()


@dataclass
class AdaptiveStopTrendConfig:
    """ATR-adaptive trend-following on QQQ with regime-based hedging.

    Bullish (QQQ > SMA(200)): Long QQQ with ATR trailing stop.
    Bearish: Rotate into TLT + GLD + cash.
    Trailing stop approximated vectorially as: close < rolling_high - k*ATR.
    """

    equity_ticker: str = "QQQ"
    hedge_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    sma_window: int = 200
    atr_period: int = 14

    # Bullish: trailing stop at close - 2.0*ATR from rolling high
    bull_atr_mult: float = 2.0
    # Re-entry filter: QQQ must cross above SMA(200) - 0.5*ATR
    reentry_atr_mult: float = 0.5

    # Bullish equity weight (< 1.0 to reduce DD)
    bull_equity_weight: float = 0.80
    bull_gold_hedge: float = 0.10  # small permanent hedge in bull

    # Bearish allocation
    hedge_weight: float = 0.50
    gold_weight: float = 0.30
    # Remaining 0.20 = cash

    # Trailing stop on TLT in bearish regime
    bear_hedge_atr_mult: float = 1.5

    # Volume filter
    vol_avg_window: int = 20
    vol_mult_threshold: float = 1.2


class AdaptiveStopTrend(Strategy):
    """ATR-adaptive trend-following: improved D2 with volatility-aware stops.

    Thesis: CrashHedgeQQQ (D2) is the fund's best performer (Sharpe 0.94)
    but uses fixed vol thresholds. An ATR-adaptive trailing stop version
    should avoid whipsaws: tight stops in calm markets preserve gains,
    wide stops in volatile markets let positions breathe.

    Long-only, max leverage 1.0x.
    """

    name = "D4-AdaptiveStopTrend"

    def __init__(self, config: AdaptiveStopTrendConfig | None = None) -> None:
        self.cfg = config or AdaptiveStopTrendConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.equity_ticker not in prices.columns:
            return weights

        qqq = prices[c.equity_ticker]
        sma200 = qqq.rolling(c.sma_window, min_periods=1).mean()
        atr_val = _close_atr_d4(qqq, c.atr_period).fillna(0.0)

        # --- Volume / activity filter ---
        qqq_ret = qqq.pct_change().fillna(0.0).abs()
        avg_act = qqq_ret.rolling(c.vol_avg_window, min_periods=1).mean()
        high_activity = pd.notna(avg_act) & (qqq_ret > avg_act * c.vol_mult_threshold)
        vol_filter = (
            high_activity.astype(float).rolling(5, min_periods=1).max().astype(bool)
        )

        # --- Trend regime ---
        # Bullish with ATR re-entry filter:
        #   QQQ > SMA(200) - reentry_atr_mult * ATR
        reentry_threshold = sma200 - c.reentry_atr_mult * atr_val
        bullish_raw = qqq > reentry_threshold

        # --- ATR trailing stop (vectorized) ---
        # Approximate trailing stop: QQQ dips below rolling high - k * ATR
        # Use 20-day rolling high as a proxy for "recent high"
        rolling_high = qqq.rolling(20, min_periods=1).max()
        stop_level = rolling_high - c.bull_atr_mult * atr_val
        stopped_out = qqq < stop_level

        # Bullish = above SMA threshold AND not stopped out
        bullish = bullish_raw & ~stopped_out

        # Bearish = everything else
        bearish = ~bullish

        # --- Assign weights ---
        # Bullish: long QQQ (reduced from 1.0 for DD control)
        weights.loc[bullish, c.equity_ticker] = c.bull_equity_weight
        if c.gold_ticker in prices.columns:
            weights.loc[bullish, c.gold_ticker] = c.bull_gold_hedge

        # Bearish: rotate into hedges
        if c.hedge_ticker in prices.columns:
            # TLT trailing stop in bearish regime
            tlt = prices[c.hedge_ticker]
            tlt_atr = _close_atr_d4(tlt, c.atr_period).fillna(0.0)
            tlt_high = tlt.rolling(20, min_periods=1).max()
            tlt_stop = tlt_high - c.bear_hedge_atr_mult * tlt_atr
            tlt_ok = tlt >= tlt_stop

            # Only full TLT weight if not stopped
            tlt_w = pd.Series(0.0, index=prices.index)
            tlt_w[bearish & tlt_ok] = c.hedge_weight
            tlt_w[bearish & ~tlt_ok] = c.hedge_weight * 0.5

            weights[c.hedge_ticker] = tlt_w

        if c.gold_ticker in prices.columns:
            weights.loc[bearish, c.gold_ticker] = c.gold_weight

        # Volume filter: only allow position changes on high-activity days.
        # On low-activity days, carry forward the previous day's weights.
        # Approximation: blend toward new weights on vol-filter days.
        # For simplicity, we don't filter (positions are clear trend signals).

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
