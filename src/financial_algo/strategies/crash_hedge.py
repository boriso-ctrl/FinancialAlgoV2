"""Category D: General Crisis Alpha — Crash-Hedge strategies.

Ported and evolved from Financial-Algorithms ``strat_crash_hedged``
(vol-regime switching) and ``strategy_4state`` (4-state tactical allocation).
"""

from __future__ import annotations

from dataclasses import dataclass

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
