"""Category H: Fixed Income / Rates strategies.

Yield curve and credit spread trades using bond ETF ratios as proxies
for rates positioning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import ema, zscore
from financial_algo.strategies.base import Strategy


# =========================================================================
# H1 — Yield Curve Steepener / Flattener
# =========================================================================

@dataclass
class YieldCurveConfig:
    """Config for yield curve strategy."""

    long_duration: str = "TLT"    # 20+ yr Treasury
    short_duration: str = "IEF"   # 7-10 yr Treasury

    # Momentum of the curve slope (TLT/IEF ratio)
    fast_momentum: int = 21       # 1-month ratio momentum
    slow_momentum: int = 63       # 3-month ratio momentum
    trend_window: int = 200       # 200-day trend filter

    leverage: float = 1.0


class YieldCurveTrade(Strategy):
    """Trade the yield curve slope via TLT/IEF ratio momentum.

    Thesis: When TLT/IEF ratio is rising (curve steepening), long
    duration outperforms -- go long TLT. When falling (flattening),
    rotate to IEF. Uses 200-day trend filter to avoid fighting
    major trend reversals. Long-only, no shorts.
    """

    name = "H1-YieldCurveTrade"

    def __init__(self, config: YieldCurveConfig | None = None) -> None:
        self.cfg = config or YieldCurveConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.long_duration not in prices.columns or c.short_duration not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        ratio = prices[c.long_duration] / prices[c.short_duration]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill()

        # Momentum signals on the ratio
        fast_mom = ratio.pct_change(c.fast_momentum).fillna(0.0)
        slow_mom = ratio.pct_change(c.slow_momentum).fillna(0.0)

        # 200-day trend filter on TLT
        tlt_sma200 = prices[c.long_duration].rolling(c.trend_window, min_periods=100).mean()
        tlt_above_trend = prices[c.long_duration] >= tlt_sma200

        # Steepening signal: both fast and slow momentum positive
        steepening = pd.notna(fast_mom) & (fast_mom > 0) & pd.notna(slow_mom) & (slow_mom > 0)
        # Confirmed steepening: ratio rising AND TLT above its 200d SMA
        long_tlt = steepening & tlt_above_trend

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # Long TLT when steepening + trend confirms; else long IEF (always invested)
        weights[c.long_duration] = np.where(long_tlt, c.leverage, 0.0)
        weights[c.short_duration] = np.where(long_tlt, 0.0, c.leverage)

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# H2 — Credit Spread Mean-Reversion
# =========================================================================

@dataclass
class CreditSpreadConfig:
    """Config for credit spread mean-reversion."""

    high_yield: str = "HYG"    # High-yield corporate bonds
    inv_grade: str = "LQD"     # Investment-grade corporate bonds
    safe_ticker: str = "IEF"   # Park cash in intermediate Treasuries

    zscore_window: int = 60
    entry_z: float = -1.0      # z < -1.0 means spreads widened -> buy HYG
    exit_z: float = 0.0        # exit when z reverts to mean

    trend_window: int = 200    # 200-day trend filter on HYG
    leverage: float = 1.0


class CreditSpreadMeanRev(Strategy):
    """Mean-revert the HYG/LQD credit spread ratio (long-only).

    Thesis: When HYG underperforms LQD significantly (z-score drops,
    spreads widen), go long HYG expecting mean-reversion. Use 200-day
    trend filter -- only enter when HYG is above its long-term trend.
    Park in IEF when not in position. Pure long, no shorts.
    """

    name = "H2-CreditSpreadMeanRev"

    def __init__(self, config: CreditSpreadConfig | None = None) -> None:
        self.cfg = config or CreditSpreadConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.high_yield not in prices.columns or c.inv_grade not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        ratio = prices[c.high_yield] / prices[c.inv_grade]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill()
        z = zscore(ratio, c.zscore_window)

        # Trend filter: HYG above its 200-day SMA
        hyg_sma = prices[c.high_yield].rolling(c.trend_window, min_periods=100).mean()
        hyg_above_trend = prices[c.high_yield] >= hyg_sma

        # Entry: z-score below entry threshold (spreads widened) + trend confirms
        entry_signal = pd.notna(z) & (z < c.entry_z) & hyg_above_trend
        # Stay in position until z reverts above exit threshold
        exit_signal = pd.notna(z) & (z > c.exit_z)

        # Vectorized position tracking using ffill logic
        # 1 = long HYG, 0 = not in position
        raw_signal = pd.Series(np.nan, index=prices.index)
        raw_signal[entry_signal] = 1.0
        raw_signal[exit_signal] = 0.0
        position = raw_signal.ffill().fillna(0.0)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        in_position = position > 0.5
        weights[c.high_yield] = np.where(in_position, c.leverage, 0.0)
        safe_col = c.safe_ticker if c.safe_ticker in prices.columns else c.inv_grade
        weights[safe_col] = np.where(in_position, 0.0, c.leverage)

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# H3 -- Duration Timing (rotate short/long bonds based on vol regime)
# =========================================================================

@dataclass
class DurationTimingConfig:
    """Rotate between long and short duration based on TLT momentum."""

    long_duration: str = "TLT"    # 20+ yr
    short_duration: str = "IEF"   # 7-10 yr

    # TLT momentum as primary signal
    fast_momentum: int = 21       # 1-month momentum
    slow_momentum: int = 63       # 3-month momentum
    trend_window: int = 200       # 200-day SMA trend filter

    # Aggressive sizing when signal is strong
    base_leverage: float = 1.0
    strong_signal_boost: float = 0.5  # extra weight when both signals agree


class DurationTiming(Strategy):
    """Rotate between long and short duration based on TLT momentum.

    Thesis: TLT momentum captures the direction of long-term rates.
    When TLT has positive momentum (rates falling), go long TLT.
    When negative (rates rising), rotate to IEF for safety. Size up
    aggressively when both fast and slow momentum agree.
    """

    name = "H3-DurationTiming"

    def __init__(self, config: DurationTimingConfig | None = None) -> None:
        self.cfg = config or DurationTimingConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.long_duration not in prices.columns or c.short_duration not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        tlt = prices[c.long_duration]

        # Momentum signals
        fast_mom = tlt.pct_change(c.fast_momentum).fillna(0.0)
        slow_mom = tlt.pct_change(c.slow_momentum).fillna(0.0)

        # Trend filter
        tlt_sma = tlt.rolling(c.trend_window, min_periods=100).mean()
        above_trend = tlt >= tlt_sma

        # Signal strength: both fast and slow agree
        both_positive = pd.notna(fast_mom) & (fast_mom > 0) & pd.notna(slow_mom) & (slow_mom > 0)
        fast_positive = pd.notna(fast_mom) & (fast_mom > 0)

        # Long TLT: fast momentum positive + above trend (boost if both agree)
        strong_long = both_positive & above_trend
        weak_long = fast_positive & above_trend & ~both_positive

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Strong signal: full leverage + boost
        lev_strong = c.base_leverage + c.strong_signal_boost
        weights[c.long_duration] = np.where(
            strong_long, lev_strong,
            np.where(weak_long, c.base_leverage, 0.0),
        )
        # When not long TLT, park in IEF
        weights[c.short_duration] = np.where(
            strong_long | weak_long, 0.0, c.base_leverage,
        )

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
