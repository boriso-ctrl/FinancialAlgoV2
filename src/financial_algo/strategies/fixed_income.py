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
    """Trade the yield curve slope via TLT/IEF ratio momentum with Fed regime detection.

    Thesis: Yield curve trades have different alpha in tightening vs easing regimes.
    When curve is steepening (TLT/IEF rising) AND TLT momentum is positive AND
    real rate expectations are improving (indicated by curve not in inversion),
    go long TLT (duration call). Else IEF (intermediate rates).
    
    Fed regime detection: Use TLT 200d SMA as proxy for long-term rate expectations.
    When TLT is below its 200d SMA, Fed is typically tightening or in high-rate regime.
    When TLT is above, Fed likely easing or rates falling.
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

        tlt = prices[c.long_duration]
        ief = prices[c.short_duration]
        ratio = tlt / ief
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill()

        # --- Curve slope signal (TLT/IEF ratio momentum) ---
        fast_mom = ratio.pct_change(c.fast_momentum).fillna(0.0)
        slow_mom = ratio.pct_change(c.slow_momentum).fillna(0.0)

        # Steepening: both momentum windows positive (curve slope rising)
        steepening = pd.notna(fast_mom) & (fast_mom > 0) & pd.notna(slow_mom) & (slow_mom > 0)

        # --- Fed regime detection via TLT long-term trend ---
        # TLT >> 200d SMA indicates falling rates / easing cycle (good for long duration)
        # TLT << 200d SMA indicates rising rates / tightening cycle (caution: avoid duration)
        tlt_sma200 = tlt.rolling(c.trend_window, min_periods=100).mean()
        tlt_above_trend = tlt >= tlt_sma200
        
        # Calculate how far above/below the 200d SMA (regime strength indicator)
        tlt_premium_pct = ((tlt - tlt_sma200) / tlt_sma200).clip(-0.15, 0.15)  # -15% to +15%
        in_easing_regime = tlt_premium_pct > 0.03  # TLT 3%+ above SMA = easing regime

        # --- Position logic ---
        # Confirmed steepening: ratio rising AND TLT in easing regime (safe for duration extension)
        long_tlt = steepening & in_easing_regime

        # Partial signal: just steepening without full regime confirmation (cautious)
        steepening_only = steepening & ~in_easing_regime

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        
        # Full allocation to TLT when steepening + easing regime
        weights[c.long_duration] = np.where(long_tlt, c.leverage, 0.0)
        
        # Reduced allocation (50%) if steepening without full regime confirmation
        weights[c.long_duration] = np.where(
            steepening_only,
            0.5 * c.leverage,
            weights[c.long_duration],
        )
        
        # Allocate IEF for the remainder (always invested, duration varies)
        weights[c.short_duration] = np.where(
            long_tlt,
            0.0,
            np.where(steepening_only, 0.5 * c.leverage, c.leverage),
        )

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
    """Mean-revert the HYG/LQD credit spread ratio with vol-scaling and Fed regime awareness.

    Thesis: Credit spreads widen in crises (HYG underperforms) then revert to mean.
    Exploit mean-reversion but gate entries with:
      1. Vol-scaled z-score thresholds (higher vol = wider entry tolerance)
      2. Fed regime filter (spreads widen naturally during tightening; safer to buy
         during easing/neutral cycles)
    
    Fed regime proxy: When TLT is rising (yields falling), Fed is easing → good time
    to enter credit long. When TLT is falling (yields rising), Fed is tightening →
    skip or reduce entry.
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

        hyg = prices[c.high_yield]
        lqd = prices[c.inv_grade]
        ratio = hyg / lqd
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill()
        z = zscore(ratio, c.zscore_window)

        # --- Vol-scaled z-score thresholds ---
        # When spreads are volatile, z-scores become more extreme faster
        # So we tighten entry thresholds when vol increases
        daily_returns = ratio.pct_change().fillna(0.0).abs()
        vol_window = max(c.zscore_window, 20)
        rolling_vol = daily_returns.rolling(vol_window, min_periods=10).std()
        rolling_vol = rolling_vol.replace(0, np.nan).ffill()
        
        # Vol percentile: compare current vol to recent range (0-1 scale)
        vol_min = rolling_vol.rolling(60, min_periods=20).min()
        vol_max = rolling_vol.rolling(60, min_periods=20).max()
        vol_percentile = ((rolling_vol - vol_min) / (vol_max - vol_min + 1e-6)).clip(0, 1)
        
        # Entry threshold adjusts with vol: in high-vol periods, tighten to z < -1.5
        # In low-vol periods, relax to z < -0.7 (more sensitive to small widening)
        entry_z_base = c.entry_z  # default -1.0
        entry_z_adjusted = entry_z_base - (vol_percentile * 0.8).clip(0, 0.8)
        
        # --- Fed regime filter via TLT trend ---
        # TLT positive 21d momentum = yields falling = easing cycle (good for credit)
        # TLT negative 21d momentum = yields rising = tightening cycle (risky for credit)
        if c.safe_ticker in prices.columns:
            tlt = prices[c.safe_ticker]
            tlt_momentum = tlt.pct_change(21).fillna(0.0)
            easing_regime = pd.notna(tlt_momentum) & (tlt_momentum > -0.01)  # Allow neutral
        else:
            # Fallback: assume easing regime
            easing_regime = pd.Series(True, index=prices.index)

        # Trend filter: HYG above its 200-day SMA (not in downtrend)
        hyg_sma = hyg.rolling(c.trend_window, min_periods=100).mean()
        hyg_above_trend = hyg >= hyg_sma

        # Entry: z-score below adjusted threshold + trend filter + Fed regime confirmation
        entry_signal = pd.notna(z) & (z < entry_z_adjusted) & hyg_above_trend & easing_regime
        
        # Exit: z reverts above a higher threshold (less aggressive exit)
        exit_z_dynamic = c.exit_z + 0.3  # Slightly wider exit than entry to reduce whipsaw
        exit_signal = pd.notna(z) & (z > exit_z_dynamic)

        # Vectorized position tracking
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
