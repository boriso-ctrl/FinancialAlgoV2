"""Category Q: Quality Trend strategies.

High-conviction strategies combining trend-following with quality filters.
These target the intersection of momentum and low-volatility factors --
two of the most robust and well-documented alpha sources in academic finance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import realized_vol, ema
from financial_algo.strategies.base import Strategy


# =========================================================================
# Q1 -- Trend + Vol Filter (the "Quality Trend")
# =========================================================================

@dataclass
class QualityTrendConfig:
    """Simple trend-following with vol filter.

    Long equities when price is above 200-day SMA AND vol is low.
    Switch to safe havens otherwise. One of the most robust signals
    in quantitative finance -- combines time-series momentum with
    volatility timing.
    """

    equity_ticker: str = "SPY"
    alt_equity: str = "QQQ"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    trend_window: int = 200  # 200-day SMA for trend
    vol_window: int = 20
    vol_threshold: float = 0.20  # below this = calm

    # Leverage by regime
    leverage_trend_up_calm: float = 2.0   # trend up + low vol
    leverage_trend_up_vol: float = 0.8    # trend up + high vol
    leverage_trend_down: float = 0.0      # trend down
    safe_weight_down: float = 0.4         # TLT when trend down
    gold_weight_down: float = 0.2         # GLD when trend down

    # Multi-asset: also trade alt equity
    alt_weight_fraction: float = 0.3  # fraction of equity allocation to QQQ


class QualityTrend(Strategy):
    """Quality Trend Following -- long equities above 200-day MA in calm vol.

    Thesis: Moskowitz, Ooi, and Pedersen (2012) "Time Series Momentum"
    shows trend-following works across all asset classes. Combining with
    vol timing (Moreira & Muir, 2017) creates a more robust signal.
    This is a simplified, high-capacity version of managed futures.
    """

    name = "Q1-QualityTrend"

    def __init__(self, config: QualityTrendConfig | None = None) -> None:
        self.cfg = config or QualityTrendConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        eq = prices[c.equity_ticker]

        # Trend signal: price > 200-day SMA
        sma = eq.rolling(c.trend_window).mean()
        trend_up = eq > sma

        # Vol filter
        vol = realized_vol(eq, c.vol_window)
        calm = vol <= c.vol_threshold

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Trend up + calm: full leverage
        up_calm = trend_up & calm
        up_vol = trend_up & ~calm
        down = ~trend_up

        main_frac = 1.0 - c.alt_weight_fraction
        alt_frac = c.alt_weight_fraction if c.alt_equity in prices.columns else 0.0
        if alt_frac > 0:
            main_frac = 1.0 - alt_frac

        weights.loc[up_calm, c.equity_ticker] = c.leverage_trend_up_calm * main_frac
        if alt_frac > 0:
            weights.loc[up_calm, c.alt_equity] = c.leverage_trend_up_calm * alt_frac

        weights.loc[up_vol, c.equity_ticker] = c.leverage_trend_up_vol * main_frac
        if alt_frac > 0:
            weights.loc[up_vol, c.alt_equity] = c.leverage_trend_up_vol * alt_frac

        weights.loc[down, c.safe_ticker] = c.safe_weight_down
        weights.loc[down, c.gold_ticker] = c.gold_weight_down

        return weights.fillna(0.0)


# =========================================================================
# Q2 -- Multi-Asset Trend Composite
# =========================================================================

@dataclass
class MultiTrendConfig:
    """Trend-follow multiple uncorrelated assets simultaneously."""

    assets: tuple = ("SPY", "QQQ", "GLD", "TLT", "EFA", "IWM")
    safe_ticker: str = "IEF"

    fast_window: int = 50    # fast trend
    slow_window: int = 200   # slow trend
    vol_window: int = 20

    leverage_per_asset: float = 0.4  # max leverage per asset when trending
    max_total_leverage: float = 2.5


class MultiAssetTrend(Strategy):
    """Trend-follow multiple uncorrelated assets with vol scaling.

    Thesis: Diversified trend-following across uncorrelated assets
    delivers consistent positive returns with low correlation to
    traditional markets. This is the core strategy of managed futures
    (Hurst, Ooi, Pedersen 2017).
    """

    name = "Q2-MultiAssetTrend"

    def __init__(self, config: MultiTrendConfig | None = None) -> None:
        self.cfg = config or MultiTrendConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.assets if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for ticker in avail:
            price = prices[ticker]
            fast_sma = price.rolling(c.fast_window).mean()
            slow_sma = price.rolling(c.slow_window).mean()

            # Dual-crossover: trend up when fast > slow
            trend_up = fast_sma > slow_sma

            # Vol-scale: inverse vol weighting (lower vol = more weight)
            vol = realized_vol(price, c.vol_window).clip(lower=0.05)
            vol_scale = (0.15 / vol).clip(upper=2.0)  # target 15% vol per position

            pos = trend_up.astype(float) * c.leverage_per_asset * vol_scale

            weights[ticker] = pos

        # Cap total leverage
        gross = weights.abs().sum(axis=1)
        cap_scale = (c.max_total_leverage / gross).clip(upper=1.0)
        weights = weights.multiply(cap_scale, axis=0)

        # When very few assets trending, add safe haven
        trending_count = (weights > 0.01).sum(axis=1)
        few_trending = trending_count <= 1
        if c.safe_ticker in prices.columns:
            weights.loc[few_trending, c.safe_ticker] = np.maximum(
                weights.loc[few_trending, c.safe_ticker], 0.3
            )

        return weights.fillna(0.0)


# =========================================================================
# Q3 -- Momentum Crash Filter
# =========================================================================

@dataclass
class MomentumCrashConfig:
    """Momentum strategy with crash protection.

    Standard momentum crashes during market reversals (2009, 2020).
    Adding a crash filter (vol + drawdown) prevents the worst losses.
    """

    equity_ticker: str = "SPY"
    alt_equity: str = "QQQ"
    safe_ticker: str = "TLT"

    # Multi-timeframe momentum
    fast_mom: int = 21      # 1-month
    med_mom: int = 63       # 3-month
    slow_mom: int = 252     # 12-month

    vol_window: int = 20
    crash_vol: float = 0.30  # flatten if vol > this (crash protection)

    leverage_strong: float = 2.0   # all momenta positive
    leverage_mixed: float = 0.8    # mixed signals
    leverage_safe: float = 0.0     # all momenta negative
    safe_weight: float = 0.5


class MomentumCrashFilter(Strategy):
    """Multi-timeframe momentum with crash protection.

    Thesis: Momentum works (Jegadeesh & Titman 1993) but crashes
    hard during bear market reversals (Daniel & Moskowitz 2016).
    Adding a high-vol filter prevents holding long into crashes.
    Multi-timeframe confirmation reduces whipsaws.
    """

    name = "Q3-MomentumCrashFilter"

    def __init__(self, config: MomentumCrashConfig | None = None) -> None:
        self.cfg = config or MomentumCrashConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        eq = prices[c.equity_ticker]

        # Three momentum signals
        fast = eq.pct_change(c.fast_mom) > 0
        med = eq.pct_change(c.med_mom) > 0
        slow = eq.pct_change(c.slow_mom) > 0

        # Count positive signals (0-3)
        score = fast.astype(int) + med.astype(int) + slow.astype(int)

        # Crash filter
        vol = realized_vol(eq, c.vol_window)
        crash = vol > c.crash_vol

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Strong: all 3 momenta positive + no crash
        strong = (score == 3) & ~crash
        mixed = (score >= 1) & (score < 3) & ~crash
        weak = (score == 0) | crash

        weights.loc[strong, c.equity_ticker] = c.leverage_strong * 0.7
        if c.alt_equity in prices.columns:
            weights.loc[strong, c.alt_equity] = c.leverage_strong * 0.3

        weights.loc[mixed, c.equity_ticker] = c.leverage_mixed
        weights.loc[weak, c.safe_ticker] = c.safe_weight

        return weights.fillna(0.0)
