"""Category C: War & Geopolitical Crisis Specialist strategies.

Dual-mode: trend-follow defense/safe-haven assets in normal markets,
amplify during geopolitical crises for crisis alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import ema, realized_vol
from financial_algo.regimes import Regime
from financial_algo.strategies.base import Strategy


# =========================================================================
# C1 - Defense Rotation (dual-mode: trend + crisis rotation)
# =========================================================================

@dataclass
class DefenseRotationConfig:
    """Config for defense rotation strategy."""

    defense_tickers: list[str] | None = None
    short_tickers: list[str] | None = None

    fast_ema: int = 20
    slow_ema: int = 50

    leverage_normal: float = 0.5
    leverage_crisis: float = 1.5
    short_leverage_crisis: float = 0.5

    vol_window: int = 20
    target_vol: float = 0.15

    def __post_init__(self) -> None:
        if self.defense_tickers is None:
            self.defense_tickers = ["ITA", "LMT", "RTX"]
        if self.short_tickers is None:
            self.short_tickers = ["EFA", "EEM"]


class DefenseRotation(Strategy):
    """Dual-mode defense sector strategy.

    Normal: Trend-follow defense stocks with vol scaling (always-on).
    Crisis: Full defense rotation + conditional shorts.
    """

    name = "DefenseRotation"

    def __init__(self, config: DefenseRotationConfig | None = None) -> None:
        self.cfg = config or DefenseRotationConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("DefenseRotation requires a regime Series")

        c = self.cfg
        crisis = regime.isin({Regime.WAR_CRISIS, Regime.GENERAL_CRISIS})
        elevated = regime.isin({Regime.ELEVATED})

        all_tickers = c.defense_tickers + c.short_tickers
        w = pd.DataFrame(0.0, index=prices.index, columns=all_tickers)

        per_normal = c.leverage_normal / len(c.defense_tickers)
        per_crisis = c.leverage_crisis / len(c.defense_tickers)
        per_elevated = (per_normal + per_crisis) / 2

        for t in c.defense_tickers:
            if t not in prices.columns:
                continue
            p = prices[t]
            fast = ema(p, c.fast_ema)
            slow = ema(p, c.slow_ema)
            trending = fast > slow

            tvol = realized_vol(p, c.vol_window).clip(lower=0.05)
            vs = (c.target_vol / tvol).clip(0.3, 2.0)
            vs = vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # Normal: trend-follow (always-on when trending)
            base = trending.astype(float) * per_normal * vs
            # Crisis: full allocation (trend filter relaxed)
            crisis_w = crisis.astype(float) * per_crisis * vs
            # Elevated: moderate boost
            elevated_w = (elevated & trending).astype(float) * per_elevated * vs

            w[t] = pd.concat([base, crisis_w, elevated_w], axis=1).max(axis=1)

        # Short international equity only in crisis + trending down
        per_short = c.short_leverage_crisis / max(len(c.short_tickers), 1)
        for t in c.short_tickers:
            if t not in prices.columns:
                continue
            p = prices[t]
            t_ema = ema(p, c.slow_ema)
            down_trend = p < t_ema
            w.loc[crisis & down_trend, t] = -per_short

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# C2 - Safe Haven Flight (always-on haven trend + crisis amplification)
# =========================================================================

@dataclass
class SafeHavenFlightConfig:
    """Config for safe-haven trend strategy."""

    haven_tickers: list[str] | None = None
    short_ticker: str = "SPY"

    fast_ema: int = 20
    slow_ema: int = 50

    leverage_normal: float = 0.5
    leverage_crisis: float = 1.0
    short_leverage: float = 0.3

    vol_window: int = 20
    target_vol: float = 0.12

    def __post_init__(self) -> None:
        if self.haven_tickers is None:
            self.haven_tickers = ["GLD", "TLT"]


class SafeHavenFlight(Strategy):
    """Dual-mode safe-haven strategy.

    Normal: Trend-follow GLD/TLT (always-on safe-haven momentum).
    Crisis: Amplify haven allocation + conditional short SPY.
    """

    name = "SafeHavenFlight"

    def __init__(self, config: SafeHavenFlightConfig | None = None) -> None:
        self.cfg = config or SafeHavenFlightConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("SafeHavenFlight requires a regime Series")

        c = self.cfg
        crisis = regime.isin({Regime.WAR_CRISIS, Regime.GENERAL_CRISIS})

        all_tickers = c.haven_tickers + [c.short_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=all_tickers)

        per_normal = c.leverage_normal / max(len(c.haven_tickers), 1)
        per_crisis = c.leverage_crisis / max(len(c.haven_tickers), 1)

        for t in c.haven_tickers:
            if t not in prices.columns:
                continue
            p = prices[t]
            fast = ema(p, c.fast_ema)
            slow = ema(p, c.slow_ema)
            trending = fast > slow

            tvol = realized_vol(p, c.vol_window).clip(lower=0.05)
            vs = (c.target_vol / tvol).clip(0.3, 2.0)
            vs = vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # Normal: trend-follow (always-on)
            base = trending.astype(float) * per_normal * vs
            # Crisis: amplified
            crisis_w = (crisis & trending).astype(float) * per_crisis * vs
            w[t] = pd.concat([base, crisis_w], axis=1).max(axis=1)

        # Short SPY in crisis when trending down
        if c.short_ticker in prices.columns:
            spy = prices[c.short_ticker]
            spy_ema = ema(spy, c.slow_ema)
            spy_down = spy < spy_ema
            w.loc[crisis & spy_down, c.short_ticker] = -c.short_leverage

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# C3 - Post-War Recovery (equity trend + recovery boost)
# =========================================================================

@dataclass
class PostWarRecoveryConfig:
    """Config for equity trend + recovery strategy."""

    equity_ticker: str = "SPY"

    fast_ema: int = 20
    slow_ema: int = 50
    momentum_window: int = 63

    leverage_trend: float = 1.0
    leverage_recovery: float = 1.5

    vol_window: int = 20
    target_vol: float = 0.15


class PostWarRecovery(Strategy):
    """Dual-mode equity strategy: trend + recovery boost.

    Normal: Trend-follow SPY when EMA crossover positive, vol-scaled.
    Recovery: Amplified long during RECOVERY regime.
    Crisis: Flat (capital preservation).
    """

    name = "PostWarRecovery"

    def __init__(self, config: PostWarRecoveryConfig | None = None) -> None:
        self.cfg = config or PostWarRecoveryConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("PostWarRecovery requires a regime Series")

        c = self.cfg
        spy = prices[c.equity_ticker]

        # Trend signal: EMA crossover
        fast = ema(spy, c.fast_ema)
        slow = ema(spy, c.slow_ema)
        trend_up = fast > slow

        # Momentum confirmation
        mom = spy.pct_change(c.momentum_window).fillna(0.0)
        mom_pos = mom > 0

        # Vol scaling
        spy_vol = realized_vol(spy, c.vol_window).clip(lower=0.05)
        vol_scale = (c.target_vol / spy_vol).clip(0.3, 2.0)
        vol_scale = vol_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Regime overlay
        recovery = regime.isin({Regime.RECOVERY})
        crisis = regime.isin({
            Regime.GENERAL_CRISIS, Regime.WAR_CRISIS, Regime.OIL_CRISIS,
        })

        # Base: trend + momentum
        signal = trend_up & mom_pos
        base = signal.astype(float) * c.leverage_trend * vol_scale

        # Recovery boost
        recovery_w = (recovery & trend_up).astype(float) * c.leverage_recovery * vol_scale

        weight = pd.concat([base, recovery_w], axis=1).max(axis=1)

        # Cut in crisis (capital preservation)
        weight[crisis] = 0.0

        w = pd.DataFrame(0.0, index=prices.index, columns=[c.equity_ticker])
        w[c.equity_ticker] = weight
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# C4 - Arms Race Momentum (defense trend + momentum boost)
# =========================================================================

@dataclass
class ArmsRaceMomentumConfig:
    """Config for defense trend-following with momentum boost."""

    defense_ticker: str = "ITA"

    fast_ema: int = 20
    slow_ema: int = 50
    momentum_window: int = 63
    momentum_threshold: float = 0.0

    leverage: float = 1.0
    momentum_boost: float = 0.5

    vol_window: int = 20
    target_vol: float = 0.15


class ArmsRaceMomentum(Strategy):
    """Defense sector trend-following with momentum boost.

    Normal: Long ITA when EMA crossover positive, vol-scaled.
    Strong momentum: Extra leverage when quarterly return is positive.
    """

    name = "ArmsRaceMomentum"

    def __init__(self, config: ArmsRaceMomentumConfig | None = None) -> None:
        self.cfg = config or ArmsRaceMomentumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        defense = prices[c.defense_ticker]

        # Trend: EMA crossover
        fast = ema(defense, c.fast_ema)
        slow = ema(defense, c.slow_ema)
        trend_up = fast > slow

        # Momentum
        mom = defense.pct_change(c.momentum_window).fillna(0.0)
        strong_mom = mom >= c.momentum_threshold

        # Vol scaling
        d_vol = realized_vol(defense, c.vol_window).clip(lower=0.05)
        vol_scale = (c.target_vol / d_vol).clip(0.3, 2.0)
        vol_scale = vol_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Base: trend-following
        base = trend_up.astype(float) * c.leverage * vol_scale

        # Momentum boost
        boost = (trend_up & strong_mom).astype(float) * c.momentum_boost * vol_scale

        weight = base + boost

        # Regime scaling (optional)
        if regime is not None:
            regime_mult = pd.Series(1.0, index=prices.index)
            regime_mult[regime.isin({Regime.WAR_CRISIS})] = 1.3
            regime_mult[regime.isin({Regime.GENERAL_CRISIS})] = 0.7
            weight = weight * regime_mult

        w = pd.DataFrame(0.0, index=prices.index, columns=[c.defense_ticker])
        w[c.defense_ticker] = weight
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w
