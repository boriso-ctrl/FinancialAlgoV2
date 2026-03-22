"""Category B: Oil Crisis Specialist strategies.

Dual-mode: trend-follow energy assets in normal markets,
amplify positioning during oil crisis episodes for crisis alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import ema, realized_vol, zscore
from financial_algo.regimes import Regime
from financial_algo.strategies.base import Strategy


# =========================================================================
# B1 - Oil Momentum Surge (energy surge + gold carry)
# =========================================================================

@dataclass
class OilMomentumConfig:
    """Config for oil momentum surge strategy."""

    oil_ticker: str = "USO"
    energy_ticker: str = "XLE"
    carry_ticker: str = "GLD"

    fast_ema: int = 20
    slow_ema: int = 50

    momentum_window: int = 20
    momentum_threshold: float = 0.03
    surge_leverage: float = 1.5
    carry_leverage: float = 0.7

    vol_window: int = 20
    target_vol: float = 0.20


class OilMomentumSurge(Strategy):
    """Dual-asset energy surge strategy.

    Surge mode: Long XLE when oil has strong momentum + uptrend.
    Carry mode: Long GLD when gold trends up and oil is not surging.
    This captures oil rally alpha while earning gold carry between events.
    """

    name = "OilMomentumSurge"

    def __init__(self, config: OilMomentumConfig | None = None) -> None:
        self.cfg = config or OilMomentumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        oil = prices[c.oil_ticker]
        xle = prices[c.energy_ticker]

        # --- Energy: oil momentum surge signal ---
        oil_ret = oil.pct_change(c.momentum_window).fillna(0.0)
        oil_ema = ema(oil, c.slow_ema)
        xle_fast = ema(xle, c.fast_ema)
        xle_slow = ema(xle, c.slow_ema)
        xle_trend = xle_fast > xle_slow

        surge = (oil_ret >= c.momentum_threshold) & (oil > oil_ema) & xle_trend

        xle_vol = realized_vol(xle, c.vol_window).clip(lower=0.05)
        xle_vs = (c.target_vol / xle_vol).clip(0.3, 2.0)
        xle_vs = xle_vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        xle_w = surge.astype(float) * c.surge_leverage * xle_vs

        # --- Gold carry: trend-follow gold when not in surge ---
        gld_w = pd.Series(0.0, index=prices.index)
        if c.carry_ticker in prices.columns:
            gld = prices[c.carry_ticker]
            gld_fast = ema(gld, c.fast_ema)
            gld_slow = ema(gld, c.slow_ema)
            gld_trend = gld_fast > gld_slow

            gld_vol = realized_vol(gld, c.vol_window).clip(lower=0.05)
            gld_vs = (c.target_vol / gld_vol).clip(0.3, 2.0)
            gld_vs = gld_vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # Gold active when trending AND oil not surging
            gld_w = (gld_trend & ~surge).astype(float) * c.carry_leverage * gld_vs

        # Regime scaling (optional)
        if regime is not None:
            regime_mult = pd.Series(1.0, index=prices.index)
            regime_mult[regime.isin({Regime.OIL_CRISIS})] = 1.3
            regime_mult[regime.isin({Regime.ELEVATED})] = 1.1
            regime_mult[regime.isin({Regime.GENERAL_CRISIS, Regime.WAR_CRISIS})] = 0.5
            xle_w = xle_w * regime_mult

        w = pd.DataFrame(0.0, index=prices.index,
                         columns=[c.energy_ticker, c.carry_ticker])
        w[c.energy_ticker] = xle_w
        w[c.carry_ticker] = gld_w
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# B2 - Oil Shock Hedge (dual-asset trend: energy + gold)
# =========================================================================

@dataclass
class OilShockHedgeConfig:
    """Config for dual-asset oil/gold trend strategy."""

    oil_ticker: str = "USO"
    energy_ticker: str = "XLE"
    hedge_ticker: str = "GLD"

    fast_ema: int = 20
    slow_ema: int = 50

    energy_leverage: float = 0.7
    gold_leverage: float = 0.7
    shock_tilt: float = 0.3

    vol_window: int = 20
    target_vol: float = 0.15


class OilShockHedge(Strategy):
    """Dual-asset trend strategy: energy + gold.

    Normal: Long XLE when trending up, long GLD when trending up.
    Oil shock: Tilt toward energy (up-shock) or gold (down-shock).
    Two uncorrelated return streams for all-weather performance.
    """

    name = "OilShockHedge"

    def __init__(self, config: OilShockHedgeConfig | None = None) -> None:
        self.cfg = config or OilShockHedgeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        # --- Energy trend ---
        xle = prices[c.energy_ticker]
        xle_fast = ema(xle, c.fast_ema)
        xle_slow = ema(xle, c.slow_ema)
        xle_trend = xle_fast > xle_slow

        # --- Gold trend ---
        gld = prices[c.hedge_ticker]
        gld_fast = ema(gld, c.fast_ema)
        gld_slow = ema(gld, c.slow_ema)
        gld_trend = gld_fast > gld_slow

        # --- Vol scaling per asset ---
        xle_vol = realized_vol(xle, c.vol_window).clip(lower=0.05)
        gld_vol = realized_vol(gld, c.vol_window).clip(lower=0.05)
        xle_vs = (c.target_vol / xle_vol).clip(0.3, 2.0)
        gld_vs = (c.target_vol / gld_vol).clip(0.3, 2.0)
        xle_vs = xle_vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        gld_vs = gld_vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # --- Oil momentum for directional tilt ---
        oil = prices[c.oil_ticker]
        oil_ret = oil.pct_change(20).fillna(0.0)
        oil_strong = oil_ret > 0.05
        oil_weak = oil_ret < -0.05

        # Base weights: always-on trend-following
        xle_w = xle_trend.astype(float) * c.energy_leverage * xle_vs
        gld_w = gld_trend.astype(float) * c.gold_leverage * gld_vs

        # Shock tilt: boost the trending side during oil shocks
        xle_w = xle_w + (oil_strong & xle_trend).astype(float) * c.shock_tilt * xle_vs
        gld_w = gld_w + (oil_weak & gld_trend).astype(float) * c.shock_tilt * gld_vs

        w = pd.DataFrame(0.0, index=prices.index, columns=[c.energy_ticker, c.hedge_ticker])
        w[c.energy_ticker] = xle_w
        w[c.hedge_ticker] = gld_w
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# B3 - Oil Mean-Reversion (fully vectorized, dual-mode)
# =========================================================================

@dataclass
class OilMeanRevConfig:
    """Config for oil mean-reversion strategy."""

    oil_ticker: str = "USO"
    gold_ticker: str = "GLD"

    fast_ema: int = 20
    slow_ema: int = 50

    zscore_window: int = 60
    entry_z: float = 2.5

    mr_leverage: float = 0.5
    gold_leverage: float = 0.6

    vol_window: int = 20
    target_vol: float = 0.12


class OilMeanReversion(Strategy):
    """Oil extreme reversion + gold trend carry.

    MR mode: Fade only the most extreme oil z-score moves (|z| > 2.5).
    Carry mode: Trend-follow gold when not in MR position.
    Fully vectorized.
    """

    name = "OilMeanReversion"

    def __init__(self, config: OilMeanRevConfig | None = None) -> None:
        self.cfg = config or OilMeanRevConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        oil = prices[c.oil_ticker]

        # Vol scaling for oil
        oil_vol = realized_vol(oil, c.vol_window).clip(lower=0.05)
        vol_scale = (c.target_vol / oil_vol).clip(0.3, 2.0)
        vol_scale = vol_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Z-score MR: only trade extreme moves
        z = zscore(oil, c.zscore_window).fillna(0.0)
        oil_pos = np.where(
            z >= c.entry_z, -c.mr_leverage,
            np.where(z <= -c.entry_z, c.mr_leverage, 0.0),
        )
        oil_pos = oil_pos * vol_scale.values
        mr_active = np.abs(z.values) >= c.entry_z

        # Gold carry: trend-follow gold when MR is not active
        gld_w = pd.Series(0.0, index=prices.index)
        if c.gold_ticker in prices.columns:
            gld = prices[c.gold_ticker]
            gld_fast = ema(gld, c.fast_ema)
            gld_slow = ema(gld, c.slow_ema)
            gld_trend = gld_fast > gld_slow

            gld_vol = realized_vol(gld, c.vol_window).clip(lower=0.05)
            gld_vs = (c.target_vol / gld_vol).clip(0.3, 2.0)
            gld_vs = gld_vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

            gld_w = (gld_trend & ~pd.Series(mr_active, index=prices.index)).astype(
                float
            ) * c.gold_leverage * gld_vs

        cols = [c.oil_ticker, c.gold_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=cols)
        w[c.oil_ticker] = oil_pos
        w[c.gold_ticker] = gld_w
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# B4 - Energy Pairs (energy trend + oil confirmation)
# =========================================================================

@dataclass
class EnergyPairsConfig:
    """Config for energy trend strategy with oil confirmation."""

    long_ticker: str = "XLE"
    oil_ticker: str = "USO"

    fast_ema: int = 20
    slow_ema: int = 50
    momentum_window: int = 63

    leverage: float = 1.0

    vol_window: int = 20
    target_vol: float = 0.20


class EnergyPairs(Strategy):
    """Energy trend-following with oil and momentum confirmation.

    Long XLE when EMA crossover is bullish, quarterly momentum is
    positive, and oil trend confirms. Vol-scaled for risk parity.
    """

    name = "EnergyPairs"

    def __init__(self, config: EnergyPairsConfig | None = None) -> None:
        self.cfg = config or EnergyPairsConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        xle = prices[c.long_ticker]

        # Trend: EMA crossover
        fast = ema(xle, c.fast_ema)
        slow = ema(xle, c.slow_ema)
        trend_up = fast > slow

        # Quarterly momentum positive
        mom = xle.pct_change(c.momentum_window).fillna(0.0)
        mom_pos = mom > 0

        # Oil trend confirmation
        oil_ok = pd.Series(True, index=prices.index)
        if c.oil_ticker in prices.columns:
            oil = prices[c.oil_ticker]
            oil_fast = ema(oil, c.fast_ema)
            oil_slow = ema(oil, c.slow_ema)
            oil_ok = oil_fast > oil_slow

        signal = trend_up & mom_pos & oil_ok

        # Vol scaling
        xle_vol = realized_vol(xle, c.vol_window).clip(lower=0.05)
        vol_scale = (c.target_vol / xle_vol).clip(0.3, 2.0)
        vol_scale = vol_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        w = pd.DataFrame(0.0, index=prices.index, columns=[c.long_ticker])
        w[c.long_ticker] = signal.astype(float) * c.leverage * vol_scale
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w
