"""Category I: Cross-Asset Momentum strategies.

Long-biased momentum across ETFs with trend filters and vol scaling.
Academic basis: Moskowitz, Ooi, Pedersen 2012; Asness et al. 2013.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financial_algo.indicators import ema, kst, realized_vol, tsi
from financial_algo.strategies.base import Strategy


# =========================================================================
# I1 — Time-Series Momentum (TSMOM) — Long-only with trend filter
# =========================================================================

@dataclass
class TSMOMConfig:
    """Config for time-series momentum strategy."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "GLD", "TLT", "XLE", "UUP", "HYG",
    ])
    lookback: int = 252      # ~12 months
    skip: int = 21           # skip most recent month
    trend_window: int = 200  # SMA trend filter
    vol_window: int = 60
    target_vol: float = 0.15
    max_weight: float = 0.30
    leverage: float = 1.5


class TimeSeriesMomentum(Strategy):
    """Long assets with positive 12-1 month momentum AND above trend.

    Thesis: Assets with positive past returns above their long-term
    trend tend to continue rising. Long-only removes the short-side
    drag that kills classic TSMOM in persistent bull markets.
    """

    name = "I1-TimeSeriesMomentum"

    def __init__(self, config: TSMOMConfig | None = None) -> None:
        self.cfg = config or TSMOMConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        p = prices[avail]

        # 12-1 month momentum signal
        ret = p.pct_change(c.lookback).shift(c.skip)

        # Long-only: positive momentum only
        signal = (ret > 0).astype(float)

        # Trend filter: TSI > 0 (double-EWM momentum, less whipsaw than SMA)
        tsi_val = pd.DataFrame(np.nan, index=p.index, columns=avail)
        for t in avail:
            tsi_val[t] = tsi(p[t])
        trend_up = (tsi_val > 0).astype(float).fillna(0.0)
        signal = signal * trend_up

        # Vol-scale each position
        rvol = p.pct_change().fillna(0.0).rolling(c.vol_window).std() * np.sqrt(252)
        rvol = rvol.clip(lower=0.05)
        vol_scale = (c.target_vol / rvol).clip(upper=3.0).fillna(0.0)

        # Normalize by number of active positions
        n_active = signal.sum(axis=1).clip(lower=1)
        weights = signal * vol_scale * c.leverage
        weights = weights.div(n_active, axis=0)
        weights = weights.clip(0, c.max_weight)

        return weights.reindex(columns=prices.columns, fill_value=0.0).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)


# =========================================================================
# I2 — Cross-Sectional Momentum — Long-only sector rotation
# =========================================================================

@dataclass
class XSMOMConfig:
    """Config for cross-sectional momentum strategy."""

    tickers: list[str] = field(default_factory=lambda: [
        "XLK", "XLF", "XLI", "XLB", "XLP",
        "XLU", "XLY", "XLV", "XLE", "GLD",
    ])
    lookback: int = 63        # 3-month returns
    skip: int = 21            # skip last month (reversal avoidance)
    top_n: int = 5            # long top N
    bottom_n: int = 0         # no shorts
    trend_window: int = 200   # SMA trend filter
    leverage: float = 1.5


class CrossSectionalMomentum(Strategy):
    """Long top-N momentum sectors with trend filter.

    Thesis: Relative winners among sector ETFs continue to outperform.
    Long-only with trend filter avoids shorting into secular trends.
    """

    name = "I2-CrossSectionalMomentum"

    def __init__(self, config: XSMOMConfig | None = None) -> None:
        self.cfg = config or XSMOMConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        p = prices[avail]

        # Momentum signal (skip last month to avoid reversal)
        ret = p.pct_change(c.lookback).shift(c.skip)

        # Trend filter: TSI > 0 per asset (faster and smoother than SMA)
        tsi_val = pd.DataFrame(np.nan, index=p.index, columns=avail)
        for t in avail:
            tsi_val[t] = tsi(p[t])
        trend_up = tsi_val > 0

        # Mask out assets not in uptrend
        filtered_ret = ret.where(trend_up, np.nan)

        # Rank and select top N
        ranks = filtered_ret.rank(axis=1, ascending=False)
        long_mask = ranks <= c.top_n

        # Only allocate to assets with valid (non-NaN) momentum
        long_mask = long_mask & pd.notna(filtered_ret)

        n_long = long_mask.sum(axis=1).clip(lower=1)

        weights = pd.DataFrame(0.0, index=prices.index, columns=avail)
        weights[long_mask] = 1.0
        weights = weights.div(n_long, axis=0) * c.leverage

        return weights.reindex(columns=prices.columns, fill_value=0.0).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)


# =========================================================================
# I3 — Dual Momentum (Absolute + Relative) — Vectorized
# =========================================================================

@dataclass
class DualMomentumConfig:
    """Config for dual momentum strategy."""

    risk_on: list[str] = field(default_factory=lambda: ["SPY", "QQQ", "EFA", "EEM"])
    safe_assets: list[str] = field(default_factory=lambda: ["TLT", "GLD"])
    abs_lookback: int = 252     # absolute momentum window
    rel_lookback: int = 126     # relative momentum window
    vol_window: int = 20
    vol_threshold: float = 0.22  # crash filter
    leverage: float = 1.5

    @property
    def safe_asset(self) -> str:
        """Primary safe asset for backwards compatibility."""
        return self.safe_assets[0] if self.safe_assets else "TLT"


class DualMomentum(Strategy):
    """Dual momentum: best risk-on asset if momentum > 0, else safe havens.

    Thesis: Combine absolute momentum (trend filter) with relative
    momentum (best asset selection). Switch to TLT+GLD during
    downtrends or high vol. Vectorized implementation.
    """

    name = "I3-DualMomentum"

    def __init__(self, config: DualMomentumConfig | None = None) -> None:
        self.cfg = config or DualMomentumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail_risk = [t for t in c.risk_on if t in prices.columns]
        safe_avail = [t for t in c.safe_assets if t in prices.columns]

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if not avail_risk:
            return weights

        p = prices[avail_risk]

        # Composite momentum: average of 3m, 6m, 12m for stability
        ret_3m = p.pct_change(63).fillna(0.0)
        ret_6m = p.pct_change(c.rel_lookback).fillna(0.0)
        ret_12m = p.pct_change(c.abs_lookback).fillna(0.0)
        composite_ret = (ret_3m + ret_6m + ret_12m) / 3.0

        # Best risk-on asset by composite momentum (vectorized)
        best_asset = composite_ret.idxmax(axis=1)
        all_nan = composite_ret.isna().all(axis=1)
        best_asset[all_nan] = np.nan

        # Composite momentum of best asset
        best_abs = pd.Series(np.nan, index=prices.index)
        for t in avail_risk:
            mask = best_asset == t
            best_abs = best_abs.where(~mask, composite_ret[t])

        # Crash filter
        spy_col = "SPY" if "SPY" in prices.columns else avail_risk[0]
        spy_vol = realized_vol(prices[spy_col], c.vol_window)
        crash = spy_vol > c.vol_threshold

        # Trend filter: SPY TSI > 0 replaces SMA(200) for faster trend detection
        trend_ok = (tsi(prices[spy_col]) > 0).fillna(False)

        # Risk-on: best asset with positive momentum, no crash, trend OK
        risk_on_ok = (best_abs > 0) & ~crash & trend_ok

        for t in avail_risk:
            is_best = best_asset == t
            weights[t] = np.where(risk_on_ok & is_best, c.leverage, 0.0)

        # Safe haven: split across available safe assets
        safe_mode = ~risk_on_ok
        if safe_avail:
            per_safe = c.leverage / len(safe_avail)
            for t in safe_avail:
                weights[t] = np.where(safe_mode, per_safe, weights[t])

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# I4 — Momentum with Volatility Scaling — Long-only
# =========================================================================

@dataclass
class MomVolScaledConfig:
    """Momentum strategy with per-asset volatility scaling."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "GLD", "TLT", "XLE", "HYG", "UUP",
        "XLK", "XLF",
    ])
    lookback: int = 126        # 6-month momentum
    skip: int = 21             # skip last month
    trend_window: int = 200    # SMA trend filter
    vol_window: int = 40
    target_vol: float = 0.12
    max_weight: float = 0.25
    leverage: float = 1.5


class MomentumVolScaled(Strategy):
    """Long-only momentum with inverse-vol position sizing.

    Thesis: Long assets with positive momentum, sized inversely to
    their volatility. Long-only avoids the catastrophic short-side
    drag seen in long-short momentum. Vol scaling equalizes risk
    contribution across positions.
    """

    name = "I4-MomentumVolScaled"

    def __init__(self, config: MomVolScaledConfig | None = None) -> None:
        self.cfg = config or MomVolScaledConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        if len(avail) < 3:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        p = prices[avail]
        ret = p.pct_change(c.lookback).shift(c.skip)

        # Long-only: positive momentum only
        signal = (ret > 0).astype(float)

        # Trend filter: TSI > 0 per asset
        tsi_val = pd.DataFrame(np.nan, index=p.index, columns=avail)
        for t in avail:
            tsi_val[t] = tsi(p[t])
        trend_up = (tsi_val > 0).astype(float).fillna(0.0)
        signal = signal * trend_up

        # Vol scaling per asset
        daily_ret = p.pct_change().fillna(0.0)
        rvol = daily_ret.rolling(c.vol_window).std() * np.sqrt(252)
        rvol = rvol.clip(lower=0.05)
        inv_vol = (c.target_vol / rvol).clip(upper=3.0).fillna(0.0)

        # Normalize by number of active positions
        n_active = signal.sum(axis=1).clip(lower=1)
        raw = signal * inv_vol * c.leverage
        raw = raw.div(n_active, axis=0)
        raw = raw.clip(0, c.max_weight)

        return raw.reindex(columns=prices.columns, fill_value=0.0).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)


# =========================================================================
# I5 — Global Momentum Rotation — Long-only, vol-scaled, top-5
# =========================================================================

@dataclass
class GlobalMomRotConfig:
    """Config for global momentum rotation strategy."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "FXI", "VGK", "EWJ", "INDA",
        "GLD", "SLV", "TLT", "DBC", "VNQ", "XBI",
    ])
    lookback: int = 252       # ~12 months
    skip: int = 21            # skip most recent month
    trend_window: int = 200   # 200-day SMA filter
    vol_window: int = 60
    target_vol: float = 0.15
    top_n: int = 5
    max_weight: float = 0.30
    leverage: float = 1.5


class GlobalMomentumRotation(Strategy):
    """Top-5 global momentum rotation with trend filter and vol scaling.

    Thesis: Time-series momentum across a broad global universe
    captures persistent trends in equities, commodities, bonds, and
    real estate. Selecting the top-5 by 12-1 month return concentrates
    exposure on the strongest trends. The 200-day SMA filter avoids
    catching falling knives, and vol-scaling equalizes risk contribution.
    Long-only avoids short-side drag.
    """

    name = "I5-GlobalMomentumRotation"

    def __init__(self, config: GlobalMomRotConfig | None = None) -> None:
        self.cfg = config or GlobalMomRotConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        if len(avail) < c.top_n:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        p = prices[avail]

        # 12-1 month momentum signal (skip last month to avoid reversal)
        mom = p.pct_change(c.lookback).shift(c.skip).fillna(0.0)

        # Trend filter: TSI > 0 per asset (more responsive than SMA)
        tsi_val = pd.DataFrame(np.nan, index=p.index, columns=avail)
        for t in avail:
            tsi_val[t] = tsi(p[t])
        trend_up = tsi_val > 0

        # Mask out assets not in uptrend
        filtered_mom = mom.where(trend_up, np.nan)

        # Rank and select top N by momentum
        ranks = filtered_mom.rank(axis=1, ascending=False)
        selected = (ranks <= c.top_n) & pd.notna(filtered_mom)

        # Vol-scale each position
        rvol = p.pct_change().fillna(0.0).rolling(c.vol_window).std() * np.sqrt(252)
        rvol = rvol.clip(lower=0.05)
        inv_vol = (c.target_vol / rvol).clip(upper=3.0).fillna(0.0)

        # Weight = selected * inverse vol
        raw = selected.astype(float) * inv_vol

        # Normalize: sum to leverage
        total = raw.sum(axis=1).clip(lower=1e-8)
        raw = raw.div(total, axis=0) * c.leverage
        raw = raw.clip(0, c.max_weight)

        return raw.reindex(columns=prices.columns, fill_value=0.0).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)


# =========================================================================
# I6 — KST Momentum (Know Sure Thing multi-horizon)
# =========================================================================

@dataclass
class KSTMomentumConfig:
    """Config for KST momentum strategy."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "GLD", "TLT", "XLE", "UUP", "HYG",
        "XLK", "XLF",
    ])
    r1: int = 10
    r2: int = 15
    r3: int = 20
    r4: int = 30
    signal_period: int = 9
    trend_window: int = 150     # price > SMA filter
    top_n: int = 5              # number of assets to hold
    vol_window: int = 60
    target_vol: float = 0.15
    leverage: float = 1.5
    max_weight: float = 0.30


class KSTMomentum(Strategy):
    """Multi-horizon momentum strategy using the Know Sure Thing (KST) indicator.

    Thesis: KST combines four ROC horizons into one smooth momentum oscillator.
    A KST crossover above its signal line is a reliable trend-entry trigger
    (Pring 1992). Multi-asset: hold the top N assets with positive crossover.
    """

    name = "I6-KSTMomentum"

    def __init__(self, config: KSTMomentumConfig | None = None) -> None:
        self.cfg = config or KSTMomentumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weight_df = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(avail) < 2:
            return weight_df

        # Compute KST for each asset (ticker loop, not row loop)
        kst_val = pd.DataFrame(np.nan, index=prices.index, columns=avail)
        kst_sig = pd.DataFrame(np.nan, index=prices.index, columns=avail)
        for t in avail:
            k = kst(prices[t], r1=c.r1, r2=c.r2, r3=c.r3, r4=c.r4,
                    signal_period=c.signal_period)
            kst_val[t] = k["kst"]
            kst_sig[t] = k["kst_signal"]

        # Bullish: KST above signal AND KST rising
        kst_above = kst_val > kst_sig
        kst_rising = kst_val > kst_val.shift(1)

        # Trend filter: price above SMA
        sma = prices[avail].rolling(c.trend_window).mean()
        trend_up = prices[avail] > sma

        bullish = kst_above & kst_rising & trend_up

        # Select top N assets ranked by KST strength
        kst_strength = kst_val.where(bullish, np.nan)
        ranks = kst_strength.rank(axis=1, ascending=False)
        selected = (ranks <= c.top_n) & pd.notna(kst_strength)

        # Vol-scale each position
        rvol = prices[avail].pct_change().fillna(0.0).rolling(c.vol_window).std() * np.sqrt(252)
        rvol = rvol.clip(lower=0.05)
        vol_scale = (c.target_vol / rvol).clip(upper=3.0).fillna(0.0)

        raw = selected.astype(float) * vol_scale
        total = raw.sum(axis=1).clip(lower=1e-8)
        raw = (raw.div(total, axis=0) * c.leverage).clip(0, c.max_weight)

        for t in avail:
            weight_df[t] = raw[t]

        return weight_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
