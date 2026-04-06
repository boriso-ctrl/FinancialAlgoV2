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
    vol_window: int = 20      # volatility window for crash filter
    vol_threshold: float = 0.35  # max vol to trade (daily ~2.2% = annualized ~35%)
    leverage: float = 1.5


class CrossSectionalMomentum(Strategy):
    """Long top-N momentum sectors with trend + crash filters.

    Thesis: Relative winners among sector ETFs continue to outperform.
    Long-only with trend and crash filters to avoid buying into crashes.
    Skip trading when market volatility spikes to reduce drawdowns.
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

        # Crash/volatility filter: market-wide vol too high -> skip trading
        spy_col = "SPY" if "SPY" in prices.columns else None
        if spy_col:
            spy_vol = realized_vol(prices[spy_col], c.vol_window)
            crash_ok = (spy_vol <= c.vol_threshold).fillna(False)
        else:
            crash_ok = pd.Series(True, index=p.index)

        # Mask out assets not in uptrend
        filtered_ret = ret.where(trend_up & crash_ok.values[:, np.newaxis], np.nan)

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
        best_asset = best_asset.where(~all_nan)

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


# I8_APPEND_MARKER


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
    lookback: int = 252       # ~12 months (12-1)
    lookback_mid: int = 126   # ~6 months (6-1)
    lookback_fast: int = 63   # ~3 months (3-1)
    skip: int = 21            # skip most recent month
    trend_window: int = 200   # long trend filter
    trend_fast_window: int = 50
    vol_window: int = 60
    target_vol: float = 0.15
    min_score: float = 0.0
    top_n: int = 5
    max_weight: float = 0.25
    leverage: float = 1.35
    crash_leverage: float = 0.45
    min_breadth: float = 0.25
    min_risk_off_defensives: int = 2
    defensive_tickers: list[str] = field(default_factory=lambda: ["GLD", "TLT", "UUP", "SHY"])


class GlobalMomentumRotation(Strategy):
    """Top-N global momentum rotation with crash-aware risk scaling.

    Thesis: Time-series momentum across a broad global universe
    captures persistent trends in equities, commodities, bonds, and
    real estate. Multi-horizon 12-1/6-1/3-1 ranking reduces single-window
    noise. A trend filter and market crash gate reduce left-tail risk,
    while inverse-vol scaling equalizes risk contribution. Long-only avoids
    short-side drag.
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
        if len(avail) < max(c.top_n, 2):
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        p = prices[avail].copy()

        # Multi-horizon momentum blend (12-1, 6-1, 3-1)
        mom_12 = p.pct_change(c.lookback).shift(c.skip)
        mom_6 = p.pct_change(c.lookback_mid).shift(c.skip)
        mom_3 = p.pct_change(c.lookback_fast).shift(c.skip)
        mom_blend = (0.5 * mom_12) + (0.3 * mom_6) + (0.2 * mom_3)

        # Risk-adjusted score reduces concentration in volatile assets
        daily_ret = p.pct_change().fillna(0.0)
        rvol = (daily_ret.rolling(c.vol_window).std() * np.sqrt(252)).clip(lower=0.05)
        score = (mom_blend / rvol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)

        # Asset trend gate: price above both fast and slow trend
        sma_fast = p.rolling(c.trend_fast_window).mean()
        sma_slow = p.rolling(c.trend_window).mean()
        trend_up = ((p > sma_slow) & (sma_fast > sma_slow)).fillna(False)

        # Market crash gate: cut leverage and allow only defensives when SPY weak
        if "SPY" in prices.columns:
            spy = prices["SPY"]
            spy_sma = spy.rolling(c.trend_window).mean()
            spy_mom = spy.pct_change(c.lookback_mid).shift(c.skip)
            risk_on = ((spy > spy_sma) & (spy_mom > 0)).fillna(False)
        else:
            risk_on = pd.Series(True, index=prices.index)

        # Breadth gate: require enough assets with positive, trend-confirmed score.
        base_tradable = trend_up & pd.notna(score) & (score > c.min_score)
        breadth = base_tradable.mean(axis=1).fillna(0.0)
        risk_on = risk_on & (breadth >= c.min_breadth)

        defensive_cols = [t for t in c.defensive_tickers if t in avail]
        defensive_mask = pd.DataFrame(False, index=prices.index, columns=avail)
        if defensive_cols:
            defensive_mask.loc[:, defensive_cols] = True

        tradable_mask = base_tradable
        tradable_mask = tradable_mask & (risk_on.to_numpy()[:, np.newaxis] | defensive_mask)
        candidate_score = score.where(tradable_mask, np.nan)

        ranks = candidate_score.rank(axis=1, ascending=False, method="average")
        selected = (ranks <= c.top_n) & pd.notna(candidate_score)

        inv_vol = (c.target_vol / rvol).clip(upper=3.0).fillna(0.0)
        positive_score = candidate_score.clip(lower=0.0).fillna(0.0)
        raw = selected.astype(float) * inv_vol * positive_score

        # Risk-off fallback: explicitly rotate into strongest defensive assets.
        risk_off_raw = pd.DataFrame(0.0, index=prices.index, columns=avail)
        if defensive_cols:
            defensive_score = score[defensive_cols].where(trend_up[defensive_cols], np.nan)
            defensive_rank = defensive_score.rank(axis=1, ascending=False, method="average")
            n_def = max(1, min(c.min_risk_off_defensives, len(defensive_cols)))
            defensive_selected = (defensive_rank <= n_def) & pd.notna(defensive_score)
            defensive_inv_vol = inv_vol[defensive_cols]
            risk_off_raw.loc[:, defensive_cols] = (
                defensive_selected.astype(float) * defensive_inv_vol * defensive_score.clip(lower=0.0).fillna(0.0)
            )

        gross = raw.sum(axis=1).replace(0.0, np.nan)
        risk_on_weights = raw.div(gross, axis=0).fillna(0.0)

        risk_off_gross = risk_off_raw.sum(axis=1).replace(0.0, np.nan)
        risk_off_weights = risk_off_raw.div(risk_off_gross, axis=0).fillna(0.0)

        risk_on_gate = risk_on.to_numpy()[:, np.newaxis]
        mixed = pd.DataFrame(
            np.where(risk_on_gate, risk_on_weights.values, risk_off_weights.values),
            index=prices.index,
            columns=avail,
        )
        leverage_t = pd.Series(np.where(risk_on, c.leverage, c.crash_leverage), index=prices.index, dtype=float)
        weights_avail = mixed.multiply(leverage_t, axis=0)
        weights_avail = weights_avail.clip(lower=0.0, upper=c.max_weight)

        return weights_avail.reindex(columns=prices.columns, fill_value=0.0).replace(
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
    trend_window: int = 100     # price > SMA filter (100d is less restrictive)
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

        # Bullish: KST above signal line.
        # Dropped the "kst_rising" requirement — triple-AND was too restrictive
        # and left the portfolio in cash most of the time.
        kst_above = kst_val > kst_sig

        # Trend filter: price above SMA
        sma = prices[avail].rolling(c.trend_window).mean()
        trend_up = prices[avail] > sma

        bullish = kst_above & trend_up

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


# =========================================================================
# I10 — Adaptive Trend Filter (Wave 2 rework)
# =========================================================================

@dataclass
class AdaptiveTrendFilterConfigV2:
    """Config for adaptive cross-asset trend filter."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "UUP",
        "XLE", "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
        "HYG", "LQD", "SLV", "SHY", "DBC", "DBA", "TIP", "AGG", "EMB",
        "FXI", "VGK", "EWJ", "INDA", "VNQ", "XBI",
    ])
    mom_windows: tuple[int, ...] = (21, 63, 126, 252)
    base_weights: tuple[float, ...] = (0.35, 0.30, 0.20, 0.15)
    stressed_weights: tuple[float, ...] = (0.10, 0.20, 0.30, 0.40)
    trend_fast: int = 63
    trend_slow: int = 126
    market_trend_window: int = 200
    vol_window: int = 20
    pos_vol_window: int = 60
    high_vol_threshold: float = 0.28
    long_n: int = 6
    target_vol: float = 0.16
    leverage: float = 1.20
    risk_off_leverage: float = 0.60
    max_weight: float = 0.30
    defensive_assets: tuple[str, ...] = ("TLT", "IEF", "GLD", "SHY")


class AdaptiveTrendFilterWave2(Strategy):
    """Adaptive long-only trend with volatility-aware risk budget."""

    name = "I10-AdaptiveTrendFilter"

    def __init__(self, config: AdaptiveTrendFilterConfigV2 | None = None) -> None:
        super().__init__()
        self.cfg = config or AdaptiveTrendFilterConfigV2()

    @staticmethod
    def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0.0, np.nan)
        z = df.sub(mu, axis=0).div(sd, axis=0)
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(avail) < c.long_n:
            return weights

        p = prices[avail]
        ret = p.pct_change().fillna(0.0)

        mom_components: list[pd.DataFrame] = []
        for w in c.mom_windows:
            mom = p.pct_change(w).replace([np.inf, -np.inf], np.nan)
            mom_components.append(self._cs_zscore(mom))

        spy_col = "SPY" if "SPY" in prices.columns else avail[0]
        spy = prices[spy_col]
        spy_ret = spy.pct_change().fillna(0.0)
        spy_vol = spy_ret.rolling(c.vol_window, min_periods=10).std() * np.sqrt(252)
        spy_vol = spy_vol.replace([np.inf, -np.inf], np.nan).fillna(c.high_vol_threshold)
        alpha = (spy_vol / max(c.high_vol_threshold, 1e-8)).clip(0.0, 1.0)

        base_w = np.array(c.base_weights)
        stress_w = np.array(c.stressed_weights)
        adapt_w = (
            (1.0 - alpha.to_numpy()[:, np.newaxis]) * base_w[np.newaxis, :]
            + alpha.to_numpy()[:, np.newaxis] * stress_w[np.newaxis, :]
        )

        mom_stack = np.stack([m.to_numpy() for m in mom_components], axis=2)
        combo = np.sum(mom_stack * adapt_w[:, np.newaxis, :], axis=2)
        combo_df = pd.DataFrame(combo, index=p.index, columns=p.columns)
        combo_df = combo_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        ema_fast = p.ewm(span=c.trend_fast, adjust=False).mean()
        ema_slow = p.ewm(span=c.trend_slow, adjust=False).mean()
        trend_up = (p > ema_fast) & (ema_fast > ema_slow)

        market_ma = spy.rolling(c.market_trend_window, min_periods=50).mean()
        risk_on = (spy > market_ma) & (spy_vol <= c.high_vol_threshold)

        score = combo_df.where(trend_up & (combo_df > 0.0), np.nan)
        rank = score.rank(axis=1, ascending=False, method="average")
        selected = (rank <= c.long_n) & pd.notna(score)

        rvol = ret.rolling(c.pos_vol_window, min_periods=20).std() * np.sqrt(252)
        rvol = rvol.replace(0.0, np.nan).fillna(c.target_vol)
        inv_vol = (c.target_vol / rvol).clip(lower=0.2, upper=4.0).fillna(0.0)

        raw = selected.astype(float) * inv_vol * score.clip(lower=0.0).fillna(0.0)
        gross = raw.sum(axis=1).replace(0.0, np.nan)
        long_weights = raw.div(gross, axis=0).fillna(0.0)

        def_cols = [t for t in c.defensive_assets if t in avail]
        defensive_weights = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        if def_cols:
            defensive_weights.loc[:, def_cols] = 1.0 / float(len(def_cols))

        on_vals = long_weights.to_numpy()
        off_vals = defensive_weights.to_numpy()
        use_on = risk_on.to_numpy()[:, np.newaxis]
        mixed = pd.DataFrame(np.where(use_on, on_vals, off_vals), index=p.index, columns=p.columns)

        lev_t = pd.Series(
            np.where(risk_on, c.leverage, c.risk_off_leverage),
            index=p.index,
            dtype=float,
        )
        scaled = mixed.mul(lev_t, axis=0).clip(lower=0.0, upper=c.max_weight)

        weights.loc[:, avail] = scaled
        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# Export Wave 2 implementation under canonical strategy name.
class AdaptiveTrendFilter(AdaptiveTrendFilterWave2):
    pass


# =========================================================================
# I7 — Drift Regime Momentum (Wave 2 rework)
# =========================================================================

@dataclass
class DriftRegimeMomentumConfig:
    """Config for drift-gated cross-sectional momentum."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV", "XLE",
        "GLD", "TLT", "DBC", "VNQ", "HYG",
    ])
    market_proxy: str = "SPY"
    fast_mom: int = 63
    slow_mom: int = 126
    trend_window: int = 100
    drift_window: int = 63
    drift_threshold: float = 0.54
    vol_window: int = 40
    target_vol: float = 0.14
    high_vol_threshold: float = 0.30
    top_n_long: int = 5
    top_n_short: int = 2
    long_leverage: float = 1.10
    short_leverage: float = 0.35
    rebalance_freq: int = 10
    max_weight: float = 0.25


class DriftRegimeMomentum(Strategy):
    """Cross-sectional momentum with drift persistence gate and crash controls."""

    name = "I7-DriftRegimeMomentum"

    def __init__(self, config: DriftRegimeMomentumConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or DriftRegimeMomentumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(avail) < (c.top_n_long + c.top_n_short):
            return weights

        p = prices[avail]
        ret = p.pct_change().fillna(0.0)
        mom_fast = p.pct_change(c.fast_mom).replace([np.inf, -np.inf], np.nan)
        mom_slow = p.pct_change(c.slow_mom).replace([np.inf, -np.inf], np.nan)
        mom = (0.6 * mom_fast) + (0.4 * mom_slow)

        drift_ratio = (ret > 0.0).astype(float).rolling(c.drift_window, min_periods=20).mean()
        drift_ratio = drift_ratio.fillna(0.0)

        ema = p.ewm(span=c.trend_window, adjust=False).mean()
        trend_up = p > ema
        trend_dn = p < ema

        market_col = c.market_proxy if c.market_proxy in prices.columns else avail[0]
        mkt_ret = prices[market_col].pct_change().fillna(0.0)
        mkt_vol = mkt_ret.rolling(c.vol_window, min_periods=10).std() * np.sqrt(252)
        mkt_vol = mkt_vol.replace([np.inf, -np.inf], np.nan).fillna(c.high_vol_threshold)
        crash = mkt_vol > c.high_vol_threshold

        long_eligible = (drift_ratio >= c.drift_threshold) & trend_up
        short_eligible = (
            (drift_ratio <= (1.0 - c.drift_threshold))
            & trend_dn
            & (~crash).reindex(p.index).fillna(False)
        )

        long_score = mom.where(long_eligible, np.nan)
        short_score = mom.where(short_eligible, np.nan)

        long_rank = long_score.rank(axis=1, ascending=False, method="average")
        short_rank = short_score.rank(axis=1, ascending=True, method="average")

        long_mask = (long_rank <= c.top_n_long) & pd.notna(long_score)
        short_mask = (short_rank <= c.top_n_short) & pd.notna(short_score)
        short_mask = short_mask & ~long_mask

        rvol = ret.rolling(c.vol_window, min_periods=20).std() * np.sqrt(252)
        rvol = rvol.replace(0.0, np.nan).fillna(c.target_vol)
        inv_vol = (c.target_vol / rvol).clip(lower=0.2, upper=4.0).fillna(0.0)

        long_raw = long_mask.astype(float) * inv_vol * long_score.clip(lower=0.0).fillna(0.0)
        short_raw = short_mask.astype(float) * inv_vol * short_score.abs().fillna(0.0)

        long_sum = long_raw.sum(axis=1).replace(0.0, np.nan)
        short_sum = short_raw.sum(axis=1).replace(0.0, np.nan)
        long_w = long_raw.div(long_sum, axis=0).fillna(0.0) * c.long_leverage
        short_w = short_raw.div(short_sum, axis=0).fillna(0.0) * c.short_leverage * -1.0

        raw = (long_w + short_w).clip(lower=-c.max_weight, upper=c.max_weight)

        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::max(c.rebalance_freq, 1)] = True
        held = raw.where(rebal_mask, axis=0).ffill().fillna(0.0)
        held = held.where(~crash, held.clip(lower=0.0), axis=0)

        weights.loc[:, avail] = held
        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# I7 — Sector Rotation Momentum — Dual-momentum sector ETFs, vol-scaled
# =========================================================================

@dataclass
class SectorRotationMomConfig:
    """Config for sector rotation momentum strategy."""

    sectors: list[str] = field(default_factory=lambda: [
        "XLK", "XLF", "XLI", "XLB", "XLP",
        "XLU", "XLY", "XLV", "XLE", "XLRE", "XLC",
    ])
    benchmark: str = "SPY"

    mom_fast: int = 63         # 3-month momentum
    mom_slow: int = 252        # 12-month momentum
    skip: int = 5              # skip last week (reversal avoidance)
    vol_window: int = 60
    target_vol: float = 0.15

    rebalance_freq: int = 21   # monthly rebalance
    top_n: int = 4             # long top-N sectors
    max_weight: float = 0.35
    leverage: float = 1.5


class SectorRotationMomentum(Strategy):
    """Dual-timeframe sector rotation: long top-N sector ETFs.

    Thesis: Sector ETFs exhibit persistent momentum from three sources:
    (1) analyst herding within sectors, (2) institutional sector-fund
    inflows that are sticky, and (3) macro-regime persistence (e.g. tech
    outperforms in loose-money regimes). Combining 3-month and 12-month
    momentum captures both intermediate and long-term trends; filtering
    by absolute momentum (sector > benchmark) avoids holding sectors in
    absolute downtrends. Monthly rebalance keeps turnover manageable.

    Documented: Moskowitz & Grinblatt (1999) — "Do Industries Explain
    Momentum?" demonstrated sector momentum is the primary driver of
    individual stock momentum.

    Signal: Composite momentum = 0.5 * rank(3m ret) + 0.5 * rank(12m ret),
    filtered by sector > SPY over 3m (absolute momentum).
    Portfolio: Top-N sectors, vol-scaled, monthly rebalance.
    """

    name = "I7-SectorRotationMomentum"

    def __init__(self, config: SectorRotationMomConfig | None = None) -> None:
        self.cfg = config or SectorRotationMomConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        avail = [t for t in c.sectors if t in prices.columns]
        if len(avail) < c.top_n:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        p = prices[avail]

        # Dual momentum scores (skip last week to avoid reversal)
        mom_fast = p.pct_change(c.mom_fast).shift(c.skip).fillna(0.0)
        mom_slow = p.pct_change(c.mom_slow).shift(c.skip).fillna(0.0)

        # Cross-sectional rank (pct) for each horizon
        rank_fast = mom_fast.rank(axis=1, pct=True).fillna(0.5)
        rank_slow = mom_slow.rank(axis=1, pct=True).fillna(0.5)

        # Composite: equal-weight blend
        composite = 0.5 * rank_fast + 0.5 * rank_slow

        # Absolute momentum filter: sector must beat benchmark over fast window
        if c.benchmark in prices.columns:
            bm_ret = prices[c.benchmark].pct_change(c.mom_fast).shift(c.skip).fillna(0.0)
            abs_filter = mom_fast.gt(bm_ret, axis=0)
        else:
            abs_filter = mom_fast > 0

        # Apply absolute momentum filter
        composite_filtered = composite.where(abs_filter, np.nan)

        # Rank top-N
        ranks = composite_filtered.rank(axis=1, ascending=False)
        selected = (ranks <= c.top_n) & pd.notna(composite_filtered)

        # Vol-scale each position
        rvol = p.pct_change().fillna(0.0).rolling(c.vol_window).std() * np.sqrt(252)
        rvol = rvol.clip(lower=0.05)
        inv_vol = (c.target_vol / rvol).clip(upper=3.0).fillna(0.0)

        raw = selected.astype(float) * inv_vol

        # Normalize to target leverage
        total = raw.sum(axis=1).clip(lower=1e-8)
        raw = raw.div(total, axis=0) * c.leverage
        raw = raw.clip(0, c.max_weight)

        # Monthly rebalance hold
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        raw = raw.where(rebal_mask).ffill().fillna(0.0)

        return raw.reindex(columns=prices.columns, fill_value=0.0).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)


# =========================================================================
# I8 — Channel Breakout (Donchian multi-horizon)
# =========================================================================

@dataclass
class ChannelBreakoutConfig:
    """Config for Donchian channel breakout momentum."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "TLT", "GLD", "QQQ", "IWM", "EFA", "EEM", "DBC", "VNQ", "HYG",
    ])
    lookbacks: tuple[int, ...] = (20, 40, 80)
    vol_window: int = 40
    target_vol: float = 0.14
    leverage: float = 1.25
    max_weight: float = 0.30


class ChannelBreakout(Strategy):
    """Multi-horizon Donchian breakout, long-only.

    Thesis: New highs across multiple horizons indicate persistent trend
    strength driven by institutional flow and slow information diffusion.
    """

    name = "I8-ChannelBreakout"

    def __init__(self, config: ChannelBreakoutConfig | None = None) -> None:
        self.cfg = config or ChannelBreakoutConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if not avail:
            return weights

        p = prices[avail]

        breakout_score = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        for lb in c.lookbacks:
            prior_high = p.rolling(lb, min_periods=lb).max().shift(1)
            breakout = p.gt(prior_high) & pd.notna(prior_high)
            breakout_score = breakout_score + breakout.astype(float)
        breakout_score = breakout_score / max(len(c.lookbacks), 1)

        rvol = p.pct_change().fillna(0.0).rolling(c.vol_window).std() * np.sqrt(252)
        rvol = rvol.replace(0.0, np.nan).clip(lower=0.05)
        inv_vol = (c.target_vol / rvol).clip(upper=3.0).fillna(0.0)

        raw = breakout_score * inv_vol
        gross = raw.sum(axis=1).clip(lower=1e-8)
        scaled = raw.div(gross, axis=0) * c.leverage
        scaled = scaled.clip(lower=0.0, upper=c.max_weight)

        weights.loc[:, avail] = scaled
        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


class TimeSeriesMomentumDrift(ChannelBreakout):
    """Backward-compatible alias for legacy I8 strategy name."""

    name = "I8-TimeSeriesMomDrift"


# =========================================================================
# I9 — Acceleration Momentum (fast-vs-slow momentum spread)
# =========================================================================

@dataclass
class AccelerationMomentumConfig:
    """Config for acceleration momentum."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "DBC", "HYG", "XLK", "XLF",
    ])
    fast_windows: tuple[int, ...] = (4, 8, 16, 32)
    norm_window: int = 126
    trend_lookback: int = 63
    leverage: float = 1.25
    max_weight: float = 0.30


class AccelerationMomentum(Strategy):
    """Acceleration momentum with robust NaN-safe normalization.

    Thesis: Returns accelerate before broad trend followers fully reposition.
    Fast-vs-slow momentum differentials capture this convex phase transition.
    """

    name = "I9-AccelerationMomentum"

    def __init__(self, config: AccelerationMomentumConfig | None = None) -> None:
        self.cfg = config or AccelerationMomentumConfig()

    @staticmethod
    def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        mu = df.mean(axis=1)
        sd = df.std(axis=1).replace(0.0, np.nan)
        z = df.sub(mu, axis=0).div(sd, axis=0)
        return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if not avail:
            return weights

        p = prices[avail]

        accel_components: list[pd.DataFrame] = []
        for w in c.fast_windows:
            fast = p.pct_change(w)
            slow = p.pct_change(w * 2)
            accel = (fast - slow).replace([np.inf, -np.inf], np.nan)
            accel_components.append(self._cs_zscore(accel))

        accel_score = pd.DataFrame(0.0, index=p.index, columns=p.columns)
        for comp in accel_components:
            accel_score = accel_score + comp
        accel_score = accel_score / max(len(accel_components), 1)
        trend = p.pct_change(c.trend_lookback).replace([np.inf, -np.inf], np.nan)

        if len(avail) == 1:
            col = avail[0]
            ts_mean = accel_score[col].rolling(c.norm_window, min_periods=20).mean()
            ts_std = accel_score[col].rolling(c.norm_window, min_periods=20).std().replace(0.0, np.nan)
            ts_z = ((accel_score[col] - ts_mean) / ts_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            pos = ts_z.clip(lower=0.0)
            signal = (pos / (1.0 + pos)).fillna(0.0)
            signal = signal.where(trend[col].fillna(0.0) > 0.0, 0.0)
            weights[col] = (signal * c.leverage).clip(lower=0.0, upper=c.max_weight)
            return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        pos_score = accel_score.clip(lower=0.0)
        pos_score = pos_score.where(trend.fillna(0.0) > 0.0, 0.0)

        gross = pos_score.sum(axis=1).clip(lower=1e-8)
        scaled = pos_score.div(gross, axis=0) * c.leverage
        scaled = scaled.clip(lower=0.0, upper=c.max_weight)

        weights.loc[:, avail] = scaled
        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


