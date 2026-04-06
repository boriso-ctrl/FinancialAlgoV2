"""Category K: Factor-Based strategies.

Quality and low-volatility factor tilts using sector ETFs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financial_algo.indicators import realized_vol, ema
from financial_algo.strategies.base import Strategy


# =========================================================================
# K1 — Quality / Low-Volatility Factor
# =========================================================================

@dataclass
class LowVolConfig:
    """Config for low-volatility factor strategy."""

    tickers: list[str] | None = None
    vol_window: int = 60
    rebalance_freq: int = 21   # monthly rebalance
    top_n: int = 4             # long top-N lowest vol
    trend_window: int = 200    # SMA trend filter window
    trend_floor: float = 0.97  # soft trend gate: allow mild dips vs SMA
    min_inv_vol: float = 0.05
    leverage: float = 1.5

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = [
                "XLP", "XLU", "XLV", "XLK", "XLF",
                "XLI", "XLB", "XLY", "XLE",
            ]


class LowVolFactor(Strategy):
    """Long the lowest-volatility sector ETFs with trend filter.

    Thesis: Low-volatility anomaly — low-vol stocks/sectors deliver
    higher risk-adjusted returns than high-vol ones. Documented by
    Baker, Bradley, Wurgler (2011). Persistent across markets and time.
    Added trend filter to reduce drawdowns in prolonged downtrends.
    Provides defensive alpha with low drawdowns.
    """

    name = "K1-LowVolFactor"

    def __init__(self, config: LowVolConfig | None = None) -> None:
        self.cfg = config or LowVolConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        if len(avail) < c.top_n:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        px = prices[avail]

        # Vol score: lower realised vol ranks higher.
        vols = px.apply(lambda s: realized_vol(s, c.vol_window))
        vol_rank = 1.0 - vols.rank(axis=1, pct=True)

        # Soft trend score avoids all-or-nothing exposure around SMA.
        sma = px.rolling(c.trend_window, min_periods=50).mean()
        rel_to_sma = px.div(sma.replace(0.0, np.nan))
        trend_score = ((rel_to_sma - c.trend_floor) / (1.03 - c.trend_floor)).clip(lower=0.0, upper=1.0)
        trend_score = trend_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        composite = (0.80 * vol_rank + 0.20 * trend_score).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rank = composite.rank(axis=1, ascending=False, method="first")
        selected = rank <= c.top_n

        # Rebalance at fixed intervals and hold between rebalances.
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        selected_held = selected.where(rebal_mask).ffill().fillna(False)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        inv_vol = (1.0 / vols.clip(lower=c.min_inv_vol)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw = selected_held.astype(float) * inv_vol * trend_score
        gross = raw.sum(axis=1).replace(0.0, np.nan)
        alloc = raw.div(gross, axis=0).fillna(0.0) * c.leverage
        weights.loc[:, avail] = alloc

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# K2 — Multi-Factor Composite (Momentum + Low-Vol + Value Proxy)
# =========================================================================

@dataclass
class MultiFactorConfig:
    """Config for multi-factor strategy."""

    tickers: list[str] | None = None

    mom_window: int = 126       # 6-month momentum
    vol_window: int = 60        # vol for low-vol score
    rebalance_freq: int = 21    # monthly

    top_n: int = 3              # long top-N composite score
    bottom_n: int = 2           # short bottom-N

    leverage: float = 1.5

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = [
                "SPY", "QQQ", "IWM", "EFA", "EEM",
                "GLD", "TLT", "XLE", "XLK", "XLF",
            ]


class MultiFactorComposite(Strategy):
    """Long top-N assets by composite score (momentum + low-vol).

    Thesis: Multi-factor approaches are more robust than single-factor.
    Long-only avoids the short-side drag that kills long-short factors.
    """

    name = "K2-MultiFactorComposite"

    def __init__(self, config: MultiFactorConfig | None = None) -> None:
        self.cfg = config or MultiFactorConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]

        # Momentum score
        mom = prices[avail].pct_change(c.mom_window)
        mom_rank = mom.rank(axis=1, pct=True)

        # Low-vol score (inverse rank)
        vols = pd.DataFrame(index=prices.index)
        for t in avail:
            vols[t] = realized_vol(prices[t], c.vol_window)
        vol_rank = 1.0 - vols.rank(axis=1, pct=True)

        # Composite: 50% momentum + 50% low-vol
        composite = 0.5 * mom_rank.reindex(columns=avail) + 0.5 * vol_rank.reindex(columns=avail)

        # Trend filter: asset above 200-day SMA
        sma200 = prices[avail].rolling(200).mean()
        trend_up = prices[avail] > sma200
        composite = composite.where(trend_up, np.nan)

        # Rank: top N
        ranks = composite.rank(axis=1, ascending=False)
        selected = (ranks <= c.top_n) & pd.notna(composite)

        # Rebalance at fixed intervals
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        selected_held = selected.where(rebal_mask).ffill().fillna(False)

        # Vol-scaled position sizing within selected assets
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for t in avail:
            base = selected_held[t].astype(float)
            vol_t = vols[t].clip(lower=0.05)
            vol_scale = (0.15 / vol_t).clip(upper=3.0).fillna(1.0)
            weights[t] = base * vol_scale

        total_w = weights[avail].sum(axis=1).clip(lower=1e-8)
        for t in avail:
            weights[t] = weights[t] / total_w * c.leverage

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# K3 -- Size Factor (Small vs Large Cap)
# =========================================================================

@dataclass
class SizeFactorConfig:
    """Tilt between small-cap (IWM) and large-cap (SPY) based on credit cycle."""

    small_cap: str = "IWM"
    large_cap: str = "SPY"

    # Credit spread proxy for cycle detection
    hyg_ticker: str = "HYG"
    lqd_ticker: str = "LQD"

    spread_window: int = 63
    ema_span: int = 21

    leverage: float = 1.5


class SizeFactor(Strategy):
    """Tilt to small-cap in expansion, large-cap in contraction.

    Thesis: Small-cap stocks outperform in early-cycle / expansion
    (when credit spreads tighten), and underperform in late-cycle /
    contraction (when spreads widen). The size premium is cyclical.
    Perez-Quiros & Timmermann (2000) document this pattern.
    """

    name = "K3-SizeFactor"

    def __init__(self, config: SizeFactorConfig | None = None) -> None:
        self.cfg = config or SizeFactorConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.small_cap not in prices.columns or c.large_cap not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Credit spread proxy for cycle detection
        if c.hyg_ticker in prices.columns and c.lqd_ticker in prices.columns:
            spread = prices[c.hyg_ticker] / prices[c.lqd_ticker]
            spread_ema = ema(spread, c.ema_span)
            spread_mom = spread_ema.pct_change(c.spread_window).fillna(0.0)
            expansion = spread_mom > 0  # Spreads tightening = expansion
        else:
            # Fallback: IWM/SPY ratio momentum
            ratio = prices[c.small_cap] / prices[c.large_cap]
            expansion = ema(ratio, c.ema_span).pct_change(c.spread_window).fillna(0.0) > 0

        weights.loc[expansion, c.small_cap] = c.leverage
        weights.loc[~expansion, c.large_cap] = c.leverage

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# K4 -- Value Factor Proxy (sector-based)
# =========================================================================

@dataclass
class ValueFactorConfig:
    """Tilt towards traditionally 'value' sectors vs 'growth'."""

    value_tickers: tuple = ("XLF", "XLE", "XLI", "XLB")
    growth_tickers: tuple = ("XLK", "XLY", "QQQ")

    # Use relative mean-reversion: when value underperforms growth, tilt to value
    lookback: int = 126
    zscore_window: int = 60
    entry_z: float = 1.0

    leverage: float = 1.5


class ValueFactor(Strategy):
    """Quality-value rotation: long quality sectors in uptrends.

    Thesis: Instead of shorting growth (suicidal in tech bull runs),
    rotate into quality value sectors when they're trending up.
    Long-only approach captures value premium without fighting trends.
    """

    name = "K4-ValueFactor"

    def __init__(self, config: ValueFactorConfig | None = None) -> None:
        self.cfg = config or ValueFactorConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        all_sectors = list(set(list(c.value_tickers) + list(c.growth_tickers)))
        avail = [t for t in all_sectors if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        p = prices[avail]

        # Quality score: low vol + positive momentum
        vols = pd.DataFrame(index=prices.index)
        for t in avail:
            vols[t] = realized_vol(prices[t], 60)
        vol_rank = 1.0 - vols.rank(axis=1, pct=True)  # low vol = high score

        mom = p.pct_change(c.lookback)
        mom_rank = mom.rank(axis=1, pct=True)

        quality = 0.5 * vol_rank.reindex(columns=avail) + 0.5 * mom_rank.reindex(columns=avail)

        # Trend filter: only long sectors above 200-day SMA
        sma200 = p.rolling(200).mean()
        trend_up = p > sma200
        quality = quality.where(trend_up, np.nan)

        # Top 4 quality sectors
        ranks = quality.rank(axis=1, ascending=False)
        selected = (ranks <= 4) & pd.notna(quality)

        # Monthly rebalance
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::21] = True
        selected_held = selected.where(rebal_mask).ffill().fillna(False)

        # Vol-scaled position sizing
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for t in avail:
            base = selected_held[t].astype(float)
            vol_t = vols[t].clip(lower=0.05)
            vol_scale = (0.15 / vol_t).clip(upper=3.0).fillna(1.0)
            weights[t] = base * vol_scale

        total_w = weights[avail].sum(axis=1).clip(lower=1e-8)
        for t in avail:
            weights[t] = weights[t] / total_w * c.leverage

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# K5 — Real Assets Factor (inflation-linked rotation)
# =========================================================================

@dataclass
class RealAssetsConfig:
    """Config for real assets vs nominal assets factor strategy."""

    real_tickers: list[str] | None = None
    nominal_tickers: list[str] | None = None

    inflation_long: str = "TIP"
    inflation_short: str = "IEF"

    signal_window: int = 63     # ~3 months for inflation trend
    trend_window: int = 200     # SMA trend filter on each asset
    rebalance_freq: int = 21    # monthly rebalance
    leverage: float = 1.5

    def __post_init__(self) -> None:
        if self.real_tickers is None:
            self.real_tickers = ["VNQ", "DBC", "GLD", "SLV", "TIP"]
        if self.nominal_tickers is None:
            self.nominal_tickers = ["SPY", "TLT", "AGG"]


class RealAssetsFactor(Strategy):
    """Rotate between real and nominal asset baskets based on inflation.

    Thesis: When inflation expectations rise (TIP/IEF ratio trending up),
    real assets (commodities, gold, silver, REITs, TIPS) outperform
    nominal assets (equities, treasuries, agg bonds). When deflation
    risks dominate (TIP/IEF falling), nominal assets outperform.
    This captures the inflation risk premium documented by Ang (2014)
    and the real asset return pattern under different inflation regimes.
    Long-only both baskets -- tilt allocation, never go short.
    """

    name = "K5-RealAssetsFactor"

    def __init__(self, config: RealAssetsConfig | None = None) -> None:
        self.cfg = config or RealAssetsConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        real_avail = [t for t in c.real_tickers if t in prices.columns]
        nom_avail = [t for t in c.nominal_tickers if t in prices.columns]
        if not real_avail or not nom_avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Inflation signal: TIP/IEF ratio momentum
        has_signal = (c.inflation_long in prices.columns
                      and c.inflation_short in prices.columns)
        if has_signal:
            tip_ief = prices[c.inflation_long] / prices[c.inflation_short]
            # Guard against division issues
            tip_ief = tip_ief.replace([np.inf, -np.inf], np.nan).ffill()
            tip_ief_sma = tip_ief.rolling(c.signal_window).mean()
            inflation_rising = (tip_ief > tip_ief_sma).fillna(False)
        else:
            # Fallback: 50/50 split
            inflation_rising = pd.Series(True, index=prices.index)

        # Trend filter per asset: only include if above 200-day SMA
        all_assets = list(set(real_avail + nom_avail))
        sma200 = prices[all_assets].rolling(c.trend_window).mean()
        trend_up = prices[all_assets] > sma200

        # Rebalance mask
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True

        # Real tickers: higher weight when inflation rising
        real_w_inf = 0.70  # 70% to real when inflation rising
        real_w_def = 0.30  # 30% to real when deflation

        real_frac = pd.Series(
            np.where(inflation_rising, real_w_inf, real_w_def),
            index=prices.index,
        )
        nom_frac = 1.0 - real_frac

        # Apply rebalance hold
        real_frac = real_frac.where(rebal_mask).ffill().fillna(0.5)
        nom_frac = nom_frac.where(rebal_mask).ffill().fillna(0.5)

        # Distribute within each basket (equal weight, trend-filtered)
        for t in real_avail:
            in_trend = trend_up[t].fillna(False) if t in trend_up.columns else True
            n_real = max(len(real_avail), 1)
            w_t = real_frac * c.leverage / n_real
            weights[t] = np.where(in_trend, w_t, 0.0)

        for t in nom_avail:
            in_trend = trend_up[t].fillna(False) if t in trend_up.columns else True
            n_nom = max(len(nom_avail), 1)
            w_t = nom_frac * c.leverage / n_nom
            weights[t] = np.where(in_trend, w_t, 0.0)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# S1 — Formulaic Alpha Momentum (WorldQuant 101 Alphas: #12, #17, #52)
# =========================================================================

@dataclass
class FormulaicAlphaMomConfig:
    """Config for WorldQuant formulaic alpha momentum composite."""

    rebalance_freq: int = 21       # monthly rebalance
    long_pct: float = 0.2          # top 20% (quintile)
    short_pct: float = 0.2         # bottom 20%
    long_only: bool = True         # long-only by default
    leverage: float = 1.0
    adv_window: int = 20           # average daily volume window
    min_assets: int = 5            # minimum assets for cross-sectional signals


class FormulaicAlphaMomentum(Strategy):
    """Composite of WorldQuant formulaic momentum alphas (#12, #17, #52).

    Thesis: Volume-price interaction signals capture informed trading
    activity. When volume diverges from price (Alpha012), momentum
    accelerates with volume confirmation (Alpha017), or long-term
    momentum aligns with volume rank (Alpha052), alpha exists from
    information asymmetry between informed and noise traders.

    Signal: Cross-sectional z-score average of 3 alpha signals.
    Portfolio: Long top quintile (or long-short via config).
    """

    name = "S1-FormulaicAlphaMomentum"

    def __init__(self, config: FormulaicAlphaMomConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or FormulaicAlphaMomConfig()
        self._volume: pd.DataFrame | None = None
        self._low: pd.DataFrame | None = None

    def set_ohlcv(
        self,
        *,
        volume: pd.DataFrame | None = None,
        low: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """Inject OHLCV data for better signal quality."""
        if volume is not None:
            self._volume = volume
        if low is not None:
            self._low = low

    def _get_volume(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self._volume is not None:
            v = self._volume.reindex(
                index=prices.index, columns=prices.columns,
            ).ffill().fillna(1e6)
            return v
        # Proxy: |return| * price as dollar-volume proxy
        ret_abs = prices.pct_change().fillna(0.0).abs()
        return (ret_abs * prices + 1e4).clip(lower=1.0)

    def _get_low(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self._low is not None:
            return self._low.reindex(
                index=prices.index, columns=prices.columns,
            ).ffill().fillna(prices)
        # Proxy: close minus half the absolute daily range
        ret_abs = prices.pct_change().fillna(0.0).abs()
        return prices * (1 - ret_abs * 0.5).clip(lower=0.5)

    @staticmethod
    def _ts_rank(x: pd.DataFrame, d: int) -> pd.DataFrame:
        """Time-series percentile rank of current value in rolling window."""
        arr = x.values
        T, N = arr.shape
        out = np.full((T, N), np.nan)
        for i in range(d - 1, T):
            window = arr[i - d + 1: i + 1, :]
            current = arr[i, :]
            with np.errstate(invalid="ignore"):
                valid_count = np.sum(~np.isnan(window), axis=0)
                le_count = np.nansum(window <= current[np.newaxis, :], axis=0)
                mask = valid_count > 1
                out[i, mask] = le_count[mask] / valid_count[mask]
        return pd.DataFrame(out, index=x.index, columns=x.columns)

    @staticmethod
    def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score per row."""
        mu = df.mean(axis=1)
        std = df.std(axis=1).replace(0, np.nan)
        result = df.sub(mu, axis=0).div(std, axis=0)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _alpha012(
        self, close: pd.DataFrame, volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """sign(delta(volume, 1)) * (-1 * delta(close, 1))"""
        delta_vol = volume.diff(1).fillna(0.0)
        delta_close = close.diff(1).fillna(0.0)
        return np.sign(delta_vol) * (-1 * delta_close)

    def _alpha017(
        self, close: pd.DataFrame, volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """-rank(ts_rank(close,10)) * rank(accel) * rank(ts_rank(rel_vol,5))"""
        ts_rank_close = self._ts_rank(close, 10)
        rank_ts_close = ts_rank_close.rank(axis=1, pct=True)

        # Momentum acceleration: delta of delta close
        delta_close = close.diff(1).fillna(0.0)
        accel = delta_close.diff(1).fillna(0.0)
        rank_accel = accel.rank(axis=1, pct=True)

        # Relative volume rank
        adv = volume.rolling(self.cfg.adv_window, min_periods=1).mean()
        rel_vol = volume / adv.replace(0, np.nan)
        rel_vol = rel_vol.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        ts_rank_vol = self._ts_rank(rel_vol, 5)
        rank_ts_vol = ts_rank_vol.rank(axis=1, pct=True)

        return -1 * rank_ts_close * rank_accel * rank_ts_vol

    def _alpha052(
        self,
        close: pd.DataFrame,
        low: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """(-delta(ts_min(low,5),5)) * rank(long_mom) * ts_rank(volume,5)"""
        ts_min_low = low.rolling(5, min_periods=1).min()
        neg_delta_min = -1 * ts_min_low.diff(5).fillna(0.0)

        # Long-term momentum ratio
        returns = close.pct_change(1).fillna(0.0)
        ts_sum_240 = returns.rolling(240, min_periods=20).sum()
        ts_sum_20 = returns.rolling(20, min_periods=5).sum()
        mom_ratio = (ts_sum_240 - ts_sum_20) / 220
        mom_ratio = mom_ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rank_mom = mom_ratio.rank(axis=1, pct=True)

        ts_rank_vol = self._ts_rank(volume, 5)

        return neg_delta_min * rank_mom * ts_rank_vol

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        if prices.shape[1] < c.min_assets:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        close = prices
        volume = self._get_volume(prices)
        low = self._get_low(prices)

        # Compute each alpha
        a12 = self._alpha012(close, volume)
        a17 = self._alpha017(close, volume)
        a52 = self._alpha052(close, low, volume)

        # Cross-sectional z-score each alpha
        z12 = self._cs_zscore(a12)
        z17 = self._cs_zscore(a17)
        z52 = self._cs_zscore(a52)

        # Equal-weight composite
        composite = (z12 + z17 + z52) / 3.0

        # Cross-sectional rank
        cs_rank = composite.rank(axis=1, pct=True)

        # Portfolio construction
        long_thresh = 1.0 - c.long_pct
        short_thresh = c.short_pct

        is_long = pd.notna(cs_rank) & (cs_rank >= long_thresh)
        is_short = pd.notna(cs_rank) & (cs_rank <= short_thresh)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.long_only:
            w_long = is_long.astype(float)
            row_sum = w_long.sum(axis=1).clip(lower=1e-8)
            weights = w_long.div(row_sum, axis=0) * c.leverage
        else:
            w_long = is_long.astype(float)
            w_short = is_short.astype(float) * -1.0
            raw = w_long + w_short
            gross = raw.abs().sum(axis=1).clip(lower=1e-8)
            weights = raw.div(gross, axis=0) * c.leverage

        # Monthly rebalance hold
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        weights = weights.where(rebal_mask).ffill().fillna(0.0)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# K6 — Quality-Momentum Composite
# =========================================================================

@dataclass
class QualityMomentumCompositeConfig:
    """Config for quality-momentum composite factor strategy."""

    tickers: list[str] = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "IEF", "UUP",
        "XLE", "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
        "HYG", "LQD", "SLV", "SHY", "DBC", "DBA", "TIP", "AGG", "EMB",
        "FXI", "VGK", "EWJ", "INDA", "VNQ", "XBI",
    ])
    sharpe_window: int = 126
    drift_window: int = 63
    mom_fast: int = 63
    mom_slow: int = 252
    vol_window: int = 63
    sma_window: int = 200
    sharpe_wt: float = 0.35
    drift_wt: float = 0.20
    mom_wt: float = 0.30
    inv_vol_wt: float = 0.15
    long_n: int = 7
    short_n: int = 2
    gross_leverage: float = 1.25
    crash_leverage: float = 0.65
    max_weight: float = 0.25
    high_vol_threshold: float = 0.30
    crash_tlt_weight: float = 0.25


class QualityMomentumComposite(Strategy):
    """Cross-sectional quality + momentum composite factor.

    Thesis: assets with persistent positive drift, strong risk-adjusted
    momentum, and controlled volatility outperform in a diversified basket.
    Exposure is reduced in stressed market states to avoid crash beta.
    """

    name = "K6-QualityMomentumComposite"

    def __init__(
        self, config: QualityMomentumCompositeConfig | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config or QualityMomentumCompositeConfig()

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
        if len(avail) < c.long_n + c.short_n:
            return weights

        p = prices[avail]
        ret = p.pct_change().fillna(0.0)

        roll_mean = ret.rolling(c.sharpe_window, min_periods=20).mean()
        roll_std = ret.rolling(c.sharpe_window, min_periods=20).std().replace(0.0, np.nan)
        rolling_sharpe = (roll_mean / roll_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        drift = (ret > 0.0).astype(float).rolling(c.drift_window, min_periods=20).mean().fillna(0.0)
        mom_fast = p.pct_change(c.mom_fast).replace([np.inf, -np.inf], np.nan)
        mom_slow = p.pct_change(c.mom_slow).replace([np.inf, -np.inf], np.nan)
        mom = (0.6 * mom_fast) + (0.4 * mom_slow)

        rvol = ret.rolling(c.vol_window, min_periods=20).std() * np.sqrt(252)
        rvol = rvol.replace(0.0, np.nan).fillna(0.20)
        inv_vol = (1.0 / rvol).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        sharpe_rank = rolling_sharpe.rank(axis=1, pct=True).fillna(0.5)
        drift_rank = drift.rank(axis=1, pct=True).fillna(0.5)
        mom_rank = mom.rank(axis=1, pct=True).fillna(0.5)
        inv_vol_rank = inv_vol.rank(axis=1, pct=True).fillna(0.5)

        composite = (
            c.sharpe_wt * sharpe_rank
            + c.drift_wt * drift_rank
            + c.mom_wt * mom_rank
            + c.inv_vol_wt * inv_vol_rank
        )

        ema200 = p.rolling(c.sma_window, min_periods=50).mean()
        trend_up = (p > ema200)
        trend_dn = (p < ema200)

        long_score = composite.where(trend_up & (mom > 0.0), np.nan)
        short_score = composite.where(trend_dn & (mom < 0.0), np.nan)

        long_rank = long_score.rank(axis=1, ascending=False, method="average")
        short_rank = short_score.rank(axis=1, ascending=True, method="average")
        is_long = (long_rank <= c.long_n) & pd.notna(long_score)
        is_short = (short_rank <= c.short_n) & pd.notna(short_score)
        is_short = is_short & ~is_long

        long_raw = is_long.astype(float) * inv_vol * long_score.clip(lower=0.0).fillna(0.0)
        short_raw = is_short.astype(float) * inv_vol * (1.0 - short_score).clip(lower=0.0).fillna(0.0)

        long_sum = long_raw.sum(axis=1).replace(0.0, np.nan)
        short_sum = short_raw.sum(axis=1).replace(0.0, np.nan)
        long_w = long_raw.div(long_sum, axis=0).fillna(0.0)
        short_w = short_raw.div(short_sum, axis=0).fillna(0.0) * -1.0

        raw = long_w + short_w
        raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        spy_col = "SPY" if "SPY" in prices.columns else avail[0]
        spy_ret = prices[spy_col].pct_change().fillna(0.0)
        spy_vol = spy_ret.rolling(c.vol_window, min_periods=20).std() * np.sqrt(252)
        spy_vol = spy_vol.replace([np.inf, -np.inf], np.nan).fillna(c.high_vol_threshold)
        spy_sma = prices[spy_col].rolling(c.sma_window, min_periods=50).mean()
        crash = (prices[spy_col] < spy_sma) | (spy_vol > c.high_vol_threshold)

        lev_t = pd.Series(np.where(crash, c.crash_leverage, c.gross_leverage), index=prices.index, dtype=float)
        gross = raw.abs().sum(axis=1).replace(0.0, np.nan)
        scaled = raw.div(gross, axis=0).fillna(0.0).mul(lev_t, axis=0)
        scaled = scaled.clip(lower=-c.max_weight, upper=c.max_weight)
        scaled = scaled.where(~crash, scaled.clip(lower=0.0), axis=0)

        if "TLT" in scaled.columns:
            tlt_overlay = pd.Series(0.0, index=prices.index)
            tlt_overlay = tlt_overlay.where(~crash, c.crash_tlt_weight)
            scaled["TLT"] = scaled["TLT"] + tlt_overlay

        weights.loc[:, avail] = scaled
        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
