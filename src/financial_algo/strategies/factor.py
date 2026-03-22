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
    leverage: float = 1.5

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = [
                "XLP", "XLU", "XLV", "XLK", "XLF",
                "XLI", "XLB", "XLY", "XLE",
            ]


class LowVolFactor(Strategy):
    """Long the lowest-volatility sector ETFs (rebalanced monthly).

    Thesis: Low-volatility anomaly — low-vol stocks/sectors deliver
    higher risk-adjusted returns than high-vol ones. Documented by
    Baker, Bradley, Wurgler (2011). Persistent across markets and time.
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
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        if len(avail) < c.top_n:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Compute rolling vol for each sector
        vols = pd.DataFrame(index=prices.index)
        for t in avail:
            vols[t] = realized_vol(prices[t], c.vol_window)

        # Rank by vol (lowest vol = rank 1) and select top-N lowest
        vol_rank = vols.rank(axis=1, method='first')
        selected = vol_rank <= c.top_n

        # Rebalance at fixed intervals — hold selections between rebalances
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        selected_held = selected.where(rebal_mask).ffill().fillna(False)

        # Assign equal weight to selected sectors
        per_weight = c.leverage / c.top_n
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for t in avail:
            weights[t] = selected_held[t].astype(float) * per_weight

        return weights.fillna(0.0)


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
