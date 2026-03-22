"""Category P: Signal Combination strategies.

Simple feature-based composite scoring using numpy/pandas.
No sklearn required -- pure numpy/pandas implementation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import realized_vol, ema, zscore
from financial_algo.strategies.base import Strategy


# =========================================================================
# P1 -- Feature Combo Signal (Simple 3-feature composite, long-only)
# =========================================================================

@dataclass
class FeatureComboConfig:
    """Simple 3-feature composite score for long-only allocation."""

    tickers: tuple = ("SPY", "QQQ", "IWM", "GLD", "TLT", "XLE", "EFA", "EEM")

    # Feature parameters
    mom_lookback: int = 252       # 12-month return
    vol_window: int = 20          # 20-day realized vol
    sma_window: int = 200         # price vs 200-day SMA

    # Weights for each feature (simple fixed blend)
    mom_weight: float = 0.40      # momentum score weight
    vol_weight: float = 0.30      # low-vol score weight
    trend_weight: float = 0.30    # above-SMA score weight

    # Allocation
    top_n: int = 4                # long top-N scoring assets
    leverage: float = 1.5
    rebalance_freq: int = 21      # monthly rebalance


class FeatureComboSignal(Strategy):
    """Simple 3-feature composite score: long top-scoring assets.

    Thesis: Combine 12-month return (momentum), 20-day vol (low-vol),
    and price vs 200-day SMA (trend) into a single composite score.
    Go long the top-N scoring assets. No shorts, no adaptive weights,
    no ML -- just a clean, simple multi-factor score.
    """

    name = "P1-FeatureComboSignal"

    def __init__(self, config: FeatureComboConfig | None = None) -> None:
        self.cfg = config or FeatureComboConfig()

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

        # Feature 1: 12-month momentum (rank: higher return = higher score)
        mom_ret = p.pct_change(c.mom_lookback).fillna(0.0)
        mom_rank = mom_ret.rank(axis=1, pct=True).fillna(0.5)

        # Feature 2: Low vol (rank: lower vol = higher score)
        vol_df = pd.DataFrame(index=prices.index, columns=avail, dtype=float)
        for t in avail:
            vol_df[t] = realized_vol(p[t], c.vol_window)
        vol_df = vol_df.fillna(0.0)
        # Invert: low vol -> high rank
        vol_rank = (1.0 - vol_df.rank(axis=1, pct=True)).fillna(0.5)

        # Feature 3: Above 200-day SMA (binary: 1 if above, 0 if below)
        sma = p.rolling(c.sma_window, min_periods=100).mean()
        above_sma = (p >= sma).astype(float).fillna(0.0)

        # Composite score (weighted sum of ranks/features)
        composite = (
            c.mom_weight * mom_rank
            + c.vol_weight * vol_rank
            + c.trend_weight * above_sma
        )

        # Select top-N assets on rebalance days, hold between rebalances
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Rank within available tickers
        top_n = min(c.top_n, len(avail))
        per_asset_weight = c.leverage / top_n

        # Create signal: top-N by composite score
        ranks = composite.rank(axis=1, ascending=False)
        is_top = ranks <= top_n

        # Apply rebalance frequency: hold positions between rebalance dates
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        # Forward-fill the rebalance signal
        held_signal = is_top.where(rebal_mask, np.nan).ffill().fillna(False).astype(bool)

        for t in avail:
            weights.loc[held_signal[t], t] = per_asset_weight

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
