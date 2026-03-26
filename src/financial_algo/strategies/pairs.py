"""Category E: Tactical Pairs & Statistical Arbitrage.

Market-neutral crisis alpha from mean-reversion pairs that decorrelate
during crises.  Ported from Financial-Algorithms v10-v18 ``pair_weights``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financial_algo.signals import pair_zscore_signal
from financial_algo.strategies.base import Strategy


# =========================================================================
# Pair definition
# =========================================================================

@dataclass
class PairSpec:
    """One pair to trade."""

    long_ticker: str
    short_ticker: str
    zscore_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    leverage: float = 2.0


# Default crisis-decorrelating pairs — selected for genuine correlation
DEFAULT_PAIRS: list[PairSpec] = [
    PairSpec("GLD", "UUP", zscore_window=40, entry_z=2.0, exit_z=0.3, leverage=1.5),
    PairSpec("HYG", "LQD", zscore_window=40, entry_z=2.0, exit_z=0.3, leverage=1.5),
    PairSpec("XLP", "XLU", zscore_window=40, entry_z=2.0, exit_z=0.3, leverage=1.5),
    PairSpec("IWM", "SPY", zscore_window=40, entry_z=2.0, exit_z=0.3, leverage=1.5),
    PairSpec("EEM", "EFA", zscore_window=40, entry_z=2.0, exit_z=0.3, leverage=1.5),
    PairSpec("XLY", "XLP", zscore_window=40, entry_z=2.0, exit_z=0.3, leverage=1.5),
]


# =========================================================================
# Single-pair strategy
# =========================================================================

class PairTrade(Strategy):
    """Z-score mean-reversion on a single pair."""

    name = "PairTrade"

    def __init__(self, pair: PairSpec) -> None:
        self.pair = pair
        self.name = f"PairTrade({pair.long_ticker}/{pair.short_ticker})"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        p = self.pair
        sig = pair_zscore_signal(
            prices[p.long_ticker],
            prices[p.short_ticker],
            window=p.zscore_window,
            entry_z=p.entry_z,
            exit_z=p.exit_z,
        )

        w = pd.DataFrame(0.0, index=prices.index, columns=[p.long_ticker, p.short_ticker])
        w[p.long_ticker] = sig * p.leverage
        w[p.short_ticker] = -sig * p.leverage

        return w


# =========================================================================
# Multi-pair portfolio
# =========================================================================

class MultiPairPortfolio(Strategy):
    """Long-only relative value across pairs with market trend filter.

    Reworked from long/short to long-only to eliminate short-side drag.
    Goes long the relatively cheap leg when spread is extended.
    """

    name = "MultiPairPortfolio"

    def __init__(self, pairs: list[PairSpec] | None = None) -> None:
        self.pairs = pairs if pairs is not None else DEFAULT_PAIRS

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Market trend + volatility throttle to reduce crisis drawdowns.
        if "SPY" in prices.columns:
            spy = prices["SPY"]
            spy_sma = spy.rolling(200, min_periods=50).mean()
            market_up = (spy > spy_sma).fillna(False)

            spy_vol = spy.pct_change().fillna(0.0).rolling(20, min_periods=10).std()
            spy_vol = (spy_vol * np.sqrt(252)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            market_calm = spy_vol <= 0.28

            vol_scale = (0.22 / spy_vol.replace(0.0, np.nan)).clip(lower=0.20, upper=1.0).fillna(0.20)
            risk_gate = (market_up & market_calm).astype(float) * vol_scale
        else:
            risk_gate = pd.Series(1.0, index=prices.index)

        per_pair = 1.0 / max(len(self.pairs), 1)

        for pair in self.pairs:
            lt, st = pair.long_ticker, pair.short_ticker
            if lt not in prices.columns or st not in prices.columns:
                continue

            sig = pair_zscore_signal(
                prices[lt], prices[st],
                window=pair.zscore_window,
                entry_z=pair.entry_z,
                exit_z=pair.exit_z,
            )

            ratio = prices[lt] / prices[st].replace(0.0, np.nan)
            ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            pair_vol = ratio.pct_change().fillna(0.0).rolling(20, min_periods=10).std()
            pair_vol = (pair_vol * np.sqrt(252)).replace([np.inf, -np.inf], np.nan)
            pair_scale = (0.18 / pair_vol.replace(0.0, np.nan)).clip(lower=0.4, upper=1.2).fillna(0.4)

            # Long-only: go long the relatively cheap leg
            w = pair.leverage * per_pair * pair_scale * risk_gate
            long_a = sig == 1
            long_b = sig == -1

            weights[lt] = weights[lt] + np.where(long_a, w, 0.0)
            weights[st] = weights[st] + np.where(long_b, w, 0.0)

        gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
        gross_cap = (0.45 / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(gross_cap, axis=0)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
