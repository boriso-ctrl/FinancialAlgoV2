"""Ensemble meta-strategy — combines multiple strategies with
correlation-aware weighting, dynamic hedging, and drawdown controls.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financial_algo.indicators import drawdown as _drawdown
from financial_algo.strategies.base import Strategy


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble combiner."""

    # Correlation-aware weighting
    correlation_lookback: int = 60
    use_inverse_vol: bool = True    # weight by inverse vol

    # Sharpe-based prior weights (tilts allocation towards higher-conviction strategies)
    # If provided, must match length of strategies list. Values are relative (auto-normalized).
    prior_weights: list[float] | None = None

    # Gradual drawdown circuit breaker (applied to ensemble equity)
    # Scale exposure linearly from 1.0 at dd_scale_start to 0.0 at dd_scale_end
    dd_scale_start: float = -0.15   # start scaling down at -15% DD
    dd_scale_end: float = -0.35     # fully flat at -35% DD

    # Max gross leverage for the ensemble
    max_gross_leverage: float = 5.0

    # Max weight for any single sub-strategy (fraction of total alloc)
    max_single_weight: float = 0.25

    # --- 1a: Correlation-based safe haven hedging ---
    # When ensemble correlation with SPY exceeds threshold, shift
    # allocation into safe haven tickers to reduce systematic risk.
    correlation_hedge_enabled: bool = False
    correlation_hedge_threshold: float = 0.65   # start hedging above this
    correlation_hedge_max: float = 0.25         # max fraction shifted to havens
    safe_haven_tickers: tuple = ("GLD", "TLT", "UUP")

    # --- 1d: Vol-regime dynamic leverage scaling ---
    # Reduce gross leverage when SPY realized vol (VIX proxy) is elevated.
    vol_regime_scaling: bool = False
    vol_elevated_threshold: float = 0.20   # annualized vol threshold (elevated)
    vol_crisis_threshold: float = 0.30     # annualized vol threshold (crisis)
    leverage_elevated: float = 2.0         # max leverage when vol elevated
    leverage_crisis: float = 1.5           # max leverage when vol crisis


class EnsembleStrategy(Strategy):
    """Combine multiple :class:`Strategy` instances.

    Each sub-strategy generates weights independently.  The ensemble
    combines them using inverse-volatility weighting (optional) and
    applies a drawdown circuit-breaker on the combined equity.
    """

    name = "Ensemble"

    def __init__(
        self,
        strategies: list[Strategy],
        config: EnsembleConfig | None = None,
    ) -> None:
        self.strategies = strategies
        self.cfg = config or EnsembleConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        sub_weights: list[pd.DataFrame] = []
        for strat in self.strategies:
            w = strat.generate_weights(prices, regime)
            sub_weights.append(w)

        if not sub_weights:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # --- Compute per-strategy returns for weighting ------------------
        asset_returns = prices.pct_change().fillna(0.0)
        strat_returns = pd.DataFrame(index=prices.index)
        for i, w in enumerate(sub_weights):
            aligned = w.reindex(columns=prices.columns, fill_value=0.0)
            sr = (aligned.shift(1).fillna(0) * asset_returns).sum(axis=1)
            strat_returns[i] = sr

        # --- Inverse-vol weighting ---------------------------------------
        n = len(sub_weights)
        if c.use_inverse_vol and n > 1:
            roll_vol = strat_returns.rolling(c.correlation_lookback).std()
            inv_vol = 1.0 / roll_vol.replace(0, np.nan)
            alloc = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1.0 / n)
        else:
            alloc = pd.DataFrame(1.0 / n, index=prices.index, columns=range(n))

        # --- Apply Sharpe-based prior weights (tilt towards winners) ------
        if c.prior_weights is not None and len(c.prior_weights) == n:
            prior = np.array(c.prior_weights, dtype=float)
            prior = prior / prior.sum()  # normalize to sum=1
            for col in range(n):
                alloc[col] = alloc[col] * prior[col]
            alloc = alloc.div(alloc.sum(axis=1), axis=0).fillna(1.0 / n)

        # Cap single strategy weight and renormalize
        alloc = alloc.clip(upper=c.max_single_weight)
        alloc = alloc.div(alloc.sum(axis=1), axis=0).fillna(1.0 / n)

        # --- Weighted combination ----------------------------------------
        all_tickers = sorted(set().union(*(w.columns for w in sub_weights)))
        combined = pd.DataFrame(0.0, index=prices.index, columns=all_tickers)

        for i, w in enumerate(sub_weights):
            aligned = w.reindex(columns=all_tickers, fill_value=0.0)
            combined += aligned.multiply(alloc[i], axis=0)

        # --- 1a: Correlation-based safe haven hedging ---------------------
        if c.correlation_hedge_enabled and "SPY" in asset_returns.columns:
            combined = self._apply_correlation_hedge(
                combined, asset_returns, strat_returns
            )

        # --- 1d: Vol-regime dynamic leverage cap -------------------------
        if c.vol_regime_scaling and "SPY" in prices.columns:
            spy_vol = (
                prices["SPY"]
                .pct_change()
                .fillna(0)
                .rolling(20)
                .std()
                .mul(np.sqrt(252))
                .fillna(0)
            )
            leverage_limit = pd.Series(
                c.max_gross_leverage, index=prices.index
            )
            leverage_limit = leverage_limit.where(
                spy_vol <= c.vol_elevated_threshold, c.leverage_elevated
            )
            leverage_limit = leverage_limit.where(
                spy_vol <= c.vol_crisis_threshold, c.leverage_crisis
            )
        else:
            leverage_limit = c.max_gross_leverage

        # --- Cap gross leverage ------------------------------------------
        gross = combined.abs().sum(axis=1)
        if isinstance(leverage_limit, pd.Series):
            scale = (leverage_limit / gross).clip(upper=1.0)
        else:
            scale = (leverage_limit / gross).clip(upper=1.0)
        combined = combined.multiply(scale, axis=0)

        # --- Drawdown circuit-breaker ------------------------------------
        combined = self._apply_circuit_breaker(combined, asset_returns)

        return combined

    def _apply_circuit_breaker(
        self,
        weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Gradually scale down weights as ensemble drawdown deepens."""
        c = self.cfg

        # Compute equity of the combined weights
        aligned = weights.reindex(columns=asset_returns.columns, fill_value=0.0)
        port_ret = (aligned.shift(1).fillna(0) * asset_returns).sum(axis=1)
        equity = (1 + port_ret).cumprod()
        dd = _drawdown(equity)

        # Linear scale: 1.0 at dd_scale_start, 0.0 at dd_scale_end
        dd_range = c.dd_scale_end - c.dd_scale_start  # negative
        scale = ((dd - c.dd_scale_start) / dd_range).clip(0.0, 1.0)
        # Invert: 1.0 when dd >= dd_scale_start (shallow), 0.0 when dd <= dd_scale_end (deep)
        scale = 1.0 - scale

        weights = weights.multiply(scale, axis=0)
        return weights

    def _apply_correlation_hedge(
        self,
        weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
        strat_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Shift allocation to safe havens when ensemble-SPY correlation spikes.

        Uses rolling correlation of the equal-weighted sub-strategy
        portfolio with SPY.  When this exceeds the threshold, linearly
        shifts up to ``correlation_hedge_max`` of exposure into GLD/TLT/UUP.
        """
        c = self.cfg

        spy_ret = asset_returns["SPY"]
        # Equal-weight portfolio of all sub-strategies as correlation input
        eq_port = strat_returns.mean(axis=1)
        corr = eq_port.rolling(c.correlation_lookback).corr(spy_ret).fillna(0)

        # Linear ramp: 0 at threshold, correlation_hedge_max at threshold+0.20
        ramp_width = 0.20
        hedge_frac = (
            (corr - c.correlation_hedge_threshold) / ramp_width
        ).clip(0, 1) * c.correlation_hedge_max

        # Scale down existing weights proportionally, add safe haven allocation
        weights = weights.multiply(1 - hedge_frac, axis=0)
        per_haven = hedge_frac / max(len(c.safe_haven_tickers), 1)
        for ticker in c.safe_haven_tickers:
            if ticker not in weights.columns:
                weights[ticker] = 0.0
            weights[ticker] = weights[ticker] + per_haven

        return weights
