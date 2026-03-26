"""Ensemble meta-strategy — combines multiple strategies with
correlation-aware weighting, dynamic hedging, and drawdown controls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from financial_algo.indicators import drawdown as _drawdown, hurst_exponent as _hurst_exp
from financial_algo.strategies.base import Strategy

logger = logging.getLogger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble combiner."""

    # --- Weighting method selection ---
    # "inverse_vol" — original inverse-volatility with optional Sharpe prior
    # "hrp"         — Hierarchical Risk Parity (skfolio)
    # "risk_budget" — Risk Budgeting / equal risk contribution (skfolio, requires cvxpy)
    weighting_method: Literal["inverse_vol", "hrp", "risk_budget"] = "inverse_vol"

    # HRP / Risk Budgeting risk measure ("variance", "cvar", "standard_deviation")
    skfolio_risk_measure: str = "variance"

    # How often to refit skfolio models (every N trading days).
    # Avoids refitting daily for performance. Set to 1 for daily refit.
    skfolio_refit_every: int = 21  # ~monthly

    # Correlation-aware weighting
    correlation_lookback: int = 60
    use_inverse_vol: bool = True    # weight by inverse vol (only used when weighting_method="inverse_vol")

    # Sharpe-based prior weights (tilts allocation towards higher-conviction strategies)
    # If provided, must match length of strategies list. Values are relative (auto-normalized).
    # Applied as a tilt on top of any weighting method.
    prior_weights: list[float] | None = None

    # Gradual drawdown circuit breaker (applied to ensemble equity)
    # Scale exposure linearly from 1.0 at dd_scale_start to 0.0 at dd_scale_end
    dd_scale_start: float = -0.15   # start scaling down at -15% DD
    dd_scale_end: float = -0.35     # fully flat at -35% DD
    dd_scale_power: float = 1.5     # >1.0 reduces shallow-DD over-triggering

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

    # --- SkewKurt overlay ---
    # Scale down exposure when SPY exhibits fat left tail (crash early warning).
    skewkurt_overlay: bool = False
    skewkurt_spy: str = "SPY"
    skewkurt_window: int = 60           # rolling window for skew/kurt
    skewkurt_scale_low: float = 0.5     # scale when skew < -0.5 AND kurt > 4
    skewkurt_scale_high: float = 0.25   # scale when skew < -1.0

    # --- Hurst regime router ---
    # Tilt allocation between momentum (I-series) and mean-reversion (J-series)
    # based on the Hurst exponent of SPY.  Disabled by default (expensive).
    hurst_router: bool = False
    hurst_spy: str = "SPY"
    hurst_window: int = 252             # rolling window for Hurst estimation
    hurst_mom_threshold: float = 0.55   # H > this → trending → boost momentum
    hurst_mr_threshold: float = 0.45    # H < this → mean-reverting → boost MR
    hurst_tilt_factor: float = 1.5      # multiply winning-regime strategies by this

    # --- Graduated RSI exit overlay ---
    # Scale down when the ensemble equity curve becomes overbought.
    # RSI is computed on the simulated portfolio daily returns.
    # rsi > rsi_exit_extreme  → scale by rsi_exit_scale_extreme  (0.50)
    # rsi > rsi_exit_moderate → scale by rsi_exit_scale_moderate (0.75)
    rsi_exit_overlay: bool = False
    rsi_exit_period: int = 14
    rsi_exit_moderate: float = 80.0
    rsi_exit_extreme: float = 90.0
    rsi_exit_scale_moderate: float = 0.75
    rsi_exit_scale_extreme: float = 0.50

    # --- VIX-Adaptive Prior Weights (differential regime tilting) ---
    # Dynamically tilt strategy allocation by VIX environment:
    # Defensive strategies (B,C,D,F,G,L,O,R) get boosted in high-VIX;
    # Risk-on strategies get boosted in low-VIX. After scaling, weights
    # are renormalized so this is a RELATIVE tilt, not a leverage change.
    vix_prior_scaling: bool = False
    vix_prior_k: float = 0.05          # exponential tilt sensitivity
    vix_prior_base: float = 20.0       # neutral VIX level
    vix_prior_lookback: int = 20       # SPY realized-vol lookback (days)

    # --- Drift Regime Filter (arXiv:2511.12490 inspired) ---
    # Scale down ensemble exposure when market lacks positive drift.
    # When <threshold fraction of trailing days are positive, reduce sizing.
    drift_filter_enabled: bool = False
    drift_lookback: int = 63           # trailing window for drift detection
    drift_threshold: float = 0.58      # min positive-day fraction for full exposure
    drift_scale_weak: float = 0.50     # min scale factor when drift is absent


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
        if c.weighting_method == "inverse_vol":
            alloc = self._alloc_inverse_vol(strat_returns, n)
        elif c.weighting_method in ("hrp", "risk_budget"):
            alloc = self._alloc_skfolio(strat_returns, n)
        else:
            alloc = pd.DataFrame(1.0 / n, index=prices.index, columns=range(n))

        # --- Apply Sharpe-based prior weights (tilt towards winners) ------
        if c.prior_weights is not None and len(c.prior_weights) == n:
            prior = np.array(c.prior_weights, dtype=float)

            if c.vix_prior_scaling and "SPY" in prices.columns:
                alloc = self._apply_vix_prior_scaling(
                    alloc, prior, prices, n
                )
            else:
                prior = prior / prior.sum()
                for col in range(n):
                    alloc[col] = alloc[col] * prior[col]

            alloc = alloc.div(alloc.sum(axis=1), axis=0).fillna(1.0 / n)

        # --- Hurst regime router: tilt alloc between momentum and MR -----
        if c.hurst_router and c.hurst_spy in prices.columns:
            alloc = self._apply_hurst_router_alloc(alloc, prices)

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

        # --- SkewKurt tail-risk overlay ----------------------------------
        if c.skewkurt_overlay and c.skewkurt_spy in prices.columns:
            combined = self._apply_skewkurt_overlay(combined, prices)

        # --- Drift regime filter overlay --------------------------------
        if c.drift_filter_enabled and "SPY" in prices.columns:
            combined = self._apply_drift_filter(combined, prices)

        # --- Drawdown circuit-breaker ------------------------------------
        combined = self._apply_circuit_breaker(combined, asset_returns)

        # --- Graduated RSI exit overlay ----------------------------------
        if c.rsi_exit_overlay:
            combined = self._apply_rsi_exit_overlay(combined, asset_returns)

        return combined

    # ------------------------------------------------------------------
    # Allocation methods
    # ------------------------------------------------------------------

    def _alloc_inverse_vol(
        self,
        strat_returns: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Original inverse-volatility weighting."""
        c = self.cfg
        if c.use_inverse_vol and n > 1:
            roll_vol = strat_returns.rolling(c.correlation_lookback).std()
            inv_vol = 1.0 / roll_vol.replace(0, np.nan)
            alloc = inv_vol.div(inv_vol.sum(axis=1), axis=0).fillna(1.0 / n)
        else:
            alloc = pd.DataFrame(1.0 / n, index=strat_returns.index, columns=range(n))
        return alloc

    # ------------------------------------------------------------------
    # VIX-adaptive prior weights
    # ------------------------------------------------------------------

    _DEFENSIVE_PREFIXES = frozenset("BCDFGLOR")

    def _apply_vix_prior_scaling(
        self,
        alloc: pd.DataFrame,
        prior: np.ndarray,
        prices: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Differentially scale prior weights by VIX environment.

        Defensive strategies get boosted in high-VIX; risk-on strategies
        get boosted in low-VIX.  After scaling, weights are renormalized
        so this is a relative tilt, not a leverage change.
        """
        c = self.cfg
        spy_vol = (
            prices["SPY"]
            .pct_change()
            .fillna(0.0)
            .rolling(c.vix_prior_lookback)
            .std()
            .mul(np.sqrt(252))
            .fillna(c.vix_prior_base / 100.0)
        )
        vix_proxy = spy_vol * 100.0  # scale to VIX-like units

        # Classify strategies: +1 = defensive, -1 = risk-on
        vix_affinity = np.array([
            1.0 if s.name[:1] in self._DEFENSIVE_PREFIXES else -1.0
            for s in self.strategies
        ])

        vix_ratio = np.asarray(vix_proxy / c.vix_prior_base - 1.0)  # (T,)
        vix_factor = np.exp(
            np.outer(vix_ratio, vix_affinity * c.vix_prior_k)
        )  # (T, n)
        dynamic_prior = prior[np.newaxis, :] * vix_factor  # (T, n)

        row_sum = dynamic_prior.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum > 0, row_sum, 1.0)
        dynamic_prior = dynamic_prior / row_sum

        for col in range(n):
            alloc[col] = alloc[col] * dynamic_prior[:, col]

        return alloc

    # ------------------------------------------------------------------
    # Drift regime filter
    # ------------------------------------------------------------------

    def _apply_drift_filter(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Scale exposure based on market positive-drift strength.

        When the fraction of positive SPY return days over a trailing
        window drops below the threshold, gradually reduce position sizes.
        Inspired by arXiv:2511.12490 drift-regime research.
        """
        c = self.cfg
        spy_ret = prices["SPY"].pct_change().fillna(0.0)
        pos_frac = (
            (spy_ret > 0)
            .astype(float)
            .rolling(c.drift_lookback)
            .mean()
            .fillna(0.5)
        )

        ramp_width = 0.15
        lower = c.drift_threshold - ramp_width
        scale = ((pos_frac - lower) / ramp_width).clip(c.drift_scale_weak, 1.0)

        return weights.multiply(scale, axis=0)

    def _alloc_skfolio(
        self,
        strat_returns: pd.DataFrame,
        n: int,
    ) -> pd.DataFrame:
        """Allocation via skfolio HRP or RiskBudgeting.

        Fits the model on a rolling window (correlation_lookback) and
        refits every ``skfolio_refit_every`` trading days to avoid
        excessive computation.  Between refits the allocation is held
        constant (forward-filled).
        """
        try:
            from skfolio import RiskMeasure
            from skfolio.optimization import HierarchicalRiskParity, RiskBudgeting
        except ImportError as exc:
            logger.warning(
                "skfolio not installed; falling back to inverse_vol. "
                "Install with: uv pip install skfolio  (%s)",
                exc,
            )
            return self._alloc_inverse_vol(strat_returns, n)

        c = self.cfg
        risk_map = {
            "variance": RiskMeasure.VARIANCE,
            "cvar": RiskMeasure.CVAR,
            "standard_deviation": RiskMeasure.STANDARD_DEVIATION,
        }
        risk_measure = risk_map.get(c.skfolio_risk_measure, RiskMeasure.VARIANCE)

        if c.weighting_method == "hrp":
            model = HierarchicalRiskParity(
                risk_measure=risk_measure,
                max_weights=c.max_single_weight,
            )
        else:  # risk_budget
            model = RiskBudgeting(
                risk_measure=risk_measure,
                max_weights=c.max_single_weight,
            )

        lookback = c.correlation_lookback
        refit_every = max(1, c.skfolio_refit_every)

        alloc = pd.DataFrame(1.0 / n, index=strat_returns.index, columns=range(n))
        last_weights = np.full(n, 1.0 / n)
        days_since_fit = refit_every  # Force fit on first eligible day

        for t in range(lookback, len(strat_returns)):
            days_since_fit += 1
            if days_since_fit >= refit_every:
                window = strat_returns.iloc[t - lookback : t].fillna(0.0)
                # Skip fit if window has zero variance (e.g. all zeros)
                if window.std().replace(0, np.nan).notna().sum() >= 2:
                    try:
                        model.fit(window)
                        last_weights = model.weights_
                        days_since_fit = 0
                    except Exception:
                        pass  # Keep previous weights on solver failure
            alloc.iloc[t] = last_weights

        return alloc

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

        # Smooth scale: 1.0 at dd_scale_start, 0.0 at dd_scale_end.
        # A power > 1.0 makes shallow drawdowns less punitive while
        # preserving full de-risking in deep drawdowns.
        dd_range = c.dd_scale_end - c.dd_scale_start  # negative
        if dd_range >= 0:
            return weights

        progress = ((dd - c.dd_scale_start) / dd_range).clip(0.0, 1.0).fillna(0.0)
        power = c.dd_scale_power if c.dd_scale_power > 0 else 1.0
        scale = 1.0 - progress.pow(power)
        scale = scale.clip(0.0, 1.0)

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

    def _apply_skewkurt_overlay(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Scale down risk when SPY exhibits a fat left tail (crash early warning).

        Regime logic:
        - skew < -1.0                        → scale by skewkurt_scale_high (0.25)
        - skew < -0.5 AND kurt > 4           → scale by skewkurt_scale_low  (0.5)
        - otherwise                          → no scaling (1.0)
        """
        c = self.cfg
        spy_ret = prices[c.skewkurt_spy].pct_change()
        skew = spy_ret.rolling(c.skewkurt_window).skew().fillna(0.0)
        kurt = spy_ret.rolling(c.skewkurt_window).kurt().fillna(0.0)

        scale = pd.Series(1.0, index=prices.index)
        stressed = (skew < -0.5) & (kurt > 4.0)
        extreme = skew < -1.0
        scale = scale.where(~stressed, c.skewkurt_scale_low)
        scale = scale.where(~extreme, c.skewkurt_scale_high)

        return weights.multiply(scale, axis=0)

    def _apply_hurst_router_alloc(
        self,
        alloc: pd.DataFrame,
        prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Tilt allocation between momentum (I-*) and mean-reversion (J-*) strategies.

        Uses the rolling Hurst exponent of SPY as a market-regime indicator:
        - H > hurst_mom_threshold  → trending market  → boost I-series strategies
        - H < hurst_mr_threshold   → mean-reverting   → boost J-series strategies
        - in between               → no change
        """
        c = self.cfg
        hurst_vals = _hurst_exp(
            prices[c.hurst_spy],
            window=c.hurst_window,
        ).fillna(0.5)

        # Classify each sub-strategy by name prefix
        mom_idx = [i for i, s in enumerate(self.strategies) if s.name.startswith("I")]
        mr_idx = [i for i, s in enumerate(self.strategies) if s.name.startswith("J")]

        if not mom_idx and not mr_idx:
            return alloc

        high_hurst = (hurst_vals > c.hurst_mom_threshold).astype(float)
        low_hurst = (hurst_vals < c.hurst_mr_threshold).astype(float)

        tilted = alloc.copy()
        for idx in mom_idx:
            tilted[idx] = alloc[idx] * (1.0 + (c.hurst_tilt_factor - 1.0) * high_hurst)
        for idx in mr_idx:
            tilted[idx] = alloc[idx] * (1.0 + (c.hurst_tilt_factor - 1.0) * low_hurst)

        # Renormalize so weights sum to 1 across strategies
        row_sum = tilted.sum(axis=1).clip(lower=1e-8)
        return tilted.div(row_sum, axis=0)

    def _apply_rsi_exit_overlay(
        self,
        weights: pd.DataFrame,
        asset_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """Scale down when ensemble equity RSI is overbought.

        Computes RSI(period) on the simulated portfolio equity curve
        built from the current weights.  This is not look-ahead biased
        because portfolio returns use ``weights.shift(1)``.

        Scaling:
        - RSI > rsi_exit_extreme  → scale by rsi_exit_scale_extreme  (0.50)
        - RSI > rsi_exit_moderate → scale by rsi_exit_scale_moderate (0.75)
        - otherwise               → no scaling
        """
        c = self.cfg
        aligned = weights.reindex(columns=asset_returns.columns, fill_value=0.0)
        port_ret = (aligned.shift(1).fillna(0) * asset_returns).sum(axis=1)
        equity = (1 + port_ret).cumprod()

        delta = equity.diff()
        gain = delta.clip(lower=0).ewm(span=c.rsi_exit_period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(span=c.rsi_exit_period, adjust=False).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_val = (100 - 100 / (1 + rs)).fillna(50.0)

        scale = pd.Series(1.0, index=weights.index)
        moderate = rsi_val > c.rsi_exit_moderate
        extreme = rsi_val > c.rsi_exit_extreme
        scale = scale.where(~moderate, c.rsi_exit_scale_moderate)
        scale = scale.where(~extreme, c.rsi_exit_scale_extreme)

        return weights.multiply(scale, axis=0)
