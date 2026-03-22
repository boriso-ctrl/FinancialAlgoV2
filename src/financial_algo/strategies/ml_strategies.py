"""Category P: ML-Enhanced Strategies.

P4 — Walk-forward optimized multi-signal strategy (sklearn).
P5 — Cross-sectional XGBoost asset ranking (xgboost).

Both strategies use strict walk-forward training: only past data is used
for parameter optimization / model fitting.  No look-ahead bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from financial_algo.strategies.base import Strategy
from financial_algo.strategies._shared_features import (
    momentum_df as _momentum,
    realized_vol_df as _realized_vol_df,
    rsi_df as _rsi_df,
    zscore_df as _zscore_df,
    above_sma as _above_sma,
)


# =====================================================================
# P4 — AdaptiveThreshold (Walk-Forward Parameter Optimization)
# =====================================================================

# Grid kept intentionally small (72 combos) to avoid overfitting
_P4_PARAM_GRID = list(product(
    [0.2, 0.4, 0.6],        # mom_weight
    [0.1, 0.3],             # vol_weight
    [0.1, 0.3],             # trend_weight
    [3, 4, 5, 6],           # top_n
))

# Default params (same as P1) used during warm-up period
_P4_DEFAULTS = {
    "mom_weight": 0.40,
    "vol_weight": 0.30,
    "trend_weight": 0.30,
    "top_n": 4,
}

_P4_TICKERS = (
    "SPY", "QQQ", "IWM", "GLD", "TLT", "XLE",
    "EFA", "EEM", "SLV", "UUP", "DBC", "VNQ",
)


@dataclass
class AdaptiveThresholdConfig:
    """Config for P4-AdaptiveThreshold."""

    tickers: tuple[str, ...] = _P4_TICKERS
    warmup_days: int = 504       # 2 years minimum before optimising
    reopt_freq: int = 63         # re-optimise every quarter
    opt_window: int = 504        # 2 years of training data
    fwd_horizon: int = 21        # target: next-month return
    cv_splits: int = 3           # TimeSeriesSplit folds
    leverage: float = 1.5
    max_weight: float = 0.50
    rebalance_freq: int = 21     # hold positions for ~1 month

    # Signal parameters
    mom12_lookback: int = 252    # 12-month momentum
    mom3_lookback: int = 63      # 3-month momentum
    rsi_period: int = 14
    zscore_window: int = 60
    sma_window: int = 200


class AdaptiveThreshold(Strategy):
    """Walk-forward optimized multi-signal composite strategy.

    Every 63 days, uses the previous 504 days to cross-validate
    signal weights (momentum, low-vol, trend) and top-N count.
    Between re-optimizations, holds the best parameter set.

    Thesis: Static parameters degrade across regimes.  Periodic
    walk-forward re-optimization adapts to the current environment
    while the short param grid and TimeSeriesSplit CV guard against
    overfitting.
    """

    name = "P4-AdaptiveThreshold"

    def __init__(self, config: AdaptiveThresholdConfig | None = None) -> None:
        self.cfg = config or AdaptiveThresholdConfig()

    # ------------------------------------------------------------------
    # Core weight generation
    # ------------------------------------------------------------------

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
        n_days = len(p)

        # Pre-compute all signals once (vectorized)
        mom12 = _momentum(p, c.mom12_lookback)
        mom3 = _momentum(p, c.mom3_lookback)
        rsi_vals = _rsi_df(p, c.rsi_period)
        zsc = _zscore_df(p, c.zscore_window)
        above = _above_sma(p, c.sma_window)

        # Forward returns for CV scoring (only used in the optimiser
        # on PAST windows — never at prediction time)
        fwd_ret = p.pct_change(c.fwd_horizon).shift(-c.fwd_horizon).fillna(0.0)

        # Walk-forward: determine which params to use on each day
        # params_timeline[i] = dict of params active on day i
        params_timeline: list[dict] = [_P4_DEFAULTS.copy()] * n_days

        # Determine re-optimisation dates
        reopt_days: list[int] = []
        for i in range(c.warmup_days, n_days, c.reopt_freq):
            reopt_days.append(i)

        current_params = _P4_DEFAULTS.copy()
        next_reopt_idx = 0

        for i in range(n_days):
            # Check if we hit a re-optimisation day
            if next_reopt_idx < len(reopt_days) and i == reopt_days[next_reopt_idx]:
                opt_start = max(0, i - c.opt_window)
                opt_end = i  # exclusive: data up to but not including today

                best_params = self._optimise_params(
                    mom12.iloc[opt_start:opt_end],
                    mom3.iloc[opt_start:opt_end],
                    rsi_vals.iloc[opt_start:opt_end],
                    zsc.iloc[opt_start:opt_end],
                    above.iloc[opt_start:opt_end],
                    fwd_ret.iloc[opt_start:opt_end],
                    avail,
                    c,
                )
                current_params = best_params
                next_reopt_idx += 1

            params_timeline[i] = current_params.copy()

        # Now build the weight matrix using the per-day params
        weights = self._build_weights(
            p, mom12, mom3, rsi_vals, zsc, above, params_timeline, c,
        )

        # --- Phase 2: intraday vol-regime overlay ---
        if self._intraday_features is not None:
            weights = self._apply_intraday_overlay(weights)

        return weights.reindex(
            columns=prices.columns, fill_value=0.0,
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # ------------------------------------------------------------------
    # Walk-forward parameter optimisation (uses only past data)
    # ------------------------------------------------------------------

    def _optimise_params(
        self,
        mom12: pd.DataFrame,
        mom3: pd.DataFrame,
        rsi_vals: pd.DataFrame,
        zsc: pd.DataFrame,
        above: pd.DataFrame,
        fwd_ret: pd.DataFrame,
        avail: list[str],
        c: AdaptiveThresholdConfig,
    ) -> dict:
        """Find the best param set via TimeSeriesSplit CV on the training window."""
        n = len(mom12)
        if n < 126:  # need at least 6 months
            return _P4_DEFAULTS.copy()

        tscv = TimeSeriesSplit(n_splits=c.cv_splits)
        indices = np.arange(n)

        best_sharpe = -np.inf
        best_params = _P4_DEFAULTS.copy()

        for mw, vw, tw, tn in _P4_PARAM_GRID:
            fold_sharpes: list[float] = []

            for train_idx, val_idx in tscv.split(indices):
                # Build composite score on validation fold
                val_mom12 = mom12.iloc[val_idx]
                val_mom3 = mom3.iloc[val_idx]
                val_rsi = rsi_vals.iloc[val_idx]
                val_zsc = zsc.iloc[val_idx]
                val_above = above.iloc[val_idx]
                val_fwd = fwd_ret.iloc[val_idx]

                composite = self._composite_score(
                    val_mom12, val_mom3, val_rsi, val_zsc, val_above, mw, vw, tw,
                )

                # Rank and pick top-N
                ranks = composite.rank(axis=1, ascending=False)
                top_n = min(tn, len(avail))
                is_top = (ranks <= top_n).astype(float)

                # Equal-weight portfolio return for top-N each day
                n_held = is_top.sum(axis=1).clip(lower=1)
                port_ret = (is_top.mul(val_fwd)).sum(axis=1) / n_held

                # Sharpe of this fold
                port_ret_clean = port_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if port_ret_clean.std() > 0:
                    sharpe = (port_ret_clean.mean() / port_ret_clean.std()) * np.sqrt(252)
                else:
                    sharpe = 0.0
                fold_sharpes.append(sharpe)

            avg_sharpe = np.mean(fold_sharpes)
            if avg_sharpe > best_sharpe:
                best_sharpe = avg_sharpe
                best_params = {
                    "mom_weight": mw,
                    "vol_weight": vw,
                    "trend_weight": tw,
                    "top_n": tn,
                }

        return best_params

    # ------------------------------------------------------------------
    # Composite score from 5 signals
    # ------------------------------------------------------------------

    @staticmethod
    def _composite_score(
        mom12: pd.DataFrame,
        mom3: pd.DataFrame,
        rsi_vals: pd.DataFrame,
        zsc: pd.DataFrame,
        above: pd.DataFrame,
        mom_weight: float,
        vol_weight: float,
        trend_weight: float,
    ) -> pd.DataFrame:
        """Weighted composite score from ranked signals.

        mom_weight  -> 12-month + 3-month momentum (averaged)
        vol_weight  -> inverse RSI rank (lower RSI = mean-reversion upside)
        trend_weight -> above-200SMA + z-score rank (mean reversion upside)
        """
        # Momentum: average of 12m and 3m return ranks
        mom_rank = (
            mom12.rank(axis=1, pct=True).fillna(0.5)
            + mom3.rank(axis=1, pct=True).fillna(0.5)
        ) / 2.0

        # Mean-reversion / vol: inverse RSI (low RSI = oversold = buy signal)
        rsi_rank = (1.0 - rsi_vals.rank(axis=1, pct=True)).fillna(0.5)

        # Trend: combo of above-SMA and inverse z-score rank
        zsc_rank = (1.0 - zsc.rank(axis=1, pct=True)).fillna(0.5)
        trend_score = (above + zsc_rank) / 2.0

        return mom_weight * mom_rank + vol_weight * rsi_rank + trend_weight * trend_score

    # ------------------------------------------------------------------
    # Build weight matrix from signals + parameter timeline
    # ------------------------------------------------------------------

    def _build_weights(
        self,
        p: pd.DataFrame,
        mom12: pd.DataFrame,
        mom3: pd.DataFrame,
        rsi_vals: pd.DataFrame,
        zsc: pd.DataFrame,
        above: pd.DataFrame,
        params_timeline: list[dict],
        c: AdaptiveThresholdConfig,
    ) -> pd.DataFrame:
        """Build daily weights using the parameter selected for each day.

        To avoid a per-row loop for the 21-day hold logic, we process
        in rebalance blocks: every rebalance_freq days, compute composite
        with that day's params, select top-N, and forward-fill until the
        next rebalance.
        """
        avail = list(p.columns)
        n_days = len(p)
        weights = pd.DataFrame(0.0, index=p.index, columns=avail)

        # Rebalance days (every rebalance_freq trading days)
        rebal_days = list(range(0, n_days, c.rebalance_freq))

        for rb_idx, rb_day in enumerate(rebal_days):
            params = params_timeline[rb_day]

            composite = self._composite_score(
                mom12.iloc[[rb_day]],
                mom3.iloc[[rb_day]],
                rsi_vals.iloc[[rb_day]],
                zsc.iloc[[rb_day]],
                above.iloc[[rb_day]],
                params["mom_weight"],
                params["vol_weight"],
                params["trend_weight"],
            )

            top_n = min(params["top_n"], len(avail))
            ranks = composite.rank(axis=1, ascending=False)
            is_top = (ranks <= top_n).iloc[0]

            per_asset = min(c.leverage / max(top_n, 1), c.max_weight)

            # Determine hold window
            next_rb = rebal_days[rb_idx + 1] if rb_idx + 1 < len(rebal_days) else n_days
            for t in avail:
                if is_top.get(t, False):
                    weights.iloc[rb_day:next_rb, weights.columns.get_loc(t)] = per_asset

        return weights

    def _apply_intraday_overlay(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Scale weights by intraday vol-regime signal from Alpaca features.

        When SPY's realised intraday volatility is elevated relative to its
        63-day rolling mean, exposure is scaled down (max 30% reduction).
        When vol is calm, a modest uplift (up to 10%) is permitted.
        This overlay is only active when ``set_intraday_features`` has been
        called with valid data; otherwise weights pass through unchanged.

        Look-ahead safety: the rolling mean is shifted by 1 day so only
        yesterday's vol level influences today's position sizes.
        """
        feat = self._intraday_features.reindex(weights.index).ffill()
        if "SPY_realized_vol_1min" not in feat.columns:
            return weights

        spy_rv = feat["SPY_realized_vol_1min"].fillna(
            feat["SPY_realized_vol_1min"].expanding().mean()
        )
        rv_mean = (
            spy_rv.rolling(63, min_periods=20).mean()
            .shift(1)
            .fillna(spy_rv.expanding().mean())
        )
        rv_ratio = (spy_rv / rv_mean.clip(lower=1e-8)).fillna(1.0).clip(0.1, 5.0)

        multiplier = pd.Series(
            np.where(rv_ratio > 1.5, 0.70,
                     np.where(rv_ratio < 0.70, 1.10, 1.0)),
            index=weights.index,
        )
        return weights.multiply(multiplier, axis=0)

    def __repr__(self) -> str:
        return f"AdaptiveThreshold(reopt={self.cfg.reopt_freq}d)"


# =====================================================================
# P5 — CrossSectionalRanker (XGBoost Asset Ranking)
# =====================================================================

_P5_FULL_UNIVERSE = (
    "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "SLV", "TLT", "IEF",
    "SHY", "UUP", "XLE", "USO", "XOP", "ITA", "LMT", "RTX", "XLK",
    "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV", "XBI", "XLC",
    "XLRE", "DBC", "DBA", "TIP", "AGG", "EMB", "FXI", "VGK", "EWJ",
    "INDA", "VNQ", "HYG", "LQD", "BTC-USD", "ETH-USD",
)

_FEATURE_NAMES = [
    "ret_1m", "ret_3m", "ret_12m",
    "rvol_20d",
    "rsi_14",
    "zscore_60d",
    "above_200sma",
    "vol_of_vol",
]


@dataclass
class CrossSectionalRankerConfig:
    """Config for P5-CrossSectionalRanker."""

    tickers: tuple[str, ...] = _P5_FULL_UNIVERSE
    min_train_days: int = 252    # 1 year minimum training
    retrain_freq: int = 63       # quarterly retraining
    fwd_horizon: int = 21        # predict next-month return
    long_n: int = 8              # long top 8
    long_weight: float = 1.0     # total long exposure (1.0x, no leverage)
    max_position: float = 0.50   # max single-position weight
    rebalance_freq: int = 21     # hold positions for ~1 month

    # XGBoost hyperparameters (conservative to avoid overfit)
    n_estimators: int = 100
    max_depth: int = 3
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


class CrossSectionalRanker(Strategy):
    """XGBoost-based cross-sectional asset ranker.

    Walk-forward expanding-window model predicts next-month returns
    for all assets.  Longs the top-ranked assets, shorts the bottom.

    Thesis: Cross-sectional return predictability from momentum, vol,
    and mean-reversion features can be captured by a gradient-boosted
    tree with walk-forward discipline.  Expanding window and conservative
    XGBoost hyperparameters mitigate overfitting.
    """

    name = "P5-CrossSectionalRanker"

    def __init__(self, config: CrossSectionalRankerConfig | None = None) -> None:
        self.cfg = config or CrossSectionalRankerConfig()

    # ------------------------------------------------------------------
    # Feature computation (vectorized for all assets)
    # ------------------------------------------------------------------

    def _compute_features(self, prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Compute all features as DataFrames (dates x tickers).

        Returns a dict mapping feature name -> DataFrame.
        """
        p = prices

        ret_1m = _momentum(p, 21)
        ret_3m = _momentum(p, 63)
        ret_12m = _momentum(p, 252)
        rvol_20d = _realized_vol_df(p, 20)
        rsi_14 = _rsi_df(p, 14)
        zscore_60d = _zscore_df(p, 60)
        above_200sma = _above_sma(p, 200)

        # Vol-of-vol proxy: 20-day return vol / 60-day return vol
        rvol_60d = _realized_vol_df(p, 60)
        rvol_60d_safe = rvol_60d.replace(0.0, np.nan)
        vol_of_vol = (rvol_20d / rvol_60d_safe).replace(
            [np.inf, -np.inf], np.nan,
        ).fillna(1.0)

        return {
            "ret_1m": ret_1m,
            "ret_3m": ret_3m,
            "ret_12m": ret_12m,
            "rvol_20d": rvol_20d,
            "rsi_14": rsi_14,
            "zscore_60d": zscore_60d,
            "above_200sma": above_200sma,
            "vol_of_vol": vol_of_vol,
        }

    def _features_to_matrix(
        self,
        feats: dict[str, pd.DataFrame],
        tickers: list[str],
        start: int,
        end: int,
    ) -> np.ndarray:
        """Stack features for [start:end] rows into (n_rows * n_tickers, n_features).

        Row order: day0-ticker0, day0-ticker1, ..., day1-ticker0, ...
        """
        arrays = []
        for fname in _FEATURE_NAMES:
            block = feats[fname].iloc[start:end][tickers].values  # (n_days, n_tickers)
            arrays.append(block.ravel())  # flatten day-major
        X = np.column_stack(arrays)
        # Final NaN/inf guard
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    def _target_vector(
        self,
        fwd_ret: pd.DataFrame,
        tickers: list[str],
        start: int,
        end: int,
    ) -> np.ndarray:
        """Forward return target, same shape ordering as _features_to_matrix."""
        y = fwd_ret.iloc[start:end][tickers].values.ravel()
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # Core weight generation
    # ------------------------------------------------------------------

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        if len(avail) < c.long_n:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        p = prices[avail]
        n_days = len(p)

        # Pre-compute all features (vectorized)
        feats = self._compute_features(p)

        # Forward returns for training (shift(-horizon) means we only use
        # this in the training window where the answer is already known)
        fwd_ret = p.pct_change(c.fwd_horizon).shift(-c.fwd_horizon).fillna(0.0)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Walk-forward: expanding window, retrain every retrain_freq days
        model: XGBRegressor | None = None
        next_retrain = c.min_train_days

        # Rebalance days for position holding
        rebal_days = list(range(c.min_train_days, n_days, c.rebalance_freq))
        rebal_set = set(rebal_days)

        # Track current positions between rebalances
        current_long: list[str] = []

        for day_idx in range(c.min_train_days, n_days):
            # --- Retrain if needed ---
            if day_idx >= next_retrain:
                # Training window: [0, day_idx - fwd_horizon)
                # We exclude the last fwd_horizon days because their forward
                # returns extend beyond day_idx (would be look-ahead)
                train_end = day_idx - c.fwd_horizon
                if train_end > c.min_train_days // 2:
                    X_train = self._features_to_matrix(feats, avail, 0, train_end)
                    y_train = self._target_vector(fwd_ret, avail, 0, train_end)

                    model = XGBRegressor(
                        n_estimators=c.n_estimators,
                        max_depth=c.max_depth,
                        learning_rate=c.learning_rate,
                        subsample=c.subsample,
                        colsample_bytree=c.colsample_bytree,
                        random_state=c.random_state,
                        verbosity=0,
                    )
                    model.fit(X_train, y_train)

                next_retrain = day_idx + c.retrain_freq

            # --- Rebalance if it's a rebalance day and model exists ---
            if day_idx in rebal_set and model is not None:
                # Features for TODAY only (no future data)
                X_today = self._features_to_matrix(feats, avail, day_idx, day_idx + 1)
                preds = model.predict(X_today)  # (n_tickers,)

                # Rank assets by predicted return
                pred_series = pd.Series(preds, index=avail)
                ranked = pred_series.rank(ascending=False)

                long_n = min(c.long_n, len(avail) // 2)

                current_long = ranked[ranked <= long_n].index.tolist()

            # --- Assign weights ---
            if model is not None:
                per_long = min(c.long_weight / max(len(current_long), 1), c.max_position)
                for t in current_long:
                    weights.iloc[day_idx, weights.columns.get_loc(t)] = per_long

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def __repr__(self) -> str:
        return (
            f"CrossSectionalRanker(long={self.cfg.long_n}, "
            f"retrain={self.cfg.retrain_freq}d)"
        )
