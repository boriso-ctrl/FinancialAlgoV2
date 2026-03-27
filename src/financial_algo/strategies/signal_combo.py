"""Category P: Signal Combination & ML-enhanced strategies.

P1 uses simple feature-based composite scoring (numpy/pandas only).
P2 uses XGBoost for walk-forward signal combination.
P3 uses Gaussian Mixture Models for soft regime classification.
P7 uses price-derived fundamental momentum (UMD + earnings surprise proxy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financial_algo.indicators import (
    breadth_count, ema, hurst_exponent, realized_vol, rmi, rolling_kurt,
    rolling_skew, rsi, rsi_change_rate, tsi, zscore,
)
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

    # Sprint-7 new feature parameters
    hurst_window: int = 126       # shorter window reduces warm-up latency

    # Weights for original 3 features
    # (scale doesn't matter — composite is ranked cross-sectionally)
    # New feature weights (each 0.05 — conservative blend)
    tsi_weight: float = 0.05
    rmi_weight: float = 0.05
    skew_weight: float = 0.05
    kurt_weight: float = 0.05
    rcr_weight: float = 0.05
    hurst_weight: float = 0.05

    # Intraday feature weights (Phase 2)
    intraday_gap_weight: float = 0.03      # opening gap rank
    intraday_vwap_weight: float = 0.03     # VWAP deviation rank
    intraday_c2h_weight: float = 0.03      # close-to-high rank

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
        self._intraday_features: pd.DataFrame | None = None

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

        # Daily returns (reused by skew, kurt features)
        returns = p.pct_change().fillna(0.0)

        # Feature 4: TSI (True Strength Index) — normalized /100, then ranked
        tsi_df = pd.DataFrame(
            {t: tsi(p[t]) / 100.0 for t in avail}, index=p.index
        ).fillna(0.0)
        tsi_rank = tsi_df.rank(axis=1, pct=True).fillna(0.5)

        # Feature 5: RMI (Relative Momentum Index) — centered & normalized, then ranked
        rmi_df = pd.DataFrame(
            {t: (rmi(p[t]) - 50.0) / 50.0 for t in avail}, index=p.index
        ).fillna(0.0)
        rmi_rank = rmi_df.rank(axis=1, pct=True).fillna(0.5)

        # Feature 6: Rolling skew — inverted (less negative = better = higher rank)
        skew_df = pd.DataFrame(
            {t: rolling_skew(returns[t]) for t in avail}, index=p.index
        ).fillna(0.0)
        skew_rank = (1.0 - skew_df.rank(axis=1, pct=True)).fillna(0.5)

        # Feature 7: Rolling kurtosis — inverted (lower tail risk = better)
        kurt_df = pd.DataFrame(
            {t: rolling_kurt(returns[t]) for t in avail}, index=p.index
        ).fillna(0.0)
        kurt_rank = (1.0 - kurt_df.rank(axis=1, pct=True)).fillna(0.5)

        # Feature 8: RSI change rate (accelerating momentum = higher rank)
        rcr_df = pd.DataFrame(
            {t: rsi_change_rate(p[t]) for t in avail}, index=p.index
        ).fillna(0.0)
        rcr_rank = rcr_df.rank(axis=1, pct=True).fillna(0.5)

        # Feature 9: Hurst exponent (trending > 0.5 = momentum-friendly)
        hurst_df = pd.DataFrame(
            {t: hurst_exponent(p[t], window=c.hurst_window) for t in avail},
            index=p.index,
        ).fillna(0.5)
        hurst_rank = hurst_df.rank(axis=1, pct=True).fillna(0.5)

        # Composite score (weighted sum — ranking handles scale differences)
        composite = (
            c.mom_weight * mom_rank
            + c.vol_weight * vol_rank
            + c.trend_weight * above_sma
            + c.tsi_weight * tsi_rank
            + c.rmi_weight * rmi_rank
            + c.skew_weight * skew_rank
            + c.kurt_weight * kurt_rank
            + c.rcr_weight * rcr_rank
            + c.hurst_weight * hurst_rank
        )

        # --- Intraday features (Phase 2) ---
        composite = self._add_intraday_features_p1(composite, avail, c)

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

        # --- Intraday vol-regime overlay ---
        if self._intraday_features is not None:
            weights = self._apply_intraday_overlay_p1(weights)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _add_intraday_features_p1(
        self,
        composite: pd.DataFrame,
        avail: list[str],
        c: FeatureComboConfig,
    ) -> pd.DataFrame:
        """Add cross-sectionally ranked intraday features to the composite."""
        if self._intraday_features is None:
            return composite
        feat = self._intraday_features.reindex(composite.index).ffill()

        # Opening gap rank (higher positive gap = momentum confirmation)
        gap_df = pd.DataFrame(
            {t: feat.get(f"{t}_opening_gap", pd.Series(0.0, index=composite.index))
             for t in avail},
            index=composite.index,
        ).fillna(0.0)
        gap_rank = gap_df.rank(axis=1, pct=True).fillna(0.5)

        # VWAP deviation rank (positive = closing above VWAP = demand)
        vwap_df = pd.DataFrame(
            {t: feat.get(f"{t}_vwap_deviation", pd.Series(0.0, index=composite.index))
             for t in avail},
            index=composite.index,
        ).fillna(0.0)
        vwap_rank = vwap_df.rank(axis=1, pct=True).fillna(0.5)

        # Close-to-high rank (inverted: closer to high = better)
        c2h_df = pd.DataFrame(
            {t: feat.get(f"{t}_close_to_high", pd.Series(0.5, index=composite.index))
             for t in avail},
            index=composite.index,
        ).fillna(0.5)
        # Lower close_to_high = closed nearer to high = better
        c2h_rank = (1.0 - c2h_df.rank(axis=1, pct=True)).fillna(0.5)

        composite = (
            composite
            + c.intraday_gap_weight * gap_rank
            + c.intraday_vwap_weight * vwap_rank
            + c.intraday_c2h_weight * c2h_rank
        )
        return composite

    def _apply_intraday_overlay_p1(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Scale P1 weights by intraday vol regime (same logic as P2)."""
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
            np.where(rv_ratio > 1.5, 0.70, np.where(rv_ratio < 0.70, 1.10, 1.0)),
            index=weights.index,
        )
        return weights.multiply(multiplier, axis=0)


# =========================================================================
# P2 -- XGBoost Signal Combination (ML-enhanced upgrade of P1)
# =========================================================================

@dataclass
class XGBoostSignalComboConfig:
    """Walk-forward XGBoost signal combo: 9 features, cross-sectional."""

    tickers: tuple = ("SPY", "QQQ", "IWM", "GLD", "TLT", "XLE", "EFA", "EEM")

    # Feature parameters (P1 originals + new)
    mom_12m: int = 252
    mom_3m: int = 63
    mom_1m: int = 21
    vol_window: int = 20
    sma_window: int = 200
    rsi_period: int = 14
    zscore_window: int = 60

    # XGBoost hyperparameters
    n_estimators: int = 200
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # Walk-forward schedule
    min_train_days: int = 504
    retrain_freq: int = 63          # quarterly retrain
    forward_return_window: int = 21  # predict 21-day fwd return
    training_stride: int = 21       # monthly stride (non-overlapping samples)

    # Allocation
    top_n: int = 4
    leverage: float = 1.5
    rebalance_freq: int = 21        # monthly rebalance

    # P1 fallback weights (warm-up period)
    p1_mom_weight: float = 0.40
    p1_vol_weight: float = 0.30
    p1_trend_weight: float = 0.30


class XGBoostSignalCombo(Strategy):
    """XGBoost-enhanced multi-factor signal combination.

    Upgrade of P1-FeatureComboSignal.  Uses XGBRegressor trained via
    walk-forward expanding window to predict next-period relative returns
    from 9 cross-sectional features.  Falls back to P1 fixed-weight logic
    during the 252-day warm-up period before enough training data exists.

    Features (per asset per date):
        1. 12-month momentum rank   5. 3-month momentum
        2. 20-day vol rank (inv.)   6. RSI(14) normalised
        3. Above-200-SMA flag       7. 60-day price z-score
        4. 1-month momentum         8. Credit spread z-score (shared)
                                    9. VIX proxy level (shared)
    """

    name = "P2-XGBoostSignalCombo"

    def __init__(self, config: XGBoostSignalComboConfig | None = None) -> None:
        self.cfg = config or XGBoostSignalComboConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        from xgboost import XGBRegressor

        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(avail) < 3:
            return weights

        p = prices[avail]
        n_days = len(p)
        n_assets = len(avail)

        # ---- Vectorised feature computation ----

        # F1: 12-month momentum rank (cross-sectional)
        mom_12m_rank = (
            p.pct_change(c.mom_12m).fillna(0.0)
            .rank(axis=1, pct=True).fillna(0.5)
        )
        # F2: 20-day vol rank (inverted: low vol = high score)
        vol_df = pd.DataFrame(
            {t: realized_vol(p[t], c.vol_window) for t in avail},
            index=p.index,
        ).fillna(0.0)
        vol_rank = (1.0 - vol_df.rank(axis=1, pct=True)).fillna(0.5)

        # F3: Above 200-day SMA flag
        sma = p.rolling(c.sma_window, min_periods=100).mean()
        above_sma = (p >= sma).astype(float).fillna(0.0)

        # F4: 1-month momentum (raw)
        mom_1m = p.pct_change(c.mom_1m).fillna(0.0)

        # F5: 3-month momentum (raw)
        mom_3m = p.pct_change(c.mom_3m).fillna(0.0)

        # F6: RSI(14) normalised to [0, 1]
        rsi_df = pd.DataFrame(index=p.index, columns=avail, dtype=float)
        for t in avail:
            rsi_df[t] = rsi(p[t].values.tolist(), c.rsi_period)
        rsi_df = rsi_df.fillna(50.0) / 100.0

        # F7: 60-day price z-score
        zscore_df = pd.DataFrame(
            {t: zscore(p[t], c.zscore_window) for t in avail},
            index=p.index,
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # F8: Credit spread z-score (HYG/LQD) -- shared across assets
        if "HYG" in prices.columns and "LQD" in prices.columns:
            credit_ratio = prices["HYG"] / prices["LQD"]
            credit_ratio = credit_ratio.replace(
                [np.inf, -np.inf], np.nan
            ).ffill().fillna(1.0)
            credit_z = zscore(credit_ratio, c.zscore_window)
            credit_z = credit_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        else:
            credit_z = pd.Series(0.0, index=p.index)

        # F9: VIX proxy (SPY 20d realised vol x 100) -- shared
        if "SPY" in prices.columns:
            vix_proxy = realized_vol(prices["SPY"], c.vol_window).fillna(0.0) * 100
        else:
            vix_proxy = pd.Series(0.0, index=p.index)

        # ---- Intraday features (Phase 2, 7 per asset) ------------------
        intra_arrays = self._build_intraday_features_p2(avail, n_days, n_assets, prices)

        # ---- Build 3-D feature tensor -----------------------------------
        n_features_base = 9
        n_intraday = len(intra_arrays)        # 0 when _intraday_features is None, 7 otherwise

        credit_z_bc = np.broadcast_to(
            credit_z.values[:, np.newaxis], (n_days, n_assets)
        ).copy()
        vix_bc = np.broadcast_to(
            vix_proxy.values[:, np.newaxis], (n_days, n_assets)
        ).copy()

        base_features = [
            mom_12m_rank[avail].values,
            vol_rank[avail].values,
            above_sma[avail].values,
            mom_1m[avail].values,
            mom_3m[avail].values,
            rsi_df[avail].values,
            zscore_df[avail].values,
            credit_z_bc,
            vix_bc,
        ]
        all_features = base_features + intra_arrays

        feat_3d = np.stack(all_features, axis=-1).astype(np.float64)
        feat_3d = np.nan_to_num(feat_3d, nan=0.0, posinf=0.0, neginf=0.0)
        n_features = n_features_base + n_intraday

        # Forward returns (target): return from day d to d + fw window
        fwd_ret = (p.shift(-c.forward_return_window) / p - 1.0)
        fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        fwd_ret_arr = fwd_ret[avail].values

        # ---- Local P1 fallback helper ----
        def p1_fallback(idx: int, date_sl: pd.DatetimeIndex) -> None:
            # Top-4 by 3-month momentum, equal weight 0.25 each
            top = min(c.top_n, n_assets)
            top_assets = mom_3m.iloc[idx].nlargest(top).index
            for t in top_assets:
                weights.loc[date_sl, t] = 0.25

        # ---- Walk-forward loop ----
        rebal_dates = list(range(0, n_days, c.rebalance_freq))
        model = None
        last_train_idx = -c.retrain_freq
        min_feat_idx = max(c.vol_window, c.rsi_period)

        for rebal_idx in rebal_dates:
            hold_end = min(rebal_idx + c.rebalance_freq, n_days)
            date_sl = p.index[rebal_idx:hold_end]

            # Phase 1: warm-up => P1 fallback
            if rebal_idx < c.min_train_days and model is None:
                if rebal_idx >= min_feat_idx:
                    p1_fallback(rebal_idx, date_sl)
                continue

            # Phase 2: retrain if schedule requires it
            if (rebal_idx - last_train_idx >= c.retrain_freq) or model is None:
                train_end = rebal_idx - c.forward_return_window
                feat_start = max(c.vol_window, c.rsi_period)
                if train_end < feat_start:
                    p1_fallback(rebal_idx, date_sl)
                    continue

                sample_idxs = np.arange(
                    feat_start, train_end + 1, c.training_stride
                )
                if len(sample_idxs) * n_assets < 10:
                    p1_fallback(rebal_idx, date_sl)
                    continue

                X_train = feat_3d[sample_idxs].reshape(-1, n_features)
                y_train = fwd_ret_arr[sample_idxs].reshape(-1)
                X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
                y_train = np.nan_to_num(y_train, nan=0.0, posinf=0.0, neginf=0.0)

                model = XGBRegressor(
                    n_estimators=c.n_estimators,
                    max_depth=c.max_depth,
                    learning_rate=c.learning_rate,
                    subsample=c.subsample,
                    colsample_bytree=c.colsample_bytree,
                    random_state=42,
                    verbosity=0,
                )
                model.fit(X_train, y_train)
                last_train_idx = rebal_idx

            if model is None:
                continue

            # Phase 3: predict & allocate top-N
            X_pred = feat_3d[rebal_idx]
            X_pred = np.nan_to_num(
                X_pred.reshape(-1, n_features), nan=0.0, posinf=0.0, neginf=0.0
            )
            scores = model.predict(X_pred)
            scored = pd.Series(scores.ravel(), index=avail)
            top_n = min(c.top_n, n_assets)
            top_assets = scored.nlargest(top_n).index
            per_w = c.leverage / top_n
            for t in top_assets:
                weights.loc[date_sl, t] = per_w

        weights = pd.DataFrame(
            np.clip(weights.values, -0.25, 0.25),
            index=prices.index,
            columns=prices.columns,
        )

        # --- Phase 2: intraday vol-regime overlay ---
        if self._intraday_features is not None:
            weights = self._apply_intraday_overlay(weights)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _apply_intraday_overlay(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Scale weights up/down based on intraday vol regime.

        Mirrors P4's overlay: high SPY intraday RV => reduce exposure 30%,
        calm intraday RV => allow modest 10% uplift.  No-op when
        ``_intraday_features`` is None or lacks the required column.

        Look-ahead safety: rolling mean is shifted 1 day so yesterday's
        vol level drives today's position sizes.
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

    def _build_intraday_features_p2(
        self,
        avail: list[str],
        n_days: int,
        n_assets: int,
        prices: pd.DataFrame,
    ) -> list[np.ndarray]:
        """Extract 7 intraday feature arrays aligned with the price index.

        Returns a list of 7 numpy arrays each shaped (n_days, n_assets),
        or an empty list when intraday features are not available.
        """
        if self._intraday_features is None:
            return []

        feat = self._intraday_features.reindex(prices.index).ffill()
        feature_names = [
            "realized_vol_1min", "vwap_deviation", "opening_gap",
            "intraday_range", "vol_of_vol", "volume_surprise", "close_to_high",
        ]
        arrays: list[np.ndarray] = []
        for fname in feature_names:
            cols = [f"{t}_{fname}" for t in avail]
            df = pd.DataFrame(
                {c: feat[c] if c in feat.columns else 0.0 for c in cols},
                index=prices.index,
            ).fillna(0.0)
            # Cross-sectional z-score for better XGBoost input
            row_mean = df.mean(axis=1)
            row_std = df.std(axis=1).clip(lower=1e-8)
            z = df.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)
            z = z.clip(-3.0, 3.0)  # winsorize
            arrays.append(z.values)
        return arrays


# =========================================================================
# P3 -- GMM Regime Classifier (soft-probability regime strategy)
# =========================================================================

@dataclass
class GMMRegimeConfig:
    """Config for GMM-based soft regime classification strategy.

    REWORK (Sprint 15.1 Phase 13):
    - Fixed look-ahead bias: train on data[0:rebal_idx], NOT including rebal_idx
    - Added feature standardization (StandardScaler)
    - Simplified regime detection logic
    - Removed short positions (toxic in crises)
    - Better allocation mapping based on crisis probability
    """

    vol_window: int = 20
    credit_zscore_window: int = 60
    breadth_lookback: int = 50

    n_components: int = 3          # reduced from 4 for better stability
    min_train_days: int = 252
    refit_freq: int = 63
    rebalance_freq: int = 21

    # Position tickers (removed shorts to prevent blow-ups)
    risk_assets: tuple = ("SPY", "QQQ", "EEM")
    safe_assets: tuple = ("GLD", "TLT")
    dollar_asset: str = "UUP"

    # Position sizing (smoother, non-binary)
    max_leverage_risk: float = 1.6   # risk-on mode
    max_leverage_safe: float = 0.8   # risk-off mode
    max_position: float = 0.50


class GMMRegimeClassifier(Strategy):
    """Gaussian Mixture Model for soft regime classification.

    REWORK: Fixed 0.26 Sharpe by:
    1. Eliminating look-ahead bias in GMM training
    2. Adding feature standardization (StandardScaler)
    3. Simplifying crisis component detection
    4. Removing short positions (problematic in crises)
    5. Smoother regime-to-allocation mapping

    Thesis: Fit a 3-component GMM on [VIX proxy, realized vol, credit ZScore]
    using strict walk-forward (no look-ahead). Identify crisis component
    (highest mean VIX). Use posterior crisis probability to continuously
    scale from risk-on (crisis_prob=0) to risk-off (crisis_prob=1).
    """

    name = "P3-GMMRegimeClassifier"

    def __init__(self, config: GMMRegimeConfig | None = None) -> None:
        self.cfg = config or GMMRegimeConfig()
        self._scaler = None

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n_days = len(prices)

        if "SPY" not in prices.columns:
            return weights

        spy = prices["SPY"]

        # Build features (NO NaN or inf propagation)
        vix_proxy = (realized_vol(spy, c.vol_window) * 100).fillna(20.0)
        real_vol = realized_vol(spy, c.vol_window).fillna(0.15)

        if "HYG" in prices.columns and "LQD" in prices.columns:
            credit_ratio = (prices["HYG"] / prices["LQD"]).replace(
                [np.inf, -np.inf], np.nan
            ).fillna(1.0)
            credit_z = zscore(credit_ratio, c.credit_zscore_window).fillna(0.0)
            credit_z = credit_z.replace([np.inf, -np.inf], 0.0)
        else:
            credit_z = pd.Series(0.0, index=prices.index)

        # Stack features: (n_days, 3)
        feat_arr = np.column_stack([
            vix_proxy.values,
            real_vol.values,
            credit_z.values,
        ]).astype(np.float64)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Available tickers (long only, no shorts)
        risk_assets = [t for t in c.risk_assets if t in prices.columns]
        safe_assets = [t for t in c.safe_assets if t in prices.columns]
        dollar_asset = c.dollar_asset if c.dollar_asset in prices.columns else None

        if not risk_assets or not safe_assets:
            return weights

        # Walk-forward GMM
        rebal_dates = list(range(c.min_train_days, n_days, c.rebalance_freq))
        gmm = None
        crisis_comp = 0
        last_fit_idx = -c.refit_freq
        scaler = StandardScaler()
        features_standardized = None

        for rebal_idx in rebal_dates:
            hold_end = min(rebal_idx + c.rebalance_freq, n_days)
            date_sl = prices.index[rebal_idx:hold_end]

            # Refit GMM on schedule (FIX: train on data[0:rebal_idx], NOT including rebal_idx)
            if (rebal_idx - last_fit_idx >= c.refit_freq) or gmm is None:
                # Use data from 0 to rebal_idx (STRICT walk-forward, no look-ahead)
                train_data = feat_arr[:rebal_idx]
                if len(train_data) < c.min_train_days:
                    continue

                # Standardize features for GMM (critical for scaling-sensitive algorithm)
                features_standardized = scaler.fit_transform(train_data)

                gmm = GaussianMixture(
                    n_components=c.n_components,
                    covariance_type="diag",  # simpler, more stable
                    random_state=42,
                    n_init=5,
                    max_iter=100,
                )
                gmm.fit(features_standardized)

                # Crisis component = highest mean VIX (column 0 of standardized data)
                crisis_comp = int(np.argmax(gmm.means_[:, 0]))
                last_fit_idx = rebal_idx

            if gmm is None or features_standardized is None:
                continue

            # Standardize current day using the fitted scaler
            current_feat = feat_arr[rebal_idx:rebal_idx + 1]
            current_feat_scaled = scaler.transform(current_feat)

            # Get posterior probability of crisis component
            probs = gmm.predict_proba(current_feat_scaled)[0]
            crisis_prob = float(np.clip(probs[crisis_comp], 0.0, 1.0))

            # Smooth allocation: continuous blend between risk-on and reisk-off
            # crisis_prob = 0 -> full risk-on
            # crisis_prob = 0.5 -> 50/50 mix
            # crisis_prob = 1 -> full risk-off
            risk_weight = 1.0 - crisis_prob
            defensive_weight = crisis_prob

            # Allocate to risk assets (long only)
            n_risk = len(risk_assets)
            if n_risk > 0:
                risk_per_asset = (c.max_leverage_risk * risk_weight) / n_risk
                risk_per_asset = min(risk_per_asset, c.max_position)
                for t in risk_assets:
                    weights.loc[date_sl, t] = risk_per_asset

            # Allocate to safe assets
            n_safe = len(safe_assets)
            if n_safe > 0:
                safe_per_asset = (c.max_leverage_safe * defensive_weight) / n_safe
                safe_per_asset = min(safe_per_asset, c.max_position)
                for t in safe_assets:
                    weights.loc[date_sl, t] = safe_per_asset

            # Optional: dollar hedge in crisis (anti-risk sentiment)
            if dollar_asset is not None and crisis_prob > 0.65:
                weights.loc[date_sl, dollar_asset] = 0.2 * defensive_weight

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# G6 -- Multi-Signal Consensus (long-only, majority-vote entry)
# =========================================================================

@dataclass
class MultiSignalConsensusConfig:
    """Config for G6-MultiSignalConsensus."""

    tickers: list = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD",
        "TLT", "XLE", "XLK", "XLF", "XLI", "XLV",
    ])
    signal_threshold: int = 3        # need >= this many signals to go long
    trend_window: int = 50           # SMA window for breadth signal
    mom_short: int = 21              # 1-month lookback
    mom_long: int = 252              # 12-month lookback
    rsi_period: int = 14
    rsi_low: float = 40.0
    rsi_high: float = 65.0
    vol_window: int = 20             # for realized vol
    spy_vol_threshold: float = 0.25  # annualized SPY vol threshold
    target_vol: float = 0.10         # per-asset vol target
    max_leverage: float = 1.5


class MultiSignalConsensus(Strategy):
    """G6: Multi-Signal Consensus -- long-only majority-vote strategy.

    Thesis: Only deploy capital when a majority of orthogonal signals agree.
    Five independent signals (each 0 or 1) are computed per asset and summed
    into a consensus score (0-5). Positions are opened only when the score
    meets the threshold (default: 3 out of 5).

    Signals:
      1. Momentum   -- 12m minus 1m return > 0
      2. Trend      -- TSI > 0
      3. Breadth    -- pct of universe above SMA(50) > 0.6  [market-wide]
      4. RSI sweet  -- RSI(14) between 40 and 65
      5. Vol regime -- SPY 20d annualized vol < 0.25        [market-wide]
    """

    name = "G6-MultiSignalConsensus"

    def __init__(self, config: MultiSignalConsensusConfig | None = None) -> None:
        self.cfg = config or MultiSignalConsensusConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if len(avail) < 3 or "SPY" not in prices.columns:
            return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        p = prices[avail]

        # --- Signal 1: Momentum (12m minus 1m return > 0) ---
        mom = (
            p.pct_change(c.mom_long).fillna(0.0)
            - p.pct_change(c.mom_short).fillna(0.0)
        )
        sig_mom = (pd.notna(mom) & (mom > 0)).astype(float)

        # --- Signal 2: Trend (TSI > 0) ---
        tsi_df = pd.DataFrame(
            {t: tsi(p[t]) for t in avail}, index=p.index
        ).fillna(0.0)
        sig_trend = (pd.notna(tsi_df) & (tsi_df > 0)).astype(float)

        # --- Signal 3: Breadth (pct universe > SMA(50) > 0.6, market-wide) ---
        sma50 = p.rolling(c.trend_window, min_periods=max(1, c.trend_window // 2)).mean()
        above_sma = (p > sma50).astype(float).fillna(0.0)
        breadth = above_sma.mean(axis=1).fillna(0.0)
        sig_breadth_vals = (breadth > 0.6).astype(float).values
        sig_breadth_df = pd.DataFrame(
            np.outer(sig_breadth_vals, np.ones(len(avail))),
            index=prices.index, columns=avail,
        )

        # --- Signal 4: Mean-reversion ready (RSI(14) between 40 and 65) ---
        rsi_df = pd.DataFrame(index=prices.index, columns=avail, dtype=float)
        rsi_period = c.rsi_period
        for t in avail:
            delta = p[t].diff()
            gain = delta.clip(lower=0).ewm(span=rsi_period, adjust=False).mean()
            loss = (-delta).clip(lower=0).ewm(span=rsi_period, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi_val = (100 - 100 / (1 + rs)).fillna(50.0)
            rsi_df[t] = rsi_val
        rsi_df = rsi_df.fillna(50.0)
        sig_rsi = ((rsi_df >= c.rsi_low) & (rsi_df <= c.rsi_high)).astype(float)

        # --- Signal 5: Vol regime (SPY 20d annualized vol < threshold, market-wide) ---
        spy_vol = realized_vol(prices["SPY"], c.vol_window).fillna(0.5)
        spy_vol = spy_vol.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        sig_vol_vals = (spy_vol < c.spy_vol_threshold).astype(float).values
        sig_vol_df = pd.DataFrame(
            np.outer(sig_vol_vals, np.ones(len(avail))),
            index=prices.index, columns=avail,
        )

        # --- Consensus score (0-5 per asset) ---
        consensus = (
            sig_mom
            + sig_trend
            + sig_breadth_df
            + sig_rsi
            + sig_vol_df
        ).fillna(0.0)

        # Qualify assets that meet the signal threshold
        qualified = (consensus >= c.signal_threshold).astype(float)
        n_qualified = qualified.sum(axis=1)

        # Equal weight among qualifying assets, normalized to sum to 1.0
        eq_weights = qualified.copy()
        has_any = n_qualified > 0
        eq_weights.loc[has_any] = qualified.loc[has_any].div(
            n_qualified.loc[has_any], axis=0
        )
        eq_weights.loc[~has_any] = 0.0

        # Vol-scale per asset: target_vol / realized_vol, clipped [0.5, 2.0]
        for t in avail:
            rv = realized_vol(p[t], c.vol_window)
            rv = rv.replace([np.inf, -np.inf], np.nan).fillna(c.target_vol)
            rv = rv.replace(0, np.nan).fillna(c.target_vol)
            vol_scale = (c.target_vol / rv).clip(0.5, 2.0)
            eq_weights[t] = eq_weights[t] * vol_scale

        # Assign into output (only available tickers)
        for t in avail:
            weights[t] = eq_weights[t]

        # Enforce max gross leverage
        total_gross = weights.abs().sum(axis=1)
        scale_mask = total_gross > c.max_leverage
        if scale_mask.any():
            scale_factor = np.where(
                scale_mask,
                c.max_leverage / total_gross.clip(lower=1e-10),
                1.0,
            )
            weights = weights.multiply(scale_factor, axis=0)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# P8 -- Alpha158 Ranker (Cross-sectional ranking via Qlib-inspired features)
# =========================================================================

@dataclass
class Alpha158RankerConfig:
    """Config for P8-Alpha158Ranker.

    Feature weights calibrated via cross-sectional IC analysis on
    2010-2025 ETF data.  Positive-IC features (KBAR_4, STD_20, MAX_20)
    predict short-term mean-reversion: beaten-down, high-vol tickers
    bounce.  Weighted long-only tilt captures this without shorting costs.
    """

    # Feature weights -- empirically-validated top-IC features
    # Sign convention: positive weight = long when feature z-score is HIGH.
    # KBAR_4 (IC +0.09): (High-Close)/Open, high = dip, mean-reversion signal
    # STD_20 (IC +0.07): high recent vol = mean-reversion opportunity
    # MAX_20 (IC +0.06): distance from 20d max, high = dip buying
    # KBAR_6 (IC -0.05): intraday position, INVERTED (low = reversal)
    # STD_60 (IC +0.07): longer-window vol confirmation
    score_features: dict = field(default_factory=lambda: {
        "KBAR_4": 0.25,       # dip-buying signal (positive IC)
        "STD_20": 0.25,       # vol mean-reversion (positive IC)
        "MAX_20": 0.20,       # distance from high (positive IC)
        "STD_60": 0.15,       # vol confirmation (positive IC)
        "KBAR_6": -0.15,      # intraday position INVERTED (negative IC)
    })

    # Portfolio construction
    top_quantile: float = 0.25     # long top 25%
    bottom_quantile: float = 0.25  # underweight bottom 25%
    long_weight: float = 1.2       # overweight winners (mild leverage)
    short_weight: float = -0.15    # very light short (minimise carry drag)
    rebalance_freq: int = 5        # weekly rebalance

    # Warmup: Alpha158 needs 60d of history + 1d shift
    warmup: int = 62


class Alpha158Ranker(Strategy):
    """P8: Cross-sectional ranker using Alpha158 features.

    Thesis: Microsoft Qlib's Alpha158 features capture mean-reversion
    dynamics across ETFs.  Empirical IC analysis shows that dip-buying
    signals (KBAR_4, MAX_20) and vol-based features (STD_20, STD_60)
    have the strongest predictive power for short-term cross-sectional
    returns.  The composite overweights tickers that are beaten-down
    and volatile (mean-reversion), rebalanced weekly.

    Signal construction (all vectorized):
      1. Compute Alpha158 features from close prices (KBAR approximated).
      2. For each chosen feature, compute daily cross-sectional z-score.
      3. Composite = IC-weighted average of z-scored features.
      4. Long top quartile, light short bottom quartile, weekly rebalance.

    No row-level loops.  All features shifted +1 day inside the feature
    module to prevent look-ahead bias.
    """

    name = "P8-Alpha158Ranker"

    def __init__(self, config: Alpha158RankerConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or Alpha158RankerConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        from financial_algo.technical.alpha158 import (
            compute_alpha158_from_close,
            pivot_feature,
        )

        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if prices.empty or prices.shape[1] < 3:
            return weights

        # 1. Compute Alpha158 features from close prices
        features = compute_alpha158_from_close(prices)
        if features.empty:
            return weights

        # 2. For each scoring feature, pivot to (date x ticker) and z-score
        z_scores: list[pd.DataFrame] = []
        total_abs_weight = 0.0
        for feat_name, feat_weight in c.score_features.items():
            if feat_name not in features.columns:
                continue
            pivoted = pivot_feature(features, feat_name)
            # Align to prices index and columns
            pivoted = pivoted.reindex(
                index=prices.index, columns=prices.columns
            ).fillna(0.0)

            # Cross-sectional z-score per day
            row_mean = pivoted.mean(axis=1)
            row_std = pivoted.std(axis=1).replace(0, np.nan)
            z = pivoted.sub(row_mean, axis=0).div(row_std, axis=0)
            z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            z = z.clip(-3.0, 3.0)  # winsorize
            # feat_weight sign encodes direction: negative = invert z
            z_scores.append(z * feat_weight)
            total_abs_weight += abs(feat_weight)

        if not z_scores or total_abs_weight == 0:
            return weights

        # 3. Composite score (weighted sum of z-scores, normalized)
        composite = sum(z_scores) / total_abs_weight

        # 4. Cross-sectional rank per day (percentile)
        rank_pct = composite.rank(axis=1, pct=True).fillna(0.5)

        # 5. Long top quartile, light short bottom quartile
        top_mask = rank_pct >= (1.0 - c.top_quantile)
        bottom_mask = rank_pct <= c.bottom_quantile

        n_long = top_mask.sum(axis=1).clip(lower=1)
        n_short = bottom_mask.sum(axis=1).clip(lower=1)

        long_w = top_mask.astype(float).div(n_long, axis=0) * c.long_weight
        short_w = bottom_mask.astype(float).div(n_short, axis=0) * c.short_weight

        raw_weights = (long_w + short_w).fillna(0.0)

        # 6. Weekly rebalance: hold positions between rebalance dates
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        held = raw_weights.where(rebal_mask, np.nan).ffill().fillna(0.0)

        # 7. Zero out warmup period
        if c.warmup < len(prices):
            held.iloc[:c.warmup] = 0.0

        weights = held.reindex(columns=prices.columns, fill_value=0.0)
        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# P7 -- Fundamental Momentum Signal (price-derived UMD + surprise proxy)
# =========================================================================

@dataclass
class FundamentalMomentumConfig:
    """Price-derived fundamental momentum: 12-1 UMD + earnings surprise proxy.

    Since our universe is ETFs (no meaningful insider trading data),
    we use price-derived proxies for fundamental momentum:
    - 12-1 month momentum (Fama-French UMD factor)
    - Earnings surprise proxy from large daily price jumps
    - Short-term reversal filter to avoid contamination
    """

    # Universe -- broad ETF set
    tickers: tuple = (
        "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "SLV", "TLT",
        "SHY", "USO", "DBA", "DBC", "HYG", "AGG", "EMB", "TIP",
        "VNQ", "XBI", "FXI", "VGK", "EWJ", "INDA",
        "XLE", "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY",
        "XLV", "XLRE", "XLC",
    )

    # 12-1 momentum parameters
    mom_total_lookback: int = 252    # 12-month total return window
    mom_skip_recent: int = 21        # skip last month (avoid reversal)

    # Earnings surprise proxy parameters
    surprise_window: int = 63        # rolling window to count surprise events
    surprise_threshold: float = 2.0  # daily return z-score threshold
    surprise_zscore_window: int = 60 # lookback for daily return z-score

    # Short-term reversal filter
    reversal_lookback: int = 21      # 1-month return for reversal detection
    reversal_penalty: float = 0.3    # score penalty for recent losers

    # Signal combination weights
    umd_weight: float = 0.55         # 12-1 momentum weight
    surprise_weight: float = 0.30    # earnings surprise proxy weight
    vol_adj_weight: float = 0.15     # vol-adjusted momentum weight

    # Allocation
    top_pct: float = 0.25            # long top quartile
    leverage: float = 1.0            # fully-invested, no leverage
    rebalance_freq: int = 21         # monthly rebalance
    vol_window: int = 20             # realized vol for vol-adjustment

    # Optional: edgartools insider data cache path (None = disabled)
    insider_cache_path: str | None = None


class FundamentalMomentumSignal(Strategy):
    """Price-derived fundamental momentum: 12-1 UMD + earnings surprise proxy.

    Thesis
    ------
    Combines three well-documented alpha sources:

    1. **12-1 Momentum (UMD)**: 12-month return excluding the most recent
       month. The skip-month avoids short-term mean-reversion contamination.
       One of the most robust factors in academic literature (Jegadeesh &
       Titman 1993, Asness et al. 2013).

    2. **Earnings surprise proxy**: Large daily price moves (>2 sigma) signal
       fundamental surprises (earnings, macro news). Net positive surprises
       in a rolling window indicate persistent fundamental improvement.

    3. **Volatility-adjusted momentum**: Raw momentum scaled by inverse vol
       equalizes signal strength across high- and low-vol assets.

    Cross-sectionally ranks all three signals, combines with fixed weights,
    and goes long the top quartile. Monthly rebalance.

    Data note
    ---------
    Designed for ETF universe. If ``edgartools`` insider data cache is
    available (via separate fetch script), incorporates insider sentiment
    as a fourth signal. Gracefully degrades when unavailable.
    """

    name = "P7-FundamentalMomentumSignal"

    def __init__(self, config: FundamentalMomentumConfig | None = None) -> None:
        self.cfg = config or FundamentalMomentumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if len(avail) < 4:
            return weights

        p = prices[avail]

        # --- Signal 1: 12-1 Momentum (UMD) ---
        total_ret = p.pct_change(c.mom_total_lookback).fillna(0.0)
        recent_ret = p.pct_change(c.mom_skip_recent).fillna(0.0)
        umd_raw = total_ret - recent_ret
        umd_rank = umd_raw.rank(axis=1, pct=True).fillna(0.5)

        # --- Signal 2: Earnings surprise proxy ---
        daily_ret = p.pct_change().fillna(0.0)
        ret_mean = daily_ret.rolling(
            c.surprise_zscore_window, min_periods=20,
        ).mean()
        ret_std = daily_ret.rolling(
            c.surprise_zscore_window, min_periods=20,
        ).std().replace(0, np.nan).fillna(daily_ret.expanding().std())
        ret_std = ret_std.clip(lower=1e-8)
        daily_zscore = ((daily_ret - ret_mean) / ret_std).fillna(0.0)
        daily_zscore = daily_zscore.replace([np.inf, -np.inf], 0.0)

        pos_surprise = (daily_zscore > c.surprise_threshold).astype(float)
        neg_surprise = (daily_zscore < -c.surprise_threshold).astype(float)
        pos_count = pos_surprise.rolling(c.surprise_window, min_periods=1).sum()
        neg_count = neg_surprise.rolling(c.surprise_window, min_periods=1).sum()

        # Net surprise ratio: (pos - neg) / (pos + neg + 1)
        surprise_score = (pos_count - neg_count) / (pos_count + neg_count + 1.0)
        surprise_score = surprise_score.replace(
            [np.inf, -np.inf], np.nan,
        ).fillna(0.0)
        surprise_rank = surprise_score.rank(axis=1, pct=True).fillna(0.5)

        # --- Signal 3: Vol-adjusted momentum ---
        vol_df = pd.DataFrame(
            {t: realized_vol(p[t], c.vol_window) for t in avail},
            index=p.index,
        ).fillna(0.0)
        vol_df = vol_df.clip(lower=1e-8)
        mom_3m = p.pct_change(63).fillna(0.0)
        vol_adj_mom = (mom_3m / vol_df).replace(
            [np.inf, -np.inf], np.nan,
        ).fillna(0.0)
        vol_adj_rank = vol_adj_mom.rank(axis=1, pct=True).fillna(0.5)

        # --- Optional: Insider sentiment from edgartools cache ---
        insider_rank = self._load_insider_signal(p.index, avail, c)

        # --- Composite score ---
        if insider_rank is not None:
            composite = (
                0.45 * umd_rank
                + 0.25 * surprise_rank
                + 0.10 * vol_adj_rank
                + 0.20 * insider_rank
            )
        else:
            composite = (
                c.umd_weight * umd_rank
                + c.surprise_weight * surprise_rank
                + c.vol_adj_weight * vol_adj_rank
            )

        # --- Short-term reversal penalty ---
        recent_1m = p.pct_change(c.reversal_lookback).fillna(0.0)
        reversal_mask = recent_1m < 0
        composite = composite - c.reversal_penalty * reversal_mask.astype(float)

        # --- Top quartile selection with monthly rebalance ---
        n_avail = len(avail)
        top_n = max(1, int(n_avail * c.top_pct))
        per_asset_weight = c.leverage / top_n

        ranks = composite.rank(axis=1, ascending=False)
        is_top = ranks <= top_n

        # Apply rebalance frequency
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        held_signal = (
            is_top.where(rebal_mask, np.nan)
            .ffill()
            .fillna(False)
            .astype(bool)
        )

        for t in avail:
            weights.loc[held_signal[t], t] = per_asset_weight

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _load_insider_signal(
        index: pd.DatetimeIndex,
        tickers: list[str],
        cfg: FundamentalMomentumConfig,
    ) -> pd.DataFrame | None:
        """Load cached insider sentiment scores if available.

        Returns cross-sectional rank DataFrame, or None if unavailable.
        The cache is populated by a separate script using edgartools.
        """
        if cfg.insider_cache_path is None:
            return None
        try:
            import os

            cache_file = os.path.join(
                cfg.insider_cache_path, "insider_scores.parquet",
            )
            if not os.path.exists(cache_file):
                return None
            cached = pd.read_parquet(cache_file)
            common_cols = [t for t in tickers if t in cached.columns]
            if len(common_cols) < 2:
                return None
            aligned = cached[common_cols].reindex(index).ffill().fillna(0.0)
            return aligned.rank(axis=1, pct=True).fillna(0.5)
        except Exception:
            return None
