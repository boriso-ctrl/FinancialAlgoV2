"""Category P (cont.): Deep Learning Strategies.

DL-1 -- TemporalCNNAlpha: 1-D CNN on rolling windows of daily features.
DL-2 -- LSTMRegimeDetector: LSTM sequence model for crisis early-warning.
DL-3 -- AttentionCrossSectionalRanker: self-attention across assets.

All three strategies:
  - Inherit from Strategy (base.py)
  - Use walk-forward expanding-window training (no look-ahead bias)
  - Fall back to simple rules during warm-up periods
  - Run on CPU-only PyTorch (small models, fast training)
  - Are NaN-safe and vectorised where possible
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from financial_algo.strategies.base import Strategy
from financial_algo.strategies._shared_features import (
    momentum_df as _momentum_df,
    realized_vol_df as _realized_vol_df,
    rsi_df as _rsi_df,
    zscore_df as _zscore_df,
    above_sma as _above_sma,
)


# =====================================================================
# DL-specific helper
# =====================================================================

def _safe_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert numpy array to float32 tensor, replacing NaN/inf."""
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return torch.from_numpy(arr)


# =====================================================================
# DL-1: Temporal CNN Alpha
# =====================================================================

class _TemporalCNNNet(nn.Module):
    """Small 1-D CNN: (batch, n_features, window) -> (batch, 1)."""

    def __init__(self, n_features: int, window: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (batch, 8, 1)
        )
        self.fc = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, window)
        h = self.conv(x).squeeze(-1)   # (batch, 8)
        return self.fc(h).squeeze(-1)  # (batch,)


_DL1_FEATURE_NAMES = [
    "ret_1m", "ret_3m", "ret_12m", "rvol_20d",
    "rsi_14", "zscore_60d", "above_200sma",
]
_N_DL1_FEATURES = len(_DL1_FEATURE_NAMES)


@dataclass
class TemporalCNNConfig:
    """Config for DL-1 TemporalCNNAlpha."""

    tickers: tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "GLD", "TLT", "XLE",
        "EFA", "EEM", "SLV", "UUP", "DBC", "VNQ",
    )
    window: int = 60             # lookback window for CNN input
    min_train_days: int = 504    # 2 years warm-up
    retrain_freq: int = 63       # quarterly retrain
    fwd_horizon: int = 21        # predict 21-day forward return
    epochs: int = 30             # training epochs per retrain
    lr: float = 0.005
    batch_size: int = 64
    top_n: int = 4
    leverage: float = 1.5
    max_weight: float = 0.50
    rebalance_freq: int = 21     # monthly rebalance


class TemporalCNNAlpha(Strategy):
    """1-D CNN trained on rolling windows of daily features.

    Thesis: Short-term price dynamics (momentum shifts, vol clustering,
    mean-reversion patterns) create temporal signatures in daily feature
    space that a CNN can detect more effectively than static linear
    combinations.  Walk-forward training ensures no look-ahead.

    Architecture: 2-layer 1D Conv with BatchNorm + AdaptiveAvgPool ->
    small FC head -> scalar alpha score per asset.
    """

    name = "DL1-TemporalCNNAlpha"

    def __init__(self, config: TemporalCNNConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or TemporalCNNConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(avail) < c.top_n:
            return weights

        p = prices[avail]
        n_days = len(p)
        n_assets = len(avail)

        # --- Compute features (vectorised) ---
        feat_dict = {
            "ret_1m": _momentum_df(p, 21),
            "ret_3m": _momentum_df(p, 63),
            "ret_12m": _momentum_df(p, 252),
            "rvol_20d": _realized_vol_df(p, 20),
            "rsi_14": _rsi_df(p, 14) / 100.0,      # normalise to [0, 1]
            "zscore_60d": _zscore_df(p, 60),
            "above_200sma": _above_sma(p, 200),
        }

        # Stack into 3D array: (n_days, n_assets, n_features)
        feat_3d = np.stack(
            [feat_dict[f][avail].values for f in _DL1_FEATURE_NAMES], axis=-1,
        ).astype(np.float64)
        feat_3d = np.nan_to_num(feat_3d, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward return target (only used for training on past data)
        fwd_ret = (p.shift(-c.fwd_horizon) / p - 1.0)
        fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        fwd_ret = np.nan_to_num(fwd_ret, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Walk-forward training + prediction ---
        model: _TemporalCNNNet | None = None
        next_retrain = c.min_train_days

        rebal_days = list(range(c.min_train_days, n_days, c.rebalance_freq))
        current_alloc: dict[str, float] = {}

        for day_idx in range(c.min_train_days, n_days):
            # Retrain?
            if day_idx >= next_retrain:
                train_end = day_idx - c.fwd_horizon
                if train_end > c.window + 50:
                    model = self._train_model(
                        feat_3d, fwd_ret, train_end, n_assets, c,
                    )
                next_retrain = day_idx + c.retrain_freq

            # Rebalance?
            if day_idx in rebal_days and model is not None:
                current_alloc = self._predict_alloc(
                    feat_3d, day_idx, avail, model, c,
                )

            # Assign weights
            for t, w in current_alloc.items():
                weights.iloc[day_idx, weights.columns.get_loc(t)] = w

        # Intraday overlay (optional)
        if self._intraday_features is not None:
            weights = self._apply_intraday_overlay(weights)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _train_model(
        feat_3d: np.ndarray,
        fwd_ret: np.ndarray,
        train_end: int,
        n_assets: int,
        c: TemporalCNNConfig,
    ) -> _TemporalCNNNet:
        """Train CNN on expanding window of past data."""
        window = c.window
        n_features = _N_DL1_FEATURES

        # Build training samples: sliding windows ending at each day
        X_list, y_list = [], []
        # Sample every 5 days to speed up training and reduce overlap
        for d in range(window, train_end, 5):
            for a in range(n_assets):
                # (n_features, window) — channels-first for Conv1d
                snippet = feat_3d[d - window:d, a, :].T
                target = fwd_ret[d, a]
                X_list.append(snippet)
                y_list.append(target)

        if len(X_list) < 50:
            # Not enough samples; return a freshly initialised model
            return _TemporalCNNNet(n_features, window)

        X_train = _safe_tensor(np.array(X_list))
        y_train = _safe_tensor(np.array(y_list))

        model = _TemporalCNNNet(n_features, window)
        optimiser = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=1e-4)
        loss_fn = nn.MSELoss()

        model.train()
        n_samples = len(X_train)
        for _ in range(c.epochs):
            # Shuffle
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, c.batch_size):
                batch_idx = perm[start:start + c.batch_size]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]

                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        model.eval()
        return model

    @staticmethod
    def _predict_alloc(
        feat_3d: np.ndarray,
        day_idx: int,
        avail: list[str],
        model: _TemporalCNNNet,
        c: TemporalCNNConfig,
    ) -> dict[str, float]:
        """Predict alpha scores for all assets and allocate top-N."""
        window = c.window
        if day_idx < window:
            return {}

        n_assets = len(avail)
        X_batch = []
        for a in range(n_assets):
            snippet = feat_3d[day_idx - window:day_idx, a, :].T
            X_batch.append(snippet)

        X_t = _safe_tensor(np.array(X_batch))
        with torch.no_grad():
            scores = model(X_t).numpy()

        scored = pd.Series(scores, index=avail)
        top_n = min(c.top_n, n_assets)
        top_assets = scored.nlargest(top_n).index
        per_w = min(c.leverage / top_n, c.max_weight)

        return {t: per_w for t in top_assets}

    def _apply_intraday_overlay(self, weights: pd.DataFrame) -> pd.DataFrame:
        """Scale weights by intraday vol-regime (same pattern as P4)."""
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


# =====================================================================
# DL-2: LSTM Regime Detector
# =====================================================================

class _LSTMRegimeNet(nn.Module):
    """Small LSTM: (batch, seq_len, n_features) -> (batch, n_regimes)."""

    def __init__(self, n_features: int, hidden_size: int = 16,
                 n_regimes: int = 3) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 8),
            nn.ReLU(),
            nn.Linear(8, n_regimes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, n_features)
        _, (h_n, _) = self.lstm(x)    # h_n: (1, batch, hidden)
        h = h_n.squeeze(0)            # (batch, hidden)
        return self.fc(h)             # (batch, n_regimes)


# Regime labels: 0 = risk-on (calm), 1 = elevated, 2 = crisis
_REGIME_RISK_ON = 0
_REGIME_ELEVATED = 1
_REGIME_CRISIS = 2


@dataclass
class LSTMRegimeConfig:
    """Config for DL-2 LSTMRegimeDetector."""

    # Market-wide feature tickers
    spy_ticker: str = "SPY"
    # Sequence length (trading days)
    seq_len: int = 30
    min_train_days: int = 504
    retrain_freq: int = 63
    epochs: int = 40
    lr: float = 0.003
    batch_size: int = 64

    # Regime thresholds (SPY drawdown from rolling max)
    # Used to label training data automatically
    crisis_dd_threshold: float = -0.15   # > 15% drawdown = crisis
    elevated_dd_threshold: float = -0.05  # > 5% drawdown = elevated

    # Position mapping
    risk_on_tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM", "EEM")
    defensive_tickers: tuple[str, ...] = ("GLD", "TLT", "UUP", "IEF")
    risk_on_weight: float = 0.25
    defensive_weight: float = 0.25
    leverage: float = 1.5
    rebalance_freq: int = 5    # weekly rebalance
    # Smoothing: require N consecutive days to switch regime
    regime_smooth_days: int = 3


class LSTMRegimeDetector(Strategy):
    """LSTM-based market regime early-warning system.

    Thesis: Regime transitions (calm -> stressed -> crisis) exhibit
    temporal patterns in vol, breadth, and spread data that are
    detectable several days before traditional threshold signals fire.
    An LSTM trained on labelled drawdown regimes can learn these
    early-warning patterns and position defensively before damage occurs.

    Labels are derived from SPY drawdown from rolling 63-day high.
    Walk-forward expanding-window training ensures no look-ahead.
    """

    name = "DL2-LSTMRegimeDetector"

    def __init__(self, config: LSTMRegimeConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or LSTMRegimeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n_days = len(prices)

        if c.spy_ticker not in prices.columns:
            return weights

        # --- Build market-wide features ---
        feat_df = self._build_market_features(prices)
        feat_arr = feat_df.values.astype(np.float64)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0)
        n_feat = feat_arr.shape[1]

        # Adapt warm-up to short validation slices while preserving
        # enough history for sequence construction and model fitting.
        effective_min_train = min(
            c.min_train_days,
            max(c.seq_len + 60, int(0.40 * n_days)),
        )

        # --- Build regime labels from SPY drawdown ---
        spy = prices[c.spy_ticker]
        spy_max = spy.rolling(63, min_periods=1).max()
        spy_dd = (spy / spy_max.replace(0, np.nan) - 1.0).fillna(0.0)

        labels = np.full(n_days, _REGIME_RISK_ON, dtype=np.int64)
        labels[spy_dd.values < c.elevated_dd_threshold] = _REGIME_ELEVATED
        labels[spy_dd.values < c.crisis_dd_threshold] = _REGIME_CRISIS

        # --- Walk-forward loop ---
        model: _LSTMRegimeNet | None = None
        next_retrain = effective_min_train
        current_regime = _REGIME_RISK_ON
        regime_counter = 0

        risk_on_avail = [t for t in c.risk_on_tickers if t in prices.columns]
        defensive_avail = [t for t in c.defensive_tickers if t in prices.columns]

        rebal_set = set(range(effective_min_train, n_days, c.rebalance_freq))

        for day_idx in range(effective_min_train, n_days):
            # Retrain?
            if day_idx >= next_retrain:
                train_end = day_idx
                if train_end > c.seq_len + 50:
                    model = self._train_model(
                        feat_arr, labels, train_end, n_feat, c,
                    )
                next_retrain = day_idx + c.retrain_freq

            # Predict regime
            if model is not None and day_idx >= c.seq_len:
                predicted = self._predict_regime(
                    feat_arr, day_idx, model, c,
                )
                # Smoothing: only switch after N consecutive days
                if predicted != current_regime:
                    regime_counter += 1
                    if regime_counter >= c.regime_smooth_days:
                        current_regime = predicted
                        regime_counter = 0
                else:
                    regime_counter = 0

            # Assign weights on rebalance days
            if day_idx in rebal_set:
                alloc = self._regime_allocation(
                    current_regime, risk_on_avail, defensive_avail, c,
                )
                # Forward-fill until next rebalance
                next_rebal = day_idx + c.rebalance_freq
                end_fill = min(next_rebal, n_days)
                for t, w in alloc.items():
                    col_idx = weights.columns.get_loc(t)
                    weights.iloc[day_idx:end_fill, col_idx] = w

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _build_market_features(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Build market-wide feature DataFrame for LSTM input."""
        c = self.cfg
        spy = prices[c.spy_ticker] if c.spy_ticker in prices.columns else prices.iloc[:, 0]

        # Feature 1: SPY 20d realized vol
        spy_vol = _realized_vol_df(spy.to_frame(), 20).iloc[:, 0]

        # Feature 2: SPY 5d return
        spy_ret5 = spy.pct_change(5).fillna(0.0)

        # Feature 3: SPY 20d return
        spy_ret20 = spy.pct_change(20).fillna(0.0)

        # Feature 4: SPY RSI / 100
        spy_rsi = _rsi_df(spy.to_frame(), 14).iloc[:, 0] / 100.0

        # Feature 5: Credit spread z-score (HYG/LQD if available)
        if "HYG" in prices.columns and "LQD" in prices.columns:
            credit_ratio = prices["HYG"] / prices["LQD"].replace(0, np.nan)
            credit_z = _zscore_df(credit_ratio.to_frame(), 60).iloc[:, 0]
        else:
            credit_z = pd.Series(0.0, index=prices.index)

        # Feature 6: SPY drawdown from 63-day high
        spy_max = spy.rolling(63, min_periods=1).max()
        spy_dd = (spy / spy_max.replace(0, np.nan) - 1.0).fillna(0.0)

        # Feature 7: Vol-of-vol (5-day rolling std of daily vol)
        daily_ret = spy.pct_change().fillna(0.0).abs()
        vov = daily_ret.rolling(20).std().fillna(0.0) * np.sqrt(252)

        # Feature 8: Breadth proxy (fraction of tickers above 50-day SMA)
        all_sma50 = prices.rolling(50, min_periods=25).mean()
        breadth = (prices >= all_sma50).mean(axis=1).fillna(0.5)

        out = pd.DataFrame({
            "spy_vol": spy_vol,
            "spy_ret5": spy_ret5,
            "spy_ret20": spy_ret20,
            "spy_rsi": spy_rsi,
            "credit_z": credit_z,
            "spy_dd": spy_dd,
            "vov": vov,
            "breadth": breadth,
        }, index=prices.index)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _train_model(
        feat_arr: np.ndarray,
        labels: np.ndarray,
        train_end: int,
        n_feat: int,
        c: LSTMRegimeConfig,
    ) -> _LSTMRegimeNet:
        """Train LSTM on past sequences with auto-labelled regimes."""
        seq_len = c.seq_len

        X_list, y_list = [], []
        # Sample every 3 days to speed up
        for d in range(seq_len, train_end, 3):
            seq = feat_arr[d - seq_len:d]   # (seq_len, n_feat)
            label = labels[d]
            X_list.append(seq)
            y_list.append(label)

        if len(X_list) < 30:
            return _LSTMRegimeNet(n_feat)

        X_train = _safe_tensor(np.array(X_list))
        y_train = torch.LongTensor(np.array(y_list))

        model = _LSTMRegimeNet(n_feat)
        optimiser = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        n_samples = len(X_train)
        for _ in range(c.epochs):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, c.batch_size):
                batch_idx = perm[start:start + c.batch_size]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]

                logits = model(xb)
                loss = loss_fn(logits, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        model.eval()
        return model

    @staticmethod
    def _predict_regime(
        feat_arr: np.ndarray,
        day_idx: int,
        model: _LSTMRegimeNet,
        c: LSTMRegimeConfig,
    ) -> int:
        """Predict regime for a single day."""
        seq = feat_arr[day_idx - c.seq_len:day_idx]
        X_t = _safe_tensor(seq[np.newaxis, ...])  # (1, seq_len, n_feat)
        with torch.no_grad():
            logits = model(X_t)
            probs = torch.softmax(logits, dim=-1).numpy()[0]
        return int(np.argmax(probs))

    @staticmethod
    def _regime_allocation(
        regime: int,
        risk_on: list[str],
        defensive: list[str],
        c: LSTMRegimeConfig,
    ) -> dict[str, float]:
        """Map regime to asset allocation."""
        alloc: dict[str, float] = {}

        if regime == _REGIME_RISK_ON:
            # Full risk-on: long equities
            n_on = max(len(risk_on), 1)
            per_w = c.leverage / n_on
            for t in risk_on:
                alloc[t] = min(per_w, c.risk_on_weight * 2)
        elif regime == _REGIME_ELEVATED:
            # Balanced: half risk-on, half defensive
            n_on = max(len(risk_on), 1)
            n_def = max(len(defensive), 1)
            for t in risk_on:
                alloc[t] = 0.5 * c.leverage / n_on
            for t in defensive:
                alloc[t] = 0.5 * c.leverage / n_def
        else:
            # Crisis: full defensive
            n_def = max(len(defensive), 1)
            per_w = c.leverage / n_def
            for t in defensive:
                alloc[t] = min(per_w, c.defensive_weight * 2)

        return alloc


# =====================================================================
# DL-3: Attention Cross-Sectional Ranker
# =====================================================================

class _CrossAssetAttentionNet(nn.Module):
    """Self-attention across assets: (batch, n_assets, n_features) -> (batch, n_assets)."""

    def __init__(self, n_features: int, d_model: int = 16,
                 n_heads: int = 2) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_assets, n_features)
        h = self.input_proj(x)             # (batch, n_assets, d_model)
        attn_out, _ = self.attention(h, h, h)  # self-attention
        h = self.norm(h + attn_out)        # residual + LayerNorm
        return self.fc(h).squeeze(-1)      # (batch, n_assets)


_DL3_FEATURE_NAMES = [
    "ret_1m", "ret_3m", "ret_12m", "rvol_20d",
    "rsi_14", "zscore_60d", "above_200sma", "vol_of_vol",
]
_N_DL3_FEATURES = len(_DL3_FEATURE_NAMES)


@dataclass
class AttentionRankerConfig:
    """Config for DL-3 AttentionCrossSectionalRanker."""

    tickers: tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "SLV",
        "TLT", "IEF", "UUP", "XLE", "DBC", "VNQ",
        "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    )
    min_train_days: int = 504
    retrain_freq: int = 63
    fwd_horizon: int = 21
    epochs: int = 30
    lr: float = 0.003
    batch_size: int = 64
    long_n: int = 5
    short_n: int = 3
    long_weight: float = 1.2      # total long exposure
    short_weight: float = 0.4     # total short exposure
    max_position: float = 0.30
    rebalance_freq: int = 21
    training_stride: int = 5      # sample every 5 days to reduce overlap


class AttentionCrossSectionalRanker(Strategy):
    """Self-attention model for cross-sectional asset ranking.

    Thesis: Cross-asset relationships (e.g., sector rotation, risk-on/off
    flows, commodity-equity linkages) are dynamic and complex.
    Multi-head self-attention can learn these inter-asset dependencies
    without hard-coding correlation assumptions, producing better
    cross-sectional return predictions than independent-asset models.

    Walk-forward expanding-window training.  Longs top-ranked assets,
    shorts bottom-ranked.
    """

    name = "DL3-AttentionRanker"

    def __init__(self, config: AttentionRankerConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or AttentionRankerConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(avail) < c.long_n + c.short_n:
            return weights

        p = prices[avail]
        n_days = len(p)
        n_assets = len(avail)

        # --- Compute features ---
        rvol_20d = _realized_vol_df(p, 20)
        rvol_60d = _realized_vol_df(p, 60).replace(0.0, np.nan)
        vol_of_vol = (rvol_20d / rvol_60d).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)

        feat_dict = {
            "ret_1m": _momentum_df(p, 21),
            "ret_3m": _momentum_df(p, 63),
            "ret_12m": _momentum_df(p, 252),
            "rvol_20d": rvol_20d,
            "rsi_14": _rsi_df(p, 14) / 100.0,
            "zscore_60d": _zscore_df(p, 60),
            "above_200sma": _above_sma(p, 200),
            "vol_of_vol": vol_of_vol,
        }

        # Stack: (n_days, n_assets, n_features)
        feat_3d = np.stack(
            [feat_dict[f][avail].values for f in _DL3_FEATURE_NAMES], axis=-1,
        ).astype(np.float64)
        feat_3d = np.nan_to_num(feat_3d, nan=0.0, posinf=0.0, neginf=0.0)

        # Forward returns target
        fwd_ret = (p.shift(-c.fwd_horizon) / p - 1.0)
        fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        fwd_ret = np.nan_to_num(fwd_ret, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Walk-forward ---
        model: _CrossAssetAttentionNet | None = None
        next_retrain = c.min_train_days
        current_long: list[str] = []
        current_short: list[str] = []

        rebal_set = set(range(c.min_train_days, n_days, c.rebalance_freq))

        for day_idx in range(c.min_train_days, n_days):
            # Retrain?
            if day_idx >= next_retrain:
                train_end = day_idx - c.fwd_horizon
                if train_end > 100:
                    model = self._train_model(
                        feat_3d, fwd_ret, train_end, n_assets, c,
                    )
                next_retrain = day_idx + c.retrain_freq

            # Rebalance?
            if day_idx in rebal_set and model is not None:
                current_long, current_short = self._predict_ranks(
                    feat_3d, day_idx, avail, model, c,
                )

            # Assign weights
            n_long = max(len(current_long), 1)
            n_short = max(len(current_short), 1)
            per_long = min(c.long_weight / n_long, c.max_position)
            per_short = min(c.short_weight / n_short, c.max_position)

            for t in current_long:
                weights.iloc[day_idx, weights.columns.get_loc(t)] = per_long
            for t in current_short:
                weights.iloc[day_idx, weights.columns.get_loc(t)] = -per_short

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    @staticmethod
    def _train_model(
        feat_3d: np.ndarray,
        fwd_ret: np.ndarray,
        train_end: int,
        n_assets: int,
        c: AttentionRankerConfig,
    ) -> _CrossAssetAttentionNet:
        """Train attention model on expanding window."""
        n_features = _N_DL3_FEATURES

        # Build samples: each sample is one day's cross-section
        sample_days = list(range(0, train_end, c.training_stride))
        if len(sample_days) < 20:
            return _CrossAssetAttentionNet(n_features)

        # X: (n_samples, n_assets, n_features)
        # y: (n_samples, n_assets) - cross-sectional target returns
        X_arr = feat_3d[sample_days]
        y_arr = fwd_ret[sample_days]

        X_train = _safe_tensor(X_arr)
        y_train = _safe_tensor(y_arr)

        model = _CrossAssetAttentionNet(n_features)
        optimiser = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=1e-4)
        # Use MSE on cross-sectional return predictions
        loss_fn = nn.MSELoss()

        model.train()
        n_samples = len(X_train)
        for _ in range(c.epochs):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, c.batch_size):
                batch_idx = perm[start:start + c.batch_size]
                xb = X_train[batch_idx]
                yb = y_train[batch_idx]

                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

        model.eval()
        return model

    @staticmethod
    def _predict_ranks(
        feat_3d: np.ndarray,
        day_idx: int,
        avail: list[str],
        model: _CrossAssetAttentionNet,
        c: AttentionRankerConfig,
    ) -> tuple[list[str], list[str]]:
        """Predict scores and return top-long and bottom-short lists."""
        X_t = _safe_tensor(feat_3d[day_idx:day_idx + 1])  # (1, n_assets, n_feat)
        with torch.no_grad():
            scores = model(X_t).numpy()[0]  # (n_assets,)

        scored = pd.Series(scores, index=avail)
        ranked = scored.rank(ascending=False)

        long_n = min(c.long_n, len(avail) // 2)
        short_n = min(c.short_n, len(avail) // 3)

        top = ranked[ranked <= long_n].index.tolist()
        bottom = ranked[ranked > len(avail) - short_n].index.tolist()

        return top, bottom
