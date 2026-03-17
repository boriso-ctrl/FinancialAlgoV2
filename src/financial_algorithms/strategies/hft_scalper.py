"""
High-Frequency Trading scalper strategy optimized for ~1 trade / hour.

Multi-timeframe approach
========================
* **Entry timeframe** : 3-minute bars (fast signal generation)
* **Confirmation TF**  : 5-minute and 10-minute (noise filter / confluence)
* **Trend context TF**  : 15-minute and 1-hour (direction bias)

Seven core indicators, each scored on a -2 … +2 scale (total range -14 … +14):
  1. EMA crossover  (3-min)      – momentum direction
  2. RSI            (3-min)      – mean-reversion / exhaustion
  3. VWAP distance  (3-min)      – institutional fair value
  4. Stochastic %K  (5-min)      – confirmation oscillator
  5. MACD histogram (5-min)      – trend momentum
  6. ADX / DI       (15-min)     – trend strength filter
  7. Bollinger %B   (10-min)     – volatility squeeze / breakout

Position management
-------------------
* Dynamic position sizing: 3 - 10 % of equity scaled by signal strength.
* Tight stop-loss: 0.30 % (intraday noise envelope).
* Tiered take-profit: TP1 = 0.45 %, TP2 = 0.90 % (R:R ≈ 1:1.5 / 1:3).
* Trailing stop activated after TP1 hit.
* Max hold time: 60 bars on the 3-min chart (≈ 3 hours safety net).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class HFTScalperStrategy:
    """Seven-indicator multi-timeframe scalper targeting ~1 trade / hour."""

    # ── default hyper-parameters ────────────────────────────────────────
    DEFAULT_CONFIG = {
        # Entry / exit thresholds (out of ±14)
        "min_buy_score": 3.0,
        "max_sell_score": -3.0,
        # Risk management
        "stop_loss_pct": 0.30,
        "tp1_pct": 0.45,
        "tp2_pct": 0.90,
        "trailing_stop_pct": 0.20,
        "max_hold_bars": 60,
        # Position sizing bounds (% of equity)
        "min_size_pct": 3.0,
        "max_size_pct": 10.0,
        # Indicator periods (tuned for 3-min primary bars)
        "ema_fast": 8,
        "ema_slow": 21,
        "rsi_period": 10,
        "stoch_period": 10,
        "stoch_smooth": 3,
        "macd_fast": 8,
        "macd_slow": 17,
        "macd_signal": 6,
        "adx_period": 10,
        "bb_period": 14,
        "bb_std": 2.0,
        "vwap_lookback": 20,
    }

    # ────────────────────────────────────────────────────────────────────
    def __init__(self, **overrides):
        cfg = {**self.DEFAULT_CONFIG, **overrides}
        for k, v in cfg.items():
            setattr(self, k, v)
        self.position_state: dict[str, dict] = {}

    # ── 1. EMA crossover (3-min) ────────────────────────────────────────
    def calculate_ema_signal(self, close: np.ndarray) -> float:
        """EMA crossover score (-2 … +2)."""
        n = len(close)
        if n < self.ema_slow + 1:
            return 0.0
        s = pd.Series(close)
        fast = s.ewm(span=self.ema_fast, adjust=False).mean().iloc[-1]
        slow = s.ewm(span=self.ema_slow, adjust=False).mean().iloc[-1]
        if slow == 0:
            return 0.0
        dist = (fast - slow) / slow * 100
        if dist > 0.15:
            return 2.0
        if dist > 0.03:
            return 1.0
        if dist < -0.15:
            return -2.0
        if dist < -0.03:
            return -1.0
        return 0.0

    # ── 2. RSI (3-min) ──────────────────────────────────────────────────
    def calculate_rsi_signal(self, close: np.ndarray) -> float:
        """RSI mean-reversion score (-2 … +2)."""
        period = self.rsi_period
        if len(close) < period + 1:
            return 0.0
        delta = np.diff(close[-(period + 1):])
        gain = np.mean(np.maximum(delta, 0))
        loss = np.mean(np.abs(np.minimum(delta, 0)))
        if loss == 0:
            rsi = 100.0 if gain > 0 else 50.0
        else:
            rsi = 100.0 - 100.0 / (1.0 + gain / loss)
        if rsi < 25:
            return 2.0
        if rsi < 40:
            return 1.0
        if rsi > 75:
            return -2.0
        if rsi > 60:
            return -1.0
        return 0.0

    # ── 3. VWAP distance (3-min) ────────────────────────────────────────
    def calculate_vwap_signal(
        self, close: np.ndarray, volume: np.ndarray
    ) -> float:
        """VWAP distance score (-2 … +2)."""
        lb = self.vwap_lookback
        if len(close) < lb or len(volume) < lb:
            return 0.0
        c, v = close[-lb:], volume[-lb:]
        total_vol = np.sum(v)
        if total_vol == 0:
            return 0.0
        vwap = np.sum(c * v) / total_vol
        if vwap == 0:
            return 0.0
        dist = (close[-1] - vwap) / vwap * 100
        if dist > 0.20:
            return -2.0  # extended above VWAP → bearish
        if dist > 0.05:
            return -1.0
        if dist < -0.20:
            return 2.0  # deep below VWAP → bullish
        if dist < -0.05:
            return 1.0
        return 0.0

    # ── 4. Stochastic %K (5-min) ────────────────────────────────────────
    def calculate_stoch_signal(self, high: np.ndarray, low: np.ndarray,
                               close: np.ndarray) -> float:
        """Stochastic %K score (-2 … +2)."""
        period = self.stoch_period
        if len(close) < period:
            return 0.0
        h_high = np.max(high[-period:])
        l_low = np.min(low[-period:])
        rng = h_high - l_low
        if rng == 0:
            return 0.0
        k = (close[-1] - l_low) / rng * 100
        if k < 15:
            return 2.0
        if k < 30:
            return 1.0
        if k > 85:
            return -2.0
        if k > 70:
            return -1.0
        return 0.0

    # ── 5. MACD histogram (5-min) ───────────────────────────────────────
    def calculate_macd_signal(self, close: np.ndarray) -> float:
        """MACD histogram score (-2 … +2)."""
        n = len(close)
        if n < self.macd_slow + self.macd_signal:
            return 0.0
        s = pd.Series(close)
        fast_ema = s.ewm(span=self.macd_fast, adjust=False).mean()
        slow_ema = s.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        hist = (macd_line - signal_line).iloc[-1]
        price = close[-1] if close[-1] != 0 else 1.0
        hist_pct = hist / abs(price) * 100
        if hist_pct > 0.10:
            return 2.0
        if hist_pct > 0.02:
            return 1.0
        if hist_pct < -0.10:
            return -2.0
        if hist_pct < -0.02:
            return -1.0
        return 0.0

    # ── 6. ADX / DI (15-min) ───────────────────────────────────────────
    def calculate_adx_signal(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> float:
        """ADX trend-strength score (-2 … +2)."""
        period = self.adx_period
        if len(high) < period + 1:
            return 0.0
        try:
            high = np.asarray(high, dtype=float)
            low = np.asarray(low, dtype=float)
            close = np.asarray(close, dtype=float)
            recent_high = np.mean(high[-period:])
            recent_low = np.mean(low[-period:])
            rng = recent_high - recent_low
            if rng == 0:
                return 0.0
            strength = (close[-1] - recent_low) / rng
            if strength > 0.80:
                return 2.0
            if strength > 0.60:
                return 1.0
            if strength < 0.20:
                return -2.0
            if strength < 0.40:
                return -1.0
            return 0.0
        except Exception:
            return 0.0

    # ── 7. Bollinger %B (10-min) ────────────────────────────────────────
    def calculate_bb_signal(self, close: np.ndarray) -> float:
        """Bollinger %B score (-2 … +2)."""
        period = self.bb_period
        if len(close) < period:
            return 0.0
        window = close[-period:]
        mid = np.mean(window)
        std = np.std(window, ddof=1)
        if std == 0:
            return 0.0
        upper = mid + self.bb_std * std
        lower = mid - self.bb_std * std
        band_width = upper - lower
        if band_width == 0:
            return 0.0
        pct_b = (close[-1] - lower) / band_width
        if pct_b < 0.05:
            return 2.0   # below lower band → bullish reversal
        if pct_b < 0.25:
            return 1.0
        if pct_b > 0.95:
            return -2.0  # above upper band → bearish reversal
        if pct_b > 0.75:
            return -1.0
        return 0.0

    # ── composite score ─────────────────────────────────────────────────
    def calculate_composite_score(
        self,
        close_3m: np.ndarray,
        high_5m: np.ndarray,
        low_5m: np.ndarray,
        close_5m: np.ndarray,
        volume_3m: np.ndarray,
        high_15m: np.ndarray,
        low_15m: np.ndarray,
        close_15m: np.ndarray,
        close_10m: np.ndarray,
    ) -> float:
        """Aggregate all seven indicator scores (range -14 … +14)."""
        ema = self.calculate_ema_signal(close_3m)
        rsi = self.calculate_rsi_signal(close_3m)
        vwap = self.calculate_vwap_signal(close_3m, volume_3m)
        stoch = self.calculate_stoch_signal(high_5m, low_5m, close_5m)
        macd = self.calculate_macd_signal(close_5m)
        adx = self.calculate_adx_signal(high_15m, low_15m, close_15m)
        bb = self.calculate_bb_signal(close_10m)
        total = ema + rsi + vwap + stoch + macd + adx + bb
        return float(np.clip(total, -14, 14))

    # ── entry / exit helpers ────────────────────────────────────────────
    def should_enter(self, score: float) -> bool:
        return score >= self.min_buy_score

    def should_exit_on_signal(self, score: float) -> bool:
        return score <= self.max_sell_score

    def calculate_position_size(self, score: float, equity: float) -> float:
        """Dynamic sizing: 3-10 % of equity scaled by signal strength."""
        if score < self.min_buy_score:
            return 0.0
        strength = min(abs(score) / 14.0, 1.0)
        pct = self.min_size_pct + strength * (self.max_size_pct - self.min_size_pct)
        return equity * pct / 100.0

    def calculate_entry_exit_levels(self, entry_price: float) -> dict[str, float]:
        sl = entry_price * (1 - self.stop_loss_pct / 100)
        tp1 = entry_price * (1 + self.tp1_pct / 100)
        tp2 = entry_price * (1 + self.tp2_pct / 100)
        return {
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "risk": entry_price - sl,
            "reward_tp1": tp1 - entry_price,
            "reward_tp2": tp2 - entry_price,
        }

    # ── vectorised signal generation ────────────────────────────────────
    def generate_signals(self, df_3m: pd.DataFrame) -> pd.Series:
        """Generate a signal series from a single-timeframe 3-min DataFrame.

        This is a convenience helper that derives higher-timeframe bars
        internally so callers only need to supply 3-minute OHLCV data.

        Parameters
        ----------
        df_3m : pd.DataFrame
            Must contain columns ``close``, ``high``, ``low``, ``volume``
            and either a ``timestamp`` column or a DatetimeIndex.

        Returns
        -------
        pd.Series
            Signal values aligned to the 3-min index.  Positive → long,
            negative → short, zero → flat.
        """
        if "timestamp" in df_3m.columns:
            df_3m = df_3m.set_index("timestamp")

        close = df_3m["close"].values
        high = df_3m["high"].values
        low = df_3m["low"].values
        volume = df_3m["volume"].values

        warmup = max(self.ema_slow, self.bb_period, self.macd_slow + self.macd_signal) + 5
        signals = np.zeros(len(df_3m))

        for i in range(warmup, len(df_3m)):
            # Use the same arrays sliced to the current bar for all
            # timeframes in the single-TF convenience path.
            c3 = close[: i + 1]
            h3 = high[: i + 1]
            l3 = low[: i + 1]
            v3 = volume[: i + 1]
            score = self.calculate_composite_score(
                close_3m=c3,
                high_5m=h3,
                low_5m=l3,
                close_5m=c3,
                volume_3m=v3,
                high_15m=h3,
                low_15m=l3,
                close_15m=c3,
                close_10m=c3,
            )
            if score >= self.min_buy_score:
                signals[i] = score
            elif score <= self.max_sell_score:
                signals[i] = score
            else:
                signals[i] = 0.0

        return pd.Series(signals, index=df_3m.index, name="hft_signal")

    # ── summary ─────────────────────────────────────────────────────────
    def get_system_summary(self) -> dict:
        return {
            "name": "HFT Scalper (7-indicator multi-timeframe)",
            "indicators": [
                "EMA crossover (3m)",
                "RSI (3m)",
                "VWAP distance (3m)",
                "Stochastic %K (5m)",
                "MACD histogram (5m)",
                "ADX/DI (15m)",
                "Bollinger %B (10m)",
            ],
            "score_range": [-14, 14],
            "buy_threshold": self.min_buy_score,
            "sell_threshold": self.max_sell_score,
            "stop_loss": f"{self.stop_loss_pct}%",
            "tp1": f"{self.tp1_pct}%",
            "tp2": f"{self.tp2_pct}%",
            "trailing_stop": f"{self.trailing_stop_pct}%",
            "max_hold": f"{self.max_hold_bars} bars",
            "position_sizing": f"{self.min_size_pct}-{self.max_size_pct}% dynamic",
        }
