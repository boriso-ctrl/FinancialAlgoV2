"""HFT / Intraday Strategy Research Pack — 11-strategy roadmap.

Phase 1 (complete):
    MMT-1  MACDHistogramAcceleration  KILLED
    MRM-1  VWAPVolNormalizedFade       KILLED
    VEF-1  RealizedVolImpulseFade      PROMOTED (Sharpe +2.20)

Phase 2 (research candidates):
    MMT-2  KSTTSIOpeningBurst
    MMT-3  RelativeVolContinuation
    MRM-2  WickRejectionFade
    MRM-3  BollingerSnapback
    VEF-2  SqueezeReleaseBreakout
    VEF-3  RangeExpansionExhaustion
    HYB-1  VolRegimeRouter
    HYB-2  GapRegimeSelector

All signals are NaN-safe, vectorized, and shift(1) for look-ahead safety.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_algo.strategies.intraday_base import (
    IntradayStrategy,
    _sanitize,
    feat_atr,
    feat_bollinger,
    feat_ema,
    feat_kst,
    feat_macd,
    feat_realized_vol,
    feat_rsi,
    feat_tsi,
    feat_vol_ratio,
    feat_vwap,
)


# ---------------------------------------------------------------------------
# Phase 1 — MMT-1  (KILLED)
# ---------------------------------------------------------------------------


class MACDHistogramAcceleration(IntradayStrategy):
    """MMT-1: MACD histogram acceleration momentum.

    Signal: sign of histogram second-derivative (acceleration).
    Status: KILLED — high turnover bleeds to transaction costs.
    """

    name = "MMT-1-MACDHistogramAcceleration"
    timeframe = "5min"

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        accel_threshold: float = 0.0,
    ) -> None:
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.accel_threshold = accel_threshold

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"].astype(float)
        _, _, hist = feat_macd(close, self.fast, self.slow, self.signal)
        accel = hist.diff().fillna(0.0)
        sig = np.sign(accel - self.accel_threshold).astype(float)
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 1 — MRM-1  (KILLED)
# ---------------------------------------------------------------------------


class VWAPVolNormalizedFade(IntradayStrategy):
    """MRM-1: VWAP deviation fade normalized by realized vol.

    Signal: mean-reversion from VWAP, size scaled by inverse vol.
    Status: KILLED — too much turnover on 1-min bars.
    """

    name = "MRM-1-VWAPVolNormalizedFade"
    timeframe = "1min"

    def __init__(
        self,
        vol_window: int = 10,
        z_cap: float = 2.0,
    ) -> None:
        self.vol_window = vol_window
        self.z_cap = z_cap

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"].astype(float)
        volume = ohlcv["volume"].astype(float).replace(0, np.nan).fillna(1.0)
        vwap = feat_vwap(close, volume)
        rvol = feat_realized_vol(close, self.vol_window).replace(0, np.nan)
        dev = ((close - vwap) / vwap.replace(0, np.nan)).fillna(0.0)
        z = (dev / rvol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        sig = (-z).clip(-self.z_cap, self.z_cap) / self.z_cap
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 1 — VEF-1  (PROMOTED)
# ---------------------------------------------------------------------------


class RealizedVolImpulseFade(IntradayStrategy):
    """VEF-1: Realized vol impulse fade — PROMOTED strategy.

    Signal: when short-term realized vol spikes (ratio > threshold),
    fade the prior bar's return direction expecting mean-reversion.

    Walk-forward gate: OOS Sharpe >= 0.30, positive fold rate >= 60%.
    Best params: shock_threshold=1.8, min_impulse=0.003, rebalance_bars=5.
    """

    name = "VEF-1-RealizedVolImpulseFade"
    timeframe = "1min"

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        shock_threshold: float = 1.8,
        min_impulse: float = 0.003,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.shock_threshold = shock_threshold
        self.min_impulse = min_impulse

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"].astype(float)
        vol_ratio = feat_vol_ratio(close, self.short_window, self.long_window)
        ret = close.pct_change().fillna(0.0)
        short_vol = feat_realized_vol(close, self.short_window)

        impulse_flag = (vol_ratio > self.shock_threshold) & (short_vol > self.min_impulse)
        # Fade the prior bar's direction when impulse detected
        sig = pd.Series(0.0, index=close.index)
        sig = sig.where(~impulse_flag, -np.sign(ret))
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 2 — MMT-2
# ---------------------------------------------------------------------------


class KSTTSIOpeningBurst(IntradayStrategy):
    """MMT-2: KST + TSI combined opening burst momentum.

    Both KST and TSI must agree on direction. Signal strongest at session open.
    Open-sensitive: higher cost bucket due to wider spreads at open.
    """

    name = "MMT-2-KSTTSIOpeningBurst"
    timeframe = "5min"

    def __init__(
        self,
        open_bars: int = 12,
        kst_threshold: float = 0.0,
        tsi_threshold: float = 0.0,
    ) -> None:
        self.open_bars = open_bars
        self.kst_threshold = kst_threshold
        self.tsi_threshold = tsi_threshold

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"].astype(float)
        kst = feat_kst(close)
        tsi = feat_tsi(close)

        # Both indicators must point same direction
        kst_bull = (kst > self.kst_threshold).astype(float)
        tsi_bull = (tsi > self.tsi_threshold).astype(float)
        agree = kst_bull * tsi_bull  # 1 when both bullish
        kst_bear = (kst < -self.kst_threshold).astype(float)
        tsi_bear = (tsi < -self.tsi_threshold).astype(float)
        agree_bear = kst_bear * tsi_bear

        sig = agree - agree_bear

        # Opening burst: boost weight in first ``open_bars`` of each session
        if hasattr(close.index, "normalize"):
            day_bar = ohlcv.groupby(close.index.normalize()).cumcount()
            open_mask = (day_bar < self.open_bars).astype(float)
            sig = sig * (1.0 + open_mask * 0.5)

        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 2 — MMT-3
# ---------------------------------------------------------------------------


class RelativeVolContinuation(IntradayStrategy):
    """MMT-3: Relative volume continuation — momentum with volume confirmation.

    High relative volume in direction of recent return = continuation signal.
    """

    name = "MMT-3-RelativeVolContinuation"
    timeframe = "1min"

    def __init__(
        self,
        rvol_threshold: float = 1.5,
        ret_window: int = 5,
        vol_window: int = 20,
    ) -> None:
        self.rvol_threshold = rvol_threshold
        self.ret_window = ret_window
        self.vol_window = vol_window

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"].astype(float)
        volume = ohlcv["volume"].astype(float).replace(0, np.nan).fillna(1.0)

        avg_vol = volume.rolling(self.vol_window, min_periods=1).mean().replace(0, np.nan)
        rvol = (volume / avg_vol).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        ret = close.pct_change().fillna(0.0)
        mom = ret.rolling(self.ret_window, min_periods=1).mean()
        direction = np.sign(mom)

        # Signal only when volume surge confirms direction
        high_vol = (rvol > self.rvol_threshold).astype(float)
        sig = direction * high_vol
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 2 — MRM-2
# ---------------------------------------------------------------------------


class WickRejectionFade(IntradayStrategy):
    """MRM-2: Wick rejection fade — price rejected at bar extremes.

    When a bar shows a long wick in one direction with a small body,
    fade the wick direction (expect reversal to body center).
    """

    name = "MRM-2-WickRejectionFade"
    timeframe = "1min"

    def __init__(
        self,
        wick_ratio_threshold: float = 0.6,
        atr_window: int = 14,
        min_atr_mult: float = 0.5,
    ) -> None:
        self.wick_ratio_threshold = wick_ratio_threshold
        self.atr_window = atr_window
        self.min_atr_mult = min_atr_mult

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        o = ohlcv["open"].astype(float)
        h = ohlcv["high"].astype(float)
        l = ohlcv["low"].astype(float)
        c = ohlcv["close"].astype(float)

        bar_range = (h - l).replace(0, np.nan)
        atr = feat_atr(h, l, c, self.atr_window).replace(0, np.nan)

        body_top = pd.concat([o, c], axis=1).max(axis=1)
        body_bot = pd.concat([o, c], axis=1).min(axis=1)
        body_size = body_top - body_bot

        upper_wick = h - body_top
        lower_wick = body_bot - l

        upper_wick_ratio = (upper_wick / bar_range).fillna(0.0)
        lower_wick_ratio = (lower_wick / bar_range).fillna(0.0)

        # Large enough move to be meaningful
        significant = (bar_range > self.min_atr_mult * atr).fillna(False)

        # Upper wick rejection -> bearish fade
        upper_reject = (upper_wick_ratio > self.wick_ratio_threshold) & significant
        # Lower wick rejection -> bullish fade
        lower_reject = (lower_wick_ratio > self.wick_ratio_threshold) & significant

        sig = pd.Series(0.0, index=c.index)
        sig = sig.where(~upper_reject, -1.0)
        sig = sig.where(~lower_reject, 1.0)
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 2 — MRM-3
# ---------------------------------------------------------------------------


class BollingerSnapback(IntradayStrategy):
    """MRM-3: Bollinger Band snapback mean reversion.

    When price closes outside the band with RSI extreme, fade back to mid.
    Signal strength proportional to distance from band.
    """

    name = "MRM-3-BollingerSnapback"
    timeframe = "5min"

    def __init__(
        self,
        bb_window: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_ob: float = 70.0,
        rsi_os: float = 30.0,
    ) -> None:
        self.bb_window = bb_window
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_ob = rsi_ob
        self.rsi_os = rsi_os

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"].astype(float)
        upper, mid, lower, _, pct_b = feat_bollinger(close, self.bb_window, self.bb_std)
        rsi = feat_rsi(close, self.rsi_period)

        band_width = (upper - lower).replace(0, np.nan)
        dist_above = ((close - upper) / band_width).fillna(0.0)
        dist_below = ((lower - close) / band_width).fillna(0.0)

        # Overbought outside upper band -> short fade
        short_cond = (close > upper) & (rsi > self.rsi_ob)
        # Oversold outside lower band -> long fade
        long_cond = (close < lower) & (rsi < self.rsi_os)

        sig = pd.Series(0.0, index=close.index)
        sig = sig.where(~short_cond, -dist_above.clip(0.0, 1.0))
        sig = sig.where(~long_cond, dist_below.clip(0.0, 1.0))
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 2 — VEF-2
# ---------------------------------------------------------------------------


class SqueezeReleaseBreakout(IntradayStrategy):
    """VEF-2: Bollinger Band squeeze release breakout.

    When BB bandwidth compresses below threshold (squeeze), then expands
    sharply, trade in the direction of the release.
    """

    name = "VEF-2-SqueezeReleaseBreakout"
    timeframe = "5min"

    def __init__(
        self,
        bb_window: int = 20,
        squeeze_pct: float = 0.1,
        release_mult: float = 1.5,
    ) -> None:
        self.bb_window = bb_window
        self.squeeze_pct = squeeze_pct
        self.release_mult = release_mult

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"].astype(float)
        _, _, _, bw, _ = feat_bollinger(close, self.bb_window)

        bw_rolling_low = bw.rolling(self.bb_window, min_periods=1).min()
        squeeze = bw <= (bw_rolling_low * (1.0 + self.squeeze_pct))
        was_squeezed = squeeze.shift(1).fillna(False)

        bw_prev = bw.shift(1).replace(0, np.nan)
        release = (~squeeze) & was_squeezed & (bw > bw_prev * self.release_mult)

        ret = close.pct_change().fillna(0.0)
        sig = pd.Series(0.0, index=close.index)
        sig = sig.where(~release, np.sign(ret))
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 2 — VEF-3
# ---------------------------------------------------------------------------


class RangeExpansionExhaustion(IntradayStrategy):
    """VEF-3: Range expansion exhaustion fade.

    When ATR-normalized bar range spikes (momentum exhaustion signal),
    fade the direction of that spike expecting short-term reversal.
    """

    name = "VEF-3-RangeExpansionExhaustion"
    timeframe = "1min"

    def __init__(
        self,
        atr_window: int = 14,
        expansion_threshold: float = 2.5,
    ) -> None:
        self.atr_window = atr_window
        self.expansion_threshold = expansion_threshold

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        h = ohlcv["high"].astype(float)
        l = ohlcv["low"].astype(float)
        c = ohlcv["close"].astype(float)

        bar_range = h - l
        atr = feat_atr(h, l, c, self.atr_window).replace(0, np.nan)
        norm_range = (bar_range / atr).replace([np.inf, -np.inf], np.nan).fillna(1.0)

        exhaustion = norm_range > self.expansion_threshold
        ret = c.pct_change().fillna(0.0)
        sig = pd.Series(0.0, index=c.index)
        sig = sig.where(~exhaustion, -np.sign(ret))
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 2 — HYB-1
# ---------------------------------------------------------------------------


class VolRegimeRouter(IntradayStrategy):
    """HYB-1: Vol regime router — momentum in low vol, fade in high vol.

    Routes between momentum and mean-reversion based on realized vol level.
    """

    name = "HYB-1-VolRegimeRouter"
    timeframe = "1min"

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        vol_threshold: float = 1.3,
        mom_window: int = 5,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window
        self.vol_threshold = vol_threshold
        self.mom_window = mom_window

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        close = ohlcv["close"].astype(float)
        vol_ratio = feat_vol_ratio(close, self.short_window, self.long_window)
        ret = close.pct_change().fillna(0.0)
        mom = ret.rolling(self.mom_window, min_periods=1).mean()

        high_vol = vol_ratio > self.vol_threshold
        mom_sig = np.sign(mom)
        fade_sig = -np.sign(mom)

        sig = pd.Series(0.0, index=close.index)
        sig = sig.where(high_vol, mom_sig)     # low vol -> momentum
        sig = sig.where(~high_vol, fade_sig)   # high vol -> fade
        return _sanitize(sig.shift(1).fillna(0.0))


# ---------------------------------------------------------------------------
# Phase 2 — HYB-2
# ---------------------------------------------------------------------------


class GapRegimeSelector(IntradayStrategy):
    """HYB-2: Gap regime selector — trade with/against opening gaps.

    Gap fills are common; large gaps tend to continue. Switch mode
    based on gap size relative to ATR.
    Open-sensitive: gap trades occur at session open.
    """

    name = "HYB-2-GapRegimeSelector"
    timeframe = "1min"

    def __init__(
        self,
        atr_window: int = 14,
        fill_threshold: float = 0.5,
        continue_threshold: float = 1.5,
    ) -> None:
        self.atr_window = atr_window
        self.fill_threshold = fill_threshold
        self.continue_threshold = continue_threshold

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series:
        o = ohlcv["open"].astype(float)
        h = ohlcv["high"].astype(float)
        l = ohlcv["low"].astype(float)
        c = ohlcv["close"].astype(float)

        atr = feat_atr(h, l, c, self.atr_window).replace(0, np.nan)

        prev_close = c.shift(1).fillna(c)
        gap = (o - prev_close) / atr.replace(0, np.nan)
        gap = gap.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Gap fill: fade the gap direction
        fill_zone = gap.abs() < self.fill_threshold
        # Gap continuation: trade with gap direction
        continue_zone = gap.abs() > self.continue_threshold

        sig = pd.Series(0.0, index=c.index)
        sig = sig.where(~fill_zone, -np.sign(gap))      # fill -> counter-gap
        sig = sig.where(~continue_zone, np.sign(gap))   # large gap -> continuation
        return _sanitize(sig.shift(1).fillna(0.0))
