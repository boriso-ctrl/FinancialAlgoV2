"""Category R: Regime Hardening — Bear Market Alpha & Crisis Hedge Adaptive.

Designed to generate positive returns in sustained equity downtrends:
2015 (oil crash + China deval), 2018 (Volmageddon + Fed), 2022 (war + inflation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_algo.regimes import Regime
from financial_algo.strategies.base import Strategy


# =========================================================================
# R1 -- Bear Market Alpha
# =========================================================================

class BearMarketAlpha(Strategy):
    """R1-BearMarketAlpha: three-signal defensive rotation.

    Combines trend (SPY vs 50/200 SMA), regime, and sector breadth
    signals to rotate into safe havens during sustained downtrends.
    """

    name = "R1-BearMarketAlpha"

    # Tickers
    _SECTOR_ETFS: list[str] = [
        "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    ]
    _SAFE_HAVENS: list[str] = ["TLT", "GLD", "UUP"]

    # SMA windows
    _SMA_SLOW: int = 200
    _SMA_FAST: int = 50

    # Signal weights
    _W_TREND: float = 0.4
    _W_REGIME: float = 0.3
    _W_BREADTH: float = 0.3

    # Breadth threshold
    _BREADTH_THRESHOLD: float = 0.40

    # Crisis regime set
    _CRISIS_REGIMES = {
        Regime.ELEVATED,
        Regime.OIL_CRISIS,
        Regime.WAR_CRISIS,
        Regime.GENERAL_CRISIS,
    }

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        tickers = ["SPY"] + self._SAFE_HAVENS
        # Ensure all needed tickers exist
        available = [t for t in tickers if t in prices.columns]
        w = pd.DataFrame(0.0, index=prices.index, columns=available)

        if "SPY" not in prices.columns:
            return w.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        spy = prices["SPY"]
        n = len(spy)

        # -- Signal 1: Trend (SPY below both 50 & 200 SMA) --
        sma200 = spy.rolling(self._SMA_SLOW, min_periods=1).mean()
        sma50 = spy.rolling(self._SMA_FAST, min_periods=1).mean()
        trend_signal = ((spy < sma200) & (spy < sma50)).astype(float)

        # -- Signal 2: Regime --
        if regime is not None:
            regime_signal = regime.isin(self._CRISIS_REGIMES).astype(float)
        else:
            regime_signal = pd.Series(0.0, index=prices.index)

        # -- Signal 3: Breadth (fraction of sector ETFs above 200 SMA) --
        sector_tickers = [t for t in self._SECTOR_ETFS if t in prices.columns]
        if len(sector_tickers) > 0:
            sector_prices = prices[sector_tickers]
            sector_sma200 = sector_prices.rolling(
                self._SMA_SLOW, min_periods=1,
            ).mean()
            above_sma = (sector_prices > sector_sma200).astype(float)
            breadth_frac = above_sma.mean(axis=1).fillna(0.0)
            breadth_signal = (breadth_frac < self._BREADTH_THRESHOLD).astype(float)
        else:
            breadth_signal = pd.Series(0.0, index=prices.index)

        # -- Composite score --
        score = (
            trend_signal * self._W_TREND
            + regime_signal * self._W_REGIME
            + breadth_signal * self._W_BREADTH
        ).fillna(0.0)

        # -- Position sizing via np.select (vectorized) --
        high_def = pd.notna(score) & (score > 0.5)
        mid_def = pd.notna(score) & (score > 0.3) & (score <= 0.5)
        # Default: score <= 0.3 -> 100% SPY

        if "SPY" in w.columns:
            w["SPY"] = np.select(
                [high_def, mid_def],
                [0.0, 0.40],
                default=1.0,
            )
        if "TLT" in w.columns:
            w["TLT"] = np.select(
                [high_def, mid_def],
                [0.50, 0.30],
                default=0.0,
            )
        if "GLD" in w.columns:
            w["GLD"] = np.select(
                [high_def, mid_def],
                [0.30, 0.20],
                default=0.0,
            )
        if "UUP" in w.columns:
            w["UUP"] = np.select(
                [high_def, mid_def],
                [0.20, 0.10],
                default=0.0,
            )

        return w.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# R5 -- Rates Tightening Alpha
# =========================================================================

class RatesTighteningAlpha(Strategy):
    """R5-RatesTighteningAlpha: profit from rate-hiking cycles.

    Detects tightening via the TLT/IEF ratio rate-of-change (long bonds
    suffering more = rates rising) combined with UUP momentum (dollar
    strengthening = tightening).  Allocates to dollar, short-duration,
    energy, and gold when tightening; shifts to defensive or risk-on
    when easing.

    Max leverage: 1.0 (fully funded, long-only).
    """

    name = "R5-RatesTighteningAlpha"

    _tlt: str = "TLT"
    _ief: str = "IEF"
    _uup: str = "UUP"
    _shy: str = "SHY"
    _xle: str = "XLE"
    _gld: str = "GLD"
    _spy: str = "SPY"

    _roc_window: int = 63
    _roc_threshold: float = -0.02
    _uup_window: int = 63

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        required = [self._tlt, self._ief, self._uup]
        if not all(t in prices.columns for t in required):
            return weights

        # --- Signals ---
        ratio = prices[self._tlt] / prices[self._ief]
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)

        ratio_roc = ratio.pct_change(self._roc_window).fillna(0.0)

        uup_ret = prices[self._uup].pct_change(self._uup_window).fillna(0.0)

        # --- Regime classification ---
        tightening_full = (
            pd.notna(ratio_roc) & (ratio_roc < self._roc_threshold)
            & pd.notna(uup_ret) & (uup_ret > 0)
        )

        tightening_mild = (
            pd.notna(ratio_roc) & (ratio_roc < 0)
            & ~tightening_full
        )

        easing = ~tightening_full & ~tightening_mild

        # --- Allocations ---
        # Full tightening: 30% UUP, 25% SHY, 25% XLE, 20% GLD
        if self._uup in prices.columns:
            weights[self._uup] = np.where(tightening_full, 0.30, weights[self._uup])
        if self._shy in prices.columns:
            weights[self._shy] = np.where(tightening_full, 0.25, weights[self._shy])
        if self._xle in prices.columns:
            weights[self._xle] = np.where(tightening_full, 0.25, weights[self._xle])
        if self._gld in prices.columns:
            weights[self._gld] = np.where(tightening_full, 0.20, weights[self._gld])

        # Mild tightening: 40% SHY, 30% GLD, 15% UUP, 15% SPY
        if self._shy in prices.columns:
            weights[self._shy] = np.where(tightening_mild, 0.40, weights[self._shy])
        if self._gld in prices.columns:
            weights[self._gld] = np.where(tightening_mild, 0.30, weights[self._gld])
        if self._uup in prices.columns:
            weights[self._uup] = np.where(tightening_mild, 0.15, weights[self._uup])
        if self._spy in prices.columns:
            weights[self._spy] = np.where(tightening_mild, 0.15, weights[self._spy])

        # Easing/normal: 60% SPY, 20% TLT, 20% GLD
        if self._spy in prices.columns:
            weights[self._spy] = np.where(easing, 0.60, weights[self._spy])
        if self._tlt in prices.columns:
            weights[self._tlt] = np.where(easing, 0.20, weights[self._tlt])
        if self._gld in prices.columns:
            weights[self._gld] = np.where(easing, 0.20, weights[self._gld])

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# R6 -- Bond-Equity Hedge
# =========================================================================

class BondEquityHedge(Strategy):
    """R6-BondEquityHedge: detect bond-equity correlation flips.

    The traditional 60/40 relies on negative stock-bond correlation
    (bonds rally when stocks fall).  When this correlation flips
    positive (both falling, as in 2022), 60/40 breaks.  This strategy
    monitors rolling correlation and switches to assets uncorrelated
    to both: gold, dollar, short-duration Treasuries.

    Max leverage: 1.0 (fully funded, long-only).
    """

    name = "R6-BondEquityHedge"

    _spy: str = "SPY"
    _tlt: str = "TLT"
    _gld: str = "GLD"
    _uup: str = "UUP"
    _shy: str = "SHY"
    _ief: str = "IEF"

    _corr_window: int = 63
    _trend_window: int = 100
    _pos_corr_threshold: float = 0.2
    _neg_corr_threshold: float = -0.2

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if self._spy not in prices.columns or self._tlt not in prices.columns:
            return weights

        spy_ret = prices[self._spy].pct_change().fillna(0.0)
        tlt_ret = prices[self._tlt].pct_change().fillna(0.0)

        rolling_corr = (
            spy_ret.rolling(self._corr_window, min_periods=30)
            .corr(tlt_ret)
            .fillna(0.0)
        )

        spy_sma = prices[self._spy].rolling(self._trend_window, min_periods=50).mean()
        spy_above_trend = prices[self._spy] >= spy_sma

        # --- Regime classification ---
        pos_corr = pd.notna(rolling_corr) & (rolling_corr > self._pos_corr_threshold)
        neg_corr = pd.notna(rolling_corr) & (rolling_corr < self._neg_corr_threshold)
        ambiguous = ~pos_corr & ~neg_corr

        pos_corr_bearish = pos_corr & ~spy_above_trend
        pos_corr_bullish = pos_corr & spy_above_trend

        # --- Allocations ---
        # Pos corr + bearish: 40% GLD, 30% UUP, 30% SHY
        if self._gld in prices.columns:
            weights[self._gld] = np.where(pos_corr_bearish, 0.40, weights[self._gld])
        if self._uup in prices.columns:
            weights[self._uup] = np.where(pos_corr_bearish, 0.30, weights[self._uup])
        if self._shy in prices.columns:
            weights[self._shy] = np.where(pos_corr_bearish, 0.30, weights[self._shy])

        # Pos corr + bullish: 50% SPY, 25% GLD, 25% SHY
        weights[self._spy] = np.where(pos_corr_bullish, 0.50, weights[self._spy])
        if self._gld in prices.columns:
            weights[self._gld] = np.where(pos_corr_bullish, 0.25, weights[self._gld])
        if self._shy in prices.columns:
            weights[self._shy] = np.where(pos_corr_bullish, 0.25, weights[self._shy])

        # Negative correlation (normal): 60% SPY, 40% TLT
        weights[self._spy] = np.where(neg_corr, 0.60, weights[self._spy])
        weights[self._tlt] = np.where(neg_corr, 0.40, weights[self._tlt])

        # Ambiguous: 40% SPY, 20% TLT, 20% GLD, 20% IEF
        weights[self._spy] = np.where(ambiguous, 0.40, weights[self._spy])
        weights[self._tlt] = np.where(ambiguous, 0.20, weights[self._tlt])
        if self._gld in prices.columns:
            weights[self._gld] = np.where(ambiguous, 0.20, weights[self._gld])
        if self._ief in prices.columns:
            weights[self._ief] = np.where(ambiguous, 0.20, weights[self._ief])

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# R3 -- Defensive Rotation (Dual Momentum + Safe Asset Rotation)
# =========================================================================

_R3_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "IEF", "UUP"]
_R3_SAFE = ["TLT", "GLD", "IEF", "UUP"]


class DefensiveRotationR3(Strategy):
    """R3-DefensiveRotation: Dual momentum rotation into safe assets.

    Signal logic:
      1. Composite score = 0.7 * 12m return + 0.3 * 1m return
      2. Rank all 6 assets; top-2 get 40 %, 3rd gets 20 %
      3. If BOTH SPY and QQQ have negative composite, force
         allocation to only safe assets (top-2 by score, 50/50)
      4. Rebalance monthly (month-end signal, forward-fill)

    Thesis: Pure price-based momentum identifies when equities
    lose absolute momentum, triggering rotation into bonds/gold.
    """

    name = "R3-DefensiveRotation"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        avail = [t for t in _R3_TICKERS if t in prices.columns]
        p = prices.reindex(columns=avail)

        # Momentum scores -------------------------------------------------
        ret_12m = p.pct_change(252).fillna(0.0)
        ret_1m = p.pct_change(21).fillna(0.0)
        composite = 0.7 * ret_12m + 0.3 * ret_1m

        # Month-end mask --------------------------------------------------
        months = p.index.to_series().dt.month
        is_month_end = months.diff().shift(-1).fillna(1) != 0

        # Vectorised ranking per row
        ranks = composite.rank(axis=1, ascending=False)

        # Boolean: both equities negative momentum
        spy_neg = composite["SPY"] < 0 if "SPY" in composite.columns else pd.Series(False, index=p.index)
        qqq_neg = composite["QQQ"] < 0 if "QQQ" in composite.columns else pd.Series(False, index=p.index)
        both_neg = spy_neg & qqq_neg

        # --- Normal mode: top-2 = 0.40, 3rd = 0.20 ----------------------
        normal_w = pd.DataFrame(0.0, index=p.index, columns=avail)
        normal_w[ranks <= 2] = 0.40
        normal_w[(ranks > 2) & (ranks <= 3)] = 0.20
        row_sum = normal_w.sum(axis=1).clip(lower=1e-9)
        normal_w = normal_w.div(row_sum, axis=0)

        # --- Safe mode: only safe assets, top-2 = 0.50 each -------------
        safe_avail = [t for t in _R3_SAFE if t in avail]
        safe_composite = composite.reindex(columns=safe_avail)
        safe_ranks = safe_composite.rank(axis=1, ascending=False)
        safe_w = pd.DataFrame(0.0, index=p.index, columns=avail)
        for t in safe_avail:
            safe_w[t] = np.where(
                pd.notna(safe_ranks[t]) & (safe_ranks[t] <= 2), 0.50, 0.0
            )
        safe_row_sum = safe_w.sum(axis=1).clip(lower=1e-9)
        safe_w = safe_w.div(safe_row_sum, axis=0)

        # Choose mode per row
        weights = pd.DataFrame(0.0, index=p.index, columns=avail)
        weights.loc[~both_neg] = normal_w.loc[~both_neg]
        weights.loc[both_neg] = safe_w.loc[both_neg]

        # Monthly rebalancing: keep month-end signals, forward-fill -------
        weights.loc[~is_month_end] = np.nan
        weights = weights.ffill().fillna(0.0)

        # Gross leverage cap = 1.0
        gross = weights.abs().sum(axis=1).clip(lower=1e-9)
        scale = np.where(gross > 1.0, 1.0 / gross, 1.0)
        weights = weights.mul(scale, axis=0)

        return (
            weights.reindex(columns=prices.columns, fill_value=0.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )


# =========================================================================
# R4 -- Adaptive Risk Budget
# =========================================================================


class AdaptiveRiskBudget(Strategy):
    """R4-AdaptiveRiskBudget: scale equity exposure by trailing drawdown.

    Signal logic:
      1. Compute SPY trailing max drawdown (63-day quarterly lookback)
      2. risk_budget = clip(1.0 + trailing_dd * 3.0, 0, 1)
      3. Equity allocation = risk_budget * (70 % SPY + 30 % QQQ)
      4. Safe allocation = (1 - risk_budget) * (50 % TLT + 30 % GLD + 20 % IEF)

    Thesis: Drawdown is a persistent, self-reinforcing process.
    Scaling down equity exposure as DD deepens avoids the worst losses
    while the safe-haven leg earns positive carry.
    """

    name = "R4-AdaptiveRiskBudget"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        tickers = ["SPY", "QQQ", "TLT", "GLD", "IEF"]
        avail = [t for t in tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "SPY" not in prices.columns:
            return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        spy = prices["SPY"]
        cummax = spy.expanding().max()
        drawdown = (spy / cummax) - 1.0
        trailing_dd = drawdown.rolling(63).min().fillna(0.0)

        # risk_budget: 1.0 when no DD, decays toward 0 as DD deepens
        risk_budget = (1.0 + trailing_dd * 3.0).clip(0.0, 1.0)

        # Equity allocation
        if "SPY" in avail:
            weights["SPY"] = risk_budget * 0.70
        if "QQQ" in avail:
            weights["QQQ"] = risk_budget * 0.30

        # Safe-haven allocation
        safe_budget = 1.0 - risk_budget
        if "TLT" in avail:
            weights["TLT"] = safe_budget * 0.50
        if "GLD" in avail:
            weights["GLD"] = safe_budget * 0.30
        if "IEF" in avail:
            weights["IEF"] = safe_budget * 0.20

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# R7 -- Vol Explosion Alpha
# =========================================================================

class VolExplosionAlpha(Strategy):
    """R7-VolExplosionAlpha: profit from volatility spikes.

    Detects vol explosions via rolling z-score and acceleration of
    realized vol, then rotates into safe havens (TLT/GLD/UUP).
    Designed to be profitable in 2018 Volmageddon and 2022 multi-crisis
    -- the exact environments where carry-oriented vol strategies bleed.

    Signal logic:
    1. Realized vol = SPY 20d rolling std * sqrt(252)
    2. Vol z-score = 252d rolling z-score of realized vol
    3. Vol acceleration = 5d diff of realized vol

    Allocation:
    - vol_zscore > 1.5 AND accel > 0: 40% TLT, 30% GLD, 30% UUP
    - vol_zscore > 0.5 AND accel > 0: 30% TLT, 20% GLD, 20% UUP, 30% SPY
    - vol_zscore < -0.5:              80% SPY, 20% QQQ
    - else:                           50% SPY, 25% TLT, 25% GLD
    """

    name = "R7-VolExplosionAlpha"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "SPY" not in prices.columns:
            return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # --- Signal construction ---
        spy_ret = prices["SPY"].pct_change().fillna(0.0)
        realized_vol = spy_ret.rolling(20, min_periods=1).std().fillna(0.0) * np.sqrt(252)

        # Vol z-score: 252d rolling z-score of realized vol
        vol_mean = realized_vol.rolling(252, min_periods=20).mean()
        vol_std = realized_vol.rolling(252, min_periods=20).std().replace(0, np.nan)
        vol_zscore = ((realized_vol - vol_mean) / vol_std)
        vol_zscore = vol_zscore.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Vol acceleration: 5d change in realized vol
        vol_accel = realized_vol.diff(5).fillna(0.0)

        # --- Regime masks (mutually exclusive, priority order) ---
        vol_exploding = pd.notna(vol_zscore) & (vol_zscore > 1.5) & (vol_accel > 0)
        vol_rising = (
            ~vol_exploding
            & pd.notna(vol_zscore)
            & (vol_zscore > 0.5)
            & (vol_accel > 0)
        )
        vol_compressed = (
            ~vol_exploding & ~vol_rising
            & pd.notna(vol_zscore)
            & (vol_zscore < -0.5)
        )
        normal = ~vol_exploding & ~vol_rising & ~vol_compressed

        # --- Allocations ---
        # Vol exploding: pure safe haven
        weights.loc[vol_exploding, "TLT"] = 0.40
        weights.loc[vol_exploding, "GLD"] = 0.30
        weights.loc[vol_exploding, "UUP"] = 0.30

        # Vol rising: partial hedge
        weights.loc[vol_rising, "TLT"] = 0.30
        weights.loc[vol_rising, "GLD"] = 0.20
        weights.loc[vol_rising, "UUP"] = 0.20
        weights.loc[vol_rising, "SPY"] = 0.30

        # Vol compressed: risk on
        weights.loc[vol_compressed, "SPY"] = 0.80
        weights.loc[vol_compressed, "QQQ"] = 0.20

        # Normal: balanced
        weights.loc[normal, "SPY"] = 0.50
        weights.loc[normal, "TLT"] = 0.25
        weights.loc[normal, "GLD"] = 0.25

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# R8 -- Vol Regime Switcher
# =========================================================================

class VolRegimeSwitcher(Strategy):
    """R8-VolRegimeSwitcher: binary regime switch on vol level.

    Compares current realized vol to its 252d rolling median.
    Simple, robust signal -- elevated vol triggers defensive rotation,
    compressed vol triggers risk-on.

    Signal logic:
    1. Realized vol = SPY 20d rolling std * sqrt(252)
    2. Normal level = 252d rolling median of realized vol
    3. Ratio = current vol / median

    Allocation:
    - ratio > 1.3 (elevated):  40% TLT, 30% GLD, 15% SHY, 15% UUP
    - ratio < 0.7 (compressed): 70% SPY, 30% QQQ
    - else (balanced):           40% SPY, 20% TLT, 20% GLD, 20% IEF
    """

    name = "R8-VolRegimeSwitcher"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "SPY" not in prices.columns:
            return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # --- Signal construction ---
        spy_ret = prices["SPY"].pct_change().fillna(0.0)
        realized_vol = spy_ret.rolling(20, min_periods=1).std().fillna(0.0) * np.sqrt(252)

        # Normal vol level: 252d rolling median
        vol_median = realized_vol.rolling(252, min_periods=20).median().fillna(0.0)

        # Ratio: current vol / median (guard division by zero)
        ratio = realized_vol / vol_median.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # --- Regime masks ---
        elevated = pd.notna(ratio) & (ratio > 1.3)
        compressed = pd.notna(ratio) & (~elevated) & (ratio < 0.7)
        balanced = ~elevated & ~compressed

        # --- Allocations ---
        # Elevated: defensive
        weights.loc[elevated, "TLT"] = 0.40
        weights.loc[elevated, "GLD"] = 0.30
        weights.loc[elevated, "SHY"] = 0.15
        weights.loc[elevated, "UUP"] = 0.15

        # Compressed: aggressive
        weights.loc[compressed, "SPY"] = 0.70
        weights.loc[compressed, "QQQ"] = 0.30

        # Balanced
        weights.loc[balanced, "SPY"] = 0.40
        weights.loc[balanced, "TLT"] = 0.20
        weights.loc[balanced, "GLD"] = 0.20
        weights.loc[balanced, "IEF"] = 0.20

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# R9 -- Multi-Asset CTA Trend Following
# =========================================================================

# Universe: equity, energy, commodities, FX (dollar), short-duration bonds,
# gold, crypto.  Designed to work across ALL macro regimes:
# - 2022: long energy/commodities/dollar, flat bonds/equities
# - 2018: monthly rebalancing avoids Volmageddon daily spikes
# - 2015: long dollar/crypto, flat energy (oil crashed)
_R9_UNIVERSE = [
    "SPY", "QQQ",       # equity
    "XLE", "DBC",       # energy / commodities
    "GLD",              # gold
    "SHY", "UUP",       # short-duration / dollar (rates-hedging)
    "BTC-USD",          # crypto (truly uncorrelated)
]


class MultiAssetCTATrend(Strategy):
    """R9-MultiAssetCTATrend: CTA-style absolute momentum trend following.

    Trades a diversified 8-asset universe using 6-month absolute momentum
    with monthly rebalancing and price-trend confirmation filter.

    Entry logic (per asset):
    - STRONG: 6m return > 0 AND price > 50d SMA  -> vol-scaled full weight
    - WEAK:   6m return > 0 AND price < 50d SMA  -> half weight
    - FLAT:   6m return <= 0                      -> zero weight

    Positions updated at month-end and forward-filled.
    Vol-scaled per position to target 10% vol contribution.
    Gross leverage capped at 1.5.
    """

    name = "R9-MultiAssetCTATrend"

    _MOM_WINDOW: int = 126    # 6-month momentum
    _SMA_WINDOW: int = 50     # Price trend confirmation
    _VOL_WINDOW: int = 20     # Vol-scaling lookback
    _VOL_TARGET: float = 0.10  # 10% vol per position
    _MAX_LEVERAGE: float = 1.5

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        avail = [t for t in _R9_UNIVERSE if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        p = prices.reindex(columns=avail).ffill()
        ret_6m = p.pct_change(self._MOM_WINDOW).fillna(0.0)
        sma50 = p.rolling(self._SMA_WINDOW, min_periods=10).mean()

        # Per-asset realized vol for position sizing
        daily_ret = p.pct_change().fillna(0.0)
        rvol = (
            daily_ret.rolling(self._VOL_WINDOW, min_periods=5).std().fillna(0.01)
            * np.sqrt(252)
        )
        # Cap at 2x equal-weight per asset
        vol_scale = (self._VOL_TARGET / rvol.clip(lower=0.01)).clip(upper=2.0 / max(len(avail), 1))

        # Momentum signals (vectorised boolean masks)
        mom_pos = ret_6m > 0
        above_sma = p > sma50

        strong_long = mom_pos & above_sma      # full vol-scaled weight
        weak_long = mom_pos & ~above_sma       # half weight (momentum fading)

        raw_weights = pd.DataFrame(0.0, index=p.index, columns=avail)
        raw_weights[strong_long] = vol_scale[strong_long]
        raw_weights[weak_long] = vol_scale[weak_long] * 0.5

        # Monthly rebalancing: signal at month-end, forward-fill
        months = p.index.to_series().dt.month
        is_month_end = months.diff().shift(-1).fillna(1) != 0
        raw_weights.loc[~is_month_end] = np.nan
        raw_weights = raw_weights.ffill().fillna(0.0)

        # Gross leverage cap
        gross = raw_weights.abs().sum(axis=1).clip(lower=1e-9)
        scale = (self._MAX_LEVERAGE / gross).clip(upper=1.0)
        raw_weights = raw_weights.multiply(scale, axis=0)

        return (
            raw_weights.reindex(columns=prices.columns, fill_value=0.0)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )


# =========================================================================
# R10 -- Commodity Macro Overlay
# =========================================================================

class CommodityMacroOverlay(Strategy):
    """R10-CommodityMacroOverlay with tightening/easing state machine.

    Thesis: Commodity inflation regimes are strongest when energy leadership,
    dollar strength, and rising rates align. Outside those windows, stay in a
    defensive macro sleeve (TLT/GLD/UUP/SHY) rather than idle cash so the
    strategy remains productive with controlled drawdowns.
    """

    name = "R10-CommodityMacroOverlay"

    _MOM_LOOKBACK: int = 63           # 3-month signal window
    _SMA_WINDOW: int = 200
    _FAST_MOM: int = 21
    _XLE_THRESHOLD: float = 0.06      # XLE 3m return > 6%
    _REL_THRESHOLD: float = 0.04      # XLE/SPY relative 3m return > 4%
    _UUP_THRESHOLD: float = 0.00      # UUP momentum must be positive
    _TLT_THRESHOLD: float = -0.01     # TLT momentum negative = tightening
    _MAX_LEVERAGE: float = 1.0
    _DD_WINDOW: int = 63

    # Tightening commodity regime
    _ACTIVE_WEIGHTS = {
        "XLE": 0.30,
        "DBC": 0.25,
        "UUP": 0.15,
        "SHY": 0.15,
        "GLD": 0.15,
    }
    # Defensive regime
    _DEFENSIVE_WEIGHTS = {
        "TLT": 0.40,
        "GLD": 0.30,
        "UUP": 0.15,
        "SHY": 0.15,
    }
    # Transition regime
    _TRANSITION_WEIGHTS = {
        "TLT": 0.15,
        "GLD": 0.30,
        "UUP": 0.15,
        "SHY": 0.40,
    }

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        required = ["XLE", "UUP", "TLT", "SPY"]
        if not all(t in prices.columns for t in required):
            return weights

        p = prices.ffill()
        m = self._MOM_LOOKBACK

        xle_ret = p["XLE"].pct_change(m).fillna(0.0)
        xle_ret_fast = p["XLE"].pct_change(self._FAST_MOM).fillna(0.0)
        uup_ret = p["UUP"].pct_change(m).fillna(0.0)
        tlt_ret = p["TLT"].pct_change(m).fillna(0.0)
        spy_ret = p["SPY"].pct_change(m).fillna(0.0)
        xle_rel = (xle_ret - spy_ret).fillna(0.0)

        if "DBC" in p.columns:
            dbc_ret = p["DBC"].pct_change(m).fillna(0.0)
        else:
            dbc_ret = xle_ret

        if "^VIX" in p.columns:
            vix = p["^VIX"]
            stress = pd.notna(vix) & (vix > 26.0)
        else:
            stress = pd.Series(False, index=p.index)

        sma200   = p["SPY"].rolling(self._SMA_WINDOW, min_periods=50).mean().fillna(p["SPY"])
        spy_bear = p["SPY"] < sma200

        commodity_regime = (
            (xle_ret > self._XLE_THRESHOLD)
            & (xle_rel > self._REL_THRESHOLD)
            & (uup_ret > self._UUP_THRESHOLD)
            & (tlt_ret < self._TLT_THRESHOLD)
            & (dbc_ret > 0.02)
            & ((xle_ret_fast > 0.0) | (dbc_ret > 0.05))
        )

        defensive_regime = (stress | spy_bear) & ~commodity_regime
        transition_regime = ~commodity_regime & ~defensive_regime

        # Monthly rebalancing: evaluate signal at each month-end, carry forward
        months = p.index.to_series().dt.month
        is_month_end = months.diff().shift(-1).fillna(1) != 0
        cm = commodity_regime.astype(float)
        df = defensive_regime.astype(float)
        tr = transition_regime.astype(float)
        cm[~is_month_end] = np.nan
        df[~is_month_end] = np.nan
        tr[~is_month_end] = np.nan
        cm = cm.ffill().fillna(0.0).astype(bool)
        df = df.ffill().fillna(0.0).astype(bool)
        tr = tr.ffill().fillna(1.0).astype(bool)

        active_target = dict(self._ACTIVE_WEIGHTS)
        if "DBC" not in weights.columns:
            active_target["XLE"] = active_target["XLE"] + active_target.get("DBC", 0.0)
            active_target.pop("DBC", None)

        for ticker, w in active_target.items():
            if ticker in weights.columns:
                weights[ticker] = np.where(cm, w, weights[ticker])

        for ticker, w in self._DEFENSIVE_WEIGHTS.items():
            if ticker in weights.columns:
                weights[ticker] = np.where(df, w, weights[ticker])

        for ticker, w in self._TRANSITION_WEIGHTS.items():
            if ticker in weights.columns:
                weights[ticker] = np.where(tr, w, weights[ticker])

        # Commodity sleeve drawdown control: reduce cyclical risk after deep pullback.
        comm_proxy = pd.Series(0.0, index=p.index)
        if "XLE" in p.columns:
            comm_proxy = comm_proxy + 0.6 * p["XLE"].pct_change().fillna(0.0)
        if "DBC" in p.columns:
            comm_proxy = comm_proxy + 0.4 * p["DBC"].pct_change().fillna(0.0)
        comm_curve = (1.0 + comm_proxy).cumprod()
        comm_dd = comm_curve / comm_curve.cummax().replace(0.0, np.nan) - 1.0
        dd_scale = np.where(comm_dd < -0.15, 0.5, np.where(comm_dd < -0.10, 0.75, 1.0))
        dd_scale = pd.Series(dd_scale, index=p.index)

        cyclical_cols = [t for t in ["XLE", "DBC"] if t in weights.columns]
        for t in cyclical_cols:
            weights[t] = weights[t] * dd_scale

        if "SHY" in weights.columns:
            lost = (self._MAX_LEVERAGE - weights.sum(axis=1)).clip(lower=0.0)
            weights["SHY"] = weights["SHY"] + lost

        gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
        scale = (self._MAX_LEVERAGE / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.multiply(scale, axis=0)

        return (
            weights
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
