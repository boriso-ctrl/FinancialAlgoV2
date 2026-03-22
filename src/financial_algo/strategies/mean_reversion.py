"""Category J: Mean-Reversion & Statistical Arbitrage strategies.

Vectorized mean-reversion with trend filters. Only buy dips in uptrends —
never fight the trend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financial_algo.indicators import dv2, realized_vol, tsi, zscore
from financial_algo.strategies.base import Strategy


# =========================================================================
# J1 — Sector ETF Mean-Reversion (vectorized, trend-filtered)
# =========================================================================

@dataclass
class SectorMeanRevConfig:
    """Config for sector mean-reversion strategy."""

    sector_tickers: list[str] | None = None
    benchmark: str = "SPY"

    zscore_window: int = 20     # shorter = better for MR
    entry_z: float = 1.5
    exit_z: float = 0.3
    trend_window: int = 200     # only MR in uptrend markets

    leverage: float = 1.0
    max_sectors: int = 3

    def __post_init__(self) -> None:
        if self.sector_tickers is None:
            self.sector_tickers = [
                "XLK", "XLF", "XLI", "XLB", "XLP",
                "XLU", "XLY", "XLV", "XLE",
            ]


class SectorMeanReversion(Strategy):
    """Buy oversold sectors when SPY is in uptrend. Long-only.

    Thesis: Sector/SPY ratios mean-revert at short horizons (5-20 days).
    Only buy dips in bull markets — mean-reversion fails in downtrends.
    Vectorized implementation for speed.
    """

    name = "J1-SectorMeanReversion"

    def __init__(self, config: SectorMeanRevConfig | None = None) -> None:
        self.cfg = config or SectorMeanRevConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.sector_tickers if t in prices.columns]
        if c.benchmark not in prices.columns or len(avail) < 3:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        benchmark = prices[c.benchmark]

        # Market trend filter: SPY TSI > 0 (faster than SMA, avoids bear-market MR)
        market_uptrend = (tsi(benchmark) > 0).fillna(False)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Base allocation: benchmark in uptrend
        weights.loc[market_uptrend, c.benchmark] = c.leverage

        # Sector MR overlay: add exposure to oversold bouncing sectors
        z_all = pd.DataFrame(index=prices.index)
        for t in avail:
            ratio = prices[t] / benchmark
            z_all[t] = zscore(ratio, c.zscore_window)

        short_mom = prices[avail].pct_change(5).fillna(0.0) > 0
        buy_signal = (z_all < -c.entry_z) & market_uptrend.values[:, np.newaxis] & short_mom.values

        z_rank = z_all.rank(axis=1, ascending=True)
        top_oversold = z_rank <= c.max_sectors
        selected = buy_signal & top_oversold

        for t in avail:
            weights[t] = weights[t] + selected[t].astype(float) * 0.3

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# J3 — RSI(2) Mean-Reversion with Trend Filter (vectorized)
# =========================================================================

@dataclass
class RSIMeanRevConfig:
    """Config for RSI extreme mean-reversion."""

    tickers: list[str] | None = None

    rsi_window: int = 2         # RSI(2) for fast mean-reversion signals
    oversold: float = 10.0      # RSI below this -> buy
    overbought: float = 90.0    # RSI above this -> sell (not used, long-only)
    exit_rsi: float = 70.0      # exit when RSI recovers above this

    trend_window: int = 200     # 200-day SMA trend filter
    vol_window: int = 20
    vol_max: float = 0.35       # skip extreme vol

    leverage: float = 2.0

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = [
                "SPY", "QQQ", "IWM", "XLK", "XLF", "XLI",
                "XLB", "XLP", "XLU", "XLY", "XLV", "XLE",
                "EFA", "EEM",
            ]


class RSIMeanReversion(Strategy):
    """Buy RSI(2) oversold dips in assets above 200-day SMA. Long-only.

    Thesis: Connors' RSI(2) strategy is one of the most well-documented
    short-term mean-reversion signals. When RSI(2) drops below 10 on
    an asset in a long-term uptrend, the bounce is highly reliable.
    """

    name = "J3-RSIMeanReversion"

    def __init__(self, config: RSIMeanRevConfig | None = None) -> None:
        self.cfg = config or RSIMeanRevConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for t in avail:
            p = prices[t]

            # RSI(2)
            delta = p.diff()
            gain = delta.clip(lower=0).ewm(span=c.rsi_window, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(span=c.rsi_window, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            # Trend filter: above 200-day SMA
            sma = p.rolling(c.trend_window).mean()
            uptrend = p > sma

            # Vol filter
            vol = realized_vol(p, c.vol_window)
            ok_vol = vol <= c.vol_max

            # Buy signal: RSI < oversold AND uptrend AND low vol
            buy = (rsi < c.oversold) & uptrend & ok_vol

            # Hold until RSI > exit
            # Use forward-fill approach: buy stays on until exit condition
            signal = pd.Series(np.nan, index=prices.index)
            signal[buy] = 1.0
            signal[rsi > c.exit_rsi] = 0.0
            signal = signal.ffill().fillna(0.0)

            weights[t] = signal * c.leverage / len(avail)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# J2 — RSI Swing Trader (vectorized, replaces overnight gap fade)
# =========================================================================

@dataclass
class OvernightGapConfig:
    """RSI(5) swing mean-reversion on liquid ETFs."""

    tickers: list[str] | None = None

    rsi_window: int = 5
    oversold: float = 20.0
    exit_rsi: float = 55.0

    trend_window: int = 200
    vol_window: int = 20
    vol_max: float = 0.40

    leverage: float = 1.5

    # Kept for backwards compatibility
    return_window: int = 1
    zscore_window: int = 40
    entry_z: float = 2.0
    exit_days: int = 3

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF", "XLE"]


class OvernightGapFade(Strategy):
    """RSI(5) swing trader: buy dips in uptrends. Long-only.

    Thesis: Short-term RSI dips on liquid ETFs in uptrends revert
    within 3-5 days. Simpler and more robust than gap-fading.
    """

    name = "J2-OvernightGapFade"

    def __init__(self, config: OvernightGapConfig | None = None) -> None:
        self.cfg = config or OvernightGapConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for t in avail:
            p = prices[t]

            # RSI(5)
            delta = p.diff()
            gain = delta.clip(lower=0).ewm(span=c.rsi_window, adjust=False).mean()
            loss = (-delta.clip(upper=0)).ewm(span=c.rsi_window, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            # Trend filter
            sma = p.rolling(c.trend_window).mean()
            uptrend = p > sma

            # Vol filter
            vol = realized_vol(p, c.vol_window)
            ok_vol = vol <= c.vol_max

            # Buy oversold dips in uptrends
            buy = (rsi < c.oversold) & uptrend & ok_vol

            signal = pd.Series(np.nan, index=prices.index)
            signal[buy] = 1.0
            signal[rsi > c.exit_rsi] = 0.0
            signal = signal.ffill().fillna(0.0)

            weights[t] = signal * c.leverage / len(avail)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# J4 — Cointegration Pairs (vectorized, better pairs)
# =========================================================================

@dataclass
class CointPairsConfig:
    """Pairs trading on highly-correlated ETF pairs."""

    pairs: list[tuple] | None = None
    zscore_window: int = 30
    entry_z: float = 2.0
    exit_z: float = 0.3
    leverage: float = 0.4    # per pair, conservative

    def __post_init__(self) -> None:
        if self.pairs is None:
            self.pairs = [
                ("GLD", "UUP"),     # Gold vs dollar (strong inverse)
                ("HYG", "LQD"),     # High yield vs investment grade
                ("XLP", "XLU"),     # Staples vs utilities (defensive pair)
                ("EEM", "EFA"),     # EM vs DM equities
                ("XLY", "XLP"),     # Discretionary vs staples (cycle pair)
            ]


class CointegrationPairs(Strategy):
    """Vectorized mean-reversion on cointegrated ETF pairs.

    Thesis: ETF pairs with strong economic links exhibit
    cointegration. Uses vectorized signal generation with
    better-selected pairs that have genuine mean-reverting spreads.
    """

    name = "J4-CointegrationPairs"

    def __init__(self, config: CointPairsConfig | None = None) -> None:
        self.cfg = config or CointPairsConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Market trend filter: SPY above 200d SMA
        if "SPY" in prices.columns:
            spy_sma = prices["SPY"].rolling(200).mean()
            market_ok = (prices["SPY"] > spy_sma).fillna(False)
        else:
            market_ok = pd.Series(True, index=prices.index)

        for leg_a, leg_b in c.pairs:
            if leg_a not in prices.columns or leg_b not in prices.columns:
                continue

            ratio = prices[leg_a] / prices[leg_b]
            z = zscore(ratio, c.zscore_window)

            w_per = c.leverage / len(c.pairs)

            # Signal with hold logic
            signal = pd.Series(np.nan, index=prices.index)
            signal[z < -c.entry_z] = 1.0      # A cheap vs B
            signal[z > c.entry_z] = -1.0       # B cheap vs A
            signal[(z > -c.exit_z) & (z < c.exit_z)] = 0.0
            signal = signal.ffill().fillna(0.0)

            # Long-only: go long the cheap leg, no shorts
            long_a = (signal == 1) & market_ok
            long_b = (signal == -1) & market_ok

            weights[leg_a] = weights[leg_a] + np.where(long_a, w_per, 0.0)
            weights[leg_b] = weights[leg_b] + np.where(long_b, w_per, 0.0)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# J5 — Global Mean Reversion (weekly z-score on regional ETFs vs SPY)
# =========================================================================

@dataclass
class GlobalMeanRevConfig:
    """Config for global mean reversion strategy."""

    regional_tickers: list[str] | None = None
    benchmark: str = "SPY"

    zscore_window: int = 130   # ~26 weeks in trading days
    entry_z: float = -1.5     # long when z < -1.5 (oversold)
    exit_z: float = 0.0       # exit when z reverts to 0

    trend_window: int = 200   # only buy dips when benchmark in uptrend
    leverage: float = 1.5
    max_regions: int = 3      # max concurrent regional longs

    def __post_init__(self) -> None:
        if self.regional_tickers is None:
            self.regional_tickers = [
                "FXI", "VGK", "EWJ", "INDA", "EEM",
            ]


class GlobalMeanReversion(Strategy):
    """Long oversold regional ETFs vs SPY using 26-week rolling z-score.

    Thesis: Regional equity markets mean-revert relative to the US.
    When a region underperforms dramatically (z < -1.5 on the
    region/SPY ratio over 26 weeks), structural rebalancing flows
    and valuation compression create a reliable bounce. Only buy
    dips when US market is in an uptrend to avoid piling into
    global contagion selloffs.
    """

    name = "J5-GlobalMeanReversion"

    def __init__(self, config: GlobalMeanRevConfig | None = None) -> None:
        self.cfg = config or GlobalMeanRevConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.regional_tickers if t in prices.columns]
        if c.benchmark not in prices.columns or not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        benchmark = prices[c.benchmark]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Market trend filter: benchmark above 200-day SMA
        spy_sma = benchmark.rolling(c.trend_window).mean()
        market_uptrend = (benchmark > spy_sma).fillna(False)

        # Compute z-score of each regional/SPY ratio
        z_all = pd.DataFrame(index=prices.index)
        for t in avail:
            ratio = prices[t] / benchmark
            # Guard division by zero
            ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill()
            z_all[t] = zscore(ratio, c.zscore_window)

        z_all = z_all.fillna(0.0)

        # Entry signal: z < entry_z (oversold) AND market uptrend
        oversold = (z_all < c.entry_z) & market_uptrend.values[:, np.newaxis]

        # Exit signal: z > exit_z (reverted)
        reverted = z_all > c.exit_z

        # Build hold signals with forward-fill logic (vectorized per ticker)
        signals = pd.DataFrame(np.nan, index=prices.index, columns=avail)
        signals[oversold] = 1.0
        signals[reverted] = 0.0
        signals = signals.ffill().fillna(0.0)

        # Limit to max_regions concurrent longs (rank by most oversold)
        active_count = signals.sum(axis=1)
        if c.max_regions > 0:
            z_rank = z_all.rank(axis=1, ascending=True)
            too_many = active_count > c.max_regions
            for t in avail:
                drop = too_many & (z_rank[t] > c.max_regions)
                signals.loc[drop, t] = 0.0

        # Assign weights: equal weight among active regions
        n_active = signals.sum(axis=1).clip(lower=1)
        for t in avail:
            weights[t] = signals[t] * c.leverage / n_active

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# J6 — DV2 Mean-Reversion (Connors/Alvarez oscillator)
# =========================================================================

@dataclass
class DV2MeanRevConfig:
    """Config for DV2 mean-reversion strategy."""

    tickers: list[str] | None = None
    benchmark: str = "SPY"
    dv2_oversold: float = 25.0      # buy when DV2 < this
    dv2_overbought: float = 75.0    # exit when DV2 > this
    trend_window: int = 200         # price > SMA = allowed to long
    ma_period: int = 2              # DV2 smoothing window
    rank_period: int = 252          # DV2 percentile lookback
    max_positions: int = 5          # max concurrent longs
    leverage: float = 1.5

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = [
                "SPY", "QQQ", "IWM", "EFA", "EEM",
                "XLK", "XLF", "XLI", "XLP", "XLB",
                "XLE", "GLD", "TLT",
            ]


class DV2MeanReversion(Strategy):
    """Short-term mean-reversion using the DV2 oscillator.

    Thesis: The DV2 percentile-rank of close/midprice ratio is a robust
    oversold/overbought indicator (Connors/Alvarez 2009).  Buy oversold
    assets that remain in uptrend; exit when DV2 normalises.

    H/L proxy: rolling 2-bar max/min of close used in place of true OHLC.
    """

    name = "J6-DV2MeanReversion"

    def __init__(self, config: DV2MeanRevConfig | None = None) -> None:
        self.cfg = config or DV2MeanRevConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(avail) < 2:
            return weights

        # Trend filter: each asset must close above SMA(trend_window)
        sma = prices[avail].rolling(c.trend_window).mean()
        in_uptrend = prices[avail] > sma

        # DV2 for each asset using rolling 2-bar H/L proxy
        dv2_df = pd.DataFrame(np.nan, index=prices.index, columns=avail)
        for t in avail:
            c_t = prices[t]
            dv2_df[t] = dv2(
                c_t.rolling(2).max(),
                c_t.rolling(2).min(),
                c_t,
                c.ma_period,
                c.rank_period,
            )

        # Signals: buy on oversold + uptrend; exit on overbought or downtrend
        oversold = (dv2_df < c.dv2_oversold) & in_uptrend
        exit_cond = (dv2_df > c.dv2_overbought) | ~in_uptrend

        positions = pd.DataFrame(np.nan, index=prices.index, columns=avail)
        positions[oversold] = 1.0
        positions[exit_cond] = 0.0
        positions = positions.ffill().fillna(0.0)

        # Cap concurrent positions by ranking most oversold (lowest DV2)
        if c.max_positions > 0:
            dv2_rank = dv2_df.rank(axis=1, ascending=True)
            active = positions.sum(axis=1)
            too_many = active > c.max_positions
            for t in avail:
                drop = too_many & (dv2_rank[t] > c.max_positions)
                positions.loc[drop, t] = 0.0

        # Equal weight, normalised to leverage
        n_active = positions.sum(axis=1).clip(lower=1)
        for t in avail:
            weights[t] = positions[t] / n_active * c.leverage

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# S2 — Formulaic Alpha Mean-Reversion (WorldQuant 101: #33, #40, #101)
# =========================================================================

@dataclass
class FormulaicAlphaMeanRevConfig:
    """Config for WorldQuant formulaic alpha mean-reversion composite."""

    rebalance_freq: int = 5        # weekly rebalance (mean-rev is faster)
    long_pct: float = 0.2          # top 20% (quintile)
    short_pct: float = 0.2         # bottom 20%
    long_only: bool = True         # long-only by default
    leverage: float = 1.0
    min_assets: int = 5            # minimum assets for cross-sectional signals


class FormulaicAlphaMeanRev(Strategy):
    """Composite of WorldQuant formulaic mean-reversion alphas (#33, #40, #101).

    Thesis: Intraday return patterns and volatility-price correlations
    exhibit short-term reversal. Alpha033 captures overnight gap reversal
    (open/close ratio), Alpha040 captures the tendency for high-vol assets
    with high price-volume correlation to revert, and Alpha101 captures
    intraday range position reversal.

    Signal: Cross-sectional z-score average of 3 alpha signals.
    Portfolio: Long top quintile (or long-short via config).
    """

    name = "S2-FormulaicAlphaMeanRev"

    def __init__(self, config: FormulaicAlphaMeanRevConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or FormulaicAlphaMeanRevConfig()
        self._open: pd.DataFrame | None = None
        self._high: pd.DataFrame | None = None
        self._low: pd.DataFrame | None = None
        self._volume: pd.DataFrame | None = None

    def set_ohlcv(
        self,
        *,
        open_df: pd.DataFrame | None = None,
        high: pd.DataFrame | None = None,
        low: pd.DataFrame | None = None,
        volume: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """Inject OHLCV data for better signal quality."""
        if open_df is not None:
            self._open = open_df
        if high is not None:
            self._high = high
        if low is not None:
            self._low = low
        if volume is not None:
            self._volume = volume

    def _get_open(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self._open is not None:
            return self._open.reindex(
                index=prices.index, columns=prices.columns,
            ).ffill().bfill()
        # Proxy: previous close (for daily ETFs, open ~ prev close)
        return prices.shift(1).bfill()

    def _get_high(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self._high is not None:
            return self._high.reindex(
                index=prices.index, columns=prices.columns,
            ).ffill().bfill()
        ret_abs = prices.pct_change().fillna(0.0).abs()
        return prices * (1 + ret_abs * 0.5)

    def _get_low(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self._low is not None:
            return self._low.reindex(
                index=prices.index, columns=prices.columns,
            ).ffill().bfill()
        ret_abs = prices.pct_change().fillna(0.0).abs()
        return prices * (1 - ret_abs * 0.5)

    def _get_volume(self, prices: pd.DataFrame) -> pd.DataFrame:
        if self._volume is not None:
            v = self._volume.reindex(
                index=prices.index, columns=prices.columns,
            ).ffill().fillna(1e6)
            return v
        ret_abs = prices.pct_change().fillna(0.0).abs()
        return (ret_abs * prices + 1e4).clip(lower=1.0)

    @staticmethod
    def _cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
        """Cross-sectional z-score per row."""
        mu = df.mean(axis=1)
        std = df.std(axis=1).replace(0, np.nan)
        result = df.sub(mu, axis=0).div(std, axis=0)
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _alpha033(
        self, close: pd.DataFrame, open_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """rank(-1 + (open / close)) -- intraday return reversal."""
        ratio = open_df / close.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        raw = -1.0 + ratio
        return raw.rank(axis=1, pct=True)

    def _alpha040(
        self,
        close: pd.DataFrame,
        high: pd.DataFrame,
        volume: pd.DataFrame,
    ) -> pd.DataFrame:
        """-rank(stddev(high, 10)) * correlation(high, volume, 10)"""
        std_high = high.rolling(10, min_periods=5).std()
        rank_std = std_high.rank(axis=1, pct=True)

        # Rolling correlation between high and volume per asset
        corr_hv = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
        for col in close.columns:
            h = high[col] if col in high.columns else close[col]
            v = volume[col] if col in volume.columns else pd.Series(1e6, index=close.index)
            corr_hv[col] = h.rolling(10, min_periods=5).corr(v)
        corr_hv = corr_hv.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        return -1 * rank_std * corr_hv

    def _alpha101(
        self,
        close: pd.DataFrame,
        open_df: pd.DataFrame,
        high: pd.DataFrame,
        low: pd.DataFrame,
    ) -> pd.DataFrame:
        """(close - open) / ((high - low) + 0.001) -- intraday range position."""
        numerator = close - open_df
        denominator = (high - low).clip(lower=0.001)
        result = numerator / denominator
        return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        if prices.shape[1] < c.min_assets:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        close = prices
        open_df = self._get_open(prices)
        high = self._get_high(prices)
        low = self._get_low(prices)
        volume = self._get_volume(prices)

        # Compute each alpha
        a33 = self._alpha033(close, open_df)
        a40 = self._alpha040(close, high, volume)
        a101 = self._alpha101(close, open_df, high, low)

        # Cross-sectional z-score each alpha
        z33 = self._cs_zscore(a33)
        z40 = self._cs_zscore(a40)
        z101 = self._cs_zscore(a101)

        # Equal-weight composite
        composite = (z33 + z40 + z101) / 3.0

        # Cross-sectional rank
        cs_rank = composite.rank(axis=1, pct=True)

        # Portfolio construction
        long_thresh = 1.0 - c.long_pct
        short_thresh = c.short_pct

        is_long = pd.notna(cs_rank) & (cs_rank >= long_thresh)
        is_short = pd.notna(cs_rank) & (cs_rank <= short_thresh)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.long_only:
            w_long = is_long.astype(float)
            row_sum = w_long.sum(axis=1).clip(lower=1e-8)
            weights = w_long.div(row_sum, axis=0) * c.leverage
        else:
            w_long = is_long.astype(float)
            w_short = is_short.astype(float) * -1.0
            raw = w_long + w_short
            gross = raw.abs().sum(axis=1).clip(lower=1e-8)
            weights = raw.div(gross, axis=0) * c.leverage

        # Weekly rebalance hold
        rebal_mask = pd.Series(False, index=prices.index)
        rebal_mask.iloc[::c.rebalance_freq] = True
        weights = weights.where(rebal_mask).ffill().fillna(0.0)

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
