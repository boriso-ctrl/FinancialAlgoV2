"""Category J: Mean-Reversion & Statistical Arbitrage strategies.

Vectorized mean-reversion with trend filters. Only buy dips in uptrends —
never fight the trend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financial_algo.indicators import realized_vol, zscore
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

        # Market trend filter
        spy_sma = benchmark.rolling(c.trend_window).mean()
        market_uptrend = (benchmark > spy_sma).fillna(False)

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
