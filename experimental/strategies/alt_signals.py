"""Category W: Alternative Signal strategies.

These use unconventional signals -- breadth divergences, dispersion
dynamics, and correlation regime shifts -- that most quants overlook.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_algo.strategies.base import Strategy
from financial_algo.indicators import realized_vol, moving_average


# =========================================================================
# W1 -- Breadth Divergence
# =========================================================================

class BreadthDivergence(Strategy):
    """Trade divergences between market breadth and index price.

    Thesis: When SPY makes new highs but fewer sectors are participating
    (declining breadth), the rally is narrow and fragile. This divergence
    historically precedes corrections. When breadth expands from a low
    (more sectors above their 50-day SMA), it's a broad-based rally
    that has legs.

    We measure breadth as the fraction of sector ETFs above their 50-day
    SMA vs SPY's position relative to its 50-day SMA.

    Academic basis: Fosback (1976) "Stock Market Logic",
    Zweig (1986) breadth thrust indicator.
    """

    name = "W1-BreadthDivergence"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        sectors = [t for t in ["XLK", "XLF", "XLI", "XLB", "XLP", "XLU",
                               "XLY", "XLV", "XLE"]
                   if t in prices.columns]
        if len(sectors) < 5 or "SPY" not in prices.columns:
            return weights

        # Breadth: fraction of sectors above their 50-day SMA
        above_sma = pd.DataFrame(index=prices.index, columns=sectors, dtype=float)
        for s in sectors:
            sma = prices[s].rolling(50).mean()
            above_sma[s] = (prices[s] > sma).astype(float)

        breadth = above_sma.mean(axis=1)  # 0 to 1

        # SPY trend
        spy_sma50 = prices["SPY"].rolling(50).mean()
        spy_sma200 = prices["SPY"].rolling(200).mean()
        spy_above_50 = prices["SPY"] > spy_sma50
        spy_trend_up = prices["SPY"] > spy_sma200

        # Breadth z-score (60-day rolling)
        breadth_ma = breadth.rolling(60).mean()
        breadth_std = breadth.rolling(60).std().replace(0, np.nan)
        breadth_z = ((breadth - breadth_ma) / breadth_std).fillna(0.0)

        # BULLISH: broad rally (breadth high + SPY trending up)
        broad_rally = (breadth > 0.7) & spy_trend_up
        weights.loc[broad_rally, "SPY"] = 0.5
        if "QQQ" in prices.columns:
            weights.loc[broad_rally, "QQQ"] = 0.3
        if "IWM" in prices.columns:
            weights.loc[broad_rally, "IWM"] = 0.2  # small caps join broad rallies

        # BREADTH THRUST: breadth z-score surges from low (powerful buy signal)
        thrust = (breadth_z > 1.5) & (breadth.shift(10) < 0.3)
        weights.loc[thrust, "SPY"] = 0.6
        if "QQQ" in prices.columns:
            weights.loc[thrust, "QQQ"] = 0.4

        # BEARISH DIVERGENCE: SPY above 50d SMA but breadth collapsing
        divergence = spy_above_50 & (breadth < 0.3)
        if "TLT" in prices.columns:
            weights.loc[divergence, "TLT"] = 0.4
        if "GLD" in prices.columns:
            weights.loc[divergence, "GLD"] = 0.3

        # DEEP WEAKNESS: breadth collapsed AND SPY below 200d
        deep_weak = (~spy_trend_up) & (breadth < 0.2)
        if "TLT" in prices.columns:
            weights.loc[deep_weak, "TLT"] = 0.5
        if "GLD" in prices.columns:
            weights.loc[deep_weak, "GLD"] = 0.3
        if "IEF" in prices.columns:
            weights.loc[deep_weak, "IEF"] = 0.2

        return weights


# =========================================================================
# W2 -- Correlation Regime Break
# =========================================================================

class CorrelationRegimeBreak(Strategy):
    """Trade shifts in cross-asset correlation structure.

    Thesis: In normal times, stocks and bonds are negatively correlated
    (the "diversification dividend"). When this correlation flips to
    positive (both stocks AND bonds falling), it signals an inflation
    regime or systemic stress -- traditional diversification breaks down.

    When stock/bond correlation is normal (negative): risk-on equities
    When correlation turns positive: switch to real assets (gold, commodities)
    When correlation spikes to extremely positive: pure haven (cash/gold)

    This is an "insurance signal" -- it detects the exact moment
    traditional portfolios stop working.

    Academic basis: Campbell, Sunderam & Viceira (2017) "Inflation Bets"
    """

    name = "W2-CorrelationRegimeBreak"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "SPY" not in prices.columns or "TLT" not in prices.columns:
            return weights

        spy_ret = prices["SPY"].pct_change().fillna(0.0)
        tlt_ret = prices["TLT"].pct_change().fillna(0.0)

        # Rolling 60-day correlation between stocks and bonds
        corr_60 = spy_ret.rolling(60).corr(tlt_ret).fillna(0.0)

        # Long-term average correlation
        corr_ma = corr_60.rolling(252).mean().fillna(corr_60.expanding().mean())
        corr_std = corr_60.rolling(252).std().replace(0, np.nan)
        corr_z = ((corr_60 - corr_ma) / corr_std).fillna(0.0)

        # 200d trend filter
        spy_sma = prices["SPY"].rolling(200).mean()
        trend_up = prices["SPY"] > spy_sma

        # NORMAL regime: stock-bond correlation negative -> alternative equity basket
        # (no SPY/QQQ to decorrelate from production ensemble)
        normal_corr = corr_60 < -0.1
        if "IWM" in prices.columns:
            weights.loc[normal_corr & trend_up, "IWM"] = 0.25
        if "GLD" in prices.columns:
            weights.loc[normal_corr & trend_up, "GLD"] = 0.20
        if "HYG" in prices.columns:
            weights.loc[normal_corr & trend_up, "HYG"] = 0.15
        if "EFA" in prices.columns:
            weights.loc[normal_corr & trend_up, "EFA"] = 0.15
        if "EEM" in prices.columns:
            weights.loc[normal_corr & trend_up, "EEM"] = 0.15
        if "XLB" in prices.columns:
            weights.loc[normal_corr & trend_up, "XLB"] = 0.10

        # INFLATION REGIME: positive or near-zero correlation (broadened threshold)
        inflation = (corr_60 > 0.10) & (corr_z < 2.0)
        if "GLD" in prices.columns:
            weights.loc[inflation, "GLD"] = 0.50
        if "XLE" in prices.columns:
            weights.loc[inflation, "XLE"] = 0.20
        if "UUP" in prices.columns:
            weights.loc[inflation, "UUP"] = 0.30

        # CORRELATION BREAKDOWN: extreme positive correlation (systemic stress)
        breakdown = corr_z > 2.0
        if "GLD" in prices.columns:
            weights.loc[breakdown, "GLD"] = 0.5
        if "UUP" in prices.columns:
            weights.loc[breakdown, "UUP"] = 0.3

        # Trend down + normal correlation: inflation-resilient defensive
        if "GLD" in prices.columns:
            weights.loc[~trend_up & normal_corr, "GLD"] = 0.35
        if "UUP" in prices.columns:
            weights.loc[~trend_up & normal_corr, "UUP"] = 0.20
        weights.loc[~trend_up & normal_corr, "TLT"] = 0.15

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# W3 -- Liquidity Vacuum Detector
# =========================================================================

class LiquidityVacuum(Strategy):
    """Detect and trade around liquidity vacuums.

    Thesis: Markets crash not because of bad news per se, but because
    liquidity evaporates. We detect liquidity vacuums using:
    1. Volume surge (panic selling = high volume)
    2. Price gap (large intraday ranges proxy via high-low / ATR)
    3. Cross-asset correlation spike (everything dumps together)

    When liquidity is thin but prices haven't crashed yet, go defensive.
    When liquidity vacuum causes an overshoot (VIX spike + breadth washout),
    buy the snapback.

    This is a microstructure + behavioral signal. Market makers widen
    spreads during uncertainty, creating positive feedback loops.

    Reference: Brunnermeier & Pedersen (2009) "Market Liquidity and
    Funding Liquidity"
    """

    name = "W3-LiquidityVacuum"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "SPY" not in prices.columns:
            return weights

        # Realized vol as liquidity proxy (high vol = low liquidity)
        spy_vol = realized_vol(prices["SPY"], 10)
        vol_ma = spy_vol.rolling(60).mean()
        vol_std = spy_vol.rolling(60).std().replace(0, np.nan)
        vol_z = ((spy_vol - vol_ma) / vol_std).fillna(0.0)

        # Cross-asset correlation as herding proxy
        risk_assets = [t for t in ["SPY", "QQQ", "IWM", "EEM", "HYG", "XLE"]
                       if t in prices.columns]
        if len(risk_assets) >= 3:
            risk_ret = prices[risk_assets].pct_change().fillna(0.0)
            # Average pairwise rolling correlation (vectorized over time)
            n_assets = len(risk_assets)
            pair_corrs = []
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    c = risk_ret.iloc[:, i].rolling(60).corr(risk_ret.iloc[:, j])
                    pair_corrs.append(c)
            if pair_corrs:
                avg_corr = pd.concat(pair_corrs, axis=1).mean(axis=1).fillna(0.0)
            else:
                avg_corr = pd.Series(0.0, index=prices.index)
        else:
            avg_corr = pd.Series(0.0, index=prices.index)

        corr_z_ma = avg_corr.rolling(120).mean()
        corr_z_std = avg_corr.rolling(120).std().replace(0, np.nan)
        corr_z = ((avg_corr - corr_z_ma) / corr_z_std).fillna(0.0)

        # 200d trend
        spy_sma = prices["SPY"].rolling(200).mean()
        trend_up = prices["SPY"] > spy_sma

        # NORMAL LIQUIDITY + trend up: alternative equity basket
        # (no SPY/QQQ to decorrelate from production ensemble)
        normal = (vol_z < 1.0) & (corr_z < 1.0) & trend_up
        if "IWM" in prices.columns:
            weights.loc[normal, "IWM"] = 0.20
        if "GLD" in prices.columns:
            weights.loc[normal, "GLD"] = 0.25
        if "HYG" in prices.columns:
            weights.loc[normal, "HYG"] = 0.15
        if "EFA" in prices.columns:
            weights.loc[normal, "EFA"] = 0.15
        if "EEM" in prices.columns:
            weights.loc[normal, "EEM"] = 0.10
        if "XLB" in prices.columns:
            weights.loc[normal, "XLB"] = 0.15

        # LIQUIDITY WARNING: vol elevated but no panic yet
        warning = (vol_z > 1.0) & (vol_z < 2.5) & (corr_z < 1.5)
        if "GLD" in prices.columns:
            weights.loc[warning, "GLD"] = 0.30
        weights.loc[warning, "SPY"] = 0.15
        if "TLT" in prices.columns:
            weights.loc[warning, "TLT"] = 0.15
        if "UUP" in prices.columns:
            weights.loc[warning, "UUP"] = 0.10

        # LIQUIDITY VACUUM: vol spike + correlation spike (everything correlated)
        vacuum = (vol_z > 2.5) | (corr_z > 2.0)
        if "GLD" in prices.columns:
            weights.loc[vacuum, "GLD"] = 0.4
        if "TLT" in prices.columns:
            weights.loc[vacuum, "TLT"] = 0.3
        if "UUP" in prices.columns:
            weights.loc[vacuum, "UUP"] = 0.2

        # SNAPBACK: vol was extreme (vacuum) but now declining rapidly
        # Use cyclicals (they bounce hardest after liquidity crises)
        vol_declining = (vol_z.shift(5) > 2.5) & (vol_z < 1.5) & trend_up
        if "IWM" in prices.columns:
            weights.loc[vol_declining, "IWM"] = 0.30
        if "EEM" in prices.columns:
            weights.loc[vol_declining, "EEM"] = 0.25
        if "XLE" in prices.columns:
            weights.loc[vol_declining, "XLE"] = 0.25
        if "XLB" in prices.columns:
            weights.loc[vol_declining, "XLB"] = 0.20

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
