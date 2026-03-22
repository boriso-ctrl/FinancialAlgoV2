"""Category Z: Structural Alpha strategies.

These exploit *mechanical* flows and market structure effects that exist
regardless of fundamentals. Fund rebalancing, index reconstitution,
option expiration flows, and roll yields create predictable price pressure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_algo.strategies.base import Strategy
from financial_algo.indicators import realized_vol


# =========================================================================
# Z1 -- Month-End Rebalancing Flow
# =========================================================================

class RebalancingFlow(Strategy):
    """Exploit predictable month-end pension/fund rebalancing flows.

    Thesis: Trillions of dollars in 60-40 pension funds, target-date funds,
    and risk-parity funds rebalance at month-end. When stocks outperform
    bonds during the month, these funds must SELL stocks and BUY bonds
    at month-end to return to target weights. Vice versa when bonds win.

    This creates predictable counter-trend pressure in the LAST 3 trading
    days of each month and a REVERSAL in the first 2 days.

    Signal: If stocks beat bonds during the month, go long bonds (short
    stocks) in last 3 days, then reverse. This is a structural alpha --
    it exists because of fund mandates, not information.

    Academic basis: Hutchinson & O'Brien (2020) "End-of-month rebalancing"
    """

    name = "Z1-RebalancingFlow"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # KILLED: Month-end rebalancing signal produced Sharpe -0.14.
        # Daily frequency too coarse to capture intraday rebalancing flows.
        # Would need intraday execution to exploit. Archived.
        return weights


# =========================================================================
# Z2 -- VIX Expiration Gamma Pin
# =========================================================================

class GammaPin(Strategy):
    """Exploit options expiration gamma effects on SPY.

    Thesis: On monthly opex (3rd Friday), dealers' gamma hedging
    pins SPY near max-pain strikes. The week BEFORE opex, dealer
    hedging compresses realized vol. The week AFTER opex, vol
    normalizes (often expands).

    Signal:
    - Days -5 to -1 before opex: low-vol environment -> sell vol proxy
      (go long SPY in calm, vol is suppressed by dealer hedging)
    - Days +1 to +3 after opex: vol unpin -> buy vol proxy (TLT/GLD)
      if VIX is elevated

    This is a market microstructure alpha. It exists because of the
    mechanical hedging behavior of options market makers.

    Reference: Cboe research, SqueezeMetrics "GEX" research
    """

    name = "Z2-GammaPin"

    @staticmethod
    def _monthly_opex_dates(index: pd.DatetimeIndex) -> pd.Series:
        """Return a Series with days-to-opex for each date.

        Monthly opex = 3rd Friday of each month.
        """
        dates = index.to_series()
        result = pd.Series(np.nan, index=index)

        # Find 3rd Friday of each month in the date range
        year_months = dates.dt.to_period("M").unique()
        opex_dates = []
        for ym in year_months:
            # Get all days in this month
            month_start = ym.start_time
            # Find the 3rd Friday
            first_day = month_start
            # Day of week: Monday=0, Friday=4
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + pd.Timedelta(days=days_until_friday)
            third_friday = first_friday + pd.Timedelta(weeks=2)
            opex_dates.append(third_friday)

        # For each trading day, compute distance to nearest opex
        opex_ts = pd.DatetimeIndex(opex_dates)
        for i, d in enumerate(index):
            # Distance to closest opex
            diffs = (opex_ts - d).total_seconds() / 86400.0
            # Find nearest upcoming opex
            future = diffs[diffs >= 0]
            if len(future) > 0:
                result.iloc[i] = future.min()
            else:
                result.iloc[i] = 30  # far from opex

        return result

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "SPY" not in prices.columns:
            return weights

        days_to_opex = self._monthly_opex_dates(prices.index)

        # Pre-opex week (5 days before): gamma pin suppresses vol
        pre_opex = (days_to_opex >= 1) & (days_to_opex <= 5)
        # Post-opex (3 days after opex -- days_to_opex was just 0, now ~28)
        # Opex day itself (days_to_opex == 0 or just rolled to ~28)
        # Use: days_to_opex > 25 (just after opex, the next one is ~28 days away)
        post_opex = days_to_opex > 25
        # Mid-period: between opex events
        mid_period = (days_to_opex > 5) & (days_to_opex <= 25)

        # 200-day trend filter
        spy_sma = prices["SPY"].rolling(200).mean()
        trend_up = prices["SPY"] > spy_sma

        # Pre-opex + trend up: concentrated long (gamma pin suppresses vol)
        weights.loc[pre_opex & trend_up, "SPY"] = 0.7
        if "QQQ" in prices.columns:
            weights.loc[pre_opex & trend_up, "QQQ"] = 0.3

        # Mid-period + trend up: moderate equity with light defense
        weights.loc[mid_period & trend_up, "SPY"] = 0.40
        if "IWM" in prices.columns:
            weights.loc[mid_period & trend_up, "IWM"] = 0.15
        if "TLT" in prices.columns:
            weights.loc[mid_period & trend_up, "TLT"] = 0.10

        # Post-opex: moderate equity with haven diversification
        weights.loc[post_opex & trend_up, "SPY"] = 0.40
        if "TLT" in prices.columns:
            weights.loc[post_opex & trend_up, "TLT"] = 0.20
        if "GLD" in prices.columns:
            weights.loc[post_opex & trend_up, "GLD"] = 0.10

        # Trend down: balanced defensive (GLD hedges inflation, TLT hedges deflation)
        if "GLD" in prices.columns:
            weights.loc[~trend_up, "GLD"] = 0.35
        weights.loc[~trend_up, "TLT"] = 0.25
        if "UUP" in prices.columns:
            weights.loc[~trend_up, "UUP"] = 0.10

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# Z3 -- Sector Dispersion Trade
# =========================================================================

class SectorDispersion(Strategy):
    """Trade sector return dispersion as a vol and regime signal.

    Thesis: When sector returns are highly dispersed (some sectors
    up big, others down big), it signals a rotation/regime shift.
    Low dispersion means calm, trend-following markets.

    High dispersion + rising VIX = crisis (defensive)
    High dispersion + falling VIX = rotation (buy laggards)
    Low dispersion + trend up = steady bull (stay long)

    This is a 2nd-order signal -- we're trading the DISTRIBUTION
    of returns, not the returns themselves.

    Academic basis: Stivers & Sun (2010) "Cross-Sectional Return Dispersion
    and Time Variation in Value and Momentum Premiums"
    """

    name = "Z3-SectorDispersion"

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

        # 20-day sector returns
        sect_ret = prices[sectors].pct_change(20).fillna(0.0)

        # Cross-sectional dispersion (std across sectors each day)
        dispersion = sect_ret.std(axis=1)

        # Z-score of dispersion
        disp_ma = dispersion.rolling(60).mean()
        disp_std = dispersion.rolling(60).std().replace(0, np.nan)
        disp_z = ((dispersion - disp_ma) / disp_std).fillna(0.0)

        # VIX proxy: 20-day realized vol of SPY
        spy_vol = realized_vol(prices["SPY"], 20)
        vol_falling = spy_vol < spy_vol.rolling(60).mean()

        # Trend
        spy_sma = prices["SPY"].rolling(200).mean()
        trend_up = prices["SPY"] > spy_sma

        # Low dispersion + trend up = calm bull -> long equity
        calm_bull = (disp_z < 0.5) & trend_up
        weights.loc[calm_bull, "SPY"] = 0.6
        if "QQQ" in prices.columns:
            weights.loc[calm_bull, "QQQ"] = 0.4

        # High dispersion + vol falling = rotation -> buy lagging sectors
        rotation = (disp_z > 1.0) & vol_falling & trend_up
        if rotation.any():
            # Buy the 3 worst-performing sectors (vectorized rank approach)
            ranks = sect_ret[sectors].rank(axis=1, ascending=True, method='first')
            bottom_3 = (ranks <= 3).astype(float) * 0.33
            for t in sectors:
                weights.loc[rotation, t] = bottom_3.loc[rotation, t].fillna(0.0)

        # High dispersion + vol rising = crisis -> defensive
        crisis = (disp_z > 1.5) & ~vol_falling
        weights.loc[crisis, "TLT"] = 0.4
        weights.loc[crisis, "GLD"] = 0.3
        weights.loc[crisis, "IEF"] = 0.2

        return weights
