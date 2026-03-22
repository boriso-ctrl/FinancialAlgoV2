"""Category N: Calendar & Seasonal strategies.

Exploit well-documented calendar effects in equity markets.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from financial_algo.strategies.base import Strategy


# =========================================================================
# N1 — Sell-in-May / Halloween Effect
# =========================================================================

@dataclass
class SeasonalConfig:
    """Config for seasonal strategy."""

    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"

    # Strong months: Nov-Apr (inclusive). Weak months: May-Oct.
    strong_months: tuple = (11, 12, 1, 2, 3, 4)

    leverage_strong: float = 1.5   # leveraged long in strong months
    leverage_weak: float = 0.0     # flat or safe in weak months
    safe_weight_weak: float = 0.5  # TLT in weak months


class SeasonalStrategy(Strategy):
    """Sell in May, buy in November — riding the Halloween effect.

    Thesis: Equities historically return more in Nov-Apr than May-Oct.
    Documented in Bouman & Jacobsen (2002), robust across markets and
    centuries. One of the most persistent calendar anomalies.
    """

    name = "N1-SeasonalStrategy"

    def __init__(self, config: SeasonalConfig | None = None) -> None:
        self.cfg = config or SeasonalConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        months = prices.index.month

        strong = pd.Series(False, index=prices.index)
        for m in c.strong_months:
            strong |= (months == m)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights.loc[strong, c.equity_ticker] = c.leverage_strong
        weights.loc[~strong, c.safe_ticker] = c.safe_weight_weak

        return weights.fillna(0.0)


# =========================================================================
# N2 — Turn-of-Month Effect
# =========================================================================

@dataclass
class TurnOfMonthConfig:
    """Config for turn-of-month strategy."""

    equity_ticker: str = "SPY"

    # Days before month-end and after month-start to be long
    days_before_end: int = 3
    days_after_start: int = 3

    leverage_tom: float = 2.0     # leveraged during turn-of-month
    leverage_other: float = 0.0   # flat otherwise


class TurnOfMonth(Strategy):
    """Long equity during the turn-of-month window.

    Thesis: Equity returns concentrate around month-end and month-start
    (last 3 + first 3 trading days). This effect is driven by institutional
    cash flows, salary payments, and pension fund rebalancing.
    Documented in Ariel (1987) and Lakonishok & Smidt (1988).
    """

    name = "N2-TurnOfMonth"

    def __init__(self, config: TurnOfMonthConfig | None = None) -> None:
        self.cfg = config or TurnOfMonthConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        idx = prices.index

        # Compute trading-day-of-month for each date
        months = idx.to_period("M")
        tom_mask = pd.Series(False, index=idx)

        for period in months.unique():
            month_dates = idx[months == period]
            n = len(month_dates)
            if n == 0:
                continue

            # First N days of month
            first_days = month_dates[:c.days_after_start]
            # Last N days of month
            last_days = month_dates[-c.days_before_end:]

            tom_mask.loc[first_days] = True
            tom_mask.loc[last_days] = True

        # Trend filter: SPY above 200d SMA
        if c.equity_ticker in prices.columns:
            sma = prices[c.equity_ticker].rolling(200).mean()
            uptrend = (prices[c.equity_ticker] > sma).fillna(False)
        else:
            uptrend = pd.Series(True, index=idx)

        weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)

        # Long equity during TOM in uptrend
        weights.loc[tom_mask & uptrend, c.equity_ticker] = c.leverage_tom

        # Safe haven: TLT outside TOM or in downtrend
        safe_mask = ~(tom_mask & uptrend)
        if "TLT" in prices.columns:
            weights.loc[safe_mask, "TLT"] = 0.5

        return weights.fillna(0.0)


# =========================================================================
# N3 -- Pre-Holiday Drift
# =========================================================================

@dataclass
class PreHolidayConfig:
    """Long equities before major US market holidays."""

    equity_ticker: str = "SPY"

    # Days before a holiday to be long
    days_before: int = 3

    leverage: float = 2.0
    flat_leverage: float = 0.0

    # Major US holidays (month, day) - approximate trading day before
    holidays: tuple = (
        (1, 1),    # New Year's Day
        (1, 15),   # MLK Day (approx)
        (2, 19),   # Presidents Day (approx)
        (5, 27),   # Memorial Day (approx)
        (7, 4),    # Independence Day
        (9, 2),    # Labor Day (approx)
        (11, 28),  # Thanksgiving (approx)
        (12, 25),  # Christmas
    )


class PreHolidayDrift(Strategy):
    """Long equities in the days before major US holidays.

    Thesis: Pre-holiday drift is one of the most robust calendar
    anomalies. Markets tend to rise in the 1-2 days before major
    holidays, driven by short-covering, positive sentiment, and
    reduced selling pressure. Documented by Ariel (1990) and others.
    """

    name = "N3-PreHolidayDrift"

    def __init__(self, config: PreHolidayConfig | None = None) -> None:
        self.cfg = config or PreHolidayConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        idx = prices.index

        # Build a mask of pre-holiday windows
        pre_holiday = pd.Series(False, index=idx)

        for year in range(idx.year.min(), idx.year.max() + 1):
            for month, day in c.holidays:
                # Find the closest trading day on or before the holiday
                try:
                    holiday_date = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue

                # Find trading days before this holiday
                before = idx[idx < holiday_date]
                if len(before) >= c.days_before:
                    pre_days = before[-c.days_before:]
                    pre_holiday.loc[pre_days] = True

        # Trend filter: SPY above 200d SMA
        if c.equity_ticker in prices.columns:
            sma = prices[c.equity_ticker].rolling(200).mean()
            uptrend = (prices[c.equity_ticker] > sma).fillna(False)
        else:
            uptrend = pd.Series(True, index=idx)

        weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)

        # Long equity before holidays in uptrend
        weights.loc[pre_holiday & uptrend, c.equity_ticker] = c.leverage

        # Moderate equity allocation outside pre-holiday windows in uptrend
        other_uptrend = ~pre_holiday & uptrend
        weights.loc[other_uptrend, c.equity_ticker] = 0.3

        return weights.fillna(0.0)
