"""Category MF: Multi-Frequency strategies.

Strategies that combine signals from multiple timeframes (daily, weekly, monthly)
to capture alpha at different horizons while reducing noise and turnover.

Academic basis:
- Moskowitz, Ooi, Pedersen (2012): Trend-following across timeframes
- Baltas & Kosowski (2013): Multi-frequency momentum
- Asness et al. (2014): Fact, Fiction and Momentum Crashes -- slower signals reduce crash risk
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from financial_algo.indicators import realized_vol, ema
from financial_algo.strategies.base import Strategy


# ---------------------------------------------------------------------------
# Helpers: resample daily prices to weekly / monthly
# ---------------------------------------------------------------------------

def _resample_weekly(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to weekly (Friday close), forward-fill to daily."""
    weekly = prices.resample("W-FRI").last()
    return weekly.reindex(prices.index, method="ffill")


def _resample_monthly(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample daily prices to month-end close, forward-fill to daily."""
    monthly = prices.resample("ME").last()
    return monthly.reindex(prices.index, method="ffill")


# =========================================================================
# MF1 -- Weekly Momentum Rotation
# =========================================================================

@dataclass
class WeeklyMomConfig:
    """Config for weekly momentum rotation."""

    tickers: list[str] | None = None
    lookback_weeks: int = 13   # ~3 months of weekly returns
    skip_weeks: int = 1        # skip most recent week (reversal avoidance)
    top_n: int = 4             # long top N
    trend_window: int = 40     # ~40 weeks (~200 daily) SMA on weekly prices
    leverage: float = 1.5
    max_weight: float = 0.40

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = [
                "SPY", "QQQ", "IWM", "EFA", "EEM",
                "XLK", "XLF", "XLE", "XLV", "GLD", "TLT",
            ]


class WeeklyMomentumRotation(Strategy):
    """Cross-sectional momentum computed on weekly returns, rebalanced weekly.

    Thesis: Weekly-frequency momentum captures the same return continuation
    effect as daily momentum but with dramatically lower turnover and less
    susceptibility to daily noise and whipsaws. The 13-week (quarterly)
    lookback is the sweet spot per Baltas & Kosowski (2013).

    Signal: Rank assets by 13-1 week returns. Long top-N that are also
    above their 40-week SMA. Equal-weight within the long basket.
    Rebalance weekly (positions held constant within each week).
    """

    name = "MF1-WeeklyMomentumRotation"

    def __init__(self, config: WeeklyMomConfig | None = None) -> None:
        self.cfg = config or WeeklyMomConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        p = prices[avail]

        # Resample to weekly
        weekly_p = p.resample("W-FRI").last().dropna(how="all")

        # Weekly momentum: 13-week return skipping most recent week
        lookback_days = c.lookback_weeks  # already in weeks since data is weekly
        skip_days = c.skip_weeks
        weekly_ret = weekly_p.pct_change(lookback_days).shift(skip_days)

        # Weekly trend filter: price above 40-week SMA
        weekly_sma = weekly_p.rolling(c.trend_window).mean()
        weekly_trend_up = weekly_p > weekly_sma

        # Filter: only rank assets in uptrend
        filtered_ret = weekly_ret.where(weekly_trend_up, np.nan)

        # Rank and pick top-N
        ranks = filtered_ret.rank(axis=1, ascending=False)
        long_mask = (ranks <= c.top_n) & pd.notna(filtered_ret)

        n_long = long_mask.sum(axis=1).clip(lower=1)
        weekly_weights = pd.DataFrame(0.0, index=weekly_p.index, columns=avail)
        weekly_weights[long_mask] = 1.0
        weekly_weights = weekly_weights.div(n_long, axis=0) * c.leverage
        weekly_weights = weekly_weights.clip(0, c.max_weight)

        # Forward-fill weekly weights to daily frequency
        daily_weights = weekly_weights.reindex(prices.index, method="ffill").fillna(0.0)

        return daily_weights.reindex(columns=prices.columns, fill_value=0.0).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)


# =========================================================================
# MF2 -- Monthly Macro Regime Allocation
# =========================================================================

@dataclass
class MonthlyMacroConfig:
    """Config for monthly macro regime strategy."""

    equity_ticker: str = "SPY"
    alt_equity: str = "QQQ"
    safe_bond: str = "TLT"
    safe_gold: str = "GLD"
    credit_risky: str = "HYG"
    credit_safe: str = "LQD"
    dollar: str = "UUP"
    duration_short: str = "IEF"

    # Macro signal windows (in months)
    yield_curve_window: int = 3   # months of TLT/IEF ratio trend
    credit_window: int = 3        # months of HYG/LQD ratio trend
    dollar_window: int = 6        # months of UUP momentum

    # Allocation
    risk_on_leverage: float = 1.8
    risk_off_leverage: float = 0.3
    neutral_leverage: float = 1.0
    safe_bond_risk_off: float = 0.4
    safe_gold_risk_off: float = 0.2


class MonthlyMacroRegime(Strategy):
    """Monthly macro regime allocation using yield curve, credit, and dollar signals.

    Thesis: Macro factors (yield curve, credit spreads, USD strength) predict
    equity returns at monthly horizons. When all three signal risk-on (steepening
    curve, tightening credit spreads, weakening dollar), increase equity exposure.
    When they signal risk-off, rotate to TLT+GLD.

    Rebalanced monthly -- holds positions constant within each month.
    """

    name = "MF2-MonthlyMacroRegime"

    def __init__(self, config: MonthlyMacroConfig | None = None) -> None:
        self.cfg = config or MonthlyMacroConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Check available tickers
        has_curve = c.safe_bond in prices.columns and c.duration_short in prices.columns
        has_credit = c.credit_risky in prices.columns and c.credit_safe in prices.columns
        has_dollar = c.dollar in prices.columns
        has_equity = c.equity_ticker in prices.columns

        if not has_equity:
            return weights

        # Resample to monthly
        monthly_p = prices.resample("ME").last().dropna(how="all")

        # --- Signal 1: Yield curve (TLT/IEF ratio change) ---
        # Rising TLT/IEF = steepening (risk-on for equities)
        if has_curve:
            curve_ratio = monthly_p[c.safe_bond] / monthly_p[c.duration_short].replace(0, np.nan)
            curve_mom = curve_ratio.pct_change(c.yield_curve_window).fillna(0.0)
            curve_signal = (curve_mom > 0).astype(float)  # 1 if steepening
        else:
            curve_signal = pd.Series(0.5, index=monthly_p.index)

        # --- Signal 2: Credit spread (HYG/LQD ratio change) ---
        # Rising HYG/LQD = tightening spreads (risk-on)
        if has_credit:
            credit_ratio = monthly_p[c.credit_risky] / monthly_p[c.credit_safe].replace(0, np.nan)
            credit_mom = credit_ratio.pct_change(c.credit_window).fillna(0.0)
            credit_signal = (credit_mom > 0).astype(float)
        else:
            credit_signal = pd.Series(0.5, index=monthly_p.index)

        # --- Signal 3: Dollar momentum (inverse: weak dollar = risk-on) ---
        if has_dollar:
            dollar_mom = monthly_p[c.dollar].pct_change(c.dollar_window).fillna(0.0)
            dollar_signal = (dollar_mom < 0).astype(float)  # weak dollar = risk-on
        else:
            dollar_signal = pd.Series(0.5, index=monthly_p.index)

        # Composite: average of 3 signals (0 = full risk-off, 1 = full risk-on)
        macro_score = (curve_signal + credit_signal + dollar_signal) / 3.0

        # Map to regime: >0.6 risk-on, <0.4 risk-off, else neutral
        risk_on = macro_score > 0.6
        risk_off = macro_score < 0.4
        neutral = ~risk_on & ~risk_off

        # Monthly target weights
        monthly_weights = pd.DataFrame(0.0, index=monthly_p.index, columns=prices.columns)

        # Risk-on: heavy equity
        eq_split = 0.7 if c.alt_equity in prices.columns else 1.0
        alt_split = 0.3 if c.alt_equity in prices.columns else 0.0

        monthly_weights.loc[risk_on, c.equity_ticker] = c.risk_on_leverage * eq_split
        if alt_split > 0:
            monthly_weights.loc[risk_on, c.alt_equity] = c.risk_on_leverage * alt_split

        # Risk-off: safe havens
        if c.safe_bond in prices.columns:
            monthly_weights.loc[risk_off, c.safe_bond] = c.safe_bond_risk_off
        if c.safe_gold in prices.columns:
            monthly_weights.loc[risk_off, c.safe_gold] = c.safe_gold_risk_off
        monthly_weights.loc[risk_off, c.equity_ticker] = c.risk_off_leverage * eq_split

        # Neutral: moderate equity
        monthly_weights.loc[neutral, c.equity_ticker] = c.neutral_leverage * eq_split
        if alt_split > 0:
            monthly_weights.loc[neutral, c.alt_equity] = c.neutral_leverage * alt_split

        # Forward-fill monthly weights to daily
        daily_weights = monthly_weights.reindex(prices.index, method="ffill").fillna(0.0)

        return daily_weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# MF3 -- Multi-Timeframe Trend Consensus
# =========================================================================

@dataclass
class MultiTFTrendConfig:
    """Config for multi-timeframe trend strategy."""

    equity_ticker: str = "SPY"
    alt_equity: str = "QQQ"
    safe_bond: str = "TLT"
    safe_gold: str = "GLD"

    # Trend windows (daily bars)
    daily_window: int = 20     # ~1 month
    weekly_window: int = 10    # 10 weeks = ~50 days
    monthly_window: int = 10   # 10 months = ~210 days

    # Leverage by consensus level
    leverage_all_up: float = 1.6     # daily+weekly+monthly all bullish
    leverage_two_up: float = 0.8     # 2 of 3 bullish
    leverage_one_up: float = 0.0     # only 1 bullish -- flat equity
    leverage_none_up: float = 0.0    # all bearish -- hedge

    safe_bond_weight: float = 0.5    # when <2 bull signals
    safe_gold_weight: float = 0.3    # when <2 bull signals


class MultiTimeframeTrend(Strategy):
    """Multi-timeframe trend consensus: long only when daily+weekly+monthly agree.

    Thesis: Single-timeframe trend signals are noisy. Requiring consensus
    across daily (20-bar), weekly (10-week), and monthly (10-month)
    timeframes dramatically reduces whipsaws and false signals. When all
    three timeframes are bullish, this is a high-conviction long. When
    they disagree, reduce exposure. When mostly bearish, rotate to safe havens.

    Lower turnover than any single-timeframe strategy because it takes
    agreement across slow and fast signals to trigger position changes.
    """

    name = "MF3-MultiTimeframeTrend"

    def __init__(self, config: MultiTFTrendConfig | None = None) -> None:
        self.cfg = config or MultiTFTrendConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.equity_ticker not in prices.columns:
            return weights

        eq = prices[c.equity_ticker]

        # --- Daily trend: price > 20-day SMA ---
        daily_sma = eq.rolling(c.daily_window).mean()
        daily_up = (eq > daily_sma).astype(int)

        # --- Weekly trend: weekly price > 10-week SMA ---
        weekly_p = eq.resample("W-FRI").last().dropna()
        weekly_sma = weekly_p.rolling(c.weekly_window).mean()
        weekly_up_raw = (weekly_p > weekly_sma).astype(int)
        # Forward-fill to daily
        weekly_up = weekly_up_raw.reindex(prices.index, method="ffill").fillna(0).astype(int)

        # --- Monthly trend: monthly price > 10-month SMA ---
        monthly_p = eq.resample("ME").last().dropna()
        monthly_sma = monthly_p.rolling(c.monthly_window).mean()
        monthly_up_raw = (monthly_p > monthly_sma).astype(int)
        monthly_up = monthly_up_raw.reindex(prices.index, method="ffill").fillna(0).astype(int)

        # Consensus score: 0 to 3
        consensus = daily_up + weekly_up + monthly_up

        # Map consensus to leverage
        all_up = consensus == 3
        two_up = consensus == 2
        one_up = consensus == 1
        none_up = consensus == 0

        # Equity allocation
        eq_split = 0.7 if c.alt_equity in prices.columns else 1.0
        alt_split = 0.3 if c.alt_equity in prices.columns else 0.0

        weights.loc[all_up, c.equity_ticker] = c.leverage_all_up * eq_split
        if alt_split > 0:
            weights.loc[all_up, c.alt_equity] = c.leverage_all_up * alt_split

        weights.loc[two_up, c.equity_ticker] = c.leverage_two_up * eq_split
        if alt_split > 0:
            weights.loc[two_up, c.alt_equity] = c.leverage_two_up * alt_split

        # One or zero up: safe haven rotation
        bearish = one_up | none_up
        if c.safe_bond in prices.columns:
            weights.loc[bearish, c.safe_bond] = c.safe_bond_weight
        if c.safe_gold in prices.columns:
            weights.loc[bearish, c.safe_gold] = c.safe_gold_weight

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# MF4 -- Weekly Sector Mean Reversion
# =========================================================================

@dataclass
class WeeklyMeanRevConfig:
    """Config for weekly sector mean reversion."""

    tickers: list[str] | None = None
    benchmark: str = "SPY"
    zscore_window: int = 20  # 20 weeks = ~5 months
    entry_z: float = -1.2    # enter when z-score below this
    exit_z: float = 0.0      # exit near zero
    max_weight: float = 0.25
    leverage: float = 1.0
    max_positions: int = 3

    def __post_init__(self) -> None:
        if self.tickers is None:
            self.tickers = [
                "XLK", "XLF", "XLI", "XLB", "XLP",
                "XLU", "XLY", "XLV", "XLE",
            ]


class WeeklyMeanReversion(Strategy):
    """Weekly z-score mean reversion on sector/SPY ratios.

    Thesis: Sector rotation is mean-reverting at the weekly-to-monthly
    horizon. Sectors that have underperformed SPY over 6 months
    (z-score < -1.5) tend to revert. Weekly resampling reduces noise
    compared to daily mean-reversion signals.

    Signal: Compute weekly relative price (sector/SPY), calculate
    26-week z-score. Long oversold sectors (z < -1.5). Exit when z
    reverts to 0. Rebalance weekly.
    """

    name = "MF4-WeeklyMeanReversion"

    def __init__(self, config: WeeklyMeanRevConfig | None = None) -> None:
        self.cfg = config or WeeklyMeanRevConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if not avail or c.benchmark not in prices.columns:
            return weights

        p = prices[avail]
        bench = prices[c.benchmark]

        # Resample to weekly
        weekly_p = p.resample("W-FRI").last().dropna(how="all")
        weekly_bench = bench.resample("W-FRI").last().dropna()

        # Relative price ratio (sector / SPY)
        rel = weekly_p.div(weekly_bench, axis=0)

        # Z-score of relative price over rolling window
        rel_mean = rel.rolling(c.zscore_window).mean()
        rel_std = rel.rolling(c.zscore_window).std().replace(0, np.nan)
        z = (rel - rel_mean) / rel_std
        z = z.fillna(0.0)

        # Trading signal: long oversold (z < entry), hold until z > exit
        # Vectorized: simple threshold-based (not stateful hold)
        # Use graded sizing: more negative z = larger position
        raw_signal = (-z - abs(c.entry_z)) / 1.0  # linear beyond threshold
        raw_signal = raw_signal.clip(lower=0.0, upper=1.0)

        # Limit to max_positions
        n_active = (raw_signal > 0).sum(axis=1)
        scale = (c.max_positions / n_active.clip(lower=1)).clip(upper=1.0)

        weekly_weights = raw_signal.mul(scale, axis=0) * c.leverage
        weekly_weights = weekly_weights.clip(0, c.max_weight)

        # Forward-fill to daily
        daily_weights = weekly_weights.reindex(prices.index, method="ffill").fillna(0.0)

        return daily_weights.reindex(columns=prices.columns, fill_value=0.0).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(0.0)
