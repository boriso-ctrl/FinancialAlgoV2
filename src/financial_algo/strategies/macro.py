"""Category M: FX & Macro strategies.

Dollar and gold macro trades using UUP and GLD as instruments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import ema
from financial_algo.strategies.base import Strategy


# =========================================================================
# M1 — Dollar Carry / Risk-On Risk-Off
# =========================================================================

@dataclass
class DollarCarryConfig:
    """Config for dollar carry trade."""

    dollar_ticker: str = "UUP"
    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"

    # Momentum of UUP as risk-on/off proxy
    momentum_window: int = 63   # 3-month dollar momentum
    ema_span: int = 21

    leverage_risk_on: float = 1.5    # long equity when dollar weakening
    leverage_risk_off: float = 0.0   # flat equity when dollar strengthening
    safe_weight: float = 0.5         # TLT allocation in risk-off


class DollarCarry(Strategy):
    """Use dollar momentum as a risk-on/risk-off regime signal.

    Thesis: A weakening dollar (UUP declining) signals risk appetite
    and global growth, favoring equities. A strengthening dollar signals
    risk aversion. Classic macro trade used by global macro funds.
    """

    name = "M1-DollarCarry"

    def __init__(self, config: DollarCarryConfig | None = None) -> None:
        self.cfg = config or DollarCarryConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.dollar_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        uup = prices[c.dollar_ticker]
        uup_ema = ema(uup, c.ema_span)

        # Dollar momentum: negative = dollar weakening = risk on
        dollar_mom = uup_ema.pct_change(c.momentum_window)
        dollar_weakening = dollar_mom < 0

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights.loc[dollar_weakening, c.equity_ticker] = c.leverage_risk_on
        weights.loc[~dollar_weakening, c.safe_ticker] = c.safe_weight

        return weights.fillna(0.0)


# =========================================================================
# M2 — Gold / Dollar Inverse Trade
# =========================================================================

@dataclass
class GoldDollarConfig:
    """Config for gold/dollar inverse trade."""

    gold_ticker: str = "GLD"
    dollar_ticker: str = "UUP"

    momentum_window: int = 42    # 2-month momentum
    ema_span: int = 10

    leverage: float = 1.5


class GoldDollarInverse(Strategy):
    """Long gold when dollar weakens, long dollar when gold weakens.

    Thesis: Gold and dollar have a well-documented inverse relationship.
    When dollar weakens, gold tends to appreciate (and vice versa).
    This exploits the structural negative correlation for steady returns.
    """

    name = "M2-GoldDollarInverse"

    def __init__(self, config: GoldDollarConfig | None = None) -> None:
        self.cfg = config or GoldDollarConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.gold_ticker not in prices.columns or c.dollar_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        gold = ema(prices[c.gold_ticker], c.ema_span)
        dollar = ema(prices[c.dollar_ticker], c.ema_span)

        gold_mom = gold.pct_change(c.momentum_window)
        dollar_mom = dollar.pct_change(c.momentum_window)

        # Relative momentum: long the one with better momentum
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        gold_stronger = gold_mom > dollar_mom
        weights.loc[gold_stronger, c.gold_ticker] = c.leverage
        weights.loc[~gold_stronger, c.dollar_ticker] = c.leverage

        return weights.fillna(0.0)


# =========================================================================
# M3 -- EM Risk Premium (with credit spread filter)
# =========================================================================

@dataclass
class EMRiskPremiumConfig:
    """Long EM equities when credit spreads are tightening (risk-on)."""

    em_ticker: str = "EEM"
    safe_ticker: str = "IEF"

    # Credit spread proxy: HYG/LQD ratio
    hyg_ticker: str = "HYG"
    lqd_ticker: str = "LQD"

    spread_momentum: int = 42   # 2-month momentum of credit spread ratio
    trend_window: int = 200     # 200-day trend filter on EEM

    leverage: float = 1.0


class EMRiskPremium(Strategy):
    """Long EEM when credit spreads tighten, IEF when widening.

    Thesis: EM equities carry a risk premium tied to global credit
    conditions. When HYG/LQD ratio is rising (spreads tightening),
    risk appetite improves -- long EEM. When ratio falls (spreads
    widening), rotate to IEF. Long-only, no shorts.
    """

    name = "M3-EMRiskPremium"

    def __init__(self, config: EMRiskPremiumConfig | None = None) -> None:
        self.cfg = config or EMRiskPremiumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.em_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Credit spread momentum
        if c.hyg_ticker in prices.columns and c.lqd_ticker in prices.columns:
            spread_ratio = prices[c.hyg_ticker] / prices[c.lqd_ticker]
            spread_ratio = spread_ratio.replace([np.inf, -np.inf], np.nan).ffill()
            spread_mom = spread_ratio.pct_change(c.spread_momentum).fillna(0.0)
            credit_improving = pd.notna(spread_mom) & (spread_mom > 0)
        else:
            # Fallback: simple EEM momentum
            em_mom = prices[c.em_ticker].pct_change(c.spread_momentum).fillna(0.0)
            credit_improving = pd.notna(em_mom) & (em_mom > 0)

        # Trend filter: EEM above 200-day SMA
        eem_sma = prices[c.em_ticker].rolling(c.trend_window, min_periods=100).mean()
        eem_above_trend = prices[c.em_ticker] >= eem_sma

        # Long EEM when credit improving AND EEM above trend
        long_eem = credit_improving & eem_above_trend

        weights[c.em_ticker] = np.where(long_eem, c.leverage, 0.0)
        safe_col = c.safe_ticker if c.safe_ticker in prices.columns else c.em_ticker
        weights[safe_col] = np.where(long_eem, 0.0, c.leverage)

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# M4 -- Commodity Momentum Basket
# =========================================================================

@dataclass
class CommodityMomConfig:
    """Long commodity ETFs with positive momentum above trend."""

    tickers: tuple = ("XLE", "GLD")
    lookback: int = 126        # 6-month momentum
    trend_window: int = 200    # 200-day SMA trend filter
    safe_ticker: str = "IEF"   # Park cash when no signal
    leverage: float = 1.0


class CommodityMomentum(Strategy):
    """Long XLE/GLD when they have positive 6-month momentum + above 200d SMA.

    Thesis: Commodities exhibit strong momentum driven by supply/demand
    cycles. Only go long when both momentum is positive AND price is
    above its 200-day SMA. Long-only, no shorts. Equal-weight qualifying
    assets; park in IEF when nothing qualifies.
    """

    name = "M4-CommodityMomentum"

    def __init__(self, config: CommodityMomConfig | None = None) -> None:
        self.cfg = config or CommodityMomConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if not avail:
            return weights

        p = prices[avail]

        # 6-month momentum
        mom = p.pct_change(c.lookback).fillna(0.0)
        # 200-day SMA trend filter
        sma200 = p.rolling(c.trend_window, min_periods=100).mean()

        # Qualify: positive momentum AND above 200d SMA
        qualify = (mom > 0) & (p >= sma200)

        # Count qualifying assets per day
        n_qualify = qualify.sum(axis=1).replace(0, np.nan)

        # Equal-weight qualifying assets
        for t in avail:
            weights[t] = np.where(
                qualify[t],
                c.leverage / n_qualify.fillna(1.0),
                0.0,
            )

        # Park in safe asset when nothing qualifies
        nothing_qualifies = qualify.sum(axis=1) == 0
        safe_col = c.safe_ticker if c.safe_ticker in prices.columns else avail[0]
        weights[safe_col] = np.where(
            nothing_qualifies,
            c.leverage,
            weights[safe_col],
        )

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# M5 — Rates Regime Trade
# =========================================================================

@dataclass
class RatesRegimeConfig:
    """Config for rates regime trade."""

    equity_ticker: str = "SPY"
    bond_ticker: str = "TLT"

    equity_trend_window: int = 200  # SPY 200-day SMA
    bond_momentum: int = 63         # 3-month TLT momentum
    leverage: float = 1.0


class RatesRegimeTrade(Strategy):
    """Rates regime indicator: flight-to-quality vs growth.

    Thesis: When SPY is below its 200-day SMA, risk is elevated --
    go long TLT (flight to quality, rate cuts expected). When TLT
    has negative 3-month momentum (rates rising = strong growth),
    go long SPY. Always invested, rotates between risk-on and
    risk-off assets based on rate regime.
    """

    name = "M5-RatesRegimeTrade"

    def __init__(self, config: RatesRegimeConfig | None = None) -> None:
        self.cfg = config or RatesRegimeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.equity_ticker not in prices.columns or c.bond_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        spy = prices[c.equity_ticker]
        tlt = prices[c.bond_ticker]

        # SPY below 200d SMA = risk-off -> long TLT
        spy_sma = spy.rolling(c.equity_trend_window, min_periods=100).mean()
        spy_below_trend = spy < spy_sma

        # TLT negative 3-month momentum = rates rising = growth -> long SPY
        tlt_mom = tlt.pct_change(c.bond_momentum).fillna(0.0)
        rates_rising = pd.notna(tlt_mom) & (tlt_mom < 0)

        # Priority: risk-off (SPY below trend) overrides growth signal
        long_tlt = spy_below_trend
        long_spy = ~spy_below_trend & rates_rising
        # Default: when SPY above trend and TLT momentum positive, stay in SPY
        long_spy_default = ~spy_below_trend & ~rates_rising

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        weights[c.bond_ticker] = np.where(long_tlt, c.leverage, 0.0)
        weights[c.equity_ticker] = np.where(
            long_spy | long_spy_default, c.leverage, 0.0,
        )

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
