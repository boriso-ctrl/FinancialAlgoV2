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

    spread_momentum: int = 21   # 1-month momentum of credit spread ratio
    trend_window: int = 100     # 100-day trend filter on EEM
    em_momentum: int = 63       # 3-month EEM momentum fallback

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
            credit_improving = pd.Series(False, index=prices.index)

        # EEM momentum fallback — own asset momentum as secondary signal
        em_mom = prices[c.em_ticker].pct_change(c.em_momentum).fillna(0.0)
        em_positive = pd.notna(em_mom) & (em_mom > 0)

        # Trend filter: EEM above 100-day SMA (shorter = faster reaction)
        eem_sma = prices[c.em_ticker].rolling(c.trend_window, min_periods=50).mean()
        eem_above_trend = prices[c.em_ticker] >= eem_sma

        # Long EEM when (credit improving OR em momentum positive) AND above trend
        long_eem = (credit_improving | em_positive) & eem_above_trend

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


# =========================================================================
# M6 -- Inflation Breakeven Trade
# =========================================================================

@dataclass
class InflationBreakevenConfig:
    """Long TIP / short IEF when breakeven inflation is rising."""

    tip_ticker: str = "TIP"
    ief_ticker: str = "IEF"

    # TIP/IEF ratio momentum as breakeven proxy
    momentum_window: int = 21   # 1-month momentum of ratio
    slow_momentum: int = 63     # 3-month momentum for confirmation
    rebalance_days: int = 21    # monthly rebalance

    long_weight: float = 1.0    # weight on long leg
    short_weight: float = -0.5  # weight on short leg (partial hedge)


class InflationBreakevenTrade(Strategy):
    """Long TIP / short IEF when breakeven inflation is rising.

    Thesis: The TIP/IEF ratio proxies breakeven inflation expectations.
    When the ratio has positive momentum (breakevens rising), inflation
    expectations are increasing -- long TIPS, short nominal Treasuries.
    When the ratio is falling (disinflation), reverse the trade.
    Monthly rebalance to reduce turnover.
    """

    name = "M6-InflationBreakevenTrade"

    def __init__(self, config: InflationBreakevenConfig | None = None) -> None:
        self.cfg = config or InflationBreakevenConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.tip_ticker not in prices.columns or c.ief_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Breakeven proxy: TIP/IEF ratio
        tip = prices[c.tip_ticker]
        ief = prices[c.ief_ticker]
        ratio = tip / ief
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)

        # Fast and slow momentum of the ratio
        fast_mom = ratio.pct_change(c.momentum_window).fillna(0.0)
        slow_mom = ratio.pct_change(c.slow_momentum).fillna(0.0)

        # Signal: rising breakevens (both fast and slow positive)
        inflation_rising = (fast_mom > 0) & (slow_mom > 0)
        inflation_falling = (fast_mom < 0) & (slow_mom < 0)

        # Monthly rebalance mask: only change positions on rebalance days
        bdays = np.arange(len(prices))
        rebal_mask = (bdays % c.rebalance_days) == 0

        # Build raw signal, then forward-fill between rebalance dates
        raw_signal = pd.Series(np.nan, index=prices.index)
        raw_signal.iloc[rebal_mask & inflation_rising.values] = 1.0
        raw_signal.iloc[rebal_mask & inflation_falling.values] = -1.0
        raw_signal.iloc[
            rebal_mask & ~inflation_rising.values & ~inflation_falling.values
        ] = 0.0
        raw_signal = raw_signal.ffill().fillna(0.0)

        # Position: long TIP / short IEF when inflation rising
        #           long IEF / short TIP when inflation falling
        weights[c.tip_ticker] = np.where(
            raw_signal > 0, c.long_weight,
            np.where(raw_signal < 0, c.short_weight, 0.0),
        )
        weights[c.ief_ticker] = np.where(
            raw_signal > 0, c.short_weight,
            np.where(raw_signal < 0, c.long_weight, 0.0),
        )

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# M7 -- Global Rotation
# =========================================================================

@dataclass
class GlobalRotationConfig:
    """Rotate among global equity regions by 6-month momentum + vol scaling."""

    region_tickers: tuple = ("SPY", "VGK", "EWJ", "FXI", "INDA", "EEM")
    safe_tickers: tuple = ("TLT", "GLD")

    momentum_window: int = 126    # 6-month momentum
    vol_window: int = 63          # 3-month vol for scaling
    top_n: int = 3                # pick top-N regions by momentum
    target_vol: float = 0.15      # target annualised vol per position
    leverage: float = 1.0


class GlobalRotation(Strategy):
    """Rotate among global equity regions by 6-month momentum with vol scaling.

    Thesis: International equity returns exhibit persistent momentum at
    the 6-month horizon driven by macro cycles, capital flows, and
    relative monetary policy. Rotate into the strongest regions, vol-scale
    for equal risk contribution. Go to TLT+GLD when all regions have
    negative momentum (global risk-off). Long-only.
    """

    name = "M7-GlobalRotation"

    def __init__(self, config: GlobalRotationConfig | None = None) -> None:
        self.cfg = config or GlobalRotationConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail = [t for t in c.region_tickers if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        p = prices[avail]

        # 6-month momentum
        mom = p.pct_change(c.momentum_window).fillna(0.0)

        # Annualised volatility for scaling
        daily_ret = p.pct_change().fillna(0.0)
        ann_vol = daily_ret.rolling(c.vol_window, min_periods=20).std() * np.sqrt(252)
        ann_vol = ann_vol.replace(0, np.nan).fillna(c.target_vol)

        # Vol-scaled weight factor
        vol_scale = (c.target_vol / ann_vol).clip(upper=3.0)

        # Rank by momentum, pick top-N with positive momentum
        mom_rank = mom.rank(axis=1, ascending=False)
        positive_mom = mom > 0
        selected = (mom_rank <= c.top_n) & positive_mom

        # Count of selected per day
        n_selected = selected.sum(axis=1).replace(0, np.nan)

        # Equal-weight among selected, scaled by vol
        for t in avail:
            raw_w = np.where(
                selected[t],
                c.leverage * vol_scale[t] / n_selected.fillna(1.0),
                0.0,
            )
            weights[t] = raw_w

        # Cap total leverage at c.leverage
        total_w = weights[avail].sum(axis=1)
        scale_down = np.where(
            pd.notna(total_w) & (total_w > c.leverage),
            c.leverage / total_w.replace(0, 1.0),
            1.0,
        )
        for t in avail:
            weights[t] = weights[t] * scale_down

        # Safe-haven allocation when ALL regions have negative momentum
        all_negative = (mom <= 0).all(axis=1)
        safe_avail = [t for t in c.safe_tickers if t in prices.columns]
        if safe_avail:
            safe_w = c.leverage / len(safe_avail)
            for t in safe_avail:
                weights[t] = np.where(all_negative, safe_w, weights[t])
            # Zero out region weights on all-negative days
            for t in avail:
                weights[t] = np.where(all_negative, 0.0, weights[t])

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# M8 -- Commodity Macro Signal
# =========================================================================

@dataclass
class CommodityMacroConfig:
    """Use commodity momentum as a macro signal for asset allocation."""

    dbc_ticker: str = "DBC"
    dba_ticker: str = "DBA"
    slv_ticker: str = "SLV"
    gld_ticker: str = "GLD"

    risk_on_tickers: tuple = ("SPY", "EEM")
    safe_tickers: tuple = ("TLT", "GLD", "SLV")

    momentum_window: int = 63    # 3-month commodity momentum
    trend_window: int = 200      # 200-day SMA trend filter
    slv_gld_window: int = 42     # 2-month SLV/GLD ratio momentum

    leverage: float = 1.0


class CommodityMacroSignal(Strategy):
    """Use commodity momentum (DBC + DBA + SLV/GLD) as a macro allocation signal.

    Thesis: Broad commodity strength (DBC, DBA rising) signals reflationary
    growth -- overweight SPY + EEM. A rising SLV/GLD ratio also confirms
    industrial demand. When commodity signals are negative (DBC+DBA falling),
    rotate defensively into TLT + GLD + SLV. This captures the macro
    regime signal embedded in commodity markets.
    """

    name = "M8-CommodityMacroSignal"

    def __init__(self, config: CommodityMacroConfig | None = None) -> None:
        self.cfg = config or CommodityMacroConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # --- Commodity momentum signals ---
        signals = []

        if c.dbc_ticker in prices.columns:
            dbc_mom = prices[c.dbc_ticker].pct_change(c.momentum_window).fillna(0.0)
            signals.append(dbc_mom > 0)

        if c.dba_ticker in prices.columns:
            dba_mom = prices[c.dba_ticker].pct_change(c.momentum_window).fillna(0.0)
            signals.append(dba_mom > 0)

        # SLV/GLD ratio momentum -- rising = industrial demand
        if c.slv_ticker in prices.columns and c.gld_ticker in prices.columns:
            slv_gld = prices[c.slv_ticker] / prices[c.gld_ticker]
            slv_gld = slv_gld.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            slv_gld_mom = slv_gld.pct_change(c.slv_gld_window).fillna(0.0)
            signals.append(slv_gld_mom > 0)

        if not signals:
            return weights

        # Composite score: fraction of signals that are positive
        signal_stack = pd.concat(signals, axis=1)
        composite = signal_stack.sum(axis=1) / len(signals)

        # Risk-on: majority of commodity signals positive (>= 0.5)
        risk_on = composite >= 0.5

        # Trend filter on DBC (if available) for confirmation
        if c.dbc_ticker in prices.columns:
            dbc_sma = prices[c.dbc_ticker].rolling(
                c.trend_window, min_periods=60,
            ).mean()
            dbc_above_trend = prices[c.dbc_ticker] >= dbc_sma
            risk_on_confirmed = risk_on & dbc_above_trend
        else:
            risk_on_confirmed = risk_on

        # Allocate to risk-on assets
        risk_on_avail = [t for t in c.risk_on_tickers if t in prices.columns]
        safe_avail = [t for t in c.safe_tickers if t in prices.columns]

        if risk_on_avail:
            w_per = c.leverage / len(risk_on_avail)
            for t in risk_on_avail:
                weights[t] = np.where(risk_on_confirmed, w_per, 0.0)

        if safe_avail:
            w_safe = c.leverage / len(safe_avail)
            for t in safe_avail:
                weights[t] = np.where(~risk_on_confirmed, w_safe, weights[t])

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
