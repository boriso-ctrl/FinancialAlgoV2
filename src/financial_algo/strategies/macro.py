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
    score_smooth_span: int = 8

    leverage_risk_on: float = 1.5    # long equity when dollar weakening
    leverage_risk_off: float = 0.0   # flat equity when dollar strengthening
    safe_weight: float = 0.7         # TLT allocation in risk-off


class DollarCarry(Strategy):
    """Use dollar regime with gradual transitions (not binary) for risk-on/risk-off.

    Thesis: A weakening dollar signals risk appetite; strengthening dollar
    signals risk aversion. But dollar doesn't move in discrete jumps -- it
    trends. Implement three states:
      1. FULL RISK-OFF: dollar 200d SMA + rising 20d momentum (strong dollar)
         → minimize equity, max safe TLT
      2. PARTIAL HEDGE: dollar in transition zone
         → moderate equity + TLT blend
      3. FULL RISK-ON: dollar 200d SMA and falling momentum (weak dollar)
         → max equity, minimize TLT
    
    This reduces whipsaw vs binary thresholds, captures multi-week trends.
    """

    name = "M1-DollarCarry"

    def __init__(self, config: DollarCarryConfig | None = None) -> None:
        self.cfg = config or DollarCarryConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if prices.empty:
            return pd.DataFrame()

        c = self.cfg
        if c.dollar_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        uup = prices[c.dollar_ticker]
        
        # --- Dollar trend signals ---
        # 200-day SMA: structural dollar regime (short-term noise filter)
        uup_sma200 = uup.rolling(200, min_periods=100).mean()
        dollar_above_sma = uup >= uup_sma200
        
        # 20-day momentum: dollar direction (is it rising or falling?)
        uup_mom20 = uup.pct_change(20).fillna(0.0)
        dollar_strengthening = pd.notna(uup_mom20) & (uup_mom20 > 0.005)  # >0.5% in 20 days = rising
        
        # Distance from 200d SMA (pct): how extreme is the current regime?
        # Positive = above SMA (stronger dollar), negative = below SMA (weaker dollar)
        sma_distance_pct = ((uup - uup_sma200) / uup_sma200).clip(-0.08, 0.08)

        # --- Three-state regime scoring ---
        # Risk-off intensity (-1 to 0): stronger dollar = risk aversion
        dollar_risk_off_signal = np.where(dollar_above_sma, 1.0, -1.0)  # +1 = strong, -1 = weak
        dollar_risk_off_signal = dollar_risk_off_signal * (1.0 + sma_distance_pct * 2).clip(0.5, 1.5)
        
        # Momentum acceleration (-1 to 0): rising dollar = increasing risk-off pressure
        momentum_multiplier = np.where(dollar_strengthening, 1.3, 0.7)
        final_risk_off_score = dollar_risk_off_signal * momentum_multiplier

        # Clip to [-1, 1] scale
        final_risk_off_score = np.clip(final_risk_off_score, -1.0, 1.0)

        # --- Allocation logic with hysteresis ---
        # Avoid extreme whipsaw: use smoothed score via simple lag
        risk_off_smooth = pd.Series(final_risk_off_score, index=prices.index).ewm(span=c.score_smooth_span).mean()
        
        # Define regime zones
        # full_risk_off_score > 0.5
        full_risk_off = risk_off_smooth > 0.5
        # full_risk_on_score < -0.5
        full_risk_on = risk_off_smooth < -0.5
        # partial zone: -0.5 to 0.5
        partial_hedge = (~full_risk_off) & (~full_risk_on)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Full risk-off: 100% TLT (safe haven)
        if c.safe_ticker in prices.columns:
            weights[c.safe_ticker] = np.where(full_risk_off, c.safe_weight, 0.0)
        
        # Full risk-on: 150% long equities (risk-on leverage)
        if c.equity_ticker in prices.columns:
            weights[c.equity_ticker] = np.where(full_risk_on, c.leverage_risk_on, 0.0)

        # Partial hedge: blend both (50/50 or adjusted by score)
        # Use the risk_off_smooth score to weight the blend.
        blend_factor = ((risk_off_smooth.clip(-0.5, 0.5) + 0.5) / 1.0).fillna(0.5)
        
        if c.equity_ticker in prices.columns:
            equity_w_partial = (1.0 - blend_factor) * c.leverage_risk_on
            weights[c.equity_ticker] = np.where(
                partial_hedge,
                equity_w_partial,
                weights[c.equity_ticker],
            )
        
        if c.safe_ticker in prices.columns:
            safe_w_partial = blend_factor * c.safe_weight
            weights[c.safe_ticker] = np.where(
                partial_hedge,
                safe_w_partial,
                weights[c.safe_ticker],
            )

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


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
    dollar_ticker: str = "UUP"
    vix_ticker: str = "^VIX"

    spread_momentum: int = 21   # 1-month momentum of credit spread ratio
    credit_health_window: int = 200  # HYG/LQD above 200d SMA = healthy credit
    trend_window: int = 120     # medium-term trend filter on EEM
    em_momentum: int = 63       # 3-month EEM momentum fallback
    uup_momentum: int = 63      # 3-month dollar momentum
    uup_trend_window: int = 200  # structural dollar regime
    vix_threshold: float = 25.0  # reduce position when VIX > this

    leverage: float = 1.0


class EMRiskPremium(Strategy):
    """Long EEM when credit+dollar+equity signals align (multi-signal confluence).

    Thesis: EM equities carry a risk premium tied to global credit, dollar flows, 
    and equity momentum. Trades fail when signals conflict (e.g., spreads tightening 
    but dollar strengthening). Implement signal *consensus* scoring:
      - Must have credit improving (HYG/LQD rising)
      - Must have weak dollar (UUP falling or below SMA)
      - Must have EEM positive momentum
      - All three signals in alignment = high confidence entry
    
    Reduce false positives by requiring 2+ signals before entering.
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

        # *** Signal 1: Credit Spread Health ***
        # Higher confidence when HYG/LQD ratio is RISING and above structural avg
        if c.hyg_ticker in prices.columns and c.lqd_ticker in prices.columns:
            spread_ratio = prices[c.hyg_ticker] / prices[c.lqd_ticker]
            spread_ratio = spread_ratio.replace([np.inf, -np.inf], np.nan).ffill()
            
            # Momentum: is spread ratio improving (rising)?
            spread_mom = spread_ratio.pct_change(c.spread_momentum).fillna(0.0)
            credit_improving = pd.notna(spread_mom) & (spread_mom > 0)

            # Structural health: is ratio above its 200d SMA?
            spread_sma = spread_ratio.rolling(c.credit_health_window, min_periods=60).mean()
            credit_healthy = spread_ratio >= spread_sma
            
            # Combined credit score: +1 for each confirmation
            credit_signal = credit_improving.astype(int) + credit_healthy.astype(int)
        else:
            credit_signal = pd.Series(0, index=prices.index)

        # *** Signal 2: Dollar Weakness ***
        # Weak dollar (UUP falling) is good for EM. Use SMA + momentum combo.
        if c.dollar_ticker in prices.columns:
            uup = prices[c.dollar_ticker]
            
            # Structural weakness: UUP below 200d SMA
            uup_sma = uup.rolling(c.uup_trend_window, min_periods=100).mean()
            dollar_structurally_weak = uup <= uup_sma
            
            # Momentum weakness: UUP falling on 20d basis
            uup_mom = uup.pct_change(20).fillna(0.0)
            dollar_momentum_weak = pd.notna(uup_mom) & (uup_mom < -0.005)  # Falling >0.5%
            
            # Combined dollar score: +1 for each confirmation
            dollar_signal = dollar_structurally_weak.astype(int) + dollar_momentum_weak.astype(int)
        else:
            dollar_signal = pd.Series(0, index=prices.index)

        # *** Signal 3: EEM Momentum ***
        # EEM itself should have positive momentum as tertiary confirmation
        em_mom = prices[c.em_ticker].pct_change(c.em_momentum).fillna(0.0)
        em_positive = pd.notna(em_mom) & (em_mom > 0.0)

        # EEM above 100d SMA (in uptrend, not oversold)
        eem_sma = prices[c.em_ticker].rolling(c.trend_window, min_periods=50).mean()
        eem_above_trend = prices[c.em_ticker] >= eem_sma
        
        # Combined EEM score
        eem_signal = em_positive.astype(int) + eem_above_trend.astype(int)

        # *** Signal 4: VIX Risk Gauge ***
        # During high vol, EM gets hit even if fundamentals improve
        if c.vix_ticker in prices.columns:
            vix = prices[c.vix_ticker]
            high_vol = pd.notna(vix) & (vix > c.vix_threshold)
        else:
            high_vol = pd.Series(False, index=prices.index)

        # *** Consensus Scoring ***
        # Total "alignment score": 0-6 (credit 0-2, dollar 0-2, EEM 0-2)
        total_alignment = credit_signal + dollar_signal + eem_signal
        
        # Confidence tiers:
        # 5-6 signals = FULL_RISK_ON (100% EEM)
        # 3-4 signals = PARTIAL_RISK_ON (50% EEM, rest in safe)
        # 0-2 signals = RISK_OFF (0% EM, go to IEF/safe)
        full_risk_on = (total_alignment >= 5) & (~high_vol)
        partial_risk_on = (total_alignment >= 3) & (total_alignment < 5) & (~high_vol)
        risk_off = high_vol | (total_alignment < 3)

        em_weight = np.where(
            full_risk_on,
            c.leverage,
            np.where(partial_risk_on, 0.5 * c.leverage, 0.0),
        )
        em_weight = pd.Series(em_weight, index=prices.index).fillna(0.0)

        weights[c.em_ticker] = em_weight
        
        # Park in safe asset (IEF or equiv) when not in EEM
        safe_col = c.safe_ticker if c.safe_ticker in prices.columns else None
        safe_weight = (c.leverage - em_weight).clip(lower=0.0)
        
        if safe_col is not None and safe_col != c.em_ticker:
            weights[safe_col] = safe_weight

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
    """Long commodities with positive momentum when real rates trend is favorable.

    Thesis: Commodity momentum is powerful, but breaks down when *real interest rates*
    are rising sharply (hurts carry). Gate long entries with:
      1. Positive 6-month momentum (XLE, GLD)
      2. Price above 200d SMA (structural uptrend)
      3. **Real rates regime**: TLT above 200d SMA OR TLT not falling sharply
         (indicates falling nominal rates → favorable for commodities)
    
    Commodity carry insight: When TLT rises (rates fall), it's often a sign of
    risk-off OR increased real inflation (both help commodities). When TLT falls
    sharply (rates up + real rates compressed), commodity carry becomes less attractive.
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

        # *** Signal 1: Commodity Momentum ***
        # 6-month momentum
        mom = p.pct_change(c.lookback).fillna(0.0)
        # 200-day SMA trend filter
        sma200 = p.rolling(c.trend_window, min_periods=100).mean()

        # Qualify: positive momentum AND above 200d SMA
        qualify = (mom > 0) & (p >= sma200)

        # *** Signal 2: Real Rates Regime Filter ***
        # When nominal bond yields (TLT) are rising sharply, real rates compressing
        # may reduce commodity carry attractiveness. Be conservative.
        # When TLT is falling (yields down), risk-off + potential inflation = good for commodities
        real_rates_favorable = pd.Series(True, index=prices.index)
        
        if "TLT" in prices.columns:
            tlt = prices["TLT"]
            
            # TLT 21d momentum (short-term rate direction)
            tlt_mom_short = tlt.pct_change(21).fillna(0.0)
            # TLT 63d momentum (medium-term rate direction)
            tlt_mom_long = tlt.pct_change(63).fillna(0.0)
            
            # TLT 200d SMA (structural rates regime)
            tlt_sma = tlt.rolling(c.trend_window, min_periods=100).mean()
            tlt_above_sma = tlt >= tlt_sma
            
            # Favorable real rates: TLT not crashing (yields falling) or at least trending sideways above SMA
            # Unfavorable: TLT sharp decline + below SMA (rates rising sharply)
            unfavorable_rates = (tlt_mom_short < -0.02) & (tlt_mom_long < -0.02) & (~tlt_above_sma)
            real_rates_favorable = ~unfavorable_rates

        # *** Signal 3: Volatility/Stress Filter ***
        # During extreme stress (VIX > 35), commodities can spike down despite momentum
        # Use a simple vol proxy: if any commodity is crashing (-5%+ in 1 day), reduce exposure
        extreme_stress = pd.Series(False, index=prices.index)
        
        if "^VIX" in prices.columns:
            vix = prices["^VIX"]
            extreme_stress = pd.notna(vix) & (vix > 35.0)

        # *** Combined Qualification ***
        # Require: positive momentum + above SMA + real rates favorable + no extreme stress
        final_qualify = qualify & real_rates_favorable.values.reshape(-1, 1) & (~extreme_stress.values.reshape(-1, 1))

        # Count qualifying assets per day
        n_qualify = final_qualify.sum(axis=1).replace(0, np.nan)

        # Equal-weight qualifying assets
        for t in avail:
            weights[t] = np.where(
                final_qualify[t],
                c.leverage / n_qualify.fillna(1.0),
                0.0,
            )

        # Park in safe asset when nothing qualifies
        nothing_qualifies = final_qualify.sum(axis=1) == 0
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
# M7 -- Global Rotation
# =========================================================================

@dataclass
class GlobalRotationConfig:
    """Macro-aware regional rotation with explicit defensive sleeve."""

    region_tickers: tuple = ("SPY", "VGK", "EWJ", "FXI", "INDA", "EEM")
    safe_tickers: tuple = ("TLT", "GLD", "UUP")

    momentum_window: int = 126     # 6-month momentum
    fast_momentum: int = 42        # 2-month momentum for regime transitions
    trend_window: int = 200        # long-term trend filter
    drawdown_window: int = 63      # 3-month drawdown check
    top_n: int = 2                 # concentrated winners only
    leverage: float = 1.0

    # Macro filters
    dollar_ticker: str = "UUP"
    bond_ticker: str = "TLT"
    vix_ticker: str = "^VIX"
    vix_risk_off: float = 24.0

    # Equity sleeve budget by regime (rest goes to safe assets)
    region_budget_risk_on: float = 0.55
    region_budget_transition: float = 0.35
    region_budget_risk_off: float = 0.10


class GlobalRotation(Strategy):
    """Rotate across regions only when macro backdrop allows equity risk.

    Thesis: Regional momentum works best when dollar stress is low and rates
    are not tightening aggressively. Cap equity sleeve and maintain a
    persistent defensive allocation (TLT/GLD/UUP) to preserve low correlation
    versus broad equity beta.
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
        if prices.empty:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        avail = [t for t in c.region_tickers if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        p = prices[avail].ffill()

        mom_slow = p.pct_change(c.momentum_window).fillna(0.0)
        mom_fast = p.pct_change(c.fast_momentum).fillna(0.0)
        trend = p.rolling(c.trend_window, min_periods=80).mean()
        trend_ok = p >= trend

        roll_max = p.rolling(c.drawdown_window, min_periods=20).max()
        region_dd = (p / roll_max.replace(0.0, np.nan) - 1.0).fillna(0.0)
        dd_ok = region_dd > -0.12

        # Blend horizons: prioritize slow momentum while reacting faster to turns.
        score = (0.7 * mom_slow + 0.3 * mom_fast).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        rank = score.rank(axis=1, ascending=False, method="first")
        selected = (rank <= c.top_n) & (score > 0.0) & trend_ok & dd_ok

        macro_flags = pd.DataFrame(index=prices.index)
        if c.dollar_ticker in prices.columns:
            uup_mom = prices[c.dollar_ticker].pct_change(c.momentum_window).fillna(0.0)
            macro_flags["dollar"] = pd.notna(uup_mom) & (uup_mom > 0.0)
        else:
            macro_flags["dollar"] = False

        if c.bond_ticker in prices.columns:
            tlt = prices[c.bond_ticker].ffill()
            tlt_sma = tlt.rolling(c.trend_window, min_periods=80).mean()
            macro_flags["rates"] = (tlt < tlt_sma) & (tlt.pct_change(42).fillna(0.0) < 0.0)
        else:
            macro_flags["rates"] = False

        if c.vix_ticker in prices.columns:
            vix = prices[c.vix_ticker]
            macro_flags["vix"] = pd.notna(vix) & (vix > c.vix_risk_off)
        else:
            macro_flags["vix"] = False

        stress_score = macro_flags.astype(int).sum(axis=1)
        risk_off = stress_score >= 2
        transition = stress_score == 1
        risk_on = stress_score == 0

        budget = pd.Series(c.region_budget_transition, index=prices.index)
        budget = budget.where(~risk_on, c.region_budget_risk_on)
        budget = budget.where(~risk_off, c.region_budget_risk_off)

        selected_float = selected.astype(float)
        n_selected = selected_float.sum(axis=1).replace(0.0, np.nan)
        region_alloc = selected_float.div(n_selected, axis=0).fillna(0.0).multiply(budget, axis=0)
        weights.loc[:, avail] = region_alloc

        residual = (c.leverage - weights[avail].sum(axis=1)).clip(lower=0.0)
        safe_avail = [t for t in c.safe_tickers if t in prices.columns]
        if safe_avail:
            # Defensive sleeve is regime-aware to limit equity beta and drawdowns.
            safe_base = pd.DataFrame(0.0, index=prices.index, columns=safe_avail)
            if "TLT" in safe_base.columns:
                safe_base["TLT"] = np.where(risk_off, 0.45, np.where(transition, 0.35, 0.30))
            if "GLD" in safe_base.columns:
                safe_base["GLD"] = np.where(risk_off, 0.35, np.where(transition, 0.30, 0.30))
            if "UUP" in safe_base.columns:
                safe_base["UUP"] = np.where(risk_off, 0.20, np.where(transition, 0.15, 0.0))

            safe_norm = safe_base.sum(axis=1).replace(0.0, np.nan)
            safe_alloc = safe_base.div(safe_norm, axis=0).fillna(0.0).multiply(residual, axis=0)
            for t in safe_avail:
                weights[t] = safe_alloc[t]

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


# =========================================================================
# M9 -- Yield Curve Regime
# =========================================================================

@dataclass
class YieldCurveRegimeConfig:
    """Position based on yield-curve slope regime (TLT/IEF ratio proxy)."""

    tlt_ticker: str = "TLT"
    ief_ticker: str = "IEF"

    # Regime assets
    equity_ticker: str = "SPY"
    bank_ticker: str = "XLF"
    gold_ticker: str = "GLD"
    dollar_ticker: str = "UUP"

    zscore_window: int = 252       # 1-year lookback for z-score
    momentum_window: int = 63      # 3-month slope momentum
    ema_span: int = 21             # EMA smoothing on z-score

    # Regime thresholds
    steep_z: float = 0.5
    flat_z: float = -0.5
    inverted_z: float = -1.5

    leverage: float = 1.0


class YieldCurveRegime(Strategy):
    """Allocate across macro regimes detected from yield-curve slope.

    Thesis: The slope of the yield curve (TLT/IEF ratio as proxy) is
    one of the most reliable macro signals. Steepening signals economic
    expansion (risk-on), flattening signals late-cycle stress (risk-off),
    and inversion signals recession risk (defensive + short equity).

    States:
      STEEPENING (z > 0.5, slope_mom > 0): SPY 1.0, XLF 0.5
      FLAT (|z| <= 0.5): SPY 0.5, TLT 0.3, GLD 0.2
      FLATTENING (z < -0.5, slope_mom < 0): TLT 0.8, GLD 0.4, UUP 0.3
      INVERTED (z < -1.5): TLT 1.0, GLD 0.5, SPY -0.3
    """

    name = "M9-YieldCurveRegime"

    def __init__(self, config: YieldCurveRegimeConfig | None = None) -> None:
        self.cfg = config or YieldCurveRegimeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.tlt_ticker not in prices.columns or c.ief_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Yield-curve slope proxy: TLT/IEF ratio
        tlt = prices[c.tlt_ticker]
        ief = prices[c.ief_ticker]
        slope_ratio = tlt / ief
        slope_ratio = slope_ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)

        # Z-score of slope ratio over 252-day lookback
        roll = slope_ratio.rolling(c.zscore_window, min_periods=60)
        roll_std = roll.std().replace(0, np.nan)
        raw_z = (slope_ratio - roll.mean()) / roll_std
        raw_z = raw_z.fillna(0.0)

        # Smooth z-score with 21-day EMA to prevent whipsaw
        z = ema(raw_z, c.ema_span)

        # Slope momentum: 63-day rate of change of the ratio
        slope_mom = slope_ratio.pct_change(c.momentum_window).fillna(0.0)

        # Regime classification (vectorized, priority: inverted > flattening > steepening > flat)
        is_inverted = z < c.inverted_z
        is_flattening = ~is_inverted & (z < c.flat_z) & (slope_mom < 0)
        is_steepening = (z > c.steep_z) & (slope_mom > 0)
        is_flat = ~is_inverted & ~is_flattening & ~is_steepening

        # STEEPENING: risk-on
        has_spy = c.equity_ticker in prices.columns
        has_xlf = c.bank_ticker in prices.columns
        if has_spy:
            weights[c.equity_ticker] = np.where(is_steepening, c.leverage, 0.0)
        if has_xlf:
            weights[c.bank_ticker] = np.where(
                is_steepening, 0.5 * c.leverage, 0.0,
            )

        # FLAT: balanced
        if has_spy:
            weights[c.equity_ticker] = np.where(
                is_flat, 0.5 * c.leverage, weights[c.equity_ticker],
            )
        weights[c.tlt_ticker] = np.where(is_flat, 0.3 * c.leverage, 0.0)
        has_gld = c.gold_ticker in prices.columns
        if has_gld:
            weights[c.gold_ticker] = np.where(
                is_flat, 0.2 * c.leverage, 0.0,
            )

        # FLATTENING: risk-off
        weights[c.tlt_ticker] = np.where(
            is_flattening, 0.8 * c.leverage, weights[c.tlt_ticker],
        )
        if has_gld:
            weights[c.gold_ticker] = np.where(
                is_flattening, 0.4 * c.leverage, weights[c.gold_ticker],
            )
        has_uup = c.dollar_ticker in prices.columns
        if has_uup:
            weights[c.dollar_ticker] = np.where(
                is_flattening, 0.3 * c.leverage, 0.0,
            )

        # INVERTED: recession -- defensive + short equity
        weights[c.tlt_ticker] = np.where(
            is_inverted, 1.0 * c.leverage, weights[c.tlt_ticker],
        )
        if has_gld:
            weights[c.gold_ticker] = np.where(
                is_inverted, 0.5 * c.leverage, weights[c.gold_ticker],
            )
        if has_spy:
            weights[c.equity_ticker] = np.where(
                is_inverted, -0.3 * c.leverage, weights[c.equity_ticker],
            )

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# M10 — Macro Signal Scoreboard
# =========================================================================

@dataclass
class MacroSignalScoreboardConfig:
    """Config for multi-signal macro scoreboard."""

    # Tickers
    tlt_ticker: str = "TLT"
    ief_ticker: str = "IEF"
    lqd_ticker: str = "LQD"
    hyg_ticker: str = "HYG"
    uup_ticker: str = "UUP"
    gld_ticker: str = "GLD"
    eem_ticker: str = "EEM"
    spy_ticker: str = "SPY"
    qqq_ticker: str = "QQQ"
    xle_ticker: str = "XLE"
    vix_ticker: str = "^VIX"

    # Window lengths
    curve_momentum: int = 63
    credit_momentum: int = 63
    dollar_momentum: int = 63
    gold_momentum: int = 126
    em_momentum: int = 63
    bond_momentum: int = 63
    commodity_momentum: int = 63

    # VIX thresholds
    vix_bull: float = 20.0
    vix_bear: float = 25.0

    # Smoothing
    ema_span: int = 5

    # Leverage per regime
    leverage_strong: float = 1.5
    leverage_mild: float = 1.0
    leverage_defensive: float = 0.8
    leverage_riskoff: float = 0.5


class MacroSignalScoreboard(Strategy):
    """Multi-signal macro scoreboard for regime detection.

    Thesis: Macro regime transitions are best detected by consensus of
    multiple indicators rather than any single signal. Count how many of
    8 macro indicators are bullish, smooth with EMA, and allocate across
    risk buckets.

    Signals (each +1 bullish, -1 bearish, 0 neutral):
      1. Yield curve slope (TLT/IEF 63d momentum) -- steepening = bullish
      2. Credit spread (HYG/LQD 63d momentum) -- narrowing = bullish
      3. Dollar strength (UUP 63d momentum) -- weakening = bullish
      4. Gold momentum (GLD 126d return) -- positive = bullish
      5. EM strength (EEM/SPY 63d ratio momentum) -- positive = bullish
      6. VIX level -- <20 bullish, >25 bearish, else neutral
      7. Bond momentum (TLT 63d return) -- positive = easing = bullish
      8. Commodity momentum (XLE 63d return) -- positive = bullish
    """

    name = "M10-MacroSignalScoreboard"

    def __init__(
        self, config: MacroSignalScoreboardConfig | None = None,
    ) -> None:
        self.cfg = config or MacroSignalScoreboardConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(prices) == 0:
            return weights

        # ------ Compute 8 indicator signals (+1 / -1 / 0) ------
        signals: list[pd.Series] = []

        # 1. Yield curve slope: TLT/IEF ratio momentum
        if c.tlt_ticker in prices.columns and c.ief_ticker in prices.columns:
            curve_ratio = prices[c.tlt_ticker] / prices[c.ief_ticker]
            curve_ratio = curve_ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            curve_mom = curve_ratio.pct_change(c.curve_momentum).fillna(0.0)
            signals.append(np.sign(curve_mom).rename("curve"))
        else:
            signals.append(pd.Series(0.0, index=prices.index, name="curve"))

        # 2. Credit spread: HYG/LQD momentum (rising = narrowing = bullish)
        if c.hyg_ticker in prices.columns and c.lqd_ticker in prices.columns:
            credit_ratio = prices[c.hyg_ticker] / prices[c.lqd_ticker]
            credit_ratio = credit_ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            credit_mom = credit_ratio.pct_change(c.credit_momentum).fillna(0.0)
            signals.append(np.sign(credit_mom).rename("credit"))
        else:
            signals.append(pd.Series(0.0, index=prices.index, name="credit"))

        # 3. Dollar strength: UUP momentum (weakening = bullish -> invert)
        if c.uup_ticker in prices.columns:
            dollar_mom = prices[c.uup_ticker].pct_change(c.dollar_momentum).fillna(0.0)
            signals.append((-np.sign(dollar_mom)).rename("dollar"))
        else:
            signals.append(pd.Series(0.0, index=prices.index, name="dollar"))

        # 4. Gold momentum: GLD 126d return
        if c.gld_ticker in prices.columns:
            gold_mom = prices[c.gld_ticker].pct_change(c.gold_momentum).fillna(0.0)
            signals.append(np.sign(gold_mom).rename("gold"))
        else:
            signals.append(pd.Series(0.0, index=prices.index, name="gold"))

        # 5. EM strength: EEM/SPY ratio momentum
        if c.eem_ticker in prices.columns and c.spy_ticker in prices.columns:
            em_ratio = prices[c.eem_ticker] / prices[c.spy_ticker]
            em_ratio = em_ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            em_mom = em_ratio.pct_change(c.em_momentum).fillna(0.0)
            signals.append(np.sign(em_mom).rename("em"))
        else:
            signals.append(pd.Series(0.0, index=prices.index, name="em"))

        # 6. VIX level: <20 bullish(+1), >25 bearish(-1), else neutral(0)
        if c.vix_ticker in prices.columns:
            vix = prices[c.vix_ticker]
            vix_sig = np.where(pd.notna(vix) & (vix < c.vix_bull), 1.0, 0.0)
            vix_sig = np.where(pd.notna(vix) & (vix > c.vix_bear), -1.0, vix_sig)
            signals.append(pd.Series(vix_sig, index=prices.index, name="vix"))
        else:
            signals.append(pd.Series(0.0, index=prices.index, name="vix"))

        # 7. Bond momentum: TLT 63d return (positive = easing = bullish)
        if c.tlt_ticker in prices.columns:
            bond_mom = prices[c.tlt_ticker].pct_change(c.bond_momentum).fillna(0.0)
            signals.append(np.sign(bond_mom).rename("bond"))
        else:
            signals.append(pd.Series(0.0, index=prices.index, name="bond"))

        # 8. Commodity momentum: XLE 63d return
        if c.xle_ticker in prices.columns:
            comm_mom = prices[c.xle_ticker].pct_change(c.commodity_momentum).fillna(0.0)
            signals.append(np.sign(comm_mom).rename("comm"))
        else:
            signals.append(pd.Series(0.0, index=prices.index, name="comm"))

        # ------ Aggregate: count bullish signals ------
        sig_df = pd.concat(signals, axis=1)
        bullish_count = (sig_df == 1.0).sum(axis=1).astype(float)

        # Smooth with EMA to avoid daily flipping
        bullish_smooth = bullish_count.ewm(span=c.ema_span, adjust=False).mean()

        # ------ Regime-based allocation ------
        strong = bullish_smooth >= 6.0
        mild = pd.notna(bullish_smooth) & (bullish_smooth >= 4.0) & ~strong
        defensive = (
            pd.notna(bullish_smooth) & (bullish_smooth >= 2.0) & ~strong & ~mild
        )
        riskoff = ~strong & ~mild & ~defensive

        # STRONG RISK-ON: SPY 0.5, QQQ 0.3, EEM 0.2 x 1.5
        lev = c.leverage_strong
        if c.spy_ticker in prices.columns:
            weights[c.spy_ticker] = np.where(strong, 0.5 * lev, weights[c.spy_ticker])
        if c.qqq_ticker in prices.columns:
            weights[c.qqq_ticker] = np.where(strong, 0.3 * lev, 0.0)
        if c.eem_ticker in prices.columns:
            weights[c.eem_ticker] = np.where(strong, 0.2 * lev, weights[c.eem_ticker])

        # MILD RISK-ON: SPY 0.4, TLT 0.2, GLD 0.1 x 1.0
        lev = c.leverage_mild
        if c.spy_ticker in prices.columns:
            weights[c.spy_ticker] = np.where(mild, 0.4 * lev, weights[c.spy_ticker])
        if c.tlt_ticker in prices.columns:
            weights[c.tlt_ticker] = np.where(mild, 0.2 * lev, weights[c.tlt_ticker])
        if c.gld_ticker in prices.columns:
            weights[c.gld_ticker] = np.where(mild, 0.1 * lev, weights[c.gld_ticker])

        # DEFENSIVE: TLT 0.4, GLD 0.3, UUP 0.1 x 0.8
        lev = c.leverage_defensive
        if c.tlt_ticker in prices.columns:
            weights[c.tlt_ticker] = np.where(defensive, 0.4 * lev, weights[c.tlt_ticker])
        if c.gld_ticker in prices.columns:
            weights[c.gld_ticker] = np.where(defensive, 0.3 * lev, weights[c.gld_ticker])
        if c.uup_ticker in prices.columns:
            weights[c.uup_ticker] = np.where(defensive, 0.1 * lev, weights[c.uup_ticker])

        # RISK-OFF: TLT 0.4, GLD 0.4, UUP 0.2 x 0.5
        lev = c.leverage_riskoff
        if c.tlt_ticker in prices.columns:
            weights[c.tlt_ticker] = np.where(riskoff, 0.4 * lev, weights[c.tlt_ticker])
        if c.gld_ticker in prices.columns:
            weights[c.gld_ticker] = np.where(riskoff, 0.4 * lev, weights[c.gld_ticker])
        if c.uup_ticker in prices.columns:
            weights[c.uup_ticker] = np.where(riskoff, 0.2 * lev, weights[c.uup_ticker])

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# M11 — Adaptive Macro Blend
# =========================================================================

@dataclass
class AdaptiveMacroBlendConfig:
    """Config for adaptive macro blend using signal accuracy weighting."""

    # Tickers
    tlt_ticker: str = "TLT"
    ief_ticker: str = "IEF"
    lqd_ticker: str = "LQD"
    hyg_ticker: str = "HYG"
    uup_ticker: str = "UUP"
    gld_ticker: str = "GLD"
    eem_ticker: str = "EEM"
    spy_ticker: str = "SPY"
    xle_ticker: str = "XLE"
    vix_ticker: str = "^VIX"

    # Signal windows (same as M10)
    curve_momentum: int = 63
    credit_momentum: int = 63
    dollar_momentum: int = 63
    gold_momentum: int = 126
    em_momentum: int = 63
    bond_momentum: int = 63
    commodity_momentum: int = 63
    vix_bull: float = 20.0
    vix_bear: float = 25.0

    # Accuracy scoring
    accuracy_window: int = 42   # correlation window (63 - 21 = 42 usable days)
    forward_return_days: int = 21
    accuracy_lag: int = 21      # lag to avoid look-ahead

    # Allocation thresholds
    risk_on_threshold: float = 0.25
    risk_off_threshold: float = -0.25

    leverage: float = 0.85
    trend_window: int = 200
    drawdown_window: int = 63


class AdaptiveMacroBlend(Strategy):
    """Dynamically weight macro signals by their recent predictive accuracy.

    Thesis: Instead of equal-weighting all macro signals, weight each by
    how well it predicted forward SPY returns over the recent past. Signals
    currently "working" get higher weight; signals that stopped working
    fade out.

    Look-ahead guard: accuracy uses lagged signals versus realized returns.
    No negative shifts are used in the signal accuracy path.
    """

    name = "M11-AdaptiveMacroBlend"

    def __init__(
        self, config: AdaptiveMacroBlendConfig | None = None,
    ) -> None:
        self.cfg = config or AdaptiveMacroBlendConfig()

    def _compute_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute the 8 macro signals (same logic as M10)."""
        c = self.cfg
        signals: dict[str, pd.Series] = {}
        idx = prices.index

        # 1. Yield curve slope
        if c.tlt_ticker in prices.columns and c.ief_ticker in prices.columns:
            cr = prices[c.tlt_ticker] / prices[c.ief_ticker]
            cr = cr.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            signals["curve"] = np.sign(cr.pct_change(c.curve_momentum).fillna(0.0))
        else:
            signals["curve"] = pd.Series(0.0, index=idx)

        # 2. Credit spread
        if c.hyg_ticker in prices.columns and c.lqd_ticker in prices.columns:
            cr2 = prices[c.hyg_ticker] / prices[c.lqd_ticker]
            cr2 = cr2.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            signals["credit"] = np.sign(cr2.pct_change(c.credit_momentum).fillna(0.0))
        else:
            signals["credit"] = pd.Series(0.0, index=idx)

        # 3. Dollar (inverted -- weakening = bullish)
        if c.uup_ticker in prices.columns:
            signals["dollar"] = -np.sign(
                prices[c.uup_ticker].pct_change(c.dollar_momentum).fillna(0.0),
            )
        else:
            signals["dollar"] = pd.Series(0.0, index=idx)

        # 4. Gold momentum
        if c.gld_ticker in prices.columns:
            signals["gold"] = np.sign(
                prices[c.gld_ticker].pct_change(c.gold_momentum).fillna(0.0),
            )
        else:
            signals["gold"] = pd.Series(0.0, index=idx)

        # 5. EM strength
        if c.eem_ticker in prices.columns and c.spy_ticker in prices.columns:
            er = prices[c.eem_ticker] / prices[c.spy_ticker]
            er = er.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            signals["em"] = np.sign(er.pct_change(c.em_momentum).fillna(0.0))
        else:
            signals["em"] = pd.Series(0.0, index=idx)

        # 6. VIX level
        if c.vix_ticker in prices.columns:
            vix = prices[c.vix_ticker]
            s = np.where(pd.notna(vix) & (vix < c.vix_bull), 1.0, 0.0)
            s = np.where(pd.notna(vix) & (vix > c.vix_bear), -1.0, s)
            signals["vix"] = pd.Series(s, index=idx)
        else:
            signals["vix"] = pd.Series(0.0, index=idx)

        # 7. Bond momentum
        if c.tlt_ticker in prices.columns:
            signals["bond"] = np.sign(
                prices[c.tlt_ticker].pct_change(c.bond_momentum).fillna(0.0),
            )
        else:
            signals["bond"] = pd.Series(0.0, index=idx)

        # 8. Commodity momentum
        if c.xle_ticker in prices.columns:
            signals["comm"] = np.sign(
                prices[c.xle_ticker].pct_change(c.commodity_momentum).fillna(0.0),
            )
        else:
            signals["comm"] = pd.Series(0.0, index=idx)

        return pd.DataFrame(signals, index=idx)

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if len(prices) == 0:
            return weights

        sig_df = self._compute_signals(prices)

        # Forward 21-day SPY return (will be lagged to avoid look-ahead)
        if c.spy_ticker not in prices.columns:
            return weights

        spy = prices[c.spy_ticker]
        # Realized 21-day return available at time t.
        horizon_ret = spy.pct_change(c.forward_return_days).fillna(0.0)

        # Rolling accuracy: corr(lagged signal, realized return), then lag again.
        accuracy_cols: dict[str, pd.Series] = {}
        for col in sig_df.columns:
            signal_lagged = sig_df[col].shift(c.forward_return_days)
            raw_corr = signal_lagged.rolling(
                c.accuracy_window, min_periods=20,
            ).corr(horizon_ret)
            raw_corr = raw_corr.where(signal_lagged.notna(), 0.0)
            accuracy_cols[col] = raw_corr.shift(c.accuracy_lag).fillna(0.0)

        accuracy_df = pd.DataFrame(accuracy_cols, index=prices.index)

        # Weight each signal by max(accuracy, 0)
        pos_accuracy = accuracy_df.clip(lower=0.0)
        total_weight = pos_accuracy.sum(axis=1).replace(0.0, np.nan)

        # Weighted score = sum(weight_i * signal_i) / sum(weight_i)
        weighted_sum = (pos_accuracy * sig_df).sum(axis=1)
        weighted_score = (weighted_sum / total_weight).fillna(0.0)
        weighted_score = weighted_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # Macro filters: tightening regime pushes allocation toward defensive sleeve.
        if c.uup_ticker in prices.columns:
            uup_mom = prices[c.uup_ticker].pct_change(c.dollar_momentum).fillna(0.0)
            dollar_strong = pd.notna(uup_mom) & (uup_mom > 0.0)
        else:
            dollar_strong = pd.Series(False, index=prices.index)

        if c.tlt_ticker in prices.columns:
            tlt = prices[c.tlt_ticker]
            tlt_sma = tlt.rolling(c.trend_window, min_periods=80).mean()
            easing_bias = tlt >= tlt_sma
        else:
            easing_bias = pd.Series(False, index=prices.index)

        if c.vix_ticker in prices.columns:
            vix = prices[c.vix_ticker]
            vix_high = pd.notna(vix) & (vix > c.vix_bear)
        else:
            vix_high = pd.Series(False, index=prices.index)

        macro_tightening = dollar_strong & (~easing_bias)

        # Equity sleeve drawdown scaling for smoother risk reduction.
        daily = prices.pct_change().fillna(0.0)
        eq_proxy = 0.0
        if c.spy_ticker in daily.columns:
            eq_proxy = eq_proxy + 0.6 * daily[c.spy_ticker]
        if c.eem_ticker in daily.columns:
            eq_proxy = eq_proxy + 0.25 * daily[c.eem_ticker]
        if c.xle_ticker in daily.columns:
            eq_proxy = eq_proxy + 0.15 * daily[c.xle_ticker]
        eq_proxy = pd.Series(eq_proxy, index=prices.index).fillna(0.0)
        eq_curve = (1.0 + eq_proxy).cumprod()
        eq_dd = eq_curve / eq_curve.cummax().replace(0.0, np.nan) - 1.0
        dd_scale = np.where(eq_dd < -0.12, 0.4, np.where(eq_dd < -0.08, 0.7, 1.0))
        dd_scale = pd.Series(dd_scale, index=prices.index)

        # --- Allocation ---
        lev = c.leverage
        risk_on = (
            pd.notna(weighted_score)
            & (weighted_score > c.risk_on_threshold)
            & (~macro_tightening)
            & (~vix_high)
        )
        risk_off = (
            pd.notna(weighted_score)
            & ((weighted_score < c.risk_off_threshold) | vix_high | macro_tightening)
        )
        neutral = ~risk_on & ~risk_off

        # Risk-on: capped equity sleeve + macro hedges.
        if c.spy_ticker in prices.columns:
            weights[c.spy_ticker] = np.where(
                risk_on,
                0.25 * lev * dd_scale,
                weights[c.spy_ticker],
            )
        if c.eem_ticker in prices.columns:
            weights[c.eem_ticker] = np.where(
                risk_on,
                0.15 * lev * dd_scale,
                weights[c.eem_ticker],
            )
        if c.xle_ticker in prices.columns:
            weights[c.xle_ticker] = np.where(
                risk_on,
                0.10 * lev * dd_scale,
                weights[c.xle_ticker],
            )
        if c.tlt_ticker in prices.columns:
            weights[c.tlt_ticker] = np.where(risk_on, 0.20 * lev, weights[c.tlt_ticker])
        if c.gld_ticker in prices.columns:
            weights[c.gld_ticker] = np.where(risk_on, 0.15 * lev, weights[c.gld_ticker])
        if c.uup_ticker in prices.columns:
            weights[c.uup_ticker] = np.where(risk_on, 0.05 * lev, weights[c.uup_ticker])
        if c.ief_ticker in prices.columns:
            weights[c.ief_ticker] = np.where(risk_on, 0.10 * lev, weights[c.ief_ticker])

        # Neutral: macro-balanced sleeve.
        if c.spy_ticker in prices.columns:
            weights[c.spy_ticker] = np.where(neutral, 0.10 * lev, weights[c.spy_ticker])
        if c.tlt_ticker in prices.columns:
            weights[c.tlt_ticker] = np.where(neutral, 0.30 * lev, weights[c.tlt_ticker])
        if c.gld_ticker in prices.columns:
            weights[c.gld_ticker] = np.where(neutral, 0.25 * lev, weights[c.gld_ticker])
        if c.uup_ticker in prices.columns:
            weights[c.uup_ticker] = np.where(neutral, 0.10 * lev, weights[c.uup_ticker])
        if c.ief_ticker in prices.columns:
            weights[c.ief_ticker] = np.where(neutral, 0.25 * lev, weights[c.ief_ticker])

        # Risk-off: explicit capital preservation sleeve.
        if c.tlt_ticker in prices.columns:
            weights[c.tlt_ticker] = np.where(risk_off, 0.35 * lev, weights[c.tlt_ticker])
        if c.gld_ticker in prices.columns:
            weights[c.gld_ticker] = np.where(risk_off, 0.30 * lev, weights[c.gld_ticker])
        if c.uup_ticker in prices.columns:
            weights[c.uup_ticker] = np.where(risk_off, 0.20 * lev, weights[c.uup_ticker])
        if c.ief_ticker in prices.columns:
            weights[c.ief_ticker] = np.where(risk_off, 0.15 * lev, weights[c.ief_ticker])

        gross = weights.clip(lower=0.0).sum(axis=1).replace(0.0, np.nan)
        scale = (lev / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.multiply(scale, axis=0)

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# M1b — Dollar Carry Scoreboard (improved M1 with signal consensus)
# =========================================================================

@dataclass
class DollarCarryScoreboardConfig:
    """Config for dollar carry trade with 3-signal consensus."""

    dollar_ticker: str = "UUP"
    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"

    # Signal 1: UUP momentum (same as original M1)
    momentum_window: int = 63
    ema_span: int = 21

    # Signal 2: UUP SMA crossover (DXY proxy)
    fast_sma: int = 20
    slow_sma: int = 60

    # Signal 3: TLT 20-day momentum as rate differential proxy
    tlt_momentum: int = 20

    # Position sizing
    leverage_risk_on: float = 1.0
    safe_weight: float = 0.5

    # Consensus: need >= 2 of 3 signals agreeing
    consensus_threshold: int = 2


class DollarCarryScoreboard(Strategy):
    """Dollar carry trade enhanced with 3-signal consensus check.

    Thesis: The original M1-DollarCarry uses a single signal (UUP momentum).
    This variant requires 2-of-3 signal consensus before entry:
      1. UUP momentum < 0 (dollar weakening)
      2. UUP 20d SMA < 60d SMA (dollar in downtrend)
      3. TLT 20d return > 0 (rates falling = dollar weakening)

    The consensus filter reduces whipsaw and improves signal quality.
    """

    name = "M1b-DollarCarryScore"

    def __init__(
        self, config: DollarCarryScoreboardConfig | None = None,
    ) -> None:
        self.cfg = config or DollarCarryScoreboardConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        if c.dollar_ticker not in prices.columns:
            return weights

        uup = prices[c.dollar_ticker]
        uup_ema = ema(uup, c.ema_span)

        # Signal 1: UUP momentum -- negative = dollar weakening = risk-on
        dollar_mom = uup_ema.pct_change(c.momentum_window).fillna(0.0)
        sig1_risk_on = (dollar_mom < 0).astype(float)

        # Signal 2: UUP fast SMA < slow SMA = dollar downtrend = risk-on
        uup_fast = uup.rolling(c.fast_sma, min_periods=10).mean()
        uup_slow = uup.rolling(c.slow_sma, min_periods=20).mean()
        sig2_risk_on = (uup_fast < uup_slow).astype(float)

        # Signal 3: TLT 20-day momentum > 0 = rates falling = risk-on
        if c.safe_ticker in prices.columns:
            tlt_mom = prices[c.safe_ticker].pct_change(c.tlt_momentum).fillna(0.0)
            sig3_risk_on = (pd.notna(tlt_mom) & (tlt_mom > 0)).astype(float)
        else:
            sig3_risk_on = pd.Series(0.0, index=prices.index)

        # Consensus score: count how many signals say risk-on
        consensus_on = sig1_risk_on + sig2_risk_on + sig3_risk_on

        # Inverse consensus for risk-off direction
        sig1_off = (dollar_mom > 0).astype(float)
        sig2_off = (uup_fast > uup_slow).astype(float)
        if c.safe_ticker in prices.columns:
            sig3_off = (pd.notna(tlt_mom) & (tlt_mom < 0)).astype(float)
        else:
            sig3_off = pd.Series(0.0, index=prices.index)
        consensus_off = sig1_off + sig2_off + sig3_off

        # Take trade only when consensus threshold met
        go_risk_on = consensus_on >= c.consensus_threshold
        go_risk_off = consensus_off >= c.consensus_threshold

        if c.equity_ticker in prices.columns:
            weights[c.equity_ticker] = np.where(
                go_risk_on, c.leverage_risk_on, 0.0,
            )
        safe_col = c.safe_ticker if c.safe_ticker in prices.columns else c.dollar_ticker
        weights[safe_col] = np.where(
            go_risk_off, c.safe_weight,
            np.where(~go_risk_on, c.safe_weight * 0.5, weights[safe_col]),
        )

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
