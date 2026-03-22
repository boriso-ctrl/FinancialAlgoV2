"""Category O: Tail Risk & Insurance strategies.

Strategies designed for capital preservation and crisis alpha.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import realized_vol, ema, drawdown as _drawdown
from financial_algo.regimes import Regime
from financial_algo.strategies.base import Strategy


# =========================================================================
# O1 -- Tail Risk Parity
# =========================================================================

@dataclass
class TailRiskParityConfig:
    """Allocate risk budget across safe havens and equities.

    In normal times: balanced risk-parity with tilt to equities.
    In crisis: shift risk budget to tail hedges (GLD, TLT).
    """

    equity_tickers: tuple = ("SPY", "QQQ", "IWM")
    safe_tickers: tuple = ("GLD", "TLT", "IEF")

    vol_window: int = 60
    crisis_vol_threshold: float = 0.25  # SPY annualized vol threshold

    # Risk budget split: fraction going to safe havens
    normal_safe_budget: float = 0.30
    crisis_safe_budget: float = 0.70

    leverage: float = 1.5


class TailRiskParity(Strategy):
    """Risk-parity allocation that shifts to safe havens in high-vol.

    Thesis: Traditional risk parity underweights tail risk.
    By dynamically increasing the safe-haven risk budget when
    vol spikes, we capture crisis alpha while maintaining
    equity upside in calm periods. Similar to Bridgewater's
    All-Weather with a dynamic overlay.
    """

    name = "O1-TailRiskParity"

    def __init__(self, config: TailRiskParityConfig | None = None) -> None:
        self.cfg = config or TailRiskParityConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        eq = [t for t in c.equity_tickers if t in prices.columns]
        sf = [t for t in c.safe_tickers if t in prices.columns]
        all_t = eq + sf
        if not all_t:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Compute inverse-vol for each asset
        vols = pd.DataFrame(index=prices.index)
        for t in all_t:
            vols[t] = realized_vol(prices[t], c.vol_window).clip(lower=0.01)

        inv_vol = 1.0 / vols
        inv_vol = inv_vol.fillna(0.0)

        # Crisis detection via SPY vol
        spy_col = "SPY" if "SPY" in prices.columns else eq[0] if eq else sf[0]
        spy_vol = realized_vol(prices[spy_col], c.vol_window)
        crisis = spy_vol > c.crisis_vol_threshold

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Vectorized: compute safe_budget per day
        safe_budget = pd.Series(
            np.where(crisis, c.crisis_safe_budget, c.normal_safe_budget),
            index=prices.index,
        )
        eq_budget = 1.0 - safe_budget

        # Inverse-vol within each bucket
        eq_iv = inv_vol[eq].fillna(0)
        sf_iv = inv_vol[sf].fillna(0)
        eq_sum = eq_iv.sum(axis=1).replace(0, np.nan)
        sf_sum = sf_iv.sum(axis=1).replace(0, np.nan)

        for t in eq:
            weights[t] = c.leverage * eq_budget * eq_iv[t] / eq_sum
        for t in sf:
            weights[t] = c.leverage * safe_budget * sf_iv[t] / sf_sum

        return weights.fillna(0.0)


# =========================================================================
# O2 -- Crisis Alpha Momentum
# =========================================================================

@dataclass
class CrisisAlphaMomConfig:
    """Regime-conditional safe-haven and recovery allocation.

    In crisis: long GLD + TLT (safe havens with trend filter).
    In recovery: long QQQ (capture V-shaped bounce).
    Otherwise: flat.
    """

    gold_ticker: str = "GLD"
    bond_ticker: str = "TLT"
    equity_ticker: str = "QQQ"

    ema_span: int = 50
    leverage_haven: float = 0.6
    leverage_recovery: float = 1.2


class CrisisAlphaMomentum(Strategy):
    """Regime-conditional allocation: safe havens in crisis, equity in recovery.

    Reworked from complex trend-following (which had negative Sharpe) to
    a simple regime-conditional strategy. Long GLD/TLT during crisis
    (with trend filter), long QQQ during recovery.
    """

    name = "O2-CrisisAlphaMomentum"

    def __init__(self, config: CrisisAlphaMomConfig | None = None) -> None:
        self.cfg = config or CrisisAlphaMomConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        c = self.cfg
        crisis = regime.isin({
            Regime.GENERAL_CRISIS, Regime.WAR_CRISIS, Regime.OIL_CRISIS,
        })
        recovery = regime.isin({Regime.RECOVERY})

        tickers = [c.gold_ticker, c.bond_ticker, c.equity_ticker]
        avail = [t for t in tickers if t in prices.columns]
        w = pd.DataFrame(0.0, index=prices.index, columns=avail)

        # Crisis: long safe havens with trend filter
        if c.gold_ticker in avail:
            gld_ema = ema(prices[c.gold_ticker], c.ema_span)
            gld_up = prices[c.gold_ticker] > gld_ema
            w.loc[crisis & gld_up, c.gold_ticker] = c.leverage_haven

        if c.bond_ticker in avail:
            tlt_ema = ema(prices[c.bond_ticker], c.ema_span)
            tlt_up = prices[c.bond_ticker] > tlt_ema
            w.loc[crisis & tlt_up, c.bond_ticker] = c.leverage_haven

        # Recovery: long equity
        if c.equity_ticker in avail:
            w.loc[recovery, c.equity_ticker] = c.leverage_recovery

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# O4 -- Black Swan Insurance
# =========================================================================

@dataclass
class BlackSwanConfig:
    """Permanent safe-haven allocation that levers up in crisis.

    Normal: small GLD + TLT allocation (insurance premium).
    Crisis: lever up safe havens + short equities.
    """

    safe_tickers: tuple = ("GLD", "TLT")
    equity_short: str = "SPY"

    vol_window: int = 20

    # Normal allocation per safe asset
    normal_weight: float = 0.15

    # Crisis thresholds (SPY annualized vol)
    crisis_threshold: float = 0.25
    severe_threshold: float = 0.40

    # Crisis weights
    crisis_safe_weight: float = 0.40
    crisis_short_weight: float = 0.20

    severe_safe_weight: float = 0.60
    severe_short_weight: float = 0.40


class BlackSwanInsurance(Strategy):
    """Permanent safe-haven allocation that scales up in crisis.

    Thesis: Maintain a constant insurance allocation to GLD and TLT.
    The cost of this insurance (drag in calm markets) is offset by
    massive outperformance during tail events. When vol spikes,
    lever up safe havens and add equity shorts for convex payoff.
    """

    name = "O4-BlackSwanInsurance"

    def __init__(self, config: BlackSwanConfig | None = None) -> None:
        self.cfg = config or BlackSwanConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        avail_safe = [t for t in c.safe_tickers if t in prices.columns]
        if not avail_safe:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        spy_col = c.equity_short if c.equity_short in prices.columns else avail_safe[0]
        spy_vol = realized_vol(prices[spy_col], c.vol_window)

        severe = spy_vol > c.severe_threshold
        crisis = (spy_vol > c.crisis_threshold) & ~severe
        normal = ~crisis & ~severe

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        n_safe = len(avail_safe)

        for t in avail_safe:
            weights.loc[normal, t] = c.normal_weight / n_safe
            weights.loc[crisis, t] = c.crisis_safe_weight / n_safe
            weights.loc[severe, t] = c.severe_safe_weight / n_safe

        if c.equity_short in prices.columns:
            weights.loc[crisis, c.equity_short] = -c.crisis_short_weight
            weights.loc[severe, c.equity_short] = -c.severe_short_weight

        return weights.fillna(0.0)


# =========================================================================
# O3 -- Crisis Rotation
# =========================================================================

@dataclass
class CrisisRotationConfig:
    """Dynamic allocation based on crisis type.

    Routes capital to the best-suited safe haven depending on
    whether the crisis is oil-driven, war-driven, or a general crash.
    """

    gold_ticker: str = "GLD"
    bond_ticker: str = "TLT"
    defense_ticker: str = "ITA"
    equity_ticker: str = "QQQ"
    energy_ticker: str = "XLE"

    ema_span: int = 50

    oil_crisis_gold: float = 0.5
    oil_crisis_energy: float = 0.5
    war_crisis_defense: float = 0.5
    war_crisis_gold: float = 0.5
    gen_crisis_gold: float = 0.5
    gen_crisis_bond: float = 0.5
    recovery_equity: float = 1.2


class CrisisRotation(Strategy):
    """Dynamic crisis-type rotation: routes to best haven per crisis type.

    Oil crisis: GLD + XLE (oil spikes = inflation hedge + energy momentum).
    War crisis: ITA + GLD (defense + geopolitical hedge).
    General crisis: GLD + TLT (classic flight-to-quality).
    Recovery: QQQ (capture V-shaped bounce).
    """

    name = "O3-CrisisRotation"

    def __init__(self, config: CrisisRotationConfig | None = None) -> None:
        self.cfg = config or CrisisRotationConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        c = self.cfg
        oil_crisis = regime.isin({Regime.OIL_CRISIS})
        war_crisis = regime.isin({Regime.WAR_CRISIS})
        gen_crisis = regime.isin({Regime.GENERAL_CRISIS})
        recovery = regime.isin({Regime.RECOVERY})

        tickers = [c.gold_ticker, c.bond_ticker, c.defense_ticker,
                   c.equity_ticker, c.energy_ticker]
        avail = [t for t in tickers if t in prices.columns]
        w = pd.DataFrame(0.0, index=prices.index, columns=avail)

        def _trend_up(ticker: str) -> pd.Series:
            if ticker in prices.columns:
                return prices[ticker] > ema(prices[ticker], c.ema_span)
            return pd.Series(False, index=prices.index)

        # Oil crisis: GLD + XLE
        if c.gold_ticker in avail:
            w.loc[oil_crisis & _trend_up(c.gold_ticker), c.gold_ticker] = c.oil_crisis_gold
        if c.energy_ticker in avail:
            w.loc[oil_crisis & _trend_up(c.energy_ticker), c.energy_ticker] = c.oil_crisis_energy

        # War crisis: ITA + GLD
        if c.defense_ticker in avail:
            w.loc[war_crisis & _trend_up(c.defense_ticker), c.defense_ticker] = c.war_crisis_defense
        if c.gold_ticker in avail:
            w.loc[war_crisis & _trend_up(c.gold_ticker), c.gold_ticker] = c.war_crisis_gold

        # General crisis: GLD + TLT
        if c.gold_ticker in avail:
            w.loc[gen_crisis & _trend_up(c.gold_ticker), c.gold_ticker] = c.gen_crisis_gold
        if c.bond_ticker in avail:
            w.loc[gen_crisis & _trend_up(c.bond_ticker), c.bond_ticker] = c.gen_crisis_bond

        # Recovery: QQQ
        if c.equity_ticker in avail:
            w.loc[recovery, c.equity_ticker] = c.recovery_equity

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# O5 -- VIX Spike Recovery
# =========================================================================

@dataclass
class VIXSpikeRecoveryConfig:
    """Go long risk assets after vol spikes above crisis level and declines.

    Captures the V-shaped recovery rally that typically follows
    a volatility spike.
    """

    equity_ticker: str = "QQQ"
    secondary_ticker: str = "SPY"

    vol_window: int = 20
    vol_spike_threshold: float = 0.30
    vol_recovery_pct: float = 0.25
    vol_max_current: float = 0.25
    vol_lookback: int = 20
    ema_span: int = 20

    leverage_primary: float = 1.5
    leverage_secondary: float = 0.5


class VIXSpikeRecovery(Strategy):
    """Long risk assets after volatility spikes and starts declining.

    Thesis: After annualised vol spikes above 0.30 and starts
    declining, equities typically rally 10-20% in the next 2-3 months.
    This strategy captures that recovery by going leveraged long QQQ + SPY
    when vol has recently been very high but is now declining.
    """

    name = "O5-VIXSpikeRecovery"

    def __init__(self, config: VIXSpikeRecoveryConfig | None = None) -> None:
        self.cfg = config or VIXSpikeRecoveryConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        spy_col = c.equity_ticker if c.equity_ticker in prices.columns else "SPY"
        vol = realized_vol(prices[spy_col], c.vol_window)

        # Detect vol was recently very high
        vol_peak = vol.rolling(c.vol_lookback, min_periods=1).max()
        vol_was_high = vol_peak > c.vol_spike_threshold
        vol_declining = vol < vol_peak * (1.0 - c.vol_recovery_pct)
        vol_moderate = vol < c.vol_max_current

        # Recovery signal: vol was high, now declining, not still extreme
        recovery_signal = vol_was_high & vol_declining & vol_moderate

        # Trend confirmation: equity above short-term EMA
        eq_ema = ema(prices[spy_col], c.ema_span)
        trend_up = prices[spy_col] > eq_ema

        signal = recovery_signal & trend_up

        tickers = [c.equity_ticker, c.secondary_ticker]
        avail = [t for t in tickers if t in prices.columns]
        w = pd.DataFrame(0.0, index=prices.index, columns=avail)

        if c.equity_ticker in avail:
            w.loc[signal, c.equity_ticker] = c.leverage_primary
        if c.secondary_ticker in avail:
            w.loc[signal, c.secondary_ticker] = c.leverage_secondary

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# O6 -- Tail Hedge Overlay
# =========================================================================

@dataclass
class TailHedgeOverlayConfig:
    """Permanent safe-haven overlay with VIX-tiered sizing.

    Always-on base allocation to GLD + TLT.
    Scales up when VIX is elevated or in crisis.
    Adds trend filter (SPY < 200d SMA) for extra safe-haven tilt.
    """

    gold_ticker: str = "GLD"
    bond_ticker: str = "TLT"
    dollar_ticker: str = "UUP"
    spy_ticker: str = "SPY"

    # Base allocation (always on)
    base_gold: float = 0.015
    base_bond: float = 0.015

    # Elevated VIX (> 25): increase allocation
    elevated_gold: float = 0.05
    elevated_bond: float = 0.05
    elevated_dollar: float = 0.03

    # Crisis VIX (> 35): maximum allocation
    crisis_gold: float = 0.10
    crisis_bond: float = 0.08
    crisis_dollar: float = 0.05

    # Trend filter: extra safe-haven when SPY < 200d SMA
    trend_sma_window: int = 200
    trend_extra_gold: float = 0.015
    trend_extra_bond: float = 0.015

    # VIX thresholds
    vix_elevated: float = 25.0
    vix_crisis: float = 35.0

    # Realized vol proxy window (when regime not available)
    vol_window: int = 20


class TailHedgeOverlay(Strategy):
    """Permanent small safe-haven allocation that scales up in stress.

    Thesis: A cheap, always-on hedge that bleeds minimally in calm
    markets (~3% total allocation) but provides meaningful protection
    during drawdowns by scaling up to 13-23% in safe havens.

    Never long equities -- pure hedge overlay.

    Signal logic:
      - Base: ~3% GLD + TLT (always on)
      - VIX > 25: 13% total (GLD + TLT + UUP)
      - VIX > 35: 23% total (heavier GLD + TLT + UUP)
      - SPY < 200d SMA: +3% extra safe haven on top
    """

    name = "O6-TailHedgeOverlay"

    def __init__(self, config: TailHedgeOverlayConfig | None = None) -> None:
        self.cfg = config or TailHedgeOverlayConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        # Determine available tickers
        avail = [t for t in [c.gold_ticker, c.bond_ticker, c.dollar_ticker]
                 if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=avail)

        # --- VIX tier detection via realized vol proxy ---
        spy_col = c.spy_ticker if c.spy_ticker in prices.columns else avail[0]
        ann_vol = realized_vol(prices[spy_col], c.vol_window).fillna(0.0)
        # Map annualized vol to approximate VIX level: VIX ~ ann_vol * 100
        vix_proxy = ann_vol * 100.0

        # Use regime if available to reinforce crisis detection
        is_crisis_regime = pd.Series(False, index=prices.index)
        is_elevated_regime = pd.Series(False, index=prices.index)
        if regime is not None:
            is_crisis_regime = regime.isin({
                Regime.GENERAL_CRISIS, Regime.WAR_CRISIS, Regime.OIL_CRISIS,
            })
            is_elevated_regime = regime.isin({Regime.ELEVATED})

        # VIX tiers (combine vol proxy with regime)
        crisis_mask = pd.notna(vix_proxy) & (vix_proxy > c.vix_crisis) | is_crisis_regime
        elevated_mask = (
            pd.notna(vix_proxy) & (vix_proxy > c.vix_elevated) | is_elevated_regime
        ) & ~crisis_mask
        base_mask = ~crisis_mask & ~elevated_mask

        # --- Trend filter: SPY below 200d SMA ---
        trend_below = pd.Series(False, index=prices.index)
        if c.spy_ticker in prices.columns:
            sma_200 = prices[c.spy_ticker].rolling(c.trend_sma_window).mean()
            trend_below = pd.notna(sma_200) & (prices[c.spy_ticker] < sma_200)

        # --- Assign weights by tier ---
        has_gold = c.gold_ticker in avail
        has_bond = c.bond_ticker in avail
        has_dollar = c.dollar_ticker in avail

        # Base allocation (always on)
        if has_gold:
            weights[c.gold_ticker] = c.base_gold
        if has_bond:
            weights[c.bond_ticker] = c.base_bond

        # Elevated: override base with larger allocation
        if has_gold:
            weights.loc[elevated_mask, c.gold_ticker] = c.elevated_gold
        if has_bond:
            weights.loc[elevated_mask, c.bond_ticker] = c.elevated_bond
        if has_dollar:
            weights.loc[elevated_mask, c.dollar_ticker] = c.elevated_dollar

        # Crisis: override with maximum allocation
        if has_gold:
            weights.loc[crisis_mask, c.gold_ticker] = c.crisis_gold
        if has_bond:
            weights.loc[crisis_mask, c.bond_ticker] = c.crisis_bond
        if has_dollar:
            weights.loc[crisis_mask, c.dollar_ticker] = c.crisis_dollar

        # Trend filter: add extra safe-haven when SPY < 200d SMA
        if has_gold:
            weights.loc[trend_below, c.gold_ticker] += c.trend_extra_gold
        if has_bond:
            weights.loc[trend_below, c.bond_ticker] += c.trend_extra_bond

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# O7 -- Precious Metals Crisis Hedge
# =========================================================================

@dataclass
class PreciousMetalsCrisisHedgeConfig:
    """Vol-tiered allocation between risk-on equities and safe havens.

    Normal (vol < 0.15): SPY + QQQ (risk-on).
    Elevated (0.15 <= vol < 0.25): partial hedge, split between equities
        and precious metals / treasuries.
    Crisis (vol >= 0.25): full crisis hedge with GLD, SLV, TLT, SHY.
        Tilts toward SLV for higher-beta precious metal convexity.
    """

    risk_on: tuple = ("SPY", "QQQ")
    safe_haven: tuple = ("GLD", "SLV", "TLT", "SHY")

    vol_window: int = 20
    normal_threshold: float = 0.15
    crisis_threshold: float = 0.25

    # Risk-on weights (split equally)
    risk_on_weight: float = 0.50

    # Elevated: partial hedge mix
    elevated_equity_weight: float = 0.30
    elevated_safe_weight: float = 0.20

    # Crisis weights: precious metals + treasuries
    # SLV gets more than GLD for convexity (higher beta = bigger crisis payoff)
    crisis_gld: float = 0.25
    crisis_slv: float = 0.35
    crisis_tlt: float = 0.25
    crisis_shy: float = 0.15


class PreciousMetalsCrisisHedge(Strategy):
    """Vol-tiered allocation: equities in calm, precious metals in crisis.

    Thesis: SLV has higher beta than GLD in panic environments,
    delivering more convexity when it matters most. Combined with
    TLT + SHY for flight-to-quality, this provides a layered
    hedge that scales with crisis severity.

    The strategy uses 20-day realized vol of SPY as a VIX proxy:
      - vol < 0.15: 100% risk-on (SPY + QQQ)
      - 0.15 <= vol < 0.25: partial hedge (mix of equities + safe havens)
      - vol >= 0.25: full crisis mode (GLD + SLV + TLT + SHY)
    """

    name = "O7-PreciousMetalsCrisisHedge"

    def __init__(self, config: PreciousMetalsCrisisHedgeConfig | None = None) -> None:
        self.cfg = config or PreciousMetalsCrisisHedgeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        all_tickers = list(c.risk_on) + list(c.safe_haven)
        avail = [t for t in all_tickers if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # SPY realized vol as VIX proxy
        spy_col = "SPY" if "SPY" in prices.columns else avail[0]
        vol = realized_vol(prices[spy_col], c.vol_window).fillna(0.0)

        # Regime masks (vectorized)
        crisis = pd.notna(vol) & (vol >= c.crisis_threshold)
        elevated = pd.notna(vol) & (vol >= c.normal_threshold) & ~crisis
        normal = ~crisis & ~elevated

        # Normal: risk-on equities
        risk_on_avail = [t for t in c.risk_on if t in prices.columns]
        n_risk = len(risk_on_avail)
        if n_risk > 0:
            per_eq = c.risk_on_weight / n_risk
            for t in risk_on_avail:
                weights.loc[normal, t] = per_eq

        # Elevated: partial hedge
        if n_risk > 0:
            per_eq_elev = c.elevated_equity_weight / n_risk
            for t in risk_on_avail:
                weights.loc[elevated, t] = per_eq_elev

        safe_avail = [t for t in c.safe_haven if t in prices.columns]
        n_safe = len(safe_avail)
        if n_safe > 0:
            per_safe_elev = c.elevated_safe_weight / n_safe
            for t in safe_avail:
                weights.loc[elevated, t] = per_safe_elev

        # Crisis: full safe-haven allocation with SLV tilt
        crisis_map = {
            "GLD": c.crisis_gld,
            "SLV": c.crisis_slv,
            "TLT": c.crisis_tlt,
            "SHY": c.crisis_shy,
        }
        for t in safe_avail:
            w_val = crisis_map.get(t, 0.0)
            if w_val > 0:
                weights.loc[crisis, t] = w_val

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# S3 -- Drawdown Recovery Timing
# =========================================================================

@dataclass
class DrawdownRecoveryTimingConfig:
    """Aggressively long risk assets during confirmed recovery from deep drawdowns.

    Trigger: SPY drawdown crosses -15%, then 5 consecutive up-days
    from the trough confirm the recovery has begun. Strategy stays
    long for 60 trading days after confirmation.
    Outside recovery windows: fully neutral (no bleed).
    """

    spy_ticker: str = "SPY"
    risk_tickers: tuple = ("SPY", "QQQ", "IWM", "EFA")

    drawdown_threshold: float = -0.15  # -15% from peak
    confirmation_days: int = 5  # consecutive up days to confirm trough
    recovery_window: int = 60  # trading days to stay long after confirmation

    per_asset_weight: float = 0.25  # equal weight per risk asset


class DrawdownRecoveryTiming(Strategy):
    """Long risk assets during early recovery from deep drawdowns.

    After SPY draws down >= 15% from peak and then shows 5 consecutive
    up-days (trough confirmation), go long SPY/QQQ/IWM/EFA with equal
    weight for 60 trading days. These recovery windows (2020 COVID
    bounce, 2022 bear market bottom) are the highest-Sharpe periods.

    Outside recovery windows the strategy is fully flat -- zero cost
    of carry in calm markets.
    """

    name = "S3-DrawdownRecoveryTiming"

    def __init__(self, config: DrawdownRecoveryTimingConfig | None = None) -> None:
        self.cfg = config or DrawdownRecoveryTimingConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        spy_col = c.spy_ticker if c.spy_ticker in prices.columns else None
        if spy_col is None:
            return weights

        spy = prices[spy_col]

        # Rolling drawdown from peak
        peak = spy.cummax()
        dd = ((spy - peak) / peak).fillna(0.0)

        # Deep drawdown flag
        in_deep_dd = dd <= c.drawdown_threshold

        # Daily returns for consecutive-up detection
        spy_ret = spy.pct_change().fillna(0.0)
        up_day = (spy_ret > 0).astype(float)

        # Consecutive up-day count (vectorized cumsum trick)
        not_up = up_day == 0
        group_id = not_up.cumsum()
        consec_up = up_day.groupby(group_id).cumsum()

        # Check if we were recently in deep DD (within lookback window)
        lookback = c.confirmation_days + 10
        was_in_deep_dd = (
            in_deep_dd.astype(float)
            .rolling(lookback, min_periods=1)
            .max()
            .astype(bool)
        )

        # Confirmation: recently in deep DD + N consecutive up-days
        confirmation = was_in_deep_dd & (consec_up >= c.confirmation_days)

        # Shift to avoid look-ahead (today's signal -> tomorrow's weight)
        confirm_shifted = confirmation.shift(1).fillna(False).astype(bool)

        # Recovery window: stay long for N days after any confirmation
        recovery_active = (
            confirm_shifted
            .astype(float)
            .rolling(c.recovery_window, min_periods=1)
            .max()
            .astype(bool)
        )

        # Assign equal weight to available risk assets
        avail = [t for t in c.risk_tickers if t in prices.columns]
        if not avail:
            return weights

        for t in avail:
            weights.loc[recovery_active, t] = c.per_asset_weight

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
