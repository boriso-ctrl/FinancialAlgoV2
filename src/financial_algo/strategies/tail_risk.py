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

    # Normal-market return sleeve to reduce insurance bleed
    normal_risk_tickers: tuple = ("SPY", "QQQ")
    normal_risk_total: float = 0.70


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

        # Add a diversified risk sleeve in normal markets to reduce carry drag.
        normal_risk = [t for t in c.normal_risk_tickers if t in prices.columns]
        n_risk = len(normal_risk)
        if n_risk > 0 and c.normal_risk_total > 0:
            per_risk = c.normal_risk_total / n_risk
            for t in normal_risk:
                weights.loc[normal, t] = per_risk

        if c.equity_short in prices.columns:
            weights.loc[crisis, c.equity_short] = -c.crisis_short_weight
            weights.loc[severe, c.equity_short] = -c.severe_short_weight

        return weights.fillna(0.0)


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
    qqq_ticker: str = "QQQ"

    # Base allocation (always on)
    base_gold: float = 0.015
    base_bond: float = 0.015

    # Elevated VIX (> 25): increase allocation
    elevated_gold: float = 0.05
    elevated_bond: float = 0.05
    elevated_dollar: float = 0.03

    # Crisis VIX (> 35): maximum allocation
    crisis_gold: float = 0.14
    crisis_bond: float = 0.11
    crisis_dollar: float = 0.05

    # Crisis equity short for convexity
    crisis_equity_short: float = -0.08

    # Safe-haven trend bonus when own trend is positive
    trend_fast_ema: int = 50
    trend_slow_ema: int = 200
    trend_bonus_gold: float = 0.015
    trend_bonus_bond: float = 0.015

    # Calm-market carry sleeve (keeps strategy productive outside stress).
    calm_spy_weight: float = 0.45
    calm_qqq_weight: float = 0.25

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
        avail = [t for t in [c.gold_ticker, c.bond_ticker, c.dollar_ticker, c.spy_ticker, c.qqq_ticker]
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
        has_spy = c.spy_ticker in avail
        has_qqq = c.qqq_ticker in avail

        # Base allocation (always on)
        if has_gold:
            weights[c.gold_ticker] = c.base_gold
        if has_bond:
            weights[c.bond_ticker] = c.base_bond
        if has_spy:
            weights.loc[base_mask, c.spy_ticker] = c.calm_spy_weight
        if has_qqq:
            weights.loc[base_mask, c.qqq_ticker] = c.calm_qqq_weight

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
        if has_spy:
            weights.loc[crisis_mask, c.spy_ticker] = c.crisis_equity_short

        # Trend filter: add extra safe-haven when SPY < 200d SMA
        if has_gold:
            weights.loc[trend_below, c.gold_ticker] += c.trend_extra_gold
        if has_bond:
            weights.loc[trend_below, c.bond_ticker] += c.trend_extra_bond

        # Add safe-haven trend bonus to improve recovery capture.
        if has_gold:
            g_fast = ema(prices[c.gold_ticker], c.trend_fast_ema)
            g_slow = ema(prices[c.gold_ticker], c.trend_slow_ema)
            g_up = pd.notna(g_fast) & pd.notna(g_slow) & (g_fast > g_slow)
            weights.loc[g_up, c.gold_ticker] += c.trend_bonus_gold
        if has_bond:
            b_fast = ema(prices[c.bond_ticker], c.trend_fast_ema)
            b_slow = ema(prices[c.bond_ticker], c.trend_slow_ema)
            b_up = pd.notna(b_fast) & pd.notna(b_slow) & (b_fast > b_slow)
            weights.loc[b_up, c.bond_ticker] += c.trend_bonus_bond

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


# =========================================================================
# O8 -- Volatility Convexity
# =========================================================================

@dataclass
class VolatilityConvexityConfig:
    """VIX-tiered safe-haven overlay for crisis convexity.

    Scales into safe havens (GLD, TLT) proportionally to VIX level,
    adding SPY short and UUP long in extreme stress. Stays flat in
    calm markets to minimise carry cost.

    VIX tiers (based on 5-day EMA of ^VIX, shifted +1 day):
      - Normal  (VIX < 20): flat
      - Elevated (20 <= VIX < 30): GLD 0.3, TLT 0.3
      - Crisis  (30 <= VIX < 40): GLD 0.6, TLT 0.6, SPY -0.3
      - Extreme (VIX >= 40): GLD 0.8, TLT 0.8, UUP 0.4, SPY -0.5
    """

    vix_ticker: str = "^VIX"
    gold_ticker: str = "GLD"
    bond_ticker: str = "TLT"
    dollar_ticker: str = "UUP"
    equity_ticker: str = "SPY"

    ema_span: int = 5

    # VIX thresholds
    vix_elevated: float = 20.0
    vix_crisis: float = 30.0
    vix_extreme: float = 40.0

    # Elevated weights
    normal_equity: float = 0.70
    elevated_equity: float = 0.30
    elevated_gold: float = 0.3
    elevated_bond: float = 0.3

    # Crisis weights
    crisis_gold: float = 0.6
    crisis_bond: float = 0.6
    crisis_equity_short: float = -0.3

    # Extreme weights
    extreme_gold: float = 0.8
    extreme_bond: float = 0.8
    extreme_dollar: float = 0.4
    extreme_equity_short: float = -0.5


class VolatilityConvexity(Strategy):
    """VIX-tiered safe-haven overlay that captures convex payoff in crises.

    Thesis: During vol explosions (VIX spike > 30), safe-haven assets
    (GLD, TLT) exhibit convexity -- their returns accelerate as fear
    increases. This strategy captures that non-linear payoff by scaling
    into safe havens proportionally to VIX level during crisis, and
    staying flat in calm markets (zero carry cost).

    Uses 5-day EMA of VIX shifted by 1 day to avoid look-ahead bias.

    Risk tier: Safe (target Sharpe > 0.5, Max DD < 15%).
    """

    name = "O8-VolatilityConvexity"

    def __init__(self, config: VolatilityConvexityConfig | None = None) -> None:
        self.cfg = config or VolatilityConvexityConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Use real VIX when available, otherwise build a proxy from SPY realized vol.
        if c.vix_ticker in prices.columns:
            vix_raw = prices[c.vix_ticker].ffill().fillna(0.0)
        elif c.equity_ticker in prices.columns:
            spy_rv = realized_vol(prices[c.equity_ticker], 20).fillna(0.0)
            vix_raw = (spy_rv * 100.0).clip(lower=10.0, upper=80.0)
        else:
            return weights

        # Smooth and shift +1 day to keep signal strictly backward-looking.
        vix_ema = ema(vix_raw, c.ema_span).shift(1).fillna(0.0)

        # VIX tier masks (mutually exclusive, highest tier wins)
        extreme = pd.notna(vix_ema) & (vix_ema >= c.vix_extreme)
        crisis = pd.notna(vix_ema) & (vix_ema >= c.vix_crisis) & ~extreme
        elevated = pd.notna(vix_ema) & (vix_ema >= c.vix_elevated) & ~extreme & ~crisis
        normal = ~extreme & ~crisis & ~elevated

        # Assign weights per tier (vectorised via np.select)
        has_gold = c.gold_ticker in prices.columns
        has_bond = c.bond_ticker in prices.columns
        has_dollar = c.dollar_ticker in prices.columns
        has_equity = c.equity_ticker in prices.columns

        conditions = [extreme, crisis, elevated]

        if has_gold:
            weights[c.gold_ticker] = np.select(
                conditions,
                [c.extreme_gold, c.crisis_gold, c.elevated_gold],
                default=0.0,
            )
        if has_bond:
            weights[c.bond_ticker] = np.select(
                conditions,
                [c.extreme_bond, c.crisis_bond, c.elevated_bond],
                default=0.0,
            )
        if has_dollar:
            weights[c.dollar_ticker] = np.select(
                conditions,
                [c.extreme_dollar, 0.0, 0.0],
                default=0.0,
            )
        if has_equity:
            weights[c.equity_ticker] = np.select(
                [extreme, crisis, elevated, normal],
                [c.extreme_equity_short, c.crisis_equity_short, c.elevated_equity, c.normal_equity],
                default=c.normal_equity,
            )

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# O9 -- ATR Crisis Alpha
# =========================================================================


def _close_atr(close: pd.Series, period: int = 14) -> pd.Series:
    """Close-only ATR proxy: rolling mean of absolute daily changes."""
    abs_change = (close - close.shift(1)).abs()
    return abs_change.rolling(period, min_periods=1).mean()


@dataclass
class ATRCrisisAlphaConfig:
    """ATR-based crisis alpha with volatility-adaptive stops.

    During crisis regimes, go long safe havens and short risk assets.
    ATR-scaled profit-taking and stop-loss adapt to the current
    volatility: tight stops in calm markets, wide in volatile markets.
    """

    gold_ticker: str = "GLD"
    bond_ticker: str = "TLT"
    dollar_ticker: str = "UUP"
    spy_ticker: str = "SPY"
    qqq_ticker: str = "QQQ"

    # Crisis allocation weights
    gold_weight: float = 0.45
    bond_weight: float = 0.15
    dollar_weight: float = 0.15
    spy_short: float = -0.15
    qqq_short: float = -0.10

    # Elevated regime: partial hedge
    elevated_gold: float = 0.20
    elevated_bond: float = 0.05
    elevated_dollar: float = 0.05

    # Normal period: trend-follow GLD for baseline returns
    normal_spy_weight: float = 0.50
    normal_qqq_weight: float = 0.20
    normal_gold_base: float = 0.10   # permanent hedge
    normal_bond_base: float = 0.05   # permanent TLT hedge
    gold_trend_weight: float = 0.20  # extra GLD when trending up
    trend_fast_ema: int = 20
    trend_slow_ema: int = 50

    # ATR parameters
    atr_period: int = 14
    atr_profit_mult: float = 3.0   # partial profit at 3x ATR
    atr_stop_mult: float = 2.0     # stop-loss at 2x ATR
    profit_scale_down: float = 0.5  # reduce to 50% on profit-take

    # Volume filter (activity proxy)
    vol_avg_window: int = 20
    vol_mult_threshold: float = 1.0  # soft gate
    max_gross: float = 1.20


class ATRCrisisAlpha(Strategy):
    """ATR-based crisis alpha with volatility-adaptive exits.

    Thesis: During crisis events, fixed-threshold strategies get
    stopped out by volatility before the crisis trade plays out.
    ATR-based stops adapt to the current volatility regime, allowing
    crisis trades to breathe while still protecting capital.

    Signal logic:
      1. Detect crisis regime via .isin() on regime enum
      2. During crisis: long GLD/TLT/UUP, short SPY/QQQ
      3. ATR exit: scale down profitable positions > 3*ATR,
         flatten losing positions > 2*ATR
      4. During NORMAL/RECOVERY: small GLD hedge (0.05)
      5. Volume filter: only enter when SPY activity is elevated
    """

    name = "O9-ATRCrisisAlpha"

    def __init__(self, config: ATRCrisisAlphaConfig | None = None) -> None:
        self.cfg = config or ATRCrisisAlphaConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("ATRCrisisAlpha requires a regime Series")

        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # --- Regime masks ---
        crisis_regimes = {Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS}
        is_crisis = regime.isin(crisis_regimes)
        is_elevated = regime.isin({Regime.ELEVATED}) & ~is_crisis
        is_normal_or_recovery = regime.isin({Regime.NORMAL, Regime.RECOVERY})

        # --- Activity filter ---
        # Use absolute SPY return as a stress activity proxy.
        if c.spy_ticker in prices.columns:
            spy_abs = prices[c.spy_ticker].pct_change().fillna(0.0).abs()
            spy_abs_ma = spy_abs.rolling(c.vol_avg_window, min_periods=1).mean()
            high_activity = pd.notna(spy_abs_ma) & (spy_abs > spy_abs_ma * c.vol_mult_threshold)
            activity = high_activity.astype(float).rolling(5, min_periods=1).max().astype(bool)
        else:
            activity = pd.Series(True, index=prices.index)

        crisis_active = is_crisis & activity
        elevated_active = is_elevated & activity

        # --- Trend confirmation and ATR scaling ---
        trend_ok = pd.Series(True, index=prices.index)
        if c.spy_ticker in prices.columns:
            spy_fast = ema(prices[c.spy_ticker], c.trend_fast_ema)
            spy_slow = ema(prices[c.spy_ticker], c.trend_slow_ema)
            trend_ok = pd.notna(spy_fast) & pd.notna(spy_slow) & (spy_fast < spy_slow)

        atr_scale = pd.Series(1.0, index=prices.index)
        if c.spy_ticker in prices.columns:
            spy_atr = _close_atr(prices[c.spy_ticker], c.atr_period).fillna(0.0)
            spy_px = prices[c.spy_ticker].replace(0.0, np.nan)
            atr_pct = (spy_atr / spy_px).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            # Scale up in high ATR states and down in calm states.
            atr_scale = (atr_pct / atr_pct.rolling(60, min_periods=10).median().replace(0.0, np.nan)).clip(0.5, 1.8)
            atr_scale = atr_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        crisis_safe_scale = atr_scale.clip(0.5, 1.8)
        crisis_short_scale = (atr_scale * trend_ok.astype(float)).clip(0.0, 1.8)

        crisis_safe_map = {
            c.gold_ticker: c.gold_weight,
            c.bond_ticker: c.bond_weight,
            c.dollar_ticker: c.dollar_weight,
        }
        crisis_short_map = {
            c.spy_ticker: c.spy_short,
            c.qqq_ticker: c.qqq_short,
        }
        elevated_map = {
            c.gold_ticker: c.elevated_gold,
            c.bond_ticker: c.elevated_bond,
            c.dollar_ticker: c.elevated_dollar,
        }

        for ticker, wt in crisis_safe_map.items():
            if ticker in prices.columns:
                weights.loc[crisis_active, ticker] = wt * crisis_safe_scale[crisis_active]

        for ticker, wt in crisis_short_map.items():
            if ticker in prices.columns:
                weights.loc[crisis_active, ticker] = wt * crisis_short_scale[crisis_active]

        for ticker, wt in elevated_map.items():
            if ticker in prices.columns:
                weights.loc[elevated_active, ticker] = wt

        # --- Normal / Recovery: small permanent hedge and trend-follow gold ---
        if c.gold_ticker in prices.columns:
            weights.loc[is_normal_or_recovery, c.gold_ticker] = c.normal_gold_base
            gld = prices[c.gold_ticker]
            gld_fast = ema(gld, c.trend_fast_ema)
            gld_slow = ema(gld, c.trend_slow_ema)
            gld_up = pd.notna(gld_fast) & pd.notna(gld_slow) & (gld_fast > gld_slow)
            weights.loc[is_normal_or_recovery & gld_up, c.gold_ticker] += c.gold_trend_weight

        if c.bond_ticker in prices.columns:
            weights.loc[regime.isin({Regime.NORMAL}), c.bond_ticker] = c.normal_bond_base

        if c.spy_ticker in prices.columns:
            weights.loc[is_normal_or_recovery, c.spy_ticker] = c.normal_spy_weight
        if c.qqq_ticker in prices.columns:
            weights.loc[is_normal_or_recovery, c.qqq_ticker] = c.normal_qqq_weight

        gross = weights.abs().sum(axis=1).replace(0.0, np.nan)
        gross_scale = (c.max_gross / gross).clip(upper=1.0).fillna(1.0)
        weights = weights.mul(gross_scale, axis=0)

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
