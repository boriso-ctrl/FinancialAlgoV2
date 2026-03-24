"""Category L: Volatility strategies.

Exploit the volatility risk premium and VIX term-structure dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import realized_vol, ema
from financial_algo.strategies.base import Strategy


# =========================================================================
# L1 — VIX Term-Structure / Vol Risk Premium
# =========================================================================

@dataclass
class VolRiskPremiumConfig:
    """Config for vol risk premium strategy.

    Proxy for VIX term-structure: compare implied vol (VIX) to
    realized vol on SPY. When VIX > realized (contango / risk premium),
    sell vol by being long equities. When VIX < realized (backwardation),
    hedge or go flat.
    """

    equity_ticker: str = "SPY"
    hedge_ticker: str = "TLT"

    realized_vol_window: int = 20
    smooth_span: int = 5

    # Threshold: VIX-to-realized-vol ratio
    contango_threshold: float = 1.2   # VIX/RV > 1.2 → sell vol (long equity)
    backwardation_threshold: float = 0.9  # VIX/RV < 0.9 → buy vol (hedge)

    leverage_contango: float = 1.5    # long equity in contango
    leverage_flat: float = 0.3        # small equity in neutral
    leverage_backwardation: float = 0.0
    hedge_backwardation: float = 0.6  # TLT hedge
    # Momentum confirmation for contango harvesting
    momentum_window: int = 63
    use_momentum_boost: bool = True
    contango_boost: float = 2.0  # extra leverage when contango + momentum up

class VolRiskPremium(Strategy):
    """Harvest the volatility risk premium using VIX-to-realized-vol ratio.

    Thesis: Implied volatility (VIX) consistently exceeds realized vol
    (variance risk premium). When this premium is large (contango),
    be long equities. When it collapses (backwardation), hedge.
    Documented in Carr & Wu (2009), well-known carry trade in vol space.
    """

    name = "L1-VolRiskPremium"

    def __init__(self, config: VolRiskPremiumConfig | None = None) -> None:
        self.cfg = config or VolRiskPremiumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        # Realized vol on equity
        rv = realized_vol(prices[c.equity_ticker], c.realized_vol_window) * 100
        rv = ema(rv, c.smooth_span)

        # We need VIX. Try to use from prices if ^VIX is there,
        # otherwise use regime/vol as proxy
        if "^VIX" in prices.columns:
            vix = prices["^VIX"]
        else:
            # Proxy: realized vol * 1.3 (overestimates, but preserves signal direction)
            vix = rv * 1.3

        # VIX-to-RV ratio
        ratio = vix / rv.replace(0, np.nan)
        ratio = ratio.fillna(1.0)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        contango = ratio >= c.contango_threshold
        backwardation = ratio <= c.backwardation_threshold
        neutral = ~contango & ~backwardation

        weights.loc[contango, c.equity_ticker] = c.leverage_contango
        weights.loc[neutral, c.equity_ticker] = c.leverage_flat
        weights.loc[backwardation, c.hedge_ticker] = c.hedge_backwardation

        return weights.fillna(0.0)


# =========================================================================
# L2 -- VIX / Realized Vol Spread Harvest
# =========================================================================

@dataclass
class VolSpreadConfig:
    """Trade the spread between implied (VIX) and realized vol."""

    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"

    realized_window: int = 20         # realized vol lookback
    vix_rv_smooth: int = 5            # EMA smoothing on ratio

    # When VIX/RV is high, vol premium is rich -> long equity (sell vol)
    # When VIX/RV is low, vol premium collapsed -> reduce / hedge
    rich_threshold: float = 1.3       # VIX/RV > 1.3 -> rich premium
    cheap_threshold: float = 0.95     # VIX/RV < 0.95 -> no premium

    leverage_rich: float = 1.5        # harvest rich vol premium
    leverage_neutral: float = 0.5     # moderate when neutral
    leverage_cheap: float = 0.0       # flat when premium gone
    hedge_cheap: float = 0.5          # TLT hedge when cheap

    # Trend confirmation: require SPY above 50-day SMA for full leverage
    trend_window: int = 50


class VolSpreadHarvest(Strategy):
    """Harvest vol risk premium via VIX-to-realized-vol ratio.

    Thesis: The gap between VIX (implied) and realized vol is the
    volatility risk premium. When VIX is high relative to realized
    vol, the premium is rich -> go long SPY (selling vol). When VIX
    is low vs realized, premium has collapsed -> reduce or hedge.
    Uses trend confirmation to avoid catching falling knives.
    """

    name = "L2-VolSpreadHarvest"

    def __init__(self, config: VolSpreadConfig | None = None) -> None:
        self.cfg = config or VolSpreadConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.equity_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Realized vol (annualized)
        rv = realized_vol(prices[c.equity_ticker], c.realized_window) * 100
        rv_safe = rv.replace(0, np.nan)

        # Get VIX or proxy
        if "^VIX" in prices.columns:
            vix = prices["^VIX"]
        else:
            # Proxy: realized vol * 1.2 (preserves signal direction)
            vix = rv * 1.2

        # VIX-to-RV ratio (smoothed)
        ratio = vix / rv_safe
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        ratio = ema(ratio, c.vix_rv_smooth)

        # Trend filter: SPY above N-day SMA
        sma = prices[c.equity_ticker].rolling(c.trend_window, min_periods=20).mean()
        uptrend = prices[c.equity_ticker] >= sma

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        rich = pd.notna(ratio) & (ratio >= c.rich_threshold)
        cheap = pd.notna(ratio) & (ratio <= c.cheap_threshold)
        neutral = ~rich & ~cheap

        # Rich premium + uptrend -> full harvest
        weights.loc[rich & uptrend, c.equity_ticker] = c.leverage_rich
        # Rich premium + downtrend -> moderate
        weights.loc[rich & ~uptrend, c.equity_ticker] = c.leverage_rich * 0.5
        # Neutral
        weights.loc[neutral, c.equity_ticker] = c.leverage_neutral
        # Cheap premium -> hedge
        if c.safe_ticker in prices.columns:
            weights.loc[cheap, c.safe_ticker] = c.hedge_cheap
        weights.loc[cheap, c.equity_ticker] = c.leverage_cheap

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L3 -- Vol Term Structure (20d vs 60d realized vol)
# =========================================================================

@dataclass
class VolTermStructureConfig:
    """Trade the realized vol term structure."""

    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    short_vol_window: int = 20   # short-term realized vol
    long_vol_window: int = 60    # long-term realized vol
    smooth_span: int = 5         # EMA smoothing on ratio

    # Contango: short-term vol < long-term vol (calm)
    # Backwardation: short-term vol > long-term vol (stress)
    contango_threshold: float = 0.92   # short/long < 0.92
    backwardation_threshold: float = 1.10  # short/long > 1.10

    # Trend filter
    trend_window: int = 50
    momentum_window: int = 63

    # Vol-inverse scaling target
    vol_target: float = 0.15  # target realized vol for scaling

    leverage_contango: float = 1.7     # long equity in contango
    leverage_contango_boost: float = 2.0  # contango + VIX + uptrend + momentum
    leverage_neutral: float = 0.9
    leverage_neutral_up: float = 1.3   # neutral + uptrend
    leverage_backwardation_safe: float = 0.4  # TLT in backwardation
    leverage_backwardation_gold: float = 0.15  # GLD in backwardation
    leverage_backwardation_equity: float = 0.4  # reduced but still invested

    # VIX/RV confirmation (implied-realized spread)
    vix_contango_threshold: float = 1.15  # VIX/RV > 1.15 = rich premium

    # Regime scaling
    regime_elevated_scale: float = 0.65


class VolTermStructure(Strategy):
    """Trade the vol term structure using short vs long realized vol.

    Thesis: When short-term vol < long-term vol (contango), vol is
    likely to stay low -- be long equities. When short-term > long-term
    (backwardation), risk is elevated -- shift to safe havens.
    Enhanced with trend filter, momentum confirmation, and vol-inverse scaling.
    """

    name = "L3-VolTermStructure"

    def __init__(self, config: VolTermStructureConfig | None = None) -> None:
        self.cfg = config or VolTermStructureConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.equity_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        equity = prices[c.equity_ticker]

        short_rv = realized_vol(equity, c.short_vol_window)
        long_rv = realized_vol(equity, c.long_vol_window)

        ratio = short_rv / long_rv.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        ratio = ema(ratio, c.smooth_span)

        # VIX/RV confirmation: implied-realized spread
        if "^VIX" in prices.columns:
            vix = prices["^VIX"]
            rv_pct = short_rv * 100
            vix_rv = vix / rv_pct.replace(0, np.nan)
            vix_rv = vix_rv.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            vix_rv = ema(vix_rv, c.smooth_span)
            vix_contango = vix_rv >= c.vix_contango_threshold
            # Continuous VIX premium scaler
            vix_scale = vix_rv.clip(0.7, 1.3)
        else:
            vix_contango = pd.Series(True, index=prices.index)
            vix_scale = pd.Series(1.0, index=prices.index)

        # Trend filter: equity above N-day SMA
        sma = equity.rolling(c.trend_window, min_periods=20).mean()
        uptrend = equity >= sma

        # Continuous momentum scaling
        mom_ret = equity.pct_change(c.momentum_window).fillna(0.0)
        mom_positive = mom_ret > 0
        mom_scale = (1.0 + mom_ret.clip(-0.10, 0.15))

        # Vol-inverse scaling: leverage up in calm, down in stress
        rv_for_scale = short_rv.fillna(c.vol_target)
        vol_scale = (c.vol_target / rv_for_scale.clip(lower=0.05)).clip(0.6, 1.3)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        contango = pd.notna(ratio) & (ratio < c.contango_threshold)
        backwardation = pd.notna(ratio) & (ratio > c.backwardation_threshold)
        neutral = ~contango & ~backwardation

        # Contango + VIX confirmation + uptrend + momentum: full boost
        contango_full = contango & vix_contango & uptrend & mom_positive
        contango_base = contango & ~contango_full
        weights.loc[contango_full, c.equity_ticker] = c.leverage_contango_boost
        weights.loc[contango_base & uptrend, c.equity_ticker] = c.leverage_contango
        weights.loc[contango_base & ~uptrend, c.equity_ticker] = c.leverage_contango * 0.7

        # Neutral: trend-dependent
        weights.loc[neutral & uptrend, c.equity_ticker] = c.leverage_neutral_up
        weights.loc[neutral & ~uptrend, c.equity_ticker] = c.leverage_neutral

        # Backwardation: safe havens, flat equity
        weights.loc[backwardation, c.equity_ticker] = c.leverage_backwardation_equity
        if c.safe_ticker in prices.columns:
            weights.loc[backwardation, c.safe_ticker] = c.leverage_backwardation_safe
        if c.gold_ticker in prices.columns:
            weights.loc[backwardation, c.gold_ticker] = c.leverage_backwardation_gold

        # Apply momentum, vol-inverse, and VIX premium scaling to equity
        weights[c.equity_ticker] = (
            weights[c.equity_ticker] * mom_scale * vol_scale * vix_scale
        )

        # Regime scaling: defensive in elevated, safe havens in crisis
        if regime is not None:
            from financial_algo.regimes import Regime
            crisis_set = {Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS}
            is_crisis = regime.isin(crisis_set)
            is_elevated = regime.isin([Regime.ELEVATED])
            weights.loc[is_crisis, c.equity_ticker] = 0.0
            if c.safe_ticker in prices.columns:
                weights.loc[is_crisis, c.safe_ticker] = c.leverage_backwardation_safe
            if c.gold_ticker in prices.columns:
                weights.loc[is_crisis, c.gold_ticker] = c.leverage_backwardation_gold
            weights.loc[is_elevated, c.equity_ticker] = (
                weights.loc[is_elevated, c.equity_ticker] * c.regime_elevated_scale
            )

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L5 -- Vol Context Breakout Quality
# =========================================================================

@dataclass
class VolContextBreakoutQualityConfig:
    """Breakout participation conditioned on volatility-quality context."""

    equity_ticker: str = "SPY"
    satellite_ticker: str = "QQQ"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"
    credit_risk_ticker: str = "HYG"
    credit_safe_ticker: str = "LQD"

    breakout_window: int = 63
    trend_window: int = 50
    momentum_window: int = 20
    rv_fast_window: int = 20
    rv_slow_window: int = 60
    vov_window: int = 20
    smooth_span: int = 5

    breakout_buffer: float = 0.01
    quality_floor: float = 0.25

    max_equity_leverage: float = 1.4
    min_equity_leverage: float = 0.10
    hedge_weight_safe: float = 0.45
    hedge_weight_gold: float = 0.20


class VolContextBreakoutQuality(Strategy):
    """Participate in breakouts only when volatility context quality is supportive.

    Thesis: naive breakout systems overtrade in hostile vol states (stress,
    fragile credit, and safety bid). This strategy keeps breakout alpha but
    scales participation continuously using a quality score from vol term
    structure, vol-of-vol, credit risk appetite, and safe-haven demand.
    """

    name = "L5-VolContextBreakoutQuality"

    def __init__(
        self,
        config: VolContextBreakoutQualityConfig | None = None,
    ) -> None:
        self.cfg = config or VolContextBreakoutQualityConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if prices.empty or c.equity_ticker not in prices.columns:
            return weights

        equity = prices[c.equity_ticker]
        satellite = prices.get(c.satellite_ticker, equity)

        breakout_ref = equity.rolling(
            c.breakout_window,
            min_periods=max(20, c.breakout_window // 3),
        ).max().shift(1)
        breakout_strength = (
            equity / breakout_ref.replace(0, np.nan) - 1.0
        ).clip(-0.25, 0.25).fillna(0.0)

        trend_ma = equity.rolling(c.trend_window, min_periods=20).mean()
        trend_strength = (
            equity / trend_ma.replace(0, np.nan) - 1.0
        ).clip(-0.20, 0.20).fillna(0.0)

        sat_momentum = satellite.pct_change(c.momentum_window).fillna(0.0)

        breakout_raw = (
            0.60 * breakout_strength +
            0.25 * trend_strength +
            0.15 * sat_momentum.clip(-0.20, 0.20)
        )
        breakout_score = (breakout_raw / max(c.breakout_buffer, 1e-6)).clip(-1.5, 1.5)
        breakout_participation = (breakout_score.clip(lower=0.0) / 1.5).clip(0.0, 1.0)

        rv_fast = realized_vol(equity, c.rv_fast_window).fillna(0.15)
        rv_slow = realized_vol(equity, c.rv_slow_window).replace(0, np.nan)
        rv_slow = rv_slow.fillna(rv_fast)
        term_ratio = (rv_fast / rv_slow).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        term_quality = ((1.05 - term_ratio) / 0.25).clip(0.0, 1.0)

        rv_delta = rv_fast.diff().abs().fillna(0.0)
        vov = rv_delta.rolling(c.vov_window, min_periods=5).mean().fillna(0.0)
        vov_med = vov.rolling(252, min_periods=40).median()
        vov_med = vov_med.fillna(vov.expanding(min_periods=5).median()).fillna(0.0)
        vov_ratio = (vov / vov_med.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        vov_quality = (1.2 - vov_ratio.fillna(1.0)).clip(0.0, 1.0)

        if "^VIX" in prices.columns:
            vix_ratio = prices["^VIX"] / (rv_fast * 100).replace(0, np.nan)
            vix_ratio = vix_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
            vix_quality = ((vix_ratio - 0.90) / 0.30).clip(0.0, 1.0)
        else:
            vix_quality = term_quality

        if (
            c.credit_risk_ticker in prices.columns
            and c.credit_safe_ticker in prices.columns
        ):
            credit_ratio = (
                prices[c.credit_risk_ticker]
                / prices[c.credit_safe_ticker].replace(0, np.nan)
            )
            credit_mom = credit_ratio.pct_change(c.momentum_window).fillna(0.0)
            credit_quality = (0.5 + 10.0 * credit_mom).clip(0.0, 1.0)
        else:
            credit_quality = pd.Series(0.5, index=prices.index)

        haven_demand = pd.Series(0.0, index=prices.index)
        if c.safe_ticker in prices.columns:
            haven_demand = haven_demand + (
                prices[c.safe_ticker] / equity.replace(0, np.nan)
            ).pct_change(c.momentum_window).fillna(0.0)
        if c.gold_ticker in prices.columns:
            haven_demand = haven_demand + (
                prices[c.gold_ticker] / equity.replace(0, np.nan)
            ).pct_change(c.momentum_window).fillna(0.0)
        haven_quality = (0.5 - 5.0 * haven_demand).clip(0.0, 1.0)

        quality_raw = (
            0.35 * vix_quality +
            0.25 * term_quality +
            0.20 * credit_quality +
            0.10 * haven_quality +
            0.10 * vov_quality
        ).fillna(0.5)
        quality = ema(quality_raw, c.smooth_span).clip(0.0, 1.0)

        equity_scale = c.min_equity_leverage + (
            c.max_equity_leverage - c.min_equity_leverage
        ) * quality
        target_equity = (breakout_participation * equity_scale).clip(0.0, c.max_equity_leverage)

        if regime is not None:
            from financial_algo.regimes import Regime

            crisis = regime.isin([Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS])
            elevated = regime.isin([Regime.ELEVATED])
            target_equity = target_equity.where(~elevated, target_equity * 0.70)
            target_equity = target_equity.where(~crisis, 0.0)

        if c.satellite_ticker in prices.columns:
            weights[c.equity_ticker] = target_equity * 0.65
            weights[c.satellite_ticker] = target_equity * 0.35
        else:
            weights[c.equity_ticker] = target_equity

        hostile = (quality < c.quality_floor) | (term_ratio > 1.15) | (vov_ratio > 1.4)
        defensive_scale = ((1.0 - quality).clip(0.0, 1.0) + hostile.astype(float) * 0.40).clip(0.0, 1.0)

        if c.safe_ticker in prices.columns:
            weights[c.safe_ticker] = c.hedge_weight_safe * defensive_scale
        if c.gold_ticker in prices.columns:
            weights[c.gold_ticker] = c.hedge_weight_gold * defensive_scale

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L6 -- Vol Spike Recovery (VIX spike fade)
# =========================================================================

@dataclass
class VolSpikeRecoveryConfig:
    """Go long when VIX spikes and starts declining."""

    equity_ticker: str = "SPY"
    hedge_ticker: str = "TLT"

    # VIX spike detection
    vix_spike_level: float = 28.0        # VIX must spike above this
    vix_decline_window: int = 5          # look for decline over N days
    vix_decline_pct: float = 0.10        # VIX must decline 10% from recent peak

    # Realized vol proxy if no VIX
    rv_window: int = 20
    rv_spike_z: float = 1.5              # z-score threshold for spike detection

    leverage_spike_long: float = 2.0     # aggressive long on spike recovery
    leverage_neutral: float = 0.3        # small long bias otherwise
    hold_days: int = 20                  # hold position for N days after signal


class VolSpikeRecovery(Strategy):
    """Go aggressively long when VIX spikes and starts declining.

    Thesis: When VIX spikes >28 AND starts declining, the worst is
    likely over and risk assets rally hard. This is a high-conviction,
    time-limited signal. Hold for ~20 trading days then revert to
    neutral. Captures the post-vol-spike bounce.
    """

    name = "L5-VolSpikeRecovery"

    def __init__(self, config: VolSpikeRecoveryConfig | None = None) -> None:
        self.cfg = config or VolSpikeRecoveryConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.equity_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Get VIX or proxy
        if "^VIX" in prices.columns:
            vix = prices["^VIX"].copy()
        else:
            rv = realized_vol(prices[c.equity_ticker], c.rv_window) * 100
            vix = rv  # use RV*100 as VIX proxy

        vix = vix.ffill().fillna(20.0)

        # Rolling peak VIX over decline window
        vix_peak = vix.rolling(c.vix_decline_window, min_periods=1).max()

        # Spike detection: VIX was above spike level recently
        was_spiked = vix_peak >= c.vix_spike_level

        # Decline detection: current VIX is below peak by threshold
        decline_pct = (vix_peak - vix) / vix_peak.replace(0, np.nan)
        decline_pct = decline_pct.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        is_declining = decline_pct >= c.vix_decline_pct

        # Signal fires when spike detected AND VIX declining from peak
        raw_signal = was_spiked & is_declining

        # Hold signal for N days after firing
        # Forward-fill the signal for hold_days using rolling max
        signal = raw_signal.astype(float).rolling(c.hold_days, min_periods=1).max().fillna(0.0)
        active = signal >= 1.0

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Neutral baseline
        weights[c.equity_ticker] = c.leverage_neutral

        # Spike recovery: aggressive long
        weights.loc[active, c.equity_ticker] = c.leverage_spike_long
        if c.hedge_ticker in prices.columns:
            weights.loc[active, c.hedge_ticker] = 0.0

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)

@dataclass
class VolOfVolConfig:
    """Switch regimes based on volatility of volatility."""

    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    vol_window: int = 20
    vov_window: int = 40      # vol-of-vol lookback
    vov_zscore_window: int = 120  # z-score lookback for VoV

    # Vol-of-vol thresholds
    calm_threshold: float = 0.03     # low vol-of-vol -> stable
    turbulent_threshold: float = 0.07  # high vol-of-vol -> unstable

    # Trend / momentum
    trend_window: int = 50
    momentum_window: int = 63

    # Position sizing (continuous, not binary)
    leverage_calm_up: float = 1.8      # calm + uptrend
    leverage_calm_down: float = 1.0    # calm + downtrend
    leverage_neutral: float = 0.7      # middle VoV zone
    leverage_neutral_up: float = 1.0   # middle VoV zone + uptrend
    leverage_turbulent_safe: float = 0.5
    leverage_turbulent_gold: float = 0.3
    leverage_turbulent_equity: float = 0.0

    # VoV z-score for dynamic adjustments
    vov_extreme_z: float = 1.5  # VoV z-score above this = extreme instability

    # Dynamic VoV inverse scaling
    vov_scale_min: float = 0.6
    vov_scale_max: float = 1.3
    vov_scale_lookback: int = 252


class VolOfVolRegime(Strategy):
    """Switch allocation based on the volatility of volatility.

    Thesis: Vol-of-vol (the variability of realized vol itself)
    captures regime uncertainty. Uses VoV as a continuous position sizer
    with trend confirmation. Low VoV + uptrend = max equity. High VoV =
    safe havens. Middle zone gets moderate equity scaled by trend.
    """

    name = "L4-VolOfVolRegime"

    def __init__(self, config: VolOfVolConfig | None = None) -> None:
        self.cfg = config or VolOfVolConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.equity_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        equity = prices[c.equity_ticker]
        rv = realized_vol(equity, c.vol_window)
        # Vol-of-vol: rolling std of realized vol
        vov = rv.rolling(c.vov_window).std().fillna(0.0)

        # VoV z-score for dynamic sizing
        vov_mu = vov.rolling(c.vov_zscore_window, min_periods=40).mean()
        vov_sigma = vov.rolling(c.vov_zscore_window, min_periods=40).std().replace(0, np.nan)
        vov_z = ((vov - vov_mu) / vov_sigma).fillna(0.0)

        # Trend filter
        sma = equity.rolling(c.trend_window, min_periods=20).mean()
        uptrend = equity >= sma

        # Momentum confirmation
        mom_ret = equity.pct_change(c.momentum_window).fillna(0.0)
        mom_positive = mom_ret > 0

        calm = vov < c.calm_threshold
        turbulent = vov > c.turbulent_threshold
        middle = ~calm & ~turbulent

        # Extreme instability: VoV z-score very high within turbulent
        extreme_turbulent = turbulent & (vov_z >= c.vov_extreme_z)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Calm zone: strong equity, boosted by trend + momentum
        weights.loc[calm & uptrend & mom_positive, c.equity_ticker] = c.leverage_calm_up
        weights.loc[calm & uptrend & ~mom_positive, c.equity_ticker] = (
            c.leverage_calm_up + c.leverage_calm_down
        ) / 2
        weights.loc[calm & ~uptrend, c.equity_ticker] = c.leverage_calm_down

        # Middle zone: moderate equity, trend-dependent
        weights.loc[middle & uptrend, c.equity_ticker] = c.leverage_neutral_up
        weights.loc[middle & ~uptrend, c.equity_ticker] = c.leverage_neutral

        # Turbulent zone: safe havens
        weights.loc[turbulent, c.equity_ticker] = c.leverage_turbulent_equity
        if c.safe_ticker in prices.columns:
            weights.loc[turbulent, c.safe_ticker] = c.leverage_turbulent_safe
            # Extreme turbulence: more to safe haven
            weights.loc[extreme_turbulent, c.safe_ticker] = c.leverage_turbulent_safe * 1.3
        if c.gold_ticker in prices.columns:
            weights.loc[turbulent, c.gold_ticker] = c.leverage_turbulent_gold

        # Dynamic VoV inverse scaling: smooth position sizing
        vov_median = vov.rolling(c.vov_scale_lookback, min_periods=60).median()
        vov_median = vov_median.fillna(vov.expanding(min_periods=10).median())
        vov_safe = vov.clip(lower=0.005)
        vov_inv_scale = (vov_median / vov_safe).clip(c.vov_scale_min, c.vov_scale_max)
        weights[c.equity_ticker] = weights[c.equity_ticker] * vov_inv_scale

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L6 -- Cross-Asset Vol Signal
# =========================================================================

@dataclass
class CrossAssetVolSignalConfig:
    """Config for cross-asset volatility dispersion strategy."""

    # Assets whose realized vol we measure for dispersion
    vol_basket: tuple[str, ...] = (
        "SPY", "EEM", "FXI", "DBC", "GLD", "TLT", "BTC-USD",
    )

    # Risk-on targets
    risk_on_tickers: tuple[str, ...] = ("SPY", "QQQ")
    risk_on_weight: float = 0.75  # per ticker in risk-on mode

    # Safe-haven targets
    safe_haven_tickers: tuple[str, ...] = ("GLD", "SLV", "TLT", "SHY")
    safe_haven_weight: float = 0.30  # per ticker in safe-haven mode

    # Realized vol parameters
    rv_window: int = 20       # realized vol lookback per asset
    smooth_span: int = 5      # EMA smoothing on dispersion

    # Dispersion thresholds (z-score based)
    dispersion_zscore_window: int = 252
    low_dispersion_z: float = -0.5   # below = stable correlations = risk-on
    high_dispersion_z: float = 1.0   # above = regime change = safe havens

    # Trend filter
    trend_window: int = 50
    equity_ticker: str = "SPY"

    # Vol-inverse scaling (reduce when overall vol is high)
    vol_target: float = 0.15


class CrossAssetVolSignal(Strategy):
    """Trade cross-asset vol dispersion for regime detection.

    Thesis: When realized vol across diverse asset classes (equities, EM,
    commodities, gold, bonds, crypto) has LOW dispersion (similar vol
    levels), correlations are stable and it is safe to lever up risk
    assets. When dispersion SPIKES (some assets calm, others volatile),
    a regime change is underway -- rotate to safe havens.

    The std-of-vols across asset classes captures regime transitions
    earlier than any single-asset vol measure.
    """

    name = "L6-CrossAssetVolSignal"

    def __init__(self, config: CrossAssetVolSignalConfig | None = None) -> None:
        self.cfg = config or CrossAssetVolSignalConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        # Compute realized vol for each asset in the basket
        available = [t for t in c.vol_basket if t in prices.columns]
        if len(available) < 3:
            # Not enough assets for meaningful dispersion
            weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            if c.equity_ticker in prices.columns:
                weights[c.equity_ticker] = 0.5
            return weights

        rv_df = pd.DataFrame(index=prices.index)
        for ticker in available:
            rv_df[ticker] = realized_vol(prices[ticker], c.rv_window)
        rv_df = rv_df.ffill().fillna(0.0)

        # Cross-asset vol dispersion: std of realized vols across assets
        dispersion = rv_df.std(axis=1)
        dispersion = ema(dispersion, c.smooth_span)

        # Z-score of dispersion for adaptive thresholds
        disp_mu = dispersion.rolling(
            c.dispersion_zscore_window, min_periods=60,
        ).mean()
        disp_sigma = dispersion.rolling(
            c.dispersion_zscore_window, min_periods=60,
        ).std().replace(0, np.nan)
        disp_z = ((dispersion - disp_mu) / disp_sigma).fillna(0.0)

        # Trend filter on SPY
        if c.equity_ticker in prices.columns:
            equity = prices[c.equity_ticker]
            sma = equity.rolling(c.trend_window, min_periods=20).mean()
            uptrend = equity >= sma
        else:
            uptrend = pd.Series(True, index=prices.index)

        # Vol-inverse scaling: reduce gross exposure when median vol is high
        median_rv = rv_df.median(axis=1).fillna(c.vol_target)
        vol_scale = (c.vol_target / median_rv.clip(lower=0.05)).clip(0.5, 1.3)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # Regime classification
        low_disp = pd.notna(disp_z) & (disp_z <= c.low_dispersion_z)
        high_disp = pd.notna(disp_z) & (disp_z >= c.high_dispersion_z)
        mid_disp = ~low_disp & ~high_disp

        # Low dispersion: risk-on
        for ticker in c.risk_on_tickers:
            if ticker in prices.columns:
                weights.loc[low_disp & uptrend, ticker] = c.risk_on_weight
                weights.loc[low_disp & ~uptrend, ticker] = c.risk_on_weight * 0.5

        # High dispersion: safe havens
        for ticker in c.safe_haven_tickers:
            if ticker in prices.columns:
                weights.loc[high_disp, ticker] = c.safe_haven_weight

        # Mid-zone: moderate risk-on, trend-dependent
        for ticker in c.risk_on_tickers:
            if ticker in prices.columns:
                weights.loc[mid_disp & uptrend, ticker] = c.risk_on_weight * 0.5
                weights.loc[mid_disp & ~uptrend, ticker] = c.risk_on_weight * 0.25

        # Apply vol-inverse scaling to risk-on tickers only
        for ticker in c.risk_on_tickers:
            if ticker in prices.columns:
                weights[ticker] = weights[ticker] * vol_scale

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L7 -- PCA Eigen-Factor Regime (Flight-to-Quality Signal)
# =========================================================================

@dataclass
class PCARigimeConfig:
    """Config for PCA-based cross-sectional regime detection strategy.

    Uses rolling PCA on a diversified ETF basket. The second principal
    component (PC2) typically captures flight-to-quality / stress dynamics.
    When PC2 is strongly negative (risk-off), allocate to safe havens.
    When PC2 is positive (risk-on), allocate to risk assets.
    """

    pca_tickers: tuple = (
        "SPY", "QQQ", "EFA", "EEM", "TLT", "GLD", "XLE", "HYG", "UUP", "IEF",
    )
    pca_lookback: int = 252       # rolling window for PCA fit (1 year)
    pca_refit_freq: int = 21      # refit monthly to limit compute cost
    stress_threshold: float = -0.5   # PC2 below this -> risk-off
    riskon_threshold: float = 0.5    # PC2 above this -> risk-on
    score_smooth: int = 5            # EMA days to smooth raw PC2 score

    safe_tickers: tuple = ("GLD", "TLT", "SHY")
    risk_tickers: tuple = ("SPY", "QQQ", "EEM")
    leverage: float = 1.0


class PCARigimeStrategy(Strategy):
    """PCA eigen-factor regime: flight-to-quality detection.

    Thesis: The second principal component (PC2) of a cross-section of
    broad ETF returns captures the 'flight-to-quality' dynamic. When PC2
    is strongly negative, capital is flowing to safe havens (risk-off).
    When PC2 is positive, risk assets are in favour (risk-on).

    PCA is fit on a rolling 252-day window and refitted every 21 trading
    days (monthly) to adapt to evolving factor structure while controlling
    computation cost. Sign is anchored so that positive PC2 = positive
    SPY loading = risk-on (flipped when needed).

    No look-ahead: PCA is fit on returns[t-lookback:t] and the score for
    day t is computed from returns[t] projected on the fitted components.
    """

    name = "L7-PCARigimeAlpha"

    def __init__(self, config: PCARigimeConfig | None = None) -> None:
        self.cfg = config or PCARigimeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        c = self.cfg
        pca_avail = [t for t in c.pca_tickers if t in prices.columns]
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if len(pca_avail) < 5:
            return weights

        safe_avail = [t for t in c.safe_tickers if t in prices.columns]
        risk_avail = [t for t in c.risk_tickers if t in prices.columns]
        if not safe_avail or not risk_avail:
            return weights

        returns = prices[pca_avail].pct_change().fillna(0.0)

        spy_idx = pca_avail.index("SPY") if "SPY" in pca_avail else 0
        n = len(prices)

        # Rolling PCA loop -- iterates over refit dates (column-level, not row-level).
        # Acceptable per vectorization policy: PCA requires sequential fitting.
        pc2_scores: list[float] = [np.nan] * n

        last_scaler: StandardScaler | None = None
        last_pca: PCA | None = None
        last_sign: float = 1.0

        for i in range(c.pca_lookback, n):
            if (i - c.pca_lookback) % c.pca_refit_freq == 0:
                window = returns.iloc[i - c.pca_lookback : i].values
                scaler = StandardScaler()
                window_std = scaler.fit_transform(window)
                pca_model = PCA(n_components=2)
                pca_model.fit(window_std)
                # Anchor sign: positive PC2 loading for SPY = risk-on
                sign = (
                    1.0 if pca_model.components_[1, spy_idx] >= 0 else -1.0
                )
                last_scaler = scaler
                last_pca = pca_model
                last_sign = sign

            if last_scaler is None:
                continue

            today_ret = returns.iloc[i].to_numpy().reshape(1, -1)  # pyright: ignore[reportAttributeAccessIssue]
            today_std = last_scaler.transform(today_ret)
            raw_score = last_pca.transform(today_std)[0, 1]  # pyright: ignore[reportOptionalMemberAccess, reportIndexIssue]
            pc2_scores[i] = last_sign * raw_score

        pc2_series = pd.Series(pc2_scores, index=prices.index, dtype=float)
        pc2_smooth = ema(pc2_series.fillna(0.0), c.score_smooth)

        # Allocate equal weight within each allocation bucket
        safe_w = c.leverage / max(len(safe_avail), 1)
        risk_w = c.leverage / max(len(risk_avail), 1)

        stress_mask = pd.notna(pc2_smooth) & (pc2_smooth < c.stress_threshold)
        riskon_mask = pd.notna(pc2_smooth) & (pc2_smooth > c.riskon_threshold)

        for t in safe_avail:
            weights.loc[stress_mask, t] = safe_w
        for t in risk_avail:
            weights.loc[riskon_mask, t] = risk_w

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L8 -- Return Skewness Signal (contrarian skew fade)
# =========================================================================

@dataclass
class ReturnSkewnessConfig:
    """Config for return-skewness contrarian strategy.

    When recent returns exhibit extreme negative skewness (left-tail
    clustering), the market is pricing in crash risk beyond what
    materialises.  Fade the skew extreme by going long.  When returns
    are extremely positively skewed (euphoria), reduce exposure.
    """

    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    # Skewness calculation
    skew_window: int = 60            # rolling window for skewness
    skew_zscore_lookback: int = 252  # z-score normalisation window

    # Thresholds (z-score of rolling skewness)
    neg_skew_threshold: float = -1.0   # extreme negative skew -> buy
    pos_skew_threshold: float = 1.0    # extreme positive skew -> reduce

    # Trend filter
    trend_window: int = 50

    # Position sizing
    leverage_neg_skew_up: float = 1.6    # neg skew + uptrend (contrarian + trend)
    leverage_neg_skew_down: float = 1.0  # neg skew + downtrend (contrarian only)
    leverage_neutral: float = 0.7        # baseline
    leverage_pos_skew: float = 0.2       # euphoria -- reduce equity
    hedge_pos_skew: float = 0.4          # hedge in euphoria

    # Vol-inverse scaling
    vol_window: int = 20
    vol_target: float = 0.15


class ReturnSkewnessSignal(Strategy):
    """Contrarian skewness fade: buy negative-skew extremes, sell positive.

    Thesis
    ------
    Return distributions exhibit time-varying skewness.  When recent
    returns are negatively skewed (fat left tail, clustering of losses),
    the market has over-priced crash risk and tends to mean-revert
    upward.  When returns are positively skewed (euphoria, blow-off
    rallies), a correction is more likely.

    This exploits a well-documented behavioral bias: investors overweight
    recent left-tail events (Barberis & Huang 2008, Bali et al. 2011).

    Signal: z-score of 60-day rolling skewness vs its 252-day history.
    Extreme negative z -> contrarian long.  Extreme positive z -> reduce.
    Trend filter prevents fighting strong downtrends on the contrarian
    leg.  Vol-inverse scaling reduces exposure in high-vol environments.
    """

    name = "L8-ReturnSkewnessSignal"

    def __init__(self, config: ReturnSkewnessConfig | None = None) -> None:
        self.cfg = config or ReturnSkewnessConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.equity_ticker not in prices.columns:
            return weights

        equity = prices[c.equity_ticker]
        daily_ret = equity.pct_change().fillna(0.0)

        # Rolling skewness
        skew = daily_ret.rolling(c.skew_window, min_periods=30).skew().fillna(0.0)

        # Z-score of skewness for adaptive thresholds
        skew_mu = skew.rolling(c.skew_zscore_lookback, min_periods=60).mean()
        skew_std = skew.rolling(
            c.skew_zscore_lookback, min_periods=60,
        ).std().replace(0, np.nan)
        skew_z = ((skew - skew_mu) / skew_std).fillna(0.0)
        skew_z = skew_z.replace([np.inf, -np.inf], 0.0)

        # Trend filter
        sma = equity.rolling(c.trend_window, min_periods=20).mean()
        uptrend = equity >= sma

        # Vol-inverse scaling
        rv = realized_vol(equity, c.vol_window).fillna(c.vol_target)
        vol_scale = (c.vol_target / rv.clip(lower=0.05)).clip(0.5, 1.3)

        # Regime classification based on skew z-score
        neg_extreme = pd.notna(skew_z) & (skew_z < c.neg_skew_threshold)
        pos_extreme = pd.notna(skew_z) & (skew_z > c.pos_skew_threshold)
        neutral = ~neg_extreme & ~pos_extreme

        # Negative skew extreme: contrarian long (fade the crash fear)
        weights.loc[neg_extreme & uptrend, c.equity_ticker] = c.leverage_neg_skew_up
        weights.loc[neg_extreme & ~uptrend, c.equity_ticker] = c.leverage_neg_skew_down

        # Neutral: baseline
        weights.loc[neutral, c.equity_ticker] = c.leverage_neutral

        # Positive skew extreme: reduce equity, hedge
        weights.loc[pos_extreme, c.equity_ticker] = c.leverage_pos_skew
        if c.safe_ticker in prices.columns:
            weights.loc[pos_extreme, c.safe_ticker] = c.hedge_pos_skew
        if c.gold_ticker in prices.columns:
            weights.loc[pos_extreme, c.gold_ticker] = c.hedge_pos_skew * 0.5

        # Vol-inverse scaling on equity leg only
        weights[c.equity_ticker] = weights[c.equity_ticker] * vol_scale

        # Regime overlay: reduce in crisis periods
        if regime is not None:
            from financial_algo.regimes import Regime
            crisis_set = {Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS}
            is_crisis = regime.isin(crisis_set)
            weights.loc[is_crisis, c.equity_ticker] *= 0.3
            if c.safe_ticker in prices.columns:
                weights.loc[is_crisis, c.safe_ticker] = c.hedge_pos_skew
            if c.gold_ticker in prices.columns:
                weights.loc[is_crisis, c.gold_ticker] = c.hedge_pos_skew * 0.5

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L7 — Implied-Realized Spread (VIX premium z-score harvesting)
# =========================================================================

@dataclass
class ImpliedRealizedSpreadConfig:
    """Harvest the spread between implied vol (VIX) and realized vol.

    When the VIX-to-realized-vol ratio z-score is high (strong contango),
    implied vol is overpriced relative to what materialises.  Harvest by
    being long equities + short bonds.  When the spread compresses or
    inverts, rotate to safe havens (TLT, GLD).

    Momentum filter prevents harvesting contango during drawdowns.
    """

    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    realized_vol_window: int = 20
    zscore_lookback: int = 126
    momentum_window: int = 63

    # Z-score thresholds
    contango_z: float = 1.0
    backwardation_z: float = -0.5

    # Allocations
    contango_equity: float = 1.5
    contango_safe_short: float = -0.3
    backwardation_safe: float = 1.0
    backwardation_gold: float = 0.5
    neutral_equity: float = 0.3


class ImpliedRealizedSpread(Strategy):
    """Harvest the VIX-to-realized-vol spread with z-score timing.

    Thesis
    ------
    When the spread between implied vol (VIX) and realized vol (SPY) is
    abnormally wide, it signals an overpriced insurance premium.  Harvest
    this by being long equities + short vol proxy.  When the spread
    compresses or inverts, rotate to safe havens.

    Signal: z-score of the VIX / 20d-realized-vol ratio over a 126-day
    lookback.  Contango confirmed only when SPY 63-day return > 0
    (momentum filter).
    """

    name = "L7-ImpliedRealizedSpread"

    def __init__(self, config: ImpliedRealizedSpreadConfig | None = None) -> None:
        self.cfg = config or ImpliedRealizedSpreadConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.equity_ticker not in prices.columns:
            return weights

        equity = prices[c.equity_ticker]

        # Realized vol (annualized, as percentage to match VIX units)
        rv = realized_vol(equity, c.realized_vol_window) * 100
        rv_safe = rv.replace(0, np.nan)

        # VIX or proxy
        if "^VIX" in prices.columns:
            vix = prices["^VIX"].ffill().fillna(20.0)
        else:
            vix = rv * 1.3  # conservative proxy

        # VIX / realized vol ratio
        ratio = vix / rv_safe
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Z-score the ratio over lookback window
        ratio_mu = ratio.rolling(c.zscore_lookback, min_periods=40).mean()
        ratio_std = ratio.rolling(
            c.zscore_lookback, min_periods=40,
        ).std().replace(0, np.nan)
        z = ((ratio - ratio_mu) / ratio_std).fillna(0.0)
        z = z.replace([np.inf, -np.inf], 0.0)

        # Momentum confirmation: SPY 63d return > 0
        mom_ret = equity.pct_change(c.momentum_window).fillna(0.0)
        mom_positive = mom_ret > 0

        # Signal classification
        contango = pd.notna(z) & (z > c.contango_z)
        backwardation = pd.notna(z) & (z < c.backwardation_z)
        neutral = ~contango & ~backwardation

        # Contango + momentum: harvest premium (long SPY, short TLT)
        contango_confirmed = contango & mom_positive
        contango_no_mom = contango & ~mom_positive

        weights.loc[contango_confirmed, c.equity_ticker] = c.contango_equity
        if c.safe_ticker in prices.columns:
            weights.loc[contango_confirmed, c.safe_ticker] = c.contango_safe_short

        # Contango without momentum: reduced — just neutral equity
        weights.loc[contango_no_mom, c.equity_ticker] = c.neutral_equity

        # Backwardation: flight to quality (TLT + GLD, zero equity)
        weights.loc[backwardation, c.equity_ticker] = 0.0
        if c.safe_ticker in prices.columns:
            weights.loc[backwardation, c.safe_ticker] = c.backwardation_safe
        if c.gold_ticker in prices.columns:
            weights.loc[backwardation, c.gold_ticker] = c.backwardation_gold

        # Neutral: minimal SPY
        weights.loc[neutral, c.equity_ticker] = c.neutral_equity

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L8 — Vol Regime Clustering (multi-signal state detection)
# =========================================================================

@dataclass
class VolRegimeClusteringConfig:
    """Cluster vol regimes using VIX percentile, vol-of-vol, and realized vol.

    Four states: LOW_VOL, NORMAL, ELEVATED, CRISIS — each with a
    distinct allocation profile.  All classification is vectorized
    (no ML, no loops).
    """

    equity_ticker: str = "SPY"
    growth_ticker: str = "QQQ"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"
    dollar_ticker: str = "UUP"

    vix_percentile_window: int = 252
    vov_window: int = 10
    realized_vol_window: int = 20

    # Percentile thresholds (0-1 scale, applied to rank output)
    low_vol_pct: float = 0.25
    elevated_pct: float = 0.75
    crisis_pct: float = 0.90

    # Realized vol thresholds (annualized decimal)
    low_rv: float = 0.12
    elevated_rv: float = 0.20
    crisis_rv: float = 0.25

    # Allocations per state
    low_vol_equity: float = 1.5
    low_vol_growth: float = 0.5
    normal_equity: float = 0.8
    normal_safe: float = 0.2
    elevated_gold: float = 0.5
    elevated_safe: float = 0.5
    crisis_safe: float = 0.8
    crisis_gold: float = 0.4
    crisis_dollar: float = 0.3


class VolRegimeClustering(Strategy):
    """Cluster vol regimes and allocate to optimal portfolio per state.

    Thesis
    ------
    Cluster vol regimes using multiple signals — VIX level percentile
    (252d), VIX vol-of-vol proxy, and SPY realized vol — to identify
    four states: Low-Vol Rally, Normal, Elevated Uncertainty, Crisis.
    Each state maps to an optimal portfolio allocation.

    All state classification is fully vectorized with priority ordering:
    CRISIS > ELEVATED > LOW_VOL > NORMAL.
    """

    name = "L8-VolRegimeClustering"

    def __init__(self, config: VolRegimeClusteringConfig | None = None) -> None:
        self.cfg = config or VolRegimeClusteringConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.equity_ticker not in prices.columns:
            return weights

        equity = prices[c.equity_ticker]

        # VIX or proxy
        if "^VIX" in prices.columns:
            vix = prices["^VIX"].ffill().fillna(20.0)
        else:
            vix = realized_vol(equity, c.realized_vol_window) * 100
            vix = vix.ffill().fillna(20.0)

        # VIX percentile rank (rolling 252d)
        vix_pct = vix.rolling(
            c.vix_percentile_window, min_periods=60,
        ).rank(pct=True).fillna(0.5)

        # Realized vol (20d, annualized decimal)
        rv = realized_vol(equity, c.realized_vol_window).fillna(0.15)

        # State classification (priority: CRISIS > ELEVATED > LOW_VOL > NORMAL)
        is_crisis = (vix_pct > c.crisis_pct) & (rv > c.crisis_rv)
        is_elevated = ~is_crisis & (
            (vix_pct > c.elevated_pct) | (rv > c.elevated_rv)
        )
        is_low_vol = ~is_crisis & ~is_elevated & (
            (vix_pct < c.low_vol_pct) & (rv < c.low_rv)
        )
        is_normal = ~is_crisis & ~is_elevated & ~is_low_vol

        # LOW_VOL: risk-on — SPY 1.5x, QQQ 0.5x
        weights.loc[is_low_vol, c.equity_ticker] = c.low_vol_equity
        if c.growth_ticker in prices.columns:
            weights.loc[is_low_vol, c.growth_ticker] = c.low_vol_growth

        # NORMAL: balanced — SPY 0.8x, TLT 0.2x
        weights.loc[is_normal, c.equity_ticker] = c.normal_equity
        if c.safe_ticker in prices.columns:
            weights.loc[is_normal, c.safe_ticker] = c.normal_safe

        # ELEVATED: defensive — GLD 0.5x, TLT 0.5x, SPY 0.0x
        weights.loc[is_elevated, c.equity_ticker] = 0.0
        if c.gold_ticker in prices.columns:
            weights.loc[is_elevated, c.gold_ticker] = c.elevated_gold
        if c.safe_ticker in prices.columns:
            weights.loc[is_elevated, c.safe_ticker] = c.elevated_safe

        # CRISIS: full flight to quality — TLT 0.8x, GLD 0.4x, UUP 0.3x
        weights.loc[is_crisis, c.equity_ticker] = 0.0
        if c.safe_ticker in prices.columns:
            weights.loc[is_crisis, c.safe_ticker] = c.crisis_safe
        if c.gold_ticker in prices.columns:
            weights.loc[is_crisis, c.gold_ticker] = c.crisis_gold
        if c.dollar_ticker in prices.columns:
            weights.loc[is_crisis, c.dollar_ticker] = c.crisis_dollar

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L9 — VIX Adaptive Carry (continuous VIX-scaled vol selling)
# =========================================================================

@dataclass
class VIXAdaptiveCarryConfig:
    """Continuously scale vol-carry exposure using exp(-k*(VIX/base-1)).

    Instead of binary on/off for vol selling, smoothly reduce exposure
    as VIX rises.  Hard cutoff at VIX > 35 for tail protection.
    """

    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    realized_vol_window: int = 20
    smooth_span: int = 5

    # VIX-scaling parameters (from Scout / CannedOrgi research)
    vix_k: float = 0.05          # decay constant
    vix_base: float = 20.0       # baseline VIX level
    vix_hard_cutoff: float = 35.0  # force flat above this VIX

    # Contango detection — VIX proxy term structure
    contango_threshold: float = 1.05   # VIX/RV > threshold = contango

    # Base allocations (before VIX scaling)
    base_equity: float = 1.3      # long SPY when selling vol
    base_safe: float = 0.2        # small TLT hedge always on
    base_gold: float = 0.1        # small GLD hedge always on
    flat_safe: float = 0.5        # TLT when flat (backwardation / cutoff)
    flat_gold: float = 0.3        # GLD when flat

    # Trend filter
    trend_window: int = 50

    # Max leverage
    max_leverage: float = 1.5


class VIXAdaptiveCarry(Strategy):
    """Continuously scale vol-carry exposure via exponential VIX factor.

    Thesis
    ------
    The volatility risk premium (implied > realized) is most safely
    harvested in calm markets.  Rather than a binary switch, use a
    continuous scaling factor:

        regime_factor = exp(-k * (VIX / base_vix - 1))

    This naturally reduces exposure as VIX rises and increases it when
    VIX is calm.  Hard cutoff at VIX > 35 for tail protection.
    Contango confirmation (VIX > realized vol) gates the carry signal.

    Target assets: SPY (long when selling vol), TLT + GLD (hedges).
    """

    name = "L9-VIXAdaptiveCarry"

    def __init__(self, config: VIXAdaptiveCarryConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or VIXAdaptiveCarryConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.equity_ticker not in prices.columns:
            return weights

        equity = prices[c.equity_ticker]

        # --- VIX ---
        if "^VIX" in prices.columns:
            vix = prices["^VIX"].ffill().fillna(c.vix_base)
        else:
            rv_proxy = realized_vol(equity, c.realized_vol_window) * 100
            vix = rv_proxy.ffill().fillna(c.vix_base)

        # --- Realized vol (annualised %) ---
        rv = realized_vol(equity, c.realized_vol_window) * 100
        rv = ema(rv, c.smooth_span)
        rv_safe = rv.replace(0, np.nan)

        # --- Contango detection ---
        ratio = vix / rv_safe
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        in_contango = pd.notna(ratio) & (ratio >= c.contango_threshold)

        # --- Trend filter ---
        sma = equity.rolling(c.trend_window, min_periods=20).mean()
        uptrend = equity >= sma

        # --- VIX regime factor: exp(-k * (VIX/base - 1)) ---
        vix_ratio = vix / c.vix_base
        regime_factor = np.exp(-c.vix_k * (vix_ratio - 1.0))
        regime_factor = regime_factor.clip(0.0, 1.2)

        # --- Hard cutoff ---
        above_cutoff = pd.notna(vix) & (vix > c.vix_hard_cutoff)

        # --- Build weights ---
        # Base case: contango + uptrend -> full carry
        carry_signal = in_contango & uptrend
        carry_partial = in_contango & ~uptrend  # contango but no trend

        # Equity: base * regime_factor, scaled by signal quality
        eq_weight = pd.Series(0.0, index=prices.index)
        eq_weight = eq_weight.where(~carry_signal, c.base_equity * regime_factor)
        eq_weight = eq_weight.where(~(carry_partial & ~carry_signal),
                                    c.base_equity * 0.5 * regime_factor)

        # Non-contango: small equity allocation scaled by regime factor
        no_carry = ~in_contango
        eq_weight = eq_weight.where(~(no_carry & uptrend),
                                    0.3 * regime_factor)
        eq_weight = eq_weight.where(~(no_carry & ~uptrend), 0.0)

        # Force flat above cutoff
        eq_weight = eq_weight.where(~above_cutoff, 0.0)

        # Cap at max leverage
        eq_weight = eq_weight.clip(0.0, c.max_leverage)
        weights[c.equity_ticker] = eq_weight

        # --- Hedges ---
        # TLT: inversely proportional to equity exposure
        if c.safe_ticker in prices.columns:
            safe_weight = c.base_safe + (1.0 - regime_factor.clip(0.0, 1.0)) * 0.3
            safe_weight = safe_weight.where(~above_cutoff, c.flat_safe)
            weights[c.safe_ticker] = safe_weight.clip(0.0, 0.8)

        if c.gold_ticker in prices.columns:
            gold_weight = c.base_gold + (1.0 - regime_factor.clip(0.0, 1.0)) * 0.2
            gold_weight = gold_weight.where(~above_cutoff, c.flat_gold)
            weights[c.gold_ticker] = gold_weight.clip(0.0, 0.5)

        # --- Regime overlay ---
        if regime is not None:
            from financial_algo.regimes import Regime
            crisis_set = {Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS}
            is_crisis = regime.isin(crisis_set)
            weights.loc[is_crisis, c.equity_ticker] = 0.0
            if c.safe_ticker in prices.columns:
                weights.loc[is_crisis, c.safe_ticker] = c.flat_safe
            if c.gold_ticker in prices.columns:
                weights.loc[is_crisis, c.gold_ticker] = c.flat_gold

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L10 — Dynamic Vol Regime Switch (smooth blend between sell-vol / buy-vol)
# =========================================================================

@dataclass
class DynamicVolRegimeSwitchConfig:
    """Smoothly blend between vol-selling and vol-buying portfolios.

    Uses the same exp(-k*(VIX/base-1)) factor to create continuous
    weights between risk-on carry (sell vol) and risk-off hedging (buy vol).
    """

    equity_ticker: str = "SPY"
    safe_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    realized_vol_window: int = 20

    # VIX-scaling parameters
    vix_k: float = 0.10
    vix_base: float = 20.0

    # Vol-sell portfolio weights (risk-on carry)
    sell_equity: float = 0.9      # long SPY
    sell_safe: float = -0.1       # slight short TLT (carry)

    # Vol-buy portfolio weights (risk-off hedging)
    buy_equity: float = 0.0       # flat SPY in risk-off
    buy_safe: float = 0.7         # long TLT
    buy_gold: float = 0.4         # long GLD

    # Trend confirmation
    trend_window: int = 50
    momentum_window: int = 63

    # Max leverage
    max_leverage: float = 1.2


class DynamicVolRegimeSwitch(Strategy):
    """Dynamically blend vol-selling and vol-buying portfolios.

    Thesis
    ------
    Different vol strategies work in different VIX regimes.  Rather than
    a discrete switch, smoothly blend between them:

        vol_sell_weight = exp(-k * (VIX/base - 1))   # high when VIX low
        vol_buy_weight  = 1 - vol_sell_weight          # high when VIX high

    Vol-sell portfolio: long SPY, short TLT (risk-on carry).
    Vol-buy portfolio:  long TLT, long GLD, short SPY (risk-off hedging).
    Final weights = vol_sell_weight * sell_portfolio + vol_buy_weight * buy_portfolio.

    This creates a smooth, continuous transition from risk-on to risk-off
    as volatility rises, avoiding the whipsaw of binary regime switches.
    """

    name = "L10-DynamicVolRegimeSwitch"

    def __init__(self, config: DynamicVolRegimeSwitchConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or DynamicVolRegimeSwitchConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.equity_ticker not in prices.columns:
            return weights

        equity = prices[c.equity_ticker]

        # --- VIX ---
        if "^VIX" in prices.columns:
            vix = prices["^VIX"].ffill().fillna(c.vix_base)
        else:
            rv_proxy = realized_vol(equity, c.realized_vol_window) * 100
            vix = rv_proxy.ffill().fillna(c.vix_base)

        # --- Blending weights ---
        vix_ratio = vix / c.vix_base
        vol_sell_w = np.exp(-c.vix_k * (vix_ratio - 1.0))
        vol_sell_w = vol_sell_w.clip(0.0, 1.0)
        vol_buy_w = 1.0 - vol_sell_w

        # --- Trend confirmation ---
        sma = equity.rolling(c.trend_window, min_periods=20).mean()
        uptrend = equity >= sma
        mom_ret = equity.pct_change(c.momentum_window).fillna(0.0)
        # Trend modifier: boost sell-vol in uptrend, boost buy-vol in downtrend
        trend_mod = np.where(uptrend, 1.1, 0.9)

        # --- Blended portfolio ---
        # Equity: sell_equity * sell_w + buy_equity * buy_w
        eq_raw = c.sell_equity * vol_sell_w * trend_mod + c.buy_equity * vol_buy_w
        # TLT: sell_safe * sell_w + buy_safe * buy_w
        safe_raw = pd.Series(0.0, index=prices.index)
        if c.safe_ticker in prices.columns:
            safe_raw = c.sell_safe * vol_sell_w + c.buy_safe * vol_buy_w
        # GLD: only from buy portfolio
        gold_raw = pd.Series(0.0, index=prices.index)
        if c.gold_ticker in prices.columns:
            gold_raw = c.buy_gold * vol_buy_w

        # --- Leverage cap ---
        gross = eq_raw.abs() + safe_raw.abs() + gold_raw.abs()
        scale = np.where(gross > c.max_leverage,
                         c.max_leverage / gross.replace(0, np.nan).fillna(1.0),
                         1.0)
        scale = pd.Series(scale, index=prices.index).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)

        weights[c.equity_ticker] = eq_raw * scale
        if c.safe_ticker in prices.columns:
            weights[c.safe_ticker] = safe_raw * scale
        if c.gold_ticker in prices.columns:
            weights[c.gold_ticker] = gold_raw * scale

        # --- Regime overlay ---
        if regime is not None:
            from financial_algo.regimes import Regime
            crisis_set = {Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS}
            is_crisis = regime.isin(crisis_set)
            is_elevated = regime.isin([Regime.ELEVATED])
            # Crisis: full risk-off
            weights.loc[is_crisis, c.equity_ticker] = 0.0
            if c.safe_ticker in prices.columns:
                weights.loc[is_crisis, c.safe_ticker] = 0.8
            if c.gold_ticker in prices.columns:
                weights.loc[is_crisis, c.gold_ticker] = 0.5
            # Elevated: tilt toward buy-vol
            weights.loc[is_elevated] = weights.loc[is_elevated] * 0.7

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# L1b — Vol Risk Premium Adaptive (VIX-scaled version of L1)
# =========================================================================

class VolRiskPremiumAdaptive(Strategy):
    """VIX-adaptive version of L1 VolRiskPremium for A/B testing.

    Thesis
    ------
    Same as L1 — harvest the vol risk premium via VIX-to-realized-vol
    ratio — but with continuous VIX scaling applied to all weights:

        regime_factor = exp(-0.05 * (VIX / 20 - 1))

    This reduces exposure during VIX spikes and increases it in calm
    periods, avoiding the binary on/off behaviour of the original L1.

    The original L1 class is untouched; this is a separate class to
    allow side-by-side comparison.
    """

    name = "L1b-VolRiskPremiumAdaptive"

    def __init__(self, config: VolRiskPremiumConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or VolRiskPremiumConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if c.equity_ticker not in prices.columns:
            return weights

        # --- Realized vol on equity ---
        rv = realized_vol(prices[c.equity_ticker], c.realized_vol_window) * 100
        rv = ema(rv, c.smooth_span)

        # --- VIX ---
        if "^VIX" in prices.columns:
            vix = prices["^VIX"].ffill().fillna(20.0)
        else:
            vix = rv * 1.3

        # --- VIX-to-RV ratio ---
        ratio = vix / rv.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # --- VIX regime factor ---
        vix_base = 20.0
        regime_factor = np.exp(-0.05 * (vix / vix_base - 1.0))
        regime_factor = regime_factor.clip(0.1, 1.2)

        # --- Signal classification (same as L1) ---
        contango = pd.notna(ratio) & (ratio >= c.contango_threshold)
        backwardation = pd.notna(ratio) & (ratio <= c.backwardation_threshold)
        neutral = ~contango & ~backwardation

        # --- Base weights (same as L1) ---
        weights.loc[contango, c.equity_ticker] = c.leverage_contango
        weights.loc[neutral, c.equity_ticker] = c.leverage_flat
        weights.loc[backwardation, c.hedge_ticker] = c.hedge_backwardation

        # --- Apply VIX regime factor to ALL weights ---
        for col in weights.columns:
            weights[col] = weights[col] * regime_factor

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
