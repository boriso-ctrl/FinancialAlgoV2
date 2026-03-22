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
# L5 -- Vol Spike Recovery (VIX spike fade)
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
