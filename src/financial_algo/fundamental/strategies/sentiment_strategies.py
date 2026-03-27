"""Sentiment-aware crisis strategies.

These strategies use fundamental/sentiment data alongside price data
to make trading decisions based on market *context*, not just
price movements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.fundamental.data.news_feeds import build_synthetic_sentiment
from financial_algo.fundamental.indicators import (
    fear_greed_composite,
    sentiment_momentum,
    sentiment_zscore,
)
from financial_algo.indicators import realized_vol
from financial_algo.fundamental.signals import (
    crisis_onset_signal,
    divergence_signal,
    fear_greed_signal,
    recovery_signal,
    sentiment_trend_signal,
)
from financial_algo.strategies.base import Strategy


# =========================================================================
# G1 — Sentiment Crisis Alpha
# =========================================================================

@dataclass
class SentimentCrisisConfig:
    """Config for sentiment-driven crisis strategy.
    
    REWORK (Sprint 15.1 Phase 13):
    - Multi-signal sentiment composition (VIX level + momentum + vol-of-vol)
    - Removed equity shorting (long TLT instead, safer in crisis)
    - Two-signal crisis confirmation (avoid false positives)
    - Better sentiment z-score: rolling calibration per quarter
    - NaN-safe signal construction
    """

    equity_ticker: str = "SPY"
    hedge_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    # Signal thresholds
    fear_z_threshold: float = 1.5       # z-score threshold for fear spike
    velocity_threshold: float = 1.5     # news velocity acceleration
    recovery_momentum: float = 0.5      # recovery momentum threshold
    
    # NEW: Multi-signal sentiment composition weights
    vix_level_weight: float = 0.40      # VIX extreme high = fear
    vix_momentum_weight: float = 0.30   # VIX rising trend = accelerating fear
    vol_of_vol_weight: float = 0.20     # Vol-of-vol spike = regime stress
    trend_weight: float = 0.10          # SPY downtrend = additional fear

    # Position sizing
    leverage_crisis: float = 0.0        # CHANGED: 0.0 equity (NO SHORTS)
    leverage_crisis_hedge: float = 2.0  # Hedge equity exposure instead
    leverage_recovery: float = 2.5
    leverage_neutral: float = 0.5

    # Refinements
    trend_window: int = 50           
    momentum_window: int = 63        
    vol_window: int = 20             
    normal_uptrend_boost: float = 1.4
    normal_base: float = 0.9
    normal_downtrend: float = 0.3
    normal_hedge_down: float = 0.2
    
    # NEW: Sentiment calibration window (quarterly)
    sentiment_cal_window: int = 63   # Recalibrate every quarter


class SentimentCrisisAlpha(Strategy):
    """Trade crisis onset and recovery using multi-signal sentiment.

    REWORK: Improved from 0.79 to target 0.88 Sharpe by:
    1. Multi-signal sentiment composition (VIX + momentum + vol-of-vol)
    2. Removing equity shorting (long TLT in crisis instead)
    3. Two-signal crisis confirmation (VIX extreme + vol spike)
    4. Rolling sentiment calibration (adaptive thresholds)
    5. NaN-safe signal construction

    States:
    - CRISIS: Multi-indicator fear spike -> LONG hedges (TLT/GLD), ZERO equity
    - RECOVERY: Sentiment improving + trend positive -> LEVERAGED long SPY
    - NEUTRAL: Trend + momentum-driven with vol scaling
    """

    name = "SentimentCrisisAlpha"

    def __init__(self, config: SentimentCrisisConfig | None = None) -> None:
        self.cfg = config or SentimentCrisisConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        if c.equity_ticker not in prices.columns:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        spy_price = prices[c.equity_ticker]
        n = len(prices)

        # --- Multi-signal sentiment composition (NEW) ---
        # VIX level (extreme high = fear)
        if "^VIX" in prices.columns:
            vix = prices["^VIX"].ffill().fillna(20.0)
        else:
            from financial_algo.indicators import realized_vol
            vix = realized_vol(spy_price, 20) * 100
        
        vix_z_20 = (vix - vix.rolling(252, min_periods=60).mean()) / vix.rolling(252, min_periods=60).std().replace(0, np.nan)
        vix_z_20 = vix_z_20.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        vix_fear_signal = np.clip(vix_z_20, -2, 2)  # Clip extremes

        # VIX momentum (rising VIX = accelerating fear)
        vix_change = vix.pct_change(5).fillna(0.0)
        vix_momentum_z = (vix_change - vix_change.rolling(60, min_periods=20).mean()) / vix_change.rolling(60, min_periods=20).std().replace(0, np.nan)
        vix_momentum_z = vix_momentum_z.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        vix_accel_signal = np.clip(vix_momentum_z, -2, 2)

        # Vol-of-vol spike detection
        from financial_algo.indicators import realized_vol
        rv = realized_vol(spy_price, 20).fillna(0.15)
        vov = rv.rolling(40).std().fillna(0.0)
        vov_z = (vov - vov.rolling(252, min_periods=60).mean()) / vov.rolling(252, min_periods=60).std().replace(0, np.nan)
        vov_z = vov_z.fillna(0.0).replace([np.inf, -np.inf], 0.0)
        vov_signal = np.clip(vov_z, -2, 2)

        # Trend component (SPY below SMA = downtrend fear)
        sma_200 = spy_price.rolling(200, min_periods=100).mean()
        downtrend = (spy_price < sma_200).astype(float)
        trend_signal = downtrend * 2 - 1  # Range: [-1, 1]

        # Composite fear signal (higher = more fear)
        composite_fear = (
            c.vix_level_weight * vix_fear_signal +
            c.vix_momentum_weight * vix_accel_signal +
            c.vol_of_vol_weight * vov_signal +
            c.trend_weight * trend_signal
        )

        # Generate legacy signals (for compatibility)
        if sentiment_df is None:
            from financial_algo.fundamental.data.news_feeds import build_synthetic_sentiment
            sentiment_df = build_synthetic_sentiment(prices, vix)

        from financial_algo.fundamental.signals import crisis_onset_signal, recovery_signal
        crisis_sig = crisis_onset_signal(sentiment_df, fear_z_threshold=c.fear_z_threshold, velocity_threshold=c.velocity_threshold)
        recov_sig = recovery_signal(sentiment_df, momentum_threshold=c.recovery_momentum)

        # --- Two-signal crisis confirmation (NEW) ---
        # Crisis only when BOTH composite fear is extreme AND crisis_sig fires
        crisis_extreme = composite_fear > 1.5  # 90th+ percentile of fear
        crisis_confirmed = (crisis_sig == 1) & crisis_extreme

        # --- Trend filters ---
        sma_trend = spy_price.rolling(c.trend_window, min_periods=20).mean()
        uptrend_200 = spy_price >= sma_200
        uptrend_short = spy_price >= sma_trend

        # Momentum
        mom_ret = spy_price.pct_change(c.momentum_window).fillna(0.0)
        mom_positive = mom_ret > 0

        # Vol scaling
        vol_scale = (0.15 / rv.clip(lower=0.05)).clip(0.5, 1.5)

        tickers = [c.equity_ticker, c.hedge_ticker, c.gold_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        # --- Normal mode: trend + momentum driven ---
        normal_up_mom = uptrend_short & mom_positive
        normal_up = uptrend_short & ~mom_positive
        normal_down = ~uptrend_short

        w[c.equity_ticker] = np.where(
            normal_up_mom, c.normal_uptrend_boost,
            np.where(normal_up, c.normal_base,
                     np.where(normal_down, c.normal_downtrend, c.normal_base))
        )
        w.loc[normal_down, c.hedge_ticker] = c.normal_hedge_down

        # --- Crisis mode: override (CHANGED: no shorts, long hedges instead) ---
        w.loc[crisis_confirmed, c.equity_ticker] = 0.0  # Zero equity
        w.loc[crisis_confirmed, c.hedge_ticker] = c.leverage_crisis_hedge  # Long TLT instead
        if c.gold_ticker in prices.columns:
            w.loc[crisis_confirmed, c.gold_ticker] = c.leverage_crisis_hedge * 0.5

        # --- Recovery mode: override ---
        recovery_mask = recov_sig == 1
        recov_up = recovery_mask & uptrend_200
        recov_down = recovery_mask & ~uptrend_200
        w.loc[recov_up, c.equity_ticker] = c.leverage_recovery
        w.loc[recov_up, c.hedge_ticker] = 0.0
        w.loc[recov_up, c.gold_ticker] = 0.0
        w.loc[recov_down, c.equity_ticker] = c.leverage_recovery * 0.5

        # Apply vol scaling to equity positions
        w[c.equity_ticker] = w[c.equity_ticker] * vol_scale

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# G2 — Fear & Greed Contrarian
# =========================================================================

@dataclass
class FearGreedConfig:
    """Config for fear/greed contrarian strategy."""

    equity_ticker: str = "QQQ"
    hedge_ticker: str = "TLT"

    fear_threshold: float = -40.0     # extreme fear -> buy
    greed_threshold: float = 40.0     # extreme greed -> hedge

    leverage_fear_buy: float = 2.0    # buy aggressively on extreme fear
    leverage_greed_short: float = -1.0  # short on extreme greed
    leverage_neutral: float = 0.5

    # Confirmation: require fear to persist for N days
    confirmation_days: int = 3


class FearGreedContrarian(Strategy):
    """Buy when others are fearful, sell when greedy.

    Uses composite Fear & Greed index as primary signal.
    Requires multi-day confirmation to avoid whipsaws.
    """

    name = "FearGreedContrarian"

    def __init__(self, config: FearGreedConfig | None = None) -> None:
        self.cfg = config or FearGreedConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        if sentiment_df is None:
            sentiment_df = build_synthetic_sentiment(prices)

        fg = fear_greed_composite(sentiment_df)

        # Multi-day confirmation
        extreme_fear = (fg <= c.fear_threshold).rolling(c.confirmation_days).min().fillna(0).astype(bool)
        extreme_greed = (fg >= c.greed_threshold).rolling(c.confirmation_days).min().fillna(0).astype(bool)

        tickers = [c.equity_ticker, c.hedge_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        # Neutral baseline
        w[c.equity_ticker] = c.leverage_neutral

        # Extreme fear: contrarian buy
        w.loc[extreme_fear, c.equity_ticker] = c.leverage_fear_buy
        w.loc[extreme_fear, c.hedge_ticker] = 0.0

        # Extreme greed: reduce / hedge
        w.loc[extreme_greed, c.equity_ticker] = c.leverage_greed_short
        w.loc[extreme_greed, c.hedge_ticker] = abs(c.leverage_greed_short) * 0.5

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# G3 — Sentiment Divergence (VIX z-score contrarian)
# =========================================================================

@dataclass
class SentimentDivergenceConfig:
    """Config for VIX z-score contrarian strategy."""

    equity_ticker: str = "SPY"
    hedge_ticker: str = "TLT"

    # VIX z-score lookback and thresholds
    vix_zscore_window: int = 60
    fear_z_threshold: float = 2.0       # extreme fear -> contrarian long
    complacency_z_threshold: float = -1.6  # complacency -> reduce

    # Trend / vol scaling
    trend_window: int = 50
    vol_window: int = 20

    # Position sizing (long-biased, enhanced normal mode)
    leverage_base: float = 0.8         # baseline long equity
    leverage_base_up: float = 1.35      # baseline + uptrend
    leverage_fear_long: float = 2.2    # contrarian long on fear extreme
    leverage_fear_long_up: float = 2.8 # fear extreme + uptrend confirmation
    leverage_complacent: float = 0.15  # minimal on complacency
    hedge_complacent: float = 0.4      # TLT hedge when complacent
    hedge_base_down: float = 0.15      # small TLT hedge in downtrend


class SentimentDivergence(Strategy):
    """VIX z-score contrarian: buy fear extremes, reduce on complacency.

    Thesis: When VIX z-score is very high (extreme fear), markets tend
    to mean-revert -> go long SPY. When VIX z-score is very low
    (complacency), risk is elevated -> reduce exposure. Enhanced with
    trend filter and vol scaling for tighter risk control.
    """

    name = "SentimentDivergence"

    def __init__(self, config: SentimentDivergenceConfig | None = None) -> None:
        self.cfg = config or SentimentDivergenceConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        if sentiment_df is None:
            sentiment_df = build_synthetic_sentiment(prices)

        # Use VIX fear z-score from sentiment
        fear = sentiment_df.get(
            "fear_index",
            pd.Series(0.0, index=prices.index),
        )
        fear_z = sentiment_zscore(fear, c.vix_zscore_window)

        # Trend filter
        equity = prices.get(c.equity_ticker, prices.iloc[:, 0])
        sma = equity.rolling(c.trend_window, min_periods=20).mean()
        uptrend = equity >= sma

        # Vol scaling: reduce when vol is elevated
        rv = realized_vol(equity, c.vol_window).fillna(0.15)
        vol_scale = (0.15 / rv.clip(lower=0.05)).clip(0.5, 1.5)

        tickers = [c.equity_ticker, c.hedge_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        # Baseline: trend-dependent
        w[c.equity_ticker] = np.where(uptrend, c.leverage_base_up, c.leverage_base)
        w.loc[~uptrend, c.hedge_ticker] = c.hedge_base_down

        # Extreme fear: contrarian long (buy the fear)
        extreme_fear = pd.notna(fear_z) & (fear_z >= c.fear_z_threshold)
        fear_up = extreme_fear & uptrend
        fear_down = extreme_fear & ~uptrend
        w.loc[fear_up, c.equity_ticker] = c.leverage_fear_long_up
        w.loc[fear_up, c.hedge_ticker] = 0.0
        w.loc[fear_down, c.equity_ticker] = c.leverage_fear_long
        w.loc[fear_down, c.hedge_ticker] = 0.0

        # Complacency: reduce exposure, add hedge
        complacent = pd.notna(fear_z) & (fear_z <= c.complacency_z_threshold)
        w.loc[complacent, c.equity_ticker] = c.leverage_complacent
        w.loc[complacent, c.hedge_ticker] = c.hedge_complacent

        # Apply vol scaling to equity positions
        w[c.equity_ticker] = w[c.equity_ticker] * vol_scale

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# G4 — Sentiment-Enhanced Regime
# =========================================================================

@dataclass
class SentimentRegimeConfig:
    """Config for sentiment-enhanced regime strategy."""

    equity_ticker: str = "QQQ"
    hedge_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    # Sentiment thresholds for regime override (tightened)
    extreme_fear_z: float = 2.2      # override to crisis
    extreme_greed_z: float = 2.2     # override to reduce exposure

    # Position sizing per state (conservative to cap DD)
    leverage_bull: float = 1.0
    leverage_neutral: float = 0.5
    leverage_crisis: float = 0.0     # flat equity in crisis (no short)
    leverage_recovery: float = 1.15   # moderate long on recovery
    hedge_crisis: float = 0.5
    hedge_gold_crisis: float = 0.2

    # Vol and trend controls
    sentiment_lookback: int = 60
    trend_window: int = 50
    vol_window: int = 20
    max_leverage: float = 1.0         # hard cap on equity leverage


class SentimentEnhancedRegime(Strategy):
    """Enhance technical regime detection with sentiment overlay + trend filter.

    Uses the technical regime (from regimes.py) as the base, but allows
    sentiment to OVERRIDE or CONFIRM regime transitions.

    Conservative sizing to cap max drawdown. No short equity positions.
    Vol-scaled to reduce exposure when realized vol is elevated.
    """

    name = "SentimentEnhancedRegime"

    def __init__(self, config: SentimentRegimeConfig | None = None) -> None:
        self.cfg = config or SentimentRegimeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        from financial_algo.regimes import Regime

        c = self.cfg

        if sentiment_df is None:
            sentiment_df = build_synthetic_sentiment(prices)

        fear = sentiment_df.get("fear_index", pd.Series(0.0, index=prices.index))
        fear_z = sentiment_zscore(fear, c.sentiment_lookback).fillna(0.0)
        sent_mom = sentiment_momentum(
            sentiment_df.get("sentiment_score", pd.Series(0.0, index=prices.index)),
            fast=5, slow=20,
        ).fillna(0.0)

        # Trend filter
        spy_price = prices.get(c.equity_ticker, prices.iloc[:, 0])
        sma = spy_price.rolling(c.trend_window, min_periods=20).mean()
        uptrend = spy_price >= sma

        # Vol scaling: reduce in high-vol regimes
        rv = realized_vol(spy_price, c.vol_window).fillna(0.15)
        vol_scale = (0.15 / rv.clip(lower=0.05)).clip(0.5, 0.90)

        tickers = [c.equity_ticker, c.hedge_ticker, c.gold_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        # --- Vectorized regime classification ---
        if regime is not None:
            crisis_regimes = {Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS}
            is_tech_crisis = regime.isin(crisis_regimes)
            is_recovery = regime.isin([Regime.RECOVERY])
            is_elevated = regime.isin([Regime.ELEVATED])
            is_normal = regime.isin([Regime.NORMAL])
        else:
            is_tech_crisis = pd.Series(False, index=prices.index)
            is_recovery = pd.Series(False, index=prices.index)
            is_elevated = pd.Series(False, index=prices.index)
            is_normal = pd.Series(True, index=prices.index)

        # Crisis: technical crisis OR extreme fear z-score
        crisis_mask = is_tech_crisis | (fear_z >= c.extreme_fear_z)
        crisis_improving = crisis_mask & (sent_mom > 0.5)
        crisis_full = crisis_mask & ~crisis_improving

        # Recovery: technical recovery
        recovery_mask = is_recovery & ~crisis_mask

        # Elevated
        elevated_fear = is_elevated & (fear_z >= 1.0) & ~crisis_mask
        elevated_neutral = is_elevated & (fear_z < 1.0) & ~crisis_mask

        # Normal with sentiment
        normal_fear = is_normal & (fear_z >= c.extreme_fear_z) & ~crisis_mask
        normal_greed = is_normal & (-fear_z >= c.extreme_greed_z) & ~crisis_mask & ~normal_fear
        normal_base = is_normal & ~crisis_mask & ~normal_fear & ~normal_greed

        # --- Assign weights vectorized (conservative, no shorts) ---
        # Crisis improving -> cautious long
        w.loc[crisis_improving, c.equity_ticker] = c.leverage_recovery * 0.7
        w.loc[crisis_improving, c.hedge_ticker] = c.hedge_crisis * 0.3

        # Crisis full -> flat equity, safe havens
        w.loc[crisis_full, c.equity_ticker] = c.leverage_crisis
        w.loc[crisis_full, c.hedge_ticker] = c.hedge_crisis
        w.loc[crisis_full, c.gold_ticker] = c.hedge_gold_crisis

        # Recovery -> moderate long, trend-dependent
        w.loc[recovery_mask & uptrend, c.equity_ticker] = c.leverage_recovery
        w.loc[recovery_mask & ~uptrend, c.equity_ticker] = c.leverage_recovery * 0.7

        # Elevated + fear -> defensive
        w.loc[elevated_fear, c.equity_ticker] = c.leverage_neutral * 0.3
        w.loc[elevated_fear, c.hedge_ticker] = c.hedge_crisis * 0.6
        w.loc[elevated_fear, c.gold_ticker] = c.hedge_gold_crisis * 0.5

        # Elevated neutral
        w.loc[elevated_neutral, c.equity_ticker] = c.leverage_neutral

        # Normal with fear warning
        w.loc[normal_fear, c.equity_ticker] = c.leverage_neutral * 0.4
        w.loc[normal_fear, c.hedge_ticker] = c.hedge_crisis * 0.4

        # Normal with greed -> reduce
        w.loc[normal_greed, c.equity_ticker] = c.leverage_neutral * 0.6

        # Normal base -> bull, trend-dependent
        w.loc[normal_base & uptrend, c.equity_ticker] = c.leverage_bull
        w.loc[normal_base & ~uptrend, c.equity_ticker] = c.leverage_bull * 0.6

        # Apply vol scaling to equity
        w[c.equity_ticker] = (w[c.equity_ticker] * vol_scale).clip(
            lower=-c.max_leverage, upper=c.max_leverage
        )

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# G5 — Crypto Sentiment Divergence
# =========================================================================

@dataclass
class CryptoSentimentDivergenceConfig:
    """Config for crypto sentiment divergence strategy."""

    btc_ticker: str = "BTC-USD"
    eth_ticker: str = "ETH-USD"

    # Equity tickers for risk-on / risk-off
    equity_tickers: tuple[str, ...] = ("SPY", "QQQ")
    safe_haven_tickers: tuple[str, ...] = ("GLD", "TLT")

    # Crypto momentum lookback
    crypto_mom_window: int = 20

    # Divergence threshold: absolute difference in normalised returns
    divergence_threshold: float = 0.15  # 15% return divergence over window

    # VIX proxy: SPY realized vol threshold for "high VIX"
    vix_rv_window: int = 20
    high_vol_percentile_window: int = 252
    high_vol_percentile: float = 0.70  # above 70th percentile = high vol

    # Position sizing
    risk_on_weight: float = 0.80       # equity weight in crypto-strength mode
    safe_haven_weight: float = 0.40    # safe-haven weight in stress mode
    neutral_equity_weight: float = 0.40  # baseline equity weight

    # Trend filter
    trend_window: int = 50
    equity_ref: str = "SPY"


class CryptoSentimentDivergence(Strategy):
    """Use BTC/ETH divergence as crypto sentiment barometer.

    Thesis: BTC and ETH normally co-move strongly. When they diverge
    significantly (one up, one down over 20 days), this signals internal
    crypto market stress -- a leading indicator for broader risk-off
    sentiment. Combined with high VIX (realized vol proxy), this
    triggers a defensive rotation to GLD+TLT.

    When BTC and ETH both show positive momentum, crypto sentiment is
    strong -- a risk-on signal that supports overweighting SPY+QQQ.

    ETH-USD data starts from 2017-11-09 -- strategy gracefully degrades
    to neutral when ETH data is unavailable.
    """

    name = "G5-CryptoSentimentDivergence"

    def __init__(
        self, config: CryptoSentimentDivergenceConfig | None = None,
    ) -> None:
        self.cfg = config or CryptoSentimentDivergenceConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        sentiment_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        has_btc = c.btc_ticker in prices.columns
        has_eth = c.eth_ticker in prices.columns

        # If BTC not available, return neutral equity weight
        if not has_btc:
            if c.equity_ref in prices.columns:
                weights[c.equity_ref] = c.neutral_equity_weight
            return weights

        # Crypto momentum
        btc_ret = prices[c.btc_ticker].pct_change(c.crypto_mom_window).fillna(0.0)

        if has_eth:
            eth_ret = prices[c.eth_ticker].pct_change(c.crypto_mom_window).fillna(0.0)
            eth_available = prices[c.eth_ticker].notna()
        else:
            eth_ret = pd.Series(0.0, index=prices.index)
            eth_available = pd.Series(False, index=prices.index)

        # Divergence: absolute difference in returns over lookback
        divergence = (btc_ret - eth_ret).abs()

        # Crypto strength: both positive momentum
        both_positive = (btc_ret > 0) & (eth_ret > 0) & eth_available

        # Crypto stress: significant divergence (one up, one down)
        crypto_stress = (
            eth_available
            & pd.notna(divergence)
            & (divergence >= c.divergence_threshold)
            & ((btc_ret > 0) != (eth_ret > 0))  # opposite signs
        )

        # High-vol proxy (VIX substitute)
        if c.equity_ref in prices.columns:
            spy_rv = realized_vol(prices[c.equity_ref], c.vix_rv_window)
            spy_rv = spy_rv.fillna(0.0)
            rv_pctl = spy_rv.rolling(
                c.high_vol_percentile_window, min_periods=60,
            ).rank(pct=True).fillna(0.5)
            high_vol = rv_pctl >= c.high_vol_percentile
        else:
            high_vol = pd.Series(False, index=prices.index)

        # Trend filter on equity
        if c.equity_ref in prices.columns:
            sma = prices[c.equity_ref].rolling(
                c.trend_window, min_periods=20,
            ).mean()
            uptrend = prices[c.equity_ref] >= sma
        else:
            uptrend = pd.Series(True, index=prices.index)

        # --- State machine ---

        # State 1: Crypto stress + high vol => safe havens
        stress_signal = crypto_stress & high_vol
        for ticker in c.safe_haven_tickers:
            if ticker in prices.columns:
                weights.loc[stress_signal, ticker] = c.safe_haven_weight

        # State 2: Crypto strength + uptrend => risk-on
        strength_signal = both_positive & uptrend & ~stress_signal
        for ticker in c.equity_tickers:
            if ticker in prices.columns:
                weights.loc[strength_signal, ticker] = c.risk_on_weight

        # State 3: Crypto strength + downtrend => moderate risk-on
        strength_down = both_positive & ~uptrend & ~stress_signal
        for ticker in c.equity_tickers:
            if ticker in prices.columns:
                weights.loc[strength_down, ticker] = c.risk_on_weight * 0.5

        # State 4: Neutral (no clear signal)
        neutral = ~stress_signal & ~strength_signal & ~strength_down
        if c.equity_ref in prices.columns:
            weights.loc[neutral & uptrend, c.equity_ref] = c.neutral_equity_weight
            weights.loc[neutral & ~uptrend, c.equity_ref] = (
                c.neutral_equity_weight * 0.6
            )

        # Before ETH data is available, fall back to neutral
        no_eth_yet = ~eth_available
        if c.equity_ref in prices.columns:
            weights.loc[no_eth_yet] = 0.0
            weights.loc[no_eth_yet & uptrend, c.equity_ref] = (
                c.neutral_equity_weight
            )
            weights.loc[no_eth_yet & ~uptrend, c.equity_ref] = (
                c.neutral_equity_weight * 0.6
            )

        return weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# G7 — Reddit Sentiment Alpha
# =========================================================================

@dataclass
class RedditSentimentConfig:
    """Config for Reddit-based sentiment alpha strategy.

    Thesis
    ------
    Reddit retail sentiment (r/wallstreetbets, r/stocks, r/options) is
    a genuinely orthogonal data source to VIX-derived sentiment.  The
    alpha comes from three distinct effects:

    1. **Mention velocity spike**: When a ticker's mention count surges
       (z-score > threshold vs 30d mean), it signals crowded attention.
       Combined with direction this becomes momentum or contrarian.

    2. **Sentiment extremes are contrarian**: When WSB is extremely
       bullish on a name (bullish_pct > 0.85, sent_zscore > 2.0), the
       crowd is typically wrong at the turning point.  Fade it.

    3. **Divergence signal**: When Reddit sentiment diverges from price
       action (price falling but sentiment rising, or vice versa), the
       smart money is already positioned and retail is late.

    Tail risk: Long-biased with explicit drawdown protection via VIX
    regime filter and position caps.
    """

    # Primary equity to trade (sentiment is aggregated across tickers
    # but we express the view through liquid ETFs)
    equity_ticker: str = "SPY"
    tech_ticker: str = "QQQ"
    hedge_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    # Which tickers to read Reddit sentiment for
    sentiment_tickers: tuple[str, ...] = ("SPY", "QQQ", "IWM")

    # Mention velocity thresholds
    mention_vel_high: float = 2.0     # z-score: buzz spike
    mention_vel_extreme: float = 3.0  # z-score: viral / meme event

    # Sentiment z-score thresholds (calibrated to ewm-smoothed range)
    sent_contrarian_bull: float = 1.3   # extreme bullish -> fade
    sent_contrarian_bear: float = -1.3  # extreme bearish -> buy
    sent_mild_bull: float = 0.3         # mild bullish -> confirm trend

    # Bullish percentage thresholds
    crowd_euphoria: float = 0.80        # >80% bullish -> contrarian short
    crowd_panic: float = 0.25           # <25% bullish -> contrarian long

    # Position sizing
    contrarian_long_weight: float = 1.8   # buy extreme fear
    contrarian_short_weight: float = -0.6  # fade extreme greed (limited)
    trend_confirm_weight: float = 1.2     # sentiment confirms trend
    neutral_weight: float = 0.5           # no clear signal
    hedge_weight: float = 0.3            # TLT hedge in contrarian short

    # Trend filter
    trend_window: int = 50
    vol_window: int = 20

    # Regime filter: reduce all positions in crisis
    crisis_scale: float = 0.3


class RedditSentimentAlpha(Strategy):
    """G7 — Reddit sentiment contrarian alpha.

    ORTHOGONAL to existing G1-G5 strategies because:
    - G1 (SentimentCrisisAlpha): Uses VIX fear spikes + news acceleration
    - G2 (FearGreedContrarian): Uses composite Fear & Greed index
    - G3 (SentimentDivergence): Uses VIX z-score mean reversion
    - G4 (SentimentEnhancedRegime): Uses VIX + regime overlay
    - G5 (CryptoSentimentDivergence): Uses BTC/ETH divergence

    G7 uses Reddit-specific signals:
    - Mention velocity (retail attention, not vol)
    - Crowd sentiment extremes (contrarian vs. trend-following retail)
    - Bullish/bearish consensus percentage (herding detection)

    None of these overlap with VIX-derived signals.  Correlation with
    existing strategies expected to be < 0.3 because the data source
    is fundamentally different (social media vs. options market).

    Carry/bleed in calm markets: +0.5x equity (small long bias).
    Worst case: Contrarian short during sustained meme mania.
    Mitigation: -0.6x max short + TLT hedge + VIX regime filter.
    """

    name = "G7-RedditSentimentAlpha"

    def __init__(self, config: RedditSentimentConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or RedditSentimentConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
        reddit_features: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Generate portfolio weights from Reddit sentiment features.

        Parameters
        ----------
        prices:
            Adjusted close prices (Date x Ticker).
        regime:
            Optional regime Series from detect_regime().
        reddit_features:
            DataFrame from ``build_synthetic_reddit_sentiment`` or
            ``extract_reddit_features``.  If None, builds synthetic.
        """
        from financial_algo.regimes import Regime

        c = self.cfg

        # Build synthetic reddit features if not provided
        if reddit_features is None:
            from financial_algo.fundamental.data.reddit_feeds import (
                build_synthetic_reddit_sentiment,
            )
            reddit_features = build_synthetic_reddit_sentiment(prices)

        # Align index
        reddit_features = reddit_features.reindex(prices.index).fillna(0.0)

        # --- Aggregate signals across tracked tickers ---
        mention_vels = []
        sent_zscores = []
        bull_pcts = []

        for ticker in c.sentiment_tickers:
            vel_col = f"{ticker}_mention_vel"
            sent_col = f"{ticker}_sent_zscore"
            bull_col = f"{ticker}_bullish_pct"

            if vel_col in reddit_features.columns:
                mention_vels.append(reddit_features[vel_col])
            if sent_col in reddit_features.columns:
                sent_zscores.append(reddit_features[sent_col])
            if bull_col in reddit_features.columns:
                bull_pcts.append(reddit_features[bull_col])

        # Use market-level aggregates if per-ticker not available
        if not mention_vels and "market_mention_vel" in reddit_features.columns:
            mention_vels = [reddit_features["market_mention_vel"]]
        if not sent_zscores and "market_sent_zscore" in reddit_features.columns:
            sent_zscores = [reddit_features["market_sent_zscore"]]

        # Mean across tickers
        if mention_vels:
            agg_vel = pd.concat(mention_vels, axis=1).mean(axis=1)
        else:
            agg_vel = pd.Series(0.0, index=prices.index)

        if sent_zscores:
            agg_sent = pd.concat(sent_zscores, axis=1).mean(axis=1)
        else:
            agg_sent = pd.Series(0.0, index=prices.index)

        if bull_pcts:
            agg_bull = pd.concat(bull_pcts, axis=1).mean(axis=1)
        else:
            agg_bull = pd.Series(0.5, index=prices.index)

        # --- Trend filter ---
        equity = prices.get(c.equity_ticker, prices.iloc[:, 0])
        sma = equity.rolling(c.trend_window, min_periods=20).mean()
        uptrend = equity >= sma
        downtrend = ~uptrend

        # Vol scaling
        rv = realized_vol(equity, c.vol_window).fillna(0.15)
        vol_scale = (0.15 / rv.clip(lower=0.05)).clip(0.5, 1.5)

        # --- Regime filter ---
        if regime is not None:
            crisis_regimes = {
                Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS,
            }
            is_crisis = regime.isin(crisis_regimes)
        else:
            is_crisis = pd.Series(False, index=prices.index)

        # --- Signal construction (vectorized) ---

        # Signal 1: Crowd euphoria -> contrarian short
        # Either extreme bullish consensus OR extreme sentiment z-score
        # (requiring both is too strict -- either alone indicates herding)
        euphoria = (
            (pd.notna(agg_bull) & (agg_bull >= c.crowd_euphoria))
            | (pd.notna(agg_sent) & (agg_sent >= c.sent_contrarian_bull))
        )

        # Signal 2: Crowd panic -> contrarian long
        # Either extreme bearish consensus OR extreme negative sentiment
        panic = (
            (pd.notna(agg_bull) & (agg_bull <= c.crowd_panic))
            | (pd.notna(agg_sent) & (agg_sent <= c.sent_contrarian_bear))
        )

        # Signal 3: Mild bullish + uptrend + high mentions -> trend confirm
        trend_confirm = (
            uptrend
            & pd.notna(agg_sent) & (agg_sent >= c.sent_mild_bull)
            & pd.notna(agg_vel) & (agg_vel >= c.mention_vel_high)
            & ~euphoria  # not at extreme
        )

        # Signal 4: Mention velocity extreme (viral event) -> reduce
        # exposure regardless of direction (event risk)
        viral = pd.notna(agg_vel) & (agg_vel >= c.mention_vel_extreme)

        # --- Build weights ---
        tickers = [c.equity_ticker, c.tech_ticker, c.hedge_ticker, c.gold_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        # Neutral baseline
        w[c.equity_ticker] = np.where(uptrend, c.neutral_weight, c.neutral_weight * 0.6)

        # Contrarian long on crowd panic
        w.loc[panic & uptrend, c.equity_ticker] = c.contrarian_long_weight
        w.loc[panic & uptrend, c.tech_ticker] = c.contrarian_long_weight * 0.5
        w.loc[panic & downtrend, c.equity_ticker] = c.contrarian_long_weight * 0.7
        w.loc[panic & downtrend, c.hedge_ticker] = c.hedge_weight * 0.5

        # Trend confirmation: boost equity
        w.loc[trend_confirm, c.equity_ticker] = c.trend_confirm_weight
        w.loc[trend_confirm, c.tech_ticker] = c.trend_confirm_weight * 0.3

        # Contrarian short on crowd euphoria
        w.loc[euphoria, c.equity_ticker] = c.contrarian_short_weight
        w.loc[euphoria, c.hedge_ticker] = c.hedge_weight
        w.loc[euphoria, c.gold_ticker] = c.hedge_weight * 0.5

        # Viral event: flatten to minimal exposure
        w.loc[viral, c.equity_ticker] = 0.1
        w.loc[viral, c.tech_ticker] = 0.0
        w.loc[viral, c.hedge_ticker] = c.hedge_weight
        w.loc[viral, c.gold_ticker] = c.hedge_weight * 0.3

        # Vol scaling on equity + tech
        w[c.equity_ticker] = w[c.equity_ticker] * vol_scale
        w[c.tech_ticker] = w[c.tech_ticker] * vol_scale

        # Crisis regime: scale everything down
        for col in tickers:
            w.loc[is_crisis, col] = w.loc[is_crisis, col] * c.crisis_scale

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w
