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
    """Config for sentiment-driven crisis strategy."""

    equity_ticker: str = "SPY"
    hedge_ticker: str = "TLT"
    gold_ticker: str = "GLD"

    # Signal thresholds
    fear_z_threshold: float = 1.5
    velocity_threshold: float = 1.5
    recovery_momentum: float = 0.5

    # Position sizing
    leverage_crisis: float = 2.0     # short equity + long hedges in crisis
    leverage_recovery: float = 2.5   # leveraged long on recovery
    leverage_neutral: float = 0.5    # small long bias when neutral

    # Normal-mode enhancement
    trend_window: int = 50           # shorter trend for normal mode
    momentum_window: int = 63        # momentum lookback
    vol_window: int = 20             # realized vol for scaling
    normal_uptrend_boost: float = 1.4  # equity weight in uptrend + momentum
    normal_base: float = 0.9          # neutral equity weight
    normal_downtrend: float = 0.3     # reduced when downtrend
    normal_hedge_down: float = 0.2    # TLT hedge in downtrend


class SentimentCrisisAlpha(Strategy):
    """Trade crisis onset and recovery using sentiment signals + trend filter.

    Unlike technical strategies that react to price drops AFTER they happen,
    this strategy detects fear spikes and news acceleration DURING crisis
    development, allowing earlier hedging.

    Enhanced normal mode: trend + momentum overlay generates returns between
    crises. Vol-scaled position sizing across all modes.

    States:
    - CRISIS: fear spiking + news accelerating -> short equity, long hedges
    - RECOVERY: fear subsiding + positive sentiment momentum -> leveraged long
    - NEUTRAL: trend/momentum-driven long with vol scaling
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

        # Build sentiment if not provided
        if sentiment_df is None:
            vix = None  # will use vol proxy
            sentiment_df = build_synthetic_sentiment(prices, vix)

        # Generate signals
        crisis_sig = crisis_onset_signal(
            sentiment_df,
            fear_z_threshold=c.fear_z_threshold,
            velocity_threshold=c.velocity_threshold,
        )
        recov_sig = recovery_signal(
            sentiment_df,
            momentum_threshold=c.recovery_momentum,
        )

        # Trend filters
        spy_price = prices.get(c.equity_ticker, prices.iloc[:, 0])
        sma_200 = spy_price.rolling(200, min_periods=100).mean()
        sma_trend = spy_price.rolling(c.trend_window, min_periods=20).mean()
        uptrend_200 = spy_price >= sma_200
        uptrend_short = spy_price >= sma_trend
        downtrend = ~uptrend_200

        # Momentum
        mom_ret = spy_price.pct_change(c.momentum_window).fillna(0.0)
        mom_positive = mom_ret > 0

        # Vol scaling: reduce positions when vol is elevated
        rv = realized_vol(spy_price, c.vol_window).fillna(0.15)
        vol_scale = (0.15 / rv.clip(lower=0.05)).clip(0.5, 1.5)

        tickers = [c.equity_ticker, c.hedge_ticker, c.gold_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        # --- Normal mode: trend + momentum driven ---
        # Uptrend + momentum: boosted long equity
        normal_up_mom = uptrend_short & mom_positive
        normal_up = uptrend_short & ~mom_positive
        normal_down = ~uptrend_short

        w[c.equity_ticker] = np.where(
            normal_up_mom, c.normal_uptrend_boost,
            np.where(normal_up, c.normal_base,
                     np.where(normal_down, c.normal_downtrend, c.normal_base))
        )
        # Small hedge allocation in downtrend
        w.loc[normal_down, c.hedge_ticker] = c.normal_hedge_down

        # --- Crisis mode: override ---
        crisis_mask = crisis_sig == 1
        crisis_down = crisis_mask & downtrend
        crisis_up = crisis_mask & uptrend_200
        w.loc[crisis_down, c.equity_ticker] = -c.leverage_crisis * 0.5
        w.loc[crisis_down, c.hedge_ticker] = c.leverage_crisis * 0.3
        w.loc[crisis_down, c.gold_ticker] = c.leverage_crisis * 0.2
        w.loc[crisis_up, c.equity_ticker] = 0.0
        w.loc[crisis_up, c.hedge_ticker] = c.leverage_crisis * 0.2
        w.loc[crisis_up, c.gold_ticker] = c.leverage_crisis * 0.1

        # --- Recovery mode: override ---
        recovery_mask = recov_sig == 1
        recov_up = recovery_mask & uptrend_200
        recov_down = recovery_mask & downtrend
        w.loc[recov_up, c.equity_ticker] = c.leverage_recovery
        w.loc[recov_up, c.hedge_ticker] = 0.0
        w.loc[recov_up, c.gold_ticker] = 0.0
        w.loc[recov_down, c.equity_ticker] = c.leverage_recovery * 0.5
        w.loc[recov_down, c.hedge_ticker] = 0.0
        w.loc[recov_down, c.gold_ticker] = 0.0

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
