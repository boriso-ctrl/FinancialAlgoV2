"""Category F: Crypto Crisis Plays.

Dual-mode: trend-follow safe-haven and crypto assets in normal markets,
with crisis-aware amplification and regime tilting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import ema, realized_vol
from financial_algo.regimes import Regime
from financial_algo.strategies.base import Strategy


# =========================================================================
# F1 - Crypto Flight-to-Quality (always-on haven trend + crypto canary)
# =========================================================================

@dataclass
class CryptoFlightConfig:
    """Config for crypto-triggered flight-to-quality strategy."""

    crypto_ticker: str = "BTC-USD"
    gold_ticker: str = "GLD"
    bond_ticker: str = "IEF"

    fast_ema: int = 20
    slow_ema: int = 50
    drawdown_window: int = 60
    drawdown_threshold: float = -0.15

    gold_leverage: float = 0.7
    bond_leverage: float = 0.3
    crisis_boost: float = 0.5

    vol_window: int = 20
    target_vol: float = 0.12


class CryptoFlightToQuality(Strategy):
    """Dual-mode safe-haven strategy with crypto canary.

    Normal: Trend-follow GLD/IEF (always-on safe-haven momentum).
    Stress: Amplify when BTC drawdown signals risk-off environment.
    """

    name = "CryptoFlightToQuality"

    def __init__(self, config: CryptoFlightConfig | None = None) -> None:
        self.cfg = config or CryptoFlightConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("CryptoFlightToQuality requires a regime Series")

        c = self.cfg

        # BTC stress canary
        btc = prices[c.crypto_ticker]
        btc_peak = btc.rolling(c.drawdown_window, min_periods=1).max()
        btc_dd = ((btc - btc_peak) / btc_peak.replace(0, np.nan)).fillna(0.0)
        btc_stressed = btc_dd < c.drawdown_threshold

        w = pd.DataFrame(0.0, index=prices.index, columns=[c.gold_ticker, c.bond_ticker])

        for t, base_lev in [(c.gold_ticker, c.gold_leverage),
                            (c.bond_ticker, c.bond_leverage)]:
            if t not in prices.columns:
                continue
            p = prices[t]
            fast = ema(p, c.fast_ema)
            slow = ema(p, c.slow_ema)
            trending = fast > slow

            tvol = realized_vol(p, c.vol_window).clip(lower=0.05)
            vs = (c.target_vol / tvol).clip(0.3, 2.0)
            vs = vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # Base: trend-follow (always-on)
            base = trending.astype(float) * base_lev * vs
            # Stress boost: amplify when BTC drawdown signals risk-off
            stress_w = (btc_stressed & trending).astype(float) * c.crisis_boost * vs
            w[t] = base + stress_w

        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# F2 - Crypto Recovery Surge (crypto trend + recovery amplification)
# =========================================================================

@dataclass
class CryptoRecoveryConfig:
    """Config for crypto trend-following with recovery boost."""

    crypto_ticker: str = "BTC-USD"

    fast_ema: int = 20
    slow_ema: int = 50

    leverage: float = 0.5
    recovery_boost: float = 0.5

    vol_window: int = 20
    target_vol: float = 0.20


class CryptoRecoverySurge(Strategy):
    """Crypto trend-following with recovery amplification.

    Normal: Long BTC when trending up, vol-scaled (small positions).
    Recovery: Amplified long for massive crypto recovery rallies.
    """

    name = "CryptoRecoverySurge"

    def __init__(self, config: CryptoRecoveryConfig | None = None) -> None:
        self.cfg = config or CryptoRecoveryConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("CryptoRecoverySurge requires a regime Series")

        c = self.cfg
        btc = prices[c.crypto_ticker]

        # Trend: EMA crossover
        fast = ema(btc, c.fast_ema)
        slow = ema(btc, c.slow_ema)
        trend_up = fast > slow

        # Vol scaling (critical for crypto extreme vol)
        btc_vol = realized_vol(btc, c.vol_window).clip(lower=0.10)
        vol_scale = (c.target_vol / btc_vol).clip(0.1, 1.5)
        vol_scale = vol_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Recovery boost
        recovery = regime.isin({Regime.RECOVERY})
        crisis = regime.isin({
            Regime.GENERAL_CRISIS, Regime.WAR_CRISIS, Regime.OIL_CRISIS,
        })

        base = trend_up.astype(float) * c.leverage * vol_scale
        boost = (recovery & trend_up).astype(float) * c.recovery_boost * vol_scale
        weight = base + boost

        # Crisis damper
        weight[crisis] = weight[crisis] * 0.3

        w = pd.DataFrame(0.0, index=prices.index, columns=[c.crypto_ticker])
        w[c.crypto_ticker] = weight
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# F3 - Crypto-Gold Divergence (dual-asset momentum: gold + crypto)
# =========================================================================

@dataclass
class CryptoGoldDivConfig:
    """Config for gold + crypto dual momentum strategy."""

    crypto_ticker: str = "BTC-USD"
    gold_ticker: str = "GLD"

    fast_ema: int = 20
    slow_ema: int = 50

    gold_leverage: float = 0.7
    crypto_leverage: float = 0.3
    regime_tilt: float = 0.3

    vol_window: int = 20
    target_vol: float = 0.15


class CryptoGoldDivergence(Strategy):
    """Dual-asset momentum: gold + crypto trend-following.

    Normal: Long each asset when trending, vol-scaled.
    Tilts toward gold in crisis, crypto in recovery. No shorting.
    """

    name = "CryptoGoldDivergence"

    def __init__(self, config: CryptoGoldDivConfig | None = None) -> None:
        self.cfg = config or CryptoGoldDivConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("CryptoGoldDivergence requires a regime Series")

        c = self.cfg

        # Gold trend
        gld = prices[c.gold_ticker]
        gld_fast = ema(gld, c.fast_ema)
        gld_slow = ema(gld, c.slow_ema)
        gld_trend = gld_fast > gld_slow

        gld_vol = realized_vol(gld, c.vol_window).clip(lower=0.05)
        gld_vs = (c.target_vol / gld_vol).clip(0.3, 2.0)
        gld_vs = gld_vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # BTC trend
        btc = prices[c.crypto_ticker]
        btc_fast = ema(btc, c.fast_ema)
        btc_slow = ema(btc, c.slow_ema)
        btc_trend = btc_fast > btc_slow

        btc_vol = realized_vol(btc, c.vol_window).clip(lower=0.10)
        btc_vs = (c.target_vol / btc_vol).clip(0.1, 1.5)
        btc_vs = btc_vs.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Base weights
        gld_w = gld_trend.astype(float) * c.gold_leverage * gld_vs
        btc_w = btc_trend.astype(float) * c.crypto_leverage * btc_vs

        # Regime tilt
        crisis = regime.isin({
            Regime.GENERAL_CRISIS, Regime.WAR_CRISIS, Regime.OIL_CRISIS,
        })
        recovery = regime.isin({Regime.RECOVERY})

        # Crisis: boost gold, reduce crypto
        gld_w = gld_w + (crisis & gld_trend).astype(float) * c.regime_tilt * gld_vs
        btc_w[crisis] = btc_w[crisis] * 0.2

        # Recovery: boost crypto
        btc_w = btc_w + (recovery & btc_trend).astype(float) * c.regime_tilt * btc_vs

        w = pd.DataFrame(0.0, index=prices.index, columns=[c.gold_ticker, c.crypto_ticker])
        w[c.gold_ticker] = gld_w
        w[c.crypto_ticker] = btc_w
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w


# =========================================================================
# F4 - Crypto Contagion Hedge
# =========================================================================

@dataclass
class CryptoContagionHedgeConfig:
    """Hedge against crypto contagion using BTC + ETH drawdown signals.

    When both BTC-USD and ETH-USD are in >20% drawdown from 60-day high,
    this signals broad crypto contagion / risk-off. Hedge with safe havens.
    When both are above their 60-day high, go risk-on.
    Mixed signals = stay flat.
    """

    btc_ticker: str = "BTC-USD"
    eth_ticker: str = "ETH-USD"

    safe_tickers: tuple = ("GLD", "SHY", "UUP")
    risk_on_tickers: tuple = ("SPY", "QQQ")

    drawdown_window: int = 60
    contagion_threshold: float = -0.20  # both must be < -20% from peak

    # Safe-haven weights during contagion
    safe_gld: float = 0.40
    safe_shy: float = 0.35
    safe_uup: float = 0.25

    # Risk-on weights when crypto strong
    risk_on_weight: float = 0.50


class CryptoContagionHedge(Strategy):
    """Crypto contagion detector: hedge when BTC+ETH both in deep drawdown.

    Thesis: When both BTC and ETH are simultaneously in >20% drawdown
    from their 60-day highs, this signals broad risk-off contagion
    spreading beyond crypto. Go long GLD + SHY + UUP (flight to quality).
    When both are above their 60-day max (strong), go risk-on SPY + QQQ.
    Mixed signals = flat (no edge).

    ETH data starts 2017-11, so periods before that treat ETH signal
    as neutral (no contagion signal without both assets).
    """

    name = "F4-CryptoContagionHedge"

    def __init__(self, config: CryptoContagionHedgeConfig | None = None) -> None:
        self.cfg = config or CryptoContagionHedgeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        all_tickers = list(c.safe_tickers) + list(c.risk_on_tickers)
        avail = [t for t in all_tickers if t in prices.columns]
        if not avail:
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        # --- Compute drawdowns for BTC and ETH ---
        def _drawdown_pct(ticker: str) -> pd.Series:
            if ticker not in prices.columns:
                return pd.Series(np.nan, index=prices.index)
            p = prices[ticker]
            peak = p.rolling(c.drawdown_window, min_periods=1).max()
            dd = (p - peak) / peak.replace(0, np.nan)
            return dd.fillna(0.0)

        btc_dd = _drawdown_pct(c.btc_ticker)
        eth_dd = _drawdown_pct(c.eth_ticker)

        # Both must have valid data for contagion signal
        btc_valid = pd.notna(prices.get(c.btc_ticker, pd.Series(dtype=float)))
        eth_valid = pd.notna(prices.get(c.eth_ticker, pd.Series(dtype=float)))
        both_valid = btc_valid & eth_valid

        # Contagion: both in deep drawdown
        btc_stressed = btc_dd < c.contagion_threshold
        eth_stressed = eth_dd < c.contagion_threshold
        contagion = both_valid & btc_stressed & eth_stressed

        # Crypto strong: both at or near highs (dd > -2%)
        btc_strong = btc_dd > -0.02
        eth_strong = eth_dd > -0.02
        crypto_strong = both_valid & btc_strong & eth_strong

        # --- Assign weights ---
        # Contagion: flight to quality
        safe_map = {"GLD": c.safe_gld, "SHY": c.safe_shy, "UUP": c.safe_uup}
        safe_avail = [t for t in c.safe_tickers if t in prices.columns]
        for t in safe_avail:
            w_val = safe_map.get(t, 0.0)
            if w_val > 0:
                weights.loc[contagion, t] = w_val

        # Crypto strong: risk-on
        risk_avail = [t for t in c.risk_on_tickers if t in prices.columns]
        n_risk = len(risk_avail)
        if n_risk > 0:
            per_risk = c.risk_on_weight / n_risk
            for t in risk_avail:
                weights.loc[crypto_strong, t] = per_risk

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# F2b - Crypto Recovery Surge with ATR-based risk management
# =========================================================================


def _close_atr_f2b(close: pd.Series, period: int = 14) -> pd.Series:
    """Close-only ATR proxy: rolling mean of absolute daily changes."""
    abs_change = (close - close.shift(1)).abs()
    return abs_change.rolling(period, min_periods=1).mean()


@dataclass
class CryptoRecoverySurgeATRConfig:
    """Crypto trend-following with ATR-based position sizing and stops.

    Improvement over F2: replaces fixed vol-scaling with ATR-based
    position sizing (size = base_size * target_risk / ATR) and adds
    ATR-based exit (flatten when loss > 2.5 * ATR from entry).
    """

    crypto_ticker: str = "BTC-USD"

    fast_ema: int = 20
    slow_ema: int = 50

    base_leverage: float = 0.35
    recovery_boost: float = 0.35

    # ATR parameters
    atr_period: int = 14
    target_risk: float = 0.02   # target daily risk per position
    max_weight: float = 0.50    # max position size (cap for DD control)
    atr_stop_mult: float = 2.5  # flatten when loss > 2.5 * ATR


class CryptoRecoverySurgeATR(Strategy):
    """Crypto trend-following with ATR-based risk management.

    Improvement over F2-CryptoRecoverySurge:
      - ATR-based position sizing: size = base * (target_risk / ATR)
        adapts position size to current crypto volatility
      - ATR-based exit: flatten when loss > 2.5 * ATR from entry
      - Recovery regime boost amplifies positions in crypto bounce

    The ATR sizing means smaller positions during extreme vol (COVID,
    FTX collapse) and larger positions in calmer trending markets.
    """

    name = "F2b-CryptoRecovSurgeATR"

    def __init__(self, config: CryptoRecoverySurgeATRConfig | None = None) -> None:
        self.cfg = config or CryptoRecoverySurgeATRConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        if regime is None:
            raise ValueError("CryptoRecoverySurgeATR requires a regime Series")

        c = self.cfg
        btc = prices[c.crypto_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=[c.crypto_ticker])

        # --- Trend: EMA crossover ---
        fast = ema(btc, c.fast_ema)
        slow = ema(btc, c.slow_ema)
        trend_up = fast > slow

        # --- ATR-based position sizing ---
        atr_val = _close_atr_f2b(btc, c.atr_period).fillna(0.0)
        safe_atr = atr_val.clip(lower=1e-8)
        # Normalise ATR as fraction of price
        atr_pct = (safe_atr / btc.clip(lower=1e-8)).fillna(0.0)
        atr_pct = atr_pct.replace([np.inf, -np.inf], np.nan).fillna(0.01)

        # Position size: scale inversely to ATR
        atr_scale = (c.target_risk / atr_pct).clip(0.1, 2.0)
        atr_scale = atr_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

        # Recovery boost
        recovery = regime.isin({Regime.RECOVERY})
        crisis = regime.isin({
            Regime.GENERAL_CRISIS, Regime.WAR_CRISIS, Regime.OIL_CRISIS,
        })

        # Base weight: trend + ATR sizing
        base = trend_up.astype(float) * c.base_leverage * atr_scale
        # Recovery boost
        boost = (recovery & trend_up).astype(float) * c.recovery_boost * atr_scale
        weight = (base + boost).clip(upper=c.max_weight)

        # Crisis damper
        weight = weight.where(~crisis, weight * 0.3)

        # --- ATR-based stop-loss ---
        # Track cumulative return since trend started
        # Detect trend transitions
        trend_int = trend_up.astype(int)
        trend_change = trend_int.diff().fillna(0.0)
        trend_start = trend_change == 1

        # Entry price at each trend start
        entry_price = btc.where(trend_start).ffill()
        entry_price = entry_price.where(trend_up)

        # Cumulative move since entry (long position)
        cum_move = (btc - entry_price).fillna(0.0)

        # Stop: flatten when loss > 2.5 * ATR
        is_stopped = pd.notna(cum_move) & (cum_move < -(c.atr_stop_mult * safe_atr))
        weight = weight.where(~(trend_up & is_stopped), 0.0)

        w[c.crypto_ticker] = weight
        w = w.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return w
