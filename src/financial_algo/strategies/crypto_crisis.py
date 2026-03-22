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
