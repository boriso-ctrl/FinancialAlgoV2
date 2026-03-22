"""Category Y: Behavioral Anomaly strategies.

Exploiting well-documented cognitive biases and market microstructure
effects that create persistent mispricings. These are harder to
arbitrage away because they stem from human psychology, not information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_algo.strategies.base import Strategy


# =========================================================================
# Y1 -- Disposition Effect Reversal
# =========================================================================

class DispositionEffectReversal(Strategy):
    """Exploit the disposition effect: investors sell winners too early.

    Thesis: Shefrin & Statman (1985) documented that investors have a
    strong tendency to sell winning positions ("lock in gains") and hold
    losers ("it'll come back"). This creates a headwind for recent winners
    that dissipates over 6-12 months, after which the fundamental trend
    reasserts itself.

    Signal: Assets that have risen over 1 month AND have positive 12-month
    momentum are being sold by retail. When selling pressure exhausts
    (detectable via declining volume or simply time), the trend resumes.

    We buy assets with 12m positive returns that had a recent 1m dip
    (disposition-driven selling) and are rebounding.
    """

    name = "Y1-DispositionReversal"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # KILLED: Disposition effect signal produced Sharpe -0.15. Too many
        # conditions (12m mom + dip + bounce + trend) caused near-zero signal
        # frequency with remaining triggers being noise. Archived.
        return weights


# =========================================================================
# Y2 -- Attention Overreaction
# =========================================================================

class AttentionOverreaction(Strategy):
    """Fade overreactions to extreme single-day moves (attention bias).

    Thesis: Barber & Odean (2008) "All That Glitters" shows that
    attention-grabbing events (big price moves, news) cause retail
    overbuying. This creates a short-term overreaction that reverts
    over the next 5-20 days.

    Signal: When a sector ETF has an extreme 1-day return (>2 sigma),
    it's likely driven by attention/herding. Fade the move if the
    broader trend is intact.

    This is NOT contrarian on the market -- it's contrarian on
    INDIVIDUAL SECTORS that spike while the market is normal.
    """

    name = "Y2-AttentionOverreaction"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        sectors = [t for t in ["XLK", "XLF", "XLI", "XLB", "XLP", "XLU",
                               "XLY", "XLV", "XLE"]
                   if t in prices.columns]
        if len(sectors) < 3 or "SPY" not in prices.columns:
            return weights

        # Daily returns
        daily_ret = prices[sectors].pct_change().fillna(0.0)

        # Rolling 60-day stats for each sector
        roll_mean = daily_ret.rolling(60).mean()
        roll_std = daily_ret.rolling(60).std().replace(0, np.nan)

        # Z-score of daily return
        z = ((daily_ret - roll_mean) / roll_std).fillna(0.0)

        # Extreme positive spike (z > 2.0): fade it (go short/underweight)
        # Extreme negative spike (z < -2.0): fade it (go long)
        # But only if SPY is in a trend (not during crisis)
        spy_sma = prices["SPY"].rolling(200).mean()
        trend_ok = prices["SPY"] > spy_sma

        # Fade negative spikes (buy the dip) on sector that crashed
        buy_fade = (z < -2.0)
        # Hold for 10 days via forward-fill (mark signal, then propagate)
        buy_fade = buy_fade.rolling(10).max().fillna(0.0).astype(bool)

        for t in sectors:
            # Only buy-fade if market trend is up (don't catch falling knives)
            signal = buy_fade[t] & trend_ok
            weights.loc[signal, t] = 0.25

        # Ensure we don't overallocate
        total = weights.abs().sum(axis=1)
        scale = (1.5 / total).clip(upper=1.0).replace([np.inf, -np.inf], 1.0).fillna(1.0)
        weights = weights.multiply(scale, axis=0)

        return weights


# =========================================================================
# Y3 -- Lunar Cycle Alpha (yes, really)
# =========================================================================

class LunarCycleAlpha(Strategy):
    """Trade the lunar cycle effect on stock returns.

    Thesis: Shockingly, there IS academic evidence for this. Dichev &
    Janes (2003, Journal of Private Equity) and Yuan, Zheng & Zhu (2006,
    Journal of Empirical Finance) found that stock returns around
    new moons are ~5-6% annualized higher than around full moons.

    Hypothesized mechanism: full moons correlate with slightly elevated
    anxiety/negative affect (Rotton & Kelly, 1985 meta-analysis), which
    may increase risk aversion at the margin. The effect is small but
    surprisingly robust across countries and time periods.

    Signal: Long equities around new moon (+/- 7 days), reduce exposure
    around full moon. We compute the lunar phase from the date.

    This is the ULTIMATE "thinking outside the box" strategy.
    Even if it doesn't work, testing it rigorously is valuable.
    """

    name = "Y3-LunarCycleAlpha"

    @staticmethod
    def _moon_phase(date: pd.Timestamp) -> float:
        """Compute approximate lunar phase (0 = new moon, 0.5 = full moon).

        Uses the synodic month approximation. Returns value in [0, 1).
        """
        # Known new moon reference: January 6, 2000 18:14 UTC
        ref = pd.Timestamp("2000-01-06 18:14:00")
        synodic_month = 29.53059  # days
        days_since = (date - ref).total_seconds() / 86400.0
        phase = (days_since % synodic_month) / synodic_month
        return phase

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "SPY" not in prices.columns:
            return weights

        # Compute lunar phase for each trading day
        phases = pd.Series(
            [self._moon_phase(d) for d in prices.index],
            index=prices.index,
        )

        # New moon: phase near 0 or 1 (within 0.25 = ~7 days)
        near_new = (phases < 0.25) | (phases > 0.75)
        # Full moon: phase near 0.5 (within 0.25 = ~7 days)
        near_full = (phases > 0.25) & (phases < 0.75)

        # 200-day trend filter
        spy_sma = prices["SPY"].rolling(200).mean()
        trend_up = prices["SPY"] > spy_sma

        # New moon + trend up: alternative equity basket (risk-on phase)
        # (small SPY + diversified alternatives to decorrelate from production)
        weights.loc[near_new & trend_up, "SPY"] = 0.40
        if "IWM" in prices.columns:
            weights.loc[near_new & trend_up, "IWM"] = 0.20
        if "GLD" in prices.columns:
            weights.loc[near_new & trend_up, "GLD"] = 0.15
        if "EEM" in prices.columns:
            weights.loc[near_new & trend_up, "EEM"] = 0.10
        if "HYG" in prices.columns:
            weights.loc[near_new & trend_up, "HYG"] = 0.10
        if "EFA" in prices.columns:
            weights.loc[near_new & trend_up, "EFA"] = 0.05

        # Full moon + trend up: defensive / alternative positioning
        if "GLD" in prices.columns:
            weights.loc[near_full & trend_up, "GLD"] = 0.30
        if "UUP" in prices.columns:
            weights.loc[near_full & trend_up, "UUP"] = 0.20
        if "XLP" in prices.columns:
            weights.loc[near_full & trend_up, "XLP"] = 0.15
        if "EFA" in prices.columns:
            weights.loc[near_full & trend_up, "EFA"] = 0.10

        # Trend down: always defensive regardless of moon
        if "GLD" in prices.columns:
            weights.loc[~trend_up, "GLD"] = 0.30
        if "UUP" in prices.columns:
            weights.loc[~trend_up, "UUP"] = 0.20
        if "TLT" in prices.columns:
            weights.loc[~trend_up, "TLT"] = 0.15
        if "EFA" in prices.columns:
            weights.loc[~trend_up, "EFA"] = 0.10

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights
