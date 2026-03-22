"""Category X: Cross-Asset Divergence strategies.

These exploit temporary breakdowns in long-run cross-asset relationships.
When correlations that "should" hold diverge, there's often a
mean-reversion trade to be had.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from financial_algo.strategies.base import Strategy
from financial_algo.indicators import realized_vol


# =========================================================================
# X1 -- Copper/Gold Ratio as Growth Signal
# =========================================================================

class CopperGoldGrowth(Strategy):
    """Use the Copper/Gold ratio as a real-time economic growth signal.

    Thesis: Copper is the "Dr. Copper" economic barometer (industrial demand).
    Gold is a fear/deflation hedge. The Copper/Gold ratio tracks global growth
    expectations better than GDP releases (which are lagged). When the ratio
    rises, go long risk assets; when it falls, go defensive.

    Since we don't have copper futures, we proxy via XLB (materials) / GLD.
    XLB is heavily exposed to copper miners (Freeport-McMoRan, etc.).

    Academic basis: Gundlach (2018) "Copper/Gold ratio predicts 10Y yield."
    The same signal predicts equity risk premium.
    """

    name = "X1-CopperGoldGrowth"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        if "XLB" not in prices.columns or "GLD" not in prices.columns:
            return weights

        # Copper/Gold proxy ratio
        ratio = prices["XLB"] / prices["GLD"].replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)

        # Z-score of ratio change (60-day window = 3 months)
        ratio_ma = ratio.rolling(60).mean()
        ratio_std = ratio.rolling(60).std().replace(0, np.nan)
        z = ((ratio - ratio_ma) / ratio_std).fillna(0.0)

        # Long-term trend filter: 200-day SMA of SPY
        spy_sma = prices["SPY"].rolling(200).mean() if "SPY" in prices.columns else None

        # Rising ratio (z > 0.3) = growth expanding -> long risk
        risk_on = z > 0.3
        # Falling ratio (z < -0.3) = growth contracting -> long havens
        risk_off = z < -0.3

        # Only go risk-on if SPY above its 200d SMA (confluence)
        if spy_sma is not None:
            trend_up = prices["SPY"] > spy_sma
            risk_on = risk_on & trend_up

        weights.loc[risk_on, "SPY"] = 0.6
        weights.loc[risk_on, "QQQ"] = 0.4

        # Neutral zone: moderate allocation when signal is ambiguous
        neutral = ~risk_on & ~risk_off
        if spy_sma is not None:
            neutral = neutral & trend_up
        weights.loc[neutral, "SPY"] = 0.3
        if "TLT" in prices.columns:
            weights.loc[neutral, "TLT"] = 0.2
        if "GLD" in prices.columns:
            weights.loc[neutral, "GLD"] = 0.1

        # Risk-off: strong defensive with diversified havens
        weights.loc[risk_off, "TLT"] = 0.40
        weights.loc[risk_off, "GLD"] = 0.30
        weights.loc[risk_off, "UUP"] = 0.15
        if "IEF" in prices.columns:
            weights.loc[risk_off, "IEF"] = 0.15

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# X2 -- Credit-Equity Divergence
# =========================================================================

class CreditEquityDivergence(Strategy):
    """Trade the divergence between credit spreads and equity prices.

    Thesis: Credit (HYG/LQD) and equities (SPY) should move together.
    When credit deteriorates but equities hold up, equities are likely
    overvalued (credit leads equities by ~2-4 weeks). When credit improves
    but equities lag, equities are likely undervalued.

    This is a well-documented lead-lag relationship. Credit markets are
    dominated by institutional investors who react faster to fundamental
    deterioration than retail-heavy equity markets.

    Academic basis: Collin-Dufresne et al. (2001), Blanco et al. (2005)
    """

    name = "X2-CreditEquityDivergence"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        needed = {"HYG", "LQD", "SPY"}
        if not needed.issubset(prices.columns):
            return weights

        # Credit spread proxy: HYG/LQD ratio (tightening = risk-on)
        credit_ratio = (prices["HYG"] / prices["LQD"].replace(0, np.nan))
        credit_ratio = credit_ratio.replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)

        # Normalize: 20-day z-score of credit ratio
        cr_ma = credit_ratio.rolling(20).mean()
        cr_std = credit_ratio.rolling(20).std().replace(0, np.nan)
        credit_z = ((credit_ratio - cr_ma) / cr_std).fillna(0.0)

        # Equity momentum: 20-day z-score of SPY returns
        spy_ret = prices["SPY"].pct_change(20).fillna(0.0)
        spy_ma = spy_ret.rolling(60).mean()
        spy_std = spy_ret.rolling(60).std().replace(0, np.nan)
        equity_z = ((spy_ret - spy_ma) / spy_std).fillna(0.0)

        # Divergence = credit_z - equity_z
        # Positive divergence: credit improving, equities lagging -> buy equities
        # Negative divergence: credit deteriorating, equities holding -> sell equities
        divergence = credit_z - equity_z

        # 200-day trend filter
        spy_sma200 = prices["SPY"].rolling(200).mean()
        trend_up = prices["SPY"] > spy_sma200

        # Buy signal: credit leading (lower threshold for more signal)
        buy = (divergence > 0.5) & trend_up
        # Defensive: credit warning
        defensive = divergence < -0.5

        # Credit-sensitive assets instead of broad equity
        if "HYG" in prices.columns:
            weights.loc[buy, "HYG"] = 0.35
        if "EEM" in prices.columns:
            weights.loc[buy, "EEM"] = 0.25
        if "XLF" in prices.columns:
            weights.loc[buy, "XLF"] = 0.20
        if "XLB" in prices.columns:
            weights.loc[buy, "XLB"] = 0.20

        weights.loc[defensive, "TLT"] = 0.40
        weights.loc[defensive, "GLD"] = 0.30
        if "UUP" in prices.columns:
            weights.loc[defensive, "UUP"] = 0.20

        weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return weights


# =========================================================================
# X3 -- Dollar Wrecking Ball
# =========================================================================

class DollarWreckingBall(Strategy):
    """Trade the "Dollar Wrecking Ball" effect on risk assets.

    Thesis: A rapidly strengthening dollar is the single most destructive
    force for global risk assets. It tightens financial conditions, hurts
    EM debtors, compresses commodity prices, and reduces S&P 500 foreign
    earnings. Brent Johnson's "Dollar Milkshake Theory" (2018).

    Signal: Rate of change of UUP (dollar index ETF). When the dollar
    accelerates higher, *everything else* sells off. When it rolls over,
    risk assets rally hard.

    This is NOT just M1-DollarCarry inverted — it specifically targets
    the *acceleration* of dollar strength (2nd derivative), not the level.
    """

    name = "X3-DollarWreckingBall"

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        # KILLED: Dollar acceleration signal produced Sharpe -0.12 after
        # two iterations. The 2nd derivative of UUP lacks sufficient
        # signal strength with daily ETF data. Archived.
        return weights
