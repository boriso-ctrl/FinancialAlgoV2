"""Category H: Crisis Spike strategies.

Profits from UPWARD price spikes during crises — oil surges, commodity
shortages, gold rallies, defense sector pops.  Complements the
crash-defensive strategies (Cat D) which only profit from drawdowns.

Captures:
  - Oil embargo / war-driven energy surges (Iran, Gulf, Russia)
  - Gold surging on geopolitical fear
  - Defense stocks rallying on conflict escalation
  - Broad commodity shortage rallies
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from financial_algo.indicators import ema, realized_vol, zscore
from financial_algo.regimes import Regime
from financial_algo.strategies.base import Strategy


def _latch(entry: np.ndarray, exit_cond: np.ndarray) -> np.ndarray:
    """Set-reset latch: ON when *entry* fires, stays ON until *exit_cond*.

    Operates on raw numpy boolean arrays for speed (avoids pd.iloc overhead).
    """
    n = len(entry)
    state = np.zeros(n, dtype=bool)
    for i in range(1, n):
        state[i] = entry[i] or (state[i - 1] and not exit_cond[i])
    return state


# =========================================================================
# H1 — Commodity Shock Rider
# =========================================================================

@dataclass
class CommodityShockConfig:
    """Config for commodity shock rider strategy."""

    oil_ticker: str = "USO"
    energy_ticker: str = "XLE"
    gold_ticker: str = "GLD"

    # Breakout: z-score threshold for detecting upward spike
    zscore_window: int = 60
    spike_z: float = 1.4

    # Fast breakout: shorter z-score for catching fast-developing spikes
    fast_zscore_window: int = 20
    fast_spike_z: float = 2.1
    fast_momentum_threshold: float = 0.06

    # Trend filter: momentum over lookback
    momentum_window: int = 10
    momentum_threshold: float = 0.03

    # Leverage
    leverage_energy: float = 1.3
    leverage_gold: float = 0.6

    # Trail stop: exit when momentum fades
    exit_momentum_threshold: float = -0.02


class CommodityShockRider(Strategy):
    """Ride upward commodity shocks — long energy + gold on confirmed breakout.

    Unlike OilShockHedge (which also shorts SPY), this strategy is PURE
    upside capture: long energy and gold without shorting equity.
    Profits from Iran-style oil surges, OPEC cuts, supply disruptions.
    """

    name = "CommodityShockRider"

    def __init__(self, config: CommodityShockConfig | None = None) -> None:
        self.cfg = config or CommodityShockConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        required = [c.oil_ticker, c.energy_ticker, c.gold_ticker]
        if not all(t in prices.columns for t in required):
            return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        oil = prices[c.oil_ticker]

        # Upward spike detection: oil z-score high AND momentum positive
        oil_z = zscore(oil, c.zscore_window)
        oil_mom = oil.pct_change(c.momentum_window)

        # Standard entry: 60-day z-score ≥ 1.5 + 3% momentum
        spike_standard = (
            pd.notna(oil_z)
            & pd.notna(oil_mom)
            & (oil_z >= c.spike_z)
            & (oil_mom >= c.momentum_threshold)
        )

        # Fast entry: 20-day z-score ≥ 2.0 + 5% momentum
        # Catches fast-developing spikes that haven't built 60 days of context
        oil_z_fast = zscore(oil, c.fast_zscore_window)
        spike_fast = (
            pd.notna(oil_z_fast)
            & pd.notna(oil_mom)
            & (oil_z_fast >= c.fast_spike_z)
            & (oil_mom >= c.fast_momentum_threshold)
        )

        # Either trigger fires entry
        spike_up = spike_standard | spike_fast

        # Exit condition: momentum reversal
        momentum_fading = oil_mom <= c.exit_momentum_threshold

        # Vectorized state machine: carry entry forward until explicit exit.
        raw_state = pd.Series(np.nan, index=prices.index)
        raw_state[momentum_fading] = 0.0
        raw_state[spike_up] = 1.0
        in_trade = raw_state.ffill().fillna(0.0) > 0.5

        # Regime filter: more aggressive during crisis regimes
        regime_mult = pd.Series(1.0, index=prices.index)
        if regime is not None:
            crisis_mask = regime.isin({
                Regime.OIL_CRISIS, Regime.WAR_CRISIS,
                Regime.GENERAL_CRISIS, Regime.ELEVATED,
            })
            regime_mult[crisis_mask] = 1.5  # 50% more leverage during crisis

        tickers = [c.energy_ticker, c.gold_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        w.loc[in_trade, c.energy_ticker] = c.leverage_energy
        w.loc[in_trade, c.gold_ticker] = c.leverage_gold

        # Scale by regime multiplier
        w[c.energy_ticker] *= regime_mult
        w[c.gold_ticker] *= regime_mult

        return w.reindex(columns=prices.columns, fill_value=0.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


# =========================================================================
# H2 — Gold Fear Rally
# =========================================================================

@dataclass
class GoldFearConfig:
    """Config for gold fear rally strategy."""

    gold_ticker: str = "GLD"
    bond_ticker: str = "TLT"
    equity_ticker: str = "SPY"

    # Gold breakout: momentum over lookback
    momentum_window: int = 10
    momentum_threshold: float = 0.02  # 2% gold move

    # Confirmation: gold outperforming equity
    relative_window: int = 20
    relative_threshold: float = 0.03  # gold > SPY by 3%

    # Leverage
    leverage_gold: float = 3.0
    leverage_bonds: float = 1.0


class GoldFearRally(Strategy):
    """Long gold + bonds when gold surges on fear / geopolitical risk.

    Captures gold rallying from $1800 -> $2500+ during war scares,
    sanctions, inflation fears. Pure upside play on safe-haven demand.
    """

    name = "GoldFearRally"

    def __init__(self, config: GoldFearConfig | None = None) -> None:
        self.cfg = config or GoldFearConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        gold = prices[c.gold_ticker]
        spy = prices[c.equity_ticker]

        # Gold momentum breakout
        gold_mom = gold.pct_change(c.momentum_window)
        gold_breaking_out = gold_mom >= c.momentum_threshold

        # Gold outperforming equity (flight-to-safety signal)
        gold_rel = gold.pct_change(c.relative_window) - spy.pct_change(c.relative_window)
        gold_outperforming = gold_rel >= c.relative_threshold

        # Combined entry
        entry = gold_breaking_out & gold_outperforming

        # Stay in trade while gold momentum positive
        gold_mom_negative = (gold_mom < 0).values
        in_trade = pd.Series(
            _latch(entry.values, gold_mom_negative),
            index=prices.index,
        )

        tickers = [c.gold_ticker, c.bond_ticker]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        w.loc[in_trade, c.gold_ticker] = c.leverage_gold
        w.loc[in_trade, c.bond_ticker] = c.leverage_bonds

        return w


# =========================================================================
# H3 — Defense Spike Breakout
# =========================================================================

@dataclass
class DefenseSpikeConfig:
    """Config for defense spike breakout strategy."""

    defense_ticker: str = "ITA"
    contractor_a: str = "LMT"
    contractor_b: str = "RTX"
    equity_ticker: str = "SPY"

    # Breakout: defense sector surging vs market
    momentum_window: int = 5    # short window — catch fast spikes
    momentum_threshold: float = 0.03  # 3% in 5 days
    relative_threshold: float = 0.02  # 2% outperformance vs SPY

    # Trend: defense above EMA
    ema_span: int = 20

    # Leverage
    leverage_etf: float = 1.5
    leverage_per_stock: float = 1.0


class DefenseSpikeBreakout(Strategy):
    """Catch defense sector surges during geopolitical escalation.

    Short-term momentum strategy — enters FAST on defense breakouts
    (war declarations, sanctions, military buildups). Unlike
    C4-ArmsRaceMomentum which uses 63-day momentum, this uses 5-day
    to capture initial spike within first week.
    """

    name = "DefenseSpikeBreakout"

    def __init__(self, config: DefenseSpikeConfig | None = None) -> None:
        self.cfg = config or DefenseSpikeConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg
        ita = prices[c.defense_ticker]
        spy = prices[c.equity_ticker]

        # Fast momentum breakout
        ita_mom = ita.pct_change(c.momentum_window)
        spy_mom = spy.pct_change(c.momentum_window)
        relative_strength = ita_mom - spy_mom

        # Entry: defense surging + outperforming broad market
        entry = (ita_mom >= c.momentum_threshold) & (relative_strength >= c.relative_threshold)

        # Trend confirm: ITA above short EMA
        ita_ema = ema(ita, c.ema_span)
        entry = entry & (ita > ita_ema)

        # Regime boost: more aggressive during war/crisis regimes
        regime_mult = pd.Series(1.0, index=prices.index)
        if regime is not None:
            war_mask = regime.isin({Regime.WAR_CRISIS, Regime.GENERAL_CRISIS})
            regime_mult[war_mask] = 1.5

        # Stay in while momentum holds
        ita_mom_negative = (ita_mom < 0).values
        in_trade = pd.Series(
            _latch(entry.values, ita_mom_negative),
            index=prices.index,
        )

        tickers = [c.defense_ticker, c.contractor_a, c.contractor_b]
        w = pd.DataFrame(0.0, index=prices.index, columns=tickers)

        w.loc[in_trade, c.defense_ticker] = c.leverage_etf
        w.loc[in_trade, c.contractor_a] = c.leverage_per_stock
        w.loc[in_trade, c.contractor_b] = c.leverage_per_stock

        # Regime scaling
        for t in tickers:
            w[t] *= regime_mult

        return w


# =========================================================================
# H4 — Multi-Asset Crisis Long
# =========================================================================

@dataclass
class MultiAssetCrisisLongConfig:
    """Config for multi-asset crisis long strategy."""

    # Assets that rally during different crisis types
    energy_ticker: str = "XLE"
    gold_ticker: str = "GLD"
    defense_ticker: str = "ITA"
    bond_ticker: str = "TLT"
    dollar_ticker: str = "UUP"
    equity_ticker: str = "SPY"

    # Breakout parameters per asset
    momentum_window: int = 10
    zscore_window: int = 60

    # Dynamic allocation: score assets by strength, allocate to strongest
    top_n: int = 3
    # REWORK: Increased leverage_per_asset from 1.5 to 2.0x for conviction
    leverage_per_asset: float = 2.0
    # REWORK: Strengthened crisis detection threshold from 1.0 to 1.2 z-score
    # Requires stronger signal to confirm crisis is active
    crisis_zscore_threshold: float = 1.2


class MultiAssetCrisisLong(Strategy):
    """Dynamically long the strongest-performing crisis assets.

    During crises, different assets spike depending on the cause:
      - Oil crisis -> XLE, GLD surge
      - War crisis -> ITA, GLD, TLT surge
      - Financial crisis -> TLT, GLD, UUP surge

    This strategy scores all crisis-beneficiary assets by recent momentum
    and goes long the top N strongest. Adapts to the TYPE of crisis
    automatically without pre-classifying it.
    """

    name = "MultiAssetCrisisLong"

    def __init__(self, config: MultiAssetCrisisLongConfig | None = None) -> None:
        self.cfg = config or MultiAssetCrisisLongConfig()

    def generate_weights(
        self,
        prices: pd.DataFrame,
        regime: pd.Series | None = None,
    ) -> pd.DataFrame:
        c = self.cfg

        crisis_assets = [
            c.energy_ticker, c.gold_ticker, c.defense_ticker,
            c.bond_ticker, c.dollar_ticker,
        ]
        # Only use assets present in prices
        available = [t for t in crisis_assets if t in prices.columns]

        if not available:
            return pd.DataFrame(0.0, index=prices.index, columns=[c.equity_ticker])

        # REWORK: Compute conviction scores = z-score * momentum
        # Strong conviction = high z-score (extreme) AND positive momentum
        # This replaces pure rank-based allocation with conviction weighting
        conviction = pd.DataFrame(index=prices.index, columns=available)
        for t in available:
            t_z = zscore(prices[t], c.zscore_window)
            t_mom = prices[t].pct_change(c.momentum_window).fillna(0.0)
            # Conviction only positive when BOTH z-score and momentum are strong
            # Clip to exclude negative contributions (long-only strategy)
            conviction[t] = (t_z * t_mom).clip(lower=0.0)

        # Rank assets by conviction each day
        conviction_ranks = conviction.rank(axis=1, ascending=False)

        # Determine "crisis active" days: at least one asset with strong conviction
        crisis_active = pd.Series(False, index=prices.index)
        extreme_spike = pd.Series(False, index=prices.index)
        
        for t in available:
            t_z = zscore(prices[t], c.zscore_window)
            t_mom = prices[t].pct_change(c.momentum_window).fillna(0.0)
            # Crisis active: z-score > threshold AND positive momentum
            crisis_active = crisis_active | ((t_z >= c.crisis_zscore_threshold) & (t_mom > 0))
            # Extreme spike: z-score > 2.0 (bypasses regime gate)
            extreme_spike = extreme_spike | (t_z >= 2.0)

        # Regime gate: require elevated/crisis regime OR extreme z-score
        # (bypasses regime gate when a truly extreme spike is happening)
        if regime is not None:
            active_regime = regime.isin({
                Regime.ELEVATED, Regime.OIL_CRISIS,
                Regime.WAR_CRISIS, Regime.GENERAL_CRISIS,
                Regime.RECOVERY,
            })
            crisis_active = crisis_active & (active_regime | extreme_spike)

        # REWORK: Conviction-weighted allocation instead of simple ranks
        # Allocate position size proportional to conviction score x leverage
        w = pd.DataFrame(0.0, index=prices.index, columns=available)

        # For each day, compute allocation across top-N assets
        for idx in prices.index:
            if not crisis_active.loc[idx]:
                continue
            
            # Get conviction scores for all assets on this day
            convictions = {t: conviction.loc[idx, t] for t in available}
            # Filter to top-N by rank
            top_assets = sorted([(t, convictions[t]) for t in available], 
                               key=lambda x: -convictions[x[0]])[:c.top_n]
            top_assets = [(t, convictions[t]) for t, _ in top_assets if convictions[t] > 0]
            
            if not top_assets:
                continue
            
            # Allocate proportionally to conviction
            total_conviction = sum(conv for _, conv in top_assets)
            if total_conviction > 0:
                for t, conv in top_assets:
                    w.loc[idx, t] = (conv / total_conviction) * c.leverage_per_asset

        return w
