"""Market regime detection — crisis classification engine (Category A).

Classifies each trading day into one of:
    NORMAL / ELEVATED / OIL_CRISIS / WAR_CRISIS / GENERAL_CRISIS / RECOVERY

Uses VIX, oil prices, defense sector momentum, credit spreads, realised
volatility, and market breadth as inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from financial_algo.indicators import breadth_count, ema, realized_vol, zscore


# ---------------------------------------------------------------------------
# Regime enum
# ---------------------------------------------------------------------------

class Regime(Enum):
    """Market regime labels."""

    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    OIL_CRISIS = "OIL_CRISIS"
    WAR_CRISIS = "WAR_CRISIS"
    GENERAL_CRISIS = "GENERAL_CRISIS"
    RECOVERY = "RECOVERY"


def is_crisis(regime: Regime) -> bool:
    """Return ``True`` if *regime* is any crisis state."""
    return regime in {Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RegimeConfig:
    """Tuneable thresholds for regime detection."""

    # VIX
    vix_elevated: float = 20.0
    vix_crisis: float = 30.0

    # Realised vol (annualised) on broad equity
    vol_elevated: float = 0.20
    vol_crisis: float = 0.35
    vol_window: int = 20

    # Oil z-score (rolling window on USO/CL-F)
    oil_zscore_window: int = 60
    oil_crisis_z: float = 2.0

    # Defense relative strength (ITA / SPY ratio z-score)
    defense_zscore_window: int = 60
    defense_war_z: float = 1.5

    # Credit spread proxy (HYG / LQD ratio z-score)
    credit_zscore_window: int = 60
    credit_crisis_z: float = -1.5  # ratio drops when spreads widen

    # Breadth
    breadth_lookback: int = 50
    breadth_crisis: float = 0.30  # < 30% above SMA → weak

    # Recovery
    recovery_vix_drop: float = 0.25  # VIX must drop 25% from recent peak
    recovery_lookback: int = 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_regime(
    prices: pd.DataFrame,
    vix: pd.Series | None = None,
    config: RegimeConfig | None = None,
) -> pd.Series:
    """Classify every bar into a :class:`Regime`.

    Parameters
    ----------
    prices:
        DataFrame of adjusted close prices.  Must include ``"SPY"`` and
        optionally ``"USO"``, ``"ITA"``, ``"HYG"``, ``"LQD"``, plus
        sector columns for breadth.
    vix:
        VIX close series (same index as *prices*). If ``None``, volatility is
        estimated from ``"SPY"`` realised vol as a proxy.
    config:
        Threshold overrides. Uses defaults when ``None``.

    Returns
    -------
    pd.Series
        Series of :class:`Regime` values, same index as *prices*.
    """
    if config is None:
        config = RegimeConfig()

    idx = prices.index
    regimes = pd.Series(Regime.NORMAL, index=idx)

    # --- VIX signal (or proxy) -------------------------------------------
    if vix is not None:
        vix = vix.reindex(idx).ffill()
    else:
        vix = realized_vol(prices["SPY"], config.vol_window) * 100  # proxy

    vix_elevated = vix >= config.vix_elevated
    vix_crisis = vix >= config.vix_crisis

    # --- Realised vol on SPY ---------------------------------------------
    rvol = realized_vol(prices["SPY"], config.vol_window)
    vol_elevated = rvol >= config.vol_elevated
    vol_crisis = rvol >= config.vol_crisis

    # --- Oil crisis flag --------------------------------------------------
    oil_flag = pd.Series(False, index=idx)
    if "USO" in prices.columns:
        oil_z = zscore(prices["USO"], config.oil_zscore_window)
        oil_flag = oil_z.abs() >= config.oil_crisis_z

    # --- War / defense flag -----------------------------------------------
    war_flag = pd.Series(False, index=idx)
    if "ITA" in prices.columns and "SPY" in prices.columns:
        defense_rel = prices["ITA"] / prices["SPY"]
        def_z = zscore(defense_rel, config.defense_zscore_window)
        war_flag = def_z >= config.defense_war_z

    # --- Credit spread flag -----------------------------------------------
    credit_flag = pd.Series(False, index=idx)
    if "HYG" in prices.columns and "LQD" in prices.columns:
        credit_ratio = prices["HYG"] / prices["LQD"]
        credit_z = zscore(credit_ratio, config.credit_zscore_window)
        credit_flag = credit_z <= config.credit_crisis_z

    # --- Breadth flag -----------------------------------------------------
    sector_cols = [c for c in prices.columns if c.startswith("XL")]
    breadth_flag = pd.Series(False, index=idx)
    if len(sector_cols) >= 3:
        bc = breadth_count(prices[sector_cols], config.breadth_lookback)
        breadth_flag = bc <= config.breadth_crisis

    # --- Recovery detection -----------------------------------------------
    vix_peak = vix.rolling(config.recovery_lookback).max()
    vix_dropped = vix <= vix_peak * (1 - config.recovery_vix_drop)

    # --- Assemble regimes (priority: specific crisis > general crisis > elevated > recovery > normal)
    regimes[vix_elevated | vol_elevated] = Regime.ELEVATED
    regimes[(vix_crisis | vol_crisis) & (credit_flag | breadth_flag)] = Regime.GENERAL_CRISIS
    regimes[oil_flag & (vix_elevated | vol_elevated)] = Regime.OIL_CRISIS
    regimes[war_flag & (vix_elevated | vol_elevated)] = Regime.WAR_CRISIS

    # Recovery: recently in crisis (within lookback window) and VIX falling
    crisis_mask = regimes.isin(
        {Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS}
    )
    was_crisis_recently = (
        crisis_mask.astype(float)
        .rolling(config.recovery_lookback, min_periods=1)
        .max()
        .shift(1)
        .fillna(0)
        .astype(bool)
    )
    regimes[was_crisis_recently & vix_dropped & ~vix_crisis] = Regime.RECOVERY

    return regimes


def detect_oil_crisis(prices: pd.DataFrame, config: RegimeConfig | None = None) -> pd.Series:
    """Return boolean Series — ``True`` on days classified as OIL_CRISIS."""
    return detect_regime(prices, config=config).isin({Regime.OIL_CRISIS})


def detect_war_crisis(prices: pd.DataFrame, config: RegimeConfig | None = None) -> pd.Series:
    """Return boolean Series — ``True`` on days classified as WAR_CRISIS."""
    return detect_regime(prices, config=config).isin({Regime.WAR_CRISIS})
