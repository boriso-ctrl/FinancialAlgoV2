"""Comprehensive Historical Crisis Backtest — All Categories.

Downloads real market data via yfinance and runs every strategy category
through historical crises with unbiased backtesting:
  - Weights shifted +1 day (no look-ahead bias)
  - Transaction costs (5 bps one-way)
  - Leverage borrowing costs (1.5 %/yr)
  - Short borrow costs (0.5 %/yr)

Crisis periods tested:
  2011       — EU Sovereign Debt Crisis
  2014-2016  — Oil Price Crash (OPEC, shale glut)
  2018 Q1/Q4 — Vol-mageddon + Fed tightening
  2020       — COVID-19 pandemic crash
  2022       — Russia-Ukraine War + inflation
  2023-2025  — Recent (baseline/recovery)

Usage:
    python scripts/production/run_crisis_backtest.py
"""

from __future__ import annotations

import csv
import sys
import uuid
import warnings
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

# Ensure the repository src/ is importable when running as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.logging_utils import get_logger, setup_logging, trace_context
from financial_algo.regimes import Regime, RegimeConfig, detect_regime

# --- Strategy imports ---
from financial_algo.strategies.crash_hedge import (
    CrashHedgeQQQ,
    FourStateTactical,
    VolCarry,
    AdaptiveStopTrend,
)
from financial_algo.strategies.oil_crisis import (
    EnergyPairs,
    OilMeanReversion,
    OilMomentumSurge,
    OilShockHedge,
)
from financial_algo.strategies.war_crisis import (
    ArmsRaceMomentum,
    DefenseRotation,
    PostWarRecovery,
    SafeHavenFlight,
)
from financial_algo.strategies.pairs import MultiPairPortfolio
from financial_algo.strategies.crypto_crisis import (
    CryptoFlightToQuality,
    CryptoGoldDivergence,
    CryptoRecoverySurge,
    CryptoContagionHedge,
    CryptoRecoverySurgeATR,
)
from financial_algo.strategies.crisis_spike import (
    CommodityShockRider,
    DefenseSpikeBreakout,
    GoldFearRally,
    MultiAssetCrisisLong,
)
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig
from financial_algo.strategies.base import Strategy

# --- New strategy imports (Categories I-P) ---
from financial_algo.strategies.momentum import (
    TimeSeriesMomentum,
    CrossSectionalMomentum,
    DualMomentum,
    MomentumVolScaled,
    GlobalMomentumRotation,
    AdaptiveTrendFilter,
    DriftRegimeMomentum,
    TimeSeriesMomentumDrift,
)
from financial_algo.strategies.mean_reversion import (
    SectorMeanReversion,
    RSIMeanReversion,
    OvernightGapFade,
    CointegrationPairs,
    GlobalMeanReversion,
    DriftReversalAlpha,
)
from financial_algo.strategies.fixed_income import (
    YieldCurveTrade,
    CreditSpreadMeanRev,
)
from financial_algo.strategies.volatility_strats import (
    VolRiskPremium,
    VolSpreadHarvest,
    VolOfVolRegime,
    VolTermStructure,
    VolSpikeRecovery,
    CrossAssetVolSignal,
    ImpliedRealizedSpread,
    VolRegimeClustering,
    VIXAdaptiveCarry,
    DynamicVolRegimeSwitch,
    VolRiskPremiumAdaptive,
)
from financial_algo.strategies.macro import (
    DollarCarry,
    GoldDollarInverse,
    EMRiskPremium,
    CommodityMomentum,
    RatesRegimeTrade,
    GlobalRotation,
    CommodityMacroSignal,
    YieldCurveRegime,
    MacroSignalScoreboard,
    AdaptiveMacroBlend,
    DollarCarryScoreboard,
)
from financial_algo.strategies.seasonal import (
    SeasonalStrategy,
    TurnOfMonth,
    PreHolidayDrift,
)
from financial_algo.strategies.factor import (
    LowVolFactor,
    MultiFactorComposite,
    SizeFactor,
    ValueFactor,
    RealAssetsFactor,
    QualityMomentumComposite,
)
from financial_algo.strategies.tail_risk import (
    TailRiskParity,
    BlackSwanInsurance,
    TailHedgeOverlay,
    PreciousMetalsCrisisHedge,
    VolatilityConvexity,
    ATRCrisisAlpha,
)
from financial_algo.strategies.signal_combo import (
    FeatureComboSignal,
    MultiSignalConsensus,
    XGBoostSignalCombo,
    GMMRegimeClassifier,
)
from financial_algo.strategies.ml_strategies import (
    AdaptiveThreshold,
    CrossSectionalRanker,
)
from financial_algo.strategies.dl_strategies import (
    TemporalCNNAlpha,
    LSTMRegimeDetector,
    AttentionCrossSectionalRanker,
)
from financial_algo.strategies.quality_trend import (
    QualityTrend,
    MultiAssetTrend,
    MomentumCrashFilter,
)
from financial_algo.strategies.multi_freq import (
    WeeklyMomentumRotation,
    MonthlyMacroRegime,
    MultiTimeframeTrend,
    WeeklyMeanReversion,
)
from financial_algo.strategies.regime_hardening import (
    BearMarketAlpha, DefensiveRotationR3,
    AdaptiveRiskBudget, RatesTighteningAlpha, BondEquityHedge,
    VolExplosionAlpha, VolRegimeSwitcher, MultiAssetCTATrend,
    CommodityMacroOverlay,
)

# --- Fundamental strategy imports ---
from financial_algo.fundamental import build_synthetic_sentiment
from financial_algo.fundamental.strategies import (
    FearGreedContrarian,
    SentimentCrisisAlpha,
    SentimentDivergence,
    SentimentEnhancedRegime,
    CryptoSentimentDivergence,
)

warnings.filterwarnings("ignore", category=FutureWarning)

logger = get_logger(__name__)

# =========================================================================
# Configuration
# =========================================================================

# Full date range — enough warm-up before first crisis window
DATA_START = "2009-01-01"
DATA_END = "2025-12-31"

# Historical crisis windows for focused analysis
CRISIS_WINDOWS: dict[str, tuple[str, str]] = {
    "Full Period (2010-2025)":        ("2010-01-01", "2025-12-31"),
    "EU Debt Crisis (2011)":          ("2011-01-01", "2012-01-01"),
    "Oil Crash (2014-2016)":          ("2014-06-01", "2016-06-01"),
    "Volmageddon + Fed (2018)":       ("2018-01-01", "2019-01-01"),
    "COVID-19 (2020)":                ("2020-01-01", "2021-01-01"),
    "Russia-Ukraine + Inflation (2022)": ("2022-01-01", "2023-01-01"),
    "Recovery & Recent (2023-2025)":  ("2023-01-01", "2025-12-31"),
    # Middle East conflict-specific windows
    "ME: Red Sea / Houthi (Jan-Mar 2024)":  ("2024-01-01", "2024-04-01"),
    "ME: Iran Tensions (Sep-Oct 2024)":     ("2024-09-01", "2024-11-01"),
    "ME: Oil Spike Jun 2025":               ("2025-05-01", "2025-07-31"),
    "ME: Full Conflict (Oct23-Dec24)":      ("2023-10-01", "2024-12-31"),
}

# Cost model with risk controls (same for all strategies -- fair comparison)
BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,              # 20% annualised vol target
    max_drawdown_trigger=-0.25,   # cut exposure at -25% drawdown
    drawdown_recovery_rate=0.10,  # resume after 10% DD improvement
)

# Ensemble gets its own config WITHOUT backtest-level DD trigger
# (the EnsembleStrategy class has its own gradual circuit breaker)
# Also enable per-strategy DD scale-down for additional protection
BT_CONFIG_ENSEMBLE = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=None,    # ensemble handles DD internally
    strategy_dd_scale_start=-0.15,   # start scaling at -15% DD
    strategy_dd_scale_end=-0.25,     # fully flat at -25% DD
)

# All tickers the strategies need (deduplicated)
TICKERS = sorted(set([
    # Broad equity
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    # Safe haven
    "GLD", "SLV", "TLT", "IEF", "SHY", "UUP",
    # Energy
    "XLE", "USO", "XOP",
    # Defense
    "ITA", "LMT", "RTX",
    # Sectors (for breadth / pairs / rotation)
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "XBI", "XLC", "XLRE",
    # Commodities
    "DBC", "DBA",
    # Fixed income / credit
    "HYG", "LQD", "TIP", "AGG", "EMB",
    # Regional
    "FXI", "VGK", "EWJ", "INDA",
    # Real estate
    "VNQ",
    # Crypto
    "BTC-USD", "ETH-USD",
]))

VIX_TICKER = "^VIX"
DECISION_STATE_CSV = REPO_ROOT / "results" / "sprint15_go_no_go.csv"

# Alpha-chasing policy guards for decision-state driven membership.
MIN_KEEP_SHARPE = 0.35
MIN_PROMOTE_SHARPE = 0.60
MAX_PROMOTIONS = 6
MAX_ENSEMBLE_MEMBERS = 32

# Keep this sprint's policy non-ML while we optimize core signal sleeves.
BLOCK_ML_PROMOTIONS = True
BLOCK_DL_PROMOTIONS = True

# Prevent over-concentration from too many promotions in one sleeve.
PROMOTION_PREFIX_CAPS = {
    "O": 1,   # tail-risk sleeve
    "L": 2,   # volatility sleeve
    "H": 1,   # crisis spike sleeve
    "B": 1,   # oil crisis sleeve
    "R": 2,   # regime sleeve
    "G": 2,   # sentiment sleeve
}


# =========================================================================
# Helpers
# =========================================================================

def run_strategy_backtest(
    name: str,
    strategy,
    prices: pd.DataFrame,
    regime: pd.Series,
    config: BacktestConfig,
    needs_regime: bool = False,
    sentiment_df: pd.DataFrame | None = None,
) -> dict | None:
    """Run a single strategy and return metrics, or None on error."""
    try:
        if sentiment_df is not None:
            # Fundamental strategies accept sentiment_df
            weights = strategy.generate_weights(prices, regime, sentiment_df)
            weights = weights.shift(1).fillna(0.0)
        elif needs_regime:
            weights = strategy.backtest_weights(prices, regime)
        else:
            weights = strategy.backtest_weights(prices)

        result = backtest(prices, weights, config)
        return result["metrics"]
    except Exception as e:
        logger.warning("Strategy backtest failed for %s: %s", name, e, exc_info=True)
        return None


def metrics_row(metrics: dict | None) -> dict:
    """Format metrics for display, handling None."""
    if metrics is None:
        return {
            "CAGR": "ERR", "Sharpe": "ERR", "Sortino": "ERR",
            "MaxDD": "ERR", "Calmar": "ERR", "AnnVol": "ERR",
            "WinRate": "ERR", "TotRet": "ERR", "Trades/Mo": "ERR",
        }
    return {
        "CAGR": f"{metrics['cagr']:.2%}",
        "Sharpe": f"{metrics['sharpe']:.2f}",
        "Sortino": f"{metrics['sortino']:.2f}",
        "MaxDD": f"{metrics['max_drawdown']:.2%}",
        "Calmar": f"{metrics['calmar']:.2f}",
        "AnnVol": f"{metrics['annual_vol']:.2%}",
        "WinRate": f"{metrics['win_rate']:.2%}",
        "TotRet": f"{metrics['total_return']:.2%}",
        "Trades/Mo": f"{metrics.get('avg_trades_per_month', 0):.1f}",
    }


def load_decision_state(path: Path) -> dict[str, dict[str, str | float]]:
    """Load strategy decisions keyed by strategy name.

    Returns an empty dict when the decision file is not present so the runner
    can still execute with static defaults.
    """
    if not path.exists():
        return {}

    out: dict[str, dict[str, str | float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("strategy") or "").strip()
            if not name:
                continue
            try:
                curr_sharpe = float((row.get("current_sharpe") or "0").strip())
            except ValueError:
                curr_sharpe = 0.0
            out[name] = {
                "decision": (row.get("decision") or "").strip(),
                "current_sharpe": curr_sharpe,
            }
    return out


def apply_decision_filters(
    registry: dict[str, list[tuple[str, object, bool]]],
    decisions: dict[str, dict[str, str | float]],
) -> dict[str, list[tuple[str, object, bool]]]:
    """Apply rollback decisions to the per-strategy registry.

    Current policy:
      - rollback_required: remove from active run registry
      - all other decisions: keep in registry
    """
    if not decisions:
        return registry

    filtered: dict[str, list[tuple[str, object, bool]]] = {}
    for cat_name, strats in registry.items():
        kept: list[tuple[str, object, bool]] = []
        for strat_name, strat, needs_regime in strats:
            decision = str(decisions.get(strat_name, {}).get("decision", ""))
            if decision == "rollback_required":
                continue
            kept.append((strat_name, strat, needs_regime))
        filtered[cat_name] = kept
    return filtered


def build_strategy_lookup(
    registry: dict[str, list[tuple[str, object, bool]]],
) -> dict[str, tuple[object, bool]]:
    """Build {strategy_name: (strategy_instance, needs_regime)} lookup."""
    lookup: dict[str, tuple[object, bool]] = {}
    for strats in registry.values():
        for strat_name, strat, needs_regime in strats:
            lookup[strat_name] = (strat, needs_regime)
    return lookup


def build_ensemble_from_decisions(
    lookup: dict[str, tuple[object, bool]],
    decisions: dict[str, dict[str, str | float]],
) -> tuple[list[object], list[float], list[str]]:
    """Build ensemble members and prior scores from decision-state policy.

    Policy:
      - Start from the previous curated baseline list
      - Remove `rollback_required` and `rework_priority`
      - Add all `promote` strategies present in the active lookup
    """
    base_names = [
        "MF2-MonthlyMacroRegime",
        "R9-MultiAssetCTATrend",
        "R6-BondEquityHedge",
        "R4-AdaptiveRiskBudget",
        "L6-CrossAssetVolSignal",
        "P1-FeatureComboSignal",
        "D2-CrashHedgeQQQ",
        "F2-CryptoRecoverySurge",
        "L1-VolRiskPremium",
        "L4-VolOfVolRegime",
        "O7-PreciousMetalsCrisisHedge",
        "O1-TailRiskParity",
        "F3-CryptoGoldDivergence",
        "G2-FearGreedContrarian",
        "D3-VolCarry",
        "R3-DefensiveRotation",
        "L3-VolTermStructure",
        "L2-VolSpreadHarvest",
        "R7-VolExplosionAlpha",
        "K1-LowVolFactor",
        "Q1-QualityTrend",
        "L5-VolSpikeRecovery",
        "Q3-MomentumCrashFilter",
        "K4-ValueFactor",
        "G1-SentimentCrisisAlpha",
        "N1-SeasonalStrategy",
        "G3-SentimentDivergence",
        "P4-AdaptiveThreshold",
        "L8-VolRegimeClustering",
        "M9-YieldCurveRegime",
    ]

    def _safe_sharpe(name: str, default: float = 0.5) -> float:
        raw = decisions.get(name, {}).get("current_sharpe", default)
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    def _family(name: str) -> str:
        token = name.split("-", 1)[0]
        letters = "".join(ch for ch in token if ch.isalpha())
        if letters:
            return letters
        return token[:1]

    selected = [n for n in base_names if n in lookup]
    if decisions:
        selected = [
            n
            for n in selected
            if str(decisions.get(n, {}).get("decision", "")) not in {"rollback_required", "rework_priority"}
        ]

        # Remove weak strategies from the active set when decision-state metrics are available.
        selected = [
            n
            for n in selected
            if _safe_sharpe(n) >= MIN_KEEP_SHARPE
        ]

        promoted_ranked = [
            (n, _safe_sharpe(n))
            for n, meta in decisions.items()
            if str(meta.get("decision", "")) == "promote" and n in lookup
        ]

        if BLOCK_ML_PROMOTIONS or BLOCK_DL_PROMOTIONS:
            filtered_promotions: list[tuple[str, float]] = []
            for n, s in promoted_ranked:
                fam = _family(n)
                is_dl = fam == "DL"
                is_ml = fam == "P"
                if BLOCK_DL_PROMOTIONS and is_dl:
                    continue
                if BLOCK_ML_PROMOTIONS and is_ml:
                    continue
                filtered_promotions.append((n, s))
            promoted_ranked = filtered_promotions

        promoted_ranked = [
            (n, s)
            for n, s in promoted_ranked
            if s >= MIN_PROMOTE_SHARPE
        ]
        promoted_ranked.sort(key=lambda x: x[1], reverse=True)

        added = 0
        added_by_prefix: dict[str, int] = {}
        for name, _ in promoted_ranked:
            if name in selected:
                continue
            fam = _family(name)
            cap = PROMOTION_PREFIX_CAPS.get(fam)
            if cap is not None and added_by_prefix.get(fam, 0) >= cap:
                continue
            if added >= MAX_PROMOTIONS or len(selected) >= MAX_ENSEMBLE_MEMBERS:
                break
            selected.append(name)
            added += 1
            added_by_prefix[fam] = added_by_prefix.get(fam, 0) + 1

    # Hard cap as final guard against decision-state expansion.
    selected = selected[:MAX_ENSEMBLE_MEMBERS]

    members = [lookup[n][0] for n in selected]
    sharpe_scores: list[float] = []
    for name in selected:
        s = _safe_sharpe(name)
        sharpe_scores.append(max(s, 0.10))

    return members, sharpe_scores, selected


# =========================================================================
# Strategy registry — every strategy, categorised
# =========================================================================

def build_strategy_registry():
    """Return ordered dict of {category: [(name, strategy_instance, needs_regime)]}."""
    return {
        "Cat B: Oil Crisis": [
            ("B1-OilMomentumSurge",  OilMomentumSurge(),   True),
            ("B2-OilShockHedge",     OilShockHedge(),      True),
            ("B3-OilMeanReversion",  OilMeanReversion(),   True),
            ("B4-EnergyPairs",       EnergyPairs(),        True),
        ],
        "Cat C: War/Geopolitical": [
            ("C1-DefenseRotation",   DefenseRotation(),    True),
            ("C2-SafeHavenFlight",   SafeHavenFlight(),    True),
            ("C3-PostWarRecovery",   PostWarRecovery(),    True),
            ("C4-ArmsRaceMomentum",  ArmsRaceMomentum(),   False),
        ],
        "Cat D: Crash-Hedge (General)": [
            ("D1-FourStateTactical", FourStateTactical(),  True),
            ("D2-CrashHedgeQQQ",     CrashHedgeQQQ(),     False),
            ("D3-VolCarry",          VolCarry(),           False),
            ("D4-AdaptiveStopTrend", AdaptiveStopTrend(),  False),
        ],
        "Cat E: Pairs Arbitrage": [
            ("E1-MultiPairPortfolio", MultiPairPortfolio(), False),
        ],
        "Cat F: Crypto Crisis": [
            ("F1-CryptoFlightToQuality", CryptoFlightToQuality(), True),
            ("F2-CryptoRecoverySurge",   CryptoRecoverySurge(),   True),
            ("F3-CryptoGoldDivergence",  CryptoGoldDivergence(),  True),
            ("F4-CryptoContagionHedge",  CryptoContagionHedge(),  False),
            ("F2b-CryptoRecoverySurgeATR", CryptoRecoverySurgeATR(), True),
        ],
        "Cat G: Fundamental/Sentiment": [
            ("G1-SentimentCrisisAlpha",   SentimentCrisisAlpha(),   True),
            ("G2-FearGreedContrarian",    FearGreedContrarian(),    True),
            ("G3-SentimentDivergence",    SentimentDivergence(),    True),
            ("G4-SentimentEnhancedRegime", SentimentEnhancedRegime(), True),
            ("G5-CryptoSentimentDivergence", CryptoSentimentDivergence(), True),
        ],
        "Cat H: Crisis Spike (Upward)": [
            ("H1-CommodityShockRider",  CommodityShockRider(),  True),
            ("H2-GoldFearRally",        GoldFearRally(),        False),
            ("H3-DefenseSpikeBreakout", DefenseSpikeBreakout(), True),
            ("H4-MultiAssetCrisisLong", MultiAssetCrisisLong(), True),
        ],
        "Cat H-FI: Fixed Income": [
            ("H1-YieldCurveTrade",        YieldCurveTrade(),         False),
            ("H2-CreditSpreadMeanRev",    CreditSpreadMeanRev(),     False),
        ],
        "Cat I: Momentum": [
            ("I1-TimeSeriesMomentum",     TimeSeriesMomentum(),      False),
            ("I2-CrossSectionalMomentum", CrossSectionalMomentum(),  False),
            ("I3-DualMomentum",           DualMomentum(),            False),
            ("I4-MomentumVolScaled",      MomentumVolScaled(),       False),
            ("I5-GlobalMomentumRotation", GlobalMomentumRotation(),  False),
            ("I10-AdaptiveTrendFilter",   AdaptiveTrendFilter(),     False),
            ("I7-DriftRegimeMomentum",    DriftRegimeMomentum(),     False),
            ("I8-TimeSeriesMomDrift",      TimeSeriesMomentumDrift(),  False),
        ],
        "Cat J: Mean Reversion": [
            ("J1-SectorMeanReversion",    SectorMeanReversion(),     False),
            ("J2-OvernightGapFade",       OvernightGapFade(),        False),
            ("J3-RSIMeanReversion",       RSIMeanReversion(),        False),
            ("J4-CointegrationPairs",     CointegrationPairs(),      False),
            ("J5-GlobalMeanReversion",    GlobalMeanReversion(),     False),
            ("J6-DriftReversalAlpha",     DriftReversalAlpha(),      False),
        ],
        "Cat K: Factor": [
            ("K1-LowVolFactor",           LowVolFactor(),            False),
            ("K2-MultiFactorComposite",   MultiFactorComposite(),    False),
            ("K3-SizeFactor",             SizeFactor(),              False),
            ("K4-ValueFactor",            ValueFactor(),             False),
            ("K5-RealAssetsFactor",       RealAssetsFactor(),        False),
            ("K6-QualityMomentumComposite", QualityMomentumComposite(), False),
        ],
        "Cat L: Volatility": [
            ("L1-VolRiskPremium",         VolRiskPremium(),          False),
            ("L2-VolSpreadHarvest",       VolSpreadHarvest(),        False),
            ("L3-VolTermStructure",       VolTermStructure(),        False),
            ("L4-VolOfVolRegime",         VolOfVolRegime(),          False),
            ("L5-VolSpikeRecovery",       VolSpikeRecovery(),        False),
            ("L6-CrossAssetVolSignal",    CrossAssetVolSignal(),     False),
            ("L7-ImpliedRealizedSpread", ImpliedRealizedSpread(),   False),
            ("L8-VolRegimeClustering",   VolRegimeClustering(),     False),
            ("L9-VIXAdaptiveCarry",      VIXAdaptiveCarry(),        False),
            ("L10-DynamicVolRegimeSwitch", DynamicVolRegimeSwitch(), False),
            ("L11-VolRiskPremiumAdaptive", VolRiskPremiumAdaptive(), False),
        ],
        "Cat M: Macro": [
            ("M1-DollarCarry",            DollarCarry(),             False),
            ("M2-GoldDollarInverse",      GoldDollarInverse(),       False),
            ("M3-EMRiskPremium",          EMRiskPremium(),           False),
            ("M4-CommodityMomentum",      CommodityMomentum(),       False),
            ("M5-RatesRegimeTrade",       RatesRegimeTrade(),        False),
            ("M7-GlobalRotation",         GlobalRotation(),          False),
            ("M8-CommodityMacroSignal",   CommodityMacroSignal(),    False),
            ("M9-YieldCurveRegime",       YieldCurveRegime(),        False),
            ("M10-MacroSignalScoreboard", MacroSignalScoreboard(),   False),
            ("M11-AdaptiveMacroBlend",    AdaptiveMacroBlend(),      False),
            ("M1b-DollarCarryScoreboard", DollarCarryScoreboard(),   False),
        ],
        "Cat N: Seasonal": [
            ("N1-SeasonalStrategy",       SeasonalStrategy(),        False),
            ("N2-TurnOfMonth",            TurnOfMonth(),             False),
            ("N3-PreHolidayDrift",        PreHolidayDrift(),         False),
        ],
        "Cat O: Tail Risk": [
            ("O1-TailRiskParity",         TailRiskParity(),          False),
            ("O4-BlackSwanInsurance",      BlackSwanInsurance(),      False),
            ("O6-TailHedgeOverlay",       TailHedgeOverlay(),        True),
            ("O7-PreciousMetalsCrisisHedge", PreciousMetalsCrisisHedge(), False),
            ("O8-VolatilityConvexity",    VolatilityConvexity(),     False),
            ("O9-ATRCrisisAlpha",         ATRCrisisAlpha(),          True),
        ],
        "Cat P: ML Signal Combo": [
            ("P1-FeatureComboSignal",     FeatureComboSignal(),      False),
            ("P2-XGBoostSignalCombo",     XGBoostSignalCombo(),      False),
            ("P3-GMMRegimeClassifier",    GMMRegimeClassifier(),     False),
            ("P4-AdaptiveThreshold",      AdaptiveThreshold(),       False),
            ("P5-CrossSectionalRanker",   CrossSectionalRanker(),    False),
            ("G6-MultiSignalConsensus",   MultiSignalConsensus(),    False),
        ],
        "Cat DL: Deep Learning": [
            ("DL1-TemporalCNNAlpha",      TemporalCNNAlpha(),        False),
            ("DL2-LSTMRegimeDetector",    LSTMRegimeDetector(),      False),
            ("DL3-AttentionRanker",       AttentionCrossSectionalRanker(), False),
        ],
        "Cat Q: Quality Trend": [
            ("Q1-QualityTrend",           QualityTrend(),            False),
            ("Q2-MultiAssetTrend",        MultiAssetTrend(),         False),
            ("Q3-MomentumCrashFilter",    MomentumCrashFilter(),     False),
        ],
        "Cat MF: Multi-Frequency": [
            ("MF1-WeeklyMomentumRotation", WeeklyMomentumRotation(), False),
            ("MF2-MonthlyMacroRegime",     MonthlyMacroRegime(),     False),
            ("MF3-MultiTimeframeTrend",    MultiTimeframeTrend(),    False),
            ("MF4-WeeklyMeanReversion",    WeeklyMeanReversion(),    False),
        ],
        "Cat R: Regime Hardening": [
            ("R1-BearMarketAlpha",        BearMarketAlpha(),         True),
            ("R3-DefensiveRotation",      DefensiveRotationR3(),     True),
            ("R4-AdaptiveRiskBudget",     AdaptiveRiskBudget(),      False),
            ("R5-RatesTighteningAlpha",   RatesTighteningAlpha(),    False),
            ("R6-BondEquityHedge",        BondEquityHedge(),         False),
            ("R7-VolExplosionAlpha",      VolExplosionAlpha(),       False),
            ("R8-VolRegimeSwitcher",      VolRegimeSwitcher(),       False),
            ("R9-MultiAssetCTATrend",     MultiAssetCTATrend(),      False),
            ("R10-CommodityMacroOverlay", CommodityMacroOverlay(),   False),
        ],
    }


# =========================================================================
# Benchmark — Buy & Hold SPY
# =========================================================================

def spy_benchmark(prices: pd.DataFrame, config: BacktestConfig) -> dict | None:
    """Simple 1x long SPY benchmark."""
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w["SPY"] = 1.0
    try:
        result = backtest(prices, w.shift(1).fillna(0), config)
        return result["metrics"]
    except Exception as e:
        logger.warning("Benchmark backtest failed: %s", e, exc_info=True)
        return None


def _inject_features(strategies, features: "pd.DataFrame") -> None:
    """Inject intraday feature DataFrame into any strategy that accepts it."""
    for strat in strategies:
        if hasattr(strat, "set_intraday_features"):
            strat.set_intraday_features(features)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    setup_logging(service="crisis_backtest", environment="production")
    run_trace_id = f"run-{uuid.uuid4().hex[:12]}"
    logger.info("Starting run_crisis_backtest", extra={"run_id": run_trace_id})

    with trace_context(run_trace_id):
        _main_impl()


def _main_impl() -> None:
    print("=" * 80)
    print("CRISIS-THRIVING STRATEGY BACKTEST - UNBIASED HISTORICAL ANALYSIS")
    print("=" * 80)
    print()

    # ------------------------------------------------------------------
    # 1. Download data
    # ------------------------------------------------------------------
    print(f"[1/4] Downloading price data for {len(TICKERS)} tickers "
          f"({DATA_START} -> {DATA_END}) ...")
    prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
    print(f"       Loaded: {prices.shape[0]} trading days x {prices.shape[1]} tickers")
    print(f"       Date range: {prices.index[0].date()} -> {prices.index[-1].date()}")
    print()

    # Download VIX separately (non-tradeable index)
    print("       Downloading VIX ...")
    try:
        vix_df = load_prices([VIX_TICKER], start=DATA_START, end=DATA_END)
        vix = vix_df[VIX_TICKER]
    except Exception:
        print("       [WARN] VIX download failed, using vol proxy")
        logger.warning("VIX download failed; using realized-vol proxy", exc_info=True)
        vix = None
    print()

    # ------------------------------------------------------------------
    # 2. Detect regime over full period
    # ------------------------------------------------------------------
    print("[2/4] Running regime detection ...")
    regime = detect_regime(prices, vix=vix)
    regime_counts = regime.value_counts()
    print("       Regime distribution (full period):")
    for r, count in regime_counts.items():
        pct = count / len(regime) * 100
        regime_name = r.value if isinstance(r, Regime) else str(r)
        print(f"         {regime_name:20s}: {count:5d} days ({pct:5.1f}%)")
    print()

    # ------------------------------------------------------------------
    # 3. Load intraday features (Phase 2: Alpaca pipeline, optional)
    # ------------------------------------------------------------------
    intraday_features: "pd.DataFrame | None" = None
    try:
        import os
        if os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET"):
            from financial_algo.data.feature_store import FeatureStore
            equity_tickers = [t for t in TICKERS
                              if not t.startswith("^") and "-USD" not in t]
            print("[2b/4] Loading Alpaca intraday features ...")
            store = FeatureStore(tickers=equity_tickers, start=DATA_START, end=DATA_END)
            intraday_features = store.load()
            if intraday_features is not None and not intraday_features.empty:
                print(f"       Loaded {intraday_features.shape[1]} intraday features "
                      f"for {intraday_features.shape[1] // 7} tickers")
            else:
                intraday_features = None
            print()
    except Exception as e:
        print(f"       [WARN] Intraday feature load failed (skipping): {e}")
        logger.warning("Intraday feature load failed (skipping): %s", e, exc_info=True)
        intraday_features = None

    # ------------------------------------------------------------------
    # 4. Build strategy registry
    # ------------------------------------------------------------------
    registry = build_strategy_registry()
    decision_state = load_decision_state(DECISION_STATE_CSV)
    registry = apply_decision_filters(registry, decision_state)
    strategy_lookup = build_strategy_lookup(registry)

    if decision_state:
        counts = {
            "promote": 0,
            "hold_watch": 0,
            "rework_priority": 0,
            "rollback_required": 0,
        }
        for meta in decision_state.values():
            d = str(meta.get("decision", ""))
            if d in counts:
                counts[d] += 1
        print(
            "[3/4] Loaded decision state "
            f"({counts['promote']} promote, {counts['hold_watch']} hold, "
            f"{counts['rework_priority']} rework, {counts['rollback_required']} rollback)"
        )
    else:
        print("[3/4] No decision-state CSV found; using static strategy set")
    print()

    ensemble_members, sharpe_scores, ensemble_names = build_ensemble_from_decisions(
        strategy_lookup,
        decision_state,
    )
    print(f"       Ensemble member count: {len(ensemble_members)}")
    print(
        "       Alpha policy "
        f"(keep>={MIN_KEEP_SHARPE:.2f}, promote>={MIN_PROMOTE_SHARPE:.2f}, "
        f"max_promotions={MAX_PROMOTIONS}, max_members={MAX_ENSEMBLE_MEMBERS})"
    )
    if decision_state:
        print("       Ensemble built from decision-state policy")
        promoted_in_set = [
            n for n in ensemble_names
            if str(decision_state.get(n, {}).get("decision", "")) == "promote"
        ]
        print(f"       Promotions retained: {len(promoted_in_set)}")
        if promoted_in_set:
            print("       " + ", ".join(promoted_in_set))
    else:
        print("       Ensemble built from static baseline policy")
    print()
    # Use Sharpe^2.5 for balanced concentration (between Sharpe^2 and Sharpe^3)
    # This tilts capital to high-Sharpe strategies without over-concentrating.
    prior_weights = [max(s, 0.15) ** 2.5 for s in sharpe_scores]
    ensemble_cfg = EnsembleConfig(
        use_inverse_vol=False,        # fixed Sharpe-proportional (less turnover)
        max_gross_leverage=2.5,
        max_single_weight=0.14,       # balanced concentration
        dd_scale_start=-0.11,         # start scaling down at -11% DD (was -12%)
        dd_scale_end=-0.21,           # fully flat at -21% DD (was -22%)
        prior_weights=prior_weights,
        # 1a: Correlation hedging
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        # 1d: Vol-regime leverage scaling
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
        # VIX-adaptive prior weights (differential regime tilting)
        vix_prior_scaling=True,
        vix_prior_k=0.02,
        vix_prior_base=20.0,
        vix_prior_lookback=20,
        # Drift regime filter -- DISABLED: threshold 0.58 > SPY avg pos-day
        # fraction (~0.53), causing chronic position scaling that kills CAGR
        drift_filter_enabled=False,
    )

    total_strategies = sum(len(v) for v in registry.values()) + 2  # +benchmark +ensemble
    print(f"[4/4] Running {total_strategies} strategy backtests across "
          f"{len(CRISIS_WINDOWS)} time windows ...")

    # Inject intraday features into strategies that support them
    if intraday_features is not None:
        _inject_features(ensemble_members, intraday_features)
        _inject_features(
            [s for cat in registry.values() for _, s, _ in cat],
            intraday_features,
        )
        print(f"       Intraday features injected into eligible strategies.")
    print()

    # ------------------------------------------------------------------
    # 4. Run backtests for each window
    # ------------------------------------------------------------------
    all_results = {}

    for window_name, (win_start, win_end) in CRISIS_WINDOWS.items():
        print("-" * 80)
        print(f"  WINDOW: {window_name}")
        print("-" * 80)

        # Slice data to window
        mask = (prices.index >= win_start) & (prices.index <= win_end)
        p_win = prices.loc[mask].copy()
        r_win = regime.loc[mask].copy()

        if len(p_win) < 30:
            print(f"  [SKIP] Not enough data ({len(p_win)} days)")
            print()
            continue

        print(f"  Data: {p_win.index[0].date()} -> {p_win.index[-1].date()} "
              f"({len(p_win)} days)")

        # Regime summary for this window
        rc = r_win.value_counts()
        crisis_days = sum(
            rc.get(r, 0)
            for r in [Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS]
        )
        print(f"  Crisis days: {crisis_days} / {len(r_win)} "
              f"({crisis_days / len(r_win) * 100:.1f}%)")

        # Build synthetic sentiment for fundamental strategies
        vix_win = vix.reindex(p_win.index).ffill() if vix is not None else None
        try:
            sent_df = build_synthetic_sentiment(p_win, vix=vix_win)
        except Exception as e:
            print(f"  [WARN] Synthetic sentiment build failed: {e}")
            logger.warning("Synthetic sentiment build failed in %s: %s", window_name, e, exc_info=True)
            sent_df = None

        rows = []

        # Benchmark
        bm = spy_benchmark(p_win, BT_CONFIG)
        rows.append({"Category": "Benchmark", "Strategy": "BuyHold-SPY", **metrics_row(bm)})

        # Each category
        for cat_name, strats in registry.items():
            use_sentiment = "Fundamental" in cat_name
            for strat_name, strat, needs_regime in strats:
                m = run_strategy_backtest(
                    strat_name, strat, p_win, r_win, BT_CONFIG, needs_regime,
                    sentiment_df=sent_df if use_sentiment else None,
                )
                rows.append({"Category": cat_name, "Strategy": strat_name, **metrics_row(m)})

        # Ensemble
        ens_result: dict | None = None
        try:
            ens = EnsembleStrategy(cast(list[Strategy], ensemble_members), ensemble_cfg)
            # Ensemble needs regime for FourStateTactical
            ens_w = ens.backtest_weights(p_win, r_win)
            ens_result = backtest(p_win, ens_w, BT_CONFIG_ENSEMBLE)
            ens_m = ens_result["metrics"]
        except Exception as e:
            print(f"  [WARN] Ensemble: {e}")
            logger.warning("Ensemble backtest failed in %s: %s", window_name, e, exc_info=True)
            ens_m = None
        rows.append({"Category": "Ensemble", "Strategy": "Ensemble-BestOfEach", **metrics_row(ens_m)})

        # Generate HTML tearsheet (optional — requires quantstats)
        if ens_result is not None:
            try:
                from financial_algo.reporting import generate_tearsheet
                _safe = window_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
                ts_path = generate_tearsheet(
                    ens_result,
                    benchmark_prices=p_win["SPY"] if "SPY" in p_win.columns else None,
                    output_path=f"results/tearsheet_{_safe}.html",
                    title=f"Ensemble - {window_name}",
                )
                print(f"  Tearsheet saved: {ts_path}")
            except ImportError:
                pass  # quantstats not installed — skip tearsheet
            except Exception as e:
                print(f"  [WARN] Tearsheet generation failed: {e}")
                logger.warning("Tearsheet generation failed for %s: %s", window_name, e, exc_info=True)

        # Display results table
        df = pd.DataFrame(rows)
        print()
        print(df.to_string(index=False))
        print()

        all_results[window_name] = df

    # ------------------------------------------------------------------
    # Summary: Best strategy per crisis period
    # ------------------------------------------------------------------
    print("=" * 80)
    print("SUMMARY - BEST STRATEGY PER CRISIS WINDOW (by Sharpe)")
    print("=" * 80)
    print()

    for window_name, df in all_results.items():
        # Parse Sharpe back to float for comparison
        df_copy = df.copy()
        df_copy["_sharpe_num"] = pd.to_numeric(
            df_copy["Sharpe"].replace("ERR", np.nan), errors="coerce"
        )
        best = df_copy.loc[df_copy["_sharpe_num"].idxmax()]
        print(f"  {window_name:45s} -> {best['Strategy']:30s} "
              f"(Sharpe={best['Sharpe']}, CAGR={best['CAGR']}, MaxDD={best['MaxDD']})")

    print()
    print("=" * 80)
    print("METHODOLOGY NOTES (Unbiased Backtesting)")
    print("=" * 80)
    print("""
  [OK] Weights shifted +1 day -- no look-ahead bias
  [OK] Transaction costs: 5 bps one-way per trade
  [OK] Leverage borrowing cost: 1.5% annualised
  [OK] Short borrow cost: 0.5% annualised
  [OK] Regime detection uses only trailing data (no future info)
  [OK] Z-scores computed on trailing rolling windows only
  [OK] No survivorship bias -- using ETFs with full history
  [OK] Initial capital: $1,000,000
  [OK] All strategies start from identical data & cost model
    """)

    # Save results to CSV
    out_dir = SCRIPTS_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    for window_name, df in all_results.items():
        safe_name = window_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        df.to_csv(out_dir / f"backtest_{safe_name}.csv", index=False)
    print(f"  Results saved to: {out_dir}")
    print()
    logger.info("Backtest run complete. Results saved to %s", out_dir)


if __name__ == "__main__":
    main()
