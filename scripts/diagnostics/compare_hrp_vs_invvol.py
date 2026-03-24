"""Compare HRP vs Inverse-Vol weighting for the ensemble.

Runs the full ensemble with both weighting methods on the full period
and prints a side-by-side comparison table.

Usage:
    .venv\\Scripts\\python.exe scripts/diagnostics/compare_hrp_vs_invvol.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable when running as a script
_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

# Check skfolio availability up front
try:
    import skfolio  # noqa: F401
    HAS_SKFOLIO = True
except ImportError:
    HAS_SKFOLIO = False

if not HAS_SKFOLIO:
    print("ERROR: skfolio is not installed.")
    print("HRP weighting requires skfolio.  Install with:  uv pip install skfolio")
    sys.exit(1)

from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy

# --- Strategy imports (same ensemble members as run_crisis_backtest.py) ---
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
from financial_algo.strategies.crypto_crisis import (
    CryptoGoldDivergence,
    CryptoRecoverySurge,
)
from financial_algo.strategies.factor import LowVolFactor, MultiFactorComposite, ValueFactor
from financial_algo.strategies.mean_reversion import RSIMeanReversion  # noqa: F401
from financial_algo.strategies.ml_strategies import AdaptiveThreshold
from financial_algo.strategies.momentum import TimeSeriesMomentum  # noqa: F401
from financial_algo.strategies.multi_freq import MonthlyMacroRegime, WeeklyMomentumRotation
from financial_algo.strategies.quality_trend import (
    MomentumCrashFilter,
    MultiAssetTrend,
    QualityTrend,
)
from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.tail_risk import TailHedgeOverlay, TailRiskParity
from financial_algo.strategies.volatility_strats import (
    VolOfVolRegime,
    VolRiskPremium,
    VolSpreadHarvest,
    VolSpikeRecovery,
    VolTermStructure,
)
from financial_algo.fundamental.strategies import (
    FearGreedContrarian,
    SentimentCrisisAlpha,
    SentimentDivergence,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================================
# Configuration (mirrored from run_crisis_backtest.py)
# =========================================================================

DATA_START = "2009-01-01"
DATA_END = "2025-12-31"

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "SLV", "TLT", "IEF", "SHY", "UUP",
    "XLE", "USO", "XOP",
    "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "XBI", "XLC", "XLRE",
    "DBC", "DBA",
    "HYG", "LQD", "TIP", "AGG", "EMB",
    "FXI", "VGK", "EWJ", "INDA",
    "VNQ",
    "BTC-USD", "ETH-USD",
]))

VIX_TICKER = "^VIX"

BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=None,
    strategy_dd_scale_start=-0.15,
    strategy_dd_scale_end=-0.25,
)


def _build_ensemble_members():
    """Return the same 26 ensemble members and Sharpe scores as run_crisis_backtest."""
    members = [
        FeatureComboSignal(),
        CrashHedgeQQQ(),
        CryptoRecoverySurge(),
        MonthlyMacroRegime(),
        VolRiskPremium(),
        VolOfVolRegime(),
        TailRiskParity(),
        VolCarry(),
        CryptoGoldDivergence(),
        MultiAssetTrend(),
        FearGreedContrarian(),
        VolTermStructure(),
        VolSpreadHarvest(),
        LowVolFactor(),
        MultiFactorComposite(),
        VolSpikeRecovery(),
        QualityTrend(),
        MomentumCrashFilter(),
        ValueFactor(),
        SentimentCrisisAlpha(),
        SeasonalStrategy(),
        WeeklyMomentumRotation(),
        SentimentDivergence(),
        TailHedgeOverlay(),
        AdaptiveThreshold(),
        MultiAssetCTATrend(),
    ]
    sharpe_scores = [
        0.97, 0.94, 0.93, 0.93, 0.92, 0.92, 0.90, 0.89, 0.89, 0.88,
        0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84, 0.83, 0.81, 0.80,
        0.79, 0.78, 0.78,
        0.50,
        0.75,
        1.14,
    ]
    prior_weights = [s ** 2 for s in sharpe_scores]
    return members, prior_weights


def _base_ensemble_config(prior_weights: list[float]) -> dict:
    """Return the shared EnsembleConfig kwargs (everything except weighting_method)."""
    return dict(
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=prior_weights,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )


def _run_ensemble(
    prices: pd.DataFrame,
    regime: pd.Series,
    members: list,
    cfg: EnsembleConfig,
    label: str,
) -> dict | None:
    """Build ensemble, generate weights, run backtest, return metrics."""
    print(f"  Running ensemble with weighting_method={cfg.weighting_method!r} ...")
    t0 = time.perf_counter()
    try:
        ens = EnsembleStrategy(members, cfg)
        weights = ens.backtest_weights(prices, regime)
        result = backtest(prices, weights, BT_CONFIG)
        elapsed = time.perf_counter() - t0
        print(f"  {label} completed in {elapsed:.1f}s")
        return result["metrics"]
    except Exception as e:
        elapsed = time.perf_counter() - t0
        print(f"  {label} FAILED after {elapsed:.1f}s: {e}")
        return None


def _fmt(val, fmt_str: str, is_pct: bool = False) -> str:
    """Format a metric value as string."""
    if val is None:
        return "ERR"
    if is_pct:
        return f"{val:.2%}"
    return f"{fmt_str}".format(val)


def main() -> None:
    print("=" * 78)
    print("HRP vs INVERSE-VOL ENSEMBLE WEIGHTING COMPARISON")
    print("=" * 78)
    print()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"[1/3] Loading price data for {len(TICKERS)} tickers ...")
    prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
    print(f"       {prices.shape[0]} trading days x {prices.shape[1]} tickers")
    print(f"       {prices.index[0].date()} -> {prices.index[-1].date()}")
    print()

    # ------------------------------------------------------------------
    # 2. Regime detection
    # ------------------------------------------------------------------
    print("[2/3] Running regime detection ...")
    try:
        vix_df = load_prices([VIX_TICKER], start=DATA_START, end=DATA_END)
        vix = vix_df[VIX_TICKER]
    except Exception:
        vix = None
    regime = detect_regime(prices, vix=vix)
    print()

    # ------------------------------------------------------------------
    # 3. Run both weighting methods on Full Period (2010-2025)
    # ------------------------------------------------------------------
    win_start, win_end = "2010-01-01", "2025-12-31"
    mask = (prices.index >= win_start) & (prices.index <= win_end)
    p_win = prices.loc[mask].copy()
    r_win = regime.loc[mask].copy()

    print(f"[3/3] Backtesting on Full Period: "
          f"{p_win.index[0].date()} -> {p_win.index[-1].date()} "
          f"({len(p_win)} days)")
    print()

    members, prior_weights = _build_ensemble_members()
    base_kwargs = _base_ensemble_config(prior_weights)

    # --- Inverse-vol (current production) ---
    cfg_iv = EnsembleConfig(
        weighting_method="inverse_vol",
        use_inverse_vol=False,  # production: fixed Sharpe-proportional
        **base_kwargs,
    )
    m_iv = _run_ensemble(p_win, r_win, members, cfg_iv, "inverse_vol")

    # --- HRP ---
    # Rebuild members to avoid any shared mutable state
    members_hrp, _ = _build_ensemble_members()
    cfg_hrp = EnsembleConfig(
        weighting_method="hrp",
        use_inverse_vol=False,
        skfolio_refit_every=21,
        skfolio_risk_measure="variance",
        **base_kwargs,
    )
    m_hrp = _run_ensemble(p_win, r_win, members_hrp, cfg_hrp, "hrp")

    # ------------------------------------------------------------------
    # 4. Print comparison table
    # ------------------------------------------------------------------
    print()
    print("=" * 78)
    print("COMPARISON TABLE -- Full Period (2010-2025)")
    print("=" * 78)
    print()

    metrics_keys = [
        ("CAGR", "cagr", True),
        ("Sharpe", "sharpe", False),
        ("Sortino", "sortino", False),
        ("Max DD", "max_drawdown", True),
        ("Calmar", "calmar", False),
    ]

    # Header
    header = f"{'Weighting Method':<20s}"
    for label, _, _ in metrics_keys:
        header += f" | {label:>10s}"
    print(header)
    print("-" * len(header))

    def _row(label: str, m: dict | None) -> str:
        row = f"{label:<20s}"
        for _, key, is_pct in metrics_keys:
            if m is None:
                row += f" | {'ERR':>10s}"
            elif is_pct:
                row += f" | {m[key]:>10.2%}"
            else:
                row += f" | {m[key]:>10.2f}"
        return row

    print(_row("inverse_vol", m_iv))
    print(_row("hrp", m_hrp))

    # Delta row
    if m_iv is not None and m_hrp is not None:
        delta_row = f"{'delta (hrp - iv)':<20s}"
        for _, key, is_pct in metrics_keys:
            d = m_hrp[key] - m_iv[key]
            sign = "+" if d >= 0 else ""
            if is_pct:
                delta_row += f" | {sign}{d:>9.2%}"
            else:
                delta_row += f" | {sign}{d:>9.2f}"
        print(delta_row)

    print()

    # ------------------------------------------------------------------
    # 5. Additional metrics
    # ------------------------------------------------------------------
    if m_iv is not None and m_hrp is not None:
        print("Additional metrics:")
        for label, m in [("inverse_vol", m_iv), ("hrp", m_hrp)]:
            print(f"  {label}: AnnVol={m['annual_vol']:.2%}  "
                  f"WinRate={m['win_rate']:.2%}  "
                  f"TotRet={m['total_return']:.2%}")
        print()

    print("METHODOLOGY:")
    print("  - Same 26 ensemble members, same prior weights, same cost model")
    print("  - inverse_vol uses fixed Sharpe-proportional (use_inverse_vol=False)")
    print("  - hrp uses skfolio HierarchicalRiskParity, refit every 21 days")
    print("  - Both use correlation hedging and vol-regime leverage scaling")
    print("  - Backtest: 5bps tx cost, 1.5% leverage cost, 0.5% short cost")
    print()


if __name__ == "__main__":
    main()
