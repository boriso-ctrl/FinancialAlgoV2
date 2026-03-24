"""Focused Middle East Crisis Backtest — runs only ME-specific windows."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import Regime, RegimeConfig, detect_regime

from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, FourStateTactical, VolCarry
from financial_algo.strategies.oil_crisis import EnergyPairs, OilMeanReversion, OilMomentumSurge, OilShockHedge
from financial_algo.strategies.war_crisis import ArmsRaceMomentum, DefenseRotation, PostWarRecovery, SafeHavenFlight
from financial_algo.strategies.pairs import MultiPairPortfolio
from financial_algo.strategies.crypto_crisis import CryptoFlightToQuality, CryptoGoldDivergence, CryptoRecoverySurge
from financial_algo.strategies.crisis_spike import CommodityShockRider, DefenseSpikeBreakout, GoldFearRally, MultiAssetCrisisLong
from financial_algo.strategies.momentum import TimeSeriesMomentum, CrossSectionalMomentum, DualMomentum
from financial_algo.strategies.mean_reversion import SectorMeanReversion, RSIMeanReversion
from financial_algo.strategies.fixed_income import YieldCurveTrade, CreditSpreadMeanRev
from financial_algo.strategies.volatility_strats import VolRiskPremium
from financial_algo.strategies.macro import DollarCarry, GoldDollarInverse
from financial_algo.strategies.seasonal import SeasonalStrategy, TurnOfMonth
from financial_algo.strategies.factor import LowVolFactor, MultiFactorComposite
from financial_algo.strategies.ensemble import EnsembleStrategy, EnsembleConfig
from financial_algo.fundamental import build_synthetic_sentiment
from financial_algo.fundamental.strategies import (
    FearGreedContrarian, SentimentCrisisAlpha, SentimentDivergence, SentimentEnhancedRegime,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_START = "2009-01-01"
DATA_END = "2025-12-31"

# Only Middle East windows
ME_WINDOWS = {
    "ME: Red Sea / Houthi (Jan-Mar 2024)":  ("2024-01-01", "2024-04-01"),
    "ME: Iran Tensions (Sep-Oct 2024)":     ("2024-09-01", "2024-11-01"),
    "ME: Oil Spike Jun 2025":               ("2025-05-01", "2025-07-31"),
    "ME: Full Conflict (Oct23-Dec24)":      ("2023-10-01", "2024-12-31"),
}

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "TLT", "IEF", "UUP",
    "XLE", "USO", "XOP",
    "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "HYG", "LQD",
    "BTC-USD",
]))
VIX_TICKER = "^VIX"

BT_CONFIG = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
    initial_capital=1_000_000.0,
    vol_target=0.20,
    max_drawdown_trigger=-0.25,
    drawdown_recovery_rate=0.10,
)


def run_strategy_backtest(name, strategy, prices, regime, config, needs_regime=False, sentiment_df=None):
    try:
        if sentiment_df is not None:
            weights = strategy.generate_weights(prices, regime, sentiment_df)
            weights = weights.shift(1).fillna(0.0)
        elif needs_regime:
            weights = strategy.backtest_weights(prices, regime)
        else:
            weights = strategy.backtest_weights(prices)
        result = backtest(prices, weights, config)
        return result["metrics"]
    except Exception as e:
        print(f"  [WARN] {name}: {e}")
        return None


def metrics_row(metrics):
    if metrics is None:
        return {"CAGR": "ERR", "Sharpe": "ERR", "Sortino": "ERR", "MaxDD": "ERR", "Calmar": "ERR", "AnnVol": "ERR", "WinRate": "ERR", "TotRet": "ERR"}
    return {
        "CAGR": f"{metrics['cagr']:.2%}", "Sharpe": f"{metrics['sharpe']:.2f}",
        "Sortino": f"{metrics['sortino']:.2f}", "MaxDD": f"{metrics['max_drawdown']:.2%}",
        "Calmar": f"{metrics['calmar']:.2f}", "AnnVol": f"{metrics['annual_vol']:.2%}",
        "WinRate": f"{metrics['win_rate']:.2%}", "TotRet": f"{metrics['total_return']:.2%}",
    }


def build_strategy_registry():
    return {
        "Cat B: Oil Crisis": [
            ("B1-OilMomentumSurge", OilMomentumSurge(), True),
            ("B2-OilShockHedge", OilShockHedge(), True),
            ("B3-OilMeanReversion", OilMeanReversion(), True),
            ("B4-EnergyPairs", EnergyPairs(), True),
        ],
        "Cat C: War/Geopolitical": [
            ("C1-DefenseRotation", DefenseRotation(), True),
            ("C2-SafeHavenFlight", SafeHavenFlight(), True),
            ("C3-PostWarRecovery", PostWarRecovery(), True),
            ("C4-ArmsRaceMomentum", ArmsRaceMomentum(), False),
        ],
        "Cat D: Crash-Hedge (General)": [
            ("D1-FourStateTactical", FourStateTactical(), True),
            ("D2-CrashHedgeQQQ", CrashHedgeQQQ(), False),
            ("D3-VolCarry", VolCarry(), False),
        ],
        "Cat E: Pairs Arbitrage": [
            ("E1-MultiPairPortfolio", MultiPairPortfolio(), False),
        ],
        "Cat F: Crypto Crisis": [
            ("F1-CryptoFlightToQuality", CryptoFlightToQuality(), True),
            ("F2-CryptoRecoverySurge", CryptoRecoverySurge(), True),
            ("F3-CryptoGoldDivergence", CryptoGoldDivergence(), True),
        ],
        "Cat G: Fundamental/Sentiment": [
            ("G1-SentimentCrisisAlpha", SentimentCrisisAlpha(), True),
            ("G2-FearGreedContrarian", FearGreedContrarian(), True),
            ("G3-SentimentDivergence", SentimentDivergence(), True),
            ("G4-SentimentEnhancedRegime", SentimentEnhancedRegime(), True),
        ],
        "Cat H: Crisis Spike (Upward)": [
            ("H1-CommodityShockRider", CommodityShockRider(), True),
            ("H2-GoldFearRally", GoldFearRally(), False),
            ("H3-DefenseSpikeBreakout", DefenseSpikeBreakout(), True),
            ("H4-MultiAssetCrisisLong", MultiAssetCrisisLong(), True),
        ],
        "Cat I: Momentum": [
            ("I1-TimeSeriesMomentum", TimeSeriesMomentum(), False),
            ("I2-CrossSectionalMomentum", CrossSectionalMomentum(), False),
            ("I3-DualMomentum", DualMomentum(), False),
        ],
        "Cat J: Mean Reversion": [
            ("J1-SectorMeanReversion", SectorMeanReversion(), False),
            ("J3-RSIMeanReversion", RSIMeanReversion(), False),
        ],
        "Cat K: Factor": [
            ("K1-LowVolFactor", LowVolFactor(), False),
            ("K2-MultiFactorComposite", MultiFactorComposite(), False),
        ],
        "Cat L: Volatility": [
            ("L1-VolRiskPremium", VolRiskPremium(), False),
        ],
        "Cat M: Macro": [
            ("M1-DollarCarry", DollarCarry(), False),
            ("M2-GoldDollarInverse", GoldDollarInverse(), False),
        ],
        "Cat N: Seasonal": [
            ("N1-SeasonalStrategy", SeasonalStrategy(), False),
            ("N2-TurnOfMonth", TurnOfMonth(), False),
        ],
    }


def spy_benchmark(prices, config):
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w["SPY"] = 1.0
    try:
        result = backtest(prices, w.shift(1).fillna(0), config)
        return result["metrics"]
    except Exception as e:
        print(f"  [WARN] Benchmark: {e}")
        return None


def main():
    print("=" * 80)
    print("MIDDLE EAST CRISIS — FOCUSED STRATEGY BACKTEST")
    print("=" * 80)
    print()

    print(f"Downloading price data for {len(TICKERS)} tickers ({DATA_START} -> {DATA_END}) ...")
    prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
    print(f"  Loaded: {prices.shape[0]} days x {prices.shape[1]} tickers")
    print()

    print("Downloading VIX ...")
    try:
        vix_df = load_prices([VIX_TICKER], start=DATA_START, end=DATA_END)
        vix = vix_df[VIX_TICKER]
    except Exception:
        vix = None
    print()

    print("Running regime detection ...")
    regime = detect_regime(prices, vix=vix)
    print()

    registry = build_strategy_registry()
    all_results = {}

    for window_name, (win_start, win_end) in ME_WINDOWS.items():
        print("=" * 80)
        print(f"  WINDOW: {window_name}")
        print("=" * 80)

        mask = (prices.index >= win_start) & (prices.index <= win_end)
        p_win = prices.loc[mask].copy()
        r_win = regime.loc[mask].copy()

        if len(p_win) < 30:
            print(f"  [SKIP] Not enough data ({len(p_win)} days)")
            continue

        print(f"  Data: {p_win.index[0].date()} -> {p_win.index[-1].date()} ({len(p_win)} days)")
        rc = r_win.value_counts()
        crisis_days = sum(rc.get(r, 0) for r in [Regime.OIL_CRISIS, Regime.WAR_CRISIS, Regime.GENERAL_CRISIS])
        print(f"  Crisis days: {crisis_days}/{len(r_win)} ({crisis_days/len(r_win)*100:.1f}%)")

        # Oil/Gold performance for context
        for asset in ["USO", "XLE", "GLD", "ITA"]:
            if asset in p_win.columns:
                ret = (p_win[asset].iloc[-1] / p_win[asset].iloc[0] - 1) * 100
                print(f"  {asset} return: {ret:+.1f}%")

        vix_win = vix.reindex(p_win.index).ffill() if vix is not None else None
        try:
            sent_df = build_synthetic_sentiment(p_win, vix=vix_win)
        except Exception:
            sent_df = None

        rows = []
        bm = spy_benchmark(p_win, BT_CONFIG)
        rows.append({"Category": "Benchmark", "Strategy": "BuyHold-SPY", **metrics_row(bm)})

        for cat_name, strats in registry.items():
            use_sentiment = "Fundamental" in cat_name
            for strat_name, strat, needs_regime in strats:
                m = run_strategy_backtest(
                    strat_name, strat, p_win, r_win, BT_CONFIG, needs_regime,
                    sentiment_df=sent_df if use_sentiment else None,
                )
                rows.append({"Category": cat_name, "Strategy": strat_name, **metrics_row(m)})

        # Ensemble
        ensemble_members = [
            CrashHedgeQQQ(), VolCarry(), DualMomentum(), TimeSeriesMomentum(),
            LowVolFactor(), SeasonalStrategy(), TurnOfMonth(), YieldCurveTrade(),
            VolRiskPremium(), DollarCarry(), GoldDollarInverse(),
        ]
        try:
            ens = EnsembleStrategy(ensemble_members, EnsembleConfig(max_gross_leverage=5.0))
            ens_w = ens.backtest_weights(p_win, r_win)
            ens_result = backtest(p_win, ens_w, BT_CONFIG)
            ens_m = ens_result["metrics"]
        except Exception as e:
            print(f"  [WARN] Ensemble: {e}")
            ens_m = None
        rows.append({"Category": "Ensemble", "Strategy": "Ensemble-BestOfEach", **metrics_row(ens_m)})

        df = pd.DataFrame(rows)
        print()
        print(df.to_string(index=False))
        print()
        all_results[window_name] = df

    # Summary
    print("=" * 80)
    print("SUMMARY — BEST STRATEGY PER MIDDLE EAST WINDOW (by Sharpe)")
    print("=" * 80)
    for window_name, df in all_results.items():
        df_c = df.copy()
        df_c["_s"] = pd.to_numeric(df_c["Sharpe"].replace("ERR", np.nan), errors="coerce")
        best = df_c.loc[df_c["_s"].idxmax()]
        print(f"  {window_name:45s} -> {best['Strategy']:30s} (Sharpe={best['Sharpe']}, CAGR={best['CAGR']}, MaxDD={best['MaxDD']})")

    # Top 5 per window
    print()
    print("=" * 80)
    print("TOP 5 STRATEGIES PER MIDDLE EAST WINDOW")
    print("=" * 80)
    for window_name, df in all_results.items():
        df_c = df.copy()
        df_c["_s"] = pd.to_numeric(df_c["Sharpe"].replace("ERR", np.nan), errors="coerce")
        top5 = df_c.nlargest(5, "_s")
        print(f"\n  {window_name}:")
        for _, row in top5.iterrows():
            print(f"    {row['Strategy']:30s}  Sharpe={row['Sharpe']:>6s}  CAGR={row['CAGR']:>8s}  MaxDD={row['MaxDD']:>8s}")


if __name__ == "__main__":
    main()
