"""A/B test DL2-LSTMRegimeDetector into ensemble v10."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy

# v9 members (26)
from financial_algo.strategies.crash_hedge import CrashHedgeQQQ, VolCarry
from financial_algo.strategies.volatility_strats import (
    VolRiskPremium, VolOfVolRegime, VolTermStructure,
    VolSpreadHarvest, VolSpikeRecovery,
)
from financial_algo.strategies.tail_risk import TailRiskParity, TailHedgeOverlay
from financial_algo.strategies.crypto_crisis import CryptoRecoverySurge, CryptoGoldDivergence
from financial_algo.strategies.signal_combo import FeatureComboSignal
from financial_algo.strategies.ml_strategies import AdaptiveThreshold
from financial_algo.strategies.quality_trend import QualityTrend, MultiAssetTrend, MomentumCrashFilter
from financial_algo.strategies.factor import LowVolFactor, MultiFactorComposite, ValueFactor
from financial_algo.strategies.seasonal import SeasonalStrategy
from financial_algo.strategies.multi_freq import WeeklyMomentumRotation, MonthlyMacroRegime
from financial_algo.strategies.regime_hardening import MultiAssetCTATrend
from financial_algo.fundamental.strategies import (
    SentimentCrisisAlpha, FearGreedContrarian, SentimentDivergence,
)
from financial_algo.strategies.dl_strategies import LSTMRegimeDetector, TemporalCNNAlpha

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM",
    "GLD", "SLV", "TLT", "IEF", "SHY", "UUP",
    "XLE", "USO", "XOP", "ITA", "LMT", "RTX",
    "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "XBI", "XLC", "XLRE", "DBC", "DBA",
    "HYG", "LQD", "TIP", "AGG", "EMB",
    "FXI", "VGK", "EWJ", "INDA", "VNQ",
    "BTC-USD", "ETH-USD",
]))

BT_ENS = BacktestConfig(
    tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
    initial_capital=1_000_000.0, vol_target=0.20,
    max_drawdown_trigger=None,
    strategy_dd_scale_start=-0.15, strategy_dd_scale_end=-0.25,
)


def build_v9():
    members = [
        FeatureComboSignal(), CrashHedgeQQQ(), CryptoRecoverySurge(),
        VolRiskPremium(), VolOfVolRegime(), TailRiskParity(),
        VolCarry(), CryptoGoldDivergence(), MultiAssetTrend(),
        FearGreedContrarian(), VolTermStructure(), VolSpreadHarvest(),
        LowVolFactor(), MultiFactorComposite(), VolSpikeRecovery(),
        QualityTrend(), MomentumCrashFilter(), ValueFactor(),
        SentimentCrisisAlpha(), SeasonalStrategy(), SentimentDivergence(),
        TailHedgeOverlay(), AdaptiveThreshold(),
        WeeklyMomentumRotation(), MonthlyMacroRegime(), MultiAssetCTATrend(),
    ]
    sharpes = [
        0.97, 0.94, 0.93, 0.92, 0.92, 0.90, 0.89, 0.89,
        0.88, 0.87, 0.86, 0.85, 0.84, 0.84, 0.84, 0.84,
        0.83, 0.81, 0.80, 0.79, 0.78, 0.50, 0.75,
        0.78, 0.93, 1.14,
    ]
    return members, sharpes


def make_ensemble(members, sharpes):
    prior = [s ** 2 for s in sharpes]
    cfg = EnsembleConfig(
        use_inverse_vol=False,
        max_gross_leverage=2.5,
        max_single_weight=0.20,
        dd_scale_start=-0.12,
        dd_scale_end=-0.22,
        prior_weights=prior,
        correlation_hedge_enabled=True,
        correlation_hedge_threshold=0.65,
        correlation_hedge_max=0.25,
        vol_regime_scaling=True,
        vol_elevated_threshold=0.20,
        vol_crisis_threshold=0.30,
        leverage_elevated=1.8,
        leverage_crisis=1.2,
    )
    return EnsembleStrategy(members, cfg)


def evaluate(ens, prices, reg, label):
    w = ens.backtest_weights(prices, reg)
    res = backtest(prices, w, BT_ENS)
    m = res["metrics"]
    sharpe = m.get("sharpe", 0.0)
    cagr = m.get("cagr", 0.0)
    maxdd = m.get("max_drawdown", 0.0)

    # Weak year analysis
    weak_years = {"2015": ("2015-01-01", "2015-12-31"),
                  "2018": ("2018-01-01", "2018-12-31"),
                  "2022": ("2022-01-01", "2022-12-31")}
    year_sharpes = {}
    for yr, (s, e) in weak_years.items():
        yr_mask = (res["returns"].index >= s) & (res["returns"].index <= e)
        yr_ret = res["returns"].loc[yr_mask]
        if len(yr_ret) > 50:
            yr_std = yr_ret.std() * np.sqrt(252)
            yr_mean = yr_ret.mean() * 252
            yr_sharpe = yr_mean / yr_std if yr_std > 0 else 0.0
            year_sharpes[yr] = yr_sharpe

    print(f"  {label:<35} Sharpe={sharpe:.2f}  CAGR={cagr:.1%}  MaxDD={maxdd:.1%}  "
          f"2015={year_sharpes.get('2015', 0):.2f}  2018={year_sharpes.get('2018', 0):.2f}  "
          f"2022={year_sharpes.get('2022', 0):.2f}")
    return sharpe


def main():
    print("=" * 90)
    print("A/B TEST: DL STRATEGIES INTO ENSEMBLE v10")
    print("=" * 90)
    print()

    print("Loading data ...")
    prices_raw = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
    vix = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")["^VIX"]
    regime = detect_regime(prices_raw, vix=vix)
    mask = (prices_raw.index >= "2010-01-01") & (prices_raw.index <= "2025-12-31")
    prices = prices_raw.loc[mask]
    reg = regime.loc[mask]
    print(f"  {prices.shape[0]} days x {prices.shape[1]} tickers")
    print()

    v9_members, v9_sharpes = build_v9()

    # v9 baseline
    print("Running variants ...")
    v9_ens = make_ensemble(v9_members, v9_sharpes)
    v9_s = evaluate(v9_ens, prices, reg, f"v9-Baseline ({len(v9_members)})")

    # v10a: +DL2
    dl2 = LSTMRegimeDetector()
    v10a_m = v9_members + [dl2]
    v10a_s = v9_sharpes + [0.67]
    ens_a = make_ensemble(v10a_m, v10a_s)
    sa = evaluate(ens_a, prices, reg, f"v10a +DL2 ({len(v10a_m)})")

    # v10b: +DL1
    dl1 = TemporalCNNAlpha()
    v10b_m = v9_members + [dl1]
    v10b_s = v9_sharpes + [0.46]
    ens_b = make_ensemble(v10b_m, v10b_s)
    sb = evaluate(ens_b, prices, reg, f"v10b +DL1 ({len(v10b_m)})")

    # v10c: +DL1 +DL2
    v10c_m = v9_members + [dl2, dl1]
    v10c_s = v9_sharpes + [0.67, 0.46]
    ens_c = make_ensemble(v10c_m, v10c_s)
    sc = evaluate(ens_c, prices, reg, f"v10c +DL1+DL2 ({len(v10c_m)})")

    print()
    print("Deltas vs v9:")
    print(f"  v10a +DL2:      dSharpe={sa - v9_s:+.3f}")
    print(f"  v10b +DL1:      dSharpe={sb - v9_s:+.3f}")
    print(f"  v10c +DL1+DL2:  dSharpe={sc - v9_s:+.3f}")
    print()

    best = max([(sa - v9_s, "v10a +DL2"), (sb - v9_s, "v10b +DL1"), (sc - v9_s, "v10c +DL1+DL2")])
    if best[0] > 0.005:
        print(f"WINNER: {best[1]} (dSharpe={best[0]:+.3f})")
    elif best[0] > -0.005:
        print(f"NEUTRAL: {best[1]} (dSharpe={best[0]:+.3f}) -- no meaningful improvement")
    else:
        print(f"NO WINNER: Best is {best[1]} (dSharpe={best[0]:+.3f}) -- keep v9")


if __name__ == "__main__":
    main()
