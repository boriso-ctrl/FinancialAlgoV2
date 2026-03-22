"""Quick backtest of P8-Alpha158Ranker strategy."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.signal_combo import Alpha158Ranker, Alpha158RankerConfig

# Full ticker universe
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

DATA_START = "2010-01-01"
DATA_END = "2025-12-31"

print(f"[1/4] Loading prices for {len(TICKERS)} tickers ...")
prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
print(f"       Loaded: {prices.shape[0]} days x {prices.shape[1]} tickers")
print(f"       Range: {prices.index[0].date()} -> {prices.index[-1].date()}")
print()

print("[2/4] Detecting regime ...")
regime = detect_regime(prices)
print()

config = BacktestConfig(
    tx_cost_bps=5.0,
    leverage_cost_annual=0.015,
    short_cost_annual=0.005,
)

print("[3/4] Running P8-Alpha158Ranker ...")
strat = Alpha158Ranker()
weights = strat.backtest_weights(prices, regime)
result = backtest(prices, weights, config)
m = result["metrics"]

print()
print("=" * 70)
print("P8-Alpha158Ranker -- Full Period Results")
print("=" * 70)
print(f"  CAGR:      {m['cagr']:.2%}")
print(f"  Sharpe:    {m['sharpe']:.2f}")
print(f"  Sortino:   {m['sortino']:.2f}")
print(f"  Max DD:    {m['max_drawdown']:.2%}")
print(f"  Calmar:    {m['calmar']:.2f}")
print(f"  Ann Vol:   {m['annual_vol']:.2%}")
print(f"  Win Rate:  {m['win_rate']:.2%}")
print(f"  Tot Ret:   {m['total_return']:.2%}")
print()

# Feature importance analysis
print("[4/4] Feature importance analysis ...")
from financial_algo.technical.alpha158 import compute_alpha158_from_close, pivot_feature

features = compute_alpha158_from_close(prices)
if not features.empty:
    # Check IC (information coefficient) for each feature
    # IC = rank correlation of feature z-score with next-day returns
    fwd_ret_1d = prices.pct_change().shift(-1).fillna(0.0)
    
    print()
    print("Feature Information Coefficients (rank corr with 1d fwd return):")
    print("-" * 55)
    
    ics = {}
    for feat_name in sorted(features.columns):
        try:
            pivoted = pivot_feature(features, feat_name)
            pivoted = pivoted.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
            
            # Cross-sectional rank correlation per day, then average
            daily_ic = []
            # Sample every 20th day for speed
            for i in range(62, len(prices) - 1, 20):
                row_feat = pivoted.iloc[i]
                row_ret = fwd_ret_1d.iloc[i]
                valid = pd.notna(row_feat) & pd.notna(row_ret) & (row_feat != 0)
                if valid.sum() >= 5:
                    ic = row_feat[valid].corr(row_ret[valid], method="spearman")
                    if pd.notna(ic):
                        daily_ic.append(ic)
            
            if daily_ic:
                mean_ic = np.mean(daily_ic)
                ics[feat_name] = mean_ic
                marker = " <<<" if abs(mean_ic) > 0.03 else ""
                print(f"  {feat_name:15s}: IC = {mean_ic:+.4f}{marker}")
        except Exception:
            pass
    
    print()
    print("Top 5 features by |IC|:")
    sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, ic in sorted_ics[:5]:
        print(f"  {name:15s}: {ic:+.4f}")

# Crisis sub-periods
print()
print("=" * 70)
print("Sub-Period Performance")
print("=" * 70)
periods = {
    "EU Crisis 2011":      ("2011-01-01", "2012-01-01"),
    "Oil Crash 2014-16":   ("2014-06-01", "2016-06-01"),
    "Volmageddon 2018":    ("2018-01-01", "2019-01-01"),
    "COVID 2020":          ("2020-01-01", "2021-01-01"),
    "Inflation 2022":      ("2022-01-01", "2023-01-01"),
    "Recovery 2023-25":    ("2023-01-01", "2025-12-31"),
}

for label, (start, end) in periods.items():
    mask = (prices.index >= start) & (prices.index <= end)
    if mask.sum() < 60:
        continue
    sub_prices = prices.loc[mask]
    sub_weights = weights.loc[mask]
    try:
        sub_result = backtest(sub_prices, sub_weights, config)
        sm = sub_result["metrics"]
        print(f"  {label:22s}: CAGR={sm['cagr']:+.2%}  Sharpe={sm['sharpe']:.2f}  "
              f"MaxDD={sm['max_drawdown']:.2%}")
    except Exception as e:
        print(f"  {label:22s}: Error - {e}")

# Correlation with SPY
print()
portfolio_ret = result["daily_returns"]
spy_ret = prices["SPY"].pct_change().fillna(0.0)
aligned = pd.DataFrame({"portfolio": portfolio_ret, "spy": spy_ret}).dropna()
corr = aligned["portfolio"].corr(aligned["spy"])
print(f"Correlation with SPY: {corr:.3f}")
