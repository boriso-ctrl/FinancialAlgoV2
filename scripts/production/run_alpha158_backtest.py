"""Standalone P8-Alpha158Ranker backtest -- writes results to results/alpha158_results.txt"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.signal_combo import Alpha158Ranker

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

outfile = os.path.join(os.path.dirname(__file__), "..", "results", "alpha158_results.txt")
lines = []

def log(msg):
    print(msg, flush=True)
    lines.append(msg)

log("Loading prices ...")
prices = load_prices(TICKERS, start="2010-01-01", end="2025-12-31")
log(f"Loaded: {prices.shape[0]} days x {prices.shape[1]} tickers")

regime = detect_regime(prices)

cfg = BacktestConfig(tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005)
strat = Alpha158Ranker()

log("Running backtest ...")
weights = strat.backtest_weights(prices, regime)
result = backtest(prices, weights, cfg)
m = result["metrics"]

log("")
log("=" * 60)
log("P8-Alpha158Ranker -- Full Period Results")
log("=" * 60)
log(f"  CAGR:      {m['cagr']:.2%}")
log(f"  Sharpe:    {m['sharpe']:.2f}")
log(f"  Sortino:   {m['sortino']:.2f}")
log(f"  Max DD:    {m['max_drawdown']:.2%}")
log(f"  Calmar:    {m['calmar']:.2f}")
log(f"  Ann Vol:   {m['annual_vol']:.2%}")
log(f"  Win Rate:  {m['win_rate']:.2%}")
log(f"  Tot Ret:   {m['total_return']:.2%}")

# SPY correlation
spy_ret = prices["SPY"].pct_change().fillna(0.0)
aligned = pd.DataFrame({"p": result["daily_returns"], "s": spy_ret}).dropna()
corr_spy = aligned["p"].corr(aligned["s"])
log(f"  Corr SPY:  {corr_spy:.3f}")

# Feature importance (IC analysis)
log("")
log("Feature Importance (Information Coefficient):")
log("-" * 50)
from financial_algo.technical.alpha158 import compute_alpha158_from_close, pivot_feature

features = compute_alpha158_from_close(prices)
fwd_ret_1d = prices.pct_change().shift(-1).fillna(0.0)

ics = {}
for feat_name in sorted(features.columns):
    try:
        pivoted = pivot_feature(features, feat_name)
        pivoted = pivoted.reindex(index=prices.index, columns=prices.columns).fillna(0.0)
        daily_ic = []
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
    except Exception:
        pass

sorted_ics = sorted(ics.items(), key=lambda x: abs(x[1]), reverse=True)
for name, ic in sorted_ics:
    marker = " <<<" if abs(ic) > 0.03 else ""
    log(f"  {name:15s}: IC = {ic:+.4f}{marker}")

log("")
log("Top 5 by |IC|:")
for name, ic in sorted_ics[:5]:
    log(f"  {name:15s}: {ic:+.4f}")

# Sub-period performance
log("")
log("=" * 60)
log("Sub-Period Performance")
log("=" * 60)
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
        sub_result = backtest(sub_prices, sub_weights, cfg)
        sm = sub_result["metrics"]
        log(f"  {label:22s}: CAGR={sm['cagr']:+.2%}  Sharpe={sm['sharpe']:.2f}  MaxDD={sm['max_drawdown']:.2%}")
    except Exception as e:
        log(f"  {label:22s}: Error - {e}")

with open(outfile, "w") as f:
    f.write("\n".join(lines))

log(f"\nResults written to {outfile}")
