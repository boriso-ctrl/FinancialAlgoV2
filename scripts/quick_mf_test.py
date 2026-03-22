"""Quick MF strategy evaluation."""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.strategies.multi_freq import (
    WeeklyMomentumRotation, MonthlyMacroRegime,
    MultiTimeframeTrend, WeeklyMeanReversion,
)

TICKERS = sorted(set([
    "SPY","QQQ","IWM","EFA","EEM","GLD","TLT","IEF","UUP",
    "XLE","USO","XOP","ITA","LMT","RTX",
    "XLK","XLF","XLI","XLB","XLP","XLU","XLY","XLV",
    "HYG","LQD","BTC-USD",
]))

print("Loading data...")
prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
p = prices.loc[mask]
print(f"Data: {len(p)} days, {p.shape[1]} tickers")

cfg = BacktestConfig(
    tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
    initial_capital=1_000_000.0, vol_target=0.20, max_drawdown_trigger=-0.25,
)

# Also compute SPY correlation
spy_ret = p["SPY"].pct_change().fillna(0)

strats = [
    ("MF1-WeeklyMomRotation", WeeklyMomentumRotation()),
    ("MF2-MonthlyMacroRegime", MonthlyMacroRegime()),
    ("MF3-MultiTimeframeTrend", MultiTimeframeTrend()),
    ("MF4-WeeklyMeanReversion", WeeklyMeanReversion()),
]

header = "{:<28} {:>7} {:>7} {:>8} {:>8} {:>7} {:>8}".format(
    "Strategy", "CAGR", "Sharpe", "MaxDD", "Sortino", "Calmar", "SPY Corr"
)
print()
print(header)
print("-" * len(header))

for name, strat in strats:
    w = strat.backtest_weights(p)
    res = backtest(p, w, cfg)
    m = res["metrics"]
    # Compute correlation with SPY
    eq = res["equity"]
    strat_ret = eq.pct_change().fillna(0)
    corr = strat_ret.corr(spy_ret)
    print("{:<28} {:>6.1f}% {:>7.2f} {:>7.1f}% {:>8.2f} {:>7.2f} {:>8.2f}".format(
        name, m["cagr"]*100, m["sharpe"], m["max_drawdown"]*100,
        m["sortino"], m["calmar"], corr
    ))

# Also print pairwise correlations among MF strategies
print("\nPairwise correlations among MF strategies:")
returns = {}
for name, strat in strats:
    w = strat.backtest_weights(p)
    res = backtest(p, w, cfg)
    returns[name] = res["equity"].pct_change().fillna(0)

corr_df = pd.DataFrame(returns).corr()
print(corr_df.to_string(float_format="{:.2f}".format))
