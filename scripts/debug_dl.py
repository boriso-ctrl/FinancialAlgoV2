"""Debug individual DL strategies."""
from __future__ import annotations
import sys, warnings, traceback
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))
sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime

TICKERS = [
    "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "SLV", "TLT", "IEF", "UUP",
    "XLE", "USO", "XOP", "XLK", "XLF", "XLI", "XLB", "XLP", "XLU", "XLY", "XLV",
    "ITA", "LMT", "RTX", "HYG", "LQD", "DBC", "VNQ", "BTC-USD",
]

print("Loading data...")
prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
try:
    vix_df = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")
    vix = vix_df["^VIX"]
except Exception:
    vix = None
regime = detect_regime(prices, vix=vix)

mask = (prices.index >= "2010-01-01") & (prices.index <= "2025-12-31")
p = prices.loc[mask]
print(f"Data: {len(p)} days, {p.shape[1]} tickers")

cfg = BacktestConfig()

# Test DL2 only
print("\n--- Testing DL2-LSTMRegimeDetector ---")
try:
    from financial_algo.strategies.dl_strategies import LSTMRegimeDetector, LSTMRegimeConfig
    strat = LSTMRegimeDetector()
    print(f"  Config: seq_len={strat.cfg.seq_len}, min_train={strat.cfg.min_train_days}")
    print(f"  Days available: {len(p)} (need {strat.cfg.min_train_days} warm-up)")
    print("  Generating weights...")
    w = strat.backtest_weights(p)
    print(f"  Weights shape: {w.shape}")
    print(f"  Non-zero rows: {(w.abs().sum(axis=1) > 0).sum()}")
    print("  Running backtest...")
    res = backtest(p, w, cfg)
    m = res["metrics"]
    print(f"  CAGR:    {m['cagr']*100:.2f}%")
    print(f"  Sharpe:  {m['sharpe']:.2f}")
    print(f"  MaxDD:   {m['max_drawdown']*100:.2f}%")
except Exception as e:
    print(f"  CRASHED: {e}")
    traceback.print_exc()

# Test DL3 only
print("\n--- Testing DL3-AttentionRanker ---")
try:
    from financial_algo.strategies.dl_strategies import AttentionCrossSectionalRanker
    strat = AttentionCrossSectionalRanker()
    print(f"  Config: min_train={strat.cfg.min_train_days}")
    print("  Generating weights...")
    w = strat.backtest_weights(p)
    print(f"  Weights shape: {w.shape}")
    print(f"  Non-zero rows: {(w.abs().sum(axis=1) > 0).sum()}")
    print("  Running backtest...")
    res = backtest(p, w, cfg)
    m = res["metrics"]
    print(f"  CAGR:    {m['cagr']*100:.2f}%")
    print(f"  Sharpe:  {m['sharpe']:.2f}")
    print(f"  MaxDD:   {m['max_drawdown']*100:.2f}%")
except Exception as e:
    print(f"  CRASHED: {e}")
    traceback.print_exc()

print("\nDone.")
