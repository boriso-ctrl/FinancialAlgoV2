"""Quick DL2+DL3 test."""
import sys, warnings, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.backtest import BacktestConfig, backtest
from financial_algo.strategies.dl_strategies import (
    LSTMRegimeDetector,
    AttentionCrossSectionalRanker,
)

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

prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
vix = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")["^VIX"]
regime = detect_regime(prices, vix=vix)
mask = prices.index >= "2010-01-01"
prices = prices.loc[mask]
reg = regime.loc[mask]

bt = BacktestConfig(
    tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005,
    initial_capital=1_000_000.0, vol_target=0.20, max_drawdown_trigger=-0.25,
)

for name, strat_cls in [("DL2", LSTMRegimeDetector), ("DL3", AttentionCrossSectionalRanker)]:
    try:
        strat = strat_cls()
        print(f"Running {name} ({strat.name}) ...")
        w = strat.backtest_weights(prices, reg)
        print(f"  Weights shape: {w.shape}, non-zero: {(w != 0).any(axis=1).sum()}")
        res = backtest(prices, w, bt)
        m = res["metrics"]
        print(f"  Sharpe={m.get('sharpe', 0):.2f}  CAGR={m.get('cagr', 0):.1%}  MaxDD={m.get('max_drawdown', 0):.1%}")
    except Exception:
        traceback.print_exc()
    print()
