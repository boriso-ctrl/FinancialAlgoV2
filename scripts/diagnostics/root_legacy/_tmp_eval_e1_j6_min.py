import json
import sys
import types
from pathlib import Path
import importlib

ROOT = Path.cwd()
SRC = ROOT / "src"
PKG = SRC / "financial_algo"
fa = types.ModuleType("financial_algo")
fa.__path__ = [str(PKG)]
sys.modules["financial_algo"] = fa
fa_strat = types.ModuleType("financial_algo.strategies")
fa_strat.__path__ = [str(PKG / "strategies")]
sys.modules["financial_algo.strategies"] = fa_strat

load_prices = importlib.import_module("financial_algo.data.loader").load_prices
bt_mod = importlib.import_module("financial_algo.backtest")
BacktestConfig = bt_mod.BacktestConfig
backtest = bt_mod.backtest
MultiPairPortfolio = importlib.import_module("financial_algo.strategies.pairs").MultiPairPortfolio
DriftReversalAlpha = importlib.import_module("financial_algo.strategies.mean_reversion").DriftReversalAlpha

TICKERS = sorted(set([
    "SPY", "QQQ", "IWM", "EFA", "EEM", "GLD", "TLT", "XLE", "XLK", "XLF",
    "UUP", "HYG", "LQD", "XLP", "XLU", "XLY",
]))

prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
prices = prices.loc["2010-01-01":"2025-12-31"]

cfg = BacktestConfig(tx_cost_bps=5.0, leverage_cost_annual=0.015, short_cost_annual=0.005, initial_capital=1_000_000.0, vol_target=0.20, max_drawdown_trigger=-0.25, drawdown_recovery_rate=0.10)

out = {}
for name, strat in [("E1-MultiPairPortfolio", MultiPairPortfolio()), ("J6-DriftReversalAlpha", DriftReversalAlpha())]:
    w = strat.backtest_weights(prices)
    out[name] = backtest(prices, w, cfg)["metrics"]
print(json.dumps(out, indent=2))
