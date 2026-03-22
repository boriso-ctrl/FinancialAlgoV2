"""Quick diagnostic: run crisis backtest, capture any traceback to file."""
import sys
import traceback
import os

# Set paths
sys.path.insert(0, "src")
os.chdir(r"c:\Users\boris\Documents\GitHub\FinancialAlgoV2")

try:
    # Minimal reproduce of run_crisis_backtest.py flow
    from financial_algo.data.loader import load_prices
    from financial_algo.regimes import detect_regime
    from financial_algo.backtest import backtest, BacktestConfig
    from financial_algo.strategies.signal_combo import FeatureComboSignal, XGBoostSignalCombo

    TICKERS = [
        "SPY","QQQ","IWM","EFA","EEM","GLD","TLT","IEF","UUP",
        "XLE","USO","XOP","ITA","LMT","RTX","XLK","XLF","XLI","XLB",
        "XLP","XLU","XLY","XLV","HYG","LQD","SHY","AGG","TIP","VNQ",
        "BTC-USD","DBC","DBA","SLV","FXI","EWJ","INDA","VGK","EMB",
        "XBI","XLRE","XLC",
    ]
    print("Loading prices ...", flush=True)
    all_prices = load_prices(TICKERS, start="2009-01-01", end="2025-12-31")
    print(f"  Prices: {all_prices.shape}", flush=True)
    vix = load_prices(["^VIX"], start="2009-01-01", end="2025-12-31")["^VIX"]
    regimes = detect_regime(all_prices, vix=vix)

    # Load intraday features
    intraday_features = None
    if os.environ.get("ALPACA_API_KEY"):
        print("Loading Alpaca intraday features ...", flush=True)
        from financial_algo.data.feature_store import FeatureStore
        equity_tickers = [t for t in TICKERS if not t.startswith("^") and "-USD" not in t]
        store = FeatureStore(tickers=equity_tickers, start="2009-01-01", end="2025-12-31")
        intraday_features = store.load()
        if intraday_features is not None and not intraday_features.empty:
            print(f"  Loaded intraday: {intraday_features.shape}", flush=True)
        else:
            intraday_features = None
            print("  No intraday features loaded", flush=True)
    else:
        print("No ALPACA_API_KEY -- skipping intraday", flush=True)

    # Test P1
    print("\n--- Testing P1 ---", flush=True)
    p1 = FeatureComboSignal()
    if intraday_features is not None:
        p1.set_intraday_features(intraday_features)
    w1 = p1.generate_weights(all_prices, regimes)
    print(f"  P1 weights: {w1.shape}, NaN: {w1.isnull().any().any()}", flush=True)

    # Test P2
    print("\n--- Testing P2 ---", flush=True)
    p2 = XGBoostSignalCombo()
    if intraday_features is not None:
        p2.set_intraday_features(intraday_features)
    w2 = p2.generate_weights(all_prices, regimes)
    print(f"  P2 weights: {w2.shape}, NaN: {w2.isnull().any().any()}", flush=True)

    # Backtest P1
    print("\n--- Backtesting P1 ---", flush=True)
    cfg = BacktestConfig()
    r1 = backtest(w1, all_prices, config=cfg)
    print(f"  P1 CAGR: {r1.cagr:.2%}, Sharpe: {r1.sharpe:.2f}, MaxDD: {r1.max_drawdown:.2%}", flush=True)

    # Backtest P2
    print("\n--- Backtesting P2 ---", flush=True)
    r2 = backtest(w2, all_prices, config=cfg)
    print(f"  P2 CAGR: {r2.cagr:.2%}, Sharpe: {r2.sharpe:.2f}, MaxDD: {r2.max_drawdown:.2%}", flush=True)

    print("\nDONE", flush=True)

except Exception:
    traceback.print_exc()
    with open("results/bt_diag_error.txt", "w") as f:
        traceback.print_exc(file=f)
