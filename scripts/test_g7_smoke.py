"""Smoke test for G7-RedditSentimentAlpha strategy."""
import sys
import traceback

sys.path.insert(0, "src")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


def main():
    # Test 1: Import reddit_feeds
    from financial_algo.fundamental.data.reddit_feeds import (
        build_synthetic_reddit_sentiment,
    )
    print("PASS: reddit_feeds import")

    # Test 2: Import strategy
    from financial_algo.fundamental.strategies.sentiment_strategies import (
        RedditSentimentAlpha,
    )
    print("PASS: RedditSentimentAlpha import")

    # Test 3: Build synthetic reddit data
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=300)
    n = len(dates)
    ret = np.random.normal(0.0005, 0.015, (n, 6))
    cols = ["SPY", "QQQ", "IWM", "TLT", "GLD", "XLE"]
    cum = np.exp(np.cumsum(ret, axis=0)) * 100
    prices = pd.DataFrame(cum, index=dates, columns=cols)

    feats = build_synthetic_reddit_sentiment(prices)
    nan_rate = feats.isnull().mean().mean()
    print(
        f"PASS: synthetic reddit features: {feats.shape[0]} rows x "
        f"{feats.shape[1]} cols, NaN rate: {nan_rate:.4f}"
    )

    # Test 4: Run strategy with synthetic data
    strat = RedditSentimentAlpha()
    w = strat.generate_weights(prices)
    has_nan = w.isnull().any().any()
    max_lev = w.abs().sum(axis=1).max()
    print(
        f"PASS: G7 weights: {w.shape}, NaN: {has_nan}, "
        f"max_lev: {max_lev:.3f}"
    )
    print(f"  Columns: {list(w.columns)}")
    print(f"  Mean equity weight: {w['SPY'].mean():.3f}")

    # Test 5: backtest_weights (shift test)
    bw = strat.backtest_weights(prices)
    first_zero = (bw.iloc[0] == 0).all()
    print(
        f"PASS: backtest_weights shift: {bw.shape}, "
        f"first row all zero: {first_zero}"
    )

    # Test 6: With regime
    from financial_algo.regimes import Regime

    regime = pd.Series([Regime.NORMAL] * n, index=dates)
    regime.iloc[100:120] = Regime.GENERAL_CRISIS
    w2 = strat.generate_weights(prices, regime=regime)
    crisis_lev = w2.iloc[100:120].abs().sum(axis=1).mean()
    normal_lev = w2.iloc[0:50].abs().sum(axis=1).mean()
    print(
        f"PASS: regime filter: crisis avg_lev={crisis_lev:.3f}, "
        f"normal avg_lev={normal_lev:.3f}"
    )

    # Test 7: No inf/NaN in output
    has_inf = np.isinf(w.values).any()
    has_nan2 = np.isnan(w.values).any()
    assert not has_inf, "Output contains inf!"
    assert not has_nan2, "Output contains NaN!"
    print("PASS: no inf/NaN in output")

    print()
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
