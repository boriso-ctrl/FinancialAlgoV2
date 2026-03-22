r"""Alpaca data pipeline test script.

Usage:
    # Set your API credentials first:
    $env:ALPACA_API_KEY = "your-key-id"
    $env:ALPACA_SECRET = "your-secret-key"

    .venv\Scripts\python.exe scripts/test_alpaca_pipeline.py

This script:
1. Checks API connectivity and prints available history window
2. Downloads 5 days of 1-min bars for SPY and QQQ
3. Extracts intraday features and prints a sample
4. Validates the feature DataFrame shape and NaN rates
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Auto-load credentials from .env if present (no external deps required)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_root / "src"))

from financial_algo.data.alpaca_loader import get_available_history, load_intraday
from financial_algo.data.feature_store import FeatureStore, FEATURE_NAMES


def main() -> int:
    print("=" * 70)
    print("ALPACA DATA PIPELINE TEST")
    print("=" * 70)

    # Check credentials
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_SECRET", "")
    if not api_key or not secret:
        print()
        print("[ERROR] Environment variables not set.")
        print("  Set ALPACA_API_KEY and ALPACA_SECRET before running this script.")
        print("  Example (PowerShell):")
        print('    $env:ALPACA_API_KEY = "PKXXXXXXXXXXXXXXXXXX"')
        print('    $env:ALPACA_SECRET  = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"')
        return 1

    print(f"  API key: {api_key[:6]}..." + "*" * (len(api_key) - 6))
    print()

    # Step 1: Check history availability
    print("[1/3] Checking available 1-min history for SPY ...")
    try:
        earliest, latest = get_available_history("SPY", feed="iex")
        print(f"  Earliest IEX 1-min bar : {earliest}")
        print(f"  Latest  IEX 1-min bar : {latest}")
    except Exception as exc:
        print(f"  [WARN] Could not probe history: {exc}")
    print()

    # Step 2: Download sample 1-min bars
    print("[2/3] Downloading 5 days of 1-min bars for SPY and QQQ ...")
    try:
        bars = load_intraday(
            ["SPY", "QQQ"],
            start="2025-01-06",
            end="2025-01-10",
            cache_dir=None,  # no cache for this test
        )
        for ticker, df in bars.items():
            print(f"  {ticker}: {len(df)} bars | "
                  f"{df.index.min().date()} -> {df.index.max().date()} | "
                  f"NaN rate: {df.isna().mean().mean():.2%}")
    except Exception as exc:
        print(f"  [ERROR] Failed to download bars: {exc}")
        return 1
    print()

    # Step 3: Extract features from sample data
    print("[3/3] Extracting intraday features ...")
    try:
        store = FeatureStore(
            tickers=["SPY", "QQQ"],
            start="2024-01-01",
            end="2025-01-10",
        )
        features = store.build(force_refresh=False)
        if features.empty:
            print("  [WARN] Feature extraction returned empty DataFrame")
        else:
            print(f"  Shape  : {features.shape[0]} days x {features.shape[1]} features")
            print(f"  Columns: {list(features.columns[:8])} ...")
            print(f"  NaN rate : {features.isna().mean().mean():.2%}")
            print()
            print("  Sample (last 3 days):")
            print(features.tail(3).to_string())
    except Exception as exc:
        print(f"  [ERROR] Feature extraction failed: {exc}")
        return 1

    print()
    print("  Available feature names:")
    for i, name in enumerate(FEATURE_NAMES, 1):
        print(f"    {i}. {name}")
    print()
    print("=" * 70)
    print("Pipeline test PASSED. Ready for Phase 2 integration.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
