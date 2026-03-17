#!/usr/bin/env python3
"""Back-test the HFT Scalper strategy on synthetic 3-minute bars.

Usage
-----
    python backtests/backtest_hft_scalper.py [--days 20] [--seed 42] [--output results/hft.json]

The script generates realistic intraday price data (trending + mean-reverting
micro-structure) and runs the :class:`HFTScalperStrategy` through the
:class:`HFTBacktestEngine`.  Results are printed to stdout and optionally
saved to a JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── path bootstrap ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from financial_algorithms.backtest.hft_engine import HFTBacktestEngine  # noqa: E402
from financial_algorithms.strategies.hft_scalper import HFTScalperStrategy  # noqa: E402

# ── synthetic data generation ───────────────────────────────────────────


def generate_intraday_bars(
    n_days: int = 20,
    bars_per_day: int = 130,  # 6.5 h × 20 bars/h for 3-min bars
    start_price: float = 150.0,
    daily_vol: float = 0.015,
    trend_strength: float = 0.00002,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic 3-minute OHLCV bars with realistic micro-structure.

    The generator mixes:
    * An upward drift component (``trend_strength`` per bar).
    * A mean-reverting noise term that captures intraday reversion.
    * Volume that spikes at open/close (U-shaped intraday profile).

    Parameters
    ----------
    n_days : int
        Number of trading days to simulate.
    bars_per_day : int
        Bars per day (130 ≈ 6.5 hours of 3-min bars).
    start_price : float
        Initial price.
    daily_vol : float
        Approximate daily volatility (as a fraction, e.g. 0.012 = 1.2 %).
    trend_strength : float
        Per-bar drift.  Positive → upward bias.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp, open, high, low, close, volume``.
    """
    rng = np.random.default_rng(seed)
    total_bars = n_days * bars_per_day
    bar_vol = daily_vol / np.sqrt(bars_per_day)

    # mean-reverting noise via an AR(1) process
    noise = np.zeros(total_bars)
    phi = 0.85  # mean-reversion strength
    for i in range(1, total_bars):
        noise[i] = phi * noise[i - 1] + rng.normal(0, bar_vol)

    # per-bar log returns: small drift + mean-reverting component + iid noise
    innovations = np.diff(noise, prepend=0)
    log_returns = trend_strength + innovations + rng.normal(0, bar_vol * 0.3, total_bars)
    prices = start_price * np.exp(np.cumsum(log_returns))

    # OHLC from close
    spread = prices * bar_vol * 0.6
    opens = prices + rng.normal(0, 1, total_bars) * spread * 0.3
    highs = np.maximum(prices, opens) + np.abs(rng.normal(0, 1, total_bars)) * spread
    lows = np.minimum(prices, opens) - np.abs(rng.normal(0, 1, total_bars)) * spread

    # U-shaped volume profile (high at open & close, low mid-day)
    bar_in_day = np.tile(np.arange(bars_per_day), n_days)
    mid = bars_per_day / 2
    u_shape = 1.0 + 0.8 * ((bar_in_day - mid) / mid) ** 2
    base_vol = rng.uniform(50_000, 150_000, total_bars) * u_shape

    # timestamps: Mon-Fri trading days starting from a Monday
    trading_days = pd.bdate_range("2025-01-06", periods=n_days, freq="B")
    timestamps = []
    for day in trading_days:
        market_open = day + pd.Timedelta(hours=9, minutes=30)
        timestamps.extend(
            [market_open + pd.Timedelta(minutes=3 * j) for j in range(bars_per_day)]
        )

    return pd.DataFrame(
        {
            "timestamp": timestamps[:total_bars],
            "open": np.round(opens[:total_bars], 4),
            "high": np.round(highs[:total_bars], 4),
            "low": np.round(lows[:total_bars], 4),
            "close": np.round(prices[:total_bars], 4),
            "volume": np.round(base_vol[:total_bars]).astype(int),
        }
    )


# ── main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="HFT Scalper Backtest")
    parser.add_argument("--days", type=int, default=20, help="Trading days to simulate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    print("=" * 70)
    print("  HFT Scalper Backtest")
    print("=" * 70)

    # Generate synthetic data
    print(f"\nGenerating {args.days} days of 3-min bars (seed={args.seed}) ...")
    df = generate_intraday_bars(n_days=args.days, seed=args.seed)
    print(f"  Bars generated : {len(df)}")
    print(f"  Date range     : {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"  Price range    : {df['close'].min():.2f} → {df['close'].max():.2f}")

    # Strategy & engine
    strategy = HFTScalperStrategy()
    engine = HFTBacktestEngine(strategy=strategy, initial_capital=100_000)

    print("\nRunning backtest ...")
    results = engine.run(df, symbol="SYNTH")
    metrics = results["metrics"]
    trades_df = results["trades"]

    # ── report ──────────────────────────────────────────────────────
    print("\n" + "─" * 50)
    print("  PERFORMANCE METRICS")
    print("─" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")

    if not trades_df.empty:
        print("\n" + "─" * 50)
        print("  TRADE SUMMARY")
        print("─" * 50)
        print(f"  Total trades   : {len(trades_df)}")
        exits = trades_df["exit_reason"].value_counts()
        for reason, count in exits.items():
            print(f"    {reason:12s}: {count}")
        print("\n  First 10 trades:")
        print(trades_df.head(10).to_string(index=False))

    # ── save ────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "strategy": strategy.get_system_summary(),
            "metrics": metrics,
            "trade_count": len(trades_df),
        }
        out_path.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\n  Results saved to {out_path}")

    print("\n✓ HFT Scalper backtest complete")
    return metrics


if __name__ == "__main__":
    main()
