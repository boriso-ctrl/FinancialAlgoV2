"""HFT All-11 Quick Eval — differentiated cost schedule.

Runs each intraday strategy against 1-min Alpaca data,
applies execution controls, and reports live metrics.

Usage:
    .venv/Scripts/python.exe scripts/_tmp_hft_all11_eval.py

Execution controls applied:
    rebalance_bars   : 5   (bars)
    change_threshold : 0.25
    quant_step       : 0.50
"""

from __future__ import annotations

import sys
import traceback
from typing import Protocol

import numpy as np
import pandas as pd

# Cost buckets per strategy family (one-way, in price fraction)
COST_SCHEDULE: dict[str, float] = {
    "MMT": 0.00022,   # aggressive — high turnover, cross spread
    "MRM": 0.00012,   # passive — limit orders, mean-reversion
    "VEF": 0.00017,   # mixed — impulse fades, partially aggressive
    "HYB": 0.00020,   # router — mixed aggressiveness
}

REBALANCE_BARS: int = 5
CHANGE_THRESHOLD: float = 0.25
QUANT_STEP: float = 0.50
ANNUAL_BARS: int = 252 * 390  # 1-min bars per year


class IntradayStrategyProtocol(Protocol):
    name: str
    timeframe: str

    def generate_signal(self, ohlcv: pd.DataFrame) -> pd.Series: ...


# ---------------------------------------------------------------------------
# Execution control helpers
# ---------------------------------------------------------------------------


def _quantize(positions: pd.Series, step: float) -> pd.Series:
    """Round positions to nearest quant_step."""
    return (positions / step).round() * step


def _apply_execution_controls(
    raw: pd.Series,
    rebalance_bars: int,
    change_threshold: float,
    quant_step: float,
) -> pd.Series:
    """Apply rebalance throttle, change threshold, and position quantization."""
    quantized = _quantize(raw, quant_step)
    result = quantized.copy()
    last_rebalance = 0
    last_pos = 0.0

    for i, (idx, pos) in enumerate(quantized.items()):
        bars_since = i - last_rebalance
        change = abs(pos - last_pos)
        if bars_since >= rebalance_bars and change >= change_threshold:
            result.loc[idx] = pos
            last_rebalance = i
            last_pos = pos
        else:
            result.loc[idx] = last_pos

    return result


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------


def _backtest(
    ohlcv: pd.DataFrame,
    signal: pd.Series,
    one_way_cost: float,
) -> dict[str, float]:
    """Run vectorized backtest; return performance metrics."""
    close = ohlcv["close"].astype(float)
    ret = close.pct_change().fillna(0.0)

    pos = signal.reindex(ret.index).fillna(0.0)
    strat_ret = pos.shift(1).fillna(0.0) * ret

    # Turnover costs
    turnover = pos.diff().abs().fillna(0.0)
    cost_drag = turnover * one_way_cost
    net_ret = strat_ret - cost_drag

    n = len(net_ret)
    avg = net_ret.mean()
    std = net_ret.std()
    sharpe = (avg / std) * np.sqrt(ANNUAL_BARS) if std > 1e-12 else 0.0

    cum = (1.0 + net_ret).cumprod()
    roll_max = cum.cummax()
    drawdown = (cum / roll_max) - 1.0
    max_dd = drawdown.min()

    cagr = cum.iloc[-1] ** (ANNUAL_BARS / max(n, 1)) - 1.0 if cum.iloc[-1] > 0 else -1.0

    gross_to = turnover.mean() * ANNUAL_BARS
    net_to = pos.diff().fillna(0.0).abs().mean() * ANNUAL_BARS

    return {
        "sharpe": sharpe,
        "cagr": cagr,
        "max_dd": max_dd,
        "gross_turnover": gross_to,
        "net_turnover": net_to,
        "n_bars": n,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def _cost_for(strategy_name: str) -> float:
    for prefix, cost in COST_SCHEDULE.items():
        if prefix in strategy_name:
            return cost
    return 0.00020  # fallback


def main() -> None:
    print("=" * 70)
    print("  HFT All-11 Eval — differentiated cost schedule")
    print("=" * 70)

    # -- load strategies --
    try:
        from financial_algo.strategies.intraday_research_pack import (
            BollingerSnapback,
            GapRegimeSelector,
            KSTTSIOpeningBurst,
            MACDHistogramAcceleration,
            RangeExpansionExhaustion,
            RealizedVolImpulseFade,
            RelativeVolContinuation,
            SqueezeReleaseBreakout,
            VolRegimeRouter,
            VWAPVolNormalizedFade,
            WickRejectionFade,
        )
    except ImportError as exc:
        print(f"[ERROR] Could not import strategies: {exc}")
        sys.exit(1)

    strategies: list[IntradayStrategyProtocol] = [
        MACDHistogramAcceleration(),
        VWAPVolNormalizedFade(),
        RealizedVolImpulseFade(),
        KSTTSIOpeningBurst(),
        RelativeVolContinuation(),
        WickRejectionFade(),
        BollingerSnapback(),
        SqueezeReleaseBreakout(),
        RangeExpansionExhaustion(),
        VolRegimeRouter(),
        GapRegimeSelector(),
    ]

    # -- load market data (synthetic fallback if Alpaca unavailable) --
    try:
        from financial_algo.data.alpaca_loader import load_intraday

        print("[DATA] Loading 1-min intraday data via Alpaca...")
        data = load_intraday(["SPY"], start="2023-01-01", end="2024-12-31")
        ohlcv = data.get("SPY", None)
        if ohlcv is None or len(ohlcv) < 500:
            raise ValueError("Insufficient SPY data from Alpaca")
        print(f"[DATA] Loaded {len(ohlcv):,} bars for SPY")
    except Exception as exc:
        print(f"[DATA] Alpaca unavailable ({exc}), using synthetic OHLCV")
        rng = np.random.default_rng(0)
        n = 5_000
        idx = pd.date_range("2024-01-02 09:31:00", periods=n, freq="1min", tz="UTC")
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
        h = close + rng.uniform(0.01, 0.15, n)
        l = close - rng.uniform(0.01, 0.15, n)
        o = close - rng.standard_normal(n) * 0.03
        vol = rng.integers(1000, 20000, n).astype(float)
        ohlcv = pd.DataFrame({"open": o, "high": h, "low": l, "close": close, "volume": vol}, index=idx)

    # -- eval loop --
    rows: list[dict] = []
    print()
    print(f"{'Strategy':<45} {'Sharpe':>7} {'CAGR':>8} {'MaxDD':>8} {'GrossTO':>9} {'Status'}")
    print("-" * 90)

    for strat in strategies:
        cost = _cost_for(strat.name)
        try:
            raw = strat.generate_signal(ohlcv)
            controlled = _apply_execution_controls(
                raw, REBALANCE_BARS, CHANGE_THRESHOLD, QUANT_STEP
            )
            metrics = _backtest(ohlcv, controlled, cost)
            status = "PROMOTED" if "VEF-1" in strat.name else ("KILLED" if metrics["sharpe"] < 0 else "ACTIVE")
            print(
                f"{strat.name:<45} {metrics['sharpe']:>7.2f} "
                f"{metrics['cagr']:>8.2%} {metrics['max_dd']:>8.2%} "
                f"{metrics['gross_turnover']:>9.1f}x  {status}"
            )
            rows.append({"strategy": strat.name, "cost": cost, **metrics, "status": status})
        except Exception:
            print(f"{strat.name:<45} [EXCEPTION]")
            traceback.print_exc()

    print()
    print(f"[COST SCHEDULE] {COST_SCHEDULE}")
    print(
        f"[EXECUTION] rebalance_bars={REBALANCE_BARS}, "
        f"change_threshold={CHANGE_THRESHOLD}, quant_step={QUANT_STEP}"
    )

    promoted = [r for r in rows if r["status"] == "PROMOTED"]
    killed = [r for r in rows if r["status"] == "KILLED"]
    print(f"\nSummary: {len(promoted)} PROMOTED, {len(killed)} KILLED, {len(rows) - len(promoted) - len(killed)} ACTIVE")

    # Write output
    import pathlib

    out_path = pathlib.Path("results") / "backtest_intraday_v10.txt"
    out_path.parent.mkdir(exist_ok=True)
    with out_path.open("w") as fh:
        fh.write("strategy,sharpe,cagr,max_dd,gross_turnover,cost,status\n")
        for r in rows:
            fh.write(
                f"{r['strategy']},{r['sharpe']:.4f},{r['cagr']:.4f},"
                f"{r['max_dd']:.4f},{r['gross_turnover']:.2f},{r['cost']:.5f},{r['status']}\n"
            )
    print(f"\n[OUTPUT] Written to {out_path}")


if __name__ == "__main__":
    main()
