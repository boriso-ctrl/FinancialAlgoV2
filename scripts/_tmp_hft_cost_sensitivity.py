"""HFT Cost Sensitivity Analysis — bucket scenarios.

Tests each strategy across four cost scenarios:
    passive    : 0.00012  (limit orders, posted liquidity)
    mixed      : 0.00020  (50/50 passive/aggressive)
    aggressive : 0.00022  (crossing the spread, IOC orders)
    router     : family-specific cost (actual deployed cost estimate)

Usage:
    .venv/Scripts/python.exe scripts/_tmp_hft_cost_sensitivity.py
"""

from __future__ import annotations

import pathlib
import sys
import traceback

import numpy as np
import pandas as pd

REBALANCE_BARS: int = 5
CHANGE_THRESHOLD: float = 0.25
QUANT_STEP: float = 0.50
ANNUAL_BARS: int = 252 * 390

COST_SCENARIOS: dict[str, float] = {
    "passive":    0.00012,
    "mixed":      0.00020,
    "aggressive": 0.00022,
    "router": -1.0,  # sentinel: use family cost
}

FAMILY_COSTS: dict[str, float] = {
    "MMT": 0.00022,
    "MRM": 0.00012,
    "VEF": 0.00017,
    "HYB": 0.00020,
}


def _family_cost(name: str) -> float:
    for prefix, cost in FAMILY_COSTS.items():
        if prefix in name:
            return cost
    return 0.00020


def _quantize(pos: pd.Series, step: float) -> pd.Series:
    return (pos / step).round() * step


def _apply_controls(raw: pd.Series, rebalance_bars: int, change_threshold: float, quant_step: float) -> pd.Series:
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


def _backtest(ohlcv: pd.DataFrame, signal: pd.Series, cost: float) -> dict[str, float]:
    close = ohlcv["close"].astype(float)
    ret = close.pct_change().fillna(0.0)
    pos = signal.reindex(ret.index).fillna(0.0)
    strat_ret = pos.shift(1).fillna(0.0) * ret
    turnover = pos.diff().abs().fillna(0.0)
    net_ret = strat_ret - turnover * cost

    n = len(net_ret)
    avg, std = net_ret.mean(), net_ret.std()
    sharpe = (avg / std) * np.sqrt(ANNUAL_BARS) if std > 1e-12 else 0.0
    cum = (1.0 + net_ret).cumprod()
    roll_max = cum.cummax()
    max_dd = ((cum / roll_max) - 1.0).min()
    cagr = cum.iloc[-1] ** (ANNUAL_BARS / max(n, 1)) - 1.0 if cum.iloc[-1] > 0 else -1.0
    gross_to = turnover.mean() * ANNUAL_BARS
    return {"sharpe": sharpe, "cagr": cagr, "max_dd": max_dd, "gross_turnover": gross_to}


def main() -> None:
    print("=" * 80)
    print("  HFT Cost Sensitivity — passive / mixed / aggressive / router buckets")
    print("=" * 80)

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
        print(f"[ERROR] {exc}")
        sys.exit(1)

    strategies = [
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

    # Data
    try:
        from financial_algo.data.alpaca_loader import load_intraday

        data = load_intraday(["SPY"], start="2023-01-01", end="2024-12-31")
        ohlcv = data.get("SPY", None)
        if ohlcv is None or len(ohlcv) < 500:
            raise ValueError("Insufficient data")
        print(f"[DATA] {len(ohlcv):,} bars (Alpaca SPY)")
    except Exception as exc:
        print(f"[DATA] Synthetic ({exc})")
        rng = np.random.default_rng(1)
        n = 5_000
        idx = pd.date_range("2024-01-02 09:31:00", periods=n, freq="1min", tz="UTC")
        close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05)
        ohlcv = pd.DataFrame({
            "open": close - rng.standard_normal(n) * 0.03,
            "high": close + rng.uniform(0.01, 0.15, n),
            "low": close - rng.uniform(0.01, 0.15, n),
            "close": close,
            "volume": rng.integers(1000, 20000, n).astype(float),
        }, index=idx)

    # Header
    scenario_names = list(COST_SCENARIOS.keys())
    header = f"{'Strategy':<45}"
    for sc in scenario_names:
        header += f"  {sc:>10}"
    print()
    print(header)
    print("-" * (45 + 12 * len(scenario_names)))

    rows: list[dict] = []
    for strat in strategies:
        row: dict = {"strategy": strat.name}
        line = f"{strat.name:<45}"
        try:
            raw = strat.generate_signal(ohlcv)
            controlled = _apply_controls(raw, REBALANCE_BARS, CHANGE_THRESHOLD, QUANT_STEP)
            for sc_name, sc_cost in COST_SCENARIOS.items():
                cost = _family_cost(strat.name) if sc_cost < 0 else sc_cost
                m = _backtest(ohlcv, controlled, cost)
                sharpe = m["sharpe"]
                row[sc_name] = sharpe
                verdict = "PASS" if sharpe >= 0.3 else "FAIL"
                line += f"  {sharpe:>+6.2f}({verdict})"
        except Exception:
            line += "  [ERROR]"
            traceback.print_exc()
        print(line)
        rows.append(row)

    print()
    print("[GATES] Promotion gate: OOS Sharpe >= 0.30 in router bucket")
    promoted = [r["strategy"] for r in rows if r.get("router", -99) >= 0.30]
    print(f"[PROMOTED at router cost] {promoted if promoted else 'None'}")

    # Write
    out_path = pathlib.Path("results") / "backtest_intraday_cost_sensitivity.txt"
    out_path.parent.mkdir(exist_ok=True)
    cols = ["strategy"] + scenario_names
    with out_path.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in rows:
            vals = [str(r.get(c, "")) for c in cols]
            fh.write(",".join(vals) + "\n")
    print(f"\n[OUTPUT] Written to {out_path}")


if __name__ == "__main__":
    main()
