from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from financial_algo.strategies.crisis_spike import CommodityShockConfig, CommodityShockRider
from financial_algo.strategies.macro import DollarCarry, DollarCarryConfig

RUNNER_PATH = ROOT / "scripts" / "production" / "run_crisis_backtest.py"
OUT_CSV = ROOT / "results" / "sprint15_priority_variant_eval.csv"


def load_runner_module():
    spec = importlib.util.spec_from_file_location("sprint15_production_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load production runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_variant_registry() -> list[tuple[str, object, bool]]:
    return [
        (
            "M1-current",
            DollarCarry(),
            False,
        ),
        (
            "M1-smoother-defensive",
            DollarCarry(
                DollarCarryConfig(
                    score_smooth_span=12,
                    leverage_risk_on=1.35,
                    safe_weight=0.80,
                )
            ),
            False,
        ),
        (
            "M1-tighter-riskoff",
            DollarCarry(
                DollarCarryConfig(
                    score_smooth_span=6,
                    leverage_risk_on=1.40,
                    safe_weight=0.85,
                )
            ),
            False,
        ),
        (
            "M1-revert-like",
            DollarCarry(
                DollarCarryConfig(
                    score_smooth_span=5,
                    leverage_risk_on=1.50,
                    safe_weight=1.00,
                )
            ),
            False,
        ),
        (
            "M1-mid-balance",
            DollarCarry(
                DollarCarryConfig(
                    score_smooth_span=8,
                    leverage_risk_on=1.45,
                    safe_weight=0.80,
                )
            ),
            False,
        ),
        (
            "H1-current",
            CommodityShockRider(),
            True,
        ),
        (
            "H1-revert-like",
            CommodityShockRider(
                CommodityShockConfig(
                    spike_z=1.5,
                    fast_spike_z=2.0,
                    fast_momentum_threshold=0.05,
                    momentum_threshold=0.03,
                    leverage_energy=2.0,
                    leverage_gold=1.0,
                    exit_momentum_threshold=-0.02,
                )
            ),
            True,
        ),
        (
            "H1-tighter-entry",
            CommodityShockRider(
                CommodityShockConfig(
                    spike_z=1.4,
                    fast_spike_z=2.1,
                    fast_momentum_threshold=0.06,
                    momentum_threshold=0.03,
                    leverage_energy=1.6,
                    leverage_gold=0.8,
                    exit_momentum_threshold=-0.02,
                )
            ),
            True,
        ),
        (
            "H1-lower-gross",
            CommodityShockRider(
                CommodityShockConfig(
                    spike_z=1.3,
                    fast_spike_z=1.9,
                    fast_momentum_threshold=0.06,
                    momentum_threshold=0.03,
                    leverage_energy=1.4,
                    leverage_gold=0.7,
                    exit_momentum_threshold=-0.02,
                )
            ),
            True,
        ),
        (
            "H1-balanced",
            CommodityShockRider(
                CommodityShockConfig(
                    spike_z=1.4,
                    fast_spike_z=2.1,
                    fast_momentum_threshold=0.06,
                    momentum_threshold=0.03,
                    leverage_energy=1.3,
                    leverage_gold=0.6,
                    exit_momentum_threshold=-0.02,
                )
            ),
            True,
        ),
        (
            "H1-balanced-plus",
            CommodityShockRider(
                CommodityShockConfig(
                    spike_z=1.35,
                    fast_spike_z=2.0,
                    fast_momentum_threshold=0.06,
                    momentum_threshold=0.03,
                    leverage_energy=1.35,
                    leverage_gold=0.65,
                    exit_momentum_threshold=-0.02,
                )
            ),
            True,
        ),
    ]


def select_windows(runner_module) -> list[str]:
    return [
        "Full Period (2010-2025)",
        "Oil Crash (2014-2016)",
        "Volmageddon + Fed (2018)",
        "Russia-Ukraine + Inflation (2022)",
        "ME: Full Conflict (Oct23-Dec24)",
    ]


def evaluate_variants() -> pd.DataFrame:
    runner = load_runner_module()
    prices = runner.load_prices(runner.TICKERS, start=runner.DATA_START, end=runner.DATA_END)
    try:
        vix_df = runner.load_prices([runner.VIX_TICKER], start=runner.DATA_START, end=runner.DATA_END)
        vix = vix_df[runner.VIX_TICKER]
    except Exception:
        vix = None
    regime = runner.detect_regime(prices, vix=vix)

    rows: list[dict[str, object]] = []
    for window_name in select_windows(runner):
        win_start, win_end = runner.CRISIS_WINDOWS[window_name]
        mask = (prices.index >= win_start) & (prices.index <= win_end)
        p_win = prices.loc[mask].copy()
        r_win = regime.loc[mask].copy()

        for variant_name, strategy, needs_regime in build_variant_registry():
            metrics = runner.run_strategy_backtest(
                variant_name,
                strategy,
                p_win,
                r_win,
                runner.BT_CONFIG,
                needs_regime=needs_regime,
            )
            if metrics is None:
                rows.append(
                    {
                        "variant": variant_name,
                        "window": window_name,
                        "sharpe": None,
                        "cagr": None,
                        "max_drawdown": None,
                        "annual_vol": None,
                        "win_rate": None,
                    }
                )
                continue

            rows.append(
                {
                    "variant": variant_name,
                    "window": window_name,
                    "sharpe": round(float(metrics["sharpe"]), 4),
                    "cagr": round(float(metrics["cagr"]), 4),
                    "max_drawdown": round(float(metrics["max_drawdown"]), 4),
                    "annual_vol": round(float(metrics["annual_vol"]), 4),
                    "win_rate": round(float(metrics["win_rate"]), 4),
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    df = evaluate_variants()
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, quoting=csv.QUOTE_MINIMAL)

    full_period = df[df["window"] == "Full Period (2010-2025)"].sort_values("sharpe", ascending=False)
    print(full_period.to_string(index=False))
    print()
    print(f"Wrote: {OUT_CSV}")


if __name__ == "__main__":
    main()