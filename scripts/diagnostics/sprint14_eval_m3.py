from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from financial_algo.data.loader import load_prices
from financial_algo.regimes import detect_regime
from financial_algo.strategies.macro import EMRiskPremium
from scripts.production.run_crisis_backtest import (
    BT_CONFIG,
    CRISIS_WINDOWS,
    DATA_END,
    DATA_START,
    TICKERS,
    VIX_TICKER,
    run_strategy_backtest,
)

CORE_WINDOWS = [
    "Full Period (2010-2025)",
    "EU Debt Crisis (2011)",
    "Oil Crash (2014-2016)",
    "Volmageddon + Fed (2018)",
    "COVID-19 (2020)",
    "Russia-Ukraine + Inflation (2022)",
    "Recovery & Recent (2023-2025)",
]


def main() -> None:
    prices = load_prices(TICKERS, start=DATA_START, end=DATA_END)
    try:
        vix_df = load_prices([VIX_TICKER], start=DATA_START, end=DATA_END)
        vix = vix_df[VIX_TICKER] if VIX_TICKER in vix_df.columns else None
    except Exception:
        vix = None

    regime = detect_regime(prices, vix=vix)
    strat = EMRiskPremium()

    rows: list[dict[str, float | str]] = []
    for window_name in CORE_WINDOWS:
        start, end = CRISIS_WINDOWS[window_name]
        p = prices.loc[start:end]
        r = regime.loc[p.index]
        metrics = run_strategy_backtest(
            name="M3-EMRiskPremium",
            strategy=strat,
            prices=p,
            regime=r,
            config=BT_CONFIG,
            needs_regime=False,
        )
        if metrics is None:
            continue
        rows.append(
            {
                "window": window_name,
                "cagr": float(metrics["cagr"]),
                "sharpe": float(metrics["sharpe"]),
                "sortino": float(metrics["sortino"]),
                "max_drawdown": float(metrics["max_drawdown"]),
                "calmar": float(metrics["calmar"]),
                "win_rate": float(metrics["win_rate"]),
                "total_return": float(metrics["total_return"]),
            }
        )

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "sprint14_m3_after.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
