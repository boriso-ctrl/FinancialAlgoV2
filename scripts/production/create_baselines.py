"""Parse existing backtest CSVs into results/baselines.json for regression detection."""
import json
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

WINDOWS = {
    "Full_Period_2010-2025": "Full Period (2010-2025)",
    "COVID-19_2020": "COVID-19 (2020)",
    "EU_Debt_Crisis_2011": "EU Debt Crisis (2011)",
    "Oil_Crash_2014-2016": "Oil Crash (2014-2016)",
    "Volmageddon_+_Fed_2018": "Volmageddon + Fed (2018)",
    "Russia-Ukraine_+_Inflation_2022": "Russia-Ukraine + Inflation (2022)",
    "Recovery_&_Recent_2023-2025": "Recovery & Recent (2023-2025)",
}


def parse_pct(s):
    if s == "ERR":
        return None
    return round(float(s.replace("%", "")) / 100, 6)


def parse_float(s):
    if s == "ERR":
        return None
    return round(float(s), 4)


def main():
    baselines = {}
    for csv_key, window_name in WINDOWS.items():
        path = RESULTS / f"backtest_{csv_key}.csv"
        if not path.exists():
            print(f"  [SKIP] {path.name} not found")
            continue
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                strat = row["Strategy"]
                if strat not in baselines:
                    baselines[strat] = {}
                baselines[strat][window_name] = {
                    "sharpe": parse_float(row["Sharpe"]),
                    "cagr": parse_pct(row["CAGR"]),
                    "max_drawdown": parse_pct(row["MaxDD"]),
                    "sortino": parse_float(row["Sortino"]),
                    "calmar": parse_float(row["Calmar"]),
                }

    out = RESULTS / "baselines.json"
    with open(out, "w") as f:
        json.dump(baselines, f, indent=2)

    print(f"Baselines created: {len(baselines)} strategies across {len(WINDOWS)} windows")
    print(f"File: {out} ({out.stat().st_size:,} bytes)")

    # Quick summary of Full Period metrics
    print()
    print("Full Period (2010-2025) Summary:")
    print(f"{'Strategy':40s} {'Sharpe':>8s} {'CAGR':>10s} {'MaxDD':>10s}")
    print("-" * 70)
    for strat, windows in sorted(baselines.items()):
        fp = windows.get("Full Period (2010-2025)")
        if fp:
            sharpe = fp["sharpe"]
            cagr = fp["cagr"]
            maxdd = fp["max_drawdown"]
            flag = " *** NEG SHARPE" if sharpe is not None and sharpe < 0 else ""
            print(
                f"{strat:40s} {sharpe:8.4f} {cagr:10.4%} {maxdd:10.4%}{flag}"
            )


if __name__ == "__main__":
    main()
