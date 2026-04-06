from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE_CSV = ROOT / "results" / "sprint15_phase12_comprehensive.csv"
CURRENT_CSV_CANDIDATES = [
    ROOT / "results" / "backtest_Full_Period_2010-2025.csv",
    ROOT / "scripts" / "results" / "backtest_Full_Period_2010-2025.csv",
]
OUT_CSV = ROOT / "results" / "sprint15_go_no_go.csv"
OUT_MD = ROOT / "results" / "sprint15_go_no_go.md"


def parse_float(value: str) -> float:
    text = (value or "").strip().replace("%", "")
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def decision_from_delta(delta: float) -> str:
    if delta <= -0.20:
        return "rollback_required"
    if delta < -0.05:
        return "rework_priority"
    if delta < 0.05:
        return "hold_watch"
    return "promote"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_current_csv() -> Path:
    existing = [p for p in CURRENT_CSV_CANDIDATES if p.exists()]
    if not existing:
        raise FileNotFoundError("No full-period backtest CSV found in results/ or scripts/results/.")
    return max(existing, key=lambda p: p.stat().st_mtime)


def main() -> None:
    baseline_rows = read_rows(BASELINE_CSV)
    current_csv = resolve_current_csv()
    current_rows = read_rows(current_csv)

    baseline = {r["strategy"]: parse_float(r.get("sharpe", "0")) for r in baseline_rows}
    current = {r["Strategy"]: parse_float(r.get("Sharpe", "0")) for r in current_rows}

    records: list[dict[str, str | float]] = []
    for strategy, baseline_sharpe in baseline.items():
        current_sharpe = current.get(strategy)
        if current_sharpe is None:
            continue
        delta = current_sharpe - baseline_sharpe
        records.append(
            {
                "strategy": strategy,
                "baseline_sharpe": round(baseline_sharpe, 4),
                "current_sharpe": round(current_sharpe, 4),
                "delta_sharpe": round(delta, 4),
                "decision": decision_from_delta(delta),
            }
        )

    records.sort(key=lambda r: float(r["delta_sharpe"]))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "strategy",
                "baseline_sharpe",
                "current_sharpe",
                "delta_sharpe",
                "decision",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    rollback = [r for r in records if r["decision"] == "rollback_required"]
    rework = [r for r in records if r["decision"] == "rework_priority"]
    hold = [r for r in records if r["decision"] == "hold_watch"]
    promote = [r for r in records if r["decision"] == "promote"]

    lines: list[str] = []
    lines.append("# Sprint 15 Go/No-Go (Full Period Sharpe Delta)")
    lines.append("")
    lines.append(f"Total strategies compared: {len(records)}")
    lines.append(f"- Promote: {len(promote)}")
    lines.append(f"- Hold/Watch: {len(hold)}")
    lines.append(f"- Rework Priority: {len(rework)}")
    lines.append(f"- Rollback Required: {len(rollback)}")
    lines.append("")

    lines.append("## Top 10 Regressions")
    lines.append("")
    lines.append("| Strategy | Baseline Sharpe | Current Sharpe | Delta | Decision |")
    lines.append("|---|---:|---:|---:|---|")
    for r in records[:10]:
        lines.append(
            f"| {r['strategy']} | {r['baseline_sharpe']:.2f} | {r['current_sharpe']:.2f} | {r['delta_sharpe']:.2f} | {r['decision']} |"
        )

    lines.append("")
    lines.append("## Top 10 Improvements")
    lines.append("")
    lines.append("| Strategy | Baseline Sharpe | Current Sharpe | Delta | Decision |")
    lines.append("|---|---:|---:|---:|---|")
    for r in records[-10:]:
        lines.append(
            f"| {r['strategy']} | {r['baseline_sharpe']:.2f} | {r['current_sharpe']:.2f} | {r['delta_sharpe']:.2f} | {r['decision']} |"
        )

    with OUT_MD.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote: {OUT_CSV}")
    print(f"Wrote: {OUT_MD}")
    print(f"Using current metrics from: {current_csv}")


if __name__ == "__main__":
    main()
