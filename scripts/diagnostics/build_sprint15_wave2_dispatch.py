"""
Build Wave 2 dispatch pack from sprint15 execution matrix.

Outputs:
- results/sprint15_wave2_dispatch.md
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
MATRIX_CSV = RESULTS_DIR / "sprint15_execution_matrix.csv"
OUT_MD = RESULTS_DIR / "sprint15_wave2_dispatch.md"


def main() -> None:
    df = pd.read_csv(MATRIX_CSV)

    # Prioritize immediate unresolved backlog.
    backlog = df[
        (df["execution_status"] == "researched_only")
        & (df["wave"].isin(["wave_1_kill_or_rebuild", "wave_1_high_urgency"]))
    ].copy()

    backlog = backlog.sort_values(["wave", "sharpe"], ascending=[True, True])

    owners = ["crisis", "systematic", "macro", "vol", "peter"]

    lines: list[str] = []
    lines.append("# Sprint 15 Wave 2 Dispatch Pack")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Candidate set size: {len(backlog)}")
    lines.append("- Source: results/sprint15_execution_matrix.csv")
    lines.append("- Selection: researched_only + (wave_1_kill_or_rebuild or wave_1_high_urgency)")
    lines.append("")
    lines.append("## Global Acceptance Criteria")
    lines.append("- Sharpe uplift target per rework: >= +0.15 (or kill if still < 0)")
    lines.append("- No look-ahead bias; no shift(-1); no double-shift in backtest path")
    lines.append("- NaN/inf-safe outputs from generate_weights")
    lines.append("- Vectorized implementation only")
    lines.append("- Pytest green after each batch")
    lines.append("")

    for owner in owners:
        owner_df = backlog[backlog["owner"] == owner].head(5)
        if owner_df.empty:
            continue

        lines.append(f"## {owner.capitalize()} Targets")
        lines.append("")
        lines.append("| Strategy | Category | Priority | Sharpe | Action |")
        lines.append("|---|---|---|---:|---|")

        for _, row in owner_df.iterrows():
            sharpe = row["sharpe"]
            sharpe_txt = "nan" if pd.isna(sharpe) else f"{sharpe:.2f}"
            action = (
                "rebuild_or_kill"
                if row["wave"] == "wave_1_kill_or_rebuild"
                else "high_urgency_rework"
            )
            lines.append(
                f"| {row['strategy']} | {row['category']} | {row['priority']} | {sharpe_txt} | {action} |"
            )

        lines.append("")

    lines.append("## Suggested Batch Order")
    lines.append("1. Run kill_or_rebuild set first (fast triage)")
    lines.append("2. Run high_urgency set by owner in parallel")
    lines.append("3. Consolidate with pytest + targeted backtests")
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
