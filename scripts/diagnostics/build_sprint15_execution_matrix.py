"""
Build Sprint 15 execution matrix for all researched strategies.

Outputs:
- results/sprint15_execution_matrix.csv
- results/sprint15_execution_matrix.md

This script reconciles:
1) Phase 12 research catalog (all strategies)
2) Phase 13 implementation set (reworked subset)

It explicitly separates "researched" from "implemented" to avoid ambiguity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

COMPREHENSIVE_CSV = RESULTS_DIR / "sprint15_phase12_comprehensive.csv"
OUT_CSV = RESULTS_DIR / "sprint15_execution_matrix.csv"
OUT_MD = RESULTS_DIR / "sprint15_execution_matrix.md"


# Strategy names reported as implemented/reworked in Phase 13 team deliverables.
PHASE13_IMPLEMENTED = {
    # Crisis (Viktor)
    "C2-SafeHavenFlight",
    "B1-OilMomentumSurge",
    "H1-CommodityShockRider",
    "H4-MultiAssetCrisisLong",
    # Systematic (Sofia)
    "E1-MultiPairPortfolio",
    "I2-CrossSectionalMomentum",
    "J1-SectorMeanReversion",
    "J3-RSIMeanReversion",
    "K1-LowVolFactor",
    "N1-SeasonalStrategy",
    "P1-FeatureComboSignal",
    # Macro (Marcus)
    "H1-YieldCurveTrade",
    "H2-CreditSpreadMeanRev",
    "M1-DollarCarry",
    "M3-EMRiskPremium",
    "M4-CommodityMomentum",
    # Vol/ML (Vera)
    "L1-VolRiskPremium",
    "L7-ImpliedRealizedSpread",
    "P3-GMMRegimeClassifier",
    "G1-SentimentCrisisAlpha",
}

# Strategies explicitly triaged for kill after rework attempts.
PHASE13_KILL_CANDIDATES = {
    "E1-MultiPairPortfolio",
    "J3-RSIMeanReversion",
}

# Strategies explicitly marked as monitor/review.
PHASE13_MONITOR = {
    "L7-ImpliedRealizedSpread",
}


@dataclass(frozen=True)
class WaveRule:
    label: str
    predicate_desc: str


OWNER_BY_CATEGORY_PREFIX: List[tuple[str, str]] = [
    # Specific prefixes first to avoid overlaps (e.g., Cat DL vs Cat D).
    ("Cat H-FI", "macro"),
    ("Cat H: Fixed Income", "macro"),
    ("Cat MF", "systematic"),
    ("Cat DL", "vol"),
    ("Cat B", "crisis"),
    ("Cat C", "crisis"),
    ("Cat D", "crisis"),
    ("Cat F", "crisis"),
    ("Cat O", "crisis"),
    ("Cat E", "systematic"),
    ("Cat I", "systematic"),
    ("Cat J", "systematic"),
    ("Cat K", "systematic"),
    ("Cat N", "systematic"),
    ("Cat Q", "systematic"),
    ("Cat M", "macro"),
    ("Cat R", "macro"),
    ("Cat G", "vol"),
    ("Cat L", "vol"),
    ("Cat P", "vol"),
]


def assign_owner(category: str) -> str:
    for prefix, owner in OWNER_BY_CATEGORY_PREFIX:
        if category.startswith(prefix):
            return owner
    return "peter"


def normalize_num(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    return vals.replace([np.inf, -np.inf], np.nan)


def assign_wave(priority: str, sharpe: float, status: str) -> str:
    if status in {"implemented", "implemented_kill", "implemented_monitor"}:
        return "done"

    if priority == "kill_or_full_rework":
        return "wave_1_kill_or_rebuild"

    if priority == "high_priority_rework":
        if pd.notna(sharpe) and sharpe < 0.35:
            return "wave_1_high_urgency"
        return "wave_2_high_priority"

    if priority == "watch_and_rework":
        if pd.notna(sharpe) and sharpe < 0.55:
            return "wave_3_watch_low_sharpe"
        return "wave_4_watch_optimize"

    if priority == "optimize_winner":
        return "wave_5_optimize_winners"

    return "wave_backlog"


def build_matrix() -> pd.DataFrame:
    df = pd.read_csv(COMPREHENSIVE_CSV)

    df["sharpe"] = normalize_num(df.get("sharpe", pd.Series(dtype=float)))
    df["cagr"] = normalize_num(df.get("cagr", pd.Series(dtype=float)))
    df["max_drawdown"] = normalize_num(df.get("max_drawdown", pd.Series(dtype=float)))

    df["owner"] = df["category"].astype(str).map(lambda c: assign_owner(c))

    def _status(strategy: str) -> str:
        if strategy in PHASE13_KILL_CANDIDATES:
            return "implemented_kill"
        if strategy in PHASE13_MONITOR:
            return "implemented_monitor"
        if strategy in PHASE13_IMPLEMENTED:
            return "implemented"
        return "researched_only"

    df["execution_status"] = df["strategy"].astype(str).map(_status)

    df["wave"] = [
        assign_wave(priority=str(p), sharpe=s, status=st)
        for p, s, st in zip(df["priority"], df["sharpe"], df["execution_status"])
    ]

    df["effort"] = np.where(
        df["priority"].isin(["kill_or_full_rework", "high_priority_rework"]),
        "medium_high",
        np.where(df["priority"].eq("watch_and_rework"), "medium", "low_medium"),
    )

    df["expected_sharpe_uplift"] = np.where(
        df["priority"].eq("kill_or_full_rework"),
        ">=0.25_or_kill",
        np.where(
            df["priority"].eq("high_priority_rework"),
            ">=0.15",
            np.where(df["priority"].eq("watch_and_rework"), ">=0.10", ">=0.05"),
        ),
    )

    out_cols = [
        "category",
        "strategy",
        "owner",
        "priority",
        "execution_status",
        "wave",
        "effort",
        "expected_sharpe_uplift",
        "sharpe",
        "cagr",
        "max_drawdown",
        "crisis_non_negative_count",
        "improvement_ideas",
    ]

    matrix = df[out_cols].copy()
    matrix = matrix.sort_values(
        by=["wave", "priority", "sharpe"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)

    return matrix


def write_markdown(matrix: pd.DataFrame) -> None:
    total = len(matrix)
    implemented = int(matrix["execution_status"].str.startswith("implemented").sum())
    researched_only = int((matrix["execution_status"] == "researched_only").sum())

    by_owner = matrix.groupby("owner", dropna=False)["strategy"].count().sort_values(ascending=False)
    by_wave = matrix.groupby("wave", dropna=False)["strategy"].count().sort_values(ascending=False)

    lines: List[str] = []
    lines.append("# Sprint 15 Execution Matrix")
    lines.append("")
    lines.append("## Coverage")
    lines.append(f"- Total researched strategies: {total}")
    lines.append(f"- Implemented/reworked in Phase 13: {implemented}")
    lines.append(f"- Researched only (not implemented yet): {researched_only}")
    lines.append("")
    lines.append("## By Owner")
    for owner, count in by_owner.items():
        lines.append(f"- {owner}: {count}")
    lines.append("")
    lines.append("## By Wave")
    for wave, count in by_wave.items():
        lines.append(f"- {wave}: {count}")
    lines.append("")

    top_wave2 = matrix[matrix["wave"] == "wave_2_high_priority"].head(15)
    if not top_wave2.empty:
        lines.append("## Next Wave Candidates (Top 15 from wave_2_high_priority)")
        lines.append("")
        lines.append("| Strategy | Owner | Priority | Sharpe |")
        lines.append("|---|---|---|---:|")
        for _, row in top_wave2.iterrows():
            sharpe = row["sharpe"]
            sharpe_txt = "nan" if pd.isna(sharpe) else f"{sharpe:.2f}"
            lines.append(
                f"| {row['strategy']} | {row['owner']} | {row['priority']} | {sharpe_txt} |"
            )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    matrix = build_matrix()
    matrix.to_csv(OUT_CSV, index=False)
    write_markdown(matrix)

    implemented = int(matrix["execution_status"].str.startswith("implemented").sum())
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_MD}")
    print(f"Strategies tracked: {len(matrix)}")
    print(f"Implemented set tracked: {implemented}")


if __name__ == "__main__":
    main()
