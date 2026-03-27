from __future__ import annotations

import ast
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FULL_PERIOD_CSV = ROOT / "scripts" / "results" / "backtest_Full_Period_2010-2025.csv"
WINDOW_GLOB = "backtest_*.csv"
STRATEGY_DIR = ROOT / "src" / "financial_algo" / "strategies"

OUT_BASELINE = ROOT / "results" / "sprint15_phase12_baseline.csv"
OUT_DOSSIER = ROOT / "results" / "sprint15_phase12_dossier.json"
OUT_COMPREHENSIVE_CSV = ROOT / "results" / "sprint15_phase12_comprehensive.csv"
OUT_COMPREHENSIVE_JSON = ROOT / "results" / "sprint15_phase12_comprehensive.json"

INDICATOR_KEYWORDS = {
    "sma": ["sma", "rolling("],
    "ema": ["ema", "ewm("],
    "rsi": ["rsi"],
    "bollinger": ["bollinger", "bb_"],
    "zscore": ["z_score", "zscore", "z-score"],
    "atr": ["atr"],
    "volatility": ["realized_vol", "volatility", "std("],
    "momentum": ["momentum", "pct_change", "return_"],
    "regime": ["regime", "vix", "crisis"],
}

SIGNAL_FAMILIES = {
    "trend": ["trend", "breakout", "momentum", "ma_", "moving average", "cross"],
    "mean_reversion": ["mean reversion", "reversion", "zscore", "rsi"],
    "carry": ["carry", "term structure", "roll"],
    "volatility": ["vol", "vix", "gamma"],
    "pairs_statarb": ["pair", "cointegration", "spread", "residual"],
    "macro_regime": ["macro", "rates", "credit", "dollar", "inflation"],
    "ml": ["xgboost", "classifier", "model", "predict", "lstm", "cnn", "attention"],
}

RESEARCH_LIBRARY = {
    "trend": {
        "evidence": "Time-series momentum has persistent cross-asset alpha when volatility-scaled and crash-managed.",
        "refs": [
            "Moskowitz, Ooi, Pedersen (2012) - Time Series Momentum",
            "Hurst, Ooi, Pedersen (2017) - A Century of Trend Following",
        ],
        "ideas": [
            "Add volatility scaling and turnover-aware rebalance cadence.",
            "Use regime filter to reduce whipsaw in high-vol mean-reverting states.",
        ],
    },
    "mean_reversion": {
        "evidence": "Short-horizon reversal improves with liquidity and volatility-state filters.",
        "refs": [
            "Lehmann (1990) - Fads, Martingales, and Market Efficiency",
            "Avellaneda, Lee (2010) - Statistical Arbitrage in US Equities",
        ],
        "ideas": [
            "Gate entries by spread/volatility regime and avoid trend breakout states.",
            "Use adaptive z-score thresholds from rolling quantiles.",
        ],
    },
    "carry": {
        "evidence": "Carry premia are structurally positive but crash-prone and require convexity overlays.",
        "refs": [
            "Koijen et al. (2018) - Carry",
            "Baltas (2019) - Alternative Risk Premia",
        ],
        "ideas": [
            "Add drawdown-aware de-risking and convex hedges in stress regimes.",
            "Use term-structure slope confidence to modulate position size.",
        ],
    },
    "volatility": {
        "evidence": "Volatility risk premium is robust but negatively skewed; timing and tail controls are essential.",
        "refs": [
            "Carr, Wu (2009) - Variance Risk Premia",
            "Israelov, Nielsen (2015) - Volatility Strategies",
        ],
        "ideas": [
            "Blend carry and momentum in vol signals to avoid left-tail episodes.",
            "Constrain gross exposure during abrupt vol-of-vol spikes.",
        ],
    },
    "pairs_statarb": {
        "evidence": "Pairs alpha decays without robust pair selection and structural-break checks.",
        "refs": [
            "Gatev, Goetzmann, Rouwenhorst (2006) - Pairs Trading",
            "Do and Faff (2010) - Pairs Performance and Risks",
        ],
        "ideas": [
            "Refresh pair universe with stability and half-life constraints.",
            "Use stop-loss and spread-regime controls for divergence risk.",
        ],
    },
    "macro_regime": {
        "evidence": "Macro allocation improves when signals are ensemble-averaged and conditioned on regimes.",
        "refs": [
            "Ilmanen (2011) - Expected Returns",
            "Ang (2014) - Asset Management: A Systematic Approach",
        ],
        "ideas": [
            "Use multi-signal scoreboards rather than single-threshold toggles.",
            "Add explicit risk-off fallback assets with confidence weighting.",
        ],
    },
    "ml": {
        "evidence": "ML alpha is strongest when feature drift is controlled and outputs are uncertainty-aware.",
        "refs": [
            "Gu, Kelly, Xiu (2020) - Empirical Asset Pricing via ML",
            "Lopez de Prado (2018) - Advances in Financial ML",
        ],
        "ideas": [
            "Use walk-forward retraining and confidence-threshold abstention.",
            "Add model-risk overlay: cap exposure when prediction entropy is high.",
        ],
    },
}


def parse_percent_or_float(raw: str) -> float:
    text = (raw or "").strip()
    if not text:
        return 0.0
    try:
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)
    except ValueError:
        return 0.0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_priority(sharpe: float, crisis_non_neg_count: int) -> str:
    if sharpe < 0.0:
        return "kill_or_full_rework"
    if sharpe < 0.30:
        return "high_priority_rework"
    if sharpe < 0.60 or crisis_non_neg_count < 4:
        return "watch_and_rework"
    return "optimize_winner"


def detect_driver_scores(indicators: list[str], families: list[str], trades_per_mo: float) -> dict[str, float]:
    scores = {
        "trend_driver": 0.0,
        "reversion_driver": 0.0,
        "carry_driver": 0.0,
        "vol_driver": 0.0,
        "macro_driver": 0.0,
        "ml_driver": 0.0,
    }

    if "trend" in families:
        scores["trend_driver"] += 1.0
    if "mean_reversion" in families:
        scores["reversion_driver"] += 1.0
    if "carry" in families:
        scores["carry_driver"] += 1.0
    if "volatility" in families:
        scores["vol_driver"] += 1.0
    if "macro_regime" in families:
        scores["macro_driver"] += 1.0
    if "ml" in families:
        scores["ml_driver"] += 1.0

    if "momentum" in indicators:
        scores["trend_driver"] += 0.5
    if "zscore" in indicators or "rsi" in indicators:
        scores["reversion_driver"] += 0.5
    if "volatility" in indicators:
        scores["vol_driver"] += 0.5
    if "regime" in indicators:
        scores["macro_driver"] += 0.4

    if trades_per_mo > 12.0:
        scores["reversion_driver"] += 0.2
        scores["ml_driver"] += 0.1
    elif trades_per_mo < 3.0:
        scores["trend_driver"] += 0.2
        scores["macro_driver"] += 0.2

    return scores


def summarize_mechanism(families: list[str], indicators: list[str], pair_universe: list[str]) -> str:
    fam_text = ", ".join(families) if families else "mixed signals"
    ind_text = ", ".join(indicators) if indicators else "custom transforms"
    uni_text = ", ".join(pair_universe[:8]) if pair_universe else "broad ETF universe"
    return (
        f"Uses {fam_text} with {ind_text}. "
        f"Primary tradable universe proxy: {uni_text}."
    )


def build_research_cross_reference(families: list[str]) -> dict[str, Any]:
    refs: list[str] = []
    ideas: list[str] = []
    evidence: list[str] = []
    for fam in families:
        lib = RESEARCH_LIBRARY.get(fam)
        if not lib:
            continue
        refs.extend(lib["refs"])
        ideas.extend(lib["ideas"])
        evidence.append(lib["evidence"])

    uniq_refs = list(dict.fromkeys(refs))
    uniq_ideas = list(dict.fromkeys(ideas))
    uniq_evidence = list(dict.fromkeys(evidence))
    return {
        "research_evidence": uniq_evidence[:3],
        "research_refs": uniq_refs[:4],
        "improvement_ideas": uniq_ideas[:4],
    }


def extract_strategy_classes(file_path: Path) -> list[dict[str, Any]]:
    src = file_path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    out: list[dict[str, Any]] = []

    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        strategy_name = ""
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name) and tgt.id == "name" and isinstance(stmt.value, ast.Constant):
                if isinstance(stmt.value.value, str):
                    strategy_name = stmt.value.value

        if not strategy_name:
            continue

        class_start = max(node.lineno - 1, 0)
        class_end = max(node.end_lineno or node.lineno, node.lineno)
        class_src = "\n".join(lines[class_start:class_end]).lower()

        indicators: list[str] = []
        for key, needles in INDICATOR_KEYWORDS.items():
            if any(n in class_src for n in needles):
                indicators.append(key)

        families: list[str] = []
        for fam, needles in SIGNAL_FAMILIES.items():
            if any(n in class_src for n in needles):
                families.append(fam)

        pair_universe = sorted(
            {
                t
                for t in re.findall(r"['\"]([A-Z]{2,6}(?:-USD)?|\^VIX)['\"]", class_src)
                if len(t) > 1
            }
        )

        out.append(
            {
                "strategy": strategy_name,
                "class_name": node.name,
                "file": str(file_path.relative_to(ROOT)).replace("\\", "/"),
                "indicators": indicators,
                "signal_families": families,
                "pair_universe": pair_universe[:30],
            }
        )

    return out


def build_source_map() -> dict[str, dict[str, Any]]:
    source_map: dict[str, dict[str, Any]] = {}
    for py in STRATEGY_DIR.glob("*.py"):
        if py.name in {"base.py", "__init__.py", "intraday_base.py"}:
            continue
        try:
            for row in extract_strategy_classes(py):
                source_map[row["strategy"]] = row
        except SyntaxError:
            continue
    return source_map


def main() -> None:
    full_rows = read_csv_rows(FULL_PERIOD_CSV)
    window_files = sorted((ROOT / "scripts" / "results").glob(WINDOW_GLOB))
    window_files = [p for p in window_files if p.name != FULL_PERIOD_CSV.name]

    window_metrics: dict[str, list[float]] = defaultdict(list)
    for wf in window_files:
        for row in read_csv_rows(wf):
            strat = row.get("Strategy", "")
            if not strat or strat == "BuyHold-SPY":
                continue
            window_metrics[strat].append(parse_percent_or_float(row.get("Sharpe", "0")))

    source_map = build_source_map()

    baseline_rows: list[dict[str, Any]] = []
    dossiers: list[dict[str, Any]] = []
    comprehensive_rows: list[dict[str, Any]] = []

    for row in full_rows:
        strat = row.get("Strategy", "")
        if not strat:
            continue

        sharpe = parse_percent_or_float(row.get("Sharpe", "0"))
        cagr = parse_percent_or_float(row.get("CAGR", "0"))
        sortino = parse_percent_or_float(row.get("Sortino", "0"))
        max_dd = parse_percent_or_float(row.get("MaxDD", "0"))
        calmar = parse_percent_or_float(row.get("Calmar", "0"))
        ann_vol = parse_percent_or_float(row.get("AnnVol", "0"))
        win_rate = parse_percent_or_float(row.get("WinRate", "0"))
        tot_ret = parse_percent_or_float(row.get("TotRet", "0"))
        trades_per_mo = parse_percent_or_float(row.get("Trades/Mo", "0"))

        crisis_sharpes = window_metrics.get(strat, [])
        crisis_non_neg = sum(1 for x in crisis_sharpes if x >= 0.0)
        crisis_avg_sharpe = sum(crisis_sharpes) / len(crisis_sharpes) if crisis_sharpes else 0.0

        priority = classify_priority(sharpe, crisis_non_neg)

        base = {
            "category": row.get("Category", ""),
            "strategy": strat,
            "cagr": round(cagr, 6),
            "sharpe": round(sharpe, 6),
            "sortino": round(sortino, 6),
            "max_drawdown": round(max_dd, 6),
            "calmar": round(calmar, 6),
            "ann_vol": round(ann_vol, 6),
            "win_rate": round(win_rate, 6),
            "total_return": round(tot_ret, 6),
            "trades_per_month": round(trades_per_mo, 6),
            "crisis_non_negative_count": crisis_non_neg,
            "crisis_avg_sharpe": round(crisis_avg_sharpe, 6),
            "priority": priority,
        }
        baseline_rows.append(base)

        src = source_map.get(strat, {})
        indicators = src.get("indicators", [])
        families = src.get("signal_families", [])
        pair_universe = src.get("pair_universe", [])
        mechanism = summarize_mechanism(families, indicators, pair_universe)
        driver_scores = detect_driver_scores(indicators, families, trades_per_mo)
        top_drivers = sorted(driver_scores.items(), key=lambda kv: kv[1], reverse=True)
        cross_ref = build_research_cross_reference(families)

        dossier = {
            **base,
            "class_name": src.get("class_name", "unknown"),
            "source_file": src.get("file", "unknown"),
            "indicators": indicators,
            "signal_families": families,
            "pair_universe": pair_universe,
            "how_it_works": mechanism,
            "drivers_ranked": [f"{k}:{round(v, 2)}" for k, v in top_drivers if v > 0.0],
            "research_evidence": cross_ref["research_evidence"],
            "research_refs": cross_ref["research_refs"],
            "improvement_ideas": cross_ref["improvement_ideas"],
            "research_focus": [f"Improve {fam} edge robustness" for fam in families[:3]],
        }
        dossiers.append(dossier)

        comprehensive_rows.append(
            {
                "category": base["category"],
                "strategy": base["strategy"],
                "priority": base["priority"],
                "sharpe": base["sharpe"],
                "cagr": base["cagr"],
                "max_drawdown": base["max_drawdown"],
                "crisis_non_negative_count": base["crisis_non_negative_count"],
                "source_file": dossier["source_file"],
                "how_it_works": dossier["how_it_works"],
                "drivers_ranked": " | ".join(dossier["drivers_ranked"]),
                "indicators": " | ".join(indicators),
                "signal_families": " | ".join(families),
                "pair_universe": " | ".join(pair_universe[:12]),
                "research_refs": " | ".join(dossier["research_refs"]),
                "improvement_ideas": " | ".join(dossier["improvement_ideas"]),
            }
        )

    baseline_rows.sort(key=lambda x: x["sharpe"])
    dossiers.sort(key=lambda x: x["sharpe"])
    comprehensive_rows.sort(key=lambda x: x["sharpe"])

    OUT_BASELINE.parent.mkdir(parents=True, exist_ok=True)
    with OUT_BASELINE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        writer.writeheader()
        writer.writerows(baseline_rows)

    with OUT_DOSSIER.open("w", encoding="utf-8") as f:
        json.dump(dossiers, f, indent=2)

    with OUT_COMPREHENSIVE_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(comprehensive_rows[0].keys()))
        writer.writeheader()
        writer.writerows(comprehensive_rows)

    with OUT_COMPREHENSIVE_JSON.open("w", encoding="utf-8") as f:
        json.dump(comprehensive_rows, f, indent=2)

    print(f"Wrote baseline snapshot: {OUT_BASELINE}")
    print(f"Wrote strategy dossier: {OUT_DOSSIER}")
    print(f"Wrote comprehensive cross-reference CSV: {OUT_COMPREHENSIVE_CSV}")
    print(f"Wrote comprehensive cross-reference JSON: {OUT_COMPREHENSIVE_JSON}")
    print(f"Strategies covered: {len(dossiers)}")


if __name__ == "__main__":
    main()
