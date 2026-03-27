# Sprint 15 Wave 2 Dispatch Pack

## Scope
- Candidate set size: 15
- Source: results/sprint15_execution_matrix.csv
- Selection: researched_only + (wave_1_kill_or_rebuild or wave_1_high_urgency)

## Global Acceptance Criteria
- Sharpe uplift target per rework: >= +0.15 (or kill if still < 0)
- No look-ahead bias; no shift(-1); no double-shift in backtest path
- NaN/inf-safe outputs from generate_weights
- Vectorized implementation only
- Pytest green after each batch

## Crisis Targets

| Strategy | Category | Priority | Sharpe | Action |
|---|---|---|---:|---|
| O8-VolatilityConvexity | Cat O: Tail Risk | high_priority_rework | 0.00 | high_urgency_rework |
| O6-TailHedgeOverlay | Cat O: Tail Risk | high_priority_rework | 0.18 | high_urgency_rework |
| B2-OilShockHedge | Cat B: Oil Crisis | high_priority_rework | 0.22 | high_urgency_rework |
| O4-BlackSwanInsurance | Cat O: Tail Risk | high_priority_rework | 0.28 | high_urgency_rework |
| O9-ATRCrisisAlpha | Cat O: Tail Risk | kill_or_full_rework | -0.03 | rebuild_or_kill |

## Systematic Targets

| Strategy | Category | Priority | Sharpe | Action |
|---|---|---|---:|---|
| K6-QualityMomentumComposite | Cat K: Factor | high_priority_rework | 0.02 | high_urgency_rework |
| J5-GlobalMeanReversion | Cat J: Mean Reversion | high_priority_rework | 0.24 | high_urgency_rework |
| I10-AdaptiveTrendFilter | Cat I: Momentum | kill_or_full_rework | -0.14 | rebuild_or_kill |
| I7-DriftRegimeMomentum | Cat I: Momentum | kill_or_full_rework | -0.06 | rebuild_or_kill |

## Macro Targets

| Strategy | Category | Priority | Sharpe | Action |
|---|---|---|---:|---|
| R10-CommodityMacroOverlay | Cat R: Regime Hardening | high_priority_rework | 0.09 | high_urgency_rework |
| M7-GlobalRotation | Cat M: Macro | high_priority_rework | 0.18 | high_urgency_rework |
| M11-AdaptiveMacroBlend | Cat M: Macro | high_priority_rework | 0.27 | high_urgency_rework |

## Vol Targets

| Strategy | Category | Priority | Sharpe | Action |
|---|---|---|---:|---|
| DL3-AttentionRanker | Cat DL: Deep Learning | high_priority_rework | 0.16 | high_urgency_rework |

## Peter Targets

| Strategy | Category | Priority | Sharpe | Action |
|---|---|---|---:|---|
| H2-GoldFearRally | Cat H: Crisis Spike (Upward) | high_priority_rework | 0.24 | high_urgency_rework |
| H3-DefenseSpikeBreakout | Cat H: Crisis Spike (Upward) | high_priority_rework | 0.26 | high_urgency_rework |

## Suggested Batch Order
1. Run kill_or_rebuild set first (fast triage)
2. Run high_urgency set by owner in parallel
3. Consolidate with pytest + targeted backtests

