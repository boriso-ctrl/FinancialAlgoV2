---
name: team-dispatch
description: "Use when: Peter needs to mobilize the full team (Viktor, Sofia, Marcus, Vera) to work on a directive in parallel, then consolidate results, run quality checks, and deliver an executive report to the boss. Orchestrates multi-agent sprint cycles."
argument-hint: "The boss's directive — what the team should accomplish this sprint, e.g. 'build 10 new strategies and fix the ensemble' or 'audit and improve all crisis strategies'"
---

# Team Dispatch — Full-Team Sprint Orchestration

Peter uses this skill to **mobilize all four department heads**, delegate domain-specific work packages, collect their outputs, run quality and performance checks, and deliver a single consolidated executive report to the boss.

## When to Use

- The boss gives a broad directive that spans multiple strategy domains
- Peter decides a coordinated team effort is needed (e.g., "build 10 new strategies", "fix all underperformers", "prepare for production")
- Any task where multiple department heads should work in parallel to maximize throughput
- Periodic team sprint cycles (weekly review, monthly rebalance prep, pre-launch hardening)

## Pre-Flight Checklist

Before dispatching the team, Peter MUST:

1. **Read the directive carefully** — understand exactly what the boss wants
2. **Review current state** — read the latest backtest results (`results/`) and known issues from Peter's own agent file
3. **Identify which heads are needed** — not every sprint requires all four; skip heads with no relevant work
4. **Decompose the directive into domain-specific work packages** — each head gets a clear, scoped task
5. **Set acceptance criteria** — define what "done" looks like for each work package (e.g., "Sharpe > 0.5", "all tests pass", "no FAIL in audit")

## Phase 1: Dispatch

Peter decomposes the boss's directive into work packages and dispatches each department head via `runSubagent`. Each dispatch MUST include:

### Dispatch Template (for each head)

```
DIRECTIVE FROM PETER — [Sprint Name]

## Your Assignment
[Specific, scoped task for this department head]

## Acceptance Criteria
- [Measurable criterion 1, e.g., "Strategy Sharpe > 0.5 over full period"]
- [Measurable criterion 2, e.g., "All 10 strategy-audit checks PASS"]
- [Measurable criterion 3, e.g., "Correlation with existing ensemble < 0.4"]

## Constraints
- Follow all coding standards (read .github/instructions/ files)
- Run pytest after any code changes
- Use vectorized operations only (no row loops)
- Report back with: what you built, backtest numbers, any blockers

## Context
[Relevant current state: existing strategies, known bugs, performance numbers]

## Deliverables
Return a structured report with:
1. What was implemented/changed (file paths, class names)
2. Backtest results table (CAGR, Sharpe, Sortino, Max DD, Calmar)
3. Any blockers or issues encountered
4. Recommendations for next steps
```

### Dispatch Order — PARALLEL BY DEFAULT

Peter dispatches **all independent agents simultaneously** using parallel `runSubagent` calls. The whole team works at the same time — that's the point. Only serialize when one agent's output is a dependency for another.

**Default (parallel):** Dispatch all relevant heads in a single batch of `runSubagent` calls:
- **Viktor** (`crisis`) — Crisis & tail risk strategies (categories B, C, D, F, O)
- **Sofia** (`systematic`) — Systematic alpha (categories E, I, J, K, N)
- **Marcus** (`macro`) — Macro & rates (categories H, M)
- **Vera** (`vol`) — Volatility & alt data (categories G, L, P)

**Serialize only when:** one head's work depends on another's output (e.g., Sofia needs Viktor's new regime detector before building momentum filters on top of it).

If the directive only affects some domains, skip irrelevant heads. If an agent returns empty output, re-dispatch immediately.

### Example Dispatches

**For a "build new strategies" sprint:**
- Viktor: "Build O1-TailRiskParity and O2-CrisisAlphaMomentum. Target Sharpe > 0.5, Max DD < 20%."
- Sofia: "Build I1-TimeSeriesMomentum and J1-SectorMeanReversion. Target Sharpe > 0.8, low correlation with D2."
- Marcus: "Build H1-YieldCurveTrade and H2-CreditSpreadMR. Target Sharpe > 0.6, Max DD < 12%."
- Vera: "Build L1-VIXTermStructure and L2-VolRiskPremium. Target Sharpe > 0.7."

**For a "fix underperformers" sprint:**
- Viktor: "B1-OilMomentumSurge has Sharpe -0.05. Diagnose and fix. Target Sharpe > 0.3."
- Sofia: "E1-MultiPairPortfolio has Sharpe -0.11. Rework the pair selection and z-score logic."
- Marcus: (skip — no macro strategies exist yet)
- Vera: (skip — no vol strategies exist yet)

## Phase 2: Collect & Validate

After each department head returns, Peter:

1. **Records their report** — saves key metrics and file changes
2. **Runs pytest** to verify no regressions: `.venv\Scripts\python.exe -m pytest tests/ -v`
3. **Runs a quick backtest** if new strategies were added: `.venv\Scripts\python.exe scripts/run_crisis_backtest.py`
4. **Dispatches Felix** (`felix`) for a code quality audit on any new/changed strategy files, using the `strategy-audit` skill
5. **Dispatches Benchmarker** (`benchmarker`) to compare metrics against baselines

### Validation Checklist

For each head's deliverables, verify:
- [ ] New strategy classes inherit from `BaseStrategy`
- [ ] Strategy registered in `strategies/__init__.py`
- [ ] `generate_weights()` returns proper DataFrame
- [ ] Pytest passes (zero failures)
- [ ] Backtest Sharpe meets acceptance criteria
- [ ] No look-ahead bias (no `.shift(-1)` in signal logic)
- [ ] No Regime enum comparison bugs (`== Regime.X`)
- [ ] Correlation with existing ensemble is acceptable (< 0.5 average pairwise)

## Phase 3: Executive Report

Peter compiles everything into a single executive report for the boss. This is the ONLY output the boss sees — make it count.

### Report Template

```markdown
# Team Sprint Report — [Sprint Name]
**Date**: [today]
**Directive**: [original boss directive, quoted]

## Executive Summary
[2-3 sentences: what was accomplished, headline metric improvements, any blockers]

## Team Deliverables

### Viktor (Crisis & Tail Risk)
- **Task**: [what was assigned]
- **Status**: DONE / PARTIAL / BLOCKED
- **Delivered**: [list of strategies built/fixed with file paths]
- **Results**:
  | Strategy | CAGR | Sharpe | Max DD | Status |
  |----------|------|--------|--------|--------|
  | ... | ... | ... | ... | ... |
- **Issues**: [any blockers or concerns]

### Sofia (Systematic Alpha)
[same format]

### Marcus (Macro & Rates)
[same format]

### Vera (Volatility & Alt Data)
[same format]

## Quality Gate
- **Pytest**: PASS / FAIL ([X] passed, [Y] failed)
- **Strategy Audit**: [X]/[Y] strategies scored 10/10
- **Benchmark Regression**: NONE / [list regressions]

## Updated Ensemble Performance
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| CAGR | X% | Y% | +Z% |
| Sharpe | X | Y | +Z |
| Max DD | X% | Y% | +Z% |
| Strategy Count | X | Y | +Z |

## Recommendations
1. [Next priority item]
2. [Second priority]
3. [Third priority]

## Next Sprint Candidates
- [Strategy or improvement that should be tackled next]
```

## Decision Rules

### When to Stop a Sprint Early
- A department head introduces a regression (Sharpe drops on existing strategy) → fix before continuing
- pytest has failures → must be resolved before dispatching next head
- Boss sends a priority interrupt → pause and address

### How to Handle Blockers
- If a head reports "blocked on data" → note it, skip, move to next head
- If a head reports "strategy doesn't work" → record the negative result, don't force it; suggest an alternative for next sprint
- If Felix finds P0 audit failures → fix them before including in ensemble

### Kill Criteria
- Any strategy with **Sharpe < 0** after a good-faith fix attempt → KILL IT. Remove from codebase.
- Any strategy with **Max DD > 40%** → requires mandatory risk review before inclusion
- Any strategy with **correlation > 0.7** with an existing ensemble member → redundant, do not add

## Anti-Patterns (DO NOT)

- **DO NOT** dispatch all heads with vague "improve things" tasks — be specific
- **DO NOT** skip the validation phase — every sprint must end with pytest + backtest
- **DO NOT** let heads introduce code that doesn't follow coding standards
- **DO NOT** report partial results to the boss — always compile the full picture
- **DO NOT** add strategies to the ensemble without checking correlation
- **DO NOT** skip Felix audit on new code — quality is non-negotiable
