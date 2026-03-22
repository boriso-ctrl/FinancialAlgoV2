---
name: strategy-audit
description: "Use when: reviewing a new or modified trading strategy, auditing strategy code for correctness, checking a strategy for common bugs (NaN, enum comparison, shift, vectorization), performing a pull request review on strategy files, validating strategy code quality before merge. Automated strategy code review checklist."
argument-hint: "Path to the strategy file to audit, e.g. src/financial_algo/strategies/momentum.py"
---

# Strategy Audit — Automated Code Review Checklist

A structured checklist for reviewing any strategy that inherits from `BaseStrategy`. Every item must be verified before a strategy is considered production-ready.

## When to Use

- A new strategy file is created
- An existing strategy is modified
- Reviewing a PR that touches `src/financial_algo/strategies/`
- Felix or Peter requests a strategy quality check

## Procedure

Read the strategy file, then evaluate each item in the checklist below. For each item, report **PASS**, **FAIL**, or **N/A** with a brief explanation.

### Checklist

#### 1. Empty DataFrame Handling
- Does `generate_weights()` handle an empty `prices` DataFrame gracefully?
- Does it return an empty DataFrame (not raise an exception) when given zero rows?
- **Check**: Look for early-return guards like `if prices.empty: return pd.DataFrame()`

#### 2. Regime Enum Comparison
- Does the code use `.isin([Regime.X, Regime.Y])` or compare `.value` for regime checks?
- **NEVER**: `regime == Regime.NORMAL` or `pd.Series == Regime.X` — this silently returns all False with the str mixin.
- **Check**: Search for `== Regime` or `!= Regime` — any match is a **P0 FAIL**.

#### 3. Vectorization
- Are there Python `for` loops iterating over DataFrame rows (`for i in range(len(df))`, `for idx, row in df.iterrows()`)?
- Can any of these loops be replaced with vectorized pandas/numpy operations?
- **Common replacements**:
  - `np.where(condition, val_a, val_b)` instead of row-by-row if/else
  - `.rank()` + `.where()` instead of looping to find top-N
  - `pd.Series.shift()` + cumulative ops instead of manual state tracking
- **Check**: Search for `for i in range`, `iterrows`, `itertuples`, `apply(lambda` — flag each for review.

#### 4. NaN Safety
- Is `NaN` handled before boolean comparisons? (`NaN > 0` is False, not an error)
- Are arithmetic results guarded against `inf`/`-inf`? (division by zero, log of zero)
- Is `.fillna()` called after `.pct_change()`, `.diff()`, or `.shift()`?
- **Check**: Search for `pct_change`, `/ `, `np.log` — verify NaN guards follow.

#### 5. Weight Bounds
- Are output weights bounded? No `inf` or `-inf` values?
- Is there a `clip()` or explicit bound on leverage?
- **Check**: Look for `.clip()`, `np.clip`, or manual min/max on the final weights.

#### 6. Single-Ticker Compatibility
- Does the strategy work when `prices` has only one column?
- Operations like `.rank(axis=1)` or cross-sectional z-scores may fail or produce NaN with a single ticker.
- **Check**: Trace the logic path with `prices.shape[1] == 1`.

#### 7. Test Coverage
- Is there at least one test in `tests/test_strategies.py` that exercises this strategy?
- Does the test cover: normal operation, empty input, single-ticker input?
- **Check**: Search `tests/` for the strategy class name.

#### 8. Signal Shift (Look-Ahead Bias)
- Does `generate_weights()` avoid using future data?
- Signals computed from today's price should only affect tomorrow's weight (handled by `backtest_weights()` shift).
- **Check**: Ensure no `.shift(-1)` or forward-looking operations inside `generate_weights()`.

#### 9. Naming & Structure
- Does the strategy inherit from `Strategy` (base class in `strategies/base.py`)?
- Is `name` set to a descriptive string?
- Is the class registered in `strategies/__init__.py`?

#### 10. Idempotency
- Does calling `generate_weights()` twice with the same inputs produce identical outputs?
- No hidden state, no random seeds without control, no global mutation.

## Output Format

```
## Strategy Audit: [ClassName] ([file_path])

| # | Check | Result | Notes |
|---|-------|--------|-------|
| 1 | Empty DataFrame handling | PASS/FAIL | ... |
| 2 | Regime enum comparison | PASS/FAIL | ... |
| 3 | Vectorization | PASS/FAIL | ... |
| 4 | NaN safety | PASS/FAIL | ... |
| 5 | Weight bounds | PASS/FAIL | ... |
| 6 | Single-ticker compat | PASS/FAIL | ... |
| 7 | Test coverage | PASS/FAIL | ... |
| 8 | Signal shift (no look-ahead) | PASS/FAIL | ... |
| 9 | Naming & structure | PASS/FAIL | ... |
| 10 | Idempotency | PASS/FAIL | ... |

### Summary
- **Score**: X/10 passed
- **Blockers**: [List P0 failures that must be fixed before merge]
- **Recommendations**: [P1-P3 items for improvement]
```
