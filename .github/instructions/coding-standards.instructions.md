---
description: "Use when: writing or editing Python code. Enforces hedge fund coding standards for performance, correctness, and maintainability in the FinancialAlgoV2 codebase."
applyTo: "**/*.py"
---

# Python Coding Standards — FinancialAlgoV2

## Correctness

- **Never compare pandas Series to Enum with `==`** — use `.isin()` or compare `.value`. The `Regime(str, Enum)` bug silently returns all False.
- **Never shift signals twice** — `backtest_weights()` already shifts +1 day. Do not shift again in `backtest()` or `_vol_target_overlay()`.
- **Handle NaN explicitly** — use `fillna()` or `dropna()` before comparisons. `NaN != NaN` causes silent logic errors.
- **Use `.loc[]` for assignment** — avoid chained indexing (`df[col][mask] = val`) which may write to a copy.

## Performance

- **Vectorize over loops** — never iterate rows of a DataFrame when a vectorized pandas/numpy operation exists.
- **Avoid redundant computation** — if an indicator (SMA, RSI, vol) is used by multiple strategies, compute it once and pass it through.
- **Use appropriate dtypes** — `float32` for large price arrays when precision allows. Categorical for regime columns.
- **Prefer `np.where()` over `apply()`** for conditional column creation.

## Style

- **ASCII-only in print/log statements** — the Windows cp1252 terminal chokes on Unicode symbols. No fancy arrows, bullets, or emoji.
- **Imports**: standard library, then third-party (`numpy`, `pandas`), then local (`financial_algo.*`). One blank line between groups.
- **Naming**: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- **Max function length**: ~50 lines. Decompose longer functions into named helpers.
- **No dead code** — remove commented-out blocks and unused imports before committing.

## Patterns

- All strategies inherit from `BaseStrategy` in `strategies/base.py`.
- Use the `Regime` enum from `regimes.py` for regime comparisons — never hardcode string values.
- Use `.venv\Scripts\python.exe` (Windows) to run Python — never bare `python`.
