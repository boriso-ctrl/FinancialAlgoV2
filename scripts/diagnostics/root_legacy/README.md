# Root Legacy Scripts

These files were moved from repository root during cleanup to keep the root focused on package code, docs, and top-level configs.

## Why this folder exists

- Preserves historical diagnostic and scratch scripts.
- Keeps root namespace clean.
- Avoids accidental execution/import confusion.

## Moved script groups

- NFI and extraction diagnostics: `_analyze_nfi*.py`, `_extract_*.py`, `_nfi_x5_temp.py`
- Temporary evaluation and profiling: `_tmp_eval_*.py`, `_profile_backtest.py`, `_eval_l9_l10.py`, `_check_v10.py`
- Ad-hoc runners/tests: `_run_macro_rework_test.py`, `_test_rework_strategies.py`, `_test_skfolio*.py`, `debug_check.py`, `run_fresh_test.py`

If a script here should become part of maintained tooling, promote it into `scripts/diagnostics/` or `scripts/production/`.
