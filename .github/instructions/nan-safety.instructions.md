---
description: "Use when: writing or editing Python code that handles NaN, inf, or missing data in pandas/numpy. Enforces NaN-aware coding rules to prevent silent failures in financial calculations."
applyTo: "**/*.py"
---

# NaN-Safety Rules — FinancialAlgoV2

These rules prevent the most common class of silent bugs in quantitative code: NaN and inf propagation causing wrong answers without any error or warning.

## Mandatory Patterns

### After `pct_change()` or `diff()` — always `fillna(0.0)`

The first row(s) will be NaN. If used in subsequent comparisons or arithmetic, NaN propagates silently.

```python
# WRONG
returns = prices.pct_change()
signal = returns > 0  # First row is False (NaN > 0 is False), not missing

# RIGHT
returns = prices.pct_change().fillna(0.0)
```

### After division — always guard against inf

Division by zero produces `inf`/`-inf`, not an error. These propagate through sums and means.

```python
# WRONG
ratio = series_a / series_b

# RIGHT
ratio = series_a / series_b
ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

### Boolean comparisons with NaN — use explicit `notna()` guard

`NaN > threshold` evaluates to `False`, not to NaN or an error. This silently excludes rows.

```python
# WRONG — silently treats NaN as "not above threshold"
mask = signal > threshold

# RIGHT — explicitly separates missing from below-threshold
mask = pd.notna(signal) & (signal > threshold)
```

### After `.std()` on small windows — guard against NaN/zero

`.std()` on an empty or single-element Series returns `NaN`. Division by this produces `inf`.

```python
# WRONG
z_score = (x - x.mean()) / x.std()

# RIGHT
std = x.std()
z_score = (x - x.mean()) / std if std > 0 else 0.0

# Or for Series:
std = x.rolling(window).std().replace(0, np.nan)
z_score = ((x - x.rolling(window).mean()) / std).fillna(0.0)
```

### After `.shift()` — consider `fillna()`

`.shift(1)` introduces NaN at the start. If the shifted series feeds into arithmetic, guard it.

```python
# Use fillna when the NaN would cause downstream issues
prev_regime = regime.shift(1).fillna(regime.iloc[0])
```

### After `np.log()` — guard against log(0) and log(negative)

```python
# WRONG
log_returns = np.log(prices / prices.shift(1))

# RIGHT
price_ratio = prices / prices.shift(1)
price_ratio = price_ratio.clip(lower=1e-10)  # Prevent log(0) and log(neg)
log_returns = np.log(price_ratio)
```

## Final Weights Sanitization

Every `generate_weights()` method should sanitize its output before returning:

```python
weights = weights.replace([np.inf, -np.inf], np.nan).fillna(0.0)
```

This is the last line of defense against inf/NaN leaking into the backtester.

## Testing NaN Safety

When writing tests for strategies, include:

```python
def test_strategy_nan_safety(self):
    """Verify no NaN or inf in output weights."""
    prices_with_nan = prices.copy()
    prices_with_nan.iloc[0:5] = np.nan  # Leading NaN (common in real data)
    weights = strategy.generate_weights(prices_with_nan)
    assert not weights.isin([np.inf, -np.inf]).any().any(), "inf in weights"
    assert not weights.isna().any().any(), "NaN in weights"
```
