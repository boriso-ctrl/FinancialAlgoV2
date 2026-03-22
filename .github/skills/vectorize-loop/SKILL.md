---
name: vectorize-loop
description: "Use when: replacing Python for-loops over DataFrames with vectorized pandas/numpy operations, optimizing slow row-by-row iteration, converting iterrows/itertuples/apply(lambda) to vectorized code, speeding up strategy generate_weights methods, eliminating for-i-in-range-len loops in financial code. Pattern library for loop vectorization."
argument-hint: "File path and function name containing the loop to vectorize, e.g. 'src/financial_algo/strategies/momentum.py generate_weights'"
---

# Vectorize Loop — Pattern Library for pandas/numpy Vectorization

A catalog of battle-tested recipes for replacing common Python loop patterns found in quantitative strategy code with equivalent vectorized operations.

## When to Use

- You see `for i in range(len(df))` in a strategy file
- `iterrows()`, `itertuples()`, or `apply(lambda ...)` appears in hot paths
- A backtest or signal generation function is slow and profiles show time in Python loops
- Felix flags a vectorization opportunity during an audit

## Procedure

1. **Identify the loop pattern** — read the loop body and classify it using the patterns below.
2. **Select the matching recipe** — find the vectorized equivalent.
3. **Apply the transformation** — rewrite the code using the recipe.
4. **Verify correctness** — run the relevant test to confirm identical output.
5. **Benchmark** — time before and after; report the speedup.

## Pattern Recipes

### Recipe 1: Conditional Assignment (if/else per row)

**Loop pattern:**
```python
for i in range(len(df)):
    if df['signal'].iloc[i] > threshold:
        df.loc[df.index[i], 'weight'] = 1.0
    else:
        df.loc[df.index[i], 'weight'] = 0.0
```

**Vectorized:**
```python
df['weight'] = np.where(df['signal'] > threshold, 1.0, 0.0)
```

**Multi-condition variant:**
```python
conditions = [
    df['signal'] > upper,
    df['signal'] < lower,
]
choices = [1.0, -1.0]
df['weight'] = np.select(conditions, choices, default=0.0)
```

---

### Recipe 2: Top-N / Bottom-N Ranking Selection

**Loop pattern:**
```python
for date in df.index:
    row = df.loc[date]
    sorted_tickers = row.sort_values(ascending=False)
    top_n = sorted_tickers.head(n).index
    weights.loc[date, top_n] = 1.0 / n
```

**Vectorized:**
```python
ranks = df.rank(axis=1, ascending=False)
mask = ranks <= n
weights = mask.astype(float).div(mask.sum(axis=1), axis=0).fillna(0.0)
```

---

### Recipe 3: Z-Score Position Tracker

**Loop pattern:**
```python
position = 0
for i in range(len(z_scores)):
    if z_scores.iloc[i] < -entry_z:
        position = 1
    elif z_scores.iloc[i] > entry_z:
        position = -1
    elif abs(z_scores.iloc[i]) < exit_z:
        position = 0
    positions.iloc[i] = position
```

**Vectorized (stateless approximation):**
```python
positions = np.where(z_scores < -entry_z, 1.0,
            np.where(z_scores > entry_z, -1.0,
            np.where(z_scores.abs() < exit_z, 0.0, np.nan)))
positions = pd.Series(positions, index=z_scores.index).ffill().fillna(0.0)
```

**Numba (when state persistence matters):**
```python
import numba

@numba.njit
def _zscore_positions(z: np.ndarray, entry: float, exit_z: float) -> np.ndarray:
    n = len(z)
    pos = np.empty(n)
    pos[0] = 0.0
    for i in range(n):
        if z[i] < -entry:
            pos[i] = 1.0
        elif z[i] > entry:
            pos[i] = -1.0
        elif abs(z[i]) < exit_z:
            pos[i] = 0.0
        elif i > 0:
            pos[i] = pos[i - 1]
        else:
            pos[i] = 0.0
    return pos

positions = pd.Series(_zscore_positions(z_scores.values, entry_z, exit_z),
                       index=z_scores.index)
```

---

### Recipe 4: Periodic Rebalance

**Loop pattern:**
```python
for i in range(len(df)):
    if i % rebalance_period == 0:
        weights.iloc[i] = compute_weights(df.iloc[:i+1])
    else:
        weights.iloc[i] = weights.iloc[i - 1]
```

**Vectorized:**
```python
# Create rebalance dates mask
rebal_mask = np.arange(len(df)) % rebalance_period == 0
# Compute weights only at rebalance points, then forward-fill
all_weights = compute_weights_vectorized(df)  # compute for all dates
weights = all_weights.where(rebal_mask).ffill().fillna(0.0)
```

**Alternative using groupby:**
```python
group_idx = np.arange(len(df)) // rebalance_period
weights = df.groupby(group_idx).transform('first')  # or apply custom logic
```

---

### Recipe 5: Rolling State Machine with Hysteresis

**Loop pattern:**
```python
state = 'off'
for i in range(len(df)):
    if state == 'off' and df['trigger'].iloc[i] > on_threshold:
        state = 'on'
    elif state == 'on' and df['trigger'].iloc[i] < off_threshold:
        state = 'off'
    states[i] = state
```

**Numba (preserves exact state semantics):**
```python
@numba.njit
def _hysteresis(trigger: np.ndarray, on_thresh: float, off_thresh: float) -> np.ndarray:
    n = len(trigger)
    state = np.zeros(n, dtype=np.int8)  # 0=off, 1=on
    for i in range(1, n):
        if state[i - 1] == 0 and trigger[i] > on_thresh:
            state[i] = 1
        elif state[i - 1] == 1 and trigger[i] < off_thresh:
            state[i] = 0
        else:
            state[i] = state[i - 1]
    return state

states = _hysteresis(df['trigger'].values, on_threshold, off_threshold)
```

---

### Recipe 6: Cumulative Max / Drawdown Tracking

**Loop pattern:**
```python
peak = equity[0]
drawdowns = []
for i in range(len(equity)):
    if equity[i] > peak:
        peak = equity[i]
    drawdowns.append((equity[i] - peak) / peak)
```

**Vectorized:**
```python
peak = equity.cummax()
drawdowns = (equity - peak) / peak
```

---

### Recipe 7: Cross-Sectional Normalization (row-wise)

**Loop pattern:**
```python
for date in df.index:
    row = df.loc[date]
    df.loc[date] = (row - row.mean()) / row.std()
```

**Vectorized:**
```python
row_mean = df.mean(axis=1)
row_std = df.std(axis=1).replace(0, np.nan)
df_normalized = df.sub(row_mean, axis=0).div(row_std, axis=0).fillna(0.0)
```

---

### Recipe 8: Expanding / Rolling Weighted Average

**Loop pattern:**
```python
for i in range(lookback, len(df)):
    window = df.iloc[i - lookback:i]
    result[i] = (window * weights_array).sum() / weights_array.sum()
```

**Vectorized:**
```python
result = df.rolling(lookback).apply(lambda x: np.dot(x, weights_array) / weights_array.sum(),
                                     raw=True)
```

**Faster with convolution (uniform or exponential weights):**
```python
# For EMA:
result = df.ewm(span=lookback).mean()
```

## Choosing the Right Approach

| Pattern | Best Approach | Speedup (typical) |
|---------|--------------|-------------------|
| Conditional assignment | `np.where` / `np.select` | 50-200x |
| Top-N selection | `.rank()` + `.where()` | 20-100x |
| Stateless position logic | `np.where` + `ffill` | 30-150x |
| Stateful position logic | `numba.njit` | 10-50x |
| Periodic rebalance | `groupby` or mask + `ffill` | 10-50x |
| Hysteresis state machine | `numba.njit` | 10-50x |
| Cumulative ops | `.cummax()`, `.cumsum()` | 100-500x |
| Cross-sectional norm | `.sub()` / `.div()` with axis | 50-200x |

## Verification Checklist

After vectorizing, confirm:

1. **Output matches** — `np.allclose(old_result, new_result, equal_nan=True)` on the full backtest period
2. **NaN handling preserved** — same NaN positions in output
3. **Edge cases** — empty input, single-row input, single-column input all work
4. **Benchmark recorded** — time both versions with `time.perf_counter()` and report the ratio
