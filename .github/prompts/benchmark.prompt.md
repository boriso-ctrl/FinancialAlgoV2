---
description: "Profile and benchmark a Python function or module to identify performance bottlenecks with before/after timing"
agent: "felix"
tools: [read, search, execute]
argument-hint: "Function or module to profile, e.g. 'backtest()' or 'src/financial_algo/signals.py'"
---

Profile the specified function or module for performance bottlenecks.

## Steps

1. **Identify the target** — read the function/module the user specified. Understand its inputs and call sites.
2. **Run cProfile** — execute a profiling run using `.venv\Scripts\python.exe -m cProfile -s cumulative` against a script that exercises the target code. If no obvious entry point exists, create a minimal profiling script.
3. **Analyze hotspots** — identify the top 10 functions by cumulative time. Focus on functions in `src/financial_algo/`.
4. **Measure baseline** — time the target function with `time.perf_counter()` around a representative call.
5. **Report findings** — present a table of hotspots with:
   - Function name and file
   - Total calls
   - Cumulative time
   - Per-call time
   - Optimization opportunity (vectorize, cache, skip, restructure)

## Output Format

```
## Benchmark Report: [target]

### Environment
- Data range: [X years]
- Tickers: [N]

### Top Hotspots
| Rank | Function | File | Calls | Cumulative (s) | Opportunity |
|------|----------|------|-------|-----------------|-------------|
| 1    | ...      | ...  | ...   | ...             | ...         |

### Recommendations
1. [Specific optimization with estimated impact]
```
