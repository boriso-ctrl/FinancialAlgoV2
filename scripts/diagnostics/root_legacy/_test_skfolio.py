"""Quick skfolio integration test."""
import time
import numpy as np
import pandas as pd
from skfolio.optimization import HierarchicalRiskParity, NestedClustersOptimization
from skfolio import RiskMeasure

np.random.seed(42)
n_days = 504
n_strats = 26  # match our 26 ensemble members
returns = pd.DataFrame(
    np.random.randn(n_days, n_strats) * 0.01,
    columns=[f"strat_{i}" for i in range(n_strats)],
    index=pd.bdate_range("2023-01-01", periods=n_days),
)

# --- Benchmark HRP ---
hrp = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
window = 60

t0 = time.perf_counter()
n_fits = 0
for end in range(window, window + 50):
    hrp.fit(returns.iloc[end - window : end])
    n_fits += 1
elapsed = time.perf_counter() - t0
print(f"HRP: {n_fits} fits in {elapsed:.3f}s = {elapsed/n_fits*1000:.1f}ms/fit")
print(f"  Extrapolated for 3800 trading days: {elapsed/n_fits * 3800:.1f}s")

# --- Benchmark NCO ---
nco = NestedClustersOptimization()
t0 = time.perf_counter()
n_fits = 0
for end in range(window, window + 50):
    nco.fit(returns.iloc[end - window : end])
    n_fits += 1
elapsed = time.perf_counter() - t0
print(f"NCO: {n_fits} fits in {elapsed:.3f}s = {elapsed/n_fits*1000:.1f}ms/fit")
print(f"  Extrapolated for 3800 trading days: {elapsed/n_fits * 3800:.1f}s")

# --- NaN handling ---
returns_nan = returns.copy()
returns_nan.iloc[0:5, 3] = np.nan
try:
    hrp.fit(returns_nan.iloc[:60])
    print(f"NaN: HRP survived, weights sum={hrp.weights_.sum():.4f}")
except Exception as e:
    print(f"NaN: HRP FAILED - {type(e).__name__}: {e}")

# --- max_weights constraint ---
hrp_capped = HierarchicalRiskParity(
    risk_measure=RiskMeasure.VARIANCE,
    max_weights=0.15,
)
hrp_capped.fit(returns.iloc[:120])
print(f"max_weights=0.15: max={hrp_capped.weights_.max():.4f}")

# --- Different risk measures ---
for rm in [RiskMeasure.VARIANCE, RiskMeasure.CVAR, RiskMeasure.STANDARD_DEVIATION]:
    h = HierarchicalRiskParity(risk_measure=rm)
    h.fit(returns.iloc[:120])
    print(f"  {rm.value}: weights std={h.weights_.std():.4f}")

print("DONE")
