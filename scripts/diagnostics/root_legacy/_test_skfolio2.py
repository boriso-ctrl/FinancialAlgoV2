"""Quick targeted skfolio tests."""
import time, sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

from skfolio.optimization import HierarchicalRiskParity, NestedClustersOptimization
from skfolio import RiskMeasure

np.random.seed(42)
returns = pd.DataFrame(
    np.random.randn(504, 26) * 0.01,
    columns=[f"s{i}" for i in range(26)],
    index=pd.bdate_range("2023-01-01", periods=504),
)

# NCO: just 5 fits
print("Testing NCO speed...", flush=True)
nco = NestedClustersOptimization()
t0 = time.perf_counter()
for end in range(60, 65):
    nco.fit(returns.iloc[end - 60 : end])
elapsed = time.perf_counter() - t0
print(f"NCO: 5 fits in {elapsed:.2f}s = {elapsed/5*1000:.0f}ms/fit", flush=True)

# NaN
print("Testing NaN handling...", flush=True)
ret_nan = returns.copy()
ret_nan.iloc[0:5, 3] = np.nan
hrp = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE)
try:
    hrp.fit(ret_nan.iloc[:60])
    print(f"NaN: survived, sum={hrp.weights_.sum():.4f}", flush=True)
except Exception as e:
    print(f"NaN: FAILED - {e}", flush=True)

# max_weights
hrp2 = HierarchicalRiskParity(risk_measure=RiskMeasure.VARIANCE, max_weights=0.15)
hrp2.fit(returns.iloc[:120])
print(f"max_weights=0.15: max={hrp2.weights_.max():.4f}", flush=True)

print("ALL DONE", flush=True)
