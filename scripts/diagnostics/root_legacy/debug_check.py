import inspect
import financial_algo.strategies.volatility_strats as v

print("Module file:", v.__file__)
src = inspect.getsource(v.CrossAssetVolSignal.generate_weights)
for i, line in enumerate(src.split("\n")):
    if "fillna" in line or "ffill" in line:
        print(f"  Line {i}: {line}")
