#!/usr/bin/env python3
import json
from pathlib import Path

data = json.loads(Path("results/sprint15_phase12_comprehensive.json").read_text())
macro_strats = [s for s in data if s['strategy'].startswith(('H1-', 'H2-', 'M1-', 'M2-', 'M3-', 'M4-', 'R10-'))]

print("=" * 100)
print("MACRO & RATES STRATEGY BASELINE METRICS")
print("=" * 100)

for s in sorted(macro_strats, key=lambda x: x['sharpe']):
    print(f"{s['strategy']:30s} | Sharpe: {s['sharpe']:7.4f} | CAGR: {s['cagr']:7.4f} | DD: {s['max_drawdown']:7.4f} | Priority: {s['priority']}")

print("\n" + "=" * 100)
print(f"Total macro strategies: {len(macro_strats)}")
