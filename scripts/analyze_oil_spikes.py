"""Analyze oil price spikes in 2023-2025 period for Middle East conflict."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from financial_algo.data.loader import load_prices

p = load_prices(["USO", "XLE", "GLD", "ITA", "SPY"], start="2023-09-01", end="2025-12-31")

uso = p["USO"]
xle = p["XLE"]
gld = p["GLD"]
ita = p["ITA"]
spy = p["SPY"]

print("=" * 70)
print("OIL (USO) MONTHLY RETURNS  — Finding Middle East Spikes")
print("=" * 70)
monthly = uso.resample("ME").last().pct_change()
for dt, ret in monthly.items():
    if pd.isna(ret):
        continue
    marker = ""
    if ret > 0.05:
        marker = "  <<< SPIKE"
    elif ret < -0.08:
        marker = "  <<< DROP"
    print(f"  {dt.strftime('%Y-%m'):>7s}: {ret:+7.1%}{marker}")

print()
print("=" * 70)
print("BIG 10-DAY MOVES IN OIL (USO)")
print("=" * 70)
uso_10d = uso.pct_change(10)
big = uso_10d[uso_10d.abs() > 0.06].dropna()
for dt, ret in big.items():
    d = "UP" if ret > 0 else "DOWN"
    print(f"  {dt.strftime('%Y-%m-%d')}: {ret:+7.1%} ({d})")

print()
print("=" * 70)
print("ASSET PERFORMANCE DURING KEY MIDDLE EAST WINDOWS")
print("=" * 70)

windows = {
    "Hamas Attack (Oct 2023)":           ("2023-10-01", "2023-11-15"),
    "Iran-Israel Escalation (Apr 2024)": ("2024-04-01", "2024-04-30"),
    "Red Sea / Houthi (Jan-Mar 2024)":   ("2024-01-01", "2024-03-31"),
    "Iran Tensions (Sep-Oct 2024)":      ("2024-09-01", "2024-10-31"),
    "Full ME Period (Oct23-Dec24)":      ("2023-10-01", "2024-12-31"),
    "Late 2025 Tensions":               ("2025-09-01", "2025-12-31"),
}

for name, (s, e) in windows.items():
    mask = (p.index >= s) & (p.index <= e)
    pw = p.loc[mask]
    if len(pw) < 5:
        print(f"\n  {name}: insufficient data")
        continue
    rets = pw.iloc[-1] / pw.iloc[0] - 1
    print(f"\n  {name} ({len(pw)} days):")
    for ticker in ["USO", "XLE", "GLD", "ITA", "SPY"]:
        print(f"    {ticker:>5s}: {rets[ticker]:+7.1%}")
