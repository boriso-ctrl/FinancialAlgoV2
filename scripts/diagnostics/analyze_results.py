"""Quick analysis of backtest results."""
import pandas as pd

df = pd.read_csv("results/backtest_Full_Period_2010-2025.csv")
df["_sharpe"] = pd.to_numeric(df["Sharpe"].replace("ERR", None), errors="coerce")
df["_cagr"] = pd.to_numeric(df["CAGR"].str.rstrip("%"), errors="coerce")

# Exclude benchmark and ensemble
strats = df[~df["Strategy"].isin(["BuyHold-SPY", "Ensemble-BestOfEach"])]
pos = strats[strats["_sharpe"] > 0]
strong = strats[strats["_sharpe"] >= 0.5]

print(f"Total strategies: {len(strats)}")
print(f"Positive Sharpe: {len(pos)}")
print(f"Sharpe >= 0.5: {len(strong)}")
print(f"Avg Sharpe (all): {strats['_sharpe'].mean():.2f}")
print(f"Avg Sharpe (positive only): {pos['_sharpe'].mean():.2f}")
print(f"Avg CAGR (positive Sharpe): {pos['_cagr'].mean():.1f}%")
print()

ens = df[df["Strategy"] == "Ensemble-BestOfEach"]
print("--- ENSEMBLE ---")
print(f"Sharpe: {ens['Sharpe'].values[0]}")
print(f"CAGR:   {ens['CAGR'].values[0]}")
print(f"MaxDD:  {ens['MaxDD'].values[0]}")
print()

bm = df[df["Strategy"] == "BuyHold-SPY"]
print("--- BENCHMARK (SPY) ---")
print(f"Sharpe: {bm['Sharpe'].values[0]}")
print(f"CAGR:   {bm['CAGR'].values[0]}")
print(f"MaxDD:  {bm['MaxDD'].values[0]}")
print()

print("--- TOP 15 by Sharpe ---")
top = strats.nlargest(15, "_sharpe")
for _, r in top.iterrows():
    print(f"  {r['Strategy']:30s}  Sharpe={r['Sharpe']}  CAGR={r['CAGR']}  MaxDD={r['MaxDD']}")
print()

print("--- BOTTOM 10 by Sharpe (cut candidates) ---")
bot = strats.nsmallest(10, "_sharpe")
for _, r in bot.iterrows():
    print(f"  {r['Strategy']:30s}  Sharpe={r['Sharpe']}  CAGR={r['CAGR']}  MaxDD={r['MaxDD']}")
