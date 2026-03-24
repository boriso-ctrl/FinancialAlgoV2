"""Wrapper to run crisis backtest and capture output to a file."""
import subprocess
import sys

result = subprocess.run(
    [sys.executable, "scripts/production/run_crisis_backtest.py"],
    capture_output=True,
    text=True,
    cwd=r"c:\Users\boris\Documents\GitHub\FinancialAlgoV2",
)
with open("results/bt_v10_stdout.txt", "w", encoding="utf-8") as f:
    f.write(result.stdout)
with open("results/bt_v10_stderr.txt", "w", encoding="utf-8") as f:
    f.write(result.stderr)
print(f"Exit code: {result.returncode}")
print(f"Stdout length: {len(result.stdout)}")
print(f"Stderr length: {len(result.stderr)}")
if result.returncode != 0:
    print("--- LAST 2000 chars of STDERR ---")
    print(result.stderr[-2000:])
    print("--- LAST 2000 chars of STDOUT ---")
    print(result.stdout[-2000:])
