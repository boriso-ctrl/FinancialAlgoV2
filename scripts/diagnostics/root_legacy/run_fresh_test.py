import subprocess
import sys
import os

os.chdir(r"c:\Users\boris\Documents\GitHub\FinancialAlgoV2")
result = subprocess.run(
    [r".venv\Scripts\python.exe", "-B", "-m", "pytest", 
     "tests/test_strategies.py", "-k", "CrossAssetVol", 
     "--tb=short", "-v"],
    capture_output=True, text=True, cwd=r"c:\Users\boris\Documents\GitHub\FinancialAlgoV2"
)
with open("fresh_test_result.txt", "w") as f:
    f.write(result.stdout)
    f.write(result.stderr)
print("Written to fresh_test_result.txt")
