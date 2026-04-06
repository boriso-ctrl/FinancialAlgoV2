"""Extract Full Period results from backtest output."""
import os

f = 'results/backtest_v10_full.txt'
if not os.path.exists(f):
    print("FILE NOT FOUND - running backtest first")
    exit(1)

print(f"File size: {os.path.getsize(f)} bytes")

with open(f, 'r', encoding='utf-8') as fh:
    content = fh.read()

print(f"Content length: {len(content)} chars")

# Find all CRISIS WINDOW markers
import re
windows = [(m.start(), m.group()) for m in re.finditer(r'CRISIS WINDOW:.*', content)]
print(f"\nFound {len(windows)} crisis windows:")
for pos, w in windows:
    print(f"  pos={pos}: {w.strip()[:80]}")

# Find Full Period section
fp_idx = None
for i, (pos, w) in enumerate(windows):
    if 'Full Period' in w:
        fp_idx = i
        break

if fp_idx is None:
    print("ERROR: Full Period window not found!")
    exit(1)

# Extract section
start = windows[fp_idx][0]
if fp_idx + 1 < len(windows):
    end = windows[fp_idx + 1][0]
else:
    end = content.find('SUMMARY', start)
    if end == -1:
        end = len(content)

section = content[start:end]

# Write extracted section
outf = 'results/_fullperiod_v10.txt'
with open(outf, 'w', encoding='utf-8') as out:
    out.write(section)
print(f"\nWrote {len(section)} chars to {outf}")

# Also extract key lines: ensemble, new strategies, SPY
print("\n=== KEY RESULTS (Full Period 2010-2025) ===")
for line in section.split('\n'):
    line_s = line.strip()
    # Match key strategies
    targets = ['Ensemble', 'B-SPY', 'I10-', 'K6-', 'L7-', 'L8-', 'O8-', 'M9-',
               'L6-', 'R4-', 'R6-', 'L4-', 'L3-', 'Q3-', 'G1-', 'D3-', 'M5-',
               'R1-', 'R3-', 'R7-', 'R5-', 'L1-']
    for t in targets:
        if t in line_s:
            print(line_s)
            break

# Extract SUMMARY section 
sum_idx = content.find('SUMMARY - BEST STRATEGY')
if sum_idx >= 0:
    sum_end = content.find('METHODOLOGY NOTES', sum_idx)
    if sum_end == -1:
        sum_end = len(content)
    summary = content[sum_idx:sum_end]
    print(f"\n=== SUMMARY ===")
    print(summary.strip())
