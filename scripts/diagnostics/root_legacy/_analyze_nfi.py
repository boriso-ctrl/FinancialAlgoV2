"""Analyze NostalgiaForInfinityX5.py structure and extract key signals."""
import re

with open('_nfi_x5_temp.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print("=== METHOD MAP (first 80) ===")
count = 0
for i, line in enumerate(lines):
    if re.match(r'    def ', line):
        print(f'L{i+1}: {line.strip()[:120]}')
        count += 1
        if count > 80:
            break

print("\n=== INDICATOR CALCULATIONS (populate_indicators) ===")
# Find populate_indicators method and extract indicator logic
in_populate = False
for i, line in enumerate(lines):
    if 'def populate_indicators' in line:
        in_populate = True
        start = i
    if in_populate:
        if i > start + 5 and re.match(r'    def ', line):
            break
        # Print lines with indicator calculations
        s = line.strip()
        if any(kw in s for kw in ['RSI', 'EMA', 'SMA', 'BB', 'bollinger', 'MACD', 'williams',
                                    'CCI', 'MFI', 'ATR', 'CMF', 'chaikin', 'aroon',
                                    'ta.', 'qtpylib', 'dataframe[', "dataframe ['",
                                    'close_max', 'close_min', 'high_max', 'low_min',
                                    'rsi_', 'ema_', 'sma_', 'bb_', 'r_', 'willr',
                                    'ewo', 'crsi', 'cci', 'roc', 'pct_change']):
            print(f'L{i+1}: {s[:150]}')
