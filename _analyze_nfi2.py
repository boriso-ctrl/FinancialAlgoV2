"""Extract key patterns from NostalgiaForInfinityX5.py"""
import re

with open('_nfi_x5_temp.py', 'r', encoding='utf-8') as f:
    content = f.read()
    lines = content.split('\n')

# 1. Extract populate_indicators_1h (informative indicators)
print("=== SECTION 1: INDICATOR DEFINITIONS (searching for indicator computation patterns) ===\n")

# Find all unique indicator column names used
cols = set()
for m in re.finditer(r'df\["([^"]+)"\]', content):
    cols.add(m.group(1))

# Print sorted unique indicators
print(f"Total unique columns referenced: {len(cols)}")
indicator_types = {}
for c in sorted(cols):
    for prefix in ['RSI_', 'EMA_', 'SMA_', 'BB', 'CCI_', 'MFI_', 'WILLR_', 'williams_r',
                   'STOCH', 'MACD', 'ATR_', 'CMF_', 'ROC_', 'T3_', 'AROONU', 'AROOND',
                   'AROON', 'CTI_', 'CRSI_', 'EWO_', 'KST_', 'TSI_', 'ADX_', 'DI_',
                   'OBV', 'VWMA', 'chop', 'close_max', 'close_min', 'high_max', 'low_min',
                   'pct_change', 'change_pct', 'r_', 'ha_', 'zlema', 'tema', 'kama',
                   'UO_', 'PPO_', 'HMA_', 'ZL_', 'DEMA_']:
        if c.upper().startswith(prefix.upper()):
            indicator_types.setdefault(prefix, []).append(c)
            break

for prefix, indicators in sorted(indicator_types.items()):
    print(f"\n{prefix}:")
    for ind in sorted(set(indicators)):
        print(f"  {ind}")

# 2. Find the actual ta-lib / indicator calculation code
print("\n\n=== SECTION 2: INDICATOR CALCULATION CODE ===\n")
# Look for lines that compute indicators (ta-lib, qtpylib, etc.)
for i, line in enumerate(lines):
    s = line.strip()
    if any(pattern in s for pattern in [
        'ta.RSI(', 'ta.EMA(', 'ta.SMA(', 'ta.BBANDS(', 'ta.CCI(', 'ta.MFI(',
        'ta.WILLR(', 'ta.STOCHRSI(', 'ta.MACD(', 'ta.ATR(', 'ta.CMF(', 'ta.ROC(',
        'ta.T3(', 'ta.AROON(', 'ta.CTI(', 'ta.ADX(', 'ta.PLUS_DI(', 'ta.MINUS_DI(',
        'ta.OBV(', 'ta.ULTOSC(', 'ta.PPO(', 'ta.HMA(',
        'qtpylib.', 'pta.', 'pandas_ta.',
        'elliott_wave', 'EWO', 'ewo',
    ]):
        print(f'L{i+1}: {s[:160]}')

# 3. Find the Williams %R conditions (key signal in NFI)
print("\n\n=== SECTION 3: WILLIAMS R USAGE PATTERNS ===\n")
for i, line in enumerate(lines):
    s = line.strip()
    if 'williams_r' in s.lower() or 'WILLR' in s:
        print(f'L{i+1}: {s[:160]}')
        if i < len(lines) - 1:
            print(f'L{i+2}: {lines[i+1].strip()[:160]}')
