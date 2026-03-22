"""Extract NFI graduated profit protection (long_exit_normal) and 5m indicators."""

with open('_nfi_x5_temp.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Long exit normal - graduated exit
print("=== LONG EXIT NORMAL (L15869-16000) ===")
for i in range(15868, min(16000, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

# 5m indicator computation (main timeframe)
print("\n\n=== 5M INDICATOR COMPUTATION ===")
# Find informative_5m or main indicators
for i in range(2500, min(2850, len(lines))):
    s = lines[i].strip()
    if any(kw in s for kw in ['pta.', 'EMA_', 'SMA_', 'RSI_', 'BBB_', 'close_max', 'close_min',
                               'high_max', 'low_min', 'num_empty', 'WILLR_480',
                               'EWO', 'pct_change', 'ha_', 'T3_']):
        print(f'L{i+1}: {s[:160]}')

# NFI parameters - stop thresholds
print("\n\n=== STOP/DERISK THRESHOLDS ===")
for i in range(0, min(300, len(lines))):
    s = lines[i].strip()
    if any(kw in s for kw in ['stop_threshold', 'derisk', 'profit_max', 'profit_min',
                               'grind_', 'rebuy_', 'trailing', 'sl_']):
        print(f'L{i+1}: {s[:160]}')

# Long exit williams_r (the Williams %R exit system)
print("\n\n=== LONG EXIT WILLIAMS R (L18139-18300) ===")
for i in range(18138, min(18350, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')
