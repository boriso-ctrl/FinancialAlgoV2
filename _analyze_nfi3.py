"""Extract NFI entry conditions, profit protection, and derisking logic."""

with open('_nfi_x5_temp.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Extract populate_entry_trend (main entry conditions)
print("=== POPULATE_ENTRY_TREND (L10125-10200) ===")
for i in range(10124, min(10250, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

print("\n\n=== ENTRY CONDITIONS SAMPLE (long_entry_conditions start around L3000) ===")
# Find long_entry_conditions
for i in range(2900, min(3200, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

print("\n\n=== EXIT PROFIT TARGET (L722-900) ===")
for i in range(721, min(900, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

print("\n\n=== DERISKING MODE (L9824-9930) ===")
for i in range(9823, min(9930, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')
