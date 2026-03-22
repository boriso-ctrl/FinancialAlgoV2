"""Extract NFI long_exit_signals and long_exit_main for graduated profit taking."""

with open('_nfi_x5_temp.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# long_exit_signals (L17939-18050)
print("=== LONG EXIT SIGNALS (L17939-18060) ===")
for i in range(17938, min(18060, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

# long_exit_main (L18036-18150)
print("\n\n=== LONG EXIT MAIN (L18036-18140) ===")
for i in range(18035, min(18140, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

# global_protections_long_pump / dump - find these
print("\n\n=== GLOBAL PROTECTIONS ===")
for i, line in enumerate(lines):
    s = line.strip()
    if 'global_protections_long' in s and ('pump' in s or 'dump' in s) and '=' in s:
        print(f'L{i+1}: {s[:160]}')
        # Print next 20 lines for context
        for j in range(i+1, min(i+25, len(lines))):
            print(f'L{j+1}: {lines[j].rstrip()[:160]}')
        print()
        break

# Long exit stoploss (L33551) - just the first 60 lines
print("\n\n=== LONG EXIT STOPLOSS (L33551-33650) ===")
for i in range(33550, min(33650, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')
