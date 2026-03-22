"""Extract NFI profit protection and indicator computation."""

with open('_nfi_x5_temp.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 1. Exit profit target logic
print("=== EXIT PROFIT TARGET (L722-950) ===")
for i in range(721, min(950, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

print("\n\n=== CUSTOM EXIT (L1318-1500) ===")
for i in range(1317, min(1500, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

print("\n\n=== INFORMATIVE 1D INDICATORS (L2100-2220) ===")
for i in range(2099, min(2220, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')

print("\n\n=== INFORMATIVE 4H INDICATORS (L2250-2410) ===")
for i in range(2249, min(2410, len(lines))):
    print(f'L{i+1}: {lines[i].rstrip()[:160]}')
