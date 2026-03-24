import csv

with open('scripts/results/backtest_Full_Period_2010-2025.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['Strategy']
        if 'Ensemble' in name or name == 'BuyHold-SPY':
            cagr = row['CAGR']
            sharpe = row['Sharpe']
            maxdd = row['MaxDD']
            print(f'{name}: CAGR={cagr}, Sharpe={sharpe}, MaxDD={maxdd}')

# Also show new strategies
print()
targets = ['I10-', 'K6-', 'L7-', 'L8-', 'O8-', 'M9-', 'G1-', 'G3-']
with open('scripts/results/backtest_Full_Period_2010-2025.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row['Strategy']
        for t in targets:
            if t in name:
                cagr = row['CAGR']
                sharpe = row['Sharpe']
                maxdd = row['MaxDD']
                print(f'{name}: CAGR={cagr}, Sharpe={sharpe}, MaxDD={maxdd}')
                break
