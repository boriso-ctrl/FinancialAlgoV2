# Experimental Strategy Lab

**Purpose**: A sandbox for testing radical, unconventional alpha ideas before they graduate to the production ensemble.

## How It Works

1. **Write a strategy** in `experimental/strategies/` — inherit from the production `Strategy` base class
2. **Run the experimental backtest**: `.venv\Scripts\python.exe experimental/run_experiments.py`
3. **If it works** (Sharpe > 0.5, uncorrelated with production strats), propose it for promotion to `src/financial_algo/strategies/`
4. **If it doesn't work**, document *why* and what you learned — negative results are valuable

## Strategy Categories

| Cat | Name | Description |
|-----|------|-------------|
| X | Cross-Asset Divergence | Exploiting breakdown in cross-asset correlations |
| Y | Behavioral Anomalies | Calendar, attention, anchoring, disposition effects |
| Z | Structural Alpha | Mechanical flows, rebalancing, roll yields |
| W | Alternative Signals | Breadth, dispersion, skew, term structure shape |

## Rules of the Lab

- **No rules on signal type** — anything goes (astrology? test it. moon phases? test it.)
- **Hard rules on methodology** — no look-ahead bias, realistic costs, proper shifting
- **Document your thesis** — every strategy needs a `# Thesis:` comment explaining WHY it should work
- **Log negative results** — add a comment if a strategy fails and why

## Promotion Criteria

To move from experimental to production:
1. Sharpe > 0.5 over 2010-2025 full period
2. Positive returns in at least 4/7 crisis windows
3. Correlation < 0.4 with ALL existing ensemble members
4. Passes full strategy-audit checklist
5. Reviewed by Peter + relevant department head
