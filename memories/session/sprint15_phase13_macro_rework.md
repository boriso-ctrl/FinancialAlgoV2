# Sprint 15.1 Phase 13 — Marcus Macro & Rates Implementation Wave
## Status: EXECUTING TOP 5 REWORKS

### Priority Rework Queue
1. **H1-YieldCurveTrade** (Sharpe 0.1400 → target 0.30+)
   - Add Fed regime detection via FFR trend
   - Add TIPS breakeven inflation signal
   - Mechanism: Steepening beta changes with Fed cycle

2. **H2-CreditSpreadMeanRev** (Sharpe 0.5000 → target 0.65+)
   - Vol-scaled z-score thresholds
   - Fed funds rate gating (don't enter spreads during tightening spikes)
   - Mechanism: Mean-reversion + regime confirmation

3. **M3-EMRiskPremium** (Sharpe 0.1600 → target 0.30+)
   - Strengthen credit+dollar+equity confluence scoring
   - Reduce false positives via multi-signal consensus
   - Mechanism: 3 signals must align (credit + dollar + EEM momentum)

4. **M1-DollarCarry** (Sharpe 0.7000 → target 0.80+)
   - Tri-state risk regime (full off / partial / full on)
   - Add carry cost overlay from interest rate differentials
   - Mechanism: Gradual transitions, not binary

5. **M4-CommodityMomentum** (Sharpe 0.5600 → target 0.70+)
   - Add real-yields and TIPS breakeven filters
   - Commodity-inflation regime alignment
   - Mechanism: Only go long when inflation expectations rising

### Testing Windows
- Fed Tightening: 2022-03, 2022-12 (aggressive rate hikes)
- QE/Easing: 2020-04, 2021-06 (accommodative policy)
- Fed Pause: 2019-02, 2023-10 (pivot periods)
- Full Period: 2010-2025

### Target Gate
- Each rework: Sharpe +0.10 minimum, optimal +0.15-0.25
- Max DD improvement: 3-5% valued
- Equity correlation: < 0.3 with SPY-heavy strategies
