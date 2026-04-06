# Sprint 15 Go/No-Go (Full Period Sharpe Delta)

Total strategies compared: 107
- Promote: 12
- Hold/Watch: 92
- Rework Priority: 2
- Rollback Required: 1

## Top 10 Regressions

| Strategy | Baseline Sharpe | Current Sharpe | Delta | Decision |
|---|---:|---:|---:|---|
| J5-GlobalMeanReversion | 0.24 | -0.17 | -0.41 | rollback_required |
| K1-LowVolFactor | 0.84 | 0.69 | -0.15 | rework_priority |
| M1-DollarCarry | 0.70 | 0.64 | -0.06 | rework_priority |
| H4-MultiAssetCrisisLong | 0.05 | -0.00 | -0.05 | hold_watch |
| M4-CommodityMomentum | 0.56 | 0.53 | -0.03 | hold_watch |
| Ensemble-BestOfEach | 1.33 | 1.30 | -0.03 | hold_watch |
| K6-QualityMomentumComposite | 0.02 | -0.00 | -0.02 | hold_watch |
| M3-EMRiskPremium | 0.16 | 0.15 | -0.01 | hold_watch |
| I10-AdaptiveTrendFilter | -0.14 | -0.14 | 0.00 | hold_watch |
| I7-DriftRegimeMomentum | -0.06 | -0.06 | 0.00 | hold_watch |

## Top 10 Improvements

| Strategy | Baseline Sharpe | Current Sharpe | Delta | Decision |
|---|---:|---:|---:|---|
| G1-SentimentCrisisAlpha | 0.79 | 0.89 | 0.10 | promote |
| H1-CommodityShockRider | 0.20 | 0.34 | 0.14 | promote |
| DL3-AttentionRanker | 0.16 | 0.33 | 0.17 | promote |
| B2-OilShockHedge | 0.22 | 0.43 | 0.21 | promote |
| P3-GMMRegimeClassifier | 0.26 | 0.59 | 0.33 | promote |
| O6-TailHedgeOverlay | 0.18 | 0.59 | 0.41 | promote |
| O9-ATRCrisisAlpha | -0.03 | 0.51 | 0.54 | promote |
| R10-CommodityMacroOverlay | 0.09 | 0.65 | 0.56 | promote |
| O4-BlackSwanInsurance | 0.28 | 1.03 | 0.75 | promote |
| O8-VolatilityConvexity | 0.00 | 0.85 | 0.85 | promote |
