# Sprint 15.1 Phase 13 — Volatility & ML Implementation Wave
## Vera's Top 12 → Top 6 Execution Plan

**Date**: March 27, 2026
**Objective**: Implement top rworks from 25-strategy deep dive. Target: Sharpe +0.10 to +0.20 per strategy.

---

## Baseline Performance (Sprint 15 Phase 12)

| Category | Strategy | Sharpe | CAGR | Status | Target |
|----------|----------|--------|------|--------|--------|
| **L (Vol)** | L1-VolRiskPremium | 0.92 | 0.1734 | watch_and_rework | → 0.97 (+0.05) |
| **L (Vol)** | L4-VolOfVolRegime | 0.92 | 0.1786 | watch_and_rework | → 0.98 (+0.06) |
| **L (Vol)** | L7-ImpliedRealizedSpread | -0.17 | -0.0135 | kill_or_full_rework | → 0.15 (+0.32) |
| **G (Sent)** | G1-SentimentCrisisAlpha | 0.79 | 0.1265 | optimize_winner | → 0.88 (+0.09) |
| **P (ML)** | P3-GMMRegimeClassifier | 0.26 | 0.0287 | high_priority_rework | → 0.40 (+0.14) |
| **P (ML)** | P4-AdaptiveThreshold | 0.75 | 0.1341 | watch_and_rework | → 0.82 (+0.07) |
| **DL** | DL2-LSTMRegimeDetector | 0.66 | 0.109 | watch_and_rework | → 0.75 (+0.09) |
| **Benchmark** | BuyHold-SPY | 0.90 | 0.1641 | — | — |

---

## Top 6 Rworks Selected (Ordered by Impact Priority)

### 1. **L7-ImpliedRealizedSpread** (KILL → REWORK)
- **Problem**: Sharpe -0.17 — Failed strategy, loses money consistently
- **Root Cause**: Oversimplified VIX/RV ratio signal; no regime adaptation; ignores vol persistence
- **Rework Approach**:
  - Add vol clustering detection (realized vol autocorrelation)
  - Implement regime-aware vol premium thresholds (crisis vs normal)
  - Add tail protection (reduce longs during vol spikes > 2σ)
  - Fix NaN handling in ratio calculation
- **Target**: Sharpe +0.15 (+0.32 delta)
- **Impact**: Converts a loss maker into positive alpha

### 2. **P3-GMMRegimeClassifier** (URGENT REWORK)
- **Problem**: Sharpe 0.26 — ML overfitting or poor feature engineering
- **Root Cause**: Likely poor walk-forward validation, potential look-ahead bias, bad features
- **Rework Approach**:
  - Audit for look-ahead bias (make sure training only uses past data)
  - Add feature importance analysis (which features actually matter)
  - Implement strict TimeSeriesSplit validation
  - Reduce model complexity (simpler HMM proxy vs complex GMM)
  - Fix NaN handling in feature pipeline
- **Target**: Sharpe +0.40 (+0.14 delta)
- **Impact**: ML strategies regain credibility

### 3. **L1-VolRiskPremium** (ENHANCE)
- **Problem**: Sharpe 0.92 — Already excellent, but can push higher
- **Root Cause**: Static contango/backwardation thresholds; no momentum confirmation
- **Rework Approach**:
  - Add VIX momentum confirmation (when VIX is rising, reduce long equity)
  - Implement dynamic thresholds based on vol regime (crisis = different threshold)
  - Add vol term structure (VIX3M vs VIX) for better timing
  - Increase leverage in identified contango regimes
- **Target**: Sharpe +0.97 (+0.05 delta)
- **Impact**: 5 bp improvement on highest-conviction strategy

### 4. **G1-SentimentCrisisAlpha** (SENTIMENT REFINEMENT)
- **Problem**: Sharpe 0.79 — Good but lagging G2-FearGreedContrarian (0.90)
- **Root Cause**: Sentiment signals only use VIX proxy; need richer emotion indicators
- **Rework Approach**:
  - Replace synthetic VIX-only sentiment with multi-signal score (VIX level, VIX momentum, vol of vol)
  - Improve crisis detection with news velocity acceleration (not just level)
  - Add sentiment z-score robustness (handle 0-vol case, clip extremes)
  - Implement rolling sentiment calibration (adapt thresholds each quarter)
- **Target**: Sharpe +0.88 (+0.09 delta)
- **Impact**: Sentiment strategy competes with macro strategies

### 5. **DL2-LSTMRegimeDetector** (DEEP LEARNING TUNE)
- **Problem**: Sharpe 0.66 — Underperforming, likely architecture or training issue
- **Root Cause**: LSTM may be too powerful for financial data and overfitting; not using GPU effectively
- **Rework Approach**:
  - Switch to GPU-accelerated training (RTX 5060 available)
  - Reduce model capacity (smaller hidden state, dropout)
  - Implement early stopping with walk-forward validation
  - Add feature scaling normalization (critical for LSTM)
  - Test on Volmageddon 2018 and COVID 2020 High-vol windows
- **Target**: Sharpe +0.75 (+0.09 delta)
- **Impact**: Deep learning regains baseline credibility

### 6. **P4-AdaptiveThreshold** (PARAMETER OPTIMIZATION)
- **Problem**: Sharpe 0.75 — ML-enhanced but parameter grid may be insufficient
- **Root Cause**: Static parameter grid size (72 combos); may miss optimal region
- **Rework Approach**:
  - Implement Bayesian optimization for parameter search (faster convergence)
  - Add regime-dependent parameter sets (different params for crisis vs normal)
  - Implement GPU-accelerated hyperparameter sweep
  - Add feature importance ranking to drop weak signals
- **Target**: Sharpe +0.82 (+0.07 delta)
- **Impact**: More robust parameter optimization = better generalization

---

## Implementation Schedule

**Phase 1 (Today)**: Reworks 1-3 (L7, P3, L1)
- Focus on vol strategies and critical ML fix
- Test in Volmageddon 2018 and COVID 2020

**Phase 2 (Tomorrow)**: Reworks 4-6 (G1, DL2, P4)
- Sentiment refinement and deep learning tuning
- Full backtest suite

**Validation Gates**:
- ✓ Pytest passes (`tests/test_strategies.py::test_strategy_validation`)
- ✓ No NaN/inf leakage in weights
- ✓ High-vol window (VIX > 25): must not crash
- ✓ Crisis-specific testing (Volmageddon 2018, COVID 2020)

---

## Target Report Format

```
## Volatility & ML Implementation Report — Sprint 15.1 Phase 13

### Reworks Completed

| Strategy | Mechanism | Before Sharpe | After Sharpe | Delta | Vol Regime | Status |
|----------|-----------|---------------|--------------|-------|-----------|--------|
| L7 | Vol clustering + regime adapt | -0.17 | +0.15 | +0.32 | Both | ✓ |
| P3 | Walk-forward validation + features | 0.26 | +0.40 | +0.14 | Both | ✓ |
| L1 | VIX momentum + dyn thresholds | 0.92 | +0.97 | +0.05 | Crisis | ✓ |
| G1 | Multi-signal sentiment score | 0.79 | +0.88 | +0.09 | Both | ✓ |
| DL2 | GPU accel + architecture tune | 0.66 | +0.75 | +0.09 | Both | ✓ |
| P4 | Bayesian param optimization | 0.75 | +0.82 | +0.07 | Both | ✓ |

**Portfolio Impact**: 
- Combined Sharpe improvement: +0.77 (weighted by size)
- Kill count: 1 (L7 from -0.17)
- GPU usage: DL2, P4 training 40% faster
- No look-ahead detected in ML models

### Key Blockers
None — all reworks proceed on schedule.
```

---

## GPU Usage Plan

**RTX 5060 (8 GB VRAM) deployment**:
1. DL2-LSTMRegimeDetector: Full walk-forward training on GPU (est. 40% speedup vs CPU)
2. P4-AdaptiveThreshold: Bayesian hyperparameter sweep on GPU (est. 3x speedup)

**Estimated runtime**:
- L7, P3, L1 (CPU): ~30 min total
- DL2 + P4 (GPU): ~15 min total (vs 40 min CPU)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| L7 rework still doesn't improve | Use simpler carry strategy, fall back to L1 |
| P3 overfitting persists | Reduce model complexity, use linear baseline |
| DL2 GPU OOM (8 GB VRAM limit) | Reduce batch size, shorter lookback windows |
| High-vol window crashes | Implement hard stops, tail hedge on short strats |

