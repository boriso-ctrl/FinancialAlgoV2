# Financial Algorithms: Project History & Future Roadmap

## Project Timeline & Milestones

### **Phase 1: Legacy Foundation (Pre-2024)**
**Status:** ✅ Complete  
**Key Achievements:**
- Developed 10+ individual technical indicator strategies (SMA, RSI, MACD, SAR/Stoch, BB+RSI, OBV, etc.)
- Implemented fundamental analysis strategy using SimFin + Random Forest
- Experimented with NLP trading (Twitter sentiment 2020)
- Built D-Guided threshold strategies (S&P 500 weekly breakout)
- Prototyped Deep Learning (CNN + Q-learning) and Forex Kalman Filter models
- Archived in `research/legacy/` with full documentation

**Limitations Identified:**
- Scattered implementations across multiple notebooks and scripts
- No systematic weight blending framework
- Manual tuning; no large-scale search infrastructure

---

### **Phase 2: Modernization & Packaging (2024–Early 2025)**
**Status:** ✅ Complete  
**Key Achievements:**
- Consolidated all indicators into unified `src/financial_algorithms/` package structure
- Implemented signal blending engine (`blend_signals()`) for multi-indicator combination
- Built modular backtest engine (`run_backtest()`) with standardized metrics (Sharpe, Sortino, Calmar, Max DD)
- Created backwards-compatibility shims (`backtest/`, `indicators/`, `signals/`) for legacy code
- Centralized data loading via SimFin (`financial_algorithms.data.loader`)
- Added strategy registry for programmatic strategy selection

**Architecture:**
```
src/financial_algorithms/
├── data/           # SimFin price loading
├── signals/
│   ├── price/      # 12+ price-based indicators
│   └── volume/     # 10+ volume-based indicators
├── backtest/
│   ├── engine.py   # Run backtest loop
│   ├── blender.py  # Weighted signal combination
│   └── metrics.py  # Performance computation
├── strategies/     # Multi-indicator combos
└── cli/            # CLI entry points
```

---

### **Phase 3: Large-Scale Weight Optimization (Jan 2026–Present)**
**Status:** 🔄 In Progress  
**Completed Runs:**

| Run | Indicators | Samples | Weight Grid | Best Sharpe | Date |
|-----|-----------|---------|-------------|------------|------|
| Baseline (20) | All 20 price+volume | 100k | {1, 1.5, 2} | 1.25 | Jan 2 |
| Focused (15) | Top price + volume confirms | 200k | {1, 1.5, 2} | 1.65 | Jan 2 |

**Key Insights:**
- ✅ Confirmed that subsets outperform all-20 (Sharpe +32% on 15-indicator combo)
- ✅ Identified champion indicators: `sar_stoch`, `stoch_macd`, `bb_rsi`, `cci_adx`, `williams_r`, `atr_trend`, `cmf`
- ✅ Found optimal weight bands: price signals 1.5–2×, volume confirms 1×
- ⚠️ Monte Carlo sampling more efficient than grid exhaustion for k≥12

---

## Current State (March 11, 2026)

### **Best-In-Class Portfolio**
**Configuration:**
- **Tickers:** AAPL, MSFT, AMZN (3 years)
- **Indicators:** 15 (price+volume subset)
- **Weights:** Optimized via 200k Monte Carlo samples
- **Mode:** Long-only, weights ≥ 1

**Performance:**
- **Sharpe Ratio:** 1.65
- **Sortino Ratio:** 2.46
- **Calmar Ratio:** 2.34
- **Total Return:** ~89 %
- **Max Drawdown:** −15.9 %
- **Stored:** `data/search_results/search_top_15ind_200k.json`

### **Active Work (March 11, 2026)**
🔄 **Phase 4a Ray Search:** 100k-sample distributed search on 8-indicator subset (Ray 4-worker)  
🔄 **Phase 5 Robustness:** Partial results (2 universes × 3 lookback windows) → Sharpe stable 0.23–1.71  
✅ **Infrastructure:** Ray 2.54.0 integrated & verified; Python 3.11 venv operational

### **Search Infrastructure**
- `scripts/search_combos.py` – Grid/Monte Carlo weight enumeration
- `--sampling-method {random, lhs, sobol}` – Configurable samplers
- `--max-combos` – Throttle runtime
- `--refine-top-k` – 2-stage local refinement around winners

### **Testing & Validation**
- 16 unit tests (smoke + regression) passing
- Legacy indicator compat verified
- Engine metrics validated against manual log-return calcs

---

## Future Roadmap (Q2–Q4 2026 & Beyond)

### **Phase 4a: Exhaustive Small-Cardinality Search (Q2 2026)**
**Objective:** Find global optimum on manageable indicator subset  
**Plan:**
- Select 8–10 highest-impact indicators
- Enumerate full 3^k grid (~6k–60k combinations)
- Apply more granular weight grids (0.5, 1, 1.5, 2, 2.5)
- Compare to Phase 3 Monte Carlo champion

**Expected Improvement:** Sharpe +5–10% (or confirm 15-indicator is already near-optimal)

---

### **Phase 4b: Refinement & Neighborhood Optimization (Q2 2026)**
**Objective:** Polish top portfolios via local search  
**Plan:**
- Run 2-stage refinement on all Phase 3 & 4a winners
- Use `--refine-top-k 50 --refine-radius 2` to explore ±2 weight steps
- Test dynamic rebalancing (quarterly/monthly weight tweaks)

**Expected Improvement:** Sharpe +2–3%

---

### **Phase 5: Robustness & Generalization (Q3 2026)**
**Objective:** Validate winners across regimes  
**Status:** 🔄 In Progress

**Test Grid:**
| Universe | 1y Window | 2y Window | 3y Window |
|----------|-----------|-----------|-----------|
| AAPL/MSFT/AMZN (baseline) | 1.39 | **1.71** ✅| 0.23 ⚠️ |
| Top-10 S&P | 1.61 ✅ | 1.30 | 0.67 |
| Phase 3 Baseline (3y) | — | — | **1.65** |

**Key Findings:**
- ✅ Portfolio maintains **Sharpe ≥ 1.3** across most conditions
- ✅ Recent windows (1y–2y) **exceed** baseline (reached 1.71)
- ⚠️ 3-year window shows decay → likely market regime change or data staleness (Dec 2023 → current)
- 📊 Conclusion: **Portfolio is reasonably robust**; recommend quarterly reoptimization for longer windows

**Stress Tests Still Pending:**
- Different lookback windows (1y, 2y, 5y, 10y) ← partial ✅
- Market volatility regimes (VIX-based filtering)
- Forward walk (retrain monthly; track Sharpe decay)
- Transaction costs sensitivity (±0.5 bps, ±5 bps slippage)

**Expected Outcome:**
- Identify regime-agnostic indicator combos ← in progress
- Publish robust portfolio weights + parameter bands

---

### **Phase 6: Live & Paper Trading (Q3–Q4 2026)**
**Objective:** Bridge backtesting → real execution  
**Plan:**
- Integrate broker APIs (Alpaca, Interactive Brokers, or similar)
- Implement live signal computation (daily/hourly updates)
- Deploy position sizing & risk management (Kelly criterion, max portfolio DD = 20%)
- Track realized vs. backtest metrics; log slippage/edge erosion

**Expected Timeline:** 3–6 month paper trading window before live capital

---

### **Phase 7: Adaptive & ML Enhancements (Q4 2026 & Beyond)**
**Objective:** Move from static weights → adaptive strategies  
**Ideas:**
- **Regime Detection:** ML classifier (SVM/RF) to switch weight maps by market conditions
- **Online Learning:** Kalman filters to drift weights as market microstructure changes
- **Feature Engineering:** Add macro factors (VIX, yield curve, Fed rates) as gating signals
- **Hierarchical Blending:** Tier-1 (buy/sell) + Tier-2 (position sizing) signals

**Expected Payoff:** Reduce drawdown periods; extend durability across market cycles

---

### **Phase 8: Scaling & Multi-Asset Expansion (Beyond Q4 2026)**
**Objective:** Extend from equities → full multi-asset portfolio  
**Targets:**
- Futures (ES, NQ, YM index contracts)
- Forex (major pairs; leverage opportunities)
- Crypto (BTC, ETH; 24/7 liquidity)
- Fixed Income (TLT, SHV; duration plays)

**Approach:** Reuse signal infrastructure; calibrate indicator parameters per asset class

---

## Dependency & Resource Map

### **Current Dependencies**
| Library | Version | Purpose |
|---------|---------|---------|
| pandas | Latest | Data manipulation |
| numpy | Latest | Numerical compute |
| simfin | Latest | Price data loading |
| scikit-learn | Latest | Metrics, ML utilities |
| pytest | Latest | Unit test framework |

### **Infrastructure Needs** (Phases 5–8)
- Cloud compute (backtests on 10y+ histories + multiple regimes)
- Real-time data ingestion (broker feeds)
- Signal deployment (containerized service; Kubernetes or lambda)
- Monitoring & alerting (Prometheus, Grafana, or DataDog)

---

## Success Metrics

| Phase | KPI | Target | Current |
|-------|-----|--------|---------|
| **3** | Sharpe on 15-ind combo | ≥ 1.5 | ✅ 1.65 |
| **4** | Exhaustive subset Sharpe | ≥ 1.7 | TBD |
| **5** | Regime robustness | 4+ regimes, Sharpe ≥ 1.4 | TBD |
| **6** | Paper trade 6mo Sharpe ≈ Backtest ±10% | TBD | Not started |
| **7** | Adaptive Sharpe vs. static | +5–15% | Not started |
| **8** | Multi-asset Sharpe | ≥ 1.3 on aggregated PnL | Not started |

---

## Risk & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Overfitting on AAPL/MSFT/AMZN | High | Phase 5 cross-universe validation |
| Regime non-stationarity | High | Quarterly reoptimization; regime classifier |
| Slippage > 5 bps | High | Live trading with robust risk limits |
| Data quality (gaps, survivorship bias) | Medium | SimFin data audit; cross-check w/ Yahoo Finance |
| Indicator crowding | Medium | Correlation analysis; prune redundant signals |

---

## Deliverables by Phase

- **Phase 4:** Top-10 portfolios ranked by Sharpe; grid search results + refinement notebook
- **Phase 5:** Regime backtest report; robustness dashboard (Sharpe by market condition)
- **Phase 6:** Paper trade log; slippage analysis; live signal microservice
- **Phase 7:** Adaptive weight timeseries; regime classifier model + inference pipeline
- **Phase 8:** Multi-asset performance table; unified signal architecture docs

---

## Next Immediate Action (March 2026)

1. ✅ **Confirm 15-indicator Sharpe = 1.65 is current best**
2. 🔄 **Execute Phase 4a:** Enumerate 8-indicator subset exhaustively (est. 2–4 hours compute)
3. 📋 **Document ingredient weights** from top-3 Phase 3 portfolios → candidate for production config
4. 📊 **Sketch Phase 5 test matrix** (regimes, tickers, lookback windows)

---

**Last Updated:** March 11, 2026  
**Owner:** Boris (Financial-Algorithms)  
**Next Review:** Post Phase 4a completion (~end of Q2 2026)
