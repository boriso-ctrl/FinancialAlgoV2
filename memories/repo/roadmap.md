# Roadmap

## Completed
- **Init 1**: Max DD Reduction -- Sharpe 1.12->1.31, MaxDD -30.9%->-18.6%, CAGR 22.6%->24.5%
- **Init 2**: Walk-Forward Validation -- WF Sharpe 1.17, 0 overfit, 11/14 folds positive
- **Init 3**: Multi-Frequency Signals -- Built 4 strategies (MF1-MF4), MF2 Sharpe 1.24 standalone.
  Ensemble v6 (24 members) with MF1+MF2: Sharpe 1.31, CAGR 24.41%, MaxDD -18.60%.
- **Init 4**: Asset Class Expansion -- Universe 26->42 tickers, 10 new strategies built.
  New tickers: SLV, SHY, ETH-USD, DBC, DBA, TIP, AGG, EMB, FXI, VGK, EWJ, INDA, VNQ, XBI.
  New strategies: I5-GlobalMomRotation, J5-GlobalMeanReversion, K5-RealAssetsFactor,
  M6-InflationBreakeven, M7-GlobalRotation, M8-CommodityMacro, O7-PreciousMetalsCH,
  F4-CryptoContagion, L6-CrossAssetVol, G5-CryptoSentDiv. Top performers:
  L6 (Sharpe 1.01), O7 (0.92), G5 (0.76), F4 (0.60). A/B tested L6+F4 in ensemble:
  dSharpe=-0.003 -- no improvement due to correlation with existing members.
  Ensemble stays at 24 members. 219/219 tests passing. 62 total strategies in registry.
- **Init 5**: Weak Spot Regime Hardening -- diagnosed 18-20/26 bleeders per weak year.
  Built 10 new R-category strategies (R1-R10). Best standalone: R9-MultiAssetCTATrend
  (WF Sharpe 1.25, beating IS=1.20 -- anti-degrades!). A/B tested 5 variants; +R9 is winner.
  Ensemble: 25->26 members. Full-period Sharpe 1.29->1.31 (+0.02), WF 1.17->1.19 (+0.02).
  Weak years: 2018 -0.10->-0.07, 2022 -0.70->-0.61 (improved). 2015 -0.23->-0.26 (slightly worse).
  Structural constraint: 2015/2018/2022 have orthogonal alpha sources; cannot fix all three.
  323/323 tests passing.
- **Init 6**: Alpaca Intraday Data Pipeline (Phase 2) -- alpaca-py 0.43.2 installed.
  DATA-1: `alpaca_loader.py` (1-min bars, gzip cache, IEX/SIP feed, env-var credentials).
  DATA-2+DATA-3: `feature_store.py` (7 microstructure features: realized_vol_1min, vwap_deviation,
  opening_gap, intraday_range, vol_of_vol, volume_surprise, close_to_high; per-ticker gzip cache).
  Injection pattern: `Strategy._intraday_features`, `set_intraday_features()` on base class.
  Feature CONSUMPTION: P4-AdaptiveThreshold + P2-XGBoostSignalCombo now apply `_apply_intraday_overlay`
  scaling exposure 0.70x when SPY intraday RV >1.5x 63-day rolling mean, 1.10x when <0.70x.
  Overlay validated: rv_ratio ~1.97x at vol spike -> 0.70x multiplier fires correctly.
  `run_crisis_backtest.py` wired for optional feature injection (requires ALPACA_API_KEY env var).
  Tests: 323/323 passing. Awaiting live API keys to test real intraday data download.


  4 ML strategies built: P2-XGBoostSignalCombo (Sharpe 0.51), P3-GMMRegimeClassifier (0.26, cut),
  P4-AdaptiveThreshold (Sharpe 0.75, WF 0.87 -- anti-degrades!), P5-CrossSectionalRanker (0.49, long-only).
  P2 tuned: stride 5->21, min_train 252->504, deeper trees, position clamp.
  P5 fixed: removed short book, long-only top-8, 1x leverage.
  Ensemble v8: P4 added as 25th member. Sharpe 1.28, CAGR 24.02%, MaxDD -18.58%.
  Walk-forward v8: IS=1.34, WF=1.19 (+0.02 vs v7), MaxDD -20.0%. 295/295 tests passing.

## Planned — ML/DL Integration Roadmap

### Phase 1: Classical ML on Daily Data (Cost: $0) -- IN PROGRESS
- **Deps**: scikit-learn 1.8.0, xgboost 3.2.0 -- INSTALLED
- **P2-XGBoostSignalCombo**: Built (Sharpe 0.51). Tuned stride 21, min_train 504, deeper trees. Below 0.5 threshold -- watching.
- **P3-GMMRegimeClassifier**: Built, Sharpe 0.26 -- CUT (below 0.3 threshold)
- **P4-AdaptiveThreshold**: Built, Sharpe 0.75, WF=0.87 (anti-degrades). IN ENSEMBLE v8.
- **P5-CrossSectionalRanker**: Built, Sharpe 0.49 (long-only fix applied). Below threshold -- watching.
- **Remaining**: Improve P2 further (Sharpe > 0.8 target), get P5 to 0.5+, consider P2 for ensemble v9

### Phase 2: Alpaca Intraday Data Pipeline (Cost: $0, needs Alpaca free account) -- COMPLETE
- **Deps**: alpaca-py 0.43.2 -- INSTALLED
- **DATA-1**: `alpaca_loader.py` -- DONE (load_intraday, load_daily_alpaca, gzip cache)
- **DATA-2**: Intraday feature extractor -- DONE (7 features: realized_vol_1min, vwap_deviation, etc.)
- **DATA-3**: `FeatureStore` class -- DONE (per-ticker gzip cache, daily DataFrame output)
- **Consumption**: P4 + P2 `_apply_intraday_overlay` -- DONE (0.70x in high-vol, 1.10x in calm vol)
- **Tests**: 323/323 passing
- **Pending**: Live test with real Alpaca credentials (user to provide API keys)

### Phase 3: Deep Learning on Intraday Patterns (Cost: $0-99/mo)
- **Deps**: torch>=2.0 (CPU-only)
- **DL-1**: Temporal CNN on 1-min OHLCV — learns microstructure patterns → daily alpha score
- **DL-2**: LSTM Regime Detector on 30-day sequences → early warning regime transitions
- **DL-3**: Attention-Based Cross-Sectional Ranker — learns dynamic cross-asset relationships
- **Target**: Ensemble Sharpe 1.5-1.8
- **Kill criteria**: If DL underperforms XGBoost after 2 weeks, kill and stay at Phase 1

### Phase 4: Live Paper Trading (Cost: $0-99/mo, Alpaca paper account)
- **LIVE-1**: Nightly signal pipeline (download data → extract features → run models → target weights)
- **LIVE-2**: Alpaca paper trading executor
- **LIVE-3**: Performance monitor (live PnL vs backtest expectations)
- **LIVE-4**: Model drift detector
- **Go-live criteria**: 3 months paper trading with Sharpe > 0.8, MaxDD < -15%

### Backlog — Weak Spot Regime Hardening (COMPLETED as Init 5)
  - See Init 5 below. 2015/2018/2022 improved. Structural constraint: orthogonal alpha sources per year.

## Current Ensemble (v9): Sharpe 1.31, CAGR 24.81%, MaxDD -18.62%, 26 members
- Walk-Forward: IS=1.34, WF=1.19, MaxDD -20.0% OOS, 11/14 folds positive (weak: 2015=-0.26, 2018=-0.07, 2022=-0.61)
- Sharpe^2 weights, max_gross_leverage=2.5, 42-ticker universe
- 323/323 tests passing
- P1-FeatureComboSignal: Sharpe 0.96, CAGR 18.33% — features upgraded to 9 (Sprint 7)
  - New features: tsi_rank, rmi_rank, skew_rank (inverted), kurt_rank (inverted), rcr_rank, hurst_rank
  - All cross-sectionally ranked [0,1]; hurst_window=126 for lower latency
- P4-AdaptiveThreshold: WF Sharpe 0.87, 86% positive folds (best OOS of all ML strategies)
- P2-XGBoostSignalCombo: Sharpe 0.51, watching for v9 candidacy
- P5-CrossSectionalRanker: Sharpe 0.49 (long-only), watching for v9 candidacy

## Alpha-Boost Sprint Plan (from open-source research) -- ALL COMPLETE
- Sprint 1: 11 new indicators -- COMPLETE
- Sprint 2: I6/J6 RETIRED (Sharpe 0.08/-0.16). SkewKurt overlay in EnsembleConfig (disabled by default).
- Sprint 3: TSI filters in I1-I5, J1 -- COMPLETE
- Sprint 4: G6-MultiSignalConsensus -- COMPLETE. 5-signal consensus strategy, long-only, 12 ETFs, threshold=3/5. 327 tests.
- Sprint 5: Hurst regime router in EnsembleConfig -- COMPLETE (disabled by default)
- Sprint 6: Graduated RSI exit overlay in EnsembleConfig -- COMPLETE.
  rsi_exit_overlay=True: RSI>80 -> scale 0.75x; RSI>90 -> scale 0.50x. Smoke-tested, no NaN.
- Sprint 7: P1 ML feature upgrade (3 -> 9 features) -- COMPLETE

## Phase 2: Alpaca Intraday Data Pipeline -- READY TO INTEGRATE
- alpaca_loader.py and feature_store.py already exist in src/financial_algo/data/
- .env file has credentials (PK7GZTZQZ43B7M4OLY5PH4W7GU / secret). Auto-loaded by test script.
- Pipeline test PASSED: 257 days x 14 features, 0.22% NaN rate, 7 feature types per ticker:
  realized_vol_1min, vwap_deviation, opening_gap, intraday_range, vol_of_vol, volume_surprise, close_to_high
- Next step: feed these 14 intraday features into P1/P2 as additional daily signals

## Next: Feed Phase 2 intraday features into P1/P2 ML strategies for Sharpe step-change to 1.6+
