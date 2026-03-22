# Deep Research Report: 3 GitHub Repos — Exact Signals & Formulas
## Date: March 22, 2026
## Researcher: Sofia (Head of Systematic Alpha)
## Status: COMPLETE — All source code fetched and analyzed

---

# REPO 1: NostalgiaForInfinity (iterativv/NostalgiaForInfinity)
## File: NostalgiaForInfinityX5.py (58,249 lines)

### 1.1 Complete Indicator Library Used (147 unique columns)

All indicators are computed via `pandas_ta` (pta) at MULTIPLE timeframes: 5m, 15m, 1h, 4h, 1d.

| Indicator | Parameters | Timeframes | Formula |
|-----------|-----------|------------|---------|
| RSI | length=3, 4, 14, 20 | 5m, 15m, 1h, 4h, 1d | Standard Wilder RSI |
| RSI change_pct | RSI pct change 1-bar | 5m, 15m, 1h, 4h, 1d | `(RSI - RSI.shift(1)) / RSI.shift(1) * 100` |
| EMA | length=3, 9, 12, 16, 20, 26, 50, 200 | 5m, 4h | Standard EMA |
| SMA | length=16, 30 | 5m | Standard SMA |
| Bollinger Bands | length=20, stdev=2.0 | 5m, 1h, 1d | BBL, BBM, BBU, BBB (bandwidth), BBP (%B) |
| Williams %R | length=14, **480** | 5m, 15m, 1h, 4h, 1d | `WILLR_14` and **`WILLR_480`** (long-term!) |
| Williams %R | length=84 | 1h | `WILLR_84_1h` (medium-term) |
| MFI | length=14 | 5m, 15m, 1h, 4h, 1d | Money Flow Index (volume-weighted RSI) |
| CMF | length=20 | 5m, 15m, 1h, 4h, 1d | Chaikin Money Flow |
| CCI | length=20 | 15m, 1h, 4h | Commodity Channel Index |
| Aroon Up/Down | length=14 | 5m, 15m, 1h, 4h, 1d | `AROONU_14`, `AROOND_14` |
| StochRSI | 14,14,3,3 | 5m, 15m, 1h, 4h, 1d | `STOCHRSIk_14_14_3_3` |
| Stochastic | 14,3,3 | 15m, 1h, 4h, 1d | `STOCHk_14_3_3` |
| KST | 10,15,20,30,10,10,10,15 | 5m, 4h | Know Sure Thing (multi-horizon momentum) |
| Ultimate Oscillator | 7,14,28 | 15m, 1h, 4h | `UO_7_14_28` |
| ROC | length=2, 9 | 5m, 15m, 1h, 4h, 1d | Rate of Change |
| OBV | - | 5m, 15m, 1h, 4h | On-Balance Volume |
| CTI | length=20 | 1d | Correlation Trend Indicator |

### 1.2 Derived Indicators

```python
# Close relative to recent range (5m)
close_max_48 = close.rolling(48).max()  # 4-hour high (at 5m bars)
close_min_48 = close.rolling(48).min()  # 4-hour low

# Empty candle count (liquidity filter)
num_empty_288 = (volume <= 0).rolling(288).sum()  # Count of zero-volume bars in 24h

# Multi-timeframe highs/lows
high_max_6_{tf}  = high.rolling(6).max()   # 6-bar high
high_max_12_{tf} = high.rolling(12).max()  # 12-bar high
high_max_20_{tf} = high.rolling(20).max()  # 20-bar high (1d only)
high_max_24_{tf} = high.rolling(24).max()  # 24-bar high (4h only)
high_max_30_{tf} = high.rolling(30).max()  # 30-bar high (1d only)
low_min_12_{tf}  = low.rolling(12).min()   # 12-bar low
low_min_24_{tf}  = low.rolling(24).min()   # 24-bar low (4h only)

# Candle change
change_pct = (close - open) / open * 100.0

# Wick analysis
top_wick_pct = (high - max(open, close)) / max(open, close) * 100.0
bot_wick_pct = abs(low - min(open, close)) / min(open, close) * 100.0
```

### 1.3 Entry Signal Architecture (162+ conditions)

Each entry condition follows this pattern:
```
Condition = Protection_filters AND Signal_logic
```

**Protection System** (`global_protections_long_pump` / `global_protections_long_dump`):
- Anti-pump protection: Requires that at least ONE of several multi-timeframe oversold conditions is met
- Anti-dump protection: Same structure with different thresholds
- Each protection is an OR-chain across timeframes (if ANY oversold indicator fires, protection passes)
- Key thresholds: RSI_3 > 2-10 (5m), RSI_3 > 10-40 (15m-4h), CCI_20 < -100 to -500, STOCHRSIk < 10-50

**Entry Signal Pattern (Condition #1 example)**:
```python
# Multi-timeframe oversold confirmation:
(RSI_3 > 2.0) | (ROC_9 > -50.0)          # 5m not flash-crashing
(RSI_3_15m > 5.0) | (RSI_3_1h > 5.0) | (CMF_20_1h > -0.30)  # 15m/1h have support
(RSI_3_15m > 5.0) | (RSI_3_4h > 15.0) | (AROONU_14_4h < 50.0)  # Not in strong downtrend
(RSI_3_15m > 5.0) | (RSI_3_1h > 15.0) | (MFI_14_4h < 50.0)  # Volume flow confirms
```

### 1.4 Graduated Profit Protection System (long_exit_main)

**KEY INSIGHT: RSI-based trailing exits with profit-tiered thresholds**

Above EMA_200 (uptrend):
| Profit Range | RSI_14 Exit Threshold |
|--------------|-----------------------|
| 0.1% - 1% | RSI < 10 |
| 1% - 2% | RSI < 28 |
| 2% - 3% | RSI < 30 |
| 3% - 4% | RSI < 32 |
| 4% - 5% | RSI < 34 |
| 5% - 6% | RSI < 36 |
| 6% - 7% | RSI < 38 |
| 7% - 8% | RSI < 40 |
| 8% - 9% | RSI < 42 |
| 9% - 10% | RSI < 44 |
| 10% - 12% | RSI < 46 |
| 12% - 20% | RSI < 44 |
| 20%+ | RSI < 42 |

Below EMA_200 (downtrend) — thresholds are 2 points higher at each level.

**Economic rationale**: As profit grows, allow wider RSI pullbacks before exiting (higher RSI threshold = easier to trigger exit). At very low profits, only exit on extreme RSI oversold (< 10) to avoid shakeouts.

### 1.5 Williams %R Exit System

**Multi-timeframe Williams %R overbought sell signals**, tiered by profit:

```python
# At profit 0.1%-1%:
if (WILLR_480 > -0.1) and (WILLR_14 >= -1.0) and (RSI_14 > 75.0):  # Triple overbought
if (WILLR_14 >= -1.0) and (RSI_14 > 84.0):  # Extreme overbought
if (WILLR_14 >= -1.0) and (RSI_14 < 40.0):  # Overbought + RSI divergence (bearish)
if (RSI_3 > 99.0) and (WILLR_14 > -4.0) and (ROC_9_1d > 50.0):  # Blowoff top
if (RSI_3 > 98.0) and (WILLR_14 > -1.0) and (ROC_9_1d < -5.0):  # Overbought into declining daily
```

**WILLR_480** (480-period Williams %R) is uniquely powerful — this is ~10 trading days at 5m, giving a medium-term overbought/oversold reading not found in standard libraries.

### 1.6 Overbought Exit Signals (long_exit_signals)

```python
# Signal 1: RSI > 84 AND close > BB_upper for 5 consecutive candles
# Signal 2: RSI > 86 AND close > BB_upper for 3 consecutive candles
# Signal 3: RSI > 88 (standalone extreme)
# Signal 4: RSI_14 > 84 AND RSI_14_1h > 80 (multi-TF overbought)
# Signal 6: Close < EMA_200 AND close > EMA_50 AND RSI > 79 (bear rally top)
# Signal 8: Close > BB_upper_1h * 1.14 (14% above hourly BB upper — extreme extension)
```

### 1.7 Stoploss Architecture

```python
# Doom stoploss: loss > 25% of initial entry cost (spot) / 60% (futures)
# Regular stoploss: loss > 10% (both spot and futures)
# Rapid mode stoploss: 25% spot / 60% futures
# Derisk mode stoploss: 25% spot / 60% futures
# Trailing stop: after +3% profit, trail at 1% from high
```

### 1.8 Grind/DCA System

Position sizing through multiple sub-entries at pre-defined loss thresholds:
```python
grind_1_stakes = [0.24, 0.26, 0.28]  # Position sizes for sub-entries
grind_1_sub_thresholds = [-0.12, -0.16, -0.20]  # Enter when down -12%, -16%, -20%
grind_1_profit_threshold = 0.018  # Exit grind when +1.8% profit
grind_1_stop_grinds = -0.50  # Hard stop at -50%
```

### 1.9 Portability Assessment for ETFs

| Signal | Portability | Notes |
|--------|-------------|-------|
| Multi-TF RSI orchestration | 5/5 | Direct port to daily/weekly |
| Williams %R 480 | 5/5 | Excellent for daily ETF data |
| Graduated RSI profit exits | 5/5 | Generic, works on any asset |
| BB + RSI overbought combo | 5/5 | Standard indicators |
| CMF + momentum confirmation | 4/5 | Needs reasonable volume data |
| KST multi-horizon momentum | 5/5 | Already in our sprint plan |
| Anti-pump/dump protection | 3/5 | Volume-based, needs adaptation |
| Grind/DCA system | 2/5 | Position management, not a signal |

---

# REPO 2: Backtrader (mementum/backtrader)
## Indicator Library — Exact Source Code Formulas

### 2.1 DV2 (David Varadi's oscillator) — **HIGH PRIORITY**

```python
# Parameters: period=252, maperiod=2
# Formula:
chl = close / ((high + low) / 2.0)        # Close-to-HL midpoint ratio
dvu = SMA(chl, period=2)                   # 2-day smoothed ratio
dv2 = PercentRank(dvu, period=252) * 100   # Percentile rank over 252 days

# PercentRank formula:
pctrank = sum(x < current_value for x in lookback_window) / len(lookback_window)
```
- **Portability**: 5/5
- **Alpha source**: Short-term mean reversion
- **Thesis**: Close relative to HL midpoint captures intraday over-extension. Percentile ranking normalizes across regimes.
- **Default params**: lookback=252 (1 year), smoothing=2 bars
- **Expected signal**: Buy when DV2 < 20, sell when DV2 > 80

### 2.2 KST (Know Sure Thing) — **HIGH PRIORITY**

```python
# Parameters:
# ROC periods: rp1=10, rp2=15, rp3=20, rp4=30
# MA periods: rma1=10, rma2=10, rma3=10, rma4=10
# Signal: rsignal=9
# Weights: rfactors=[1.0, 2.0, 3.0, 4.0]

rcma1 = SMA(ROC(close, 10), period=10)
rcma2 = SMA(ROC(close, 15), period=10)
rcma3 = SMA(ROC(close, 20), period=10)
rcma4 = SMA(ROC(close, 30), period=10)

kst = 1.0 * rcma1 + 2.0 * rcma2 + 3.0 * rcma3 + 4.0 * rcma4
signal = SMA(kst, period=9)
```
- **Portability**: 5/5
- **Alpha source**: Multi-horizon momentum
- **Thesis**: Weighted sum of 4 different ROC timeframes captures short+medium+long momentum confluence.
- **Signal**: KST > signal = bullish, KST < signal = bearish. Zero-line crossovers.

### 2.3 TSI (True Strength Index) — **HIGH PRIORITY**

```python
# Parameters: period1=25, period2=13, pchange=1

price_change = close - close.shift(1)

# Double-smoothed momentum
sm1 = EMA(price_change, period=25)
sm12 = EMA(sm1, period=13)

# Double-smoothed absolute momentum
sm2 = EMA(abs(price_change), period=25)
sm22 = EMA(sm2, period=13)

tsi = 100.0 * sm12 / sm22
```
- **Portability**: 5/5
- **Alpha source**: Momentum with smoothing (reduces whipsaws)
- **Thesis**: Double-smoothed momentum ratio shows trend strength without noise. Oscillates -100 to +100.
- **Signal**: TSI > 0 bullish, TSI < 0 bearish. Divergence detection.

### 2.4 Aroon Up/Down/Oscillator

```python
# Parameters: period=14, upperband=70, lowerband=30

aroon_up = 100.0 * (period - bars_since_highest_high) / period
aroon_down = 100.0 * (period - bars_since_lowest_low) / period
aroon_osc = aroon_up - aroon_down
```
- **Portability**: 5/5
- **Alpha source**: Trend identification
- **Thesis**: Measures how recently highs/lows occurred. Both > 70 = trending. Osc > 0 = bullish.

### 2.5 Ultimate Oscillator

```python
# Parameters: p1=7, p2=14, p3=28

BP = close - min(low, close.shift(1))   # Buying Pressure
TR = max(high, close.shift(1)) - min(low, close.shift(1))  # True Range

avg7  = sum(BP, 7) / sum(TR, 7)
avg14 = sum(BP, 14) / sum(TR, 14)
avg28 = sum(BP, 28) / sum(TR, 28)

UO = 100 * (4*avg7 + 2*avg14 + avg28) / 7
```
- **Portability**: 5/5
- **Alpha source**: Multi-period mean reversion
- **Bands**: UO > 70 = overbought, UO < 30 = oversold

### 2.6 Williams %R

```python
# Parameters: period=14
percR = -100.0 * (highest_high(period) - close) / (highest_high(period) - lowest_low(period))
# Range: -100 to 0. Above -20 = overbought, below -80 = oversold.
```

### 2.7 Hurst Exponent — **UNIQUE, HIGH VALUE**

```python
# Parameters: period=40 (ideally 2000), lag_start=2, lag_end=period/2

lags = range(lag_start, lag_end)
tau = [sqrt(std(ts[lag:] - ts[:-lag])) for lag in lags]
hurst = polyfit(log10(lags), log10(tau), 1)[0] * 2.0

# Interpretation:
# H = 0.5: Random walk (no alpha)
# H < 0.5: Mean-reverting (trade mean reversion)
# H > 0.5: Trending (trade momentum)
```
- **Portability**: 4/5 (needs numpy, non-vectorized)
- **Alpha source**: Regime detection
- **Thesis**: Determines whether to deploy momentum or mean-reversion strategies. Can be used as a regime filter across our entire ensemble.
- **CRITICAL INSIGHT**: Better than SMA(200) for regime detection.

### 2.8 Laguerre RSI — **UNIQUE, HIGH VALUE**

```python
# Parameters: gamma=0.5

# Recursive Laguerre filter (4 taps):
L0 = (1 - gamma) * price + gamma * L0_prev
L1 = -gamma * L0 + L0_prev + gamma * L1_prev
L2 = -gamma * L1 + L1_prev + gamma * L2_prev
L3 = -gamma * L2 + L2_prev + gamma * L3_prev

# RSI calculation on Laguerre-filtered data:
cu = max(L0-L1, 0) + max(L1-L2, 0) + max(L2-L3, 0)
cd = max(L1-L0, 0) + max(L2-L1, 0) + max(L3-L2, 0)
lrsi = cu / (cu + cd)  # Range 0 to 1
```
- **Portability**: 4/5 (recursive, needs iterative computation)
- **Alpha source**: Fast mean-reversion detection
- **Thesis**: Laguerre filter provides "time warp" — faster reaction to price changes than standard RSI. gamma=0.5 is optimal balance per Ehlers.
- **Signal**: Buy LRSI < 0.2, sell LRSI > 0.8. Faster signals than RSI(14).

### 2.9 RMI (Relative Momentum Index)

```python
# Parameters: period=20, lookback=5
# Standard RSI formula but comparing close to close(lookback_ago) instead of close(1_ago)
# Smoother than standard RSI.
```
- **Portability**: 5/5
- **Alpha source**: Smoothed momentum/mean-reversion

### 2.10 Pretty Good Oscillator (PGO)

```python
# Parameters: period=14
pgo = (close - SMA(close, period)) / ATR(period)
# PGO > 3.0 = go long (breakout), PGO < -3.0 = go short
```
- **Portability**: 5/5
- **Alpha source**: Normalized trend breakout
- **Thesis**: Distance from SMA normalized by volatility. ATR normalization makes thresholds regime-independent.

### 2.11 Vortex Indicator

```python
# Parameters: period=14
VM_plus = sum(abs(high - low.shift(1)), period)
VM_minus = sum(abs(low - high.shift(1)), period)
TR_sum = sum(max(high-low, abs(high-close.shift(1)), abs(low-close.shift(1))), period)

VI_plus = VM_plus / TR_sum
VI_minus = VM_minus / TR_sum
# VI_plus > VI_minus = uptrend, crossovers signal trend changes
```
- **Portability**: 5/5
- **Alpha source**: Trend direction

### 2.12 Acceleration/Deceleration Oscillator

```python
# Parameters: period=5
AO = SMA((high + low)/2, 5) - SMA((high + low)/2, 34)  # Awesome Oscillator
AccDecOsc = AO - SMA(AO, 5)
```

### 2.13 CCI (Commodity Channel Index)

```python
# Parameters: period=20, factor=0.015
tp = (high + low + close) / 3
tpmean = SMA(tp, 20)
dev = tp - tpmean
meandev = MeanDeviation(tp, 20)
cci = dev / (0.015 * meandev)
# CCI > 100 = overbought, CCI < -100 = oversold
```

### 2.14 Parabolic SAR

```python
# Parameters: af=0.02, afmax=0.20
# Wilder's SAR with acceleration factor starting at 0.02, max 0.20
# Reversal-based trend following
```

### 2.15 Detrended Price Oscillator (DPO)

```python
# Parameters: period=20
dpo = close - SMA(close, period).shift(period/2 + 1)
# Removes trend, isolates cycles
```
- **Portability**: 5/5
- **Alpha source**: Cycle identification

### 2.16 OLS (Ordinary Least Squares) — in `ols.py`

Linear regression slope as a trend indicator.

### 2.17 Heikin-Ashi

```python
ha_close = (open + high + low + close) / 4
ha_open = (ha_open_prev + ha_close_prev) / 2
ha_high = max(high, ha_open, ha_close)
ha_low = min(low, ha_open, ha_close)
```

---

# REPO 3: Freqtrade FreqAI
## Feature Engineering & Walk-Forward Framework

### 3.1 Feature Engineering: Expandable Features (feature_engineering_expand_all)

These features auto-expand across MULTIPLE periods, timeframes, shifted candles, and correlation pairs:

```python
# For each period in indicator_periods_candles (e.g., [10, 20]):
features["%-rsi-period"] = ta.RSI(close, timeperiod=period)
features["%-mfi-period"] = ta.MFI(high, low, close, volume, timeperiod=period)
features["%-adx-period"] = ta.ADX(high, low, close, timeperiod=period)
features["%-sma-period"] = ta.SMA(close, timeperiod=period)
features["%-ema-period"] = ta.EMA(close, timeperiod=period)

# Bollinger Band features (stds=2.2, not default 2.0!)
bb = bollinger_bands(typical_price, window=period, stds=2.2)
features["%-bb_width-period"] = (bb_upper - bb_lower) / bb_middle
features["%-close-bb_lower-period"] = close / bb_lower

# Rate of Change
features["%-roc-period"] = ta.ROC(close, timeperiod=period)

# Relative Volume (UNIQUE!)
features["%-relative_volume-period"] = volume / volume.rolling(period).mean()
```

**Total feature count formula**:
`N_features = N_base_features * N_periods * N_timeframes * N_shifted_candles * (1 + N_corr_pairs)`

With defaults: 10 features x 2 periods x 3 timeframes x 3 shifts x 3 pairs = **540 features!**

### 3.2 Basic Features (feature_engineering_expand_basic)

```python
features["%-pct-change"] = close.pct_change()  # 1-bar return
features["%-raw_volume"] = volume
features["%-raw_price"] = close
```

### 3.3 Standard Features (feature_engineering_standard)

```python
features["%-day_of_week"] = date.dt.dayofweek  # 0-6
features["%-hour_of_day"] = date.dt.hour        # 0-23 (for intraday)
```

### 3.4 Target Variable

```python
# Target: forward-looking smoothed return
target = close.shift(-label_period_candles).rolling(label_period_candles).mean() / close
# This is a REGRESSION target (predict future return ratio)
```

### 3.5 Walk-Forward Architecture (split_timerange)

```python
# Parameters:
# train_period_days = 28 (training window in days)
# backtest_period_days = 7 (forward test window in days)

# Algorithm:
while backtest_end < total_end:
    train_window = [current_start, current_start + train_period_days]
    backtest_window = [train_end, train_end + backtest_period_days]

    train_model(data[train_window])
    predict(data[backtest_window])

    current_start += backtest_period_days  # Slide forward
```

This is an **expanding-start sliding window**: training window slides forward by `backtest_period_days` each iteration.

### 3.6 Recency-Weighted Training

```python
# weight_factor: higher = more recent data weighted more heavily
weights = set_weights_higher_recent(N)
# Applies exponential decay so recent training data matters more
```

### 3.7 Feature Expansion Pattern (populate_features)

```python
for each timeframe in include_timeframes:
    for each period in indicator_periods_candles:
        df = feature_engineering_expand_all(df, period, metadata)

    df = feature_engineering_expand_basic(df, metadata)

    # Auto-shift features by N candles (lagged features)
    for shift in range(1, include_shifted_candles + 1):
        df_shift = features.shift(shift)
        df = concat(df, df_shift.add_suffix(f"_shift-{shift}"))
```

### 3.8 Hybrid Strategy Pattern

The `FreqaiExampleHybridStrategy` shows how to combine ML predictions with traditional indicators:
```python
# Use ML prediction as a filter ON TOP of RSI signals
# entry: RSI < buy_rsi AND ml_prediction > threshold
# exit: RSI > sell_rsi OR ml_prediction < threshold
```

### 3.9 Configuration Parameters

```python
freqai_config = {
    "train_period_days": 15,        # Training window
    "backtest_period_days": 7,       # Default forward test
    "include_timeframes": ["3m", "15m", "1h"],
    "include_corr_pairlist": ["BTC/USDT", "ETH/USDT"],
    "label_period_candles": 20,      # Forward-looking target
    "include_shifted_candles": 2,    # Lagged features
    "DI_threshold": 0.9,            # Dissimilarity Index (novelty filter)
    "weight_factor": 0.9,           # Recency weighting
    "use_SVM_to_remove_outliers": True,  # Outlier detection
    "indicator_periods_candles": [10, 20],
}
```

### 3.10 Portability Assessment

| Feature | Portability | Notes |
|---------|-------------|-------|
| Feature expansion framework | 5/5 | Direct pandas implementation |
| Walk-forward split | 5/5 | Already have similar in our walk_forward.py |
| Recency-weighted training | 5/5 | Simple exponential decay weights |
| BB width as feature | 5/5 | Good mean-reversion predictor |
| Close / BB_lower ratio | 5/5 | Normalized distance from support |
| Relative volume | 5/5 | Volume normalization across regimes |
| Day-of-week | 5/5 | Seasonal signal (Category N) |
| Multi-period auto-expansion | 4/5 | Config-driven, need adapter |
| DI_threshold novelty filter | 3/5 | Needs ML infrastructure |

---

# CONSOLIDATED SIGNAL CATALOG — Prioritized for Implementation

## Tier 1: IMMEDIATE Implementation (Sprint 1)

| # | Signal | Source | Formula | Alpha Type | Portability |
|---|--------|--------|---------|------------|-------------|
| 1 | **DV2** | Backtrader | `PercentRank(SMA(close/((H+L)/2), 2), 252) * 100` | Mean-rev | 5/5 |
| 2 | **KST** | Backtrader/NFI | `w1*SMA(ROC10,10) + w2*SMA(ROC15,10) + w3*SMA(ROC20,10) + w4*SMA(ROC30,10)` | Multi-horizon momentum | 5/5 |
| 3 | **TSI** | Backtrader | `100 * EMA(EMA(pc,25),13) / EMA(EMA(abs(pc),25),13)` | Smoothed momentum | 5/5 |
| 4 | **Graduated RSI Exit** | NFI | Profit-tiered RSI thresholds (see table 1.4) | Exit management | 5/5 |
| 5 | **Multi-TF RSI filter** | NFI | `RSI_3 > X OR RSI_3_1h > Y OR RSI_3_4h > Z` | Protection | 5/5 |
| 6 | **WILLR_480** | NFI | `Williams %R with period=480` | Long-term overbought/oversold | 5/5 |
| 7 | **Relative Volume** | FreqAI | `volume / volume.rolling(period).mean()` | Volume regime | 5/5 |
| 8 | **BB Width** | FreqAI/NFI | `(BBU - BBL) / BBM` (stds=2.2) | Volatility regime | 5/5 |

## Tier 2: HIGH VALUE (Sprint 2)

| # | Signal | Source | Formula | Alpha Type | Portability |
|---|--------|--------|---------|------------|-------------|
| 9 | **Hurst Exponent** | Backtrader | `2 * polyfit(log10(lags), log10(tau), 1)[0]` | Regime detection | 4/5 |
| 10 | **Laguerre RSI** | Backtrader | 4-tap Laguerre filter + RSI, gamma=0.5 | Fast mean-rev | 4/5 |
| 11 | **PGO** | Backtrader | `(close - SMA) / ATR` | Normalized breakout | 5/5 |
| 12 | **RSI Change Rate** | NFI | `(RSI - RSI.shift(1)) / RSI.shift(1) * 100` | Momentum acceleration | 5/5 |
| 13 | **CMF 20** | NFI/Backtrader | Chaikin Money Flow, period=20 | Volume-weighted trend | 5/5 |
| 14 | **UO 7-14-28** | NFI/Backtrader | `100 * (4*avg7 + 2*avg14 + avg28) / 7` | Multi-period mean-rev | 5/5 |
| 15 | **Vortex** | Backtrader | `VM+/TR_sum vs VM-/TR_sum` | Trend direction | 5/5 |
| 16 | **RMI(20,5)** | Backtrader | RSI with lookback=5 instead of 1 | Smoothed overbought/oversold | 5/5 |

## Tier 3: OVERLAY / SUPPORTING

| # | Signal | Source | Formula | Alpha Type | Portability |
|---|--------|--------|---------|------------|-------------|
| 17 | **Walk-forward recency weighting** | FreqAI | Exponential decay on training samples | Training improvement | 5/5 |
| 18 | **Day-of-week** | FreqAI | `date.dt.dayofweek` | Seasonal | 5/5 |
| 19 | **Close/BB_lower ratio** | FreqAI | `close / bollinger_lower` | Distance from support | 5/5 |
| 20 | **DPO** | Backtrader | `close - SMA.shift(period/2+1)` | Cycle identification | 5/5 |
| 21 | **Acc/Dec Oscillator** | Backtrader | `AO - SMA(AO, 5)` | Momentum acceleration | 5/5 |
| 22 | **BB extension signal** | NFI | `close > BB_upper_1h * 1.14` | Extreme overbought | 5/5 |
| 23 | **Consecutive BB breakout** | NFI | `close > BBU for N consecutive bars` | Overbought persistence | 5/5 |
| 24 | **Wick analysis** | NFI | `top_wick_pct, bot_wick_pct` | Rejection candles | 5/5 |
| 25 | **Empty candle filter** | NFI | `count(volume=0) in 288 bars` | Liquidity filter | 3/5 |

---

## Implementation Priority Mapping to Our Categories

| Signal | Our Strategy Category | Target File |
|--------|----------------------|-------------|
| DV2 | J6-DV2MeanRev | mean_reversion.py |
| KST | I1/I2 improvement | momentum.py |
| TSI | Regime filter | regimes.py or signals.py |
| Graduated RSI exit | Ensemble overlay | ensemble.py |
| Multi-TF RSI | Protection filter | All strategies |
| WILLR_480 | J-series exit improvement | mean_reversion.py |
| Hurst Exponent | Regime detection L-series | regimes.py |
| Laguerre RSI | J-series fast signals | mean_reversion.py |
| PGO | K-series factor | factor.py |
| Walk-forward weighting | Walk-forward improvement | walk_forward.py |
| BB Width/Close ratio | Vol regime signals | signals.py |
| Relative Volume | Volume factor | signals.py |
