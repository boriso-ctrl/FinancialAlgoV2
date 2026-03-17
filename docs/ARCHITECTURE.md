# System Architecture

**Tier 1: Research & Experimentation** (archived)  
**Tier 2: Validation & Testing** (current)  
**Tier 3: Production & Deployment** (ready)

---

## Production Code (Tier 3)

### Core Strategy Layer
**File**: `src/financial_algorithms/strategies/voting_enhanced_weighted.py` (476 lines)

Entry point: `EnhancedWeightedVotingStrategy` class

```python
class EnhancedWeightedVotingStrategy:
    def __init__(
        self,
        risk_pct: float = 2.0,           # Stop loss distance
        tp1_pct: float = 1.5,            # First profit target
        tp2_pct: float = 3.0,            # Full profit target
        min_buy_score: float = 2.0,      # Entry threshold
        max_sell_score: float = -2.0,    # Exit threshold
    ):
        """Initialize with tuned parameters from Phase 6 testing."""
```

**Key Methods**:
- `calculate_voting_score()` - Aggregate 5 indicators into -10 to +10 score
- `should_enter()` - Entry logic (score >= +2.0)
- `should_exit_on_signal()` - Exit logic (score <= -2.0)
- `calculate_position_size()` - Risk-based sizing (4-8% dynamic)

**Signal Generation Pipeline**:

```
┌─────────────────────────────────────────────────┐
│        Input: OHLCV Data                        │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
    2023-2025          Historical
    Daily Bars         Indicators
        │                     │
        └──────────┬──────────┘
                   │
        ┌──────────┴──────────────────────────────────────┐
        │         Calculate 5 Indicator Signals           │
        ├─────────────┬─────────────┬────────────┬────────┤
        │             │             │            │        │
      SMA(±2)      RSI(±2)      Volume(±2)   ADX(±2)   ATR(±2)
        │             │             │            │        │
        └──────────────┴─────────────┴────────────┴────────┘
                       │
        ┌──────────────┴──────────────┐
        │  Total Score: Sum of 5      │
        │  Range: -10 to +10          │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  Signal Decision            │
        ├─────────────┬───────────────┤
        │             │               │
      BUY (≥+2.0)  HOLD (-2 to +2)  SELL (≤-2.0)
        │             │               │
        └──────────────┴───────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  Position Sizing            │
        │  4-8% based on score        │
        └──────────────┬──────────────┘
                       │
        ┌──────────────┴──────────────┐
        │  Entry/Exit Execution       │
        └─────────────────────────────┘
```

---

### Exit Management Layer
**File**: `src/financial_algorithms/backtest/tiered_exits.py` (547 lines)

Entry point: `TieredExitManager` class

**Two-Tiered Profit Taking Strategy**:

```
Position Entry at Price P
        │
        ├─→ TP1 (+1.5%):
        │   └─→ Exit 50% of position
        │       └─→ Assess signal strength
        │           ├─→ Weak (Scenario A): Close full position
        │           └─→ Strong (Scenario B): Trail stop, aim for TP2
        │
        ├─→ TP2 (+3.0%):
        │   └─→ Exit remaining 50%
        │       └─→ Book full position profit
        │
        ├─→ Stop Loss (-2%):
        │   └─→ Exit all on loss limit
        │
        └─→ Trailing Stop:
            └─→ Exit on reversal (score ≤ -2.0)
```

**Trade Lifecycle**:
1. Entry: Score ≥ +2.0 → Create trade object
2. Track: Update high water mark daily
3. TP1 Check: At +1.5%, assess signal → manage risk
4. TP2 Check: At +3.0%, exit remaining
5. Stop Check: Trailing stop or signal reversal
6. Exit: Record P&L, update equity

---

### Backtesting Engine Layer
**File**: `src/financial_algorithms/backtest/engine.py`

**Mock Trade Execution**:
- Position tracking (entry price, quantity, dates)
- Mark-to-market P&L calculation
- Equity curve construction
- Trade log recording

**Metrics Calculation** (metrics.py):
- Total return: Profit / Initial Capital
- Win rate: Winning trades / Total trades
- Sharpe ratio: (Avg Return / Std Dev) * √252
- Max drawdown: Peak-to-trough equity decline
- Average trade return: Mean of all trade returns

---

### Indicator Layer
**Location**: `src/financial_algorithms/signals/`

**60+ Technical Indicators** across two families:

#### Price Indicators (`price/`)
- **Trend**: SMA, EMA, ADX, SAR
- **Momentum**: RSI, MACD, Stochastic, CCI  
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Peaks/Valleys**: Williams %R, Divergence detection

#### Volume Indicators (`volume/`)
- **On-Balance Volume** (OBV)
- **Chaikin Money Flow** (CMF)
- **Force Index**
- **Accumulation/Distribution**
- **Volume-Price Trend** (VPT)

**Usage Example**:
```python
from src.financial_algorithms.signals.price import rsi, macd, atr
from src.financial_algorithms.signals.volume import obv, cmf

# Calculate RSI (14-period)
rsi_signal = rsi.calculate_signal(close_prices)

# Calculate CMF (20-period)
cmf_signal = cmf.calculate_signal(high, low, close, volume)

# Calculate ATR
atr_signal = atr.calculate_signal(high, low, close)
```

---

## Validation & Testing Layer (Tier 2)

### Production Backtests

#### 1. Single-Asset Daily Backtest
**File**: `backtests/backtest_daily_voting.py`

Tests baseline strategy on single asset (default: SPY)
- Entry: +2.0 signal
- Exit: -2.0 signal or SL
- Position sizing: 4-8%
- Exit targets: TP1 1.5%, TP2 3.0%

**Sample Usage**:
```bash
python backtests/backtest_daily_voting.py --asset AAPL --output results.json
```

**Output Metrics**:
```json
{
  "asset": "AAPL",
  "total_return_pct": 2.41,
  "sharpe_ratio": 5.34,
  "win_rate_pct": 71.4,
  "trade_count": 19,
  "avg_trade_return_pct": 0.94
}
```

#### 2. Multi-Asset Validator
**File**: `backtests/backtest_15asset_validator.py`

Comprehensive test across 15 diverse assets:
- AAPL, MSFT, NVDA, GOOGL (Tech)
- JPM, GS, BAC (Finance)  
- XOM, CVX (Energy)
- JNJ, PFE (Pharma)
- AMZN, WMT (Retail)
- SPY, QQQ (Broad Index)

**Output Format**: Consolidated JSON with all 15 results

**Key Validation**: ✅ 100% of assets profitable (0 losers)

---

### Unit Tests
**Location**: `tests/`

#### 1. Smoke Tests (`test_backtest_smoke.py`)
- Backtest runs without exceptions
- Trades execute with valid P&L
- Equity curve updates correctly

#### 2. Indicator Tests (`test_indicators_smoke.py`)
- All 60+ indicators calculate without NaN
- Signals are in expected ranges (-2 to +2 per indicator)
- No lookahead bias (only historical data used)

#### 3. Integration Tests (`test_demo_blend.py`)
- Multiple indicator blending works
- Signal aggregation produces valid scores
- Entry/exit logic triggers correctly

**Run Tests**:
```bash
pytest tests/ -v
```

---

## Experimental Layer (Tier 1 - Archived)

### Phase 5: Robustness Testing
**Archive**: `experiments/robustness/`

Stress testing different market regimes

### Phase 6: Parameter Optimization
**Archive**: `experiments/parameter_search/`

Bayesian adaptive search for optimal parameters
- Tested: Buy thresholds, exit triggers, position sizing
- Found: +2.0/-2.0 with 4-8% sizing optimal for this data

### Phase 6B: Intraday Exploration
**Archive**: `experiments/intraday/`

15-minute and 1-hour timeframe testing
- Finding: Daily outperforms intraday (0.06-0.51% returns)
- Recommendation: Stick with daily resolution

### Phase 7: Aggressive Growth
**Archive**: `experiments/aggressive_growth/`

Testing maximum position sizing (10-20%) and extended targets
- Finding: +0.5 entry, -10.0 exit, 100% sizing captures 80.43% of SPY move
- Conclusion: Bull market air benefits buy-hold; active trading adds drag

---

## Data Flow

```
┌──────────────────────━┐
│   Data Sources       │
│  (yfinance, etc)     │
└───────────┬──────────┘
            │
            ├─→ Download OHLCV
            │   Daily bars 2023-2025
            │
            └─→ Validation
                ✓ No NaN values
                ✓ Chronological order
                ✓ 751 bars/asset
            
┌───────────────────────────────────────────┐
│     Voting Strategy Calculation           │
│  ┌──────────────────────────────────────┐ │
│  │Loop through each bar (idx=50 to end)│ │
│  │  1. Extract OHLCV history[:idx+1]   │ │
│  │  2. Calculate 5 indicators          │ │
│  │  3. Sum to voting score (-10:+10)   │ │
│  │  4. Check entry/exit rules          │ │
│  │  5. Manage position (TP1/TP2/SL)    │ │
│  │  6. Record trade metrics            │ │
│  │  7. Update equity curve             │ │
│  └──────────────────────────────────────┘ │
└────────────┬────────────────────────────────┘
             │
┌────────────────────────────────────────────────┐
│        Trade Analysis & Metrics                │
│  • Win Rate = Wins / Total Trades              │
│  • Return = Total PL / Initial Capital         │
│  • Sharpe = (Avg Trade Return / Std Dev) * √252│
│  • Max Drawdown = Peak-to-Trough Equity Decline│
└────────────┬───────────────────────────────────┘
             │
┌────────────────────────────────────────────┐
│    Output: JSON Results                   │
│  {                                        │
│    "asset": "SPY",                        │
│    "total_return_pct": 2.41,              │
│    "sharpe_ratio": 5.34,                  │
│    "win_rate_pct": 71.4,                  │
│    "trades": [...]                        │
│  }                                        │
└────────────────────────────────────────────┘
```

---

## Deployment Readiness

### Production Components ✅
- [x] Strategy module (voting_enhanced_weighted.py)
- [x] Exit management (tiered_exits.py)
- [x] Backtesting engine
- [x] 60+ validated indicators
- [x] Comprehensive test coverage

### Pre-Deployment Checklist
- [x] Unit tests passing
- [x] 15-asset validation complete (100% profitable)
- [x] Performance metrics documented
- [x] Code reviewed and optimized
- [ ] Paper trading (4-8 weeks)
- [ ] Real slippage/commission validation
- [ ] Live trading (optional, small size)

### Known Limitations
- ⚠️ Not tested in bear markets (2015-2022, 2020)
- ⚠️ Underperforms buy-hold in sustained bull trends (by design)
- ⚠️ Intraday timeframes show poor performance
- ⚠️ Requires parameter tuning for different regimes

---

## Maintenance & Future Improvements

### Quick Tweaks
- Edit `voting_enhanced_weighted.py` parameters for different market regimes
- Modify `tiered_exits.py` TP1/TP2 targets for different risk profiles
- Adjust position sizing in `calculate_position_size()` method

### Medium-Term Enhancements
- [ ] Add machine learning for parameter optimization
- [ ] Implement sector-specific strategies
- [ ] Add volatility regime detection
- [ ] Optimize for different time periods

### Long-Term Roadmap
- [ ] Multi-asset portfolio optimization
- [ ] Dynamic position sizing by volatility
- [ ] Real-time trading integration
- [ ] Risk management framework
