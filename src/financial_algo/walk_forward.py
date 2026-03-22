"""Walk-forward validation framework.

Tests whether strategies that performed well in-sample continue to
deliver alpha out-of-sample.  This is the gold-standard for detecting
overfitting in quantitative strategies.

Design
------
- **Expanding training window**: starts at ``min_train_years`` and grows
  each fold.  Test window is always ``test_years`` long.
- **Per-strategy WF**: run each strategy on train → compute Sharpe, then
  test on unseen data.  Concatenate all test-period returns to get
  aggregate out-of-sample metrics.
- **Ensemble WF**: recompute ensemble prior_weights from training-period
  Sharpes each fold (no future peeking).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from financial_algo.backtest import BacktestConfig, backtest, compute_metrics
from financial_algo.regimes import detect_regime
from financial_algo.strategies.base import Strategy
from financial_algo.strategies.ensemble import EnsembleConfig, EnsembleStrategy


@dataclass
class WalkForwardConfig:
    """Parameters for walk-forward validation."""

    min_train_years: int = 3       # minimum training window (years)
    test_years: int = 1            # test window length (years)
    step_years: int = 1            # step between folds (years)
    warmup_days: int = 252         # extra warm-up before train start

    # Backtest config for both train and test
    bt_config: BacktestConfig | None = None


def _year_offset(base: pd.Timestamp, years: int) -> pd.Timestamp:
    """Add *years* to a timestamp (approximate with 365 days)."""
    return base + pd.DateOffset(years=years)


def generate_folds(
    prices: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> list[dict]:
    """Generate train/test fold boundaries.

    Returns list of dicts with keys:
        fold, train_start, train_end, test_start, test_end
    """
    data_start = prices.index[0]
    data_end = prices.index[-1]

    folds = []
    fold_idx = 0
    train_start = data_start

    while True:
        train_end = _year_offset(data_start, cfg.min_train_years + fold_idx * cfg.step_years)
        test_start = train_end
        test_end = _year_offset(test_start, cfg.test_years)

        # Stop if test window extends past data
        if test_end > data_end + pd.Timedelta(days=30):
            break

        # Clip test_end to data_end
        if test_end > data_end:
            test_end = data_end

        # Need at least 60 trading days in test
        test_mask = (prices.index >= str(test_start.date())) & (prices.index <= str(test_end.date()))
        if test_mask.sum() < 60:
            break

        folds.append({
            "fold": fold_idx + 1,
            "train_start": train_start,
            "train_end": train_end,
            "test_start": test_start,
            "test_end": test_end,
        })
        fold_idx += 1

    return folds


def walk_forward_strategy(
    strategy: Strategy,
    prices: pd.DataFrame,
    regime: pd.Series,
    cfg: WalkForwardConfig,
    needs_regime: bool = True,
    vix: pd.Series | None = None,
) -> dict:
    """Run walk-forward validation for a single strategy.

    Returns
    -------
    dict with keys:
        - oos_metrics: aggregate out-of-sample metrics
        - per_fold: list of per-fold metrics dicts
        - oos_returns: concatenated out-of-sample daily returns
    """
    bt_cfg = cfg.bt_config or BacktestConfig(
        tx_cost_bps=5.0,
        leverage_cost_annual=0.015,
        short_cost_annual=0.005,
        initial_capital=1_000_000.0,
        vol_target=0.20,
        max_drawdown_trigger=-0.25,
        drawdown_recovery_rate=0.10,
    )

    folds = generate_folds(prices, cfg)
    all_oos_returns = []
    per_fold = []

    for fold in folds:
        ts = str(fold["train_start"].date())
        te = str(fold["train_end"].date())
        xs = str(fold["test_start"].date())
        xe = str(fold["test_end"].date())

        # Slice price data — include warmup before train for indicators
        warmup_start = fold["train_start"] - pd.Timedelta(days=cfg.warmup_days + 30)
        p_full = prices.loc[str(warmup_start.date()):xe].copy()
        r_full = regime.reindex(p_full.index).ffill().bfill()

        # Generate weights on full range (strategy needs warmup)
        if needs_regime:
            weights = strategy.backtest_weights(p_full, r_full)
        else:
            weights = strategy.backtest_weights(p_full)

        # -- Training period metrics --
        train_mask = (p_full.index >= ts) & (p_full.index < te)
        if train_mask.sum() < 60:
            continue

        p_train = p_full.loc[train_mask]
        w_train = weights.loc[train_mask]
        try:
            train_result = backtest(p_train, w_train, bt_cfg)
            train_m = train_result["metrics"]
        except Exception:
            train_m = {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

        # -- Test period metrics (out-of-sample) --
        test_mask = (p_full.index >= xs) & (p_full.index <= xe)
        if test_mask.sum() < 30:
            continue

        p_test = p_full.loc[test_mask]
        w_test = weights.loc[test_mask]
        try:
            test_result = backtest(p_test, w_test, bt_cfg)
            test_m = test_result["metrics"]
            all_oos_returns.append(test_result["returns"])
        except Exception:
            test_m = {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

        per_fold.append({
            "fold": fold["fold"],
            "train": f"{ts} -> {te}",
            "test": f"{xs} -> {xe}",
            "train_sharpe": train_m.get("sharpe", 0.0),
            "test_sharpe": test_m.get("sharpe", 0.0),
            "train_cagr": train_m.get("cagr", 0.0),
            "test_cagr": test_m.get("cagr", 0.0),
            "test_maxdd": test_m.get("max_drawdown", 0.0),
        })

    # Aggregate out-of-sample returns
    if all_oos_returns:
        oos_ret = pd.concat(all_oos_returns)
        # Remove duplicate dates (overlapping folds) — keep first
        oos_ret = oos_ret[~oos_ret.index.duplicated(keep="first")]
        oos_ret = oos_ret.sort_index()
        oos_metrics = compute_metrics(oos_ret)
    else:
        oos_ret = pd.Series(dtype=float)
        oos_metrics = {}

    return {
        "oos_metrics": oos_metrics,
        "per_fold": per_fold,
        "oos_returns": oos_ret,
    }


def walk_forward_ensemble(
    strategies: list[Strategy],
    prices: pd.DataFrame,
    regime: pd.Series,
    cfg: WalkForwardConfig,
    ensemble_cfg_template: EnsembleConfig | None = None,
    vix: pd.Series | None = None,
) -> dict:
    """Run walk-forward validation for the full ensemble.

    For each fold, recompute ensemble prior_weights from training-period
    Sharpes (no future peeking).  Then test on the unseen test period.

    Returns same structure as walk_forward_strategy.
    """
    bt_cfg = cfg.bt_config or BacktestConfig(
        tx_cost_bps=5.0,
        leverage_cost_annual=0.015,
        short_cost_annual=0.005,
        initial_capital=1_000_000.0,
        vol_target=0.20,
        max_drawdown_trigger=None,
        strategy_dd_scale_start=-0.15,
        strategy_dd_scale_end=-0.25,
    )

    base_cfg = ensemble_cfg_template or EnsembleConfig()
    folds = generate_folds(prices, cfg)
    all_oos_returns = []
    per_fold = []

    for fold in folds:
        ts = str(fold["train_start"].date())
        te = str(fold["train_end"].date())
        xs = str(fold["test_start"].date())
        xe = str(fold["test_end"].date())

        warmup_start = fold["train_start"] - pd.Timedelta(days=cfg.warmup_days + 30)
        p_full = prices.loc[str(warmup_start.date()):xe].copy()
        r_full = regime.reindex(p_full.index).ffill().bfill()

        # -- Step 1: Compute per-strategy Sharpe on training data --
        train_sharpes = []
        for strat in strategies:
            try:
                w = strat.backtest_weights(p_full, r_full)
                train_mask = (p_full.index >= ts) & (p_full.index < te)
                p_tr = p_full.loc[train_mask]
                w_tr = w.loc[train_mask]
                res = backtest(p_tr, w_tr, bt_cfg)
                s = max(res["metrics"].get("sharpe", 0.0), 0.01)
            except Exception:
                s = 0.01
            train_sharpes.append(s)

        # -- Step 2: Build ensemble with training-derived weights --
        prior_w = [max(s, 0.01) ** 2 for s in train_sharpes]
        fold_cfg = EnsembleConfig(
            use_inverse_vol=base_cfg.use_inverse_vol,
            correlation_lookback=base_cfg.correlation_lookback,
            max_gross_leverage=base_cfg.max_gross_leverage,
            max_single_weight=base_cfg.max_single_weight,
            dd_scale_start=base_cfg.dd_scale_start,
            dd_scale_end=base_cfg.dd_scale_end,
            prior_weights=prior_w,
            correlation_hedge_enabled=base_cfg.correlation_hedge_enabled,
            correlation_hedge_threshold=base_cfg.correlation_hedge_threshold,
            correlation_hedge_max=base_cfg.correlation_hedge_max,
            safe_haven_tickers=base_cfg.safe_haven_tickers,
            vol_regime_scaling=base_cfg.vol_regime_scaling,
            vol_elevated_threshold=base_cfg.vol_elevated_threshold,
            vol_crisis_threshold=base_cfg.vol_crisis_threshold,
            leverage_elevated=base_cfg.leverage_elevated,
            leverage_crisis=base_cfg.leverage_crisis,
        )
        ensemble = EnsembleStrategy(strategies, fold_cfg)
        ens_w = ensemble.backtest_weights(p_full, r_full)

        # -- Training metrics --
        train_mask = (p_full.index >= ts) & (p_full.index < te)
        try:
            res_train = backtest(p_full.loc[train_mask], ens_w.loc[train_mask], bt_cfg)
            train_m = res_train["metrics"]
        except Exception:
            train_m = {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

        # -- Test metrics (OOS) --
        test_mask = (p_full.index >= xs) & (p_full.index <= xe)
        if test_mask.sum() < 30:
            continue
        try:
            res_test = backtest(p_full.loc[test_mask], ens_w.loc[test_mask], bt_cfg)
            test_m = res_test["metrics"]
            all_oos_returns.append(res_test["returns"])
        except Exception:
            test_m = {"sharpe": 0.0, "cagr": 0.0, "max_drawdown": 0.0}

        per_fold.append({
            "fold": fold["fold"],
            "train": f"{ts} -> {te}",
            "test": f"{xs} -> {xe}",
            "train_sharpe": train_m.get("sharpe", 0.0),
            "test_sharpe": test_m.get("sharpe", 0.0),
            "train_cagr": train_m.get("cagr", 0.0),
            "test_cagr": test_m.get("cagr", 0.0),
            "test_maxdd": test_m.get("max_drawdown", 0.0),
            "train_sharpes_used": [round(s, 2) for s in train_sharpes],
        })

    if all_oos_returns:
        oos_ret = pd.concat(all_oos_returns)
        oos_ret = oos_ret[~oos_ret.index.duplicated(keep="first")]
        oos_ret = oos_ret.sort_index()
        oos_metrics = compute_metrics(oos_ret)
    else:
        oos_ret = pd.Series(dtype=float)
        oos_metrics = {}

    return {
        "oos_metrics": oos_metrics,
        "per_fold": per_fold,
        "oos_returns": oos_ret,
    }
