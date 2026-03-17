"""Command-line interface for running ad-hoc backtests."""

from __future__ import annotations

import argparse
import json
from typing import Dict

import matplotlib.pyplot as plt

from financial_algorithms.backtest import run_backtest
from financial_algorithms.backtest.strategy_registry import registry
from financial_algorithms.data import load_daily_prices


def _summarize_prices(prices) -> None:
    print(f"  Data shape: {prices.shape[0]} rows x {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index.min()} to {prices.index.max()}")
    missing = prices.isna().sum().sum()
    if missing:
        pct = missing / prices.size * 100
        print(f"  Missing values: {missing} ({pct:.2f}% of matrix) [filled forward/backward in engine]")
    dupes = prices.index.duplicated().sum()
    if dupes:
        print(f"  Warning: {dupes} duplicate dates in index; engine will raise if not deduped")


def _parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple strategy backtest.")
    parser.add_argument('--strategy', default='sma', help="Strategy name (see registry)")
    parser.add_argument('--tickers', nargs='+', default=['AAPL', 'MSFT', 'AMZN'], help="List of tickers")
    parser.add_argument('--years', type=int, default=3, help="Limit to last N years (0 = all)")
    parser.add_argument('--capital', type=float, default=100000.0, help="Starting capital")
    parser.add_argument('--fast', type=int, help="Fast period (for SMA-like strategies)")
    parser.add_argument('--slow', type=int, help="Slow period (for SMA-like strategies)")
    parser.add_argument('--sl', type=float, help="Stop-loss/threshold param where applicable")
    parser.add_argument('--cost-bps', type=float, default=1.0, help="Commission/fees per turnover in basis points")
    parser.add_argument('--slippage-bps', type=float, default=5.0, help="Slippage per turnover in basis points")
    parser.add_argument('--allow-short', action='store_true', help="Enable short signals (-1) when provided")
    parser.add_argument('--risk-free-rate', type=float, default=0.0, help="Annual risk-free rate (e.g., 0.05 for 5%)")
    parser.add_argument('--max-gross-leverage', type=float, default=1.0, help="Cap on sum of absolute weights")
    parser.add_argument('--max-position-weight', type=float, default=1.0, help="Cap on absolute weight per position")
    parser.add_argument('--max-signal-abs', type=float, default=5.0, help="Clip raw signal magnitudes to this abs value")
    parser.add_argument('--strict-data', action='store_true', help="Raise on duplicate dates or NaNs after fill")
    parser.add_argument('--report-prefix', type=str, help="If set, saves metrics/equity/plot with this prefix")
    return parser.parse_args(args=args)


def _resolve_strategy(name: str) -> Dict:
    strat = registry.get(name)
    if strat is None:
        available = ', '.join(registry.list().keys())
        raise SystemExit(f"Unknown strategy '{name}'. Available: {available}")
    return strat


def main(args: list[str] | None = None) -> None:
    cli_args = _parse_args(args)

    print("=" * 60)
    print("Strategy Backtest")
    print("=" * 60)

    print("\n[1/4] Loading price data...")
    prices = load_daily_prices(cli_args.tickers)
    prices = prices.last(f"{cli_args.years}Y") if cli_args.years else prices
    print(f"  Loaded {len(prices)} days for {len(cli_args.tickers)} tickers")
    _summarize_prices(prices)

    strat = _resolve_strategy(cli_args.strategy)
    print(f"\n[2/4] Generating signals with strategy '{cli_args.strategy}'...")
    params = strat['params'].copy()
    if cli_args.fast:
        params['fast'] = cli_args.fast
    if cli_args.slow:
        params['slow'] = cli_args.slow
    if cli_args.sl is not None:
        params['sl'] = cli_args.sl

    signals = strat['func'](prices, **params)
    if hasattr(signals, 'columns') and signals.shape[1] >= 1 and not signals.equals(prices):
        signal_cols = [c for c in signals.columns if 'signal' in c.lower()]
        if signal_cols:
            signals = signals[signal_cols]
    signals = signals.reindex(prices.index).fillna(0)
    print(f"  Signals generated: {len(signals)} rows")

    print("\n[3/4] Running backtest...")
    results = run_backtest(
        prices,
        signals,
        initial_capital=cli_args.capital,
        cost_bps=cli_args.cost_bps,
        slippage_bps=cli_args.slippage_bps,
        allow_short=cli_args.allow_short,
        risk_free_rate=cli_args.risk_free_rate,
        max_gross_leverage=cli_args.max_gross_leverage,
        max_position_weight=cli_args.max_position_weight,
        max_signal_abs=cli_args.max_signal_abs,
        strict_data=cli_args.strict_data,
    )

    print("\n[4/4] Results")
    print("=" * 60)
    print(f"Initial Capital: ${cli_args.capital:,.2f}\n")
    for key, value in results['metrics'].items():
        print(f"{key:.<25} {value}")

    print("\n" + "=" * 60)
    print("Generating equity curve plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    results['equity_curve'].plot(ax=ax1, linewidth=2)
    ax1.set_title(f"{cli_args.strategy} Backtest: Equity Curve", fontsize=14)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.grid(alpha=0.3)
    ax1.axhline(cli_args.capital, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.legend()

    results['num_positions'].plot(ax=ax2, linewidth=1, alpha=0.7)
    ax2.set_title("Number of Active Positions", fontsize=14)
    ax2.set_ylabel("# Positions")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plot_path = f"{cli_args.report_prefix}_equity.png" if cli_args.report_prefix else "backtest_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f"✓ Plot saved to: {plot_path}")
    plt.show()

    if cli_args.report_prefix:
        metrics_path = f"{cli_args.report_prefix}_metrics.json"
        equity_path = f"{cli_args.report_prefix}_equity.csv"
        results['equity_curve'].to_csv(equity_path, header=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(results['metrics'], f, indent=2)
        print(f"✓ Metrics saved to: {metrics_path}")
        print(f"✓ Equity curve saved to: {equity_path}")

    print("\n" + "=" * 60)
    print("Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover
    main()
