"""
CLI entry point for the BTC 4h backtesting engine.

Usage examples (one flag per line — PowerShell 5.1 doesn't support && chaining):
    python run.py
    python run.py --start 2021-01-01 --end 2024-01-01 --fast 20 --slow 100
    python run.py --no-plot
"""
import argparse
import logging

from data.loader import load_candles
from engine.engine import run_backtest
from metrics.metrics import compute_all_metrics, print_metrics_table, print_trade_blotter
from reporting.plot import plot_results
from strategies.ma_crossover import MACrossover

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BTC 4h correctness-first backtesting engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol",    default="BTCUSDT",  help="Binance trading pair")
    parser.add_argument("--start",     default="2022-01-01", help="Start date YYYY-MM-DD (UTC)")
    parser.add_argument("--end",       default=None,         help="End date YYYY-MM-DD (UTC); defaults to today")
    parser.add_argument("--capital",   type=float, default=10_000.0, help="Initial capital (USDT)")
    parser.add_argument("--fee",       type=float, default=0.001,    help="Taker fee per side (0.001 = 0.1%%)")
    parser.add_argument("--slippage",  type=float, default=0.0,      help="Fill slippage in basis points")
    parser.add_argument("--fast",      type=int,   default=50,        help="Fast SMA period")
    parser.add_argument("--slow",      type=int,   default=200,       help="Slow SMA period")
    parser.add_argument("--stop",      type=float, default=0.08,      help="Drawdown stop threshold (0.08 = 8%%)")
    parser.add_argument("--fraction",  type=float, default=1.0,       help="Position size as fraction of equity")
    parser.add_argument("--no-plot",   action="store_true",           help="Skip chart generation")
    parser.add_argument("--save-plot", default="backtest_results.png", help="Chart output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"\nLoading {args.symbol} 4h candles ({args.start} to {args.end or 'today'})...")
    data = load_candles(args.symbol, "4h", args.start, args.end)
    print(f"Loaded {len(data):,} candles  ({data.index[0].date()} to {data.index[-1].date()})")

    strategy = MACrossover(
        fast_period=args.fast,
        slow_period=args.slow,
        stop_pct=args.stop,
    )
    print(
        f"\nRunning SMA({args.fast},{args.slow}) crossover + {args.stop*100:.0f}% drawdown stop..."
    )
    result = run_backtest(
        strategy=strategy,
        data=data,
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_bps=args.slippage,
        position_fraction=args.fraction,
    )

    metrics = compute_all_metrics(result)
    print_metrics_table(metrics)

    print("\nTrade Blotter:")
    print_trade_blotter(result.trades)

    if not args.no_plot:
        plot_results(result.equity, result.data, result.trades, save_path=args.save_plot)


if __name__ == "__main__":
    main()
