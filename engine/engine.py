"""
Event-driven backtest loop.

Bar-by-bar execution (not vectorized) makes lookahead impossible by construction:
when the strategy runs on bar t, only bars 0..t exist in the Context.
Vectorized execution would be faster but hides the very class of bugs this engine
exists to prevent; vectorizing is a deliberate later optimization.

Execution order per bar t:
    1. Fill any orders queued on bar t-1 at bar t's OPEN price.
    2. Record whether a position is held at bar-start (for exposure metric).
    3. Call strategy.on_bar(context) — strategy may queue orders for bar t+1.
    4. If this is the final bar and a position is still open, force-close it
       at bar t's CLOSE (no bar t+1 exists to fill a queued order).
    5. Mark equity to market at bar t's CLOSE.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.broker import Broker, Trade
from engine.context import Context, Order, PositionInfo
from strategies.base import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series            # per-bar equity curve (one entry per bar)
    trades: list[Trade]          # completed round-trips
    data: pd.DataFrame           # the OHLCV data used
    in_position: pd.Series       # bool per bar: True if position held at bar-start


def run_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_bps: float = 0.0,
    position_fraction: float = 1.0,
) -> BacktestResult:
    """
    Run the backtest. Returns equity curve, trade blotter, and per-bar position state.

    Parameters
    ----------
    strategy         : Strategy instance (init() has not been called yet)
    data             : OHLCV DataFrame indexed by open_time (UTC), sorted ascending
    initial_capital  : starting cash in USDT
    fee_rate         : taker fee per side (0.001 = 0.1%)
    slippage_bps     : fill slippage in basis points (fills move against you)
    position_fraction: fraction of equity deployed per entry (1.0 = 100%)
    """
    broker = Broker(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        position_fraction=position_fraction,
    )

    # strategy.init() receives the full dataset ONCE — the only place full data is allowed.
    # It is expected to precompute only backward-looking (strictly causal) indicators.
    strategy.init(data)

    pending_orders: list[Order] = []
    n_bars = len(data)
    in_position_list: list[bool] = []

    for t in range(n_bars):
        bar = data.iloc[t]
        timestamp = data.index[t]
        is_final_bar = t == n_bars - 1

        # ── Step 1: Fill orders from bar t-1 at bar t's OPEN ──────────────────
        # This is the "next-bar open" fill rule: signals queue here, fill next bar.
        if pending_orders:
            open_price = float(bar["open"])
            for order in pending_orders:
                if order.action == "buy":
                    broker.fill_buy(open_price, timestamp, bar_idx=t)
                elif order.action == "close":
                    broker.fill_close(open_price, timestamp, reason=order.reason, bar_idx=t)
            pending_orders.clear()

        # ── Step 2: Record position state at bar-start (after fills) ──────────
        in_position_list.append(broker.has_position)

        # ── Step 3: Build context (lookahead firewall) and call strategy ───────
        position_info = PositionInfo(
            has_position=broker.has_position,
            entry_price=broker.entry_price or 0.0,
            size=broker.btc_held,
        )
        context = Context(data, t, position_info)
        strategy.on_bar(context)

        if not is_final_bar:
            pending_orders = context.pop_orders()
        else:
            # Discard any orders the strategy queued (no bar t+1 to fill them)
            context.pop_orders()
            # Force-close any remaining position at this bar's CLOSE
            if broker.has_position:
                broker.fill_close(
                    float(bar["close"]), timestamp, reason="end_of_data", bar_idx=t
                )

        # ── Step 5: Mark equity to market at bar t's CLOSE ────────────────────
        broker.mark_to_market(timestamp, float(bar["close"]))

    return BacktestResult(
        equity=broker.equity_series(),
        trades=broker.trades,
        data=data,
        in_position=pd.Series(in_position_list, index=data.index, name="in_position"),
    )
