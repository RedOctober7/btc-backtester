"""
Event-driven backtest loop.

Bar-by-bar execution (not vectorized) makes lookahead impossible by construction:
when the strategy runs on bar t, only bars 0..t exist in the Context.

Execution order per bar t:
    1. Fill any orders queued on bar t-1 at bar t's OPEN price.
    2. FUNDING (perps, leverage runs): on 8h-aligned bars, charge/credit one
       funding interval at the bar's open price. Applied BEFORE the liquidation
       check because eroded margin pulls the liquidation trigger closer — the
       order matters and this is the conservative one.
    3. LIQUIDATION CHECK (leverage > 1 only): if the position's liquidation
       trigger falls inside this bar's range, force-close. Gap-aware: a bar
       that OPENS beyond the trigger fills at the open (you eat the gap), a bar
       that trades through it intrabar fills at the trigger. Checked against
       high/low, never just close — a real exchange liquidates the instant
       price touches the level; close-only checking would let leveraged
       positions "survive" bars that would have wiped them out live.
       ORDERING NOTE (documented ambiguity): when a single bar hits both the
       strategy's stop (evaluated on close, exits next bar) and the liquidation
       trigger (intrabar), liquidation wins. OHLC data cannot reveal which came
       first inside the bar; resolving in favor of liquidation is the
       pessimistic choice, consistent with this engine's design rule.
    4. Record whether a position is held at bar-start (exposure metric).
    5. Call strategy.on_bar(context) — may queue orders for bar t+1.
    6. Final bar: discard queued orders, force-close any open position at close.
    7. Mark equity to market at bar t's CLOSE.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.broker import Broker, Trade
from engine.context import Context, Order, PositionInfo
from strategies.base import Strategy


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: list[Trade]
    data: pd.DataFrame
    in_position: pd.Series


def _is_funding_bar(timestamp: pd.Timestamp) -> bool:
    """Perp funding settles every 8h at 00:00/08:00/16:00 UTC. On 4h bars that
    is every second bar; on other frequencies only the aligned bars charge."""
    return timestamp.hour % 8 == 0 and timestamp.minute == 0


def run_backtest(
    strategy: Strategy,
    data: pd.DataFrame,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
    slippage_bps: float = 0.0,
    position_fraction: float = 1.0,
    leverage: float = 1.0,
    maintenance_margin_rate: float = 0.005,
    funding_rate_8h: float = 0.0,
    liquidation_penalty_bps: float = 0.0,
) -> BacktestResult:
    """
    Run the backtest. All new parameters default to the v1 spot behavior:
    leverage=1, funding=0, penalty=0 reproduces the original engine exactly.

    funding_rate_8h : signed per-8h perp funding rate. Positive = longs pay,
        shorts receive (the common resting state, ~0.0001). Set this whenever
        leverage > 1 or the backtest will flatter every leveraged hold.
    liquidation_penalty_bps : extra adverse slippage applied to forced
        liquidation fills, modeling cascade conditions.
    """
    broker = Broker(
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        position_fraction=position_fraction,
        leverage=leverage,
        maintenance_margin_rate=maintenance_margin_rate,
        funding_rate_8h=funding_rate_8h,
        liquidation_penalty_bps=liquidation_penalty_bps,
    )

    strategy.init(data)

    pending_orders: list[Order] = []
    n_bars = len(data)
    in_position_list: list[bool] = []

    for t in range(n_bars):
        bar = data.iloc[t]
        timestamp = data.index[t]
        is_final_bar = t == n_bars - 1
        bar_open = float(bar["open"])

        # ── 1. Fill queued orders at this bar's OPEN ─────────────────────────
        if pending_orders:
            for order in pending_orders:
                if order.action == "buy":
                    broker.fill_buy(bar_open, timestamp, bar_idx=t)
                elif order.action == "short":
                    broker.fill_short(bar_open, timestamp, bar_idx=t)
                elif order.action == "close":
                    broker.fill_close(bar_open, timestamp, reason=order.reason, bar_idx=t)
            pending_orders.clear()

        # ── 2. Funding (before liquidation: eroded margin moves the trigger) ──
        if _is_funding_bar(timestamp):
            broker.apply_funding(bar_open, timestamp)

        # ── 3. Intrabar, gap-aware liquidation check ─────────────────────────
        broker.maybe_liquidate(
            bar_open, float(bar["high"]), float(bar["low"]), timestamp, t
        )

        # ── 4. Position state at bar-start (after fills, funding, liq) ────────
        in_position_list.append(broker.has_position)

        # ── 5. Strategy sees the world through the lookahead firewall ─────────
        position_info = PositionInfo(
            has_position=broker.has_position,
            entry_price=broker.entry_price or 0.0,
            size=broker.size,
            direction=broker.direction,
        )
        context = Context(data, t, position_info)
        strategy.on_bar(context)

        if not is_final_bar:
            pending_orders = context.pop_orders()
        else:
            context.pop_orders()
            if broker.has_position:
                broker.fill_close(float(bar["close"]), timestamp, reason="end_of_data", bar_idx=t)

        # ── 7. Mark to market at close ────────────────────────────────────────
        broker.mark_to_market(timestamp, float(bar["close"]))

    return BacktestResult(
        equity=broker.equity_series(),
        trades=broker.trades,
        data=data,
        in_position=pd.Series(in_position_list, index=data.index, name="in_position"),
    )
