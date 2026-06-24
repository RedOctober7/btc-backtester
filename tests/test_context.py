"""
Tests for the lookahead firewall.

The critical invariant: on bar t, the strategy must see exactly t+1 rows of price data.
A strategy that tries to access data beyond the current bar gets an IndexError — not a
subtle wrong-number, a hard crash — because the data literally isn't there.
"""
import pandas as pd
import pytest
from conftest import make_ohlcv
from engine.context import Context, PositionInfo, Order
from strategies.base import Strategy


def _no_position() -> PositionInfo:
    # Helper: build a PositionInfo representing 'no open position'
    return PositionInfo(has_position=False, entry_price=0.0, size=0.0)


# ---------------------------------------------------------------------------
# Structural guarantee: context is sliced to bar t
# ---------------------------------------------------------------------------

def test_context_exposes_only_current_and_past_bars():
    """
    On each bar t, the context contains exactly t+1 rows.
    This is the structural guarantee that makes lookahead impossible by construction.
    """
    data = make_ohlcv([(100 + i, 110 + i, 90 + i, 105 + i) for i in range(10)])

    visible_lengths = []

    class LengthRecorder(Strategy):
        def init(self, d): pass
        def on_bar(self, ctx):
            # Access the internal slice — confirms the engine truncates correctly
            visible_lengths.append(len(ctx._data))

    # Simulate what the engine does: build a Context for each bar t
    for t in range(len(data)):
        ctx = Context(data, t, _no_position())
        LengthRecorder().on_bar(ctx)

    expected = list(range(1, len(data) + 1))
    assert visible_lengths == expected, (
        f"Context slicing broken: on bar t strategy should see t+1 rows. Got {visible_lengths}"
    )


def test_context_future_bar_access_raises():
    """Trying to read beyond the current bar raises IndexError, not a silent wrong value."""
    data = make_ohlcv([(100.0, 110.0, 90.0, 105.0)] * 5)
    ctx = Context(data, 2, _no_position())  # bar 2 = 3 rows visible

    with pytest.raises(IndexError):
        _ = ctx._data.iloc[3]  # bar 3 is the future — must not be accessible


# ---------------------------------------------------------------------------
# Lookahead test that BITES
# ---------------------------------------------------------------------------

def test_no_lookahead_signal_fires_only_on_current_close():
    """
    This test is designed so that a cheating engine would produce a different result.

    Setup:
    - Bars 0..3: close = 100 (boring)
    - Bar 4:     close = 999 (spike)

    Strategy: "buy if context.close > 500"

    Honest engine: signal fires on bar 4 (the last bar). There is no bar 5 to fill
    the order, so the engine discards it. Result: no completed buy fills.

    Cheating engine (bar 3 can see bar 4's close): signal fires on bar 3, fills
    on bar 4's open (999), force-closed at bar 4's close (999). Result: 1 trade.

    We assert: 0 trades. This assertion passes only for the honest engine.
    To verify this test has teeth: temporarily change Context to expose all data
    (not sliced), re-run, and confirm this test goes red.
    """
    from engine.engine import run_backtest

    data = make_ohlcv([
        (100.0, 110.0, 90.0, 100.0),
        (100.0, 110.0, 90.0, 100.0),
        (100.0, 110.0, 90.0, 100.0),
        (100.0, 110.0, 90.0, 100.0),
        (999.0, 1010.0, 990.0, 999.0),  # bar 4: the spike
    ])

    class BuyOnHighClose(Strategy):
        def init(self, d): pass
        def on_bar(self, ctx):
            if ctx.close > 500:
                ctx.buy()

    result = run_backtest(BuyOnHighClose(), data, initial_capital=1000.0)

    assert len(result.trades) == 0, (
        f"LOOKAHEAD DETECTED: expected 0 trades (signal fires on final bar, no fill bar exists), "
        f"got {len(result.trades)}. "
        f"If the engine is feeding future data to the strategy, the signal fires on bar 3 "
        f"instead and gets filled on bar 4."
    )


# ---------------------------------------------------------------------------
# Context properties
# ---------------------------------------------------------------------------

def test_context_close_returns_current_bar():
    data = make_ohlcv([(100.0, 110.0, 90.0, 105.0), (200.0, 210.0, 190.0, 205.0)])
    ctx = Context(data, 1, _no_position())
    assert ctx.close == 205.0
    assert ctx.open == 200.0
    assert ctx.high == 210.0
    assert ctx.low == 190.0
    assert ctx.current_idx == 1


def test_context_position_info_is_passed_through():
    data = make_ohlcv([(100.0, 110.0, 90.0, 105.0)])
    pos = PositionInfo(has_position=True, entry_price=95.0, size=0.5)
    ctx = Context(data, 0, pos)
    assert ctx.position.has_position is True
    assert ctx.position.entry_price == 95.0


# ---------------------------------------------------------------------------
# Order queueing
# ---------------------------------------------------------------------------

def test_buy_queues_order():
    data = make_ohlcv([(100.0, 110.0, 90.0, 105.0)])
    ctx = Context(data, 0, _no_position())
    ctx.buy()
    orders = ctx.pop_orders()
    assert len(orders) == 1
    assert orders[0].action == "buy"


def test_close_position_queues_order_with_reason():
    data = make_ohlcv([(100.0, 110.0, 90.0, 105.0)])
    ctx = Context(data, 0, _no_position())
    ctx.close_position(reason="stop")
    orders = ctx.pop_orders()
    assert len(orders) == 1
    assert orders[0].action == "close"
    assert orders[0].reason == "stop"


def test_pop_orders_clears_queue():
    data = make_ohlcv([(100.0, 110.0, 90.0, 105.0)])
    ctx = Context(data, 0, _no_position())
    ctx.buy()
    ctx.pop_orders()           # first pop — should clear
    assert ctx.pop_orders() == []  # second pop — must be empty
