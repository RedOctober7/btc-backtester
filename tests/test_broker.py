"""
Broker correctness tests.

Key things being verified:
1. Fills happen at the price we hand the broker — it's the engine's job to pass
   the correct (next) bar's open, so the test just confirms the price lands on the trade.
2. Fee arithmetic is sequential, not simplified: entry fee shrinks the BTC bought,
   so the exit fee lands on a slightly smaller notional.
3. Cash never goes negative on a 100% position size allocation.
"""
import pytest
import pandas as pd
from engine.broker import Broker, Trade


def _make_ts(offset_hours: int = 0) -> pd.Timestamp:
    return pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(hours=offset_hours)


# ---------------------------------------------------------------------------
# Next-bar fill price
# ---------------------------------------------------------------------------

def test_fill_price_is_whatever_engine_passes():
    """
    The broker records the exact price handed to fill_buy().
    It is the ENGINE's responsibility to pass the correct bar's open price.
    This test verifies the broker doesn't silently alter the fill price
    (slippage aside — slippage is tested separately).
    """
    broker = Broker(initial_capital=1000.0, fee_rate=0.0, slippage_bps=0.0)
    fill_price = 200.0  # represents bar t+1's open
    broker.fill_buy(fill_price, _make_ts(4))  # t+1's timestamp
    assert broker.trades == []  # no closed trade yet
    assert broker.entry_price == fill_price


def test_next_bar_fill_simulation():
    """
    Full simulation: signal on bar 0 (via engine) should fill at bar 1's open.
    We test this end-to-end using run_backtest so the engine's fill timing is exercised.
    """
    from conftest import make_ohlcv
    from engine.engine import run_backtest
    from strategies.base import Strategy

    # bar 0: close=150 (signal bar); bar 1: open=200 (fill price); bar 2: close=250
    data = make_ohlcv([
        (100.0, 160.0, 90.0, 150.0),   # bar 0
        (200.0, 220.0, 180.0, 210.0),  # bar 1 — fill here
        (200.0, 260.0, 190.0, 250.0),  # bar 2
    ])

    class BuyOnBar0(Strategy):
        def init(self, d): pass
        def on_bar(self, ctx):
            if ctx.current_idx == 0:
                ctx.buy()

    result = run_backtest(BuyOnBar0(), data, initial_capital=1000.0, fee_rate=0.0, slippage_bps=0.0)
    assert len(result.trades) == 1
    trade = result.trades[0]
    # Fill must be at bar 1's open (200), NOT bar 0's close (150)
    assert trade.entry_price == 200.0, (
        f"Wrong fill price: expected bar 1 open (200.0), got {trade.entry_price}. "
        f"This means the engine filled at bar 0's close instead of bar 1's open — "
        f"a next-bar-fill violation."
    )


# ---------------------------------------------------------------------------
# Fee arithmetic
# ---------------------------------------------------------------------------

def test_fee_exact_round_trip():
    """
    Exact round-trip fee when buying and selling at the same flat price.

    The naive formula  loss = 2 * fee_rate * capital  is WRONG.
    Entry fee shrinks the BTC purchased, so exit fee lands on a slightly smaller notional:
        btc_bought = capital / (price * (1 + fee_rate))
        entry_fee  = btc_bought * price * fee_rate = capital * fee_rate / (1 + fee_rate)
        exit_gross = btc_bought * price = capital / (1 + fee_rate)
        exit_fee   = exit_gross * fee_rate = capital * fee_rate / (1 + fee_rate)
        net_after  = exit_gross - exit_fee = capital * (1 - fee_rate) / (1 + fee_rate)
        total_loss = capital - net_after = capital * 2 * fee_rate / (1 + fee_rate)
    """
    initial_capital = 1000.0
    fee_rate = 0.001
    price = 100.0

    broker = Broker(initial_capital=initial_capital, fee_rate=fee_rate, slippage_bps=0.0)
    broker.fill_buy(price, _make_ts(0))
    broker.fill_close(price, _make_ts(4), reason="signal")
    broker.mark_to_market(_make_ts(4), price)

    expected_final_equity = initial_capital * (1 - fee_rate) / (1 + fee_rate)
    actual_final_equity = broker.equity_series().iloc[-1]

    assert abs(actual_final_equity - expected_final_equity) < 1e-8, (
        f"Fee arithmetic wrong. "
        f"Expected {expected_final_equity:.10f} (sequential fees), "
        f"got {actual_final_equity:.10f}. "
        f"Simple 2*fee_rate*capital would give {initial_capital - 2*fee_rate*initial_capital:.10f}."
    )


def test_fee_zero_means_flat_price_is_breakeven():
    broker = Broker(initial_capital=1000.0, fee_rate=0.0, slippage_bps=0.0)
    broker.fill_buy(100.0, _make_ts(0))
    broker.fill_close(100.0, _make_ts(4), reason="signal")
    broker.mark_to_market(_make_ts(4), 100.0)
    assert abs(broker.equity_series().iloc[-1] - 1000.0) < 1e-8


def test_cash_never_negative_on_full_allocation():
    """A 100% position allocation must not cause negative cash (fee is paid from equity)."""
    broker = Broker(initial_capital=1000.0, fee_rate=0.001, slippage_bps=0.0, position_fraction=1.0)
    broker.fill_buy(100.0, _make_ts(0))
    assert broker.cash >= -1e-10, f"Cash went negative: {broker.cash}"
    assert abs(broker.cash) < 1e-6  # should be ~0, not exactly 0 due to float arithmetic


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------

def test_slippage_moves_buy_price_up():
    """A buy with 10 bps slippage fills at price * (1 + 0.001)."""
    broker = Broker(initial_capital=1000.0, fee_rate=0.0, slippage_bps=10.0)
    broker.fill_buy(100.0, _make_ts(0))
    expected_price = 100.0 * (1 + 10 / 10_000)
    assert abs(broker.entry_price - expected_price) < 1e-8


def test_slippage_moves_sell_price_down():
    """A sell with 10 bps slippage fills at price * (1 - 0.001)."""
    broker = Broker(initial_capital=1000.0, fee_rate=0.0, slippage_bps=10.0)
    broker.fill_buy(100.0, _make_ts(0))
    broker.fill_close(100.0, _make_ts(4), reason="signal")
    trade = broker.trades[0]
    expected_exit = 100.0 * (1 - 10 / 10_000)
    assert abs(trade.exit_price - expected_exit) < 1e-8


# ---------------------------------------------------------------------------
# Trade blotter
# ---------------------------------------------------------------------------

def test_trade_blotter_fields():
    """Closed trade records all required fields."""
    broker = Broker(initial_capital=1000.0, fee_rate=0.001, slippage_bps=0.0)
    t0 = _make_ts(0)
    t1 = _make_ts(4)
    broker.fill_buy(100.0, t0)
    broker.fill_close(110.0, t1, reason="crossover")

    assert len(broker.trades) == 1
    trade = broker.trades[0]
    assert trade.entry_time == t0
    assert trade.exit_time == t1
    assert trade.entry_price == 100.0
    assert trade.exit_price == 110.0
    assert trade.exit_reason == "crossover"
    assert trade.fee_entry > 0
    assert trade.fee_exit > 0
    assert trade.fees_paid == trade.fee_entry + trade.fee_exit
    assert trade.pnl > 0  # profitable at 110 vs 100


def test_one_position_at_a_time():
    """Second buy while position is open is silently ignored."""
    broker = Broker(initial_capital=1000.0, fee_rate=0.0, slippage_bps=0.0)
    broker.fill_buy(100.0, _make_ts(0))
    btc_after_first = broker.btc_held
    broker.fill_buy(110.0, _make_ts(4))  # should be ignored
    assert broker.btc_held == btc_after_first
