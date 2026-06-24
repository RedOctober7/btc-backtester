"""
Engine loop correctness tests:
- Warmup: no trades while indicators are NaN
- Force-close: open position on final bar gets closed at that bar's close, tagged 'end_of_data'
- Equity marked to market every bar
- in_position series tracks position state at bar-start
"""
import pandas as pd
from conftest import make_ohlcv
from engine.engine import run_backtest
from strategies.base import Strategy


# ---------------------------------------------------------------------------
# Warmup test
# ---------------------------------------------------------------------------

def test_no_trades_during_warmup():
    """
    Strategy must not trade while its longest indicator is NaN.
    A 5-period SMA is NaN for bars 0-3; the first valid bar is index 4.
    The earliest possible entry fill is at bar 5's open.
    """
    n = 10
    # Rising prices so "close > SMA" is true on every valid bar
    prices = [float(100 + i * 10) for i in range(n)]
    data = make_ohlcv([(p, p + 5, p - 5, p) for p in prices])

    class SMAStrategy(Strategy):
        """Buys whenever close is above the 5-period SMA. Warmup = bars 0-3."""
        def init(self, d):
            # rolling(5) is NaN for first 4 rows (indices 0-3); first valid at index 4
            self.sma = d["close"].rolling(5).mean()

        def on_bar(self, ctx):
            val = self.sma.iloc[ctx.current_idx]
            if pd.isna(val):
                return  # warmup — guard explicitly; NaN > NaN is always False in Python
            if ctx.close > val and not ctx.position.has_position:
                ctx.buy()

    result = run_backtest(SMAStrategy(), data, initial_capital=1000.0)
    assert result.trades, "Expected at least one trade after warmup"

    # The first fill cannot happen before bar 5's open_time
    # (signal fires on bar 4, fills on bar 5)
    bar5_open_time = data.index[5]
    first_entry_time = result.trades[0].entry_time
    assert first_entry_time >= bar5_open_time, (
        f"Trade entered during warmup! entry_time={first_entry_time}, "
        f"bar5_open_time={bar5_open_time}. "
        f"The strategy must guard against NaN indicator values."
    )


# ---------------------------------------------------------------------------
# Force-close test
# ---------------------------------------------------------------------------

def test_force_close_at_final_bar():
    """
    A position still open on the final bar must be force-closed at that bar's close.
    The exit_reason must be 'end_of_data' and the final equity must reflect realized PnL.
    """
    data = make_ohlcv([
        (100.0, 110.0, 90.0, 105.0),   # bar 0: signal bar (buy)
        (110.0, 120.0, 100.0, 115.0),  # bar 1: fill at open=110
        (120.0, 130.0, 110.0, 125.0),  # bar 2: final bar — force-close at close=125
    ])

    class AlwaysBuyNeverSell(Strategy):
        def init(self, d): pass
        def on_bar(self, ctx):
            if not ctx.position.has_position:
                ctx.buy()

    result = run_backtest(AlwaysBuyNeverSell(), data, initial_capital=1000.0, fee_rate=0.0)

    assert len(result.trades) == 1
    trade = result.trades[0]

    # Force-close tag
    assert trade.exit_reason == "end_of_data", (
        f"Expected 'end_of_data', got '{trade.exit_reason}'"
    )
    # Exit at final bar's close (125)
    assert trade.exit_price == 125.0, f"Expected 125.0, got {trade.exit_price}"
    # Entry at bar 1's open (110)
    assert trade.entry_price == 110.0

    # Final equity reflects the realized gain (no fees)
    # Entry: bought 1000/110 BTC at 110; exit: sold at 125
    expected_btc = 1000.0 / 110.0
    expected_equity = expected_btc * 125.0
    assert abs(result.equity.iloc[-1] - expected_equity) < 0.01


def test_force_close_appears_in_trade_count():
    """Force-close must appear as a completed trade — not an open position."""
    data = make_ohlcv([
        (100.0, 110.0, 90.0, 100.0),
        (100.0, 110.0, 90.0, 100.0),
    ])

    class ImmediateBuy(Strategy):
        def init(self, d): pass
        def on_bar(self, ctx):
            if not ctx.position.has_position:
                ctx.buy()

    result = run_backtest(ImmediateBuy(), data, initial_capital=1000.0, fee_rate=0.0)
    # 2-bar series: buy signal on bar 0, fills on bar 1 (also the final bar),
    # then force-close fires at bar 1's close
    assert len(result.trades) == 1
    assert result.trades[0].exit_reason == "end_of_data"


# ---------------------------------------------------------------------------
# Equity is marked to market every bar
# ---------------------------------------------------------------------------

def test_equity_marked_every_bar():
    """Equity series must have one entry per bar, even when no trade occurs."""
    n = 5
    data = make_ohlcv([(100.0, 110.0, 90.0, 100.0)] * n)

    class DoNothing(Strategy):
        def init(self, d): pass
        def on_bar(self, ctx): pass

    result = run_backtest(DoNothing(), data, initial_capital=1000.0)
    assert len(result.equity) == n


# ---------------------------------------------------------------------------
# In-position tracking for exposure metric
# ---------------------------------------------------------------------------

def test_in_position_series_matches_trades():
    """in_position Series must be True on bars where a position is held."""
    data = make_ohlcv([
        (100.0, 110.0, 90.0, 100.0),   # bar 0: buy signal
        (110.0, 120.0, 100.0, 115.0),  # bar 1: fill at open; held
        (115.0, 125.0, 105.0, 120.0),  # bar 2: held
        (120.0, 130.0, 110.0, 125.0),  # bar 3: final bar, force-close
    ])

    class HoldForever(Strategy):
        def init(self, d): pass
        def on_bar(self, ctx):
            if not ctx.position.has_position:
                ctx.buy()

    result = run_backtest(HoldForever(), data, initial_capital=1000.0, fee_rate=0.0)

    # Bar 0: not yet in position (fill happens at bar 1's open)
    # Bars 1, 2, 3: in position (filled at bar 1's open, held through force-close on bar 3)
    assert result.in_position.iloc[0] is False or not result.in_position.iloc[0]
    assert result.in_position.iloc[1]
    assert result.in_position.iloc[2]
