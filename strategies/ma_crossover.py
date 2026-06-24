"""
SMA crossover strategy with a position drawdown stop.

Entry:  fast SMA crosses above slow SMA (bullish crossover)
Exit:   fast SMA crosses below slow SMA  OR  position drawdown from peak >= stop_pct

Stop semantics (conservative, NOT a guaranteed stop):
    The stop is evaluated on each bar's CLOSE and, if triggered, exits at the
    NEXT bar's OPEN. On a large or gapping 4h candle, the realized loss can exceed
    stop_pct because we wait for the open. An intrabar stop (exits exactly at the
    stop level using bar's low) would be more optimistic but less realistic — that
    is left as a v2 option. The conservative version is what ships here.
"""
from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class MACrossover(Strategy):
    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        stop_pct: float = 0.08,  # 8% drawdown from position peak triggers exit
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_pct = stop_pct

        # Precomputed in init(); read by index in on_bar()
        self.sma_fast: pd.Series | None = None
        self.sma_slow: pd.Series | None = None

        self._peak_close: float = 0.0  # highest close seen since entering the position

    def init(self, data: pd.DataFrame) -> None:
        """
        Compute both SMAs on the full series ONCE.
        pandas .rolling() is backward-looking by default — each value depends only
        on rows at or before its own index. No forward leakage is possible here.
        The NaN values during warmup are intentional: on_bar() guards for them explicitly.
        """
        self.sma_fast = data["close"].rolling(self.fast_period).mean()
        self.sma_slow = data["close"].rolling(self.slow_period).mean()
        # Reset mutable state so the same instance can be passed to run_backtest() twice
        # without carrying forward the peak from the previous run.
        self._peak_close = 0.0

    def on_bar(self, context) -> None:
        i = context.current_idx
        fast = self.sma_fast.iloc[i]
        slow = self.sma_slow.iloc[i]

        # Guard NaN explicitly — NaN > NaN is False in Python/pandas and fails silently,
        # which could create ghost crossover signals at the very first valid bar.
        if pd.isna(fast) or pd.isna(slow):
            return

        # Need the previous bar's values to detect a crossover (sign change in fast-slow)
        if i == 0:
            return
        prev_fast = self.sma_fast.iloc[i - 1]
        prev_slow = self.sma_slow.iloc[i - 1]
        if pd.isna(prev_fast) or pd.isna(prev_slow):
            return

        if context.position.has_position:
            # Track the highest close since entry to measure drawdown
            # Using closes (not intrabar highs) for consistency with all other price points
            self._peak_close = max(self._peak_close, context.close)
            drawdown = (self._peak_close - context.close) / self._peak_close

            stop_hit = drawdown >= self.stop_pct
            crossed_below = fast < slow and prev_fast >= prev_slow

            if stop_hit:
                context.close_position(reason="stop")
                self._peak_close = 0.0
            elif crossed_below:
                context.close_position(reason="crossover")
                self._peak_close = 0.0
        else:
            crossed_above = fast > slow and prev_fast <= prev_slow
            if crossed_above:
                context.buy()
                # Initialize peak at the signal bar's close, not the fill bar's open,
                # because peak tracking begins from when we commit to the trade
                self._peak_close = context.close
