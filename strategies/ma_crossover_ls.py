"""
Long/Short SMA crossover with a symmetric drawdown stop.

The trend-follower's argument for shorting: the plain long/flat MACrossover sits
in cash during downtrends, earning nothing while price falls. A long/short
version stays engaged — long above the slow SMA, short below it — so it can
profit from the down-legs that the long-only version merely sidesteps. Whether
that helps or just doubles the whipsaw exposure in chop is an empirical question;
run it through walk_forward.py before believing it.

Signals (desired side follows the most recent crossover):
    fast crosses above slow  -> desired = long
    fast crosses below slow  -> desired = short

Each bar:
    - if flat and a desired side is set        -> enter that side
    - if holding the wrong side (a flip)        -> close (reason "flip"); the
      opposite entry lands the bar after, since next-bar-open fills make an
      instantaneous reversal impossible anyway
    - drawdown stop, symmetric:
        long : (peak_close  - close) / peak_close  >= stop_pct
        short: (close - trough_close) / trough_close >= stop_pct

Stop semantics match the long-only version: evaluated on close, exits at the
NEXT bar's open, so a gapping bar can realize more than stop_pct. Conservative
by design.
"""
from __future__ import annotations

import pandas as pd

from strategies.base import Strategy


class MACrossoverLS(Strategy):
    def __init__(
        self,
        fast_period: int = 50,
        slow_period: int = 200,
        stop_pct: float = 0.08,
    ):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.stop_pct = stop_pct

        self.sma_fast: pd.Series | None = None
        self.sma_slow: pd.Series | None = None

        self._desired: str | None = None      # "long" | "short" | None
        self._peak_close: float = 0.0          # for long stop
        self._trough_close: float = 0.0        # for short stop

    def init(self, data: pd.DataFrame) -> None:
        self.sma_fast = data["close"].rolling(self.fast_period).mean()
        self.sma_slow = data["close"].rolling(self.slow_period).mean()
        self._desired = None
        self._peak_close = 0.0
        self._trough_close = 0.0

    def on_bar(self, context) -> None:
        i = context.current_idx
        if i == 0:
            return
        fast = self.sma_fast.iloc[i]
        slow = self.sma_slow.iloc[i]
        prev_fast = self.sma_fast.iloc[i - 1]
        prev_slow = self.sma_slow.iloc[i - 1]
        if pd.isna(fast) or pd.isna(slow) or pd.isna(prev_fast) or pd.isna(prev_slow):
            return

        # Update desired side on a fresh crossover event
        if fast > slow and prev_fast <= prev_slow:
            self._desired = "long"
        elif fast < slow and prev_fast >= prev_slow:
            self._desired = "short"

        pos = context.position

        if pos.has_position:
            close = context.close

            if pos.is_long:
                self._peak_close = max(self._peak_close, close)
                drawdown = (self._peak_close - close) / self._peak_close
                if drawdown >= self.stop_pct:
                    context.close_position(reason="stop")
                    return
                if self._desired == "short":        # flip
                    context.close_position(reason="flip")
                    return

            elif pos.is_short:
                self._trough_close = min(self._trough_close, close) if self._trough_close else close
                rally = (close - self._trough_close) / self._trough_close
                if rally >= self.stop_pct:
                    context.close_position(reason="stop")
                    return
                if self._desired == "long":          # flip
                    context.close_position(reason="flip")
                    return
        else:
            # Flat: take the desired side if we have one
            if self._desired == "long":
                context.buy()
                self._peak_close = context.close
                self._trough_close = 0.0
            elif self._desired == "short":
                context.sell_short()
                self._trough_close = context.close
                self._peak_close = 0.0
