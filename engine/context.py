"""
The lookahead firewall.

When the engine calls the strategy on bar t, it constructs a Context from
data.iloc[:t+1]. The strategy can only see bars 0..t. Bar t+1 and beyond are
not in the object — accessing them raises IndexError, not a silent wrong value.
"""
from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)  # frozen=True: strategies cannot mutate position info
class PositionInfo:
    """Read-only snapshot of the broker's position, passed through Context."""
    has_position: bool
    entry_price: float   # 0.0 when no position
    size: float          # BTC held; 0.0 when no position


@dataclass
class Order:
    """A pending order queued by the strategy; filled by the broker on the next bar's open."""
    action: str      # "buy" or "close"
    reason: str = "signal"  # "crossover", "stop", "end_of_data" — recorded in trade blotter


class Context:
    """
    Windowed view of market data, handed to the strategy on each bar t.

    Internal: self._data = data.iloc[:t+1]  — structurally excludes future bars.
    """

    def __init__(self, data: pd.DataFrame, current_idx: int, position: PositionInfo):
        # Slice to bars 0..current_idx inclusive; future rows are not present
        self._data = data.iloc[: current_idx + 1]
        self._orders: list[Order] = []
        self.current_idx = current_idx
        self.position = position

    @property
    def close(self) -> float:
        """Current bar's close price."""
        return float(self._data["close"].iloc[-1])

    @property
    def open(self) -> float:
        """Current bar's open price."""
        return float(self._data["open"].iloc[-1])

    @property
    def high(self) -> float:
        """Current bar's high price."""
        return float(self._data["high"].iloc[-1])

    @property
    def low(self) -> float:
        """Current bar's low price."""
        return float(self._data["low"].iloc[-1])

    @property
    def timestamp(self) -> pd.Timestamp:
        """Current bar's open_time."""
        return self._data.index[-1]

    def buy(self) -> None:
        """Queue a buy order; executed at the NEXT bar's open — never the current close."""
        self._orders.append(Order(action="buy"))

    def close_position(self, reason: str = "signal") -> None:
        """Queue a close order; executed at the NEXT bar's open."""
        self._orders.append(Order(action="close", reason=reason))

    def pop_orders(self) -> list[Order]:
        """Return and clear all queued orders. Called by the engine after on_bar()."""
        orders = list(self._orders)
        self._orders.clear()
        return orders
