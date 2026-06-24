from abc import ABC, abstractmethod
import pandas as pd


class Strategy(ABC):
    """
    Abstract base for all strategies. Two-method interface keeps the surface tiny.

    Asymmetry is intentional and critical:
    - init() sees the full series ONCE; use it ONLY for backward-looking indicator
      precomputation (rolling means, etc.). Never compute anything here that uses
      future values (centered windows, z-scores, min-max over the full history).
    - on_bar() is called once per bar and receives a Context windowed to the
      current bar only. It cannot see the future because the Context doesn't have it.
    """

    @abstractmethod
    def init(self, data: pd.DataFrame) -> None:
        """Called once before the backtest run. Receives the full OHLCV DataFrame."""
        ...

    @abstractmethod
    def on_bar(self, context) -> None:
        """Called once per bar. Use context.buy() / context.close_position() to act."""
        ...
