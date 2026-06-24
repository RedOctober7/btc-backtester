"""
Broker: order execution, fees, slippage, position tracking, and trade blotter.

Fills happen at the price the ENGINE passes in. The engine is responsible for
passing the correct bar's open price (the next bar after the signal). The broker
never selects a price itself — separating concerns makes timing bugs visible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class Trade:
    """One completed round-trip (entry + exit)."""
    entry_time: pd.Timestamp
    entry_price: float
    size: float          # BTC amount purchased
    fee_entry: float     # USDT paid at entry
    entry_bar_idx: int   # bar index of the fill (for bars_held calculation)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    fee_exit: float = 0.0
    exit_bar_idx: Optional[int] = None
    exit_reason: str = ""  # "crossover", "stop", "end_of_data"

    @property
    def fees_paid(self) -> float:
        """Total fees for this round-trip in USDT."""
        return self.fee_entry + self.fee_exit

    @property
    def pnl(self) -> float:
        """
        Realized PnL in USDT.
        = net exit proceeds  −  (cost basis incl. entry fee)
        """
        if self.exit_price is None:
            return 0.0
        gross_proceeds = self.size * self.exit_price
        net_proceeds = gross_proceeds - self.fee_exit
        cost_basis = self.size * self.entry_price + self.fee_entry
        return net_proceeds - cost_basis

    @property
    def return_pct(self) -> float:
        """PnL as a percentage of invested capital (cost basis)."""
        if self.exit_price is None or self.size == 0:
            return 0.0
        cost_basis = self.size * self.entry_price + self.fee_entry
        return (self.pnl / cost_basis) * 100.0

    @property
    def bars_held(self) -> Optional[int]:
        """Number of bars between entry and exit fills."""
        if self.exit_bar_idx is None:
            return None
        return self.exit_bar_idx - self.entry_bar_idx


class Broker:
    """
    Stateful broker: tracks cash, BTC position, and produces a trade blotter.

    Position size formula (entry):
        We want:  btc * price + btc * price * fee_rate = target_cash
        Solving:  btc = target_cash / (price * (1 + fee_rate))
    This ensures cash never goes negative on a 100% allocation — the entry fee
    is paid from the same cash pool, so we buy slightly less than equity/price.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,        # 0.1% taker fee per side (Binance spot default)
        slippage_bps: float = 0.0,      # basis points; fills move against you by this amount
        position_fraction: float = 1.0,  # fraction of equity to deploy per entry
    ):
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.position_fraction = position_fraction

        self.cash: float = initial_capital
        self.btc_held: float = 0.0
        self.entry_price: Optional[float] = None
        self._entry_bar_idx: Optional[int] = None

        self.trades: list[Trade] = []
        self._open_trade: Optional[Trade] = None

        # Per-bar equity snapshots: list of (timestamp, equity_value)
        self._equity_history: list[tuple[pd.Timestamp, float]] = []

    @property
    def has_position(self) -> bool:
        return self.btc_held > 0.0

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Slip the fill price against us by slippage_bps.
        Buys fill higher; sells fill lower — both hurt the strategy.
        """
        slip = self.slippage_bps / 10_000.0
        return price * (1 + slip) if is_buy else price * (1 - slip)

    def fill_buy(self, fill_price: float, timestamp: pd.Timestamp, bar_idx: int = 0) -> None:
        """
        Execute a buy at fill_price.
        Size = position_fraction * current_equity, accounting for entry fee so
        that cost + fee exactly equals the target cash allocation.
        """
        if self.has_position:
            return  # one open position at a time; silently ignore

        fill_price = self._apply_slippage(fill_price, is_buy=True)
        target_cash = self.cash * self.position_fraction

        # Derive BTC size so that (btc * price) + (btc * price * fee_rate) = target_cash
        btc_bought = target_cash / (fill_price * (1 + self.fee_rate))
        entry_fee = btc_bought * fill_price * self.fee_rate
        total_cost = btc_bought * fill_price + entry_fee  # ≈ target_cash

        self.cash -= total_cost
        self.btc_held = btc_bought
        self.entry_price = fill_price
        self._entry_bar_idx = bar_idx

        self._open_trade = Trade(
            entry_time=timestamp,
            entry_price=fill_price,
            size=btc_bought,
            fee_entry=entry_fee,
            entry_bar_idx=bar_idx,
        )

    def fill_close(
        self, fill_price: float, timestamp: pd.Timestamp, reason: str = "signal", bar_idx: int = 0
    ) -> None:
        """Execute a sell (close) at fill_price. Exit fee is deducted from gross proceeds."""
        if not self.has_position:
            return

        fill_price = self._apply_slippage(fill_price, is_buy=False)
        gross_proceeds = self.btc_held * fill_price
        exit_fee = gross_proceeds * self.fee_rate
        net_proceeds = gross_proceeds - exit_fee

        self.cash += net_proceeds

        if self._open_trade is not None:
            self._open_trade.exit_time = timestamp
            self._open_trade.exit_price = fill_price
            self._open_trade.fee_exit = exit_fee
            self._open_trade.exit_bar_idx = bar_idx
            self._open_trade.exit_reason = reason
            self.trades.append(self._open_trade)
            self._open_trade = None

        self.btc_held = 0.0
        self.entry_price = None
        self._entry_bar_idx = None

    def mark_to_market(self, timestamp: pd.Timestamp, close_price: float) -> None:
        """Record equity = cash + mark-to-market position value at this bar's close."""
        equity = self.cash + self.btc_held * close_price
        self._equity_history.append((timestamp, equity))

    def equity_series(self) -> pd.Series:
        """Return the per-bar equity curve as a UTC-indexed Series."""
        if not self._equity_history:
            return pd.Series(dtype=float, name="equity")
        times, values = zip(*self._equity_history)
        return pd.Series(
            list(values),
            index=pd.DatetimeIndex(list(times), tz="UTC"),
            name="equity",
        )
