"""
Broker: order execution, fees, slippage, funding, position tracking, and blotter.

Fills happen at the price the ENGINE passes in. The engine owns timing (which
bar, which price); the broker owns accounting. Separating them keeps timing bugs
visible.

v2 adds short selling and leverage. Guiding invariant: at leverage=1 with
funding disabled (both defaults), every number is bit-for-bit identical to the
original spot-only broker — existing backtests and README figures do not move
unless you deliberately opt into leverage, shorting, or funding.

The leverage model is a linear perpetual-futures approximation. What it models,
and the simplifications it makes, are documented at each method rather than
hidden. The design rule throughout: when OHLC data leaves a question unanswerable
(intrabar path, exact fill in a cascade), resolve it PESSIMISTICALLY, so the
backtest never flatters a leveraged strategy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

Direction = str  # "long" | "short"


@dataclass
class Trade:
    """One completed round-trip (entry + exit)."""
    entry_time: pd.Timestamp
    entry_price: float
    size: float          # BTC amount (always positive; direction is separate)
    fee_entry: float     # USDT paid at entry
    entry_bar_idx: int
    direction: Direction = "long"
    margin: float = 0.0  # USDT collateral posted (net of entry fee)
    leverage: float = 1.0
    funding_paid: float = 0.0  # net USDT paid in funding over the hold (negative = received)
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    fee_exit: float = 0.0
    exit_bar_idx: Optional[int] = None
    exit_reason: str = ""  # "crossover" | "stop" | "flip" | "end_of_data" | "liquidation"

    @property
    def fees_paid(self) -> float:
        return self.fee_entry + self.fee_exit

    @property
    def pnl(self) -> float:
        """
        Realized PnL in USDT: price move − fees − net funding. Direction-aware.
        At direction='long', funding=0, this equals the original v1 definition
        (net exit proceeds − cost basis incl. entry fee).
        """
        if self.exit_price is None:
            return 0.0
        if self.direction == "long":
            price_pnl = self.size * (self.exit_price - self.entry_price)
        else:
            price_pnl = self.size * (self.entry_price - self.exit_price)
        return price_pnl - self.fee_entry - self.fee_exit - self.funding_paid

    @property
    def return_pct(self) -> float:
        """PnL as a percentage of margin posted (capital actually at risk)."""
        if self.exit_price is None or self.margin == 0:
            return 0.0
        return (self.pnl / self.margin) * 100.0

    @property
    def bars_held(self) -> Optional[int]:
        if self.exit_bar_idx is None:
            return None
        return self.exit_bar_idx - self.entry_bar_idx


class Broker:
    """
    Stateful broker: cash, one optional leveraged long/short position, blotter.

    Sizing at entry (unified across direction and leverage):
        margin_required = cash * position_fraction
        notional        = margin_required * leverage
        size            = notional / (price * (1 + fee_rate))    [fee-inclusive]
        entry_fee       = size * price * fee_rate
        margin_locked   = margin_required - entry_fee            [collateral backing the position]
        cash           -= margin_required

    At leverage=1 this reduces algebraically to the v1 formulas.

    LIQUIDATION (leverage > 1). Liquidation price is derived from the exchange
    condition "position equity == maintenance margin on CURRENT notional":
        long : margin + size*(P - entry) = mmr * size * P
               =>  P_liq = (size*entry - margin) / (size * (1 - mmr))
        short: margin + size*(entry - P) = mmr * size * P
               =>  P_liq = (size*entry + margin) / (size * (1 + mmr))
    This matches how real linear-perp venues compute isolated liquidation
    (current-mark maintenance), not the cruder entry-notional shortcut.

    Fill on liquidation is GAP-AWARE and pessimistic: if the bar opens beyond
    the liquidation price (a crash gap), the fill is the OPEN, not the trigger —
    because a real account is liquidated into the gap and eats it. Only when the
    bar trades THROUGH the level intrabar (open still on the safe side) does it
    fill at the trigger. An optional liquidation_penalty_bps models the extra
    slippage of a forced fill in a cascade.

    Loss is floored at the posted margin (equity can never go negative). Real
    venues backstop the sub-zero tail with an insurance fund / ADL; that
    mechanism is not modeled — the floor is the documented simplification.

    FUNDING (perpetual futures). If funding_rate_8h != 0, the engine applies
    funding on 8h-aligned bars: longs pay (and shorts receive) rate * notional
    when the rate is positive. Deducted from margin_locked, so sustained funding
    both erodes PnL and pulls the liquidation price closer — as it does live.
    Default 0.0 keeps spot behavior exact; set it (~0.0001 per 8h is a typical
    resting rate, far higher in a squeeze) whenever you run leverage, or the
    backtest will overstate every leveraged hold.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_rate: float = 0.001,
        slippage_bps: float = 0.0,
        position_fraction: float = 1.0,
        leverage: float = 1.0,
        maintenance_margin_rate: float = 0.005,
        funding_rate_8h: float = 0.0,
        liquidation_penalty_bps: float = 0.0,
    ):
        if leverage < 1.0:
            raise ValueError(f"leverage must be >= 1.0, got {leverage}")

        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_bps = slippage_bps
        self.position_fraction = position_fraction
        self.leverage = leverage
        self.maintenance_margin_rate = maintenance_margin_rate
        self.funding_rate_8h = funding_rate_8h
        self.liquidation_penalty_bps = liquidation_penalty_bps

        self.cash: float = initial_capital
        self.size: float = 0.0
        self.direction: Optional[Direction] = None
        self.entry_price: Optional[float] = None
        self.margin_locked: float = 0.0
        self._entry_bar_idx: Optional[int] = None

        self.trades: list[Trade] = []
        self._open_trade: Optional[Trade] = None
        self._equity_history: list[tuple[pd.Timestamp, float]] = []

    # ── backward-compat: old code checked broker.btc_held > 0 ────────────────
    @property
    def btc_held(self) -> float:
        return self.size if self.direction == "long" else 0.0

    @property
    def has_position(self) -> bool:
        return self.direction is not None

    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        slip = self.slippage_bps / 10_000.0
        return price * (1 + slip) if is_buy else price * (1 - slip)

    # ── open ─────────────────────────────────────────────────────────────────
    def _open(self, direction: Direction, price: float, timestamp: pd.Timestamp, bar_idx: int) -> None:
        if self.has_position:
            return
        margin_required = self.cash * self.position_fraction
        if margin_required <= 0:
            return  # broke account cannot open (prevents zero-size ghost positions)

        is_buy = direction == "long"
        fill_price = self._apply_slippage(price, is_buy=is_buy)

        notional = margin_required * self.leverage
        size = notional / (fill_price * (1 + self.fee_rate))
        entry_fee = size * fill_price * self.fee_rate
        margin_locked = margin_required - entry_fee

        self.cash -= margin_required
        self.size = size
        self.direction = direction
        self.entry_price = fill_price
        self.margin_locked = margin_locked
        self._entry_bar_idx = bar_idx

        self._open_trade = Trade(
            entry_time=timestamp, entry_price=fill_price, size=size,
            fee_entry=entry_fee, entry_bar_idx=bar_idx, direction=direction,
            margin=margin_locked, leverage=self.leverage,
        )

    def fill_buy(self, fill_price, timestamp, bar_idx=0) -> None:
        """Open a long. Original method name kept for backward compat."""
        self._open("long", fill_price, timestamp, bar_idx)

    def fill_short(self, fill_price, timestamp, bar_idx=0) -> None:
        """Open a short."""
        self._open("short", fill_price, timestamp, bar_idx)

    # ── close ──────────────────────────────────────────────────────────────
    def fill_close(self, fill_price, timestamp, reason="signal", bar_idx=0) -> None:
        if not self.has_position:
            return
        is_buy = self.direction == "short"   # closing a short = buying back
        fill_price = self._apply_slippage(fill_price, is_buy=is_buy)
        self._settle(fill_price, timestamp, reason, bar_idx)

    def _settle(self, fill_price, timestamp, reason, bar_idx) -> None:
        exit_fee = self.size * fill_price * self.fee_rate
        if self.direction == "long":
            realized = self.size * (fill_price - self.entry_price)
        else:
            realized = self.size * (self.entry_price - fill_price)

        # A closed position returns at most its remaining value, never below 0.
        returned = max(0.0, self.margin_locked + realized - exit_fee)
        self.cash += returned

        if self._open_trade is not None:
            self._open_trade.exit_time = timestamp
            self._open_trade.exit_price = fill_price
            self._open_trade.fee_exit = exit_fee
            self._open_trade.exit_bar_idx = bar_idx
            self._open_trade.exit_reason = reason
            self.trades.append(self._open_trade)
            self._open_trade = None

        self.size = 0.0
        self.direction = None
        self.entry_price = None
        self.margin_locked = 0.0
        self._entry_bar_idx = None

    # ── liquidation ────────────────────────────────────────────────────────
    @property
    def liquidation_price(self) -> Optional[float]:
        """Trigger price for the open position (current-notional maintenance)."""
        if not self.has_position or self.size == 0:
            return None
        mmr = self.maintenance_margin_rate
        q, e, m = self.size, self.entry_price, self.margin_locked
        if self.direction == "long":
            return (q * e - m) / (q * (1 - mmr))
        else:
            return (q * e + m) / (q * (1 + mmr))

    def maybe_liquidate(self, bar_open, bar_high, bar_low, timestamp, bar_idx) -> bool:
        """
        If this bar's range crosses the liquidation trigger, force-close the
        position and return True. Gap-aware, pessimistic fill (see class docs).
        """
        if not self.has_position or self.leverage <= 1.0:
            return False
        trig = self.liquidation_price
        if trig is None:
            return False
        pen = self.liquidation_penalty_bps / 10_000.0

        if self.direction == "long" and bar_low <= trig:
            fill = min(bar_open, trig)      # gapped below open? eat the gap
            fill = fill * (1 - pen)         # forced sell slips down
            self._settle(fill, timestamp, "liquidation", bar_idx)
            return True
        if self.direction == "short" and bar_high >= trig:
            fill = max(bar_open, trig)      # gapped above open? eat the gap
            fill = fill * (1 + pen)         # forced buy-back slips up
            self._settle(fill, timestamp, "liquidation", bar_idx)
            return True
        return False

    # ── funding ──────────────────────────────────────────────────────────────
    def apply_funding(self, mark_price, timestamp) -> None:
        """Charge/credit one funding interval. No-op when flat or rate is 0."""
        if not self.has_position or self.funding_rate_8h == 0.0:
            return
        amount = self.funding_rate_8h * self.size * mark_price  # signed by rate
        if self.direction == "long":
            self.margin_locked -= amount     # long pays when rate > 0
            paid = amount
        else:
            self.margin_locked += amount     # short receives when rate > 0
            paid = -amount
        if self._open_trade is not None:
            self._open_trade.funding_paid += paid

    # ── equity ───────────────────────────────────────────────────────────────
    def unrealized_pnl(self, mark_price) -> float:
        if not self.has_position:
            return 0.0
        if self.direction == "long":
            return self.size * (mark_price - self.entry_price)
        return self.size * (self.entry_price - mark_price)

    def equity(self, mark_price) -> float:
        return self.cash + self.margin_locked + self.unrealized_pnl(mark_price)

    def mark_to_market(self, timestamp, close_price) -> None:
        self._equity_history.append((timestamp, self.equity(close_price)))

    def equity_series(self) -> pd.Series:
        if not self._equity_history:
            return pd.Series(dtype=float, name="equity")
        times, values = zip(*self._equity_history)
        return pd.Series(list(values), index=pd.DatetimeIndex(list(times), tz="UTC"), name="equity")
