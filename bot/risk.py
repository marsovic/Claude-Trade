"""Position sizing, volatility-adapted leverage and position management.

Risk is defined by the stop, not by the leverage: the bot loses at most
``risk_per_trade`` of its equity (fees included) when a stop is hit. Leverage
only decides how much margin the position locks, and it shrinks as volatility
grows so that a normal adverse move never comes close to the liquidation price.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

from bot.binance_client import SymbolInfo
from bot.strategy import LONG


def direction(side: str) -> int:
    return 1 if side == LONG else -1


@dataclass
class TradePlan:
    symbol: str
    side: str
    qty: float
    price: float
    stop_loss: float
    take_profit: float
    leverage: int
    atr: float
    notional: float
    margin: float
    risk_amount: float


@dataclass
class Position:
    symbol: str
    side: str
    qty: float
    entry_price: float
    leverage: int
    stop_loss: float
    take_profit: float
    initial_stop: float
    atr: float
    margin: float
    risk_amount: float
    opened_at: int
    entry_fee: float = 0.0
    best_price: float = 0.0
    breakeven_done: bool = False
    sl_algo_id: int | None = None
    tp_algo_id: int | None = None
    # Revolut X (spot): resting take-profit and the quote amount spent, fees included.
    tp_order_id: str | None = None
    tp_client_id: str | None = None
    cost: float = 0.0
    exit_order_ids: list[str] = field(default_factory=list)  # every order that sold coins
    exit_client_ids: list[str] = field(default_factory=list)  # sells whose reply was lost
    sentiment: float | None = None
    signal: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.best_price:
            self.best_price = self.entry_price

    @property
    def dir(self) -> int:
        return direction(self.side)

    @property
    def r_distance(self) -> float:
        return abs(self.entry_price - self.initial_stop)

    def unrealized(self, price: float) -> float:
        return self.dir * (price - self.entry_price) * self.qty

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Position:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ClosedTrade:
    symbol: str
    side: str
    qty: float
    entry_price: float
    exit_price: float
    leverage: int
    margin: float
    opened_at: int
    closed_at: int
    pnl: float
    fees: float
    r_multiple: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------- sizing
def volatility_leverage(atr_pct: float, s) -> int:
    """Calm market -> up to ``max_leverage``; agitated market -> down to x1."""
    if atr_pct <= 0:
        return 1
    return max(1, min(s.max_leverage, math.floor(s.leverage_vol_target / atr_pct)))


def liquidation_safe_leverage(leverage: int, stop_pct: float, s) -> int:
    """Lower leverage until the stop sits well inside the liquidation distance.

    With isolated margin the liquidation price is roughly
    ``1/leverage - maintenance_margin_rate`` away from entry. We keep the stop
    within half of that distance.
    """
    while leverage > 1 and stop_pct > 0.5 * (1 / leverage - s.maintenance_margin_rate):
        leverage -= 1
    return leverage


def plan_trade(symbol: str, side: str, price: float, atr: float, atr_pct: float,
               equity: float, used_margin: float, info: SymbolInfo, s
               ) -> tuple[TradePlan | None, str]:
    """Build an order plan, or return ``(None, reason)`` when the trade is refused."""
    if price <= 0 or atr <= 0:
        return None, "invalid price/ATR"
    d = direction(side)
    stop_dist = atr * s.stop_atr_mult
    stop_pct = stop_dist / price
    if stop_pct >= 0.5:
        return None, "stop distance too large"
    stop = price - d * stop_dist
    target = price + d * stop_dist * s.take_profit_r

    leverage = volatility_leverage(atr_pct, s)
    leverage = liquidation_safe_leverage(leverage, stop_pct, s)

    risk_budget = equity * s.risk_per_trade
    # Loss per unit if the stop is hit, including entry and exit taker fees.
    loss_per_unit = stop_dist + s.fee_rate * (price + stop) + price * s.slippage_bps / 1e4
    qty = risk_budget / loss_per_unit

    free_margin = min(equity * s.max_margin_per_trade,
                      equity * s.max_total_margin - used_margin)
    if free_margin <= 0:
        return None, "no free margin"
    if qty * price / leverage > free_margin:
        qty = free_margin * leverage / price  # smaller position, risk < budget

    rq = info.round_qty(qty)
    if rq <= 0 or rq < info.min_qty:
        return None, f"quantity below exchange minimum ({info.min_qty})"
    if rq * info.round_price(price) < info.min_notional:
        return None, f"notional below exchange minimum ({info.min_notional} USDT)"
    qty = float(rq)
    notional = qty * price
    return TradePlan(
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        stop_loss=float(info.round_price(stop)),
        take_profit=float(info.round_price(target)),
        leverage=leverage,
        atr=atr,
        notional=notional,
        margin=notional / leverage,
        risk_amount=qty * loss_per_unit,
    ), "ok"


# ----------------------------------------------------------------- management
def _better(side: str, a: float, b: float) -> float:
    """The more protective of two stops."""
    return max(a, b) if side == LONG else min(a, b)


def update_stops(pos: Position, price: float, s) -> bool:
    """Move the stop to break-even, then trail it. Returns True if it moved."""
    d = pos.dir
    if d * (price - pos.best_price) > 0:
        pos.best_price = price
    r = pos.r_distance
    if r <= 0:
        return False
    gain = d * (pos.best_price - pos.entry_price)
    new_stop = pos.stop_loss
    if s.breakeven_r > 0 and not pos.breakeven_done and gain >= s.breakeven_r * r:
        # Slightly beyond entry so that the round-trip fees are covered.
        breakeven = pos.entry_price * (1 + d * 2 * s.fee_rate)
        new_stop = _better(pos.side, new_stop, breakeven)
        pos.breakeven_done = True
    if s.trailing and gain >= s.trail_start_r * r:
        trail = pos.best_price - d * s.trail_atr_mult * pos.atr
        new_stop = _better(pos.side, new_stop, trail)
    if new_stop != pos.stop_loss:
        pos.stop_loss = new_stop
        return True
    return False


def check_exit(pos: Position, high: float, low: float,
               open_: float | None = None) -> tuple[float, str] | None:
    """Return ``(fill_price, reason)`` if the stop or the target was reached.

    When both are touched inside the same bar the stop wins (pessimistic).
    If the bar opened beyond a level (gap), the fill is the open price.
    """
    stop_reason = "stop_loss" if pos.stop_loss == pos.initial_stop else "trailing_stop"
    if pos.side == LONG:
        if low <= pos.stop_loss:
            fill = pos.stop_loss if open_ is None else min(pos.stop_loss, open_)
            return fill, stop_reason
        if high >= pos.take_profit:
            fill = pos.take_profit if open_ is None else max(pos.take_profit, open_)
            return fill, "take_profit"
    else:
        if high >= pos.stop_loss:
            fill = pos.stop_loss if open_ is None else max(pos.stop_loss, open_)
            return fill, stop_reason
        if low <= pos.take_profit:
            fill = pos.take_profit if open_ is None else min(pos.take_profit, open_)
            return fill, "take_profit"
    return None


def apply_slippage(price: float, side: str, opening: bool, s) -> float:
    """Adverse slippage: buy higher, sell lower."""
    buying = (side == LONG) == opening
    factor = s.slippage_bps / 1e4
    return price * (1 + factor) if buying else price * (1 - factor)


def realize(pos: Position, fill_price: float, closed_at: int, reason: str, s
            ) -> tuple[ClosedTrade, float]:
    """Simulated close. Returns the trade and the cash delta for the balance.

    The entry fee was already taken from the balance when the position opened,
    so the balance delta is the gross PnL minus the exit fee.
    """
    exit_px = apply_slippage(fill_price, pos.side, opening=False, s=s)
    gross = pos.dir * (exit_px - pos.entry_price) * pos.qty
    if s.margin_type == "ISOLATED":
        gross = max(gross, -pos.margin)  # isolated margin: liquidation caps the loss
    exit_fee = exit_px * pos.qty * s.fee_rate
    net = gross - exit_fee - pos.entry_fee
    trade = ClosedTrade(
        symbol=pos.symbol, side=pos.side, qty=pos.qty,
        entry_price=pos.entry_price, exit_price=exit_px,
        leverage=pos.leverage, margin=pos.margin,
        opened_at=pos.opened_at, closed_at=closed_at,
        pnl=net, fees=pos.entry_fee + exit_fee,
        r_multiple=net / pos.risk_amount if pos.risk_amount else 0.0,
        reason=reason,
    )
    return trade, gross - exit_fee
