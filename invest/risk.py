"""Aggressive but bounded sizing, stops and pyramiding.

* Initial stop: 3 x ATR(22) below entry, clamped to 4-20 % of the price.
* Position weight = min(risk / stop distance, 1 / slots, 40 %), so a stop
  costs at most ``risk_per_trade`` of equity (gaps aside). No margin: the sum
  of weights never exceeds 100 %.
* Drawdown ladder: deeper than -20 % -> 3 % risk and one slot less;
  deeper than -35 % -> 2 % risk, a single slot and no leveraged ETP.
* Chandelier stop: max(previous stop, highest high since entry - 3 x ATR).
* Pyramiding: one add of 50 % of the initial size at +1.5 R on a new
  20-day closing high, then the whole position's stop goes to break-even.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from invest.config import FeeModel
from invest.portfolio import InvestPosition


@dataclass
class RiskBudget:
    risk_per_trade: float
    slots: int
    etp_allowed: bool
    level: int  # 0 normal, 1 and 2 = drawdown ladder steps


def risk_budget(equity: float, drawdown: float, s) -> RiskBudget:
    slots = s.slots_small_equity if equity < s.small_equity else s.slots
    if drawdown >= s.dd_level2:
        return RiskBudget(min(s.risk_per_trade, 0.02), 1, False, 2)
    if drawdown >= s.dd_level1:
        return RiskBudget(min(s.risk_per_trade, 0.03), max(1, slots - 1), s.etp_enabled, 1)
    return RiskBudget(s.risk_per_trade, slots, s.etp_enabled, 0)


def stop_distance_pct(price: float, atr: float, s, mult: float | None = None) -> float:
    mult = s.stop_atr_mult if mult is None else mult
    if price <= 0 or atr <= 0:
        return s.stop_max_pct
    return min(max(mult * atr / price, s.stop_min_pct), s.stop_max_pct)


@dataclass
class SizedOrder:
    qty: float
    notional_eur: float
    weight: float
    stop_price: float
    stop_pct: float
    risk_eur: float
    cost_eur: float  # estimated round-trip costs


def size_position(price: float, atr: float, equity: float, cash: float, open_risk_eur: float,
                  budget: RiskBudget, usd_per_eur: float, fees: FeeModel, s,
                  max_weight: float | None = None, foreign: bool = True
                  ) -> tuple[SizedOrder | None, str]:
    """Quantity for a new position, or (None, reason)."""
    if price <= 0 or equity <= 0:
        return None, "invalid price/equity"
    stop_pct = stop_distance_pct(price, atr, s)
    cap = max_weight if max_weight is not None else s.max_weight
    weight = min(budget.risk_per_trade / stop_pct, 1 / budget.slots, cap)
    risk_left = equity * s.max_open_risk - open_risk_eur
    if risk_left <= 0:
        return None, "total open risk limit reached"
    weight = min(weight, risk_left / (equity * stop_pct))
    notional_eur = min(weight * equity, cash * 0.995, equity * s.max_gross)
    if notional_eur <= 0:
        return None, "no cash available"
    rate = usd_per_eur if foreign else 1.0
    qty = notional_eur * rate / price
    if not s.fractional:
        qty = math.floor(qty)
        if qty < 1:
            return None, f"one share costs {price / rate:.2f} EUR, more than the slot"
    else:
        qty = math.floor(qty * 1e4) / 1e4  # Trading 212 accepts at most ~4 decimals
    if qty <= 0:
        return None, "quantity rounds to zero"
    notional_eur = qty * price / rate
    if notional_eur < s.min_order_eur:
        return None, f"order {notional_eur:.2f} EUR below the {s.min_order_eur:g} EUR minimum"
    risk_eur = notional_eur * stop_pct
    round_trip = (fees.cost(qty, price, foreign) * 2) / rate
    if risk_eur > 0 and round_trip / risk_eur > s.max_cost_r:
        return None, (f"costs {round_trip:.2f} EUR = {round_trip / risk_eur:.2f}R "
                      f"> {s.max_cost_r:g}R")
    return SizedOrder(qty=qty, notional_eur=notional_eur, weight=notional_eur / equity,
                      stop_price=price * (1 - stop_pct), stop_pct=stop_pct,
                      risk_eur=risk_eur, cost_eur=round_trip), "ok"


def chandelier_stop(pos: InvestPosition, atr: float, s, mult: float | None = None) -> float:
    """Ratchets up only."""
    mult = s.stop_atr_mult if mult is None else mult
    if atr <= 0:
        return pos.stop_loss
    # Same floor as the initial stop: never trail closer than stop_min_pct.
    distance = max(mult * atr, s.stop_min_pct * pos.highest)
    return max(pos.stop_loss, pos.highest - distance)


def stop_reason(fill: float, stop: float, entry: float, slip: float = 0.0) -> str:
    """Name a stop exit by its outcome, not by where the stop stood."""
    if fill >= entry:
        return "trailing stop"
    if fill < stop * (1 - slip) * 0.995:
        return "stop (gap)"  # the price jumped through the stop
    return "stop"


def pyramid_ok(pos: InvestPosition, close: float, new_high: bool, pct_rank: float | None,
               s) -> bool:
    return (s.pyramid and pos.adds == 0 and pos.sleeve == "stock" and new_high
            and pos.r_multiple(close) >= s.pyramid_r
            and pct_rank is not None and pct_rank >= 1 - s.top_pct_entry)


def breakeven_stop(pos: InvestPosition, fee_rate: float) -> float:
    return pos.entry_price * (1 + 2 * fee_rate)
