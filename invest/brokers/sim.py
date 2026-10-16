"""Daily-bar simulator used by paper trading and the backtester.

Orders decided on a closed bar fill on the NEXT bar:

* market orders at the open (adverse slippage),
* buy limits at the open if it is below the limit, else at the limit if the
  day's low reaches it, else they expire (no chasing gaps),
* resting stops at the stop price, or at the open when the bar gaps through it.

Costs follow the configured fee model (Trading 212 by default: no commission,
0.15 % FX fee on USD instruments for a EUR account).
"""

from __future__ import annotations

from datetime import time as dtime

import pandas as pd

from invest.brokers.base import Fill, OrderRequest
from invest.data import ET
from invest.risk import stop_reason


def session_ts(day: pd.Timestamp, at: dtime) -> int:
    ts = pd.Timestamp.combine(pd.Timestamp(day).date(), at).tz_localize(ET)
    return int(ts.timestamp() * 1000)


class SimBroker:
    name = "sim"
    resting_stops = True

    def __init__(self, s, state, fees):
        self.s = s
        self.state = state
        self.fees = fees
        self.fractional = s.fractional

    def queue(self, order: OrderRequest) -> None:
        order.status = "submitted"
        self.state.pending.append(order.to_dict())

    def cancel_symbol(self, symbol: str) -> None:
        self.state.pending = [o for o in self.state.pending if o["symbol"] != symbol]

    def _bar(self, data, symbol: str, day: pd.Timestamp):
        df = data.bars.get(symbol)
        if df is None or day not in df.index:
            return None
        return df.loc[day]

    def _fee_eur(self, qty: float, price: float, currency: str, fx: float) -> float:
        foreign = currency != "EUR"
        cost = self.fees.cost(qty, price, foreign)
        return cost / fx if foreign else cost

    def settle_open(self, day: pd.Timestamp, data) -> tuple[list[Fill], list[OrderRequest]]:
        """Fill the orders queued on the previous close. Returns (fills, expired)."""
        fills, expired = [], []
        fx = data.fx(day)
        slip = self.s.slippage_bps / 1e4
        ts = session_ts(day, dtime(9, 30))
        orders = [OrderRequest.from_dict(o) for o in self.state.pending]
        orders.sort(key=lambda o: o.side != "SELL")  # sells first: they free cash
        carried = []
        for o in orders:
            bar = self._bar(data, o.symbol, day)
            if bar is None:
                if o.side == "SELL":
                    carried.append(o.to_dict())  # no bar (halt, data gap): sell next session
                    continue
                o.status = "cancelled"
                expired.append(o)
                continue
            if o.side == "SELL":
                price = bar["open"] * (1 - slip)
            elif o.kind == "LIMIT" and o.limit_price:
                if bar["open"] <= o.limit_price:
                    price = min(bar["open"] * (1 + slip), o.limit_price)
                elif bar["low"] <= o.limit_price:
                    price = o.limit_price
                else:
                    o.status = "cancelled"
                    expired.append(o)
                    continue
            else:
                price = bar["open"] * (1 + slip)
            o.status = "filled"
            fills.append(Fill(symbol=o.symbol, side=o.side, qty=o.qty, price=float(price),
                              fees_eur=self._fee_eur(o.qty, price, o.currency, fx),
                              usd_per_eur=fx, day=str(day.date()), ts=ts, reason=o.reason,
                              order=o.to_dict(), fill_id=o.id))
        self.state.pending = carried
        return fills, expired

    def settle_stops(self, day: pd.Timestamp, data, positions: dict) -> list[Fill]:
        fills = []
        fx = data.fx(day)
        slip = self.s.slippage_bps / 1e4
        ts = session_ts(day, dtime(15, 59))
        for sym, pos in list(positions.items()):
            bar = self._bar(data, sym, day)
            if bar is None or bar["low"] > pos.stop_loss:
                continue
            # A stop-market order fills worse when the market moves fast: scale with the
            # stock's own volatility instead of a flat 5 bp.
            stop_slip = max(slip, self.s.stop_slippage_atr * pos.atr / max(pos.stop_loss, 1e-9))
            price = min(bar["open"], pos.stop_loss) * (1 - stop_slip)
            reason = stop_reason(float(price), pos.stop_loss, pos.entry_price, stop_slip)
            fills.append(Fill(symbol=sym, side="SELL", qty=pos.qty, price=float(price),
                              fees_eur=self._fee_eur(pos.qty, price, pos.currency, fx),
                              usd_per_eur=fx, day=str(day.date()), ts=ts, reason=reason,
                              fill_id=f"stop-{sym}-{day.date()}"))
        return fills
