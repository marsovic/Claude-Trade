"""Glue between the engine and a broker.

* ``SimExecution``  - fills on the next daily bar (paper trading, backtests).
* ``LiveExecution`` - real broker. Orders decided after the close are kept as
  "planned" and sent when the US session opens; fills are reconciled by
  polling the broker; protective stops rest at the broker when it allows it,
  otherwise the bot watches prices and sells itself.

Real-money safety rules implemented here:

* a resting stop is cancelled in two phases: request, then wait until the
  broker confirms the cancellation before selling or placing a new stop
  (Trading 212 locks shares under a pending stop; IBKR margin accounts could
  go short if a sell raced a stop fill);
* an order whose submission timed out is looked up at the broker, never resent;
* partial fills are booked as they happen, so the filled part gets a stop;
* entries planned for a previous session are dropped, exits are de-duplicated;
* a position only counts as "closed by hand" after two consecutive checks.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime

import pandas as pd

from invest.brokers.base import BrokerError, Fill, LiveBroker, OrderRequest
from invest.brokers.sim import SimBroker
from invest.data import ET

log = logging.getLogger("invest")


class SimExecution:
    fractional_default = True

    def __init__(self, s, state, fees):
        self.s = s
        self.state = state
        self.broker = SimBroker(s, state, fees)
        self.fractional = s.fractional

    def resolve(self, symbol: str) -> str | None:
        return symbol

    def has_open_sell(self, symbol: str) -> bool:
        return any(o["symbol"] == symbol and o["side"] == "SELL" for o in self.state.pending)

    def queue(self, engine, order: OrderRequest) -> bool:
        if order.side == "SELL" and self.has_open_sell(order.symbol):
            return False
        self.broker.queue(order)
        return True

    def settle(self, engine, day, data) -> None:
        fills, expired = self.broker.settle_open(day, data)
        for f in fills:
            engine.apply_fill(f)
        for o in expired:
            engine.on_order_dead(o, "limit not reached at the open" if o.kind == "LIMIT"
                                 else "no bar for the symbol")
        for f in self.broker.settle_stops(day, data, engine.state.positions):
            engine.apply_fill(f)

    def on_stop_change(self, engine, pos) -> None:
        pass  # the simulator reads pos.stop_loss directly

    def on_position_closed(self, engine, pos, fill) -> None:
        self.broker.cancel_symbol(pos.symbol)

    def cancel_entries(self, engine) -> None:
        self.state.pending = [o for o in self.state.pending if o["side"] != "BUY"]

    def liquidate(self, engine, pos, day) -> None:
        slip = self.s.slippage_bps / 1e4
        price = pos.last_price * (1 - slip)
        fx = engine.fx_now
        fees = self.broker._fee_eur(pos.qty, price, pos.currency, fx)
        engine.apply_fill(Fill(symbol=pos.symbol, side="SELL", qty=pos.qty, price=price,
                               fees_eur=fees, usd_per_eur=fx, day=str(day.date()),
                               ts=int(pd.Timestamp(day).timestamp() * 1000), reason="death"))

    def broker_equity(self) -> float | None:
        return None

    def broker_cash(self) -> float | None:
        return None


class LiveExecution:
    UNKNOWN_LIMIT = 5  # syncs before an order the broker does not know is dropped
    SUBMIT_ATTEMPTS = 3

    def __init__(self, s, state, store, broker: LiveBroker, clock=time.time):
        self.s = s
        self.state = state
        self.store = store
        self.broker = broker
        self.clock = clock
        self.fractional = broker.fractional and s.fractional
        self._tickers: dict[str, str | None] = {}
        self._unknown: dict[str, int] = {}
        self._account = None

    # ------------------------------------------------------------ helpers
    def _log(self, kind, message, symbol=None):
        self.store.log_decision(kind, message, symbol, "LONG" if symbol else None)
        log.info("%s %s %s", kind, symbol or "", message)

    def _call(self, fn, *args):
        """Every broker call goes through here: any failure becomes a BrokerError."""
        try:
            return fn(*args)
        except BrokerError:
            raise
        except Exception as exc:  # e.g. ConnectionError from a dropped IB Gateway
            raise BrokerError(f"{type(exc).__name__}: {exc}") from exc

    def resolve(self, symbol: str) -> str | None:
        if symbol not in self._tickers:
            try:
                self._tickers[symbol] = self._call(self.broker.resolve, symbol)
            except BrokerError as exc:
                self._log("error", f"cannot resolve instrument: {exc}", symbol)
                return None
        return self._tickers[symbol]

    def refresh_account(self):
        try:
            self._account = self._call(self.broker.account)
        except BrokerError as exc:
            self._log("error", f"account unavailable: {exc}")
            self._account = None
        return self._account

    def broker_equity(self) -> float | None:
        eq = self._account.equity_eur if self._account else None
        return eq if eq is not None and eq > 0 else None  # never trust a missing/zero read

    def broker_cash(self) -> float | None:
        return self._account.cash_eur if self._account else None

    def _open_for(self, symbol: str, side: str | None = None, kind: str | None = None) -> list:
        return [od for od in self.state.open_orders.values()
                if od["symbol"] == symbol and (side is None or od["side"] == side)
                and (kind is None or od["kind"] == kind)]

    def has_open_sell(self, symbol: str) -> bool:
        planned = any(o["symbol"] == symbol and o["side"] == "SELL" for o in self.state.pending)
        sent = any(od["kind"] != "STOP" for od in self._open_for(symbol, "SELL"))
        return planned or sent

    # ------------------------------------------------------------- orders
    def queue(self, engine, order: OrderRequest) -> bool:
        if order.side == "SELL" and self.has_open_sell(order.symbol):
            return False  # one exit per position is enough
        if not order.broker_ticker:
            order.broker_ticker = self.resolve(order.symbol) or ""
        if not order.broker_ticker:
            self._log("skip", "not tradable at this broker", order.symbol)
            return False
        order.status = "planned"
        self.state.pending.append(order.to_dict())
        return True

    def _track(self, order: OrderRequest, **extra) -> None:
        self.state.open_orders[order.id] = {**order.to_dict(), **extra}

    def _submit(self, order: OrderRequest) -> bool:
        """Send one order. A timeout is tracked as 'unconfirmed', never resent."""
        submitted_at = int(self.clock() * 1000)
        try:
            order.broker_id = self._call(self.broker.submit, order)
        except BrokerError as exc:
            if "outcome unknown" in str(exc):
                order.status = "unconfirmed"
                self._track(order, submitted_at=submitted_at)
                self._log("warning", f"{order.side} {order.qty:g} outcome unknown: it will "
                          "be looked up at the broker, not resent", order.symbol)
                return True
            raise
        order.status = "submitted"
        self._track(order, submitted_at=submitted_at)
        return True

    def submit_planned(self, engine) -> int:
        """Send the orders planned after the last close (call during the session)."""
        self.sync(engine)
        sent = 0
        keep = []
        orders = [OrderRequest.from_dict(o) for o in self.state.pending]
        orders.sort(key=lambda o: o.side != "SELL")
        for o in orders:
            raw = next(p for p in self.state.pending if p["id"] == o.id)
            if o.side == "BUY":
                if self.state.last_day and o.day and o.day < self.state.last_day:
                    self._log("skip", f"entry planned on {o.day} is stale: dropped", o.symbol)
                    continue
                if o.reason == "entry" and o.symbol in self.state.positions:
                    continue
            else:
                pos = self.state.positions.get(o.symbol)
                if pos is None or any(od["kind"] != "STOP"
                                      for od in self._open_for(o.symbol, "SELL")):
                    continue
                if pos.stop_order_id:
                    self._request_stop_cancel(pos)
                    keep.append(raw)  # sell once the broker confirms the stop is gone
                    continue
                o.qty = pos.qty
            try:
                self._submit(o)
            except BrokerError as exc:
                attempts = raw.get("attempts", 0) + 1
                self._log("error", f"{o.side} {o.qty:g} failed (attempt {attempts}): {exc}",
                          o.symbol)
                if attempts < self.SUBMIT_ATTEMPTS:
                    keep.append({**raw, "attempts": attempts})
                continue
            sent += 1
            self._log("order", f"sent {o.side} {o.kind} {o.qty:g}"
                      + (f" limit {o.limit_price:.4g}" if o.limit_price else "")
                      + f" ({o.reason}) id {o.broker_id or '?'}", o.symbol)
        self.state.pending = keep
        self.ensure_stops(engine)
        return sent

    # -------------------------------------------------------------- stops
    def _request_stop_cancel(self, pos) -> None:
        od = self.state.open_orders.get(pos.stop_order_id)
        if od is None:
            pos.stop_order_id = ""
            return
        if od.get("cancel_requested"):
            return
        if not od.get("broker_id"):
            return  # unconfirmed stop: wait until sync finds it
        try:
            self._call(self.broker.cancel, od["broker_id"])
            od["cancel_requested"] = True
        except BrokerError as exc:
            self._log("error", f"could not cancel the stop: {exc}", pos.symbol)

    def on_stop_change(self, engine, pos) -> None:
        self.ensure_stops(engine, only=pos.symbol)

    def ensure_stops(self, engine, only: str | None = None) -> None:
        """Every position gets exactly one resting stop at its current level and size."""
        if not self.broker.resting_stops:
            return  # watched by check_monitored_stops
        for sym, pos in list(self.state.positions.items()):
            if only and sym != only:
                continue
            if self.has_open_sell(sym):
                continue  # an exit is planned or working: no new stop (shares are busy)
            qty = pos.qty if self.fractional else float(int(pos.qty))
            if qty <= 0:
                continue
            current = self.state.open_orders.get(pos.stop_order_id) if pos.stop_order_id \
                else None
            if current is not None:
                if current.get("cancel_requested"):
                    continue  # wait for the confirmation
                stale = (abs(current["qty"] - qty) > 1e-9
                         or (current.get("stop_price") or 0) < pos.stop_loss * (1 - 1e-6))
                if stale:
                    self._request_stop_cancel(pos)
                continue
            pos.stop_order_id = ""
            order = OrderRequest(symbol=sym, side="SELL", qty=qty, kind="STOP",
                                 stop_price=pos.stop_loss, reason="stop", sleeve=pos.sleeve,
                                 currency=pos.currency, broker_ticker=pos.broker_ticker)
            try:
                self._submit(order)
            except BrokerError as exc:
                self._log("error", f"stop not placed ({exc}): watched by the bot, retried "
                          "next cycle", sym)
                continue
            pos.stop_order_id = order.id

    def on_position_closed(self, engine, pos, fill) -> None:
        if pos.stop_order_id and pos.stop_order_id != (fill.order or {}).get("id"):
            self._request_stop_cancel(pos)

    def cancel_entries(self, engine) -> None:
        """Death: no new exposure, ever. Planned and working buys are cancelled."""
        self.state.pending = [o for o in self.state.pending if o["side"] != "BUY"]
        for od in list(self.state.open_orders.values()):
            if od["side"] == "BUY" and od.get("broker_id") and not od.get("cancel_requested"):
                try:
                    self._call(self.broker.cancel, od["broker_id"])
                    od["cancel_requested"] = True
                except BrokerError as exc:
                    self._log("error", f"could not cancel entry, CANCEL IT BY HAND: {exc}",
                              od["symbol"])

    def liquidate(self, engine, pos, day) -> None:
        """Queue a market sale; it is sent as soon as the stop is confirmed cancelled."""
        order = OrderRequest(symbol=pos.symbol, side="SELL", qty=pos.qty, kind="MARKET",
                             reason="death", sleeve=pos.sleeve, currency=pos.currency,
                             broker_ticker=pos.broker_ticker)
        self.state.pending = [o for o in self.state.pending if o["symbol"] != pos.symbol]
        self.state.pending.append(order.to_dict())
        self._log("order", f"liquidation SELL {pos.qty:g} planned", pos.symbol)

    def busy(self) -> bool:
        return bool(self.state.pending or self.state.open_orders)

    # ---------------------------------------------------------- reconcile
    def sync(self, engine) -> int:
        if not self.state.open_orders:
            return 0
        try:
            self._call(self.broker.begin_sync)
        except BrokerError as exc:
            self._log("error", f"sync failed: {exc}")
            return 0
        applied = 0
        for oid, od in list(self.state.open_orders.items()):
            o = OrderRequest.from_dict(od)
            try:
                if not o.broker_id:
                    found = self._call(self.broker.find, o, od.get("submitted_at", 0))
                    if not found:
                        self._unconfirmed_miss(oid, o)
                        continue
                    o.broker_id = od["broker_id"] = found
                upd = self._call(self.broker.order_update, o)
            except BrokerError as exc:
                self._log("error", f"order status unavailable: {exc}", o.symbol)
                continue
            applied += self._apply_update(engine, oid, od, o, upd)
        self.ensure_stops(engine)
        return applied

    def _unconfirmed_miss(self, oid: str, o: OrderRequest) -> None:
        self._unknown[oid] = self._unknown.get(oid, 0) + 1
        if self._unknown[oid] >= self.UNKNOWN_LIMIT:
            del self.state.open_orders[oid]
            pos = self.state.positions.get(o.symbol)
            if pos is not None and pos.stop_order_id == oid:
                pos.stop_order_id = ""
            self._log("warning", f"{o.side} {o.kind} {o.qty:g} never showed up at the broker: "
                      "dropped (check the broker app)", o.symbol)

    def _apply_update(self, engine, oid, od, o: OrderRequest, upd) -> int:
        applied_qty = od.get("applied_qty", 0.0)
        total = upd.filled_qty or (o.qty if upd.status == "filled" else 0.0)
        increment = total - applied_qty
        terminal = upd.status in ("filled", "cancelled", "rejected")
        if increment > 1e-9:
            if upd.avg_price <= 0 and not (o.limit_price or o.stop_price):
                self._log("warning", "fill without a price yet: retried next sync", o.symbol)
                return 0
            if o.kind == "STOP" and o.symbol in self.state.positions:
                pos = self.state.positions[o.symbol]
                if pos.stop_order_id == oid and terminal:
                    pos.stop_order_id = ""
            od["applied_qty"] = total
            if terminal:
                del self.state.open_orders[oid]
            engine.apply_fill(self._fill(engine, o, upd, increment))
            return 1
        if terminal:
            del self.state.open_orders[oid]
            if o.kind == "STOP":
                pos = self.state.positions.get(o.symbol)
                if pos and pos.stop_order_id == oid:
                    pos.stop_order_id = ""
                    if not od.get("cancel_requested"):
                        self._log("warning", f"stop {upd.status} by the broker: watched by "
                                  "the bot until it is placed again", o.symbol)
            elif not od.get("cancel_requested"):
                engine.on_order_dead(o, upd.status)
        elif upd.status == "unknown":
            self._unconfirmed_miss(oid, o)
        return 0

    def _fill(self, engine, o: OrderRequest, upd, qty: float) -> Fill:
        price = upd.avg_price or o.limit_price or o.stop_price or 0.0
        fx = engine.fx_now
        fees = upd.fees_eur * (qty / upd.filled_qty) if upd.fees_eur and upd.filled_qty \
            else 0.0
        if not fees:
            foreign = o.currency != "EUR"
            cost = self.s.fees.cost(qty, price, foreign)
            fees = cost / fx if foreign else cost
        ts = upd.filled_at or int(self.clock() * 1000)
        day = datetime.fromtimestamp(ts / 1000, tz=ET).date()
        reason = o.reason
        if o.kind == "STOP":
            pos = self.state.positions.get(o.symbol)
            reason = "trailing stop" if pos is not None and pos.stop_loss >= pos.entry_price \
                else "stop"
        return Fill(symbol=o.symbol, side=o.side, qty=qty, price=price, fees_eur=fees,
                    usd_per_eur=fx, day=str(day), ts=ts, reason=reason, order=o.to_dict(),
                    fill_id=o.broker_id)

    def settle(self, engine, day, data) -> None:
        """End of day: reconcile fills, cancel unfilled entries, detect manual closes."""
        self.sync(engine)
        for od in list(self.state.open_orders.values()):
            if od["side"] == "BUY" and od.get("broker_id") and not od.get("cancel_requested"):
                try:
                    self._call(self.broker.cancel, od["broker_id"])
                    od["cancel_requested"] = True
                    self._log("skip", "entry not filled today: cancelled", od["symbol"])
                except BrokerError as exc:
                    self._log("error", f"could not cancel stale entry: {exc}", od["symbol"])
        self.sync(engine)
        account = self.refresh_account()
        if account is None:
            return
        held = {p.ticker for p in account.positions.values() if p.qty > 0}
        missing = self.state.missing_checks
        for sym, pos in list(self.state.positions.items()):
            busy = self._open_for(sym, "SELL") and any(
                od["kind"] != "STOP" for od in self._open_for(sym, "SELL"))
            if not pos.broker_ticker or pos.broker_ticker in held or busy:
                missing.pop(sym, None)
                continue
            missing[sym] = missing.get(sym, 0) + 1
            if missing[sym] < 2:
                self._log("warning", "position not seen at the broker: will check again",
                          sym)
                continue
            missing.pop(sym, None)
            self._log("warning", "position gone at the broker twice in a row (closed by "
                      "hand?): booked at the last price", sym)
            if pos.stop_order_id:
                self._request_stop_cancel(pos)
            engine.apply_fill(Fill(symbol=sym, side="SELL", qty=pos.qty,
                                   price=pos.last_price, fees_eur=0.0,
                                   usd_per_eur=engine.fx_now, day=str(day.date()),
                                   ts=int(self.clock() * 1000),
                                   reason="closed outside the bot"))

    def check_monitored_stops(self, engine, prices: dict[str, float]) -> None:
        """Positions without a resting stop: sell when the last price breaks it."""
        for sym, pos in list(self.state.positions.items()):
            if self.broker.resting_stops and pos.stop_order_id:
                continue
            price = prices.get(sym)
            if price is None or price > pos.stop_loss or self.has_open_sell(sym):
                continue
            order = OrderRequest(symbol=sym, side="SELL", qty=pos.qty, kind="MARKET",
                                 reason="stop (watched)", sleeve=pos.sleeve,
                                 currency=pos.currency, broker_ticker=pos.broker_ticker)
            try:
                self._submit(order)
            except BrokerError as exc:
                self._log("error", f"stop sale failed: {exc}", sym)
                continue
            self._log("order", f"price {price:.4g} broke the stop {pos.stop_loss:.4g}: "
                      "SELL at market", sym)

    def broker_prices(self) -> dict[str, float]:
        """Last prices of held positions, from the broker (map back to symbols)."""
        if self._account is None:
            return {}
        by_ticker = {p.ticker: p.price for p in self._account.positions.values()}
        return {sym: by_ticker[pos.broker_ticker]
                for sym, pos in self.state.positions.items()
                if pos.broker_ticker in by_ticker and by_ticker[pos.broker_ticker] > 0}
