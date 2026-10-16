"""Real spot orders on Revolut X (long only, no leverage, EUR by default).

Differences with the Binance futures broker:

* spot: a position is coins really bought. No short selling, no leverage,
  no liquidation. A sell is capped by the coins actually available and by the
  position's own quantity, so a repeated sell can never sell twice and never
  touches coins the bot did not buy.
* the REST API has no stop orders: the bot reads the bid every poll
  (POLL_SECONDS) and sells at market once the stop is touched. While the bot
  is down nobody watches the stop: run it under a supervisor (systemd...).
* the take-profit rests on the exchange as a GTC limit sell (0 % maker fee),
  so it still executes while the bot is offline.
* entries are IOC limit buys capped at ``ask * (1 + REVX_ENTRY_SLIPPAGE)``.
* every order carries a client_order_id that the exchange refuses twice. An
  entry whose reply was lost is saved before sending and looked up by that id
  on the next cycles; it is never sent again under a new id.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import asdict

from bot.revolutx_client import (
    TERMINAL_STATES,
    RevolutXClient,
    RevolutXDuplicateOrder,
    RevolutXError,
    RevolutXOutcomeUnknown,
    order_fill,
)
from bot.risk import ClosedTrade, Position, TradePlan, update_stops
from bot.state import BotState, StateStore
from bot.strategy import LONG

log = logging.getLogger(__name__)

PENDING_TIMEOUT_MS = 15 * 60_000  # a lost entry not found after this is assumed unplaced
SETTLE_POLLS = 10
SELL_ATTEMPTS = 3


class RevolutXBroker:
    name = "revolutx"

    def __init__(self, settings, state: BotState, store: StateStore, client: RevolutXClient,
                 symbols: dict, sleep=time.sleep):
        self.s = settings
        self.state = state
        self.store = store
        self.client = client
        self.symbols = symbols
        self.sleep = sleep
        self._balances: dict[str, dict[str, float]] = {}
        self._low_readings = 0
        self._gone: dict[str, int] = {}

    # ----------------------------------------------------------- preflight
    def preflight(self) -> list[str]:
        """Checks before trading real money. Returns warnings, raises if unsafe."""
        try:
            bal = self.client.balances()  # proves the key, the signature and the IP rule
        except RevolutXError as exc:
            hint = " (is this server's IP whitelisted on the API key?)" if exc.status == 401 else ""
            raise RuntimeError(f"Revolut X refused the API key: {exc}{hint}") from exc
        self._balances = bal
        quote = self.s.quote_asset
        cash = bal.get(quote, {}).get("total", 0.0)
        warnings = []
        if not self.state.positions and not self.state.pending_entries:
            if cash <= self.s.survival_floor:
                raise RuntimeError(
                    f"the Revolut X account holds {cash:.2f} {quote}: deposit at least "
                    f"INITIAL_CAPITAL ({self.s.initial_capital:.2f} {quote}) before going live.")
            if cash < self.s.initial_capital:
                warnings.append(f"{quote} balance {cash:.2f} is below INITIAL_CAPITAL "
                                f"{self.s.initial_capital:.2f}: equity is capped to the account.")
        own = {self.symbols[s].base_asset for s in self.state.positions if s in self.symbols}
        held = sorted(c for c, b in bal.items() if c != quote and b["total"] > 0 and c not in own)
        if held:
            warnings.append(f"coins already held ({', '.join(held[:10])}): never sold by the bot, "
                            "which only sells what it bought itself.")
        mine = {p.tp_order_id for p in self.state.positions.values() if p.tp_order_id}
        try:
            foreign = [o for o in self.client.active_orders() if o.get("id") not in mine]
        except RevolutXError:
            foreign = []
        if foreign:
            warnings.append(f"{len(foreign)} open order(s) not placed by the bot: left untouched, "
                            "but they may reserve coins or cash.")
        return warnings

    # -------------------------------------------------------------- helpers
    def _quote(self, symbol: str, side: str, fallback: float) -> float:
        return (self.client.last_quotes.get(symbol) or {}).get(side) or fallback

    def _settle(self, order_id: str) -> dict:
        """The order document once terminal (a fresh order takes a moment to show)."""
        doc: dict = {"id": order_id, "status": "unknown"}
        for i in range(SETTLE_POLLS):
            try:
                doc = self.client.order(order_id) or doc
            except RevolutXError as exc:
                if exc.status != 404:
                    raise
            if doc.get("status") in TERMINAL_STATES:
                return doc
            self.sleep(0.5 if i < 4 else 1.0)
        return doc

    def _fill_value(self, order: dict, base: str) -> tuple[float, float, float]:
        """``(qty, quote proceeds or cost, fee in quote)`` of one order."""
        qty, avg, fee, fee_ccy = order_fill(order)
        return qty, qty * avg, fee * avg if fee_ccy == base else fee

    # ---------------------------------------------------------------- entry
    def open(self, plan: TradePlan, price: float, now: int) -> Position:
        if plan.side != LONG:
            raise RuntimeError("Revolut X is a spot exchange: no short selling")
        info = self.symbols[plan.symbol]
        limit = info.round_price(self._quote(plan.symbol, "ask", price)
                                 * (1 + self.s.revx_entry_slippage))
        qty = info.round_qty(plan.qty)
        if qty <= 0:
            raise RuntimeError("quantity rounds to zero")
        cid = str(uuid.uuid4())
        # Written before sending: a crash or a lost reply cannot lose the order.
        self.state.pending_entries[cid] = {"plan": asdict(plan), "placed_at": now}
        self.store.save(self.state)
        try:
            order_id = self.client.place_order(plan.symbol, "buy", base_size=qty, price=limit,
                                               time_in_force="ioc",
                                               client_order_id=cid)["venue_order_id"]
        except RevolutXDuplicateOrder:
            order_id = None  # placed: looked up by its client_order_id
        except RevolutXOutcomeUnknown as exc:
            raise RuntimeError(f"entry outcome unknown ({exc}): looked up by "
                               "client_order_id on the next cycles") from exc
        except RevolutXError:
            self.state.pending_entries.pop(cid, None)  # refused: nothing was placed
            raise
        # From here the order exists: any failure leaves it to _reconcile_pending.
        try:
            if order_id is None:
                found = self.client.find_order(cid, plan.symbol, now)
                if found is None:
                    raise RuntimeError("entry accepted but not visible yet: will reconcile")
                order_id = found["id"]
            self.state.pending_entries[cid]["order_id"] = order_id
            order = self._settle(order_id)
        except RevolutXError as exc:
            raise RuntimeError(f"entry placed, status unavailable ({exc}): "
                               "will reconcile") from exc
        if order.get("status") not in TERMINAL_STATES:
            raise RuntimeError("entry order not settled yet: will reconcile")
        self.state.pending_entries.pop(cid, None)
        pos = self._position_from_order(plan, order, now)
        if pos is None:
            raise RuntimeError(f"IOC buy not filled at <= {limit} (the price moved away)")
        self._place_take_profit(pos)
        return pos

    def _position_from_order(self, plan: TradePlan, order: dict, now: int) -> Position | None:
        info = self.symbols[plan.symbol]
        filled, avg, fee, fee_ccy = order_fill(order)
        if filled <= 0 or avg <= 0:
            return None
        fee_in_base = fee_ccy == info.base_asset
        held = filled - fee if fee_in_base else filled
        cost = filled * avg + (0.0 if fee_in_base else fee)
        stop = float(info.round_price(avg - (plan.price - plan.stop_loss)))
        target = float(info.round_price(avg + (plan.take_profit - plan.price)))
        scale = filled / plan.qty if plan.qty else 1.0
        return Position(
            symbol=plan.symbol, side=LONG, qty=held, entry_price=avg, leverage=1,
            stop_loss=stop, take_profit=target, initial_stop=stop, atr=plan.atr,
            margin=cost, risk_amount=plan.risk_amount * scale, opened_at=now,
            entry_fee=fee * avg if fee_in_base else fee, cost=cost)

    def _place_take_profit(self, pos: Position) -> None:
        info = self.symbols[pos.symbol]
        qty = info.round_qty(pos.qty)
        price = info.round_price(pos.take_profit)
        if qty <= 0 or qty < info.min_qty or qty * price < info.min_notional:
            return
        pos.tp_client_id = str(uuid.uuid4())
        try:
            placed = self.client.place_order(pos.symbol, "sell", base_size=qty, price=price,
                                             client_order_id=pos.tp_client_id)
            pos.tp_order_id = placed["venue_order_id"]
        except RevolutXDuplicateOrder:
            pass  # placed: adopted by its client_order_id in sync()
        except RevolutXError as exc:  # the bot watches the target itself meanwhile
            log.warning("%s: take-profit not placed (%s): watched by the bot", pos.symbol, exc)
            if not isinstance(exc, RevolutXOutcomeUnknown):
                pos.tp_client_id = None

    # ---------------------------------------------------------------- exit
    def _cancel_take_profit(self, pos: Position) -> None:
        """Cancel the resting take-profit; what it already sold stays on the books."""
        ids = [pos.tp_order_id] if pos.tp_order_id else []
        if pos.tp_client_id:
            ids += [o["id"] for o in self.client.active_orders([pos.symbol])
                    if o.get("client_order_id") == pos.tp_client_id and o["id"] not in ids]
        for oid in ids:
            # Raises on a network error: the coins stay reserved, nothing can be sold.
            self.client.cancel_order(oid)
            if oid not in pos.exit_order_ids:
                pos.exit_order_ids.append(oid)
        pos.tp_order_id = pos.tp_client_id = None

    def _exit_fills(self, pos: Position) -> tuple[float, float, float]:
        """``(sold, proceeds, fees)`` over every exit order, recomputed each time."""
        for cid in list(pos.exit_client_ids):
            found = self.client.find_order(cid, pos.symbol, pos.opened_at)
            if found:
                pos.exit_client_ids.remove(cid)
                pos.exit_order_ids.append(found["id"])
        base = self.symbols[pos.symbol].base_asset
        sold = proceeds = fees = 0.0
        for oid in pos.exit_order_ids:
            q, value, fee = self._fill_value(self._settle(oid), base)
            sold, proceeds, fees = sold + q, proceeds + value, fees + fee
        return sold, proceeds, fees

    def close(self, pos: Position, price: float, reason: str, now: int) -> ClosedTrade:
        info = self.symbols[pos.symbol]
        self._cancel_take_profit(pos)
        sold, proceeds, fees = self._exit_fills(pos)
        for _ in range(SELL_ATTEMPTS):
            bal = self.client.balances()
            self._balances = bal
            available = bal.get(info.base_asset, {}).get("available", 0.0)
            qty = info.round_qty(min(pos.qty - sold, available))
            bid = self._quote(pos.symbol, "bid", price)
            if qty <= 0 or qty < info.min_qty or float(qty) * bid < float(info.min_notional):
                break  # nothing (left) worth selling: dust or coins sold outside the bot
            cid = str(uuid.uuid4())
            pos.exit_client_ids.append(cid)
            self.store.save(self.state)  # a lost reply is found again by this id
            try:
                placed = self.client.place_order(pos.symbol, "sell", base_size=qty,
                                                 client_order_id=cid)
            except RevolutXDuplicateOrder:
                pass  # it exists: found by its client_order_id below
            except RevolutXOutcomeUnknown as exc:
                # Retried next cycle: the balance check makes a double sale impossible.
                raise RuntimeError(f"sell outcome unknown ({exc}): retried next cycle") from exc
            except RevolutXError:
                pos.exit_client_ids.remove(cid)  # refused: nothing was placed
                raise
            else:
                pos.exit_client_ids.remove(cid)
                pos.exit_order_ids.append(placed["venue_order_id"])
            new_sold, proceeds, fees = self._exit_fills(pos)
            if new_sold <= sold:
                raise RuntimeError("market sell not filled: retried next cycle")
            sold = new_sold
        if sold <= 0:
            # Nothing sold by the bot: the coins left the account (sold in the app).
            sold, proceeds = pos.qty, pos.qty * self._quote(pos.symbol, "bid", price)
            reason = "closed_outside"
        return self._finalize(pos, sold, proceeds, fees, reason, now)

    def _finalize(self, pos: Position, sold: float, proceeds: float, fees: float, reason: str,
                  now: int) -> ClosedTrade:
        # The whole cost is charged: unsold dust is written off.
        pnl = proceeds - fees - pos.cost
        self.state.balance += pnl
        return ClosedTrade(
            symbol=pos.symbol, side=LONG, qty=sold, entry_price=pos.entry_price,
            exit_price=proceeds / sold if sold else pos.entry_price, leverage=1,
            margin=pos.cost, opened_at=pos.opened_at, closed_at=now, pnl=pnl,
            fees=pos.entry_fee + fees,
            r_multiple=pnl / pos.risk_amount if pos.risk_amount else 0.0, reason=reason)

    # ----------------------------------------------------------------- sync
    def sync(self, prices: dict[str, float], now: int) -> list[ClosedTrade]:
        self._reconcile_pending(now)
        try:
            self._balances = self.client.balances()
        except RevolutXError as exc:  # stops are still watched without balances
            log.warning("balances unavailable: %s", exc)
            self._balances = {}
        closed = []
        for symbol, pos in list(self.state.positions.items()):
            if symbol not in self.symbols:
                continue
            try:
                trade = self._sync_position(pos, prices, now)
            except RuntimeError as exc:  # one position must not block the others
                log.error("%s: %s", symbol, exc)
                self.store.log_decision("error", str(exc), symbol, pos.side, ts=now)
                continue
            if trade:
                closed.append(trade)
        return closed

    def _sync_position(self, pos: Position, prices: dict[str, float],
                       now: int) -> ClosedTrade | None:
        info = self.symbols[pos.symbol]
        bid = self._quote(pos.symbol, "bid", prices.get(pos.symbol) or 0.0)
        if bid <= 0:
            return None
        tp_sold = 0.0
        if pos.tp_order_id:
            try:
                tp = self.client.order(pos.tp_order_id)
            except RevolutXError as exc:
                log.warning("%s: take-profit status unavailable: %s", pos.symbol, exc)
                tp = {}
            tp_sold = order_fill(tp)[0] if tp else 0.0
            if tp.get("status") == "filled":
                q, value, fee = self._fill_value(tp, info.base_asset)
                pos.tp_order_id = pos.tp_client_id = None
                return self._finalize(pos, q, value, fee, "take_profit", now)
            if tp.get("status") in ("cancelled", "rejected", "replaced"):
                self.store.log_decision("warning", "take-profit order gone from the exchange: "
                                        "the bot now watches the target itself", pos.symbol,
                                        pos.side, ts=now)
                pos.tp_order_id = pos.tp_client_id = None
        elif pos.tp_client_id:
            try:
                found = self.client.find_order(pos.tp_client_id, pos.symbol, pos.opened_at)
            except RevolutXError as exc:
                log.warning("%s: take-profit lookup failed: %s", pos.symbol, exc)
                found = {}
            if found and found.get("status") not in TERMINAL_STATES:
                pos.tp_order_id = found["id"]
            elif found is None and now - pos.opened_at > PENDING_TIMEOUT_MS:
                pos.tp_client_id = None

        quote = self.s.quote_asset
        if self._balances and quote in self._balances:
            held = self._balances.get(info.base_asset, {}).get("total", 0.0)
            expected = pos.qty - tp_sold
            # Two readings in a row, so that one odd balance reply cannot close anything.
            gone = expected > 0 and held < expected * 0.5
            self._gone[pos.symbol] = self._gone.get(pos.symbol, 0) + 1 if gone else 0
            if self._gone[pos.symbol] >= 2:
                self._gone.pop(pos.symbol, None)
                pos.tp_order_id = pos.tp_client_id = None
                return self._finalize(pos, pos.qty, pos.qty * bid, 0.0, "closed_outside", now)

        old_stop = pos.stop_loss
        if update_stops(pos, bid, self.s):
            self.store.log_decision("stop_moved", f"stop {old_stop:.6g} -> {pos.stop_loss:.6g} "
                                    "(watched by the bot)", pos.symbol, pos.side,
                                    {"bid": bid}, ts=now)
        if bid <= pos.stop_loss:
            reason = "stop_loss" if pos.stop_loss <= pos.initial_stop else "trailing_stop"
            return self.close(pos, bid, reason, now)
        if not pos.tp_order_id and not pos.tp_client_id and bid >= pos.take_profit:
            return self.close(pos, bid, "take_profit", now)
        return None

    def _reconcile_pending(self, now: int) -> None:
        """Adopt or drop entries whose reply was lost."""
        for cid, p in list(self.state.pending_entries.items()):
            plan = TradePlan(**p["plan"])
            try:
                order = (self.client.order(p["order_id"]) if p.get("order_id")
                         else self.client.find_order(cid, plan.symbol, p["placed_at"]))
            except RevolutXError as exc:
                if exc.status != 404:
                    log.warning("%s: pending entry lookup failed: %s", plan.symbol, exc)
                    continue
                order = None
            if not order:
                if now - p["placed_at"] > PENDING_TIMEOUT_MS:
                    self.state.pending_entries.pop(cid)
                    self.store.log_decision("warning", f"entry {cid} never appeared on the "
                                            "exchange: assumed not placed", plan.symbol,
                                            plan.side, ts=now)
                continue
            if order.get("status") not in TERMINAL_STATES:
                if now - p["placed_at"] > 60_000:
                    try:
                        self.client.cancel_order(order["id"])  # an IOC should be long done
                    except RevolutXError as exc:
                        log.warning("%s: could not cancel a stale entry: %s", plan.symbol, exc)
                continue
            self.state.pending_entries.pop(cid)
            pos = self._position_from_order(plan, order, now)
            if pos is None:
                self.store.log_decision("skip", "lost entry found unfilled", plan.symbol,
                                        plan.side, ts=now)
                continue
            self._place_take_profit(pos)
            self.state.positions[plan.symbol] = pos
            self.store.log_decision("open", f"entry found by its client_order_id: qty "
                                    f"{pos.qty:g} @ {pos.entry_price:.6g}", plan.symbol,
                                    plan.side, pos.to_dict(), ts=now)

    # --------------------------------------------------------------- equity
    def equity(self, prices: dict[str, float]) -> float:
        own = self.state.balance + sum(p.qty * prices.get(sym, p.entry_price) - p.cost
                                       for sym, p in self.state.positions.items())
        bal, quote = self._balances, self.s.quote_asset
        if not bal or quote not in bal:
            return own  # balances unreadable: trust the books for now
        account = bal[quote]["total"]
        for sym, p in self.state.positions.items():
            info = self.symbols.get(sym)
            held = bal.get(info.base_asset, {}).get("total", 0.0) if info else p.qty
            account += min(p.qty, held) * prices.get(sym, p.entry_price)
        if account <= self.s.survival_floor < own:
            # Only the exchange reading is under the floor: confirm before dying.
            self._low_readings += 1
            if self._low_readings < 2:
                log.warning("account reads %.2f %s vs %.2f in the books: confirming",
                            account, quote, own)
                return own
        else:
            self._low_readings = 0
        # Never pretend to own more than the account holds (cash taken out...).
        return min(own, account)
