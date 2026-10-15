"""Order execution: simulated (paper) or real (Binance testnet / live).

Both brokers expose the same interface so the trading loop does not care
which one it drives:

* ``open(plan, price, now)``          -> Position
* ``close(pos, price, reason, now)``  -> ClosedTrade
* ``sync(prices, now)``               -> list of ClosedTrade (stops, targets, ...)
* ``equity(prices)``                  -> float
"""

from __future__ import annotations

import logging
import uuid

from bot.binance_client import BinanceAPIError, BinanceFuturesClient, SymbolInfo
from bot.risk import (
    ClosedTrade,
    Position,
    TradePlan,
    apply_slippage,
    check_exit,
    direction,
    realize,
    update_stops,
)
from bot.state import BotState, StateStore
from bot.strategy import LONG

log = logging.getLogger(__name__)

PNL_INCOME_TYPES = {"REALIZED_PNL", "COMMISSION", "FUNDING_FEE"}


class PaperBroker:
    """Fills at the last price with fees and adverse slippage. No order is sent."""

    name = "paper"

    def __init__(self, settings, state: BotState, store: StateStore):
        self.s = settings
        self.state = state
        self.store = store

    def open(self, plan: TradePlan, price: float, now: int) -> Position:
        fill = apply_slippage(price, plan.side, opening=True, s=self.s)
        # Keep the planned risk distance relative to the actual fill.
        d = direction(plan.side)
        stop_dist = abs(plan.price - plan.stop_loss)
        target_dist = abs(plan.take_profit - plan.price)
        fee = fill * plan.qty * self.s.fee_rate
        self.state.balance -= fee
        return Position(
            symbol=plan.symbol, side=plan.side, qty=plan.qty, entry_price=fill,
            leverage=plan.leverage, stop_loss=fill - d * stop_dist,
            take_profit=fill + d * target_dist, initial_stop=fill - d * stop_dist,
            atr=plan.atr, margin=plan.notional / plan.leverage,
            risk_amount=plan.risk_amount, opened_at=now, entry_fee=fee,
        )

    def close(self, pos: Position, price: float, reason: str, now: int) -> ClosedTrade:
        trade, cash = realize(pos, price, now, reason, self.s)
        self.state.balance += cash
        return trade

    def sync(self, prices: dict[str, float], now: int) -> list[ClosedTrade]:
        closed = []
        for symbol, pos in list(self.state.positions.items()):
            price = prices.get(symbol)
            if price is None:
                continue
            old_stop = pos.stop_loss
            if update_stops(pos, price, self.s):
                self.store.log_decision(
                    "stop_moved", f"stop {old_stop:.6g} -> {pos.stop_loss:.6g}",
                    symbol, pos.side, {"price": price}, ts=now)
            hit = check_exit(pos, high=price, low=price, open_=price)
            if hit:
                fill, reason = hit
                closed.append(self.close(pos, fill, reason, now))
        return closed

    def equity(self, prices: dict[str, float]) -> float:
        unrealized = sum(p.unrealized(prices.get(sym, p.entry_price))
                         for sym, p in self.state.positions.items())
        return self.state.balance + unrealized


class BinanceBroker:
    """Real orders on Binance USDⓈ-M futures (one-way mode).

    Every position is protected on the exchange itself by a STOP_MARKET and a
    TAKE_PROFIT_MARKET (closePosition=true), so it stays protected even if the
    bot crashes. If the stop cannot be placed, the position is closed at once.
    """

    name = "binance"

    def __init__(self, settings, state: BotState, store: StateStore,
                 client: BinanceFuturesClient, symbols: dict[str, SymbolInfo]):
        self.s = settings
        self.state = state
        self.store = store
        self.client = client
        self.symbols = symbols
        self._exchange_positions: dict[str, dict] = {}

    # ----------------------------------------------------------- preflight
    def preflight(self) -> list[str]:
        """Safety checks before trading with real keys. Returns warnings."""
        warnings = []
        self.client.sync_time()
        mode = self.client.position_mode()
        if mode.get("dualSidePosition"):
            raise RuntimeError("Hedge mode is enabled: switch the futures account to "
                               "one-way mode (the bot assumes one position per symbol).")
        if self.s.mode == "live":
            perms = self.client.api_restrictions()
            if perms.get("enableWithdrawals"):
                raise RuntimeError("This API key can WITHDRAW funds. Create a key without "
                                   "withdrawal permission (and ideally IP-restricted).")
            if not perms.get("enableFutures", False):
                raise RuntimeError("This API key has no futures permission.")
            if not perms.get("ipRestrict", False):
                warnings.append("API key is not IP-restricted: consider restricting it.")
        account = self.client.account()
        wallet = float(account.get("totalWalletBalance", 0))
        if wallet < self.s.initial_capital:
            warnings.append(f"wallet balance {wallet:.2f} USDT is below INITIAL_CAPITAL "
                            f"{self.s.initial_capital:.2f}: equity is capped to the wallet.")
        known = set(self.state.positions)
        for row in self.client.position_risk():
            if float(row.get("positionAmt", 0)) != 0 and row["symbol"] not in known:
                warnings.append(f"{row['symbol']}: open position not managed by the bot "
                                "(left untouched).")
        return warnings

    # ---------------------------------------------------------------- orders
    def _client_id(self, tag: str) -> str:
        return f"stb-{tag}-{uuid.uuid4().hex[:16]}"

    def _place_stop(self, pos: Position, stop: float) -> int:
        info = self.symbols[pos.symbol]
        side = "SELL" if pos.side == LONG else "BUY"
        resp = self.client.conditional_close_order(
            pos.symbol, side, "STOP_MARKET", info.round_price(stop), self._client_id("sl"))
        return int(resp["algoId"])

    def _place_target(self, pos: Position) -> int:
        info = self.symbols[pos.symbol]
        side = "SELL" if pos.side == LONG else "BUY"
        resp = self.client.conditional_close_order(
            pos.symbol, side, "TAKE_PROFIT_MARKET", info.round_price(pos.take_profit),
            self._client_id("tp"))
        return int(resp["algoId"])

    def open(self, plan: TradePlan, price: float, now: int) -> Position:
        info = self.symbols[plan.symbol]
        self.client.set_margin_type(plan.symbol, self.s.margin_type)
        self.client.set_leverage(plan.symbol, plan.leverage)
        side = "BUY" if plan.side == LONG else "SELL"
        order = self.client.market_order(plan.symbol, side, info.round_qty(plan.qty),
                                         client_order_id=self._client_id("in"))
        qty = float(order.get("executedQty") or plan.qty)
        fill = _avg_price(order) or price
        d = direction(plan.side)
        stop_dist = abs(plan.price - plan.stop_loss)
        target_dist = abs(plan.take_profit - plan.price)
        pos = Position(
            symbol=plan.symbol, side=plan.side, qty=qty, entry_price=fill,
            leverage=plan.leverage,
            stop_loss=float(info.round_price(fill - d * stop_dist)),
            take_profit=float(info.round_price(fill + d * target_dist)),
            initial_stop=float(info.round_price(fill - d * stop_dist)),
            atr=plan.atr, margin=qty * fill / plan.leverage,
            risk_amount=plan.risk_amount, opened_at=now,
            entry_fee=qty * fill * self.s.fee_rate,
        )
        try:
            pos.sl_algo_id = self._place_stop(pos, pos.stop_loss)
        except BinanceAPIError as exc:
            log.error("%s: stop-loss rejected (%s), closing immediately", plan.symbol, exc)
            self._market_close(pos)
            raise RuntimeError(f"stop-loss could not be placed: {exc}") from exc
        try:
            pos.tp_algo_id = self._place_target(pos)
        except BinanceAPIError as exc:  # the stop protects us; the bot can live without TP
            log.warning("%s: take-profit rejected (%s)", plan.symbol, exc)
        return pos

    def _market_close(self, pos: Position) -> dict:
        info = self.symbols[pos.symbol]
        side = "SELL" if pos.side == LONG else "BUY"
        return self.client.market_order(pos.symbol, side, info.round_qty(pos.qty),
                                         reduce_only=True,
                                         client_order_id=self._client_id("out"))

    def _cancel_protection(self, symbol: str) -> None:
        try:
            self.client.cancel_all_algo_orders(symbol)
        except BinanceAPIError as exc:
            log.warning("%s: could not cancel algo orders: %s", symbol, exc)

    def close(self, pos: Position, price: float, reason: str, now: int) -> ClosedTrade:
        self._cancel_protection(pos.symbol)
        order = self._market_close(pos)
        exit_px = _avg_price(order) or price
        return self._finalize(pos, exit_px, reason, now)

    def _finalize(self, pos: Position, exit_px: float, reason: str, now: int) -> ClosedTrade:
        """Build the trade from the exchange's own income records."""
        pnl = fees = 0.0
        try:
            for row in self.client.income(pos.opened_at - 1000, symbol=pos.symbol):
                amount = float(row.get("income", 0))
                if row.get("incomeType") in PNL_INCOME_TYPES:
                    pnl += amount
                if row.get("incomeType") == "COMMISSION":
                    fees += -amount
        except BinanceAPIError as exc:
            log.warning("%s: income history unavailable (%s), estimating PnL", pos.symbol, exc)
            gross = pos.dir * (exit_px - pos.entry_price) * pos.qty
            fees = (pos.entry_price + exit_px) * pos.qty * self.s.fee_rate
            pnl = gross - fees
        return ClosedTrade(
            symbol=pos.symbol, side=pos.side, qty=pos.qty, entry_price=pos.entry_price,
            exit_price=exit_px, leverage=pos.leverage, margin=pos.margin,
            opened_at=pos.opened_at, closed_at=now, pnl=pnl, fees=fees,
            r_multiple=pnl / pos.risk_amount if pos.risk_amount else 0.0, reason=reason,
        )

    def _last_fill(self, pos: Position) -> float | None:
        try:
            trades = self.client.user_trades(pos.symbol, pos.opened_at)
        except BinanceAPIError:
            return None
        closing = "SELL" if pos.side == LONG else "BUY"
        fills = [t for t in trades if t.get("side") == closing]
        return float(fills[-1]["price"]) if fills else None

    # ------------------------------------------------------------------ sync
    def sync(self, prices: dict[str, float], now: int) -> list[ClosedTrade]:
        rows = self.client.position_risk()
        self._exchange_positions = {r["symbol"]: r for r in rows
                                    if float(r.get("positionAmt", 0)) != 0}
        closed = []
        for symbol, pos in list(self.state.positions.items()):
            if symbol not in self._exchange_positions:
                # Closed on the exchange: stop, target, liquidation or by hand.
                self._cancel_protection(symbol)
                exit_px = self._last_fill(pos) or prices.get(symbol, pos.entry_price)
                closed.append(self._finalize(pos, exit_px, _guess_reason(pos, exit_px), now))
                continue
            mark = float(self._exchange_positions[symbol].get("markPrice") or
                         prices.get(symbol, pos.entry_price))
            old_stop, old_breakeven = pos.stop_loss, pos.breakeven_done
            if update_stops(pos, mark, self.s):
                info = self.symbols[symbol]
                if info.round_price(pos.stop_loss) == info.round_price(old_stop):
                    continue
                trade = self._move_stop(pos, old_stop, old_breakeven, mark, now)
                if trade:
                    closed.append(trade)
        self._sync_income()
        return closed

    def _move_stop(self, pos: Position, old_stop: float, old_breakeven: bool, mark: float,
                   now: int) -> ClosedTrade | None:
        """Replace the exchange stop. Never leaves the position unprotected.

        Binance accepts only one closePosition STOP_MARKET per direction
        (error -4130), so the old stop is cancelled before the new one is placed.
        """
        symbol = pos.symbol
        if pos.sl_algo_id:
            try:
                self.client.cancel_algo_order(pos.sl_algo_id)
            except BinanceAPIError as exc:
                log.warning("%s: could not cancel the stop (%s), keeping it", symbol, exc)
                pos.stop_loss, pos.breakeven_done = old_stop, old_breakeven
                return None
        try:
            pos.sl_algo_id = self._place_stop(pos, pos.stop_loss)
        except BinanceAPIError as exc:
            if exc.code == -2021:  # would trigger immediately: price is already past it
                return self.close(pos, mark, "trailing_stop", now)
            log.warning("%s: could not move stop (%s), restoring the old one", symbol, exc)
            pos.stop_loss, pos.breakeven_done = old_stop, old_breakeven
            try:
                pos.sl_algo_id = self._place_stop(pos, old_stop)
            except BinanceAPIError as exc2:
                log.error("%s: old stop rejected too (%s): closing at market", symbol, exc2)
                return self.close(pos, mark, "stop_loss", now)
            return None
        self.store.log_decision("stop_moved", f"stop {old_stop:.6g} -> {pos.stop_loss:.6g}",
                                symbol, pos.side, {"mark": mark}, ts=now)
        return None

    def _sync_income(self) -> None:
        """Accumulate realized PnL, fees and funding since the bot started."""
        st = self.state
        try:
            rows = self.client.income(st.income_cursor)
        except BinanceAPIError as exc:
            log.warning("income sync failed: %s", exc)
            return
        seen = set(st.income_seen)
        for row in rows:
            tran = int(row.get("tranId", 0))
            if tran in seen or row.get("incomeType") not in PNL_INCOME_TYPES:
                continue
            seen.add(tran)
            st.income_total += float(row.get("income", 0))
            st.income_cursor = max(st.income_cursor, int(row.get("time", st.income_cursor)))
        st.income_seen = sorted(seen)[-500:]
        st.balance = st.initial_capital + st.income_total

    def equity(self, prices: dict[str, float]) -> float:
        unrealized = 0.0
        for symbol in self.state.positions:
            row = self._exchange_positions.get(symbol)
            if row:
                unrealized += float(row.get("unRealizedProfit", 0))
        equity = self.state.balance + unrealized
        try:
            account = self.client.account()
            # Never pretend to own more than the futures wallet actually holds.
            equity = min(equity, float(account.get("totalMarginBalance", equity)))
        except BinanceAPIError as exc:
            log.warning("account unavailable: %s", exc)
        return equity


def _avg_price(order: dict) -> float | None:
    try:
        avg = float(order.get("avgPrice") or 0)
        if avg > 0:
            return avg
        qty = float(order.get("executedQty") or 0)
        quote = float(order.get("cumQuote") or 0)
        return quote / qty if qty > 0 and quote > 0 else None
    except (TypeError, ValueError):
        return None


def _guess_reason(pos: Position, exit_px: float) -> str:
    to_stop = abs(exit_px - pos.stop_loss)
    to_target = abs(exit_px - pos.take_profit)
    if to_target < to_stop:
        return "take_profit"
    return "stop_loss" if pos.stop_loss == pos.initial_stop else "trailing_stop"

