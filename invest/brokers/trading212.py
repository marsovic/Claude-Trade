"""Trading 212 Public API v0 (beta) - Invest account.

Facts from the official OpenAPI spec (docs.trading212.com/_bundle/api.json):

* LIVE https://live.trading212.com/api/v0, DEMO https://demo.trading212.com/api/v0
* HTTP Basic auth with API key / API secret (keys are per environment)
* quantity > 0 buys, quantity < 0 sells; orders by quantity only
* order POSTs are NOT idempotent: never retried automatically
* no market data in the API: prices come from the data provider
* rate limits per account: market 50/min, limit/stop 1 per 2 s, pending
  orders 1 per 5 s, positions 1/s, summary 1 per 5 s, history 6/min,
  instruments 1 per 50 s

POLICY WARNING: Trading 212's own agent-skills repository states "As required
by MiFID II, you are prohibited from using this tool for algorithmic trading".
Get Trading 212's written agreement before running this bot on a live account.
"""

from __future__ import annotations

import base64
import logging
import math
import time

import pandas as pd
import requests

from invest.brokers.base import (
    AccountInfo,
    BrokerError,
    BrokerOutcomeUnknown,
    BrokerPosition,
    LiveBroker,
    OrderRequest,
    OrderUpdate,
)

log = logging.getLogger(__name__)

URLS = {"live": "https://live.trading212.com/api/v0",
        "paper": "https://demo.trading212.com/api/v0"}
MIN_INTERVAL = {  # seconds between calls, per endpoint family
    "summary": 5.2, "positions": 1.1, "orders": 5.2, "market": 1.3, "limit": 2.1,
    "stop": 2.1, "cancel": 1.3, "order": 1.1, "history": 10.5, "instruments": 51.0,
}
QTY_DECIMALS = 4  # the API rejects more precision than this on many instruments


def floor_qty(qty: float, decimals: int = QTY_DECIMALS) -> float:
    factor = 10 ** decimals
    return math.floor(abs(qty) * factor + 1e-9) / factor


class Trading212Broker(LiveBroker):
    name = "trading212"
    resting_stops = True
    fractional = True

    def __init__(self, api_key: str, api_secret: str, environment: str = "paper",
                 etp_ticker: str = "", session: requests.Session | None = None,
                 sleep=time.sleep, clock=time.monotonic, fx=None):
        self.base = URLS[environment]
        token = base64.b64encode(f"{api_key}:{api_secret}".encode()).decode()
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Basic {token}",
                                     "Content-Type": "application/json"})
        self.sleep, self.clock = sleep, clock
        self.etp_ticker = etp_ticker
        self.fx = fx or (lambda: 1.10)  # USD per EUR, used to convert USD fees
        self._last: dict[str, float] = {}
        self._instruments: dict[str, dict] | None = None
        self._instruments_at = 0.0
        self._pending: dict[str, dict] = {}
        self._history: list[dict] | None = None

    # ------------------------------------------------------------- transport
    def _throttle(self, family: str) -> None:
        wait = self._last.get(family, -1e18) + MIN_INTERVAL[family] - self.clock()
        if wait > 0:
            self.sleep(wait)
        self._last[family] = self.clock()

    def _request(self, method: str, path: str, family: str, json: dict | None = None,
                 params: dict | None = None):
        idempotent = method == "GET"
        for attempt in range(3 if idempotent else 1):
            self._throttle(family)
            try:
                resp = self.session.request(method, self.base + path, json=json,
                                            params=params, timeout=20)
            except requests.RequestException as exc:
                if idempotent and attempt < 2:
                    self.sleep(2 * (attempt + 1))
                    continue
                if method == "POST":
                    raise BrokerOutcomeUnknown(f"Trading 212 network error on {path}: {exc}") \
                        from exc
                raise BrokerError(f"Trading 212 network error on {method} {path}: {exc}") \
                    from exc
            if resp.status_code == 429 and idempotent and attempt < 2:
                reset = resp.headers.get("x-ratelimit-reset")
                self.sleep(max(1.0, float(reset) - time.time()) if reset else 5.0)
                continue
            if resp.status_code == 404 and method == "DELETE":
                return None
            if method == "POST" and (resp.status_code >= 500 or resp.status_code == 408):
                # Order POSTs are not idempotent: the order may exist anyway.
                raise BrokerOutcomeUnknown(f"Trading 212 HTTP {resp.status_code} on {path}")
            if resp.status_code >= 400:
                raise BrokerError(f"Trading 212 HTTP {resp.status_code} on {path}: "
                                  f"{resp.text[:300]}")
            return resp.json() if resp.content else None
        raise BrokerError(f"Trading 212: rate limited on {path}")

    # ------------------------------------------------------------- metadata
    def instruments(self) -> dict[str, dict]:
        if self._instruments is None or time.time() - self._instruments_at > 86400:
            rows = self._request("GET", "/equity/metadata/instruments", "instruments")
            self._instruments = {r["ticker"]: r for r in rows}
            self._instruments_at = time.time()
        return self._instruments

    def resolve(self, symbol: str) -> str | None:
        if symbol.startswith("ETP:"):
            return self.etp_ticker or None
        inst = self.instruments()
        guess = f"{symbol.replace('.', '_')}_US_EQ"
        if guess in inst:
            return guess
        for ticker, row in inst.items():  # fallback: same short name, USD stock
            if (row.get("shortName") == symbol and row.get("currencyCode") == "USD"
                    and row.get("type") == "STOCK"):
                return ticker
        return None

    # -------------------------------------------------------------- account
    def account(self) -> AccountInfo:
        summary = self._request("GET", "/equity/account/summary", "summary")
        cash = summary.get("cash", {})
        invest = summary.get("investments", {})
        cash_eur = float(cash.get("availableToTrade", 0) or 0)
        equity = (cash_eur + float(cash.get("reservedForOrders", 0) or 0)
                  + float(cash.get("inPies", 0) or 0)
                  + float(invest.get("currentValue", 0) or 0))
        if summary.get("currency") not in (None, "EUR"):
            log.warning("Trading 212 account currency is %s, the bot assumes EUR",
                        summary.get("currency"))
        positions = {}
        for row in self._request("GET", "/equity/positions", "positions") or []:
            inst = row.get("instrument", {})
            ticker = inst.get("ticker", "")
            available = row.get("quantityAvailableForTrading")
            positions[ticker] = BrokerPosition(
                ticker=ticker,
                # Total quantity: shares under our own pending stop are still ours.
                qty=float(row.get("quantity", 0) or 0),
                avg_price=float(row.get("averagePricePaid", 0) or 0),
                price=float(row.get("currentPrice", 0) or 0),
                currency=inst.get("currency", "USD"),
                available=None if available is None else float(available))
        return AccountInfo(equity_eur=equity, cash_eur=cash_eur, positions=positions)

    # --------------------------------------------------------------- orders
    def submit(self, order: OrderRequest) -> str:
        if not order.broker_ticker:
            raise BrokerError(f"{order.symbol}: no Trading 212 ticker")
        qty = floor_qty(order.qty)
        if qty <= 0:
            raise BrokerError(f"{order.symbol}: quantity {order.qty} rounds to zero")
        signed = qty if order.side == "BUY" else -qty
        body = {"ticker": order.broker_ticker, "quantity": signed}
        if order.kind == "MARKET":
            resp = self._request("POST", "/equity/orders/market", "market",
                                 json={**body, "extendedHours": False})
        elif order.kind == "LIMIT":
            # Entries live for one session only (DAY), stops until cancelled.
            resp = self._request("POST", "/equity/orders/limit", "limit", json={
                **body, "limitPrice": round(float(order.limit_price), 4),
                "timeValidity": "DAY"})
        elif order.kind == "STOP":
            resp = self._request("POST", "/equity/orders/stop", "stop", json={
                **body, "stopPrice": round(float(order.stop_price), 4),
                "timeValidity": "GOOD_TILL_CANCEL"})
        else:
            raise BrokerError(f"unsupported order kind {order.kind}")
        return str(resp["id"])

    def cancel(self, broker_id: str) -> None:
        if broker_id:
            self._request("DELETE", f"/equity/orders/{broker_id}", "cancel")

    def begin_sync(self) -> None:
        rows = self._request("GET", "/equity/orders", "orders") or []
        self._pending = {str(r["id"]): r for r in rows}
        self._history = None

    def _history_items(self) -> list[dict]:
        if self._history is None:
            items, path, params = [], "/equity/history/orders", {"limit": 50}
            for _ in range(2):  # 100 most recent orders is plenty for a 3-slot bot
                page = self._request("GET", path, "history", params=params) or {}
                items.extend(page.get("items", []))
                nxt = page.get("nextPagePath")
                if not nxt:
                    break
                path = nxt.split("/api/v0", 1)[-1].split("?", 1)[0]
                params = dict(p.split("=", 1) for p in nxt.split("?", 1)[1].split("&")) \
                    if "?" in nxt else {}
            self._history = items
        return self._history

    def order_update(self, order: OrderRequest) -> OrderUpdate:
        bid = order.broker_id
        if bid in self._pending:
            row = self._pending[bid]
            filled = abs(float(row.get("filledQuantity", 0) or 0))
            value = abs(float(row.get("filledValue", 0) or 0))
            return OrderUpdate("partially_filled" if filled else "pending", filled_qty=filled,
                               avg_price=value / filled if filled and value else 0.0)
        for item in self._history_items():
            o = item.get("order", {})
            if str(o.get("id")) != bid:
                continue
            status = o.get("status", "")
            fill = item.get("fill") or {}
            if status == "FILLED" or fill:
                fees = 0.0
                for tax in (fill.get("walletImpact") or {}).get("taxes", []) or []:
                    amount = abs(float(tax.get("quantity", 0) or 0))
                    fees += amount if tax.get("currency", "EUR") == "EUR" else amount / self.fx()
                when = fill.get("filledAt") or o.get("createdAt")
                return OrderUpdate(
                    "filled",
                    filled_qty=abs(float(fill.get("quantity") or o.get("filledQuantity") or 0)),
                    avg_price=float(fill.get("price") or 0), fees_eur=fees,
                    filled_at=int(pd.Timestamp(when).timestamp() * 1000) if when else 0)
            if status in ("CANCELLED", "REJECTED"):
                return OrderUpdate("cancelled" if status == "CANCELLED" else "rejected")
        return OrderUpdate("unknown")

    def find(self, order: OrderRequest, submitted_at: int) -> str | None:
        """Match a timed-out submission by ticker, signed quantity and type."""
        qty = floor_qty(order.qty)
        signed = qty if order.side == "BUY" else -qty
        kind = {"MARKET": "MARKET", "LIMIT": "LIMIT", "STOP": "STOP"}[order.kind]

        def same(o: dict) -> bool:
            created = o.get("createdAt")
            recent = not created or pd.Timestamp(created).timestamp() * 1000 >= \
                submitted_at - 120_000
            return (o.get("ticker") == order.broker_ticker and o.get("type") == kind
                    and abs(float(o.get("quantity") or 0) - signed) < 1e-9 and recent)

        for oid, o in self._pending.items():
            if same(o):
                return oid
        for item in self._history_items():
            o = item.get("order", {})
            if same(o):
                return str(o.get("id"))
        return None

    def preflight(self) -> list[str]:
        warnings = []
        summary = self._request("GET", "/equity/account/summary", "summary")
        if summary.get("currency") != "EUR":
            warnings.append(f"account currency is {summary.get('currency')}, expected EUR")
        warnings.append("Trading 212 forbids algorithmic trading with its own AI tool "
                        "(MiFID II wording): make sure you have its written OK.")
        return warnings
