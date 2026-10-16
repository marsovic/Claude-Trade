"""Alpaca Trading API v2 (US stocks, commission-free, fractional).

* LIVE https://api.alpaca.markets/v2, PAPER https://paper-api.alpaca.markets/v2
  (paper accounts are free for anyone and are a realistic broker simulator)
* headers APCA-API-KEY-ID / APCA-API-SECRET-KEY
* ``client_order_id`` makes order submission idempotent
* fractional orders are market orders with time_in_force=day, so stops are
  watched by the bot (no resting stop) - see ``resting_stops``
* the account is in USD: values are converted to EUR with the ECB rate

Eligibility of French residents for a LIVE account was not confirmed by an
official Alpaca source in Oct 2026: check the sign-up form before relying on it.
"""

from __future__ import annotations

import logging

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
URLS = {"live": "https://api.alpaca.markets/v2", "paper": "https://paper-api.alpaca.markets/v2"}
STATUS = {"new": "pending", "accepted": "pending", "pending_new": "pending",
          "accepted_for_bidding": "pending", "held": "pending", "calculated": "pending",
          "partially_filled": "partially_filled", "filled": "filled",
          "done_for_day": "pending", "canceled": "cancelled", "expired": "cancelled",
          "replaced": "cancelled", "pending_cancel": "pending", "pending_replace": "pending",
          "rejected": "rejected", "suspended": "rejected", "stopped": "pending"}


class AlpacaBroker(LiveBroker):
    name = "alpaca"
    resting_stops = False
    fractional = True

    def __init__(self, api_key: str, api_secret: str, environment: str = "paper",
                 session: requests.Session | None = None, fx=None):
        self.base = URLS[environment]
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": api_key,
                                     "APCA-API-SECRET-KEY": api_secret})
        self.fx = fx or (lambda: 1.10)

    def _request(self, method: str, path: str, **kwargs):
        posting = method == "POST" and path == "/orders"
        try:
            resp = self.session.request(method, self.base + path, timeout=20, **kwargs)
        except requests.RequestException as exc:
            if posting:
                raise BrokerOutcomeUnknown(f"Alpaca network error on POST /orders: {exc}") \
                    from exc
            raise BrokerError(f"Alpaca network error on {method} {path}: {exc}") from exc
        if resp.status_code == 404 and method == "DELETE":
            return None
        if posting and (resp.status_code >= 500 or (
                resp.status_code == 422 and "client_order_id" in resp.text)):
            # 5xx, or a resend of an order Alpaca already has: look it up by its id.
            raise BrokerOutcomeUnknown(f"Alpaca HTTP {resp.status_code} on POST /orders")
        if resp.status_code >= 400:
            raise BrokerError(f"Alpaca HTTP {resp.status_code} on {path}: {resp.text[:300]}")
        return resp.json() if resp.content else None

    def resolve(self, symbol: str) -> str | None:
        if symbol.startswith("ETP:"):
            return None  # US broker: no EU-listed leveraged ETPs
        return symbol

    def account(self) -> AccountInfo:
        acct = self._request("GET", "/account")
        rate = self.fx()
        positions = {}
        for row in self._request("GET", "/positions") or []:
            positions[row["symbol"]] = BrokerPosition(
                ticker=row["symbol"], qty=float(row.get("qty_available") or row["qty"]),
                avg_price=float(row["avg_entry_price"]),
                price=float(row.get("current_price") or 0), currency="USD")
        return AccountInfo(equity_eur=float(acct["equity"]) / rate,
                           cash_eur=float(acct["cash"]) / rate, positions=positions)

    def submit(self, order: OrderRequest) -> str:
        whole = float(order.qty).is_integer()
        body = {"symbol": order.broker_ticker or order.symbol,
                "qty": f"{order.qty:.6f}".rstrip("0").rstrip("."),
                "side": order.side.lower(), "time_in_force": "day",
                "client_order_id": order.id}
        if order.kind == "LIMIT":
            # Fractional limit orders are accepted with time_in_force=day.
            body.update(type="limit", limit_price=f"{order.limit_price:.2f}")
        elif order.kind == "STOP" and whole:
            body.update(type="stop", stop_price=f"{order.stop_price:.2f}", time_in_force="gtc")
        elif order.kind == "STOP":
            raise BrokerError("fractional stops cannot rest at Alpaca: watched by the bot")
        else:
            body["type"] = "market"
        resp = self._request("POST", "/orders", json=body)
        return str(resp["id"])

    def cancel(self, broker_id: str) -> None:
        if broker_id:
            self._request("DELETE", f"/orders/{broker_id}")

    def order_update(self, order: OrderRequest) -> OrderUpdate:
        row = self._request("GET", f"/orders/{order.broker_id}")
        status = STATUS.get(row.get("status", ""), "unknown")
        filled_at = row.get("filled_at")
        return OrderUpdate(status, filled_qty=float(row.get("filled_qty") or 0),
                           avg_price=float(row.get("filled_avg_price") or 0),
                           filled_at=int(pd.Timestamp(filled_at).timestamp() * 1000)
                           if filled_at else 0)

    def find(self, order: OrderRequest, submitted_at: int) -> str | None:
        try:
            row = self._request("GET", "/orders:by_client_order_id",
                                params={"client_order_id": order.id})
        except BrokerError:
            return None
        return str(row["id"]) if row and row.get("id") else None

    def preflight(self) -> list[str]:
        acct = self._request("GET", "/account")
        warnings = []
        if acct.get("status") != "ACTIVE":
            warnings.append(f"Alpaca account status is {acct.get('status')}")
        warnings.append("Alpaca accounts are in USD: the bot converts with the ECB rate; "
                        "EU-listed ETPs are not available (ETP sleeve disabled).")
        return warnings
