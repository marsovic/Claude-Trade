"""Broker adapters against in-memory fakes (no network)."""

import base64
import json
from types import SimpleNamespace

import pytest
import requests

from invest.brokers.alpaca import AlpacaBroker
from invest.brokers.base import BrokerError, BrokerOutcomeUnknown, OrderRequest
from invest.brokers.ibkr import IbkrBroker
from invest.brokers.trading212 import Trading212Broker, floor_qty


class Resp:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.content = b"" if payload is None else json.dumps(payload).encode()
        self.text = self.content.decode()

    def json(self):
        return self._payload


class Session:
    def __init__(self, routes):
        self.routes = routes  # (method, path suffix) -> Resp | list[Resp] | Exception
        self.calls = []
        self.headers = {}

    def request(self, method, url, json=None, params=None, timeout=None):
        self.calls.append((method, url, json, params))
        for (m, suffix), resp in self.routes.items():
            if m == method and url.split("?")[0].endswith(suffix):
                if isinstance(resp, list):
                    resp = resp.pop(0) if len(resp) > 1 else resp[0]
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return Resp(404, {"error": "no route"})


def t212(routes, **kw):
    return Trading212Broker("KEY", "SECRET", "paper", session=Session(routes),
                            sleep=lambda _: None, clock=lambda: 0.0, **kw)


# ------------------------------------------------------------------ Trading 212
def test_t212_basic_auth_and_demo_url():
    b = t212({})
    token = base64.b64decode(b.session.headers["Authorization"].split()[1]).decode()
    assert token == "KEY:SECRET"
    assert b.base == "https://demo.trading212.com/api/v0"


def test_t212_order_bodies():
    b = t212({("POST", "/equity/orders/market"): Resp(200, {"id": 1}),
              ("POST", "/equity/orders/limit"): Resp(200, {"id": 2}),
              ("POST", "/equity/orders/stop"): Resp(200, {"id": 3})})
    sell = OrderRequest("NVDA", "SELL", 0.123456789, broker_ticker="NVDA_US_EQ")
    assert b.submit(sell) == "1"
    assert b.session.calls[-1][2] == {"ticker": "NVDA_US_EQ", "quantity": -0.1234,
                                      "extendedHours": False}
    buy = OrderRequest("NVDA", "BUY", 0.5, kind="LIMIT", limit_price=123.456789,
                       broker_ticker="NVDA_US_EQ")
    assert b.submit(buy) == "2"
    assert b.session.calls[-1][2] == {"ticker": "NVDA_US_EQ", "quantity": 0.5,
                                      "limitPrice": 123.4568,
                                      "timeValidity": "DAY"}  # entries: one session only
    stop = OrderRequest("NVDA", "SELL", 0.5, kind="STOP", stop_price=100.0,
                        broker_ticker="NVDA_US_EQ")
    b.submit(stop)
    assert b.session.calls[-1][2]["quantity"] == -0.5
    assert b.session.calls[-1][2]["stopPrice"] == 100.0


def test_t212_orders_are_never_retried():
    b = t212({("POST", "/equity/orders/market"): requests.ConnectionError("reset")})
    with pytest.raises(BrokerOutcomeUnknown):
        b.submit(OrderRequest("AMD", "BUY", 1, broker_ticker="AMD_US_EQ"))
    assert len(b.session.calls) == 1
    b5 = t212({("POST", "/equity/orders/limit"): Resp(504, {"error": "gateway"})})
    with pytest.raises(BrokerOutcomeUnknown):  # a 5xx answer may hide an accepted order
        b5.submit(OrderRequest("AMD", "BUY", 1, kind="LIMIT", limit_price=1.0,
                               broker_ticker="AMD_US_EQ"))


def test_t212_get_retries_on_429_and_parses_account():
    summary = {"currency": "EUR", "cash": {"availableToTrade": 40.0, "reservedForOrders": 5.0,
                                           "inPies": 0.0},
               "investments": {"currentValue": 60.0}}
    positions = [{"instrument": {"ticker": "AMD_US_EQ", "currency": "USD"}, "quantity": 0.3,
                  "quantityAvailableForTrading": 0.3, "averagePricePaid": 150.0,
                  "currentPrice": 160.0}]
    b = t212({("GET", "/equity/account/summary"): [Resp(429, {}), Resp(200, summary)],
              ("GET", "/equity/positions"): Resp(200, positions)})
    acct = b.account()
    assert acct.equity_eur == pytest.approx(105.0) and acct.cash_eur == 40.0
    assert acct.positions["AMD_US_EQ"].price == 160.0


def test_t212_resolve_and_reconcile():
    instruments = [{"ticker": "AMD_US_EQ", "shortName": "AMD", "currencyCode": "USD",
                    "type": "STOCK"},
                   {"ticker": "BRK_B_US_EQ", "shortName": "BRK.B", "currencyCode": "USD",
                    "type": "STOCK"}]
    history = {"items": [
        {"order": {"id": 11, "status": "FILLED", "filledQuantity": 0.3},
         "fill": {"price": 151.0, "quantity": 0.3, "filledAt": "2026-10-14T13:30:05Z",
                  "walletImpact": {"taxes": [
                      {"name": "CURRENCY_CONVERSION_FEE", "quantity": -0.07, "currency": "EUR"}]}}},
        {"order": {"id": 12, "status": "CANCELLED"}, "fill": None},
    ], "nextPagePath": None}
    b = t212({("GET", "/equity/metadata/instruments"): Resp(200, instruments),
              ("GET", "/equity/orders"): Resp(200, [{"id": 13, "filledQuantity": 0}]),
              ("GET", "/equity/history/orders"): Resp(200, history)})
    assert b.resolve("AMD") == "AMD_US_EQ"
    assert b.resolve("BRK.B") == "BRK_B_US_EQ"
    assert b.resolve("NOPE") is None
    b.begin_sync()
    filled = b.order_update(OrderRequest("AMD", "BUY", 0.3, broker_id="11"))
    assert filled.status == "filled" and filled.avg_price == 151.0
    assert filled.fees_eur == pytest.approx(0.07) and filled.filled_at > 0
    assert b.order_update(OrderRequest("AMD", "BUY", 1, broker_id="12")).status == "cancelled"
    assert b.order_update(OrderRequest("AMD", "BUY", 1, broker_id="13")).status == "pending"
    assert b.order_update(OrderRequest("AMD", "BUY", 1, broker_id="99")).status == "unknown"


def test_floor_qty():
    assert floor_qty(0.99999) == 0.9999
    assert floor_qty(1.0) == 1.0


# ---------------------------------------------------------------------- Alpaca
def test_alpaca_order_and_account():
    session = Session({("POST", "/orders"): Resp(200, {"id": "abc"}),
                       ("GET", "/orders/abc"): Resp(200, {
                           "status": "filled", "filled_qty": "0.25", "filled_avg_price": "200",
                           "filled_at": "2026-10-14T13:31:00Z"}),
                       ("GET", "/account"): Resp(200, {"equity": "110", "cash": "55",
                                                       "status": "ACTIVE"}),
                       ("GET", "/positions"): Resp(200, [{
                           "symbol": "AMD", "qty": "0.25", "qty_available": "0.25",
                           "avg_entry_price": "200", "current_price": "210"}])})
    b = AlpacaBroker("k", "s", "paper", session=session, fx=lambda: 1.1)
    assert session.headers["APCA-API-KEY-ID"] == "k"
    frac = OrderRequest("AMD", "BUY", 0.25, kind="LIMIT", limit_price=201.0)
    assert b.submit(frac) == "abc"
    body = session.calls[-1][2]
    assert body["type"] == "limit" and body["qty"] == "0.25"  # gap cap kept
    assert body["limit_price"] == "201.00" and body["time_in_force"] == "day"
    assert body["client_order_id"] == frac.id  # idempotent
    with pytest.raises(BrokerError, match="fractional stops"):
        b.submit(OrderRequest("AMD", "SELL", 0.25, kind="STOP", stop_price=190.0))
    b.submit(OrderRequest("AMD", "SELL", 2, kind="STOP", stop_price=190.0))
    assert session.calls[-1][2]["type"] == "stop"
    assert session.calls[-1][2]["time_in_force"] == "gtc"
    upd = b.order_update(OrderRequest("AMD", "BUY", 0.25, broker_id="abc"))
    assert upd.status == "filled" and upd.avg_price == 200.0
    acct = b.account()
    assert acct.equity_eur == pytest.approx(100.0) and acct.positions["AMD"].qty == 0.25
    assert b.resolve("ETP:QQQx3") is None and not b.resting_stops


# ------------------------------------------------------------------------ IBKR
class FakeIB:
    def __init__(self):
        self.placed = []
        self._trades = []

    def isConnected(self):
        return True

    def connect(self, *a, **k):
        raise AssertionError("should not reconnect")

    def reqMarketDataType(self, t):
        self.data_type = t

    def qualifyContracts(self, c):
        return [c]

    def placeOrder(self, contract, order):
        order.orderId = len(self.placed) + 1
        trade = SimpleNamespace(order=order, contract=contract, fills=[],
                                orderStatus=SimpleNamespace(status="Submitted", filled=0,
                                                            avgFillPrice=0.0))
        self.placed.append(trade)
        self._trades.append(trade)
        return trade

    def trades(self):
        return self._trades

    def openTrades(self):
        return [t for t in self._trades if t.orderStatus.status == "Submitted"]

    def cancelOrder(self, order):
        order.cancelled = True

    def reqExecutions(self):
        return []

    def sleep(self, _):
        pass

    def accountSummary(self):
        return [SimpleNamespace(tag="NetLiquidation", currency="EUR", value="101.5"),
                SimpleNamespace(tag="TotalCashValue", currency="EUR", value="20")]

    def portfolio(self):
        return [SimpleNamespace(contract=SimpleNamespace(symbol="AMD", currency="USD"),
                                position=1.0, averageCost=150.0, marketPrice=155.0)]


class FakeLib:
    @staticmethod
    def Stock(symbol, exchange, currency):
        return SimpleNamespace(symbol=symbol, exchange=exchange, currency=currency)

    class _Order(SimpleNamespace):
        pass

    @classmethod
    def MarketOrder(cls, action, qty, **kw):
        return cls._Order(orderType="MKT", action=action, totalQuantity=qty, **kw)

    @classmethod
    def LimitOrder(cls, action, qty, price, **kw):
        return cls._Order(orderType="LMT", action=action, totalQuantity=qty, lmtPrice=price,
                          **kw)

    @classmethod
    def StopOrder(cls, action, qty, price, **kw):
        return cls._Order(orderType="STP", action=action, totalQuantity=qty, auxPrice=price,
                          **kw)


def ibkr(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "ib_async", FakeLib)
    return IbkrBroker(ib=FakeIB(), fx=lambda: 1.1)


def test_ibkr_whole_shares_and_reconcile(monkeypatch):
    b = ibkr(monkeypatch)
    assert b.ib.data_type == 3  # delayed data, no subscription needed
    with pytest.raises(BrokerError, match="whole shares"):
        b.submit(OrderRequest("AMD", "BUY", 0.5))
    order = OrderRequest("AMD", "BUY", 2.7, kind="LIMIT", limit_price=150.123)
    b.submit(order)
    trade = b.ib.placed[-1]
    assert trade.order.totalQuantity == 2 and trade.order.lmtPrice == 150.12
    assert trade.order.orderRef == order.id and trade.order.tif == "DAY"
    stop = OrderRequest("AMD", "SELL", 2, kind="STOP", stop_price=140.0)
    b.submit(stop)
    assert b.ib.placed[-1].order.tif == "GTC"
    assert b.order_update(order).status == "pending"
    trade.orderStatus.status, trade.orderStatus.filled = "Filled", 2
    trade.orderStatus.avgFillPrice = 150.0
    trade.fills = [SimpleNamespace(commissionReport=SimpleNamespace(commission=1.1),
                                   time=SimpleNamespace(timestamp=lambda: 1_700_000_000))]
    upd = b.order_update(order)
    assert upd.status == "filled" and upd.fees_eur == pytest.approx(1.0)
    acct = b.account()
    assert acct.equity_eur == 101.5 and acct.positions["AMD"].price == 155.0
