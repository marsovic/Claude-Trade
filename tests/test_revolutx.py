"""Revolut X venue: signed client, spot broker and long-only trading loop."""

import base64
import json
import os
from decimal import Decimal
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from bot.binance_client import SymbolInfo
from bot.config import ConfigError, Settings
from bot.revolutx_broker import PENDING_TIMEOUT_MS, RevolutXBroker
from bot.revolutx_client import (
    RevolutXClient,
    RevolutXDuplicateOrder,
    RevolutXError,
    RevolutXOutcomeUnknown,
    resolve_credentials,
    signing_message,
)
from bot.risk import TradePlan
from bot.state import BotState, StateStore
from bot.trading_bot import TradingBot
from tests.conftest import Clock, FakeMarketClient, FakeSentiment, signal_frame

NOW = 1_800_000_000_000


# ================================================================== fixtures
@pytest.fixture
def key_file(tmp_path):
    key = Ed25519PrivateKey.generate()
    path = tmp_path / "private.pem"
    path.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                                       serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption()))
    os.chmod(path, 0o600)
    return path, key


@pytest.fixture
def revx_settings(tmp_path, key_file, monkeypatch):
    monkeypatch.setenv("REVOLUTX_CONFIG_DIR", str(tmp_path / "no-cli-config"))
    s = Settings.for_venue("revolutx", state_dir=str(tmp_path / "state"),
                           revx_api_key="k" * 64, revx_private_key_path=str(key_file[0]))
    s.validate()
    return s


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self.ok = status < 400
        self._body = body
        self.content = b"" if body is None else json.dumps(body).encode()
        self.text = self.content.decode()
        self.headers = headers or {}

    def json(self):
        if self._body is None:
            raise ValueError("no body")
        return self._body


class FakeSession:
    """Records requests; answers from a queue of responses or exceptions."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []
        self.headers = {}

    def request(self, method, url, data=None, headers=None, timeout=None):
        data = data.decode() if isinstance(data, bytes) else data
        self.calls.append(SimpleNamespace(method=method, url=url, data=data,
                                          headers=dict(headers or {})))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def make_client(key, *answers) -> tuple[RevolutXClient, FakeSession]:
    session = FakeSession(*answers)
    return RevolutXClient("k" * 64, private_key=key, session=session,
                          sleep=lambda _s: None), session


# ==================================================================== client
def test_requests_are_signed_like_the_official_client(key_file):
    _, key = key_file
    client, session = make_client(key, Resp(body={"data": {"venue_order_id": "v1",
                                                           "client_order_id": "c1",
                                                           "state": "new"}}))
    client.place_order("BTC-EUR", "buy", base_size=Decimal("0.00012000"), price=70000.5,
                       time_in_force="ioc", client_order_id="c1")
    call = session.calls[0]
    assert call.url == "https://revx.revolut.com/api/1.0/orders"
    assert call.data == ('{"client_order_id":"c1","symbol":"BTC-EUR","side":"buy",'
                         '"order_configuration":{"limit":{"price":"70000.5",'
                         '"base_size":"0.00012","time_in_force":"ioc"}}}')
    h = call.headers
    assert h["X-Revx-API-Key"] == "k" * 64
    message = signing_message(h["X-Revx-Timestamp"], "POST", "/api/1.0/orders", "", call.data)
    key.public_key().verify(base64.b64decode(h["X-Revx-Signature"]), message)  # no raise


def test_query_string_is_sorted_encoded_and_signed(key_file):
    _, key = key_file
    client, session = make_client(key, Resp(body={"data": [], "metadata": {}}))
    client.active_orders(["BTC/EUR", "ETH-EUR"])
    call = session.calls[0]
    query = urlsplit(call.url).query
    assert query == "symbols=BTC-EUR%2CETH-EUR"
    message = signing_message(call.headers["X-Revx-Timestamp"], "GET",
                              "/api/1.0/orders/active", query)
    key.public_key().verify(base64.b64decode(call.headers["X-Revx-Signature"]), message)


def test_clock_drift_is_corrected_once(key_file):
    _, key = key_file
    server = NOW
    client, session = make_client(
        key, Resp(409, {"message": "Request timestamp is in the future", "timestamp": server}),
        Resp(body=[{"currency": "EUR", "available": "50", "reserved": "0", "total": "50"}]))
    assert client.balances()["EUR"]["total"] == 50.0
    retried = int(session.calls[1].headers["X-Revx-Timestamp"])
    assert abs(retried - server) < 5_000


def test_order_post_is_retried_with_the_same_client_order_id(key_file):
    _, key = key_file
    client, session = make_client(
        key, Resp(503, {"message": "Something went wrong!"}),
        Resp(400, {"message": "Order with client order id 'c1' has already been placed"}))
    with pytest.raises(RevolutXDuplicateOrder):
        client.place_order("BTC-EUR", "buy", base_size=0.001, client_order_id="c1")
    assert session.calls[0].data == session.calls[1].data  # same id: no second order


def test_lost_order_reply_is_outcome_unknown_but_refused_connection_is_not(key_file):
    _, key = key_file
    timeout = requests.exceptions.ReadTimeout("read timed out")
    client, _ = make_client(key, timeout, timeout, timeout)
    with pytest.raises(RevolutXOutcomeUnknown):
        client.place_order("BTC-EUR", "buy", base_size=0.001, client_order_id="c1")
    refused = requests.exceptions.ConnectTimeout("connect timed out")
    client, _ = make_client(key, refused, refused, refused)
    with pytest.raises(RevolutXError) as info:
        client.place_order("BTC-EUR", "buy", base_size=0.001, client_order_id="c2")
    assert not isinstance(info.value, RevolutXOutcomeUnknown)


def test_pairs_tickers_and_chunked_candles(key_file):
    _, key = key_file
    pairs = {"BTC/EUR": {"base": "BTC", "quote": "EUR", "base_step": "0.00000001",
                         "quote_step": "0.01", "min_order_size": "0.00000001",
                         "min_order_size_quote": "0.1", "status": "active"},
             "OLD/EUR": {"base": "OLD", "quote": "EUR", "base_step": "1", "quote_step": "1",
                         "status": "inactive"}}
    tickers = {"data": [{"symbol": "BTC/EUR", "bid": "72000", "ask": "72010", "mid": "72005",
                         "last_price": "72003", "volume_24h": "100", "quote_volume_24h": ""}]}
    client, session = make_client(key, Resp(body=pairs), Resp(body=tickers), Resp(body=tickers))
    infos = client.symbol_infos()
    assert list(infos) == ["BTC-EUR"]
    assert infos["BTC-EUR"].tick_size == Decimal("0.01")
    assert infos["BTC-EUR"].min_notional == Decimal("0.1")
    assert client.ticker_prices() == {"BTC-EUR": 72005.0}
    assert client.last_quotes["BTC-EUR"] == {"bid": 72000.0, "ask": 72010.0}
    row = client.ticker_24h()[0]
    assert float(row["quoteVolume"]) == pytest.approx(100 * 72003)

    step = 15 * 60_000
    answers = []
    end = NOW // step * step
    start = end - 249 * step
    for chunk_start in range(start, end + 1, 100 * step):
        chunk = [{"start": t, "open": "1", "high": "2", "low": "0.5", "close": "1.5",
                  "volume": "3"} for t in range(chunk_start, min(end, chunk_start + 99 * step)
                                                + 1, step)]
        answers.append(Resp(body={"data": chunk}))
    session.answers = answers
    rows = client.klines("BTC-EUR", "15m", limit=250, end_time=NOW)
    assert len(rows) == 250 and rows[-1][0] == end and rows[0][0] == start
    assert rows[0][6] == start + step - 1
    candle_calls = [c for c in session.calls if "/candles/" in c.url]
    assert len(candle_calls) == 3
    for c in candle_calls:
        q = dict(p.split("=") for p in urlsplit(c.url).query.split("&"))
        assert (int(q["until"]) - int(q["since"])) // step < 100


def test_credentials_come_from_the_revx_cli_folder(tmp_path, monkeypatch, key_file):
    folder = tmp_path / "revx"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"api_key": "a" * 64}))
    (folder / "private.pem").write_bytes(key_file[0].read_bytes())
    monkeypatch.setenv("REVOLUTX_CONFIG_DIR", str(folder))
    assert resolve_credentials("", "") == ("a" * 64, str(folder / "private.pem"))
    assert resolve_credentials("b" * 64, "/x.pem") == ("b" * 64, "/x.pem")


def test_world_readable_private_key_is_refused(key_file):
    path, _ = key_file
    os.chmod(path, 0o644)
    with pytest.raises(RevolutXError, match="chmod 600"):
        RevolutXClient("k" * 64, str(path))


# ==================================================================== config
def test_revolutx_settings_defaults(revx_settings):
    s = revx_settings
    assert s.quote_asset == "EUR" and s.fee_rate == 0.0009 and s.max_leverage == 1
    assert s.long_only and s.currency_symbol == "€"
    public = s.public_dict()
    assert public["revx_api_key"] is True and "k" * 64 not in json.dumps(public)
    assert public["venue_label"].startswith("Revolut X")


def test_revolutx_env_forces_spot_rules(tmp_path, key_file, monkeypatch):
    for k in ("MODE", "QUOTE_ASSET", "FEE_RATE", "INTERVAL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("REVOLUTX_CONFIG_DIR", str(tmp_path / "none"))
    monkeypatch.setenv("EXCHANGE", "revolutx")
    monkeypatch.setenv("MAX_LEVERAGE", "5")
    monkeypatch.setenv("REVX_API_KEY", "k" * 64)
    monkeypatch.setenv("REVX_PRIVATE_KEY_PATH", str(key_file[0]))
    s = Settings.from_env(None)
    assert (s.exchange, s.quote_asset, s.max_leverage, s.fee_rate) == ("revolutx", "EUR", 1,
                                                                       0.0009)
    with pytest.raises(ConfigError, match="no testnet"):
        Settings.from_env(None, mode="testnet")
    with pytest.raises(ConfigError, match="INTERVAL"):
        Settings.from_env(None, interval="2h")
    monkeypatch.setenv("REVX_PRIVATE_KEY_PATH", str(tmp_path / "missing.pem"))
    with pytest.raises(ConfigError, match="no file"):
        Settings.from_env(None)
    monkeypatch.delenv("REVX_API_KEY")
    with pytest.raises(ConfigError, match="REVX_API_KEY"):
        Settings.from_env(None)


# ============================================================ fake exchange
def btc_info():
    return SymbolInfo(symbol="BTC-EUR", base_asset="BTC", quote_asset="EUR",
                      tick_size=Decimal("0.01"), step_size=Decimal("0.00000001"),
                      min_qty=Decimal("0.00000001"), min_notional=Decimal("0.1"))


class FakeRevX:
    """In-memory Revolut X: balances, IOC limit buys, market sells, resting limits."""

    def __init__(self, eur=50.0, btc=0.0, bid=70_000.0, ask=70_010.0):
        self.bal = {"EUR": {"available": eur, "reserved": 0.0, "total": eur},
                    "BTC": {"available": btc, "reserved": 0.0, "total": btc}}
        self.last_quotes = {"BTC-EUR": {"bid": bid, "ask": ask}}
        self.orders: dict[str, dict] = {}
        self.placed: list[dict] = []
        self.fee_in_base = False
        self.fill_buys = True
        self.fail_next: Exception | None = None
        self.record_anyway = False
        self.n = 0

    # helpers
    def _move(self, ccy, avail=0.0, reserved=0.0):
        b = self.bal.setdefault(ccy, {"available": 0.0, "reserved": 0.0, "total": 0.0})
        b["available"] += avail
        b["reserved"] += reserved
        b["total"] = b["available"] + b["reserved"]

    def set_bid(self, bid):
        self.last_quotes["BTC-EUR"] = {"bid": bid, "ask": bid + 10}

    # API
    def balances(self):
        return json.loads(json.dumps(self.bal))

    def place_order(self, symbol, side, *, base_size, price=None, time_in_force=None,
                    client_order_id=None):
        if any(o["client_order_id"] == client_order_id for o in self.orders.values()):
            raise RevolutXDuplicateOrder(400, "already been placed")
        err, self.fail_next = self.fail_next, None
        if err and not self.record_anyway:
            raise err
        self.n += 1
        oid = f"o{self.n}"
        qty = float(base_size)
        o = {"id": oid, "client_order_id": client_order_id, "symbol": "BTC/EUR",
             "side": side, "type": "limit" if price is not None else "market",
             "quantity": str(qty), "filled_quantity": "0", "status": "new",
             "price": str(price) if price is not None else None}
        q = self.last_quotes["BTC-EUR"]
        if side == "buy":
            fill_px = q["ask"]
            if self.fill_buys and fill_px <= float(price):
                fee = qty * 0.0009 if self.fee_in_base else qty * fill_px * 0.0009
                self._move("EUR", avail=-(qty * fill_px + (0 if self.fee_in_base else fee)))
                self._move("BTC", avail=qty - (fee if self.fee_in_base else 0))
                o.update(filled_quantity=str(qty), average_fill_price=str(fill_px),
                         status="filled", total_fee=str(fee),
                         fee_currency="BTC" if self.fee_in_base else "EUR")
            else:
                o["status"] = "cancelled"
        elif price is None:  # market sell
            qty = min(qty, self.bal["BTC"]["available"])
            self._move("BTC", avail=-qty)
            self._move("EUR", avail=qty * q["bid"] * (1 - 0.0009))
            o.update(filled_quantity=str(qty), average_fill_price=str(q["bid"]),
                     status="filled", total_fee=str(qty * q["bid"] * 0.0009),
                     fee_currency="EUR")
        else:  # resting limit sell (take-profit)
            self._move("BTC", avail=-qty, reserved=qty)
        self.orders[oid] = o
        self.placed.append(o)
        if err:
            raise err
        return {"venue_order_id": oid, "client_order_id": client_order_id, "state": o["status"]}

    def fill_take_profit(self, oid):
        o = self.orders[oid]
        qty, px = float(o["quantity"]), float(o["price"])
        self._move("BTC", reserved=-qty)
        self._move("EUR", avail=qty * px)
        o.update(filled_quantity=str(qty), average_fill_price=str(px), status="filled",
                 total_fee="0", fee_currency="EUR")

    def order(self, oid):
        if oid not in self.orders:
            raise RevolutXError(404, "not found")
        return dict(self.orders[oid])

    def cancel_order(self, oid):
        o = self.orders.get(oid)
        if not o or o["status"] not in ("new", "partially_filled"):
            return False
        left = float(o["quantity"]) - float(o["filled_quantity"])
        self._move("BTC", avail=left, reserved=-left)
        o["status"] = "cancelled"
        return True

    def active_orders(self, symbols=None):
        return [dict(o) for o in self.orders.values() if o["status"] in ("new",
                                                                         "partially_filled")]

    def find_order(self, cid, symbol, since_ms):
        for o in self.orders.values():
            if o["client_order_id"] == cid:
                return dict(o)
        return None


def make_broker(settings, ex: FakeRevX):
    state = BotState.fresh("live", settings.initial_capital, "revolutx")
    store = StateStore(settings.state_dir)
    broker = RevolutXBroker(settings, state, store, ex, {"BTC-EUR": btc_info()},
                            sleep=lambda _s: None)
    return broker, state


def plan(price=70_000.0, qty=0.00025):
    return TradePlan(symbol="BTC-EUR", side="LONG", qty=qty, price=price,
                     stop_loss=price - 700, take_profit=price + 1400, leverage=1, atr=466.0,
                     notional=qty * price, margin=qty * price, risk_amount=0.2)


# ==================================================================== broker
def test_entry_is_an_ioc_buy_and_the_take_profit_rests_on_the_exchange(revx_settings):
    ex = FakeRevX()
    broker, state = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    buy, tp = ex.placed
    assert buy["side"] == "buy" and float(buy["price"]) == pytest.approx(70_010 * 1.003, 0.01)
    assert pos.entry_price == 70_010.0 and pos.leverage == 1
    assert pos.cost == pytest.approx(0.00025 * 70_010 * 1.0009)
    assert pos.stop_loss == pytest.approx(70_010 - 700)
    assert tp["side"] == "sell" and float(tp["price"]) == pytest.approx(70_010 + 1400)
    assert pos.tp_order_id == tp["id"] and state.pending_entries == {}


def test_fee_taken_in_coins_shrinks_the_position(revx_settings):
    ex = FakeRevX()
    ex.fee_in_base = True
    broker, _ = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    assert pos.qty == pytest.approx(0.00025 * (1 - 0.0009))


def test_unfilled_ioc_leaves_nothing_behind(revx_settings):
    ex = FakeRevX()
    ex.fill_buys = False
    broker, state = make_broker(revx_settings, ex)
    with pytest.raises(RuntimeError, match="not filled"):
        broker.open(plan(), 70_000.0, NOW)
    assert state.pending_entries == {} and len(ex.placed) == 1


def test_lost_entry_reply_is_reconciled_never_resent(revx_settings):
    ex = FakeRevX()
    ex.fail_next = RevolutXOutcomeUnknown(0, "read timed out")
    ex.record_anyway = True  # the order did reach the exchange
    broker, state = make_broker(revx_settings, ex)
    with pytest.raises(RuntimeError, match="unknown"):
        broker.open(plan(), 70_000.0, NOW)
    assert len(state.pending_entries) == 1 and state.positions == {}
    assert StateStore(revx_settings.state_dir).load().pending_entries  # saved before sending
    broker.sync({"BTC-EUR": 70_005.0}, NOW + 20_000)
    assert state.pending_entries == {}
    pos = state.positions["BTC-EUR"]
    assert pos.qty == pytest.approx(0.00025) and pos.tp_order_id
    assert sum(o["side"] == "buy" for o in ex.placed) == 1


def test_entry_status_failure_after_placing_keeps_it_pending(revx_settings):
    ex = FakeRevX()
    broker, state = make_broker(revx_settings, ex)
    real_order = ex.order
    ex.order = lambda oid: (_ for _ in ()).throw(RevolutXError(0, "network error"))
    with pytest.raises(RuntimeError, match="reconcile"):
        broker.open(plan(), 70_000.0, NOW)
    (pending,) = state.pending_entries.values()
    assert pending["order_id"] == ex.placed[0]["id"]  # the bought coins are not lost
    ex.order = real_order
    broker.sync({}, NOW + 20_000)
    assert "BTC-EUR" in state.positions and state.pending_entries == {}


def test_stop_is_still_watched_when_balances_fail(revx_settings):
    ex = FakeRevX()
    broker, state = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    state.positions["BTC-EUR"] = pos
    real_balances = ex.balances
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RevolutXError(503, "Something went wrong!")
        return real_balances()

    ex.balances = flaky
    ex.set_bid(pos.stop_loss - 5)
    trades = broker.sync({}, NOW + 20_000)
    assert trades and trades[0].reason == "stop_loss"


def test_entry_that_never_shows_up_is_dropped_after_the_timeout(revx_settings):
    ex = FakeRevX()
    ex.fail_next = RevolutXOutcomeUnknown(0, "read timed out")
    broker, state = make_broker(revx_settings, ex)
    with pytest.raises(RuntimeError):
        broker.open(plan(), 70_000.0, NOW)
    broker.sync({}, NOW + 60_000)
    assert len(state.pending_entries) == 1
    broker.sync({}, NOW + PENDING_TIMEOUT_MS + 1)
    assert state.pending_entries == {} and state.positions == {}


def test_stop_sells_only_the_bots_coins(revx_settings):
    ex = FakeRevX(btc=1.0)  # the user's own bitcoin, never to be sold
    broker, state = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    state.positions["BTC-EUR"] = pos
    ex.set_bid(pos.stop_loss - 50)
    trades = broker.sync({"BTC-EUR": pos.stop_loss}, NOW + 60_000)
    assert len(trades) == 1 and trades[0].reason == "stop_loss"
    sell = ex.placed[-1]
    assert sell["type"] == "market" and float(sell["quantity"]) == pytest.approx(0.00025)
    assert ex.orders[pos.tp_order_id or ex.placed[1]["id"]]["status"] == "cancelled"
    assert ex.bal["BTC"]["total"] == pytest.approx(1.0)
    assert trades[0].pnl < 0 and abs(trades[0].pnl) < 0.5
    assert state.balance == pytest.approx(revx_settings.initial_capital + trades[0].pnl)


def test_take_profit_filled_on_the_exchange_is_booked(revx_settings):
    ex = FakeRevX()
    broker, state = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    state.positions["BTC-EUR"] = pos
    ex.fill_take_profit(pos.tp_order_id)
    trades = broker.sync({"BTC-EUR": 71_500.0}, NOW + 60_000)
    assert trades[0].reason == "take_profit"
    assert trades[0].pnl == pytest.approx(0.00025 * 71_410 - pos.cost)
    assert trades[0].pnl > 0


def test_trailing_stop_is_watched_by_the_bot(revx_settings):
    ex = FakeRevX()
    broker, state = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    state.positions["BTC-EUR"] = pos
    initial = pos.stop_loss
    ex.set_bid(pos.entry_price + 1.6 * pos.r_distance)
    assert broker.sync({}, NOW + 60_000) == []
    assert pos.stop_loss > initial and pos.breakeven_done
    n = len(ex.placed)
    ex.set_bid(pos.stop_loss - 1)
    trades = broker.sync({}, NOW + 120_000)
    assert trades[0].reason == "trailing_stop" and len(ex.placed) == n + 1


def test_coins_sold_in_the_app_close_the_position_after_two_readings(revx_settings):
    ex = FakeRevX()
    broker, state = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    state.positions["BTC-EUR"] = pos
    ex.cancel_order(pos.tp_order_id)
    ex._move("BTC", avail=-ex.bal["BTC"]["available"])  # sold by hand
    pos.tp_order_id = pos.tp_client_id = None
    assert broker.sync({}, NOW + 20_000) == []
    trades = broker.sync({}, NOW + 40_000)
    assert trades[0].reason == "closed_outside"
    assert not any(o["type"] == "market" for o in ex.placed)


def test_sell_with_lost_reply_is_never_doubled(revx_settings):
    ex = FakeRevX()
    broker, state = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    state.positions["BTC-EUR"] = pos
    ex.fail_next = RevolutXOutcomeUnknown(0, "reset")
    ex.record_anyway = True
    ex.set_bid(pos.stop_loss - 10)
    assert broker.sync({}, NOW + 20_000) == []  # error logged, position kept
    assert "BTC-EUR" in state.positions
    ex.record_anyway = False
    trades = broker.sync({}, NOW + 40_000)
    sells = [o for o in ex.placed if o["type"] == "market"]
    assert len(sells) == 1 and trades[0].reason == "stop_loss"  # found by its id
    assert trades[0].exit_price == pytest.approx(pos.stop_loss - 10)


def test_partial_take_profit_then_lost_sell_keeps_every_fill(revx_settings):
    ex = FakeRevX()
    broker, state = make_broker(revx_settings, ex)
    pos = broker.open(plan(), 70_000.0, NOW)
    state.positions["BTC-EUR"] = pos
    tp = ex.orders[pos.tp_order_id]  # half of the take-profit fills
    ex._move("BTC", reserved=-0.000125)
    ex._move("EUR", avail=0.000125 * float(tp["price"]))
    tp.update(filled_quantity="0.000125", average_fill_price=tp["price"],
              status="partially_filled", total_fee="0", fee_currency="EUR")
    ex.fail_next = RevolutXOutcomeUnknown(0, "reset")
    ex.record_anyway = True
    ex.set_bid(pos.stop_loss - 10)
    assert broker.sync({}, NOW + 20_000) == []
    ex.record_anyway = False
    trades = broker.sync({}, NOW + 40_000)
    assert trades[0].qty == pytest.approx(0.00025)
    expected = 0.000125 * float(tp["price"]) + 0.000125 * (pos.stop_loss - 10) * (1 - 0.0009)
    assert trades[0].pnl == pytest.approx(expected - pos.cost, rel=1e-6)


def test_equity_is_capped_by_the_account_and_confirmed_before_death(revx_settings):
    ex = FakeRevX(eur=500.0)
    broker, state = make_broker(revx_settings, ex)
    broker.sync({}, NOW)
    assert broker.equity({}) == pytest.approx(50.0)  # the bot only owns its 50 EUR
    ex.bal["EUR"] = {"available": 1.0, "reserved": 0.0, "total": 1.0}
    broker.sync({}, NOW + 1)
    assert broker.equity({}) == pytest.approx(50.0)  # first low reading: confirm
    assert broker.equity({}) == pytest.approx(1.0)


def test_preflight_refuses_an_empty_account_and_lists_foreign_coins(revx_settings):
    broker, _ = make_broker(revx_settings, FakeRevX(eur=0.0))
    with pytest.raises(RuntimeError, match="deposit"):
        broker.preflight()
    broker, _ = make_broker(revx_settings, FakeRevX(eur=60.0, btc=0.5))
    warnings = broker.preflight()
    assert any("BTC" in w and "never sold" in w for w in warnings)


# ================================================================ main loop
class FakeEurMarket(FakeMarketClient):
    def symbol_infos(self):
        return {sym: SymbolInfo(symbol=sym, base_asset=sym.split("-")[0], quote_asset="EUR",
                                tick_size=Decimal("0.0001"), step_size=Decimal("0.0001"),
                                min_qty=Decimal("0.0001"), min_notional=Decimal("0.1"))
                for sym in self.frames}


def test_paper_bot_on_revolutx_is_long_only(revx_settings):
    frames = {"AAA-EUR": signal_frame("LONG"), "BBB-EUR": signal_frame("SHORT"),
              "USDC-EUR": signal_frame("LONG")}
    client = FakeEurMarket(frames)
    last = max(int(df["close_time"].iloc[-1]) for df in frames.values())
    bot = TradingBot(revx_settings, client=client, sentiment=FakeSentiment(),
                     clock=Clock((last + 10_000) / 1000))
    bot.setup()
    assert "USDC-EUR" not in bot.state.universe  # stablecoins never traded
    bot.cycle(force_scan=True)
    assert list(bot.state.positions) == ["AAA-EUR"]
    pos = bot.state.positions["AAA-EUR"]
    assert pos.side == "LONG" and pos.leverage == 1
    assert pos.margin <= revx_settings.initial_capital * revx_settings.max_margin_per_trade + 1e-6
    skips = [d for d in bot.store.recent_decisions() if d["kind"] == "skip"]
    assert any(d["symbol"] == "BBB-EUR" and "spot" in d["message"] for d in skips)
    assert bot.state.last_scan["long_only"] is True
    assert bot.state.settings["currency_symbol"] == "€"


def test_state_dir_of_another_venue_is_refused(revx_settings):
    store = StateStore(revx_settings.state_dir)
    store.save(BotState.fresh("paper", 50.0, "binance"))
    bot = TradingBot(revx_settings, client=FakeEurMarket({"AAA-EUR": signal_frame("LONG")}),
                     sentiment=FakeSentiment())
    with pytest.raises(ConfigError, match="different STATE_DIR"):
        bot.setup()
