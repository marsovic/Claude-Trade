from decimal import Decimal
from urllib.parse import parse_qs, urlparse

import pytest

from bot.binance_client import (
    BinanceAPIError,
    BinanceFuturesClient,
    fmt_decimal,
    parse_symbol_info,
    sign,
)
from tests.conftest import symbol_info


def test_signature_matches_binance_documentation_example():
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    query = ("symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1"
             "&recvWindow=5000&timestamp=1499827319559")
    assert sign(secret, query) == \
        "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"


def test_rounding_and_formatting():
    info = symbol_info("AAAUSDT", step="0.001", tick="0.01")
    assert info.round_qty(1.23456) == Decimal("1.234")
    assert info.round_qty(0.0009) == 0
    assert info.round_price(101.235) == Decimal("101.24")
    assert fmt_decimal(Decimal("1.2300")) == "1.23"
    assert fmt_decimal(Decimal("1E+2")) == "100"
    assert fmt_decimal(0.00000123) == "0.00000123"


def test_parse_symbol_info_and_sentiment_asset():
    raw = {"symbol": "1000PEPEUSDT", "baseAsset": "1000PEPE", "quoteAsset": "USDT",
           "filters": [
               {"filterType": "PRICE_FILTER", "tickSize": "0.0000001"},
               {"filterType": "LOT_SIZE", "stepSize": "1", "minQty": "1"},
               {"filterType": "MARKET_LOT_SIZE", "stepSize": "1", "minQty": "1"},
               {"filterType": "MIN_NOTIONAL", "notional": "5"},
           ]}
    info = parse_symbol_info(raw)
    assert info.step_size == Decimal("1") and info.min_notional == Decimal("5")
    assert info.sentiment_asset == "PEPE"
    assert symbol_info("BTCUSDT").sentiment_asset == "BTC"


class Resp:
    def __init__(self, status, payload, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}
        self.text = str(payload)

    def json(self):
        return self._payload


class RecordingSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.headers = {}

    def request(self, method, url, headers=None, timeout=None):
        self.requests.append((method, url, headers))
        return self.responses.pop(0)


def test_signed_request_carries_key_timestamp_and_signature():
    session = RecordingSession([Resp(200, {"algoId": 42})])
    client = BinanceFuturesClient("https://example.test", "KEY", "SECRET", session=session)
    out = client.conditional_close_order("BTCUSDT", "SELL", "STOP_MARKET", Decimal("50000.10"))
    assert out == {"algoId": 42}
    method, url, headers = session.requests[0]
    assert method == "POST" and url.startswith("https://example.test/fapi/v1/algoOrder?")
    assert headers["X-MBX-APIKEY"] == "KEY"
    query = urlparse(url).query
    params = parse_qs(query)
    assert params["algoType"] == ["CONDITIONAL"]
    assert params["closePosition"] == ["true"]
    assert params["triggerPrice"] == ["50000.1"]
    assert params["workingType"] == ["MARK_PRICE"]
    unsigned, signature = query.rsplit("&signature=", 1)
    assert signature == sign("SECRET", unsigned)


def test_api_errors_are_raised_with_code():
    session = RecordingSession([Resp(400, {"code": -2019, "msg": "Margin is insufficient."})])
    client = BinanceFuturesClient("https://example.test", "K", "S", session=session)
    with pytest.raises(BinanceAPIError) as err:
        client.set_leverage("BTCUSDT", 5)
    assert err.value.code == -2019


def test_margin_type_already_set_is_not_an_error():
    session = RecordingSession([Resp(400, {"code": -4046, "msg": "No need to change"})])
    client = BinanceFuturesClient("https://example.test", "K", "S", session=session)
    assert client.set_margin_type("BTCUSDT", "ISOLATED") is None


def test_signed_endpoint_without_keys_is_refused():
    client = BinanceFuturesClient("https://example.test", session=RecordingSession([]))
    with pytest.raises(BinanceAPIError):
        client.account()


def test_symbol_infos_keeps_trading_perpetuals_only():
    payload = {"symbols": [
        {"symbol": "BTCUSDT", "contractType": "PERPETUAL", "status": "TRADING",
         "baseAsset": "BTC", "quoteAsset": "USDT", "filters": []},
        {"symbol": "BTCUSDT_261225", "contractType": "CURRENT_QUARTER", "status": "TRADING",
         "baseAsset": "BTC", "quoteAsset": "USDT", "filters": []},
        {"symbol": "OLDUSDT", "contractType": "PERPETUAL", "status": "SETTLING",
         "baseAsset": "OLD", "quoteAsset": "USDT", "filters": []},
    ]}
    client = BinanceFuturesClient("https://example.test",
                                  session=RecordingSession([Resp(200, payload)]))
    assert list(client.symbol_infos()) == ["BTCUSDT"]
