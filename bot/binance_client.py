"""Minimal Binance USDⓈ-M futures REST client (requests + HMAC-SHA256).

Only the endpoints the bot needs are implemented. Conditional orders
(STOP_MARKET / TAKE_PROFIT_MARKET) go through the Algo Order API
(``/fapi/v1/algoOrder``), which replaced ``/fapi/v1/order`` for those order
types in December 2025.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)


class BinanceAPIError(RuntimeError):
    def __init__(self, status: int, code: int | None, message: str):
        super().__init__(f"HTTP {status} / code {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SymbolInfo:
    symbol: str
    base_asset: str
    quote_asset: str
    tick_size: Decimal
    step_size: Decimal
    min_qty: Decimal
    min_notional: Decimal

    def round_qty(self, qty: float) -> Decimal:
        """Round a quantity down to the lot step (never up: never over-size)."""
        q = Decimal(str(qty))
        return (q / self.step_size).to_integral_value(rounding=ROUND_DOWN) * self.step_size

    def round_price(self, price: float) -> Decimal:
        p = Decimal(str(price))
        return (p / self.tick_size).to_integral_value(rounding=ROUND_HALF_UP) * self.tick_size

    @property
    def sentiment_asset(self) -> str:
        """Base asset as news/social sources know it (1000PEPE -> PEPE)."""
        asset = self.base_asset
        for prefix in ("1000000", "1000", "1M"):
            if asset.startswith(prefix) and len(asset) > len(prefix):
                return asset[len(prefix):]
        return asset


def fmt_decimal(value: Decimal | float) -> str:
    """Format a number the way Binance expects (no exponent, no trailing zeros)."""
    d = Decimal(str(value)) if not isinstance(value, Decimal) else value
    text = format(d.normalize(), "f")
    return text if text not in ("-0", "") else "0"


def sign(secret: str, query: str) -> str:
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def parse_symbol_info(raw: dict) -> SymbolInfo:
    filters = {f["filterType"]: f for f in raw.get("filters", [])}
    price_f = filters.get("PRICE_FILTER", {})
    lot_f = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE", {})
    if Decimal(lot_f.get("stepSize", "0") or "0") == 0:
        lot_f = filters.get("LOT_SIZE", {})
    notional_f = filters.get("MIN_NOTIONAL", {})
    return SymbolInfo(
        symbol=raw["symbol"],
        base_asset=raw.get("baseAsset", raw["symbol"].removesuffix("USDT")),
        quote_asset=raw.get("quoteAsset", "USDT"),
        tick_size=Decimal(price_f.get("tickSize", "0.0001")),
        step_size=Decimal(lot_f.get("stepSize", "0.001")),
        min_qty=Decimal(lot_f.get("minQty", "0")),
        min_notional=Decimal(notional_f.get("notional", notional_f.get("minNotional", "5"))),
    )


class BinanceFuturesClient:
    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        api_key: str = "",
        api_secret: str = "",
        spot_base_url: str = "https://api.binance.com",
        timeout: float = 10.0,
        recv_window: int = 5000,
        session: requests.Session | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.spot_base_url = spot_base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret
        self.timeout = timeout
        self.recv_window = recv_window
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "survival-trading-bot/2"})
        self._time_offset_ms = 0

    # ------------------------------------------------------------------ core
    def _request(self, method: str, path: str, params: dict | None = None,
                 signed: bool = False, base_url: str | None = None, retries: int = 3):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        url = (base_url or self.base_url) + path
        headers = {}
        if signed:
            if not (self.api_key and self.api_secret):
                raise BinanceAPIError(0, None, "API key/secret required for signed endpoint")
            headers["X-MBX-APIKEY"] = self.api_key
        attempt = 0
        while True:
            attempt += 1
            query_params = dict(params)
            if signed:
                query_params["recvWindow"] = self.recv_window
                query_params["timestamp"] = int(time.time() * 1000) + self._time_offset_ms
                query = urlencode(query_params)
                query += "&signature=" + sign(self.api_secret, query)
            else:
                query = urlencode(query_params)
            full_url = f"{url}?{query}" if query else url
            try:
                resp = self.session.request(method, full_url, headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                # Retrying a POST could duplicate an order: only GETs are retried.
                if method == "GET" and attempt < retries:
                    time.sleep(2 ** attempt)
                    continue
                raise BinanceAPIError(0, None, f"network error: {exc}") from exc
            if resp.status_code in (418, 429):
                wait = int(resp.headers.get("Retry-After", "5"))
                log.warning("Binance rate limit hit (%s), sleeping %ss", resp.status_code, wait)
                if attempt < retries:
                    time.sleep(min(wait, 60))
                    continue
            if resp.status_code >= 400:
                try:
                    body = resp.json()
                    raise BinanceAPIError(resp.status_code, body.get("code"), body.get("msg", ""))
                except ValueError:
                    raise BinanceAPIError(resp.status_code, None, resp.text[:200]) from None
            return resp.json()

    def sync_time(self) -> int:
        server = self._request("GET", "/fapi/v1/time")["serverTime"]
        self._time_offset_ms = int(server - time.time() * 1000)
        return self._time_offset_ms

    # ---------------------------------------------------------------- public
    def ping(self) -> dict:
        return self._request("GET", "/fapi/v1/ping")

    def exchange_info(self) -> dict:
        return self._request("GET", "/fapi/v1/exchangeInfo")

    def ticker_24h(self) -> list[dict]:
        return self._request("GET", "/fapi/v1/ticker/24hr")

    def ticker_prices(self) -> dict[str, float]:
        rows = self._request("GET", "/fapi/v1/ticker/price")
        return {row["symbol"]: float(row["price"]) for row in rows}

    def klines(self, symbol: str, interval: str, limit: int = 300,
               start_time: int | None = None, end_time: int | None = None) -> list[list]:
        return self._request("GET", "/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "limit": limit,
            "startTime": start_time, "endTime": end_time,
        })

    # ---------------------------------------------------------------- signed
    def account(self) -> dict:
        return self._request("GET", "/fapi/v3/account", signed=True)

    def position_risk(self, symbol: str | None = None) -> list[dict]:
        return self._request("GET", "/fapi/v3/positionRisk", {"symbol": symbol}, signed=True)

    def position_mode(self) -> dict:
        return self._request("GET", "/fapi/v1/positionSide/dual", signed=True)

    def set_leverage(self, symbol: str, leverage: int) -> dict:
        return self._request("POST", "/fapi/v1/leverage",
                             {"symbol": symbol, "leverage": leverage}, signed=True)

    def set_margin_type(self, symbol: str, margin_type: str) -> dict | None:
        try:
            return self._request("POST", "/fapi/v1/marginType",
                                 {"symbol": symbol, "marginType": margin_type}, signed=True)
        except BinanceAPIError as exc:
            if exc.code == -4046:  # "No need to change margin type."
                return None
            raise

    def market_order(self, symbol: str, side: str, quantity: Decimal,
                     reduce_only: bool = False, client_order_id: str | None = None) -> dict:
        return self._request("POST", "/fapi/v1/order", {
            "symbol": symbol, "side": side, "type": "MARKET",
            "quantity": fmt_decimal(quantity),
            "reduceOnly": "true" if reduce_only else None,
            "newClientOrderId": client_order_id,
            "newOrderRespType": "RESULT",
        }, signed=True)

    def conditional_close_order(self, symbol: str, side: str, order_type: str,
                                trigger_price: Decimal, client_algo_id: str | None = None) -> dict:
        """STOP_MARKET / TAKE_PROFIT_MARKET that closes the whole position."""
        return self._request("POST", "/fapi/v1/algoOrder", {
            "algoType": "CONDITIONAL",
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "triggerPrice": fmt_decimal(trigger_price),
            "closePosition": "true",
            "workingType": "MARK_PRICE",
            "priceProtect": "TRUE",
            "clientAlgoId": client_algo_id,
        }, signed=True)

    def cancel_algo_order(self, algo_id: int) -> dict:
        return self._request("DELETE", "/fapi/v1/algoOrder", {"algoId": algo_id}, signed=True)

    def cancel_all_algo_orders(self, symbol: str) -> dict:
        return self._request("DELETE", "/fapi/v1/algoOpenOrders", {"symbol": symbol}, signed=True)

    def open_algo_orders(self, symbol: str | None = None) -> list[dict]:
        return self._request("GET", "/fapi/v1/openAlgoOrders", {"symbol": symbol}, signed=True)

    def income(self, start_time: int, symbol: str | None = None,
               income_type: str | None = None, limit: int = 1000) -> list[dict]:
        return self._request("GET", "/fapi/v1/income", {
            "symbol": symbol, "incomeType": income_type,
            "startTime": start_time, "limit": limit,
        }, signed=True)

    def user_trades(self, symbol: str, start_time: int, limit: int = 500) -> list[dict]:
        return self._request("GET", "/fapi/v1/userTrades",
                             {"symbol": symbol, "startTime": start_time, "limit": limit},
                             signed=True)

    def api_restrictions(self) -> dict:
        """Key permissions (spot SAPI): used to refuse keys that can withdraw."""
        return self._request("GET", "/sapi/v1/account/apiRestrictions", signed=True,
                             base_url=self.spot_base_url)

    # --------------------------------------------------------------- helpers
    def symbol_infos(self) -> dict[str, SymbolInfo]:
        info = self.exchange_info()
        out = {}
        for raw in info.get("symbols", []):
            if raw.get("contractType") == "PERPETUAL" and raw.get("status") == "TRADING":
                out[raw["symbol"]] = parse_symbol_info(raw)
        return out
