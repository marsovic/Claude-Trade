"""Minimal Revolut X REST client (spot crypto exchange, Ed25519-signed requests).

Reference: the official Revolut X REST API (https://developer.revolut.com/docs/x-api,
mirrored in github.com/revolut-engineering/revolut-x-api). Base URL
``https://revx.revolut.com/api/1.0``. Every request carries three headers::

    X-Revx-API-Key    the 64-character API key
    X-Revx-Timestamp  Unix time in milliseconds
    X-Revx-Signature  base64(Ed25519 signature of
                      timestamp + METHOD + /api/1.0/path + query + minified JSON body)

Market data (pairs, tickers, candles) needs the key as well: only the last
trades and a 5-level order book are public. The REST API places limit and
market orders only (conditional and TP/SL orders exist in the app, not in the
API), so stop-losses are watched by the bot itself.

The market-data methods mirror :class:`bot.binance_client.BinanceFuturesClient`
(``symbol_infos``, ``ticker_24h``, ``ticker_prices``, ``klines``) so the trading
loop and the backtester work with either venue. Symbols use the path format
``BTC-EUR``; responses use ``BTC/EUR`` and are converted.
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import stat
import sys
import time
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

import requests

from bot.binance_client import SymbolInfo, fmt_decimal

log = logging.getLogger(__name__)

BASE_URL = "https://revx.revolut.com"
API_PREFIX = "/api/1.0"

# Candle intervals accepted by GET /candles (minutes), keyed by the bot's names.
INTERVAL_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}
# The endpoint reference caps a request at 100 candles: stay within it.
CANDLES_PER_REQUEST = 100

TERMINAL_STATES = {"filled", "cancelled", "rejected", "replaced"}
STABLECOINS = {"USDC", "USDT", "EURC", "EURT", "DAI", "PYUSD", "FDUSD", "TUSD", "USDE",
               "RLUSD", "USD1", "USDS", "USDG", "EURQ", "EURR"}


class RevolutXError(RuntimeError):
    def __init__(self, status: int, message: str, error_id: str | None = None,
                 server_time: int | None = None):
        super().__init__(f"Revolut X HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.error_id = error_id
        self.server_time = server_time


class RevolutXOutcomeUnknown(RevolutXError):
    """An order request may have reached the exchange (timeout, reset, 5xx).

    Look the order up by its client_order_id before doing anything else; never
    send it again under a new id.
    """


class RevolutXDuplicateOrder(RevolutXError):
    """The exchange refused a client_order_id it already accepted: the order exists."""


# ================================================================ credentials
def revx_config_dir() -> Path:
    """Folder used by the official ``revx`` CLI and the Revolut X MCP server."""
    custom = os.environ.get("REVOLUTX_CONFIG_DIR")
    if custom:
        return Path(custom).expanduser()
    if sys.platform == "win32" and os.environ.get("APPDATA"):
        return Path(os.environ["APPDATA"]) / "revolut-x"
    return Path.home() / ".config" / "revolut-x"


def resolve_credentials(api_key: str, key_path: str) -> tuple[str, str]:
    """Explicit settings first, then the ``revx`` CLI/MCP config folder."""
    if api_key and key_path:
        return api_key, str(Path(key_path).expanduser())
    folder = revx_config_dir()
    config = {}
    with contextlib.suppress(OSError, ValueError):
        config = json.loads((folder / "config.json").read_text())
    if not isinstance(config, dict):
        config = {}
    api_key = api_key or str(config.get("api_key") or "")
    key_path = key_path or str(config.get("private_key_path") or "")
    if not key_path and (folder / "private.pem").exists():
        key_path = str(folder / "private.pem")
    return api_key, str(Path(key_path).expanduser()) if key_path else ""


def load_private_key(path: str):
    """Load an Ed25519 PEM private key, refusing files other users can read."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    p = Path(path).expanduser()
    if os.name == "posix" and stat.S_IMODE(p.stat().st_mode) & 0o077:
        raise RevolutXError(0, f"{p} is readable by other users: run `chmod 600 {p}`")
    key = serialization.load_pem_private_key(p.read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise RevolutXError(0, f"{p} is not an Ed25519 private key")
    return key


def signing_message(timestamp: str, method: str, path: str, query: str = "",
                    body: str = "") -> bytes:
    return f"{timestamp}{method.upper()}{path}{query}{body}".encode()


def build_query(params: dict | None) -> str:
    """Sorted, URL-encoded query string, exactly as it is signed and sent."""
    items = sorted((k, v) for k, v in (params or {}).items() if v is not None)
    return "&".join(f"{quote(str(k), safe='')}={quote(str(v), safe='')}" for k, v in items)


def to_ms(value) -> int:
    """Epoch milliseconds from an int, a numeric string or an ISO-8601 string."""
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value)
    if text.lstrip("-").isdigit():
        return int(text)
    return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000)


def _num(value, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def path_symbol(symbol: str) -> str:
    return symbol.replace("/", "-").upper()


def _unsent(exc: Exception) -> bool:
    """True when the request certainly never reached the exchange."""
    if isinstance(exc, (requests.exceptions.ConnectTimeout, requests.exceptions.SSLError,
                        requests.exceptions.InvalidURL)):
        return True
    text = repr(exc)
    return any(k in text for k in ("NewConnectionError", "NameResolutionError",
                                   "Failed to resolve", "Connection refused"))


# ===================================================================== client
class RevolutXClient:
    def __init__(self, api_key: str = "", private_key_path: str = "",
                 base_url: str = BASE_URL, timeout: float = 10.0, retries: int = 3,
                 session: requests.Session | None = None, private_key=None,
                 sleep=time.sleep):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._key = private_key
        if self._key is None and private_key_path:
            self._key = load_private_key(private_key_path)
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "survival-trading-bot/2 (revolutx)"})
        self.sleep = sleep
        self._time_offset_ms = 0
        self.last_quotes: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ core
    def _headers(self, method: str, full_path: str, query: str, body: str) -> dict:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key and self._key is not None:
            ts = str(int(time.time() * 1000) + self._time_offset_ms)
            sig = self._key.sign(signing_message(ts, method, full_path, query, body))
            headers.update({"X-Revx-API-Key": self.api_key, "X-Revx-Timestamp": ts,
                            "X-Revx-Signature": base64.b64encode(sig).decode()})
        return headers

    def _request(self, method: str, path: str, params: dict | None = None,
                 body: dict | None = None):
        if not (self.api_key and self._key is not None) and not path.startswith("/public"):
            raise RevolutXError(0, "API key and Ed25519 private key required")
        query = build_query(params)
        body_text = json.dumps(body, separators=(",", ":")) if body is not None else ""
        full_path = API_PREFIX + path
        url = self.base_url + full_path + (f"?{query}" if query else "")
        # Order POSTs carry a client_order_id the exchange refuses twice, so they
        # can be retried like reads; what matters is whether one may have landed.
        maybe_sent = False
        clock_fixed = False
        attempt = 0
        while True:
            attempt += 1
            headers = self._headers(method, full_path, query, body_text)
            try:
                resp = self.session.request(method, url, data=body_text.encode() or None,
                                            headers=headers, timeout=self.timeout)
            except requests.RequestException as exc:
                maybe_sent = maybe_sent or not _unsent(exc)
                if attempt < self.retries:
                    self.sleep(min(2 ** attempt, 10))
                    continue
                cls = RevolutXOutcomeUnknown if maybe_sent and method != "GET" else RevolutXError
                raise cls(0, f"network error: {exc}") from exc
            if resp.status_code == 204 or (resp.ok and not resp.content):
                return {}
            if resp.ok:
                try:
                    return resp.json()
                except ValueError as exc:
                    raise RevolutXOutcomeUnknown(resp.status_code, "unreadable reply") from exc
            err = _error(resp)
            if resp.status_code == 409 and err.server_time and not clock_fixed:
                # "Request timestamp is in the future" / too old: resync and retry.
                self._time_offset_ms = err.server_time - int(time.time() * 1000)
                clock_fixed = True
                continue
            if resp.status_code == 429 and attempt < self.retries:
                self.sleep(min(_num(resp.headers.get("Retry-After"), 5.0), 60.0))
                continue
            if resp.status_code >= 500:
                maybe_sent = True
                if attempt < self.retries:
                    self.sleep(min(2 ** attempt, 10))
                    continue
                if method != "GET":
                    raise RevolutXOutcomeUnknown(err.status, err.message, err.error_id)
            if resp.status_code == 400 and "already been placed" in err.message.lower():
                raise RevolutXDuplicateOrder(err.status, err.message, err.error_id)
            raise err

    # ---------------------------------------------------------------- account
    def balances(self) -> dict[str, dict[str, float]]:
        raw = self._request("GET", "/balances")
        rows = raw.get("data", []) if isinstance(raw, dict) else raw
        return {r["currency"]: {k: _num(r.get(k)) for k in ("available", "reserved", "total")}
                for r in rows or []}

    # ----------------------------------------------------------- market data
    def pairs(self) -> dict[str, dict]:
        raw = self._request("GET", "/configuration/pairs")
        data = raw.get("data", raw) if isinstance(raw, dict) else {}
        return {path_symbol(k): v for k, v in data.items()}

    def symbol_infos(self) -> dict[str, SymbolInfo]:
        out = {}
        for symbol, p in self.pairs().items():
            if p.get("status", "active") != "active":
                continue
            out[symbol] = SymbolInfo(
                symbol=symbol, base_asset=p["base"], quote_asset=p["quote"],
                tick_size=Decimal(str(p["quote_step"])), step_size=Decimal(str(p["base_step"])),
                min_qty=Decimal(str(p.get("min_order_size", "0"))),
                min_notional=Decimal(str(p.get("min_order_size_quote", "0"))))
        return out

    def tickers(self, symbols: list[str] | None = None) -> list[dict]:
        params = {"symbols": ",".join(symbols)} if symbols else None
        raw = self._request("GET", "/tickers", params)
        rows = raw.get("data", []) if isinstance(raw, dict) else raw
        for row in rows:
            row["symbol"] = path_symbol(row["symbol"])
        return rows

    def ticker_24h(self) -> list[dict]:
        """Binance-shaped 24 h tickers: ``symbol``, ``lastPrice``, ``quoteVolume``."""
        out = []
        for t in self.tickers():
            last = _num(t.get("last_price")) or _num(t.get("mid"))
            qv = _num(t.get("quote_volume_24h")) or _num(t.get("volume_24h")) * last
            out.append({"symbol": t["symbol"], "lastPrice": str(last), "quoteVolume": str(qv)})
        return out

    def ticker_prices(self) -> dict[str, float]:
        """Mid prices; bid/ask are kept in ``last_quotes`` for the broker."""
        prices, quotes = {}, {}
        for t in self.tickers():
            bid, ask = _num(t.get("bid")), _num(t.get("ask"))
            mid = _num(t.get("mid"))
            if not mid and bid and ask:
                mid = (bid + ask) / 2
            price = mid or _num(t.get("last_price"))
            if price <= 0:
                continue
            prices[t["symbol"]] = price
            quotes[t["symbol"]] = {"bid": bid or price, "ask": ask or price}
        self.last_quotes = quotes
        return prices

    def candles(self, symbol: str, minutes: int, since: int, until: int) -> list[dict]:
        raw = self._request("GET", f"/candles/{path_symbol(symbol)}",
                            {"interval": minutes, "since": since, "until": until})
        return raw.get("data", []) if isinstance(raw, dict) else raw

    def klines(self, symbol: str, interval: str, limit: int = 300,
               start_time: int | None = None, end_time: int | None = None) -> list[list]:
        """Binance-shaped kline rows (``[open_time, o, h, l, c, v, close_time]``).

        With ``start_time``: up to ``limit`` candles from it. Without: the last
        ``limit`` candles, the one still forming included (callers drop it).
        """
        minutes = INTERVAL_MINUTES.get(interval)
        if minutes is None:
            raise ValueError(f"Revolut X has no {interval!r} candles")
        step = minutes * 60_000
        now = int(time.time() * 1000)
        if start_time is not None:
            since = start_time // step * step
            until = min(end_time or now, since + (limit - 1) * step)
        else:
            until = (end_time or now) // step * step
            since = until - (limit - 1) * step
        by_start: dict[int, list] = {}
        cursor = since
        while cursor <= until:
            chunk_end = min(until, cursor + (CANDLES_PER_REQUEST - 1) * step)
            for c in self.candles(symbol, minutes, cursor, chunk_end):
                t = to_ms(c["start"])
                by_start[t] = [t, str(c["open"]), str(c["high"]), str(c["low"]),
                               str(c["close"]), str(c.get("volume", "0")), t + step - 1]
            cursor = chunk_end + step
        rows = [by_start[t] for t in sorted(by_start) if since <= t <= until]
        return rows[:limit] if start_time is not None else rows[-limit:]

    # ----------------------------------------------------------------- orders
    def place_order(self, symbol: str, side: str, *, base_size: Decimal | float,
                    price: Decimal | float | None = None, time_in_force: str | None = None,
                    client_order_id: str | None = None) -> dict:
        """Limit order when ``price`` is given, market order otherwise."""
        if price is not None:
            config: dict = {"limit": {"price": fmt_decimal(price),
                                      "base_size": fmt_decimal(base_size)}}
            if time_in_force:
                config["limit"]["time_in_force"] = time_in_force
        else:
            config = {"market": {"base_size": fmt_decimal(base_size)}}
        body = {"client_order_id": client_order_id or str(uuid.uuid4()),
                "symbol": path_symbol(symbol), "side": side.lower(),
                "order_configuration": config}
        raw = self._request("POST", "/orders", body=body)
        data = raw.get("data") if isinstance(raw, dict) else None
        if not isinstance(data, dict) or not data.get("venue_order_id"):
            raise RevolutXOutcomeUnknown(200, "reply without the order's details")
        return data

    def order(self, venue_order_id: str) -> dict:
        raw = self._request("GET", f"/orders/{venue_order_id}")
        return raw.get("data", raw) if isinstance(raw, dict) else {}

    def cancel_order(self, venue_order_id: str) -> bool:
        """False if the order was already gone (filled, cancelled, unknown)."""
        try:
            self._request("DELETE", f"/orders/{venue_order_id}")
            return True
        except RevolutXError as exc:
            if exc.status in (400, 404):
                return False
            raise

    def _paged(self, path: str, params: dict) -> list[dict]:
        out, cursor = [], None
        for _ in range(50):
            raw = self._request("GET", path, {**params, "cursor": cursor})
            out.extend(raw.get("data", []))
            cursor = (raw.get("metadata") or {}).get("next_cursor")
            if not cursor:
                break
        return out

    def active_orders(self, symbols: list[str] | None = None) -> list[dict]:
        params = {"symbols": ",".join(path_symbol(s) for s in symbols) if symbols else None}
        return self._paged("/orders/active", params)

    def historical_orders(self, symbols: list[str] | None, start_ms: int,
                          end_ms: int) -> list[dict]:
        params = {"symbols": ",".join(path_symbol(s) for s in symbols) if symbols else None,
                  "start_date": start_ms, "end_date": end_ms}
        return self._paged("/orders/historical", params)

    def find_order(self, client_order_id: str, symbol: str, since_ms: int) -> dict | None:
        """An order looked up by its client_order_id (active, then finished)."""
        for row in self.active_orders([symbol]):
            if row.get("client_order_id") == client_order_id:
                return row
        now = int(time.time() * 1000)
        start = max(since_ms - 60_000, now - 6 * 86_400_000)  # the API allows one week
        for row in self.historical_orders([symbol], start, now):
            if row.get("client_order_id") == client_order_id:
                return row
        return None


def _error(resp: requests.Response) -> RevolutXError:
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    message = str(body.get("message") or resp.text[:200] or f"HTTP {resp.status_code}")
    server_time = body.get("timestamp")
    return RevolutXError(resp.status_code, message, body.get("error_id"),
                         int(server_time) if isinstance(server_time, (int, float)) else None)


def order_fill(order: dict) -> tuple[float, float, float, str]:
    """``(filled_qty, avg_price, fee, fee_currency)`` of an order document."""
    qty = _num(order.get("filled_quantity"))
    avg = _num(order.get("average_fill_price"))
    if not avg and qty:
        avg = _num(order.get("filled_amount")) / qty if _num(order.get("filled_amount")) \
            else _num(order.get("price"))
    return qty, avg, _num(order.get("total_fee")), str(order.get("fee_currency") or "")
