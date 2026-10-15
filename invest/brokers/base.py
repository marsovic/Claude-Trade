"""Broker-independent order and fill records, and the live-broker interface."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field, fields


@dataclass
class OrderRequest:
    symbol: str  # engine symbol (US ticker, or "ETP:QQQx3")
    side: str  # BUY | SELL
    qty: float
    kind: str = "MARKET"  # MARKET | LIMIT | STOP
    limit_price: float | None = None
    stop_price: float | None = None
    reason: str = ""  # entry | add | exit:<why> | stop
    sleeve: str = "stock"
    currency: str = "USD"
    broker_ticker: str = ""
    stop_pct: float = 0.0  # entries: protective stop distance applied to the fill
    atr: float = 0.0
    risk_eur: float = 0.0
    day: str = ""  # decision day
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    broker_id: str = ""
    status: str = "planned"  # planned | submitted | filled | cancelled | rejected

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> OrderRequest:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float  # instrument currency
    fees_eur: float
    usd_per_eur: float
    day: str  # YYYY-MM-DD
    ts: int  # ms
    reason: str
    order: dict = field(default_factory=dict)  # the originating OrderRequest
    fill_id: str = ""


@dataclass
class BrokerPosition:
    ticker: str
    qty: float
    avg_price: float
    price: float
    currency: str = "USD"


@dataclass
class AccountInfo:
    equity_eur: float
    cash_eur: float
    positions: dict[str, BrokerPosition] = field(default_factory=dict)


@dataclass
class OrderUpdate:
    status: str  # pending | filled | partially_filled | cancelled | rejected | unknown
    filled_qty: float = 0.0
    avg_price: float = 0.0
    fees_eur: float = 0.0
    filled_at: int = 0


class BrokerError(RuntimeError):
    pass


class LiveBroker:
    """Interface implemented by Trading 212, Alpaca and IBKR adapters."""

    name = "live"
    resting_stops = True  # stop orders can rest at the broker
    fractional = True

    def resolve(self, symbol: str) -> str | None:
        """Broker identifier for an engine symbol (None if not tradable)."""
        raise NotImplementedError

    def account(self) -> AccountInfo:
        raise NotImplementedError

    def submit(self, order: OrderRequest) -> str:
        raise NotImplementedError

    def cancel(self, broker_id: str) -> None:
        raise NotImplementedError

    def order_update(self, order: OrderRequest) -> OrderUpdate:
        raise NotImplementedError

    def begin_sync(self) -> None:
        """Hook to batch the expensive calls of one synchronisation pass."""

    def preflight(self) -> list[str]:
        return []
