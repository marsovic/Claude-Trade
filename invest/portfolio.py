"""Positions, closed trades and the persistent state of the investment bot.

Money is tracked in EUR (the account currency); prices stay in the
instrument currency (USD for US stocks). The JSON shape matches what the
dashboard reads, so ``dashboard/dashboard.py --state-dir state_invest`` works.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields

EQUITY_HISTORY_MAX = 5000


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class InvestPosition:
    symbol: str
    qty: float
    entry_price: float  # average price, instrument currency
    cost_eur: float  # total paid in EUR, fees included (PMP basis)
    initial_stop: float
    stop_loss: float
    atr: float
    risk_amount: float  # EUR at risk at entry (qty * stop distance)
    opened_at: int  # ms
    entry_day: str  # YYYY-MM-DD of the first fill
    sleeve: str = "stock"  # stock | etp
    currency: str = "USD"
    broker_ticker: str = ""
    highest: float = 0.0
    initial_qty: float = 0.0
    first_entry: float = 0.0  # price of the first fill: R stays anchored to it
    adds: int = 0
    stop_order_id: str = ""
    side: str = "LONG"
    leverage: float = 1.0  # 3 for a 3x ETP (shown in the dashboard)
    take_profit: float | None = None  # momentum exits have no fixed target
    unrealized_eur: float = 0.0
    last_price: float = 0.0

    def __post_init__(self):
        if not self.highest:
            self.highest = self.entry_price
        if not self.initial_qty:
            self.initial_qty = self.qty
        if not self.first_entry:
            self.first_entry = self.entry_price
        if not self.last_price:
            self.last_price = self.entry_price

    @property
    def margin(self) -> float:  # "invested" in the dashboard
        return self.cost_eur

    @property
    def r_distance(self) -> float:
        return max(self.first_entry - self.initial_stop, 1e-12)

    def r_multiple(self, price: float) -> float:
        """Move since the first entry, in units of the initial risk (adds do not
        shrink R)."""
        return (price - self.first_entry) / self.r_distance

    def apply_split(self, ratio: float) -> None:
        """``ratio`` new shares per old share (10 for a 10-for-1 split)."""
        self.qty *= ratio
        self.initial_qty *= ratio
        for name in ("entry_price", "first_entry", "initial_stop", "stop_loss", "highest",
                     "last_price", "atr"):
            setattr(self, name, getattr(self, name) / ratio)

    def value_eur(self, price: float, usd_per_eur: float) -> float:
        rate = usd_per_eur if self.currency == "USD" else 1.0
        return self.qty * price / rate

    def to_dict(self) -> dict:
        d = asdict(self)
        d["margin"] = self.margin
        return d

    @classmethod
    def from_dict(cls, data: dict) -> InvestPosition:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class InvestTrade:
    symbol: str
    qty: float
    entry_price: float
    exit_price: float
    cost_eur: float
    proceeds_eur: float
    pnl: float  # EUR, net of all fees
    fees: float  # EUR, exit side (entry fees are inside cost_eur)
    r_multiple: float
    reason: str
    opened_at: int
    closed_at: int
    sleeve: str = "stock"
    side: str = "LONG"
    leverage: float = 1.0

    @property
    def margin(self) -> float:
        return self.cost_eur

    def to_dict(self) -> dict:
        d = asdict(self)
        d["margin"] = self.margin
        return d


@dataclass
class InvestState:
    mode: str
    initial_capital: float
    cash: float  # EUR (simulator / bot ledger)
    started_at: int
    status: str = "running"
    equity: float = 0.0
    peak_equity: float = 0.0
    positions: dict[str, InvestPosition] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)
    pending: list[dict] = field(default_factory=list)  # simulator orders for next open
    open_orders: dict[str, dict] = field(default_factory=dict)  # real-broker orders by id
    last_day: str = ""  # last processed trading day (YYYY-MM-DD)
    realized_eur: float = 0.0
    fees_eur: float = 0.0
    fee_log: list[list] = field(default_factory=list)  # [day, eur] for the fee halt
    week_id: str = ""
    week_start_equity: float = 0.0
    pause_until: str = ""
    daily_halt: bool = False  # True while new entries are paused (dashboard badge)
    regime: dict = field(default_factory=dict)
    equity_history: list[list[float]] = field(default_factory=list)
    universe: list[str] = field(default_factory=list)
    last_scan: dict = field(default_factory=dict)
    sentiment: dict = field(default_factory=dict)
    stats: dict = field(default_factory=lambda: {
        "trades": 0, "wins": 0, "losses": 0,
        "gross_profit": 0.0, "gross_loss": 0.0, "fees": 0.0,
    })
    heartbeat: int = 0
    next_scan_at: int = 0
    settings: dict = field(default_factory=dict)
    broker_sync_ms: int = 0
    seen_fills: list[str] = field(default_factory=list)
    missing_checks: dict[str, int] = field(default_factory=dict)  # "closed by hand?" checks
    prev_close_equity: float = 0.0  # equity at the previous processed close
    etp_anchor: list = field(default_factory=list)  # [day, price] of the synthetic ETP
    floor_breaches: int = 0  # consecutive intraday readings below the survival floor
    applied_splits: list[str] = field(default_factory=list)  # "SYM:YYYY-MM-DD"

    @classmethod
    def fresh(cls, mode: str, capital: float) -> InvestState:
        t = now_ms()
        return cls(mode=mode, initial_capital=capital, cash=capital, started_at=t,
                   equity=capital, peak_equity=capital, week_start_equity=capital,
                   broker_sync_ms=t)

    def invested_eur(self) -> float:
        return sum(p.cost_eur for p in self.positions.values())

    def record_equity(self, t: int, equity: float) -> None:
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if not self.equity_history or t > self.equity_history[-1][0]:
            self.equity_history.append([t, round(equity, 4)])
        elif t == self.equity_history[-1][0]:
            self.equity_history[-1][1] = round(equity, 4)
        if len(self.equity_history) > EQUITY_HISTORY_MAX:
            half = EQUITY_HISTORY_MAX // 2
            self.equity_history = self.equity_history[:half:2] + self.equity_history[half:]

    def record_trade(self, trade: InvestTrade) -> None:
        st = self.stats
        st["trades"] += 1
        if trade.pnl > 0:
            st["wins"] += 1
            st["gross_profit"] += trade.pnl
        else:
            st["losses"] += 1
            st["gross_loss"] += -trade.pnl

    def drawdown(self) -> float:
        return 0.0 if self.peak_equity <= 0 else 1 - self.equity / self.peak_equity

    def to_dict(self) -> dict:
        data = asdict(self)
        data["positions"] = {k: p.to_dict() for k, p in self.positions.items()}
        data["stats"]["fees"] = round(self.fees_eur, 4)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> InvestState:
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        clean["positions"] = {k: InvestPosition.from_dict(v)
                              for k, v in data.get("positions", {}).items()}
        return cls(**clean)
