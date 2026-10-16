"""Persistent state shared between the bot and the dashboard.

Files inside ``STATE_DIR``:

* ``state.json``      current snapshot (equity, positions, last scan, ...)
* ``decisions.jsonl`` append-only journal of every decision the bot takes
* ``trades.jsonl``    append-only list of closed trades
* ``TOMBSTONE.json``  written once when the capital is lost; while it exists
                      the bot refuses to start (the survival rule)
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from bot.risk import ClosedTrade, Position

EQUITY_HISTORY_MAX = 10_000
EQUITY_SAMPLE_MS = 60_000


def now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class BotState:
    mode: str
    initial_capital: float
    balance: float
    started_at: int
    status: str = "running"  # running | stopped | dead
    venue: str = "binance"  # binance | revolutx
    equity: float = 0.0
    peak_equity: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    marks: dict[str, float] = field(default_factory=dict)  # last price of open positions
    cooldowns: dict[str, int] = field(default_factory=dict)
    day: str = ""
    day_start_equity: float = 0.0
    daily_halt: bool = False
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
    # Live mode: running total of futures income (realized PnL, fees, funding).
    income_cursor: int = 0
    income_total: float = 0.0
    income_seen: list[int] = field(default_factory=list)
    # Revolut X: entries sent but not confirmed yet, by client_order_id.
    pending_entries: dict[str, dict] = field(default_factory=dict)

    @classmethod
    def fresh(cls, mode: str, initial_capital: float, venue: str = "binance") -> BotState:
        t = now_ms()
        return cls(mode=mode, initial_capital=initial_capital, balance=initial_capital,
                   started_at=t, equity=initial_capital, peak_equity=initial_capital,
                   income_cursor=t, venue=venue)

    def used_margin(self) -> float:
        return sum(p.margin for p in self.positions.values())

    def record_equity(self, t: int, equity: float) -> None:
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)
        if not self.equity_history or t - self.equity_history[-1][0] >= EQUITY_SAMPLE_MS:
            self.equity_history.append([t, round(equity, 4)])
            if len(self.equity_history) > EQUITY_HISTORY_MAX:
                # Keep the shape of old history at half resolution.
                head = self.equity_history[: EQUITY_HISTORY_MAX // 2 : 2]
                self.equity_history = head + self.equity_history[EQUITY_HISTORY_MAX // 2:]

    def record_trade(self, trade: ClosedTrade) -> None:
        st = self.stats
        st["trades"] += 1
        st["fees"] += trade.fees
        if trade.pnl > 0:
            st["wins"] += 1
            st["gross_profit"] += trade.pnl
        else:
            st["losses"] += 1
            st["gross_loss"] += -trade.pnl

    def to_dict(self) -> dict:
        data = asdict(self)
        data["positions"] = {k: p.to_dict() for k, p in self.positions.items()}
        return data

    @classmethod
    def from_dict(cls, data: dict) -> BotState:
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in data.items() if k in known}
        clean["positions"] = {k: Position.from_dict(v)
                              for k, v in data.get("positions", {}).items()}
        return cls(**clean)


class StateStore:
    """``state_cls`` lets another bot persist its own state type (needs
    ``from_dict``/``to_dict``) while sharing the journals and the tombstone."""

    def __init__(self, state_dir: str | Path, state_cls=None):
        self.state_cls = state_cls or BotState
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.decisions_path = self.dir / "decisions.jsonl"
        self.trades_path = self.dir / "trades.jsonl"
        self.tombstone_path = self.dir / "TOMBSTONE.json"

    # ------------------------------------------------------------ snapshot
    def load(self):
        if not self.state_path.exists():
            return None
        return self.state_cls.from_dict(json.loads(self.state_path.read_text()))

    def save(self, state) -> None:
        _atomic_write(self.state_path, json.dumps(state.to_dict(), separators=(",", ":"),
                                                  default=_json_default))

    # ------------------------------------------------------------- journals
    def log_decision(self, kind: str, message: str, symbol: str | None = None,
                     side: str | None = None, data: dict | None = None,
                     ts: int | None = None) -> dict:
        entry = {"ts": ts or now_ms(), "kind": kind, "symbol": symbol, "side": side,
                 "message": message, "data": data or {}}
        _append_line(self.decisions_path, entry)
        return entry

    def log_trade(self, trade: ClosedTrade) -> None:
        _append_line(self.trades_path, trade.to_dict())

    def recent_decisions(self, n: int = 200) -> list[dict]:
        return _tail_jsonl(self.decisions_path, n)

    def recent_trades(self, n: int = 200) -> list[dict]:
        return _tail_jsonl(self.trades_path, n)

    # ------------------------------------------------------------ tombstone
    def tombstone(self) -> dict | None:
        if not self.tombstone_path.exists():
            return None
        return json.loads(self.tombstone_path.read_text())

    def write_tombstone(self, info: dict) -> None:
        _atomic_write(self.tombstone_path, json.dumps(info, indent=2))


def _json_default(obj):
    """numpy scalars (np.bool_, np.float64...) -> plain Python values."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _append_line(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, separators=(",", ":"),
                            default=lambda o: o.item() if hasattr(o, "item") else str(o))
                 + "\n")


def _tail_lines(path: Path, n: int, block: int = 65536) -> list[str]:
    """Last ``n`` lines, reading backwards: cheap even on a huge journal."""
    with path.open("rb") as fh:
        fh.seek(0, os.SEEK_END)
        pos = fh.tell()
        data = b""
        while pos > 0 and data.count(b"\n") <= n:
            step = min(block, pos)
            pos -= step
            fh.seek(pos)
            data = fh.read(step) + data
    lines = data.decode("utf-8", errors="replace").splitlines()
    return list(deque(lines, maxlen=n))


def _tail_jsonl(path: Path, n: int) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in _tail_lines(path, n):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line being written right now
    return out
