"""LiveExecution + InvestBot against a fake broker and a fake data loader."""

from datetime import datetime

import pandas as pd
import pytest

from bot.state import StateStore
from invest.brokers.base import AccountInfo, BrokerPosition, LiveBroker, OrderUpdate
from invest.data import ET, MarketData
from invest.engine import Engine
from invest.execution import LiveExecution
from invest.invest_bot import BotIsDead, InvestBot, main
from invest.portfolio import InvestPosition, InvestState
from invest.tax import TaxLedger
from tests.invest_helpers import SYMS, make_market, settings, signal_case


class FakeBroker(LiveBroker):
    name = "fake"

    def __init__(self, resting_stops=True):
        self.resting_stops = resting_stops
        self.fractional = True
        self.orders = {}  # broker id -> {"order", "status", "filled", "price"}
        self.positions = {}  # ticker -> qty
        self.cancelled = []
        self.equity = 100.0
        self.cash = 100.0
        self.next_id = 1
        self.raise_next = None  # exception raised by the next submit
        self.record_anyway = False  # ...while the order still reaches the broker

    def resolve(self, symbol):
        return None if symbol == "UNTRADABLE" else f"{symbol}_US_EQ"

    def account(self):
        return AccountInfo(self.equity, self.cash, {
            t: BrokerPosition(t, q, 100.0, 105.0) for t, q in self.positions.items() if q > 0})

    def submit(self, order):
        exc, self.raise_next = self.raise_next, None
        if exc is not None and not self.record_anyway:
            raise exc
        bid = str(self.next_id)
        self.next_id += 1
        self.orders[bid] = {"order": order, "status": "pending", "filled": 0.0, "price": 0.0}
        if exc is not None:
            raise exc
        return bid

    def find(self, order, submitted_at):
        for bid, rec in self.orders.items():
            if rec["order"].id == order.id:
                return bid
        return None

    def cancel(self, broker_id):
        self.cancelled.append(broker_id)
        if broker_id in self.orders and self.orders[broker_id]["status"] in (
                "pending", "partially_filled"):
            self.orders[broker_id]["status"] = "cancelled"

    def fill(self, bid, price, qty=None):
        rec = self.orders[bid]
        o = rec["order"]
        qty = o.qty - rec["filled"] if qty is None else qty
        rec["filled"] += qty
        rec["price"] = price
        rec["status"] = "filled" if rec["filled"] >= o.qty - 1e-9 else "partially_filled"
        sign = 1 if o.side == "BUY" else -1
        self.positions[o.broker_ticker] = self.positions.get(o.broker_ticker, 0) + sign * qty

    def order_update(self, order):
        rec = self.orders.get(order.broker_id)
        if rec is None:
            return OrderUpdate("unknown")
        return OrderUpdate(rec["status"], rec["filled"], rec["price"],
                           0.05 if rec["filled"] else 0.0,
                           1_760_000_000_000 if rec["filled"] else 0)

    def by_kind(self, kind, status="pending"):
        return [b for b, r in self.orders.items()
                if r["order"].kind == kind and r["status"] == status]


def live_engine(tmp_path, broker=None, **kw):
    s = settings(state_dir=str(tmp_path), **kw)
    state = InvestState.fresh("trading212-paper", 100.0)
    state.universe = list(SYMS)
    store = StateStore(tmp_path, InvestState)
    broker = broker or FakeBroker()
    ex = LiveExecution(s, state, store, broker, clock=lambda: 1_760_000_000)
    engine = Engine(s, state, store, ex, ledger=TaxLedger(tmp_path))
    return engine, ex, broker, state, store


def next_open_clock(bars, day):
    """Clock at 10:00 New York on the session after ``day`` (when planned orders go out)."""
    nxt = bars["SPY"].index[bars["SPY"].index.get_loc(day) + 1]
    return lambda: datetime(nxt.year, nxt.month, nxt.day, 10, 0, tzinfo=ET).timestamp()


def test_planned_entry_submitted_filled_and_protected(tmp_path):
    bars, day, hits = signal_case()
    engine, ex, broker, state, store = live_engine(tmp_path)
    ex.clock = next_open_clock(bars, day)
    engine.process_day(day, MarketData(bars=bars))
    assert state.pending and state.pending[0]["status"] == "planned"
    assert broker.orders == {}  # nothing sent before the open
    assert ex.submit_planned(engine) == len(state.open_orders)
    bid = broker.by_kind("LIMIT")[0]
    broker.fill(bid, 101.0)
    ex.sync(engine)
    sym = hits[0]
    pos = state.positions[sym]
    assert pos.entry_price == 101.0 and pos.broker_ticker == f"{sym}_US_EQ"
    stops = broker.by_kind("STOP")
    assert len(stops) == 1 and pos.stop_order_id  # protective stop resting at the broker
    assert broker.orders[stops[0]]["order"].stop_price == pytest.approx(pos.stop_loss)
    # The stop fills while the bot sleeps: the position is closed.
    broker.fill(stops[0], pos.stop_loss)
    ex.sync(engine)
    assert sym not in state.positions
    assert store.recent_trades()[-1]["reason"] == "stop"


def test_exit_cancels_stop_before_selling(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    pos = InvestPosition(symbol="AMD", qty=0.3, entry_price=100, cost_eur=27.3,
                         initial_stop=94, stop_loss=94, atr=2, risk_amount=1.6, opened_at=0,
                         entry_day="2026-01-02", broker_ticker="AMD_US_EQ")
    state.positions["AMD"] = pos
    broker.positions["AMD_US_EQ"] = 0.3
    ex.on_stop_change(engine, pos)
    stop_bid = broker.by_kind("STOP")[0]
    engine._exit("AMD", pos, "rank exit", pd.Timestamp("2026-10-14"))
    ex.submit_planned(engine)
    assert stop_bid in broker.cancelled  # shares unlocked before the market sell...
    assert broker.by_kind("MARKET") == []  # ...but nothing sold until it is confirmed
    assert state.pending and state.pending[0]["side"] == "SELL"
    ex.submit_planned(engine)  # next cycle: the cancellation is confirmed -> sell
    assert broker.by_kind("STOP") == []  # no new stop placed in between
    sell = broker.by_kind("MARKET")[0]
    broker.fill(sell, 110.0)
    ex.sync(engine)
    assert "AMD" not in state.positions
    assert store.recent_trades()[-1]["pnl"] > 0


def test_stale_entry_cancelled_and_manual_close_detected(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    bars, day, hits = signal_case()
    ex.clock = next_open_clock(bars, day)
    engine.process_day(day, MarketData(bars=bars))
    ex.submit_planned(engine)
    pos = InvestPosition(symbol="ZZZ", qty=1, entry_price=10, cost_eur=9.1, initial_stop=9,
                         stop_loss=9, atr=0.3, risk_amount=0.9, opened_at=0,
                         entry_day="2026-01-02", broker_ticker="ZZZ_US_EQ")
    state.positions["ZZZ"] = pos  # but the broker does not hold it any more
    ex.settle(engine, day, MarketData(bars=bars))
    assert broker.by_kind("LIMIT", "cancelled")  # unfilled entry cancelled at the close
    assert "ZZZ" in state.positions  # one missing reading only raises a warning
    ex.settle(engine, day, MarketData(bars=bars))
    assert "ZZZ" not in state.positions
    assert store.recent_trades()[-1]["reason"] == "closed outside the bot"


def test_monitored_stop_when_broker_has_no_resting_stops(tmp_path):
    broker = FakeBroker(resting_stops=False)
    engine, ex, broker, state, store = live_engine(tmp_path, broker=broker)
    pos = InvestPosition(symbol="AMD", qty=0.3, entry_price=100, cost_eur=27.3,
                         initial_stop=94, stop_loss=94, atr=2, risk_amount=1.6, opened_at=0,
                         entry_day="2026-01-02", broker_ticker="AMD_US_EQ")
    state.positions["AMD"] = pos
    ex.on_stop_change(engine, pos)
    assert broker.orders == {}  # nothing resting
    ex.check_monitored_stops(engine, {"AMD": 95.0})
    assert broker.orders == {}
    ex.check_monitored_stops(engine, {"AMD": 93.5})
    ex.check_monitored_stops(engine, {"AMD": 93.0})  # no duplicate sell
    assert len(broker.by_kind("MARKET")) == 1


def test_untradable_symbol_is_skipped(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    from invest.brokers.base import OrderRequest

    assert not ex.queue(engine, OrderRequest("UNTRADABLE", "BUY", 1))
    assert state.pending == []


def test_equity_capped_by_broker(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    broker.equity = 5000.0
    ex.refresh_account()
    assert engine.equity(ex.broker_equity()) == pytest.approx(100.0)  # bot's own 100 EUR
    broker.equity = 60.0
    ex.refresh_account()
    assert engine.equity(ex.broker_equity()) == pytest.approx(60.0)


# -------------------------------------------------------------------- InvestBot
class Clock:
    def __init__(self, dt):
        self.t = dt.timestamp()

    def __call__(self):
        return self.t


def fake_loader(bars):
    def load(s, symbols, end):
        cut = {k: v[v.index <= pd.Timestamp(end)] for k, v in bars.items()}
        return MarketData(bars=cut)
    return load


def sim_bot(tmp_path, bars, when, **kw):
    s = settings(state_dir=str(tmp_path), **kw)
    clock = Clock(when)
    return InvestBot(s, data_loader=fake_loader(bars), clock=clock), clock


def test_bot_first_run_processes_only_the_last_close(tmp_path):
    bars = make_market()
    last = bars["SPY"].index[-1]
    bot, _ = sim_bot(tmp_path, bars, datetime(last.year, last.month, last.day, 17, 0,
                                             tzinfo=ET))
    bot.setup()
    assert bot.daily_job()
    assert bot.state.last_day == str(last.date())
    assert not bot.daily_job()  # nothing new


def test_bot_catches_up_missed_days_in_simulation(tmp_path):
    bars = make_market()
    idx = bars["SPY"].index
    bot, clock = sim_bot(tmp_path, bars, datetime(2025, 1, 1, tzinfo=ET))
    bot.setup()
    bot.state.last_day = str(idx[-6].date())
    clock.t = datetime(idx[-1].year, idx[-1].month, idx[-1].day, 17, 0, tzinfo=ET).timestamp()
    bot.daily_job()
    assert bot.state.last_day == str(idx[-1].date())
    assert len(bot.state.equity_history) == 5  # the 5 missed closes, in order
    times = [t for t, _ in bot.state.equity_history]
    assert times == sorted(times)


def test_daily_due_and_session_windows(tmp_path):
    bars = make_market()
    bot, clock = sim_bot(tmp_path, bars, datetime(2026, 10, 14, 12, 0, tzinfo=ET))
    bot.setup()
    assert bot.in_session(bot.now_et()) and not bot.daily_due(bot.now_et())
    clock.t = datetime(2026, 10, 14, 16, 45, tzinfo=ET).timestamp()
    assert bot.daily_due(bot.now_et())
    bot.state.next_scan_at = int(datetime(2026, 10, 15, 16, 30, tzinfo=ET).timestamp() * 1000)
    assert not bot.daily_due(bot.now_et())  # already processed today's close


def test_bot_refuses_to_start_when_dead(tmp_path):
    bars = make_market()
    StateStore(tmp_path, InvestState).write_tombstone({"died_at_iso": "x", "equity": 3})
    bot, _ = sim_bot(tmp_path, bars, datetime(2026, 1, 5, tzinfo=ET))
    with pytest.raises(BotIsDead):
        bot.setup()
    code = main(["--state-dir", str(tmp_path), "--env-file", str(tmp_path / "none"),
                 "--once"])
    assert code == 0  # dead and flat: exits cleanly, launchd does not restart it


def test_cli_rejects_live_without_unlock(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("T212_API_KEY", "k")
    monkeypatch.setenv("T212_API_SECRET", "s")
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    code = main(["--broker", "trading212", "--environment", "live", "--state-dir",
                 str(tmp_path), "--env-file", str(tmp_path / "none"), "--once"])
    assert code == 1
    assert "locked" in capsys.readouterr().err


def test_live_bot_never_replays_history(tmp_path):
    bars = make_market()
    idx = bars["SPY"].index
    s = settings(state_dir=str(tmp_path), broker="trading212", t212_api_key="k",
                 t212_api_secret="s")
    clock = Clock(datetime(idx[-1].year, idx[-1].month, idx[-1].day, 17, 0, tzinfo=ET))
    bot = InvestBot(s, broker=FakeBroker(), data_loader=fake_loader(bars), clock=clock)
    bot.setup()
    bot.state.last_day = str(idx[-10].date())
    bot.daily_job()
    assert bot.state.last_day == str(idx[-1].date())
    assert len(bot.state.equity_history) == 1  # only the latest close, never a replay


def test_entry_that_missed_its_session_is_dropped(tmp_path):
    """The bot was down on the session the entry was planned for: never buy a day late."""
    bars, day, hits = signal_case()
    engine, ex, broker, state, store = live_engine(tmp_path)
    engine.process_day(day, MarketData(bars=bars))
    later = bars["SPY"].index[bars["SPY"].index.get_loc(day) + 3]  # (a holiday may sit between)
    ex.clock = lambda: datetime(later.year, later.month, later.day, 10, 0,
                                tzinfo=ET).timestamp()
    assert ex.submit_planned(engine) == 0 and broker.orders == {}
    assert any("missed its session" in d["message"] for d in store.recent_decisions())
