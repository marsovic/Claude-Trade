"""Regression tests for the defects found by the adversarial review."""

from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bot.state import StateStore
from invest.backtest import run_backtest
from invest.brokers.base import BrokerError, OrderRequest
from invest.data import ET, MarketData
from invest.engine import Engine
from invest.execution import SimExecution
from invest.invest_bot import InvestBot
from invest.portfolio import InvestPosition, InvestState
from invest.signals import build_panel
from invest.tax import TaxLedger
from tests.invest_helpers import SYMS, make_market, settings, signal_case
from tests.test_invest_live import Clock, FakeBroker, fake_loader, live_engine


def _pos(sym="AMD", **kw):
    base = dict(symbol=sym, qty=0.3, entry_price=100.0, cost_eur=27.3, initial_stop=94.0,
                stop_loss=94.0, atr=2.0, risk_amount=1.6, opened_at=0, entry_day="2026-01-02",
                broker_ticker=f"{sym}_US_EQ")
    base.update(kw)
    return InvestPosition(**base)


# ------------------------------------------------------------- execution
def test_trading212_total_quantity_includes_shares_under_a_stop():
    from tests.test_invest_brokers import Resp, t212

    positions = [{"instrument": {"ticker": "AMD_US_EQ", "currency": "USD"}, "quantity": 0.3,
                  "quantityAvailableForTrading": 0.0, "averagePricePaid": 150.0,
                  "currentPrice": 160.0}]
    b = t212({("GET", "/equity/account/summary"): Resp(200, {"currency": "EUR", "cash": {},
                                                           "investments": {}}),
              ("GET", "/equity/positions"): Resp(200, positions)})
    pos = b.account().positions["AMD_US_EQ"]
    assert pos.qty == 0.3 and pos.available == 0.0


def test_position_protected_by_its_stop_is_not_closed(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    state.positions["AMD"] = _pos()
    broker.positions["AMD_US_EQ"] = 0.3
    ex.on_stop_change(engine, state.positions["AMD"])
    day = pd.Timestamp("2026-10-14")
    for _ in range(3):
        ex.settle(engine, day, MarketData(bars={}))
    assert "AMD" in state.positions and broker.by_kind("STOP")


def test_death_cancels_entries_and_books_the_liquidation(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    state.positions["AMD"] = _pos()
    broker.positions["AMD_US_EQ"] = 0.3
    ex.on_stop_change(engine, state.positions["AMD"])
    ex.queue(engine, OrderRequest("NVDA", "BUY", 0.1, kind="LIMIT", limit_price=150,
                                  reason="entry", day="2026-10-14"))
    ex.submit_planned(engine)
    buy = broker.by_kind("LIMIT")[0]
    engine.day = pd.Timestamp("2026-10-15")
    engine.die(engine.day, 9.0, 1_760_000_000_000)
    assert buy in broker.cancelled  # no new exposure after death
    assert store.tombstone() is not None
    for _ in range(3):  # wind-down: stop cancel confirmed, then the sale goes out
        ex.submit_planned(engine)
        for bid in broker.by_kind("MARKET"):
            broker.fill(bid, 95.0)
        ex.sync(engine)
    assert state.positions == {}
    rows = TaxLedger(tmp_path).rows()
    assert rows[-1]["side"] == "SELL" and rows[-1]["symbol"] == "AMD"  # in the 2074 ledger
    assert store.recent_trades()[-1]["reason"] == "death"


def test_timed_out_submission_is_found_not_resent(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    broker.raise_next = BrokerError("network error (order outcome unknown: reconciled)")
    broker.record_anyway = True
    ex.queue(engine, OrderRequest("AMD", "BUY", 0.3, kind="LIMIT", limit_price=101,
                                  reason="entry", day="2026-10-14", stop_pct=0.06))
    ex.submit_planned(engine)
    assert len(broker.orders) == 1  # sent once, never resent
    tracked = list(state.open_orders.values())[0]
    assert tracked["status"] == "unconfirmed" and not tracked["broker_id"]
    broker.fill("1", 100.0)
    ex.sync(engine)
    assert state.positions["AMD"].qty == pytest.approx(0.3)


def test_partial_fills_get_a_stop_and_are_not_pyramids(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    ex.queue(engine, OrderRequest("AMD", "BUY", 0.4, kind="LIMIT", limit_price=101,
                                  reason="entry", day="2026-10-14", stop_pct=0.06))
    ex.submit_planned(engine)
    bid = broker.by_kind("LIMIT")[0]
    broker.fill(bid, 100.0, qty=0.1)
    ex.sync(engine)
    pos = state.positions["AMD"]
    assert pos.qty == pytest.approx(0.1) and broker.by_kind("STOP")  # protected at once
    broker.fill(bid, 100.0)  # the rest
    ex.sync(engine)
    ex.sync(engine)  # old stop cancellation confirmed -> new stop for the full size
    assert pos.qty == pytest.approx(0.4) and pos.adds == 0
    stops = broker.by_kind("STOP")
    assert len(stops) == 1 and broker.orders[stops[0]]["order"].qty == pytest.approx(0.4)


def test_stale_entries_dropped_and_exits_deduplicated(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    state.last_day = "2026-10-14"
    ex.queue(engine, OrderRequest("AMD", "BUY", 0.3, kind="LIMIT", limit_price=101,
                                  reason="entry", day="2026-10-09"))
    ex.submit_planned(engine)
    assert broker.orders == {} and state.pending == []
    state.positions["NVDA"] = _pos("NVDA")
    assert ex.queue(engine, OrderRequest("NVDA", "SELL", 0.3, reason="rank exit"))
    assert not ex.queue(engine, OrderRequest("NVDA", "SELL", 0.3, reason="time stop"))


def test_connection_errors_keep_the_order_for_a_retry(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    broker.raise_next = ConnectionError("gateway down")
    ex.queue(engine, OrderRequest("AMD", "BUY", 0.3, kind="LIMIT", limit_price=101,
                                  reason="entry", day="2026-10-14"))
    ex.submit_planned(engine)  # must not crash
    assert broker.orders == {} and state.pending[0]["attempts"] == 1
    ex.submit_planned(engine)
    assert broker.by_kind("LIMIT")


def test_missing_or_zero_broker_equity_is_ignored(tmp_path):
    engine, ex, broker, state, store = live_engine(tmp_path)
    broker.equity = 0.0
    ex.refresh_account()
    assert ex.broker_equity() is None
    assert engine.equity(ex.broker_equity()) == pytest.approx(100.0)


def test_ibkr_restored_fill_without_details_uses_executions(monkeypatch):
    from tests.test_invest_brokers import ibkr

    b = ibkr(monkeypatch)
    order = OrderRequest("AMD", "BUY", 2, kind="LIMIT", limit_price=150.0)
    b.submit(order)
    trade = b.ib.placed[-1]
    trade.orderStatus.status, trade.orderStatus.filled = "Filled", 0  # restored, no detail
    ex = SimpleNamespace(shares=2, price=149.5, orderRef=order.id)
    fill = SimpleNamespace(execution=ex, commissionReport=SimpleNamespace(commission=1.1),
                           time=datetime(2026, 10, 14, 15, 0, tzinfo=ET))
    b.ib.reqExecutions = lambda: [fill]
    upd = b.order_update(order)
    assert upd.status == "filled" and upd.filled_qty == 2 and upd.avg_price == 149.5


# ---------------------------------------------------------------- engine
def sim_engine(tmp_path, **kw):
    s = settings(state_dir=str(tmp_path), **kw)
    state = InvestState.fresh("sim", 100.0)
    state.universe = list(SYMS)
    store = StateStore(tmp_path, InvestState)
    return Engine(s, state, store, SimExecution(s, state, s.fees)), state


def test_r_multiple_stays_anchored_to_the_first_entry():
    pos = _pos(entry_price=100.0, initial_stop=94.0)
    pos.entry_price = 105.0  # averaged up by a pyramid add
    assert pos.r_multiple(112.0) == pytest.approx(2.0)


def test_weekly_pause_counts_the_monday_gap(tmp_path):
    bars, day, hits = signal_case()
    engine, state = sim_engine(tmp_path)
    monday = next(d for d in bars["SPY"].index if d > day and d.weekday() == 0)
    state.week_id = "previous"
    state.prev_close_equity = 130.0  # last Friday's close; today equity is 100 (-23 %)
    engine.process_day(monday, MarketData(bars=bars))
    assert state.week_start_equity == 130.0 and state.daily_halt


def test_held_winner_not_sold_when_a_jump_makes_it_ineligible(tmp_path):
    bars = make_market()
    engine, state = sim_engine(tmp_path, max_daily_jump=0.0)  # nothing is entry-eligible
    days = bars["SPY"].index
    friday = next(d for d in days[420:] if d.weekday() == 4)
    panel = build_panel(bars, SYMS, engine.s)
    best = max(panel, key=lambda k: panel[k].loc[friday, "pct_rank_hold"])
    assert np.isnan(panel[best].loc[friday, "pct_rank"])  # ineligible for entries...
    price = bars[best].loc[friday, "close"]
    state.positions[best] = _pos(best, qty=0.1, entry_price=price * 0.9,
                                 initial_stop=price * 0.5, stop_loss=price * 0.5,
                                 entry_day=str(days[410].date()))
    engine.process_day(friday, MarketData(bars=bars), panel=panel)
    assert not any(o["side"] == "SELL" for o in state.pending)  # ...but still held


def test_split_rescales_positions_once(tmp_path):
    bars = make_market()
    engine, state = sim_engine(tmp_path, rebalance_weekday=9, time_stop_days=999)
    day = bars["SPY"].index[450]
    sym = "AAPL"
    p = float(bars[sym].loc[day, "low"])  # post-split (adjusted) price level
    # The position was opened before a 10-for-1 split, in pre-split prices.
    state.positions[sym] = _pos(sym, qty=0.2, entry_price=9.5 * p, initial_stop=5 * p,
                                stop_loss=5 * p, highest=9.5 * p, last_price=9.5 * p,
                                entry_day=str(bars["SPY"].index[440].date()))
    data = MarketData(bars=bars, splits={sym: (str(day.date()), 10.0)})
    engine.process_day(day, data)
    pos = state.positions[sym]
    assert pos.qty == pytest.approx(2.0)
    assert pos.initial_stop == pytest.approx(0.5 * p) and pos.entry_price == pytest.approx(0.95 * p)
    assert f"{sym}:{day.date()}" in state.applied_splits
    engine.process_day(bars["SPY"].index[451], data)  # not applied twice
    assert state.positions[sym].qty == pytest.approx(2.0)


def test_synthetic_etp_does_not_jump_when_the_window_moves(tmp_path):
    bars = make_market()
    engine, state = sim_engine(tmp_path, etp_enabled=True)
    days = bars["SPY"].index
    d1, d2 = days[400], days[401]
    window1 = {k: v.iloc[:401] for k, v in bars.items()}
    window2 = {k: v.iloc[30:402] for k, v in bars.items()}  # older bars dropped
    engine.process_day(d1, MarketData(bars=window1))
    md2 = MarketData(bars=window2)
    engine.process_day(d2, md2)
    sym = "ETP:QQQx3"
    assert state.etp_anchor[0] == str(d2.date())
    first = MarketData(bars={k: v.iloc[:401] for k, v in bars.items()})
    engine2, _ = sim_engine(tmp_path / "b", etp_enabled=True)
    engine2._synthetic_etp(first)
    assert md2.bars[sym].loc[d1, "close"] == pytest.approx(first.bars[sym].loc[d1, "close"])


def test_backtest_rerun_does_not_double_count(tmp_path):
    bars = make_market()
    s = settings()
    r1 = run_backtest(bars, s, out_dir=tmp_path)
    r2 = run_backtest(bars, s, out_dir=tmp_path)
    assert len(r1.trades) == len(r2.trades) > 0
    buys = [r for r in TaxLedger(tmp_path).rows() if r["side"] == "BUY"]
    assert len(buys) == len({r["day"] + r["symbol"] for r in buys})  # not appended twice
    fx = pd.Series(1.0, index=bars["SPY"].index)
    fx.iloc[400:] = 2.0  # EUR doubles vs USD after the backtest started (day 260)
    r3 = run_backtest(bars, s, eurusd=fx, out_dir=tmp_path / "fx")
    usd = bars["SPY"]["close"].reindex(r3.curve.index)
    expected = usd.iloc[-1] / usd.iloc[0] * 100 / 2.0
    assert r3.benchmark.iloc[-1] == pytest.approx(expected)


# ------------------------------------------------------------------ bot loop
def test_intraday_floor_needs_two_readings_then_winds_down(tmp_path):
    bars = make_market()
    s = settings(state_dir=str(tmp_path), broker="trading212", t212_api_key="k",
                 t212_api_secret="s")
    broker = FakeBroker()
    clock = Clock(datetime(2026, 10, 14, 11, 0, tzinfo=ET))
    bot = InvestBot(s, broker=broker, data_loader=fake_loader(bars), clock=clock)
    bot.setup()
    bot.state.cash = 5.0  # equity collapses below the 10 EUR floor
    bot.intraday(bot.now_et())
    assert bot.state.status == "running" and bot.state.floor_breaches == 1
    bot.intraday(bot.now_et())
    assert bot.state.status == "dead" and bot.store.tombstone()
    assert bot.wind_down(max_days=0) == 2


def test_restart_after_death_only_finishes_the_liquidation(tmp_path):
    bars = make_market()
    s = settings(state_dir=str(tmp_path), broker="trading212", t212_api_key="k",
                 t212_api_secret="s")
    store = StateStore(tmp_path, InvestState)
    state = InvestState.fresh(s.mode, 100.0)
    state.positions["AMD"] = _pos()
    store.save(state)
    store.write_tombstone({"died_at_iso": "x", "equity": 9})
    broker = FakeBroker()
    broker.positions["AMD_US_EQ"] = 0.3
    bot = InvestBot(s, broker=broker, data_loader=fake_loader(bars),
                    clock=Clock(datetime(2026, 10, 14, 11, 0, tzinfo=ET)))
    bot.setup()
    assert bot.winding_down and bot.state.status == "dead"
    bot.engine.die(pd.Timestamp("2026-10-14"), 9.0, 1)
    assert bot.wind_down(max_days=0) == 2
    assert broker.orders  # the liquidation is being worked on
