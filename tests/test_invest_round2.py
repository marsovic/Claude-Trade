"""Regression tests for the second adversarial review of the invest bot."""

from datetime import date, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from bot.state import StateStore
from invest.backtest import InvestBacktest, _lag
from invest.config import ConfigError
from invest.data import ET, BarStore, load_market_data
from invest.invest_bot import InvestBot
from invest.portfolio import InvestState
from tests.invest_helpers import make_market, settings
from tests.test_invest_data_research import FakeProvider
from tests.test_invest_live import Clock, FakeBroker, fake_loader
from tests.test_invest_review_fixes import _pos


def market_ending(day: str) -> dict[str, pd.DataFrame]:
    start = (pd.Timestamp(day) - pd.offsets.BDay(519)).strftime("%Y-%m-%d")
    return make_market(start=start)


# ---------------------------------------------------------------------- data
def test_split_survives_a_restart(tmp_path):
    p = FakeProvider()
    BarStore(tmp_path, p, history_days=60).update(["AMD"], date(2026, 10, 12))
    p.split = True
    BarStore(tmp_path, p, history_days=60).update(["AMD"], date(2026, 10, 14))
    # The history is now consistent: only the saved record still knows the split.
    again = BarStore(tmp_path, p, history_days=60)
    again.update(["AMD"], date(2026, 10, 14))
    assert again.splits == {"AMD": ("2026-10-13", 10.0)}


def test_split_seen_on_a_weekend_takes_effect_next_session(tmp_path):
    p = FakeProvider()
    store = BarStore(tmp_path, p, history_days=60)
    store.update(["AMD"], date(2026, 10, 9))  # Friday
    p.split = True
    store.update(["AMD"], date(2026, 10, 11))  # Sunday: no new bar
    assert store.splits == {"AMD": ("2026-10-12", 10.0)}


def test_etp_prices_fetched_only_for_a_new_close(monkeypatch):
    spy = make_market()["SPY"]
    calls = []

    def boom(*a, **k):
        raise OSError("offline")

    monkeypatch.setattr("invest.data.alphavantage_daily",
                        lambda *a, **k: calls.append(a) or pd.DataFrame())
    monkeypatch.setattr("invest.data.fred_series", boom)
    monkeypatch.setattr("invest.data.ecb_eurusd", boom)
    store = SimpleNamespace(splits={}, update=lambda symbols, end: {"SPY": spy})
    s = settings(broker="trading212", t212_api_key="k", t212_api_secret="s",
                 etp_enabled=True, etp_price_symbol="QQQ3.LON", alphavantage_api_key="a")
    md = load_market_data(s, ["SPY"], spy.index[-1].date(), store=store, fx_fallback=1.25,
                          last_day=spy.index[-1].date())
    assert calls == [] and md.eurusd_fallback == 1.25  # retry before a new bar: no AV call
    load_market_data(s, ["SPY"], spy.index[-1].date(), store=store,
                     last_day=spy.index[-2].date())
    assert len(calls) == 1


# ------------------------------------------------------------------- the loop
def test_nyse_holiday_is_not_a_session(tmp_path):
    bot = InvestBot(settings(state_dir=str(tmp_path)), data_loader=fake_loader(make_market()),
                    clock=Clock(datetime(2026, 11, 26, 12, 0, tzinfo=ET)))  # Thanksgiving
    bot.setup()
    assert not bot.in_session(bot.now_et()) and bot.daily_due(bot.now_et())


def test_next_scan_skips_the_holiday_and_today_bar_waits_for_the_close(tmp_path):
    bars = market_ending("2026-11-25")
    clock = Clock(datetime(2026, 11, 25, 15, 0, tzinfo=ET))
    bot = InvestBot(settings(state_dir=str(tmp_path)), data_loader=fake_loader(bars),
                    clock=clock)
    bot.setup()
    bot.daily_job()
    assert bot.state.last_day == "2026-11-24"  # 15:00: today's bar is still forming
    clock.t = datetime(2026, 11, 25, 17, 0, tzinfo=ET).timestamp()
    bot.state.next_scan_at = 0
    bot.daily_job()
    assert bot.state.last_day == "2026-11-25"
    nxt = datetime.fromtimestamp(bot.state.next_scan_at / 1000, tz=ET)
    assert nxt.date() == date(2026, 11, 27)  # Thanksgiving skipped


def test_catch_up_days_run_without_research(tmp_path):
    bars = make_market()
    idx = bars["SPY"].index
    marker = SimpleNamespace()
    clock = Clock(datetime(idx[-1].year, idx[-1].month, idx[-1].day, 17, 0, tzinfo=ET))
    bot = InvestBot(settings(state_dir=str(tmp_path)), research=marker,
                    data_loader=fake_loader(bars), clock=clock)
    bot.setup()
    bot.state.last_day = str(idx[-4].date())
    seen = []
    bot.engine.process_day = lambda day, data, **kw: seen.append(bot.engine.research)
    bot.daily_job()
    assert seen == [None, None, marker] and bot.engine.research is marker


def test_once_is_refused_with_a_real_broker(tmp_path):
    s = settings(state_dir=str(tmp_path), broker="trading212", t212_api_key="k",
                 t212_api_secret="s")
    bot = InvestBot(s, broker=FakeBroker(), data_loader=fake_loader(make_market()),
                    clock=Clock(datetime(2026, 10, 14, 17, 0, tzinfo=ET)))
    with pytest.raises(ConfigError, match="--once"):
        bot.run_once()


def test_wind_down_sends_a_new_sale_after_a_rejected_one(tmp_path):
    s = settings(state_dir=str(tmp_path), broker="trading212", t212_api_key="k",
                 t212_api_secret="s")
    store = StateStore(tmp_path, InvestState)
    state = InvestState.fresh(s.mode, 100.0)
    state.positions["AMD"] = _pos()
    store.save(state)
    store.write_tombstone({"died_at_iso": "x", "equity": 9})
    broker = FakeBroker()
    broker.positions["AMD_US_EQ"] = 0.3
    bot = InvestBot(s, broker=broker, data_loader=fake_loader(make_market()),
                    clock=Clock(datetime(2026, 10, 14, 11, 0, tzinfo=ET)))
    bot.setup()
    bot.engine.die(pd.Timestamp("2026-10-14"), 9.0, 1)
    bot.wind_down(max_days=0)
    (first,) = broker.by_kind("MARKET")
    broker.orders[first]["status"] = "rejected"
    bot.wind_down(max_days=0)  # re-planned; the safety stop placed meanwhile is cancelled
    assert any(o["reason"] == "death" for o in bot.state.pending)
    bot.wind_down(max_days=0)  # stop cancellation confirmed: the new sale goes out
    sells = [b for b, r in broker.orders.items()
             if r["order"].kind == "MARKET" and r["order"].side == "SELL"]
    assert len(sells) == 2 and broker.orders[sells[-1]]["status"] == "pending"


# ------------------------------------------------------------------ backtest
def test_yearly_returns_start_from_the_previous_close():
    curve = pd.Series([100.0, 110.0, 121.0],
                      index=pd.to_datetime(["2025-12-30", "2025-12-31", "2026-01-02"]))
    bt = InvestBacktest(state=InvestState.fresh("sim", 100.0), trades=[], curve=curve,
                        benchmark=pd.Series(dtype=float), days=3)
    assert bt.stats()["yearly_return_pct"] == {2025: 10.0, 2026: 10.0}


def test_vix_is_lagged_one_session():
    d = pd.to_datetime(["2026-10-13", "2026-10-14"])
    assert _lag(pd.Series([10.0, 20.0], index=d)).to_dict() == {d[1]: 10.0}
