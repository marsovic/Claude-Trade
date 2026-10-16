"""Engine defects from the audit: drawdown ladder, entry-day high, holiday rebalance,
carried sell orders, volatility-scaled stop slippage."""

import pandas as pd
import pytest

from invest.brokers.base import OrderRequest
from invest.portfolio import InvestPosition
from invest.risk import ladder_level, risk_budget, size_position
from tests.invest_helpers import make_market, market_data, settings
from tests.test_invest_engine import make_engine


def pos(sym="AMD", **kw):
    base = dict(symbol=sym, qty=0.3, entry_price=100.0, cost_eur=27.0, initial_stop=90.0,
                stop_loss=90.0, atr=3.0, risk_amount=2.0, opened_at=0, entry_day="2026-01-02")
    base.update(kw)
    return InvestPosition(**base)


# ------------------------------------------------------------ drawdown ladder
def test_ladder_has_hysteresis():
    s = settings()  # trigger 20 %, release at 60 % of it = 12 %
    assert ladder_level(0.19, 0, s) == 0
    assert ladder_level(0.21, 0, s) == 1
    assert ladder_level(0.15, 1, s) == 1  # recovering but not enough: stays at level 1
    assert ladder_level(0.11, 1, s) == 0
    assert ladder_level(0.30, 1, s) == 1 and ladder_level(0.36, 1, s) == 2
    assert ladder_level(0.25, 2, s) == 2  # 25 % >= 35 % x 0.6
    assert ladder_level(0.15, 2, s) == 1  # level 2 left, level 1 still held


def test_derisking_never_makes_a_position_bigger():
    s = settings()
    sizes = []
    for dd in (0.0, 0.25, 0.40):
        budget = risk_budget(100.0, dd, s)
        sized, _ = size_position(50.0, 1.0, 100.0, 100.0, 0.0, budget, 1.1, s.fees, s)
        sizes.append(sized.notional_eur)
        assert budget.weight_slots == 3  # the weight cap still assumes 3 slots
    assert sizes[0] >= sizes[1] >= sizes[2]


def test_engine_remembers_the_ladder_level(tmp_path):
    bars = make_market()
    engine, state, _ = make_engine(tmp_path)
    state.peak_equity = 100.0
    state.cash = 78.0  # -22 %
    day = bars["SPY"].index[300]
    engine.process_day(day, market_data(bars))
    assert state.dd_level == 1
    state.cash = 86.0  # -14 %: above the 12 % release, so it stays at level 1
    engine.process_day(bars["SPY"].index[301], market_data(bars))
    assert state.dd_level == 1
    state.cash = 95.0
    engine.process_day(bars["SPY"].index[302], market_data(bars))
    assert state.dd_level == 0


# ---------------------------------------------------------- entry-day high
def test_limit_fill_after_a_gap_up_does_not_inherit_the_morning_high(tmp_path):
    bars = make_market()
    engine, state, _ = make_engine(tmp_path)
    day = bars["SPY"].index[300]
    sym = "AMD"
    df = bars[sym].copy()
    df.loc[day, ["open", "high", "low", "close"]] = [110.0, 120.0, 101.0, 103.0]
    bars[sym] = df
    data = market_data(bars)
    gap_fill = pos(entry_price=102.0, entry_day=str(day.date()))  # limit hit after a gap-up
    assert engine._usable_high(gap_fill, data, day, 120.0, 103.0) == 103.0
    open_fill = pos(entry_price=110.05, entry_day=str(day.date()))  # filled at the open
    assert engine._usable_high(open_fill, data, day, 120.0, 103.0) == 120.0
    older = pos(entry_price=102.0, entry_day="2026-01-02")
    assert engine._usable_high(older, data, day, 120.0, 103.0) == 120.0


# ------------------------------------------------------------ rebalance day
def test_rank_exit_runs_on_the_last_session_of_the_week(tmp_path):
    engine, _, _ = make_engine(tmp_path)
    ts = pd.Timestamp
    assert engine._rebalance_day(ts("2026-10-16"))  # an ordinary Friday
    assert not engine._rebalance_day(ts("2026-10-15"))  # Thursday
    assert engine._rebalance_day(ts("2026-04-02"))  # Good Friday 2026-04-03 is a holiday
    assert not engine._rebalance_day(ts("2026-04-01"))


# ----------------------------------------------------------- carried sells
def test_a_sell_without_a_bar_is_carried_to_the_next_session(tmp_path):
    bars = make_market()
    engine, state, _ = make_engine(tmp_path)
    day = bars["SPY"].index[300]
    sell = OrderRequest(symbol="GHOST", side="SELL", qty=1.0, kind="MARKET", reason="rank",
                        broker_ticker="GHOST")
    buy = OrderRequest(symbol="GHOST2", side="BUY", qty=1.0, kind="MARKET", reason="entry",
                       broker_ticker="GHOST2")
    state.pending = [sell.to_dict(), buy.to_dict()]
    fills, expired = engine.exec.broker.settle_open(day, market_data(bars))
    assert fills == [] and [o.symbol for o in expired] == ["GHOST2"]  # entries last one session
    assert [o["symbol"] for o in state.pending] == ["GHOST"]  # the sell waits for a bar


# ------------------------------------------------------------ stop slippage
def test_stop_slippage_scales_with_the_stocks_volatility(tmp_path):
    bars = make_market()
    day = bars["SPY"].index[300]
    fills = []
    for atr in (0.5, 8.0):
        engine, state, _ = make_engine(tmp_path / f"a{atr}")
        df = bars["AMD"].copy()
        df.loc[day, ["open", "high", "low", "close"]] = [95.0, 95.0, 85.0, 90.0]
        b = {**bars, "AMD": df}
        state.positions["AMD"] = pos(stop_loss=95.0, atr=atr)
        (f,) = engine.exec.broker.settle_stops(day, market_data(b), state.positions)
        fills.append(f.price)
    calm, wild = fills
    assert wild < calm  # the volatile stock fills further below its stop
    s = settings()
    assert calm == pytest.approx(95.0 * (1 - max(5e-4, s.stop_slippage_atr * 0.5 / 95.0)))
    assert wild == pytest.approx(95.0 * (1 - s.stop_slippage_atr * 8.0 / 95.0))
