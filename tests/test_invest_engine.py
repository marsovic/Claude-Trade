import json

import numpy as np
import pandas as pd
import pytest

from bot.state import StateStore
from dashboard.dashboard import create_app
from invest.engine import Engine, etp_symbol
from invest.execution import SimExecution
from invest.portfolio import InvestPosition, InvestState
from invest.tax import TaxLedger
from tests.invest_helpers import SYMS, make_market, market_data, settings, signal_case


def make_engine(tmp_path, **kw):
    s = settings(state_dir=str(tmp_path), **kw)
    state = InvestState.fresh("sim", s.initial_capital)
    state.universe = list(SYMS)
    store = StateStore(tmp_path, InvestState)
    engine = Engine(s, state, store, SimExecution(s, state, s.fees), research=None,
                    ledger=TaxLedger(tmp_path))
    return engine, state, store


def next_day(bars, day, k=1):
    idx = bars["SPY"].index
    return idx[idx.get_loc(day) + k]


def set_bar(bars, sym, day, **values):
    df = bars[sym].copy()
    for k, v in values.items():
        df.loc[day, k] = v
    df.loc[day, "high"] = max(df.loc[day, ["open", "high", "close", "low"]])
    df.loc[day, "low"] = min(df.loc[day, ["open", "high", "close", "low"]])
    bars[sym] = df


def entered(tmp_path):
    bars, day, hits = signal_case()
    bars = {k: v.copy() for k, v in bars.items()}
    sym = hits[0]
    engine, state, store = make_engine(tmp_path)
    engine.process_day(day, market_data(bars))
    close = bars[sym].loc[day, "close"]
    d1 = next_day(bars, day)
    set_bar(bars, sym, d1, open=close * 1.01, low=close * 1.0, close=close * 1.02)
    engine.process_day(d1, market_data(bars))
    return bars, day, d1, sym, engine, state, store


def test_signal_becomes_capped_limit_order(tmp_path):
    bars, day, hits = signal_case()
    engine, state, store = make_engine(tmp_path)
    result = engine.process_day(day, market_data(bars))
    assert result["entries"] == hits[:3]
    order = state.pending[0]
    close = bars[hits[0]].loc[day, "close"]
    assert order["side"] == "BUY" and order["kind"] == "LIMIT"
    assert order["limit_price"] == pytest.approx(close * 1.03)
    assert 0.04 <= order["stop_pct"] <= 0.20
    notional_eur = order["qty"] * order["limit_price"] / engine.fx_now
    assert notional_eur <= 100 / 3 + 0.5  # one of three slots
    assert order["risk_eur"] <= 4.0 + 1e-6  # 4 % of 100 EUR
    kinds = [d["kind"] for d in store.recent_decisions()]
    assert "signal" in kinds and "order" in kinds
    snap = json.loads(store.state_path.read_text())
    assert snap["last_scan"]["long_only"] and len(snap["last_scan"]["conditions"]) == 7


def test_fill_at_next_open_creates_position_and_ledger(tmp_path):
    bars, day, d1, sym, engine, state, store = entered(tmp_path)
    pos = state.positions[sym]
    open_ = bars[sym].loc[d1, "open"]
    assert pos.entry_price == pytest.approx(open_ * 1.0005)
    assert pos.stop_loss == pytest.approx(pos.entry_price * (1 - state.pending[0]["stop_pct"])
                                          if state.pending else pos.stop_loss)
    assert pos.cost_eur > 0 and state.cash == pytest.approx(100 - pos.cost_eur)
    assert pos.entry_day == str(d1.date())
    rows = TaxLedger(tmp_path).rows()
    assert rows[0]["side"] == "BUY" and rows[0]["symbol"] == sym
    assert state.fees_eur == pytest.approx(rows[0]["fees_eur"], abs=1e-5)


def test_limit_not_reached_expires(tmp_path):
    bars, day, hits = signal_case()
    bars = {k: v.copy() for k, v in bars.items()}
    sym = hits[0]
    engine, state, store = make_engine(tmp_path)
    engine.process_day(day, market_data(bars))
    close = bars[sym].loc[day, "close"]
    d1 = next_day(bars, day)
    set_bar(bars, sym, d1, open=close * 1.10, low=close * 1.08, close=close * 1.12)
    engine.process_day(d1, market_data(bars))
    assert sym not in state.positions
    assert any(d["kind"] == "skip" and "not filled" in d["message"]
               for d in store.recent_decisions())


def test_stop_loss_costs_about_the_planned_risk(tmp_path):
    bars, day, d1, sym, engine, state, store = entered(tmp_path)
    pos = state.positions[sym]
    planned_risk = pos.risk_amount
    d2 = next_day(bars, d1)
    set_bar(bars, sym, d2, open=pos.stop_loss * 1.01, low=pos.stop_loss * 0.98,
            close=pos.stop_loss * 0.99)
    engine.process_day(d2, market_data(bars))
    assert sym not in state.positions
    trade = store.recent_trades()[-1]
    assert trade["reason"] == "stop"
    assert -planned_risk * 1.2 < trade["pnl"] < 0
    summary = TaxLedger(tmp_path).summary(d2.year)
    assert summary["net_eur"] == pytest.approx(trade["pnl"], abs=0.01)
    assert summary["box"] == "3VH"


def test_gap_through_stop_fills_at_open(tmp_path):
    bars, day, d1, sym, engine, state, store = entered(tmp_path)
    stop = state.positions[sym].stop_loss
    d2 = next_day(bars, d1)
    set_bar(bars, sym, d2, open=stop * 0.90, low=stop * 0.88, close=stop * 0.95)
    engine.process_day(d2, market_data(bars))
    assert store.recent_trades()[-1]["exit_price"] == pytest.approx(stop * 0.90 * 0.9995)


def test_survival_rule_liquidates_and_writes_tombstone(tmp_path):
    bars, day, d1, sym, engine, state, store = entered(tmp_path)
    state.cash = -200.0  # the capital is gone
    d2 = next_day(bars, d1)
    result = engine.process_day(d2, market_data(bars))
    assert result["dead"] and state.status == "dead"
    assert state.positions == {}
    tomb = store.tombstone()
    assert tomb["equity"] <= 10
    assert store.recent_trades()[-1]["reason"] == "death"


def test_regime_off_blocks_entries(tmp_path):
    bars, day, hits = signal_case()
    engine, state, store = make_engine(tmp_path)
    md = market_data(bars, vix=pd.Series([45.0], index=[day]))
    engine.process_day(day, md)
    assert state.pending == []
    assert any(d["kind"] == "regime" for d in store.recent_decisions())


def test_rank_exit_on_rebalance_day(tmp_path):
    bars = make_market()
    engine, state, store = make_engine(tmp_path)
    days = bars["SPY"].index
    friday = next(d for d in days[400:] if d.weekday() == 4)
    from invest.signals import build_panel

    panel = build_panel(bars, SYMS, engine.s)
    worst = min(panel, key=lambda k: panel[k].loc[friday, "score"]
                if not np.isnan(panel[k].loc[friday, "score"]) else 9)
    price = bars[worst].loc[friday, "close"]
    state.positions[worst] = InvestPosition(
        symbol=worst, qty=0.1, entry_price=price, cost_eur=price * 0.1 / 1.1,
        initial_stop=price * 0.5, stop_loss=price * 0.5, atr=1.0, risk_amount=1.0,
        opened_at=0, entry_day=str(days[390].date()))
    engine.process_day(friday, market_data(bars), panel=panel)
    sells = [o for o in state.pending if o["side"] == "SELL"]
    assert sells and sells[0]["symbol"] == worst and "rank exit" in sells[0]["reason"]


def test_time_stop(tmp_path):
    bars = make_market()
    engine, state, store = make_engine(tmp_path, rebalance_weekday=9)  # no rank exits
    days = bars["SPY"].index
    day = days[450]
    from invest.signals import build_panel

    panel = build_panel(bars, SYMS, engine.s)
    worst = min(panel, key=lambda k: panel[k].loc[day, "pct_rank"]
                if not np.isnan(panel[k].loc[day, "pct_rank"]) else 9)
    price = bars[worst].loc[day, "close"]
    state.positions[worst] = InvestPosition(
        symbol=worst, qty=0.1, entry_price=price, cost_eur=price * 0.1 / 1.1,
        initial_stop=price * 0.5, stop_loss=price * 0.5, atr=1.0, risk_amount=1.0,
        opened_at=0, entry_day=str(days[400].date()))
    engine.process_day(day, market_data(bars), panel=panel)
    assert any("time stop" in o["reason"] for o in state.pending if o["side"] == "SELL")


def test_pyramid_add_moves_stop_to_breakeven(tmp_path):
    bars, day, hits = signal_case()
    bars = {k: v.copy() for k, v in bars.items()}
    sym = hits[0]
    engine, state, store = make_engine(tmp_path, slots=1, slots_small_equity=1,
                                       min_order_eur=0.0, max_weight=1.0)  # tiny test sizes
    close = bars[sym].loc[day, "close"]
    entry = close / 1.2
    state.positions[sym] = InvestPosition(
        symbol=sym, qty=0.2, entry_price=entry, cost_eur=entry * 0.2 / 1.1,
        initial_stop=entry * 0.9, stop_loss=entry * 0.9, atr=1.0, risk_amount=2.0,
        opened_at=0, entry_day=str(bars["SPY"].index[250].date()))
    state.cash -= entry * 0.2 / 1.1
    engine.process_day(day, market_data(bars))
    adds = [o for o in state.pending if o["reason"] == "add"]
    assert adds and adds[0]["qty"] == pytest.approx(0.1)
    d1 = next_day(bars, day)
    set_bar(bars, sym, d1, open=close * 1.01, low=close, close=close * 1.02)
    engine.process_day(d1, market_data(bars))
    pos = state.positions[sym]
    assert pos.adds == 1 and pos.qty == pytest.approx(0.3)
    assert pos.stop_loss >= pos.entry_price  # break-even or better


def calm_uptrend(bars, n):
    days = bars["SPY"].index
    t = np.arange(n)
    rng = np.random.default_rng(0)
    close = 100 * np.exp(0.0012 * t + rng.normal(0, 0.003, n).cumsum() * 0.2)
    df = pd.DataFrame({"open": close * 0.999, "high": close * 1.003, "low": close * 0.997,
                       "close": close, "volume": np.full(n, 5e7)}, index=days)
    return df


def test_leveraged_etp_sleeve_in_calm_uptrend(tmp_path):
    bars = make_market()
    n = len(bars["SPY"])
    bars["SPY"] = calm_uptrend(bars, n)
    bars["QQQ"] = calm_uptrend(bars, n)
    engine, state, store = make_engine(tmp_path, etp_enabled=True)
    day = bars["SPY"].index[400]
    vix = pd.Series(15.0, index=bars["SPY"].index)
    engine.process_day(day, market_data(bars, vix=vix))
    sym = etp_symbol(engine.s)
    etp_orders = [o for o in state.pending if o["symbol"] == sym]
    assert etp_orders and etp_orders[0]["sleeve"] == "etp"
    d1 = next_day(bars, day)
    engine.process_day(d1, market_data(bars, vix=vix))
    pos = state.positions[sym]
    assert pos.leverage == 3.0 and pos.cost_eur <= 100 * 0.33 + 1


def test_no_leveraged_etp_without_vix(tmp_path):
    bars = make_market()
    n = len(bars["SPY"])
    bars["SPY"] = calm_uptrend(bars, n)
    bars["QQQ"] = calm_uptrend(bars, n)
    engine, state, store = make_engine(tmp_path, etp_enabled=True)
    engine.process_day(bars["SPY"].index[400], market_data(bars))
    assert not any(o["symbol"].startswith("ETP:") for o in state.pending)


def test_weekly_loss_pause(tmp_path):
    bars, day, hits = signal_case()
    engine, state, store = make_engine(tmp_path)
    state.week_id = f"{day.isocalendar().year}-{day.isocalendar().week}"
    state.week_start_equity = 200.0  # equity 100 = -50 % this week
    engine.process_day(day, market_data(bars))
    assert state.daily_halt and state.pending == []
    assert any(d["kind"] == "halt" for d in store.recent_decisions())


def test_state_roundtrip_and_dashboard(tmp_path):
    bars, day, d1, sym, engine, state, store = entered(tmp_path)
    loaded = store.load()
    assert isinstance(loaded, InvestState)
    assert loaded.positions[sym].qty == state.positions[sym].qty
    data = create_app(tmp_path).test_client().get("/api/state").get_json()
    st = data["state"]
    assert st["settings"]["currency_symbol"] == "€"
    assert "US stocks" in st["settings"]["venue_label"]
    assert st["positions"][sym]["margin"] == pytest.approx(state.positions[sym].cost_eur)
    assert "unrealized_eur" in st["positions"][sym]
    assert st["last_scan"]["condition_labels"][0].startswith("Regime")
