"""Changes after the third review of the invest bot (engine audit by independent reviewers)."""

import pandas as pd

from invest.backtest import placebo_summary, run_backtest, run_placebo
from invest.config import InvestSettings
from invest.portfolio import InvestPosition
from invest.risk import chandelier_stop, risk_budget, size_position, stop_reason
from invest.signals import StockEval, build_panel
from invest.universe import CURATED_US, THEMES, theme_of
from tests.invest_helpers import SYMS, make_market, market_data, settings, signal_case
from tests.test_invest_engine import make_engine, next_day, set_bar


def ev(sym, day, score=1.0):
    return StockEval(symbol=sym, day=day, price=50.0, atr=2.0, atr_pct=4.0, score=score,
                     pct_rank=0.99, eligible=True, checks={"regime": True}, values={"rsi": 60.0})


def enter(engine, state, day, evals):
    budget = risk_budget(state.cash, 0.0, engine.s)
    return engine._enter_stocks(day, {e.symbol: e for e in evals}, {}, budget, state.cash, {},
                                set())


# ------------------------------------------------------------------ defaults
def test_defaults_follow_the_audit():
    s = InvestSettings()
    assert s.min_history == 260  # 12-1 momentum needs 253 bars: no dead first two months
    assert s.weekly_loss_pause < s.max_open_risk  # reachable: about two stops
    assert s.etp_enabled is False and s.max_per_theme == 1 and s.min_order_eur >= 5
    assert s.trail_atr_mult == 0.0  # same as the initial stop until validated


def test_every_curated_stock_has_a_theme():
    assert [x for x in CURATED_US if theme_of(x) is None] == []
    assert theme_of("cohr") == "ai_chips_hardware" and theme_of("NOPE") is None
    assert sum(len(v) for v in THEMES.values()) == len({s for v in THEMES.values() for s in v})


def test_first_session_with_eligible_stocks_is_not_delayed():
    bars = make_market(n_days=520)
    s = settings()
    panel = build_panel(bars, SYMS, s)
    eligible_days = [d for d in bars["SPY"].index
                     if any(f.loc[d, "eligible"] for f in panel.values() if d in f.index)]
    assert bars["SPY"].index.get_loc(eligible_days[0]) <= s.regime_sma + 60  # sim start


# ------------------------------------------------------------------- labels
def test_stop_exits_are_named_by_outcome():
    assert stop_reason(fill=105.0, stop=95.0, entry=100.0) == "trailing stop"
    assert stop_reason(fill=95.0, stop=95.0, entry=100.0) == "stop"
    assert stop_reason(fill=80.0, stop=95.0, entry=100.0) == "stop (gap)"
    # a stop that stood above entry but was jumped over for a loss is not "trailing"
    assert stop_reason(fill=96.0, stop=101.0, entry=100.0) == "stop (gap)"


# --------------------------------------------------------------- theme cap
def test_one_position_per_theme(tmp_path):
    bars = make_market()
    day = bars["SPY"].index[-1]
    engine, state, _ = make_engine(tmp_path, slots=3, slots_small_equity=3)
    # COHR, LRCX and MRVL are all AI chips; JPM is a financial
    entered = enter(engine, state, day, [ev("COHR", day, 3), ev("LRCX", day, 2),
                                         ev("MRVL", day, 1.5), ev("JPM", day, 1)])
    assert entered == ["COHR", "JPM"]
    journal = [d["message"] for d in engine.store.recent_decisions() if d["kind"] == "skip"]
    assert sum("ai_chips_hardware theme" in m for m in journal) == 2


def test_theme_cap_counts_open_positions_and_can_be_disabled(tmp_path):
    bars = make_market()
    day = bars["SPY"].index[-1]
    engine, state, _ = make_engine(tmp_path, slots=3, slots_small_equity=3)
    state.positions["AMD"] = InvestPosition(
        symbol="AMD", qty=0.1, entry_price=100.0, cost_eur=9.0, initial_stop=90.0,
        stop_loss=90.0, atr=3.0, risk_amount=1.0, opened_at=0, entry_day="2026-01-02")
    assert enter(engine, state, day, [ev("NVDA", day)]) == []  # a chip is already held
    off, state2, _ = make_engine(tmp_path / "off", slots=3, slots_small_equity=3,
                                 max_per_theme=0)
    assert enter(off, state2, day, [ev("COHR", day, 3), ev("LRCX", day, 2)]) == ["COHR",
                                                                                "LRCX"]


# -------------------------------------------------------------------- sizing
def test_dust_orders_are_refused():
    s = settings()
    budget = risk_budget(100.0, 0.0, s)
    sized, why = size_position(50.0, 2.0, 100.0, 5.0, 0.0, budget, 1.1, s.fees, s)
    assert sized is None and "minimum" in why  # 5 EUR of cash would be dust
    ok, _ = size_position(50.0, 2.0, 100.0, 100.0, 0.0, budget, 1.1, s.fees, s)
    assert ok is not None and ok.notional_eur >= s.min_order_eur


def test_separate_trailing_multiple():
    pos = InvestPosition(symbol="X", qty=1, entry_price=100.0, cost_eur=90.0,
                         initial_stop=90.0, stop_loss=90.0, atr=2.0, risk_amount=1.0,
                         opened_at=0, entry_day="2026-01-02", highest=150.0)
    tight = chandelier_stop(pos, 5.0, settings(), mult=3.0)
    wide = chandelier_stop(pos, 5.0, settings(trail_atr_mult=4.0), mult=4.0)
    assert wide < tight  # 4 ATR leaves more room than 3 ATR


# ------------------------------------------------------------------ pyramids
def pyramid_setup(tmp_path, qty, **kw):
    bars, day, hits = signal_case()
    bars = {k: v.copy() for k, v in bars.items()}
    sym = hits[0]
    engine, state, store = make_engine(tmp_path, slots=1, slots_small_equity=1,
                                       min_order_eur=0.0, **kw)
    close = bars[sym].loc[day, "close"]
    entry = close / 1.2
    state.positions[sym] = InvestPosition(
        symbol=sym, qty=qty, entry_price=entry, cost_eur=entry * qty / 1.1,
        initial_stop=entry * 0.9, stop_loss=entry * 0.9, atr=1.0, risk_amount=2.0,
        opened_at=0, entry_day=str(bars["SPY"].index[250].date()))
    state.cash -= entry * qty / 1.1
    return bars, day, sym, close, engine, state


def test_pyramid_never_pushes_a_position_above_the_weight_cap(tmp_path):
    # the position already weighs more than max_weight (40 %): no add
    bars, day, sym, close, engine, state = pyramid_setup(tmp_path, qty=0.2)
    engine.process_day(day, market_data(bars))
    assert [o for o in state.pending if o["reason"] == "add"] == []


def test_pyramid_is_trimmed_to_the_room_left_under_the_cap(tmp_path):
    bars, day, sym, close, engine, state = pyramid_setup(tmp_path, qty=0.12)
    engine.process_day(day, market_data(bars))
    (add,) = [o for o in state.pending if o["reason"] == "add"]
    room_qty = (engine.s.max_weight * engine.equity() - 0.12 * close / 1.1) * 1.1 / close
    assert 0 < add["qty"] <= room_qty + 1e-9 and add["qty"] < 0.06  # trimmed from 50 % (0.06)


def test_risk_of_an_add_counts_in_r(tmp_path):
    bars, day, sym, close, engine, state = pyramid_setup(tmp_path, qty=0.2, max_weight=1.0)
    engine.process_day(day, market_data(bars))
    before = state.positions[sym].risk_amount
    d1 = next_day(bars, day)
    set_bar(bars, sym, d1, open=close * 1.01, low=close, close=close * 1.02)
    engine.process_day(d1, market_data(bars))
    assert state.positions[sym].adds == 1
    assert state.positions[sym].risk_amount > before  # R is measured on everything at risk


# ------------------------------------------------------ backtest instruments
def test_backtest_reports_an_equal_weight_baseline_and_a_placebo(tmp_path):
    bars = make_market(n_days=520)
    s = settings()
    real = run_backtest(bars, s, out_dir=tmp_path / "real")
    st = real.stats()
    assert st["benchmark_equal_weight_universe_final_eur"] > 0
    raw = run_placebo(bars, s, n_entries=max(st["trades"], 3), runs=4, seed=1,
                      out_dir=tmp_path / "placebo")
    assert raw["runs"] == 4 and len(raw["results"]) == 4
    summary = placebo_summary(st, raw)
    assert 0 <= summary["real_beats_pct_of_random_runs"] <= 100
    assert set(summary["random_return_pct"]) == {"median", "p10", "p90"}
    # same seed, same result: the placebo is reproducible
    again = run_placebo(bars, s, n_entries=max(st["trades"], 3), runs=4, seed=1,
                        out_dir=tmp_path / "placebo2")
    assert [r["return_pct"] for r in again["results"]] == [r["return_pct"]
                                                          for r in raw["results"]]


def test_placebo_entries_do_not_need_signals(tmp_path):
    bars = make_market(n_days=520)
    s = settings(top_pct_entry=0.0001, near_high_min=2.0)  # no real signal can ever fire
    assert run_backtest(bars, s, out_dir=tmp_path / "a").stats()["trades"] == 0
    raw = run_placebo(bars, s, n_entries=15, runs=2, seed=3, out_dir=tmp_path / "b")
    assert sum(r["trades"] for r in raw["results"]) > 0
    assert isinstance(pd.Timestamp("2026-01-01"), pd.Timestamp)
