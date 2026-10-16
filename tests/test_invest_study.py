"""Tests of the signal-level walk-forward study (invest/study.py).

The most important one is ``test_trades_match_the_real_engine``: the study re-implements the
exit logic of ``Engine.process_day`` + ``SimBroker`` on whole arrays. If the engine's exit
logic changes (stops, slippage, entry-day high, rank exit...), that test fails and
``study.simulate`` must follow.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from invest import study
from invest.backtest import run_backtest
from invest.config import InvestSettings
from invest.risk import stop_distance_pct
from invest.signals import build_panel, regime_at
from invest.study import (
    BASELINE,
    EventSet,
    Grid,
    LeakError,
    Params,
    StudyError,
    Variant,
)
from tests.invest_helpers import SYMS, make_market, ohlcv, settings

FAST = Params(n_boot=300, placebo_draws=60, audit_days=9, min_design_events=5)
FASTER = dataclasses.replace(FAST, audit="off")  # mirror only: for tests that need no audit
# baseline only (no variants), no placebo: enough to compare the trades with the engine
BASELINE_ONLY = Params(n_boot=50, placebo_draws=0, audit="off", min_design_events=0)


# ------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def market():
    return make_market(n_days=1700, seed=3, start="2017-01-02")


@pytest.fixture(scope="module")
def s_base():
    return settings(universe=SYMS)


@pytest.fixture(scope="module")
def result(market, s_base):
    return study.run_study(market, s_base, params=FAST)


def crafted_grid(rows, atr: float = 2.0) -> Grid:
    """One stock 'X' with hand-written (open, high, low, close) rows and a constant ATR."""
    n = len(rows)
    cal = pd.bdate_range("2024-01-02", periods=n)
    A = {k: np.full((n, 1), np.nan) for k in study.GRID_COLS}
    for i, (o, h, lo, c) in enumerate(rows):
        A["open"][i, 0], A["high"][i, 0], A["low"][i, 0], A["close"][i, 0] = o, h, lo, c
    A["atr"][:, 0] = atr
    A["pct_rank_hold"][:, 0] = 0.99  # always in the top: no rank exit
    A["r6_1"][:, 0] = 0.5
    return Grid(cal=cal, syms=["X"], A=A, eligible=np.ones((n, 1), dtype=bool))


def flat(n: int, price: float):
    return [(price, price + 0.1, price - 0.1, price)] * n


def run_one(rows, s, t0: int = 2):
    g = crafted_grid(rows)
    off = np.zeros(g.T, dtype=bool)
    return study.simulate(g, np.array([t0]), np.array([0]), s, off, horizon=250)


# ----------------------------------------------------------- crafted single trades
def test_gap_through_stop_gives_the_expected_r_and_reason():
    s = settings(fee_preset="alpaca")  # zero fees: exact arithmetic
    rows = (flat(2, 100) + [(100, 100.2, 99.8, 100)]            # t0 = 2: signal close 100
            + [(100.5, 102.0, 100.2, 101.5)]                     # entry day
            + [(90.0, 91.0, 88.0, 89.0)]                         # gap far below the stop
            + flat(6, 89))
    r = run_one(rows, s)
    slip = s.slippage_bps / 1e4
    fill = 100.5 * (1 + slip)
    sp = min(max(3 * 2.0 / (100 * 1.03), 0.04), 0.20)
    # entry-day high 102 lifts the chandelier to 102 - 3 ATR = 96
    stop = 102.0 - 6.0
    stop_slip = max(slip, getattr(s, "stop_slippage_atr", 0.0) * 2.0 / stop)
    px = 90.0 * (1 - stop_slip)
    assert r.filled[0] and r.kind[0] == 0
    assert r.fill[0] == pytest.approx(fill)
    assert r.sp[0] == pytest.approx(sp)
    assert r.exit_t[0] == 4
    assert r.exit_px[0] == pytest.approx(px)
    assert r.reason[0] == "stop (gap)"
    expected_r = (px / fill - 1) / sp
    assert expected_r < -1.5  # a gap loses more than the planned 1 R
    assert r.net_ret[0] / r.sp[0] == pytest.approx(expected_r)


def test_ordinary_stop_and_trailing_stop_reasons():
    s = settings(fee_preset="alpaca")
    entry = flat(2, 100) + [(100, 100.2, 99.8, 100), (100.5, 102.0, 100.2, 101.5)]
    r = run_one(entry + [(99.0, 99.5, 95.0, 96.0)] + flat(5, 96), s)  # touches 96, no gap
    assert r.reason[0] == "stop" and r.exit_t[0] == 4
    # the same stop after a rally above the entry is a trailing stop
    rally = entry + [(101.6, 112.0, 101.5, 111.0), (110.0, 111.0, 100.0, 105.0)] + flat(4, 105)
    r2 = run_one(rally, s)
    assert r2.reason[0] == "trailing stop" and r2.exit_px[0] > r2.fill[0]


def test_limit_fill_after_a_gap_up_does_not_inherit_the_morning_high():
    s = settings(fee_preset="alpaca")
    sig = [(100, 100.2, 99.8, 100)]
    # open 105 is above the limit 103: filled at the limit after the high of 110 was printed
    gap = flat(2, 100) + sig + [(105.0, 110.0, 102.0, 103.5), (103.8, 104.0, 103.0, 103.5)]
    r = run_one(gap + flat(6, 103.5), s)
    assert r.kind[0] == 1 and r.fill[0] == pytest.approx(103.0)  # the limit, no slippage
    # the chandelier starts from max(fill, close) = 103.5, not from 110: no stop at 104
    assert r.exit_t[0] > 4 and r.reason[0].startswith("open")
    # an open fill lives the whole day: the 110 high counts and the stop is lifted to 104
    openfill = flat(2, 100) + sig + [(101.0, 110.0, 100.5, 109.0), (105.0, 105.5, 103.9, 104.5)]
    r2 = run_one(openfill + flat(5, 104.5), s)
    assert r2.kind[0] == 0
    assert r2.exit_t[0] == 4 and r2.reason[0] == "trailing stop"


def test_unfilled_when_the_gap_is_beyond_the_limit():
    s = settings(fee_preset="alpaca")
    rows = flat(2, 100) + [(100, 100.2, 99.8, 100), (106.0, 108.0, 105.0, 107.0)] + flat(5, 107)
    r = run_one(rows, s)  # open and low both above the 103 limit
    assert not r.filled[0] and np.isnan(r.net_ret[0])


def test_r_is_net_of_fees_and_slippage():
    rows = (flat(2, 100) + [(100, 100.2, 99.8, 100), (101.0, 110.0, 100.5, 109.0),
                            (105.0, 105.5, 103.9, 104.5)] + flat(5, 104.5))
    free = run_one(rows, settings(fee_preset="alpaca", slippage_bps=0.0))
    paid = run_one(rows, settings(fee_preset="sim", slippage_bps=0.0))
    rate = InvestSettings(fee_preset="sim").fees.fx_fee_pct
    assert rate > 0
    gap = (free.net_ret[0] - paid.net_ret[0])
    assert gap == pytest.approx(rate * (1 + paid.exit_px[0] / paid.fill[0]))
    assert (free.net_ret[0] / free.sp[0]) > (paid.net_ret[0] / paid.sp[0])  # R is net
    slipped = run_one(rows, settings(fee_preset="alpaca", slippage_bps=50.0))
    assert slipped.net_ret[0] < free.net_ret[0]


def test_horizon_and_end_of_data_are_marked_with_exit_costs():
    s = settings(fee_preset="alpaca")
    rows = flat(2, 100) + [(100, 100.2, 99.8, 100), (100.5, 100.8, 100.2, 100.6)] + flat(30, 100.6)
    g = crafted_grid(rows)
    off = np.zeros(g.T, dtype=bool)
    r = study.simulate(g, np.array([2]), np.array([0]), s, off, horizon=5)
    assert r.reason[0] == "horizon cap" and r.exit_t[0] == 3 + 5
    r2 = study.simulate(g, np.array([2]), np.array([0]), s, off, horizon=250)
    assert r2.reason[0] == "open (end of data)" and r2.exit_t[0] == g.T - 1


# ----------------------------------------------- the study mirrors the real engine
def _engine_settings(**kw):
    base = dict(universe=SYMS, pyramid=False, slots=1, slots_small_equity=1, max_per_theme=0,
                initial_capital=1000.0, small_equity=0.0, max_weight=1.0, max_open_risk=1.0,
                weekly_loss_pause=0.99, dd_level1=0.99, dd_level2=0.999)
    base.update(kw)
    return settings(**base)


def _gappy(seed: int, n: int = 1500, gap_sd: float = 0.02):
    bars = make_market(n_days=n, seed=seed, start="2017-01-02")
    rng = np.random.default_rng(seed + 100)
    out = {}
    for sym, df in bars.items():
        df = df.copy()
        prev = df["close"].shift(1).fillna(df["open"])
        df["open"] = prev * np.exp(rng.normal(0, gap_sd, len(df)))
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
        out[sym] = df
    return out


def _compare_with_engine(bars, s, events: pd.DataFrame, out_dir) -> tuple[int, list, dict]:
    """Every trade of the real backtest must be one of the study's raw baseline events with
    the same entry, exit price, exit day and R."""
    bt = run_backtest(bars, s, out_dir=out_dir)
    cal = bars["SPY"].index
    nxt = {d: cal[i + 1] for i, d in enumerate(cal[:-1])}
    ev = events[events["filled"]]
    entry = [nxt[pd.Timestamp(x)].date() for x in ev["signal_day"]]
    ev = ev.assign(entry=entry)
    bad, kinds = [], {}
    for t in bt.trades:
        d = pd.to_datetime(t["opened_at"], unit="ms", utc=True).tz_convert("America/New_York")
        x = pd.to_datetime(t["closed_at"], unit="ms", utc=True).tz_convert("America/New_York")
        hit = ev[(ev["symbol"] == t["symbol"]) & (ev["entry"] == d.date())]
        if hit.empty:
            bad.append(("no event", t["symbol"], d.date()))
            continue
        e = hit.iloc[0]
        kinds[e["fill_kind"]] = kinds.get(e["fill_kind"], 0) + 1
        ok = (abs(e["fill_price"] / t["entry_price"] - 1) < 1e-9
              and abs(e["exit_price"] / t["exit_price"] - 1) < 1e-9
              and e["exit_day"] == str(x.date()) and abs(e["R"] - t["r_multiple"]) < 2e-3)
        if not ok:
            bad.append((t["symbol"], d.date(), t["exit_price"], e["exit_price"], t["r_multiple"],
                        e["R"]))
    return len(bt.trades), bad, kinds


def test_trades_match_the_real_engine(tmp_path):
    bars = make_market(n_days=1100, seed=3, start="2017-01-02")
    s = _engine_settings()
    res = study.run_study(bars, s, variants=[], params=BASELINE_ONLY)
    n, bad, _ = _compare_with_engine(bars, s, res.events, tmp_path)
    assert n >= 10, "the synthetic market should give a dozen trades at least"
    assert not bad, f"study.simulate no longer mirrors the engine: {bad[:3]}"


def test_trades_match_the_engine_with_gaps_time_stops_and_other_exits(tmp_path):
    """Gappy opens: limit fills after a gap-up, gap-through-stops, time stops, rank exits."""
    bars = _gappy(7, n=1100)
    s = _engine_settings(time_stop_days=8, time_stop_min_r=2.0, stop_atr_mult=2.5,
                         trail_atr_mult=4.0)
    res = study.run_study(bars, s, variants=[], params=BASELINE_ONLY)
    n, bad, kinds = _compare_with_engine(bars, s, res.events, tmp_path)
    assert n >= 10 and not bad, bad[:3]
    reasons = set(res.events["exit_reason"])
    assert any(r.startswith("time stop") for r in reasons)
    assert "limit_after_gap" in set(res.events["fill_kind"])


def test_regime_off_tightening_matches_the_engine(monkeypatch, tmp_path):
    """Force 'regime off' on random sessions in BOTH the engine and the study."""
    import invest.engine as engine_mod

    bars = make_market(n_days=1100, seed=5, start="2017-01-02")
    cal = bars["SPY"].index
    rng = np.random.default_rng(0)
    forced_days = set(cal[rng.random(len(cal)) < 0.25])
    real = regime_at

    def forced(day, bars_, vix, vix3m, s_):
        reg = real(day, bars_, vix, vix3m, s_)
        return dataclasses.replace(reg, off=True) if day in forced_days else reg

    monkeypatch.setattr(engine_mod, "regime_at", forced)
    monkeypatch.setattr(study, "regime_at", forced)
    s = _engine_settings(regime_off_stop_mult=0.8)
    res = study.run_study(bars, s, variants=[], params=BASELINE_ONLY)
    n, bad, _ = _compare_with_engine(bars, s, res.events, tmp_path)
    assert n >= 10 and not bad, bad[:3]


# ------------------------------------------------------------------ signals mirror
def test_signal_mirror_equals_evaluate_day_for_variants_too(market, s_base):
    panel = build_panel(market, SYMS, s_base)
    cal = market["SPY"].index
    g = study.build_grid(panel, cal)
    regs = study.regime_list(market, pd.Series(dtype=float), pd.Series(dtype=float), s_base, cal)
    on, _ = study.regime_arrays(regs)
    days = list(range(s_base.regime_sma + 60, g.T, 7))
    for kw in ({}, {"top_pct_entry": 0.2, "volume_factor": 1.0}, {"near_high_min": 0.95},
               {"max_extension": 0.1, "rsi_max": 60.0}):
        sv = dataclasses.replace(s_base, **kw)
        m = study.signal_matrix(g, on, sv)
        assert study.audit_signals(panel, g, regs, sv, m, days, str(kw)) > 0
    # a doctored mirror must be caught
    m = study.signal_matrix(g, on, s_base)
    hit = np.argwhere(m[days])[0] if m[days].any() else None
    if hit is not None:
        m = m.copy()
        m[days[hit[0]], hit[1]] = False
        with pytest.raises(StudyError):
            study.audit_signals(panel, g, regs, s_base, m, days, "doctored")


def test_events_are_every_signal_and_the_baseline_equals_evaluate_day(result):
    ev = result.events
    assert result.baseline["raw_signals"] == len(ev) > 30
    counted = ev[ev["counted"]]
    assert result.baseline["events"] == len(counted)
    assert len(counted) <= len(ev)
    assert (counted["filled"]).all()
    assert result.audit["baseline_mismatches"] == 0


# -------------------------------------------------------------------- features
def test_gap_extension_event_bar_and_discreteness_features():
    n = 400
    rows = [(100.0, 100.5, 99.5, 100.0)] * n
    rows = [list(r) for r in rows]
    # a news bar on session 300: opens +10 (5 ATR) on a volume spike
    rows[300] = [110.0, 111.0, 109.0, 110.5]
    g = crafted_grid(rows, atr=2.0)
    g.A["ema20"][:, 0] = 98.0
    g.A["vol_ratio"][:, 0] = 1.0
    g.A["vol_ratio"][300, 0] = 3.0
    t0 = np.array([300, 302, 340])
    j = np.zeros(3, dtype=int)
    f = study.candidate_features(g, t0, j)
    assert f["gap_atr"][2] == pytest.approx((100.0 - 100.0) / 2.0)  # flat open
    assert f["gap_atr"][0] == pytest.approx((100.0 - 110.5) / 2.0)  # next open vs the signal close
    assert f["ext_ema20_atr"][2] == pytest.approx((100.0 - 98.0) / 2.0)
    # event bar: |110 - 100| / 2 = 5 ATR >= 2 and volume 3x: flagged for 3 sessions
    assert f["event_bar_1"].tolist() == [True, False, False]
    assert f["event_bar_3"].tolist() == [True, True, False]
    assert f["event_bar_5"].tolist() == [True, True, False]


def test_information_discreteness_matches_its_definition():
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, (400, 2)), axis=0))
    t0 = np.array([300, 350, 399])
    j = np.array([0, 1, 0])
    got = study.information_discreteness(c, t0, j)
    for k in range(3):
        lr = np.diff(np.log(c[:, j[k]]))  # lr[i] = return of session i + 1
        w = lr[max(0, t0[k] - 252 - 1): t0[k] - 21]  # sessions t-252 .. t-21
        want = np.sign(w.sum()) * ((w < 0).mean() - (w > 0).mean())
        assert got[k] == pytest.approx(want)


# ----------------------------------------------------------------- statistics
def _weeks(n_weeks: int = 200, per_week: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    weeks = np.repeat(np.arange(n_weeks), per_week)
    wk_effect = rng.normal(0, 0.5, n_weeks)[weeks]  # events of a week move together
    r = wk_effect + rng.normal(0.1, 1.0, len(weeks))
    return weeks, r


def test_block_bootstrap_is_deterministic_for_a_seed():
    weeks, r = _weeks()
    mask = r > np.median(r)
    a = study.block_boot_diff(weeks, r, mask, n_boot=500, seed=11)
    b = study.block_boot_diff(weeks, r, mask, n_boot=500, seed=11)
    c = study.block_boot_diff(weeks, r, mask, n_boot=500, seed=12)
    assert a == b
    assert (a["lo"], a["hi"]) != (c["lo"], c["hi"])
    assert a["diff"] == c["diff"]  # the point estimate does not depend on the seed
    assert a["lo"] <= a["diff"] <= a["hi"]


def test_bootstrap_detects_a_planted_effect_but_not_a_random_variant():
    weeks, r = _weeks(seed=2)
    rng = np.random.default_rng(3)
    # a variant that keeps only the winners of the baseline: a planted effect
    planted = study.block_boot_diff(weeks, r, r > np.percentile(r, 40), n_boot=1000, seed=1)
    assert planted["lo"] > 0 and planted["p"] < 0.01
    # random variants keeping 70% of the events: no effect, the interval covers 0
    misses = 0
    for k in range(20):
        mask = rng.random(len(r)) < 0.7
        out = study.block_boot_diff(weeks, r, mask, n_boot=500, seed=k)
        misses += not (out["lo"] <= 0 <= out["hi"])
    assert misses <= 3  # about 5% expected at the 95% level
    # a variant that is the baseline itself has an exactly null difference
    same = study.block_boot_diff(weeks, r, np.ones(len(r), dtype=bool), n_boot=200, seed=0)
    assert same["diff"] == 0 and same["lo"] == same["hi"] == 0 and same["p"] == 1.0


def test_holm_and_effective_sample_size():
    adj = study.holm_adjust(np.array([0.01, 0.04, 0.03, 0.2]))
    assert adj.tolist() == pytest.approx([0.04, 0.09, 0.09, 0.2])
    rng = np.random.default_rng(0)
    iid = rng.normal(size=600)
    grp = np.repeat(np.arange(200), 3)
    n_eff, rho, _ = study.effective_n(iid, grp)
    assert n_eff > 450 and rho < 0.15
    clustered = np.repeat(rng.normal(size=200), 3) + 0.2 * rng.normal(size=600)
    n_eff2, rho2, m_bar = study.effective_n(clustered, grp)
    assert n_eff2 < 0.5 * 600 and rho2 > 0.8 and m_bar == pytest.approx(3.0)


def test_one_position_per_symbol_chain():
    t0 = np.array([5, 6, 9, 20, 5])
    j = np.array([0, 0, 0, 0, 1])
    exit_t = np.array([8, 12, 15, 25, 7])
    order = np.lexsort((t0, j))
    keep = study.one_position_per_symbol(order, j, t0, exit_t, np.ones(5, dtype=bool))
    # symbol 0: 5 taken (held to 8), 6 and 9? 9 >= 8 is free again, 20 >= 15 free
    assert keep.tolist() == [True, False, True, True, True]


# --------------------------------------------------- decision rule on planted data
def _fake_sets(rng, planted=None, only_year=None, n=600, week_icc=0.3):
    cal = study.make_calendar(pd.bdate_range("2017-01-02", "2024-12-31"))
    L = int(np.searchsorted(cal.yr, 2022))
    t0 = np.sort(rng.integers(0, L - 300, n)).astype(np.int64)
    wk_eff = rng.normal(0, np.sqrt(week_icc), cal.wk.max() + 1)
    base_r = wk_eff[cal.wk[t0]] + rng.normal(0, 1.5, n) + 0.2
    tests = [v for v in study.VARIANTS if v.role == "test"]

    def make(name, mask, r):
        ix = np.nonzero(mask)[0]
        return EventSet(name, t0[ix], np.zeros(len(ix), dtype=np.int64), t0[ix] + 10, r[ix],
                        r[ix], r[ix] * 2, rng.random(len(ix)), np.full(len(ix), 10),
                        np.array([""] * len(ix), dtype=object))

    sets = {BASELINE: make(BASELINE, np.ones(n, dtype=bool), base_r)}
    for v in tests:
        r = base_r + rng.normal(0, 0.3, n)
        if planted and v.name in planted:
            add = planted[v.name]
            r = r + (add if only_year is None else add * (cal.yr[t0] == only_year))
        sets[v.name] = make(v.name, rng.random(n) < rng.uniform(0.7, 1.0), r)
    return sets, cal, L, [v.name for v in tests]


def _families(tests):
    """Adoptable ('better') variants and 'worse' diagnostics are corrected separately."""
    expect = {v.name: v.expect for v in study.VARIANTS}
    return [[n for n in tests if expect[n] == "better"], [n for n in tests if expect[n] == "worse"]]


def _run_rule(rng, planted=None, only_year=None, seed=0):
    sets, cal, L, tests = _fake_sets(rng, planted, only_year)
    p = Params(n_boot=3000)
    s = settings()
    vmap = {v.name: v for v in study.VARIANTS}
    tbl = study.analyse_segment(sets, cal, 0, L, _families(tests), p, seed)
    axis = study.axis_values(study.VARIANTS, s)
    tbl = study.select_on_design(sets, tbl, vmap, axis, p, s, L, L / 252.0)
    return [n for n in tests if bool(tbl.at[n, "design_pass"])], tbl


def test_decision_rule_selects_a_planted_plateau_and_nothing_under_the_null():
    rng = np.random.default_rng(1)
    chosen, _ = _run_rule(rng, planted={"stop_atr_mult=3.5": 0.35, "stop_atr_mult=4": 0.45})
    assert {"stop_atr_mult=3.5", "stop_atr_mult=4"} <= set(chosen)
    for k in range(6):  # no effect anywhere: nothing may be selected
        chosen, tbl = _run_rule(np.random.default_rng(100 + k), seed=k)
        assert chosen == []
        # the correction matters: many raw p-values are small by chance alone
    tests = [v.name for v in study.VARIANTS if v.role == "test"]
    assert (tbl.loc[tests, "p_holm"] >= tbl.loc[tests, "p_raw"]).all()
    assert (tbl.loc[tests, "ci_adj_hi"] - tbl.loc[tests, "ci_adj_lo"]
            >= tbl.loc[tests, "dR_hi"] - tbl.loc[tests, "dR_lo"] - 1e-12).all()


def test_decision_rule_rejects_spikes_and_single_year_effects():
    rng = np.random.default_rng(4)
    # a spike: the effect at 4.0 but the neighbour 3.5 points the other way
    chosen, tbl = _run_rule(rng, planted={"stop_atr_mult=4": 0.6, "stop_atr_mult=3.5": -0.4})
    assert "stop_atr_mult=4" not in chosen
    assert tbl.at["stop_atr_mult=4", "plateau"] == "fail"
    # an effect that lives in a single calendar year is not stable across years
    chosen2, tbl2 = _run_rule(np.random.default_rng(5),
                              planted={"trail_atr_mult=4": 2.0, "trail_atr_mult=5": 2.0},
                              only_year=2019)
    assert "trail_atr_mult=4" not in chosen2 and not tbl2.at["trail_atr_mult=4", "c_years"]


def test_selection_function_refuses_locked_events():
    rng = np.random.default_rng(0)
    sets, cal, L, tests = _fake_sets(rng)
    p, s = Params(n_boot=100), settings()
    vmap = {v.name: v for v in study.VARIANTS}
    tbl = study.analyse_segment(sets, cal, 0, L, _families(tests), p, 0)
    axis = study.axis_values(study.VARIANTS, s)
    leaky = dict(sets)
    leaky[BASELINE] = sets[BASELINE].sel(np.ones(sets[BASELINE].n, dtype=bool))
    leaky[BASELINE].t0[-1] = L + 5  # one LOCKED event
    with pytest.raises(LeakError):
        study.select_on_design(leaky, tbl, vmap, axis, p, s, L, L / 252.0)


# ------------------------------------------------------ the walk-forward protocol
def test_locked_rows_never_reach_the_selection_and_unselected_variants_stay_hidden(
        market, s_base, monkeypatch):
    seen = {"design_calls": [], "analyse": []}
    orig_select, orig_analyse = study.select_on_design, study.analyse_segment

    def spy_select(design_sets, table, variants, axis, params, s, locked_start, n_years):
        seen["design_calls"].append((locked_start, max(
            max(ev.t0.max(), ev.exit_t.max()) for ev in design_sets.values() if ev.n)))
        out = orig_select(design_sets, table, variants, axis, params, s, locked_start, n_years)
        out.loc["stop_atr_mult=4", "design_pass"] = True  # force ONE selection
        return out

    def spy_analyse(sets, cal, a, b, family, params, seed):
        seen["analyse"].append((a, b, sorted(sets), max(ev.t0.max() for ev in sets.values()
                                                         if ev.n)))
        return orig_analyse(sets, cal, a, b, family, params, seed)

    monkeypatch.setattr(study, "select_on_design", spy_select)
    monkeypatch.setattr(study, "analyse_segment", spy_analyse)
    res = study.run_study(market, s_base, params=FASTER)
    (locked_start, newest), = seen["design_calls"]
    assert newest < locked_start  # no design event starts or ends in the locked years
    design_call, locked_call = seen["analyse"]
    assert design_call[1] == locked_start and design_call[3] < locked_start
    assert locked_call[0] == locked_start
    assert locked_call[2] == [BASELINE, "stop_atr_mult=4"]  # only the selected one, once
    # no other variant carries any locked number
    vt = res.variants_table
    others = vt.drop(index=["stop_atr_mult=4"])
    assert "locked_dR" in vt.columns
    assert others["locked_dR"].isna().all() and others["locked_n"].isna().all()
    assert list(res.locked.index) == [BASELINE, "stop_atr_mult=4"]
    report = study.render_report(res)
    locked_section = report.split("## Locked years")[1].split("## Honesty")[0]
    assert "stop_atr_mult=4" in locked_section and "stop_atr_mult=3.5" not in locked_section


def test_split_follows_locked_from_warns_and_purges_straddling_trades(market, s_base,
                                                                      result):
    # cut the calendar through a real trade: one that was opened in a year and closed in the next
    ev = result.events[result.events["counted"]]
    straddle = ev[ev["signal_day"].str[:4] != ev["exit_day"].str[:4]]
    year = int(straddle["exit_day"].iloc[0][:4])
    res = study.run_study(market, s_base, params=dataclasses.replace(FASTER, locked_from=year))
    assert res.meta["locked_from"] == year
    assert max(res.meta["design_years"]) == year - 1 and min(res.meta["locked_years"]) == year
    assert any("LOCKED segment has only" in w for w in res.warnings)  # fewer than 150 events
    ev = res.events[res.events["counted"]]
    split = f"{year}-01-01"
    design, purged = ev[ev["segment"] == "design"], ev[ev["segment"] == "purged"]
    assert (design["exit_day"] < split).all()  # a design trade ends before the split
    assert (purged["signal_day"] < split).all() and (purged["exit_day"] >= split).all()
    assert res.baseline["purged_events"] == len(purged) >= 1
    with pytest.raises(StudyError):
        study.run_study(market, s_base, params=dataclasses.replace(FASTER, locked_from=2040))


def test_default_result_is_keep_the_defaults_with_consistent_tables(result):
    assert result.verdict.startswith(("KEEP THE DEFAULTS", "NO CONCLUSION"))
    assert result.recommended == [] and result.env == {}
    d = result.design
    assert d.at[BASELINE, "dR"] == 0
    assert (d["n"] >= 0).all()
    # the identity control reproduces the baseline exactly (self-test of the machinery)
    assert result.checks["control trail_atr_mult=3 reproduces the baseline exactly"]
    assert d.at["trail_atr_mult=3", "dR"] == 0
    assert d.at["trail_atr_mult=3", "n"] == d.at[BASELINE, "n"]
    # only the decision rule selects: a higher mean alone never does
    passed = [n for n in d.index if n != BASELINE and bool(d.at[n, "design_pass"])]
    assert passed == result.selected


def test_declared_variants_cover_the_required_grid():
    names = {v.name: v for v in study.VARIANTS}
    for need in ("stop_atr_mult=2.5", "stop_atr_mult=4", "trail_atr_mult=3", "trail_atr_mult=4",
                 "trail_atr_mult=5", "top_pct_entry=0.05", "top_pct_entry=0.15",
                 "top_pct_entry=0.2", "near_high_min=0.8", "near_high_min=0.9",
                 "volume_factor=1", "volume_factor=1.6", "max_extension=0.15",
                 "max_extension=0.35", "rsi_max=70"):
        assert need in names, need
    flt = {v.flt for v in study.VARIANTS if v.flt}
    assert flt == {"gap_atr", "ext_ema20_atr", "event_bar", "id_smooth"}
    assert all(v.expect in ("better", "worse") for v in study.VARIANTS)
    assert names["trail_atr_mult=3"].role == "control"
    # every filter variant leaves the engine untouched: no settings override
    assert all(not v.overrides for v in study.VARIANTS if v.flt)


def test_variants_changing_the_panel_are_refused(market, s_base):
    bad = Variant("min_price=50", "min_price", {"min_price": 50.0}, "", 50.0, "better", "")
    with pytest.raises(StudyError):
        study.run_study(market, s_base, variants=[bad], params=FAST)


# --------------------------------------------------------------------- placebo
def test_placebo_runs_and_is_reproducible(market, s_base, result):
    for key in ("design", "design_by_year", "all", "all_by_year"):
        pl = result.placebo[key]
        assert pl["draws"] == FAST.placebo_draws
        assert 0 <= pl["baseline_percentile"] <= 100
        assert pl["random_avg_R_p5"] <= pl["random_avg_R_mean"] <= pl["random_avg_R_p95"]
        assert 0 < pl["p_value_one_sided"] <= 1
    again = study.run_study(market, s_base, params=FAST)
    assert again.placebo == result.placebo
    row = result.variants_table.loc["random_entries (placebo)"]
    assert row["role"] == "placebo" and row["n"] > 0


def test_allow_overlap_counts_more_events(market, s_base, result):
    over = study.run_study(market, s_base, params=dataclasses.replace(FASTER, allow_overlap=True))
    assert over.baseline["events"] >= result.baseline["events"]
    assert over.baseline["events"] == int(over.events["filled"].sum())


# ------------------------------------------------------------ edge cases and CLI
def test_market_without_signals_is_handled():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2019-01-02", periods=700)
    bars = {"SPY": ohlcv(rng.normal(-0.002, 0.008, 700), days, 300, 8e7, rng)}
    for sym in SYMS:
        bars[sym] = ohlcv(rng.normal(0.0, 0.015, 700), days, 100, 3e6, rng)
    res = study.run_study(bars, settings(universe=SYMS), params=FASTER)
    assert res.baseline["events"] == 0
    assert res.verdict.startswith("NO CONCLUSION")
    assert "Verdict" in study.render_report(res)


def test_cli_runs_end_to_end_on_csv_files(market, tmp_path, monkeypatch, capsys):
    data = tmp_path / "csv"
    data.mkdir()
    for sym, df in market.items():
        df.to_csv(data / f"{sym}.csv", index_label="date")
    monkeypatch.setenv("INV_MIN_BETA", "0")
    out = tmp_path / "out"
    code = study.main(["--csv-dir", str(data), "--out", str(out), "--no-vix", "--boot", "200",
                       "--placebo-draws", "40", "--env-file", str(tmp_path / "none"),
                       "--locked-from", "2021", "--audit", "sample"])
    assert code == 0
    for name in ("report.md", "events.csv", "variants.csv", "summary.json"):
        assert (out / name).exists()
    def _no_nan(token):
        raise ValueError(f"not strict JSON: {token}")

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"),
                         parse_constant=_no_nan)
    assert summary["verdict"].startswith(("KEEP THE DEFAULTS", "NO CONCLUSION",
                                          "ONE CHANGE"))
    assert summary["meta"]["locked_from"] == 2021
    assert summary["baseline"]["events"] > 0
    printed = capsys.readouterr().out
    assert "VERDICT:" in printed and "placebo" in printed and "avg R per year" in printed
    variants = pd.read_csv(out / "variants.csv", index_col=0)
    assert BASELINE in variants.index and len(variants) >= 30
    events = pd.read_csv(out / "events.csv")
    assert {"signal_day", "symbol", "R", "exit_reason", "segment", "counted"} <= set(events)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Verdict" in report and "En français" in report


def test_recommended_environment_variables_round_trip_through_the_settings(monkeypatch):
    """The INV_* names and values the report prints are exactly what the settings read."""
    checked = 0
    for v in study.VARIANTS:
        if v.flt:
            assert study.env_for(v) == {}  # a filter has no INV_ setting: it needs a code change
            continue
        env = study.env_for(v)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        s2 = InvestSettings.from_env(None, broker="sim")
        for field, value in v.overrides.items():
            assert getattr(s2, field) == pytest.approx(value)
        for name in env:
            monkeypatch.delenv(name)
        checked += 1
    assert checked >= 20
    assert study.env_for(next(v for v in study.VARIANTS if v.name == "stop_atr_mult=4")) == {
        "INV_STOP_ATR_MULT": "4.0"}


def test_verdict_names_the_variables_or_a_code_change():
    vmap = {v.name: v for v in study.VARIANTS}
    meta = {"family_size": 26, "variants_declared": 33, "design_years": [2019, 2023],
            "locked_years": [2024, 2026]}
    baseline = {"design_events": 300, "design_avg_R": 0.3}
    for name, wanted in (("stop_atr_mult=4", "INV_STOP_ATR_MULT=4.0"),
                         ("gap_atr<=0.5", "code change")):
        table = pd.DataFrame({"dR": [0.3], "p_holm": [0.01], "years_same_sign": ["4/5"]},
                             index=[name])
        locked = pd.DataFrame({"dR": [0.25]}, index=[name])
        text, fr = study._verdict(meta, baseline, {}, table, locked, [name], [name], [name],
                                  vmap, 300, Params())
        assert text.startswith(f"ONE CHANGE IS SUPPORTED: {name}") and wanted in text
        assert fr.startswith("UN SEUL CHANGEMENT")
    text, _ = study._verdict(meta, baseline, {}, table, pd.DataFrame(), [], [], [], vmap, 300,
                             Params())
    assert text.startswith("KEEP THE DEFAULTS") and "No variant was looked at" in text
    text, _ = study._verdict(meta, baseline, {}, table, locked, [name], [], [], vmap, 300,
                             Params())
    assert text.startswith("KEEP THE DEFAULTS") and "none held on the locked years" in text


def test_cli_needs_a_data_source(capsys):
    with pytest.raises(SystemExit):
        study.main([])


def test_stop_distance_is_the_risk_module_one(market, s_base):
    """R is measured against the bot's own initial stop (risk.stop_distance_pct)."""
    panel = build_panel(market, SYMS, s_base)
    row = panel["AAPL"].iloc[400]
    sp = stop_distance_pct(row["close"] * 1.03, row["atr"], s_base)
    cal = market["SPY"].index
    g = study.build_grid(panel, cal)
    j = g.syms.index("AAPL")
    t0 = int(cal.get_loc(row.name))
    off = np.zeros(g.T, dtype=bool)
    r = study.simulate(g, np.array([t0]), np.array([j]), s_base, off)
    if r.filled[0]:
        assert r.sp[0] == pytest.approx(sp)
