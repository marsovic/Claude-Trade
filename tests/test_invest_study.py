"""Tests of the signal-level walk-forward study (invest/study.py).

The most important one is ``test_trades_match_the_real_engine``: the study re-implements the
exit logic of ``Engine.process_day`` + ``SimBroker`` on whole arrays. If the engine's exit
logic changes (stops, slippage, entry-day high, rank exit...), that test fails and
``study.simulate`` must follow.
"""

from __future__ import annotations

import dataclasses
import json
import logging

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
def _fake_sets(rng, planted=None, only_year=None, n=600, week_icc=0.3, keep=None):
    cal = study.make_calendar(pd.bdate_range("2017-01-02", "2024-12-31"))
    L = int(np.searchsorted(cal.yr, 2022))
    t0 = np.sort(rng.integers(0, L - 300, n)).astype(np.int64)
    wk_eff = rng.normal(0, np.sqrt(week_icc), cal.wk.max() + 1)
    base_r = wk_eff[cal.wk[t0]] + rng.normal(0, 1.5, n) + 0.2
    tests = [v for v in study.VARIANTS if v.role == "test"]

    def make(name, mask, r):
        ix = np.nonzero(mask)[0]
        return EventSet(name, t0[ix], ix.astype(np.int64), t0[ix] + 10, r[ix],
                        r[ix], r[ix] * 2, rng.random(len(ix)), np.full(len(ix), 10),
                        np.array([""] * len(ix), dtype=object))

    sets = {BASELINE: make(BASELINE, np.ones(n, dtype=bool), base_r)}
    for v in tests:
        r = base_r + rng.normal(0, 0.3, n)
        if planted and v.name in planted:
            add = planted[v.name]
            r = r + (add if only_year is None else add * (cal.yr[t0] == only_year))
        u = rng.random(n)
        frac = rng.uniform(0.7, 1.0)
        sets[v.name] = make(v.name, u < (keep or {}).get(v.name, frac), r)
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
    # the small test market has fewer than the 150 locked events the hard gate needs
    res = study.run_study(market, s_base, params=dataclasses.replace(FASTER, min_locked_events=0))
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
        assert pl["diff_lo"] <= pl["diff"] <= pl["diff_hi"]
        assert pl["diff"] == pytest.approx(pl["baseline_avg_R"] - pl["random_avg_R_mean"])
        assert pl["random_avg_R_p5"] <= pl["random_avg_R_p95"]
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
    code = study.main(["--csv-dir", str(data), "--out", str(out), "--no-vix", "--boot", "2000",
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
    assert text.startswith("KEEP THE DEFAULTS") and "none was confirmed on the locked years" in text


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


# ==================================================================================
# Review fixes: false-recommendation risks (S1-S5), data and input robustness (S6, F3-F9)
# ==================================================================================
def _drifting_market(seed=3, n_syms=45, n_days=2300, mu=0.0009):
    """Random walks with a positive drift and NO predictability: any "edge" is just drift."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2015-01-02", periods=n_days)
    mkt = rng.normal(mu, 0.009, n_days)
    bars = {"SPY": ohlcv(mkt, days, 300, 8e7, rng)}
    syms = [f"S{i:02d}" for i in range(n_syms)]
    for i, sym in enumerate(syms):
        beta = 0.8 + 0.8 * rng.random()
        bars[sym] = ohlcv(beta * mkt + rng.normal(mu / 2, 0.017, n_days), days, 40 + 3 * i, 3e6,
                          rng)
    return bars, syms


@pytest.fixture(scope="module")
def drift_run():
    bars, syms = _drifting_market()
    p = Params(n_boot=400, placebo_draws=0, audit="off", min_locked_events=0)
    return study.run_study(bars, settings(universe=syms), params=p)


def _grid_of_closes(close: np.ndarray) -> Grid:
    t, n = close.shape
    A = {k: np.full((t, n), np.nan) for k in study.GRID_COLS}
    A["close"] = close
    return Grid(cal=pd.bdate_range("2020-01-02", periods=t), syms=[f"X{i}" for i in range(n)],
                A=A, eligible=np.ones((t, n), dtype=bool))


# ------------------------------------------------------------------ S1 / F1: drift
def test_market_move_beta_is_trailing_and_the_drift_is_beta_times_the_window_return():
    rng = np.random.default_rng(0)
    t = 400
    m = rng.normal(0.0005, 0.01, t)
    loads = np.array([2.0, 0.5, 1.0])
    close = 100 * np.cumprod(1 + m[:, None] * loads + rng.normal(0, 1e-4, (t, 3)), axis=0)
    mm = study.market_move(_grid_of_closes(close), window=126)
    want = 0.5 * loads / loads.mean() + 0.5  # shrunk halfway to 1
    assert mm.beta[300] == pytest.approx(want, abs=0.03)
    assert mm.beta[300, 0] > mm.beta[300, 2] > mm.beta[300, 1]
    # nothing after session 300 may change the beta known at the close of session 300
    later = close.copy()
    later[301:] *= np.exp(np.cumsum(rng.normal(0, 0.05, (t - 301, 3)), axis=0))
    mm2 = study.market_move(_grid_of_closes(later), window=126)
    assert np.allclose(mm2.beta[:301], mm.beta[:301]) and np.allclose(mm2.idx[:301], mm.idx[:301])
    # the window return is the ratio of the index; the drift charged to a trade is beta x it
    t_in, t_out, j = np.array([10, 20]), np.array([50, 25]), np.array([0, 2])
    win = mm.idx[t_out] / mm.idx[t_in] - 1
    assert mm.window(t_in, t_out) == pytest.approx(win)
    assert mm.drift(t_in, j, t_out) == pytest.approx(mm.beta[t_in, j] * win)
    assert study.annual_growth(mm.idx, 0, t) == pytest.approx(
        (mm.idx[-1] / mm.idx[0]) ** (252 / (t - 1)) - 1)


def test_one_bad_price_cannot_move_the_universe_index():
    rng = np.random.default_rng(1)
    close = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, (300, 40)), axis=0)
    base = study.market_move(_grid_of_closes(close), window=126).idx
    glitch = close.copy()
    glitch[150:, 7] *= 100.0  # a 100x print in one of 40 stocks (a data error)
    bad = study.market_move(_grid_of_closes(glitch), window=126).idx
    assert bad[160] / base[160] < 1.04  # unclipped it would be about 3.5x


def test_holding_longer_in_a_drifting_market_is_not_an_edge(drift_run):
    adj = drift_run
    assert adj.meta["universe_cagr"] > study.DRIFT_WARN
    assert any("equal-weight universe grew" in w for w in adj.warnings)
    for nm in ("trail_atr_mult=5", "stop_atr_mult=4"):
        # most of the raw gain of a wider stop or trail is the market's drift over the longer
        # hold: no predictability was planted in this market
        assert adj.design.at[nm, "dR_raw"] > 0.05, nm
        assert adj.design.at[nm, "dR_raw"] - adj.design.at[nm, "dR"] > 0.15, nm
        assert adj.design.at[nm, "dR"] < 0.5 * adj.design.at[nm, "dR_raw"], nm
    assert adj.design.at["trail_atr_mult=5", "dR_raw"] > 0.3
    assert not adj.verdict.startswith("ONE CHANGE")
    assert adj.meta["metric"].startswith("R net of beta")
    assert "market's move over its holding window" in study.render_report(adj)


def test_without_the_adjustment_the_decision_metric_is_the_raw_r(market, s_base):
    raw = study.run_study(market, s_base, params=dataclasses.replace(FASTER, drift_adjust=False))
    d = raw.design
    assert np.allclose(d["dR"], d["dR_raw"], equal_nan=True)
    assert raw.meta["metric"] == "raw R"
    adj = study.run_study(market, s_base, params=FASTER)
    assert np.allclose(adj.design["dR_raw"], d["dR_raw"], equal_nan=True)  # same trades
    assert not np.allclose(adj.design["dR"], adj.design["dR_raw"], equal_nan=True)


# ------------------------------------------------------- S2 / F2: the locked confirmation
def test_locked_confirmation_is_a_significance_test_not_just_same_sign_and_half():
    names = list("abcd")
    design = pd.DataFrame({"dR": [0.20] * 4}, index=names)
    locked = pd.DataFrame({"dR": [0.12, 0.12, -0.05, 0.05],
                           "p_holm": [0.04, 0.40, 0.04, 0.04],
                           "p_holm_blk": [0.06, 0.04, 0.04, 0.04]}, index=names)
    passed, ratios, p1s, oks = study.confirm_locked(locked, design, names, Params())
    # a: significant one-sided (0.06 / 2 <= 0.05) and above half the design size
    # b: same sign and above half the size (the OLD rule passed it) but p = 0.4
    # c: wrong sign;  d: significant but under half of the design size
    assert passed == ["a"] and oks == {"a": True, "b": False, "c": False, "d": False}
    assert ratios["b"] == pytest.approx(0.6) and p1s["b"] == pytest.approx(0.2)


def test_too_small_a_locked_segment_gives_no_recommendation_at_all(market, s_base, monkeypatch):
    orig = study.select_on_design

    def force(*a, **k):
        out = orig(*a, **k)
        out.loc["stop_atr_mult=4", "design_pass"] = True
        return out

    monkeypatch.setattr(study, "select_on_design", force)
    seen = []
    orig_analyse = study.analyse_segment
    monkeypatch.setattr(study, "analyse_segment",
                        lambda *a, **k: (seen.append(a[2:4]), orig_analyse(*a, **k))[1])
    res = study.run_study(market, s_base, params=FASTER)  # default: >= 150 locked events
    assert res.baseline["locked_events"] < FASTER.min_locked_events
    assert res.selected == ["stop_atr_mult=4"]
    assert res.final == [] and res.recommended == [] and res.env == {}
    assert res.locked.empty and res.meta["locked_blocked"]
    assert len(seen) == 1  # the locked years were not even looked at
    assert res.verdict.startswith("NO CONCLUSION (PROVISIONAL)")
    report = study.render_report(res)
    assert "NOT evaluated" in report and "Environment variables" not in report


# ---------------------------------------- S3: few changed events / micro gains cannot pass
def _select(sets, cal, L, tests, seed=0, n_boot=300):
    p = Params(n_boot=n_boot)
    s = settings()
    vmap = {v.name: v for v in study.VARIANTS}
    tbl = study.analyse_segment(sets, cal, 0, L, _families(tests), p, seed)
    axis = study.axis_values(study.VARIANTS, s)
    return study.select_on_design(sets, tbl, vmap, axis, p, s, L, L / 252.0)


def test_a_variant_that_changes_few_events_or_gains_little_cannot_pass_the_size_test():
    sets, cal, L, tests = _fake_sets(np.random.default_rng(7), n=500)
    base = sets[BASELINE]
    order = np.argsort(base.R)

    def drop_worst(k):
        keep = np.ones(base.n, dtype=bool)
        keep[order[:k]] = False
        return base.sel(keep)

    sets["near_high_min=0.9"] = drop_worst(20)  # 20 events changed: too few
    sets["near_high_min=0.95"] = drop_worst(60)  # 60 events changed
    sets["rsi_max=70"] = dataclasses.replace(base, Rref=base.Rref + 0.01, Rx=base.Rx + 0.01)
    tbl = _select(sets, cal, L, tests)
    assert tbl.at["near_high_min=0.9", "n_changed"] == 20
    assert tbl.at["near_high_min=0.9", "dR"] >= Params().min_gain_r  # the gain alone is enough
    assert not tbl.at["near_high_min=0.9", "c_size"]
    assert not tbl.at["near_high_min=0.9", "design_pass"]
    assert tbl.at["near_high_min=0.95", "n_changed"] == 60
    assert tbl.at["near_high_min=0.95", "c_size"]
    # every event changed but by a hundredth of R: under the minimum gain
    assert tbl.at["rsi_max=70", "n_changed"] == base.n
    assert 0 < tbl.at["rsi_max=70", "dR"] < Params().min_gain_r
    assert not tbl.at["rsi_max=70", "c_size"]
    assert study._n_changed(base, base) == 0


# ------------------------------------------------------------ S4: strong but not adoptable
def test_a_strong_filter_that_fails_only_coverage_is_said_to_be_real_but_not_adoptable():
    rng = np.random.default_rng(11)
    planted = {"volume_factor=1.6": 0.6, "volume_factor=2": 0.8}
    sets, cal, L, tests = _fake_sets(rng, planted=planted,
                                     keep={"volume_factor=1.6": 0.95, "volume_factor=2": 0.3})
    tbl = _select(sets, cal, L, tests, n_boot=3000)  # Holm over 26 tests needs many draws
    row = tbl.loc["volume_factor=2"]
    assert row["coverage_only"] and not row["c_keep"] and not row["design_pass"]
    assert row["c_holm"] and row["c_years"] and row["c_plateau"] and row["c_size"]
    why = study._why_not({**row.to_dict(), "role": "test", "expect": "better"}, 0.05)
    assert "slots would sit idle" in why and "29%" in why
    meta = {"family_size": 26, "variants_declared": 33, "design_years": [2019, 2023],
            "locked_years": [2024, 2026], "coverage_only": ["volume_factor=2"]}
    text, _ = study._verdict(meta, {"design_events": 300, "design_avg_R": 0.3}, {}, tbl,
                             pd.DataFrame(), [], [], [], {}, 300, Params())
    assert text.startswith("KEEP THE DEFAULTS") and "except coverage (volume_factor=2)" in text


# ----------------------------------------------------------------------- S5: placebo
def _event_set(cal, rng, n, mean, weeks=None):
    t0 = np.sort(rng.integers(0, len(cal.wk) - 30, n)).astype(np.int64)
    r = rng.normal(mean, 1.0, n)
    return EventSet("x", t0, np.zeros(n, dtype=np.int64), t0 + 5, r, r, r, np.zeros(n),
                    np.full(n, 5), np.array([""] * n, dtype=object))


def test_placebo_summary_is_a_week_bootstrap_of_the_difference_not_a_percentile():
    cal = study.make_calendar(pd.bdate_range("2017-01-02", "2020-12-31"))
    rng = np.random.default_rng(0)
    hi = len(cal.wk)
    pool = _event_set(cal, rng, 3000, 0.0)
    same = study.placebo_summary(pool, _event_set(cal, rng, 300, 0.0), cal, 600, 1, False, 0, hi)
    assert same["diff_lo"] < 0 < same["diff_hi"] and same["p_value_one_sided"] > 0.05
    better = study.placebo_summary(pool, _event_set(cal, rng, 300, 0.6), cal, 600, 1, False, 0,
                                   hi)
    assert better["diff"] > 0.4 and better["diff_lo"] > 0 and better["p_value_one_sided"] < 0.01
    # the baseline's own noise counts: 15 events give a far wider interval than 300 events
    small = study.placebo_summary(pool, _event_set(cal, rng, 15, 0.6), cal, 600, 1, False, 0, hi)
    assert (small["diff_hi"] - small["diff_lo"]) > 2 * (better["diff_hi"] - better["diff_lo"])
    assert study.placebo_summary(pool, _event_set(cal, np.random.default_rng(5), 300, 0.0), cal,
                                 100, 3, True, 0, hi) == study.placebo_summary(
        pool, _event_set(cal, np.random.default_rng(5), 300, 0.0), cal, 100, 3, True, 0, hi)
    assert study.placebo_summary(None, pool, cal, 10, 1, False, 0, hi) == {"draws": 0}
    for key in ("baseline_percentile", "beats"):
        assert key not in better  # the old "beat X% of the draws" is gone


def test_placebo_entries_come_from_the_entry_band_not_from_all_stocks(market, s_base,
                                                                     monkeypatch):
    calls = []
    real = study.simulate

    def spy(g, t0, j, s, off, horizon=250, ref_equity=None):
        calls.append((g, t0.copy(), j.copy()))
        return real(g, t0, j, s, off, horizon, ref_equity)

    monkeypatch.setattr(study, "simulate", spy)
    study.run_study(market, s_base, params=dataclasses.replace(FAST, audit="off"))
    g, t0, j = calls[-1]  # the placebo pool is simulated last
    assert len(t0) > 50
    assert (g.A["pct_rank"][t0, j] >= 1 - s_base.top_pct_entry).all()
    assert (g.A["score"][t0, j] > 0).all() and g.eligible[t0, j].all()


def test_design_placebo_does_not_depend_on_the_locked_years(market, s_base):
    p = dataclasses.replace(FAST, audit="off", placebo_pool=40, locked_from=2021)
    a = study.run_study(market, s_base, params=p)
    rng = np.random.default_rng(5)
    changed = {}
    for k, df in market.items():
        d = df.copy()
        late = np.asarray(d.index >= pd.Timestamp("2021-01-01"))
        factor = np.exp(np.cumsum(rng.normal(0, 0.03, late.sum())))
        d.loc[late, ["open", "high", "low", "close"]] = (
            d.loc[late, ["open", "high", "low", "close"]].to_numpy() * factor[:, None])
        changed[k] = d
    b = study.run_study(changed, s_base, params=p)
    assert a.placebo["design"]["draws"] and a.placebo["design_by_year"]["draws"]
    assert a.placebo["design"] == b.placebo["design"]
    assert a.placebo["design_by_year"] == b.placebo["design_by_year"]
    assert a.placebo["all"] != b.placebo["all"]  # the locked years themselves do differ


def test_events_csv_keeps_the_locked_rows_features_empty(result):
    ev = result.events
    locked, design = ev[ev["segment"] == "locked"], ev[ev["segment"] == "design"]
    assert len(locked) > 0 and len(design) > 0
    cols = ["score", "pct_rank", "rsi", "vol_ratio", "h52", "ext50", "atr_pct", "ext_ema20_atr",
            "gap_atr", "id_smooth", "event_bar_1", "event_bar_3", "event_bar_5"]
    assert locked[cols].isna().all().all()  # nothing to rebuild an unselected variant from
    assert design[["score", "pct_rank", "gap_atr"]].notna().all().all()
    assert locked["R"].notna().any()  # the baseline's own result stays visible
    out = study.render_report(result)
    assert "INFORMATION ONLY" in out and "left empty on purpose" in out


# ----------------------------------------------------- S6: a bad bar must not blind the tool
def test_a_zero_price_bar_is_dropped_loudly_instead_of_turning_everything_into_nan(
        market, s_base, result):
    ev = result.events[result.events["counted"]]
    row = ev.iloc[len(ev) // 2]
    cal = market["SPY"].index
    nxt = cal[cal.get_loc(pd.Timestamp(row["signal_day"])) + 1]
    bars = {k: v.copy() for k, v in market.items()}
    bars[row["symbol"]].loc[nxt, ["open", "low"]] = 0.0
    with np.errstate(divide="raise"):  # no hidden divide-by-zero either
        res = study.run_study(bars, s_base, params=FASTER)
    assert np.isfinite(res.baseline["avg_R"]) and np.isfinite(res.baseline["design_avg_R"])
    assert np.isfinite(res.design.at[BASELINE, "avg_R"])
    assert any("could not be traded" in w and row["symbol"] in w for w in res.warnings)
    assert any("zero or negative price" in w for w in res.warnings)
    bad = res.events[(res.events["symbol"] == row["symbol"])
                     & (res.events["signal_day"] == row["signal_day"])]
    assert not bad["filled"].iloc[0]
    assert res.baseline["events"] == result.baseline["events"] - 1


def test_simulate_marks_invalid_prices_and_counts_refused_entries():
    s = settings(fee_preset="alpaca")
    rows = flat(2, 100) + [(100, 100.2, 99.8, 100), (0.0, 100.8, 0.0, 100.6)] + flat(6, 100.6)
    r = run_one(rows, s)
    assert not r.filled[0] and r.invalid[0] and np.isnan(r.net_ret[0])
    rows = flat(2, 100) + [(100, 100.2, 99.8, 100), (100.5, 100.8, 100.2, 100.6)] + flat(6, 100.6)
    g = crafted_grid(rows)
    tight = study.simulate(g, np.array([2]), np.array([0]), settings(max_cost_r=1e-9), np.zeros(
        g.T, dtype=bool))
    assert tight.refused[0] and not tight.filled[0] and not tight.invalid[0]


# -------------------------------------- F7: cost cap and fee preset are said out loud
def test_entries_refused_by_the_cost_cap_are_named_in_the_verdict(market):
    s = settings(universe=SYMS, max_cost_r=1e-6)
    res = study.run_study(market, s, params=FASTER)
    assert res.baseline["events"] == 0 and res.meta["refused_by_cost"] > 0
    assert res.verdict.startswith("NO CONCLUSION") and "--ref-equity" in res.verdict
    assert any("refused because the round trip costs" in w for w in res.warnings)


def test_fee_preset_and_reference_equity_are_listed_and_a_different_live_preset_warned(
        market, s_base):
    res = study.run_study(market, s_base, params=dataclasses.replace(
        FASTER, live_fee_preset="ibkr", ref_equity=5000.0))
    assert any("INV_FEE_PRESET=ibkr" in w for w in res.warnings)
    report = study.render_report(res)
    assert "fee preset" in report and "5,000 EUR" in report and "INV_FEE_PRESET=ibkr" in report
    assert res.meta["ref_equity_eur"] == 5000.0


# ------------------------------------------------------------- S8: gap_atr looks ahead
def test_gap_filter_verdict_says_it_needs_a_check_at_the_open():
    vmap = {v.name: v for v in study.VARIANTS}
    meta = {"family_size": 26, "variants_declared": 33, "design_years": [2019, 2023],
            "locked_years": [2024, 2026]}
    table = pd.DataFrame({"dR": [0.3], "p_holm": [0.01], "years_same_sign": ["4/5"]},
                         index=["gap_atr<=0.5"])
    locked = pd.DataFrame({"dR": [0.25], "locked_p_one_sided": [0.02]}, index=["gap_atr<=0.5"])
    text, _ = study._verdict(meta, {"design_events": 300, "design_avg_R": 0.3}, {}, table, locked,
                             ["gap_atr<=0.5"], ["gap_atr<=0.5"], ["gap_atr<=0.5"], vmap, 300,
                             Params())
    assert "NEXT session's open" in text and "check made at the open" in text
    assert "one-sided p 0.020" in text
    assert "NEXT session's open" in study.__doc__


# ----------------------------------------------------------------------- F3: VIX status
def test_without_vix_the_study_says_so_and_a_failed_download_blocks_any_recommendation(
        market, s_base, monkeypatch):
    orig = study.select_on_design

    def force(*a, **k):
        out = orig(*a, **k)
        out.loc["stop_atr_mult=4", "design_pass"] = True
        return out

    monkeypatch.setattr(study, "select_on_design", force)
    monkeypatch.setattr(study, "confirm_locked", lambda locked, design, selected, p: (
        list(selected), {n: 1.0 for n in selected}, {n: 0.01 for n in selected},
        {n: True for n in selected}))
    p = dataclasses.replace(FASTER, min_locked_events=0)
    ok = study.run_study(market, s_base, params=p)
    assert ok.recommended == ["stop_atr_mult=4"] and ok.meta["vix_status"] == "off"
    assert any("VIX / VIX3M are not used" in w for w in ok.warnings)
    assert "SPY trend only" in study.render_report(ok)
    chosen = study.run_study(market, s_base, params=dataclasses.replace(p, vix_disabled=True))
    assert any("chosen with --no-vix" in w for w in chosen.warnings)
    failed = study.run_study(market, s_base, params=dataclasses.replace(p, vix_failed=True))
    assert failed.meta["vix_status"] == "unavailable"
    assert failed.recommended == [] and failed.env == {}
    assert failed.verdict.startswith("NO CONCLUSION: VIX")
    assert any("could not be downloaded" in w for w in failed.warnings)
    assert "could NOT be loaded" in study.render_report(failed)


def test_vix_coverage_is_reported(market, s_base):
    cal = market["SPY"].index
    vix = pd.Series(15.0, index=cal)
    vix3m = pd.Series(17.0, index=cal)
    full = study.run_study(market, s_base, vix=vix, vix3m=vix3m, params=FASTER)
    assert full.meta["vix_status"] == "loaded" and full.meta["vix"]["vix3m"]["n"] == len(cal)
    assert "VIX from FRED" in study.render_report(full)
    assert not any("VIX starts" in w or "VIX ends" in w or "VIX3M is missing" in w
                   for w in full.warnings)
    late = study.run_study(market, s_base, vix=vix.iloc[900:1500], params=FASTER)
    assert any("VIX starts on" in w for w in late.warnings)
    assert any("VIX ends on" in w for w in late.warnings)
    assert any("VIX3M is missing" in w for w in late.warnings)


# ------------------------------------------------------------------ F9: small counts
def test_raw_signals_do_not_count_the_last_session_and_horizon_days_are_held(
        market, s_base, result):
    last = result.events["signal_day"].iloc[len(result.events) // 2]
    res = study.run_study(market, s_base, params=dataclasses.replace(FASTER, end=last))
    assert res.meta["last_day"] == last
    assert res.baseline["raw_signals"] == len(res.events)  # a last-session signal cannot trade
    s = settings(fee_preset="alpaca")
    rows = flat(2, 100) + [(100, 100.2, 99.8, 100), (100.5, 100.8, 100.2, 100.6)] + flat(30, 100.6)
    g = crafted_grid(rows)
    off = np.zeros(g.T, dtype=bool)
    r = study.simulate(g, np.array([2]), np.array([0]), s, off, horizon=7)
    assert r.reason[0] == "horizon cap" and r.exit_t[0] == 3 + 7 and r.held[0] == 7
    r2 = study.simulate(g, np.array([2]), np.array([0]), s, off, horizon=2)
    assert r2.exit_t[0] == 3 + 2 and r2.held[0] == 2


# ==================================================================================
# command line: inputs, providers, cache (F3-F7)
# ==================================================================================
class FakeProvider:
    """A bar provider with a request counter, a rate limit and optional failures."""

    name = "fake"
    batch_size = 1
    supports_grouped = False

    def __init__(self, bars, fail=None, interval=75.0):
        self.bars, self.fail, self.calls = bars, fail or {}, []
        self.limiter = type("L", (), {"interval": interval})()

    def history(self, symbol, start, end):
        self.calls.append(symbol)
        if symbol in self.fail:
            raise self.fail[symbol]
        df = self.bars.get(symbol)
        return study.empty_frame() if df is None else df

    def history_many(self, symbols, start, end):
        return {s: self.history(s, start, end) for s in symbols}

    def grouped(self, day):
        return None


def _csv_dir(tmp_path, market, name="csv", drop_volume=False, rows=None):
    data = tmp_path / name
    data.mkdir()
    for sym, df in market.items():
        df = df if rows is None else df.iloc[:rows]
        df = df.drop(columns=["volume"]) if drop_volume else df
        df.to_csv(data / f"{sym}.csv", index_label="date")
    return data


BASE_ARGS = ["--no-vix", "--boot", "2000", "--placebo-draws", "0", "--audit", "off"]


def _cli(tmp_path, *args, env_file=True):
    out = tmp_path / "out"
    extra = ["--env-file", str(tmp_path / "none")] if env_file else []
    return study.main([*args, "--out", str(out), *extra]), out


@pytest.mark.parametrize("args, fragment", [
    (["--start", "2019-13-45"], "not a date"),
    (["--start", "yesterday"], "not a date"),
    (["--end", "notadate"], "not a date"),
    (["--boot", "-5"], "out of range"),
    (["--boot", "200"], "at least 2000"),
    (["--design-frac", "0"], "strictly between"),
    (["--design-frac", "1"], "strictly between"),
    (["--design-frac", "5"], "strictly between"),
    (["--design-frac", "-0.5"], "strictly between"),
    (["--horizon", "1"], "out of range"),
    (["--horizon", "0"], "out of range"),
    (["--horizon", "3"], "shorter than the time stop"),
    (["--ref-equity", "0"], "out of range"),
    (["--ref-equity", "-5"], "out of range"),
    (["--days", "10"], "out of range"),
    (["--placebo-draws", "-1"], "out of range"),
    (["--locked-from", "abc"], "not a valid number"),
])
def test_bad_arguments_stop_cleanly_before_any_work(args, fragment, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        study.main(["--csv-dir", str(tmp_path), *args, "--out", str(tmp_path / "o"),
                    "--env-file", str(tmp_path / "none")])
    assert exc.value.code == 2
    assert fragment in capsys.readouterr().err


def test_an_unwritable_out_folder_is_refused_before_the_run(tmp_path, capsys):
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    with pytest.raises(SystemExit) as exc:
        study.main(["--csv-dir", str(tmp_path), "--out", str(blocker),
                    "--env-file", str(tmp_path / "none")])
    assert exc.value.code == 2 and "cannot write there" in capsys.readouterr().err


@pytest.mark.parametrize("var, value", [("INV_SLOTS", "abc"), ("INV_RISK_PER_TRADE", "0.5")])
def test_bad_settings_give_a_message_not_a_traceback(var, value, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(var, value)
    code, _ = _cli(tmp_path, "--csv-dir", str(tmp_path), *BASE_ARGS)
    assert code == 2 and "STUDY STOPPED: invalid settings" in capsys.readouterr().err


def test_unreadable_or_too_short_data_gives_a_message_not_a_traceback(tmp_path, monkeypatch,
                                                                      capsys, market):
    monkeypatch.setenv("INV_MIN_BETA", "0")
    nov = _csv_dir(tmp_path, market, "novol", drop_volume=True)
    code, _ = _cli(tmp_path, "--csv-dir", str(nov), *BASE_ARGS)
    err = capsys.readouterr().err
    assert code == 2 and "STUDY STOPPED: cannot read the price data" in err
    assert "volume" in err
    short = _csv_dir(tmp_path, market, "short", rows=30)
    code, _ = _cli(tmp_path, "--csv-dir", str(short), *BASE_ARGS)
    assert code == 2 and "STUDY STOPPED" in capsys.readouterr().err
    with pytest.raises(StudyError):  # the same through the API: no empty-panel crash
        study.run_study({k: v.iloc[:30] for k, v in market.items()}, settings(universe=SYMS),
                        params=FASTER)


def test_provider_failures_stop_with_a_message_that_names_the_cause(tmp_path, monkeypatch,
                                                                    capsys, market):
    monkeypatch.setenv("INV_MIN_BETA", "0")
    args = ["--provider", "tiingo", "--data-dir", str(tmp_path / "cache"), *BASE_ARGS]
    # no API key
    def no_key(s):
        raise study.DataError("TIINGO_API_KEY is required")

    monkeypatch.setattr(study, "make_provider", no_key)
    code, _ = _cli(tmp_path, *args)
    err = capsys.readouterr().err
    assert code == 2 and "STUDY STOPPED" in err and "TIINGO_API_KEY" in err
    # no network
    boom = requests_connection_error()
    monkeypatch.setattr(study, "make_provider", lambda s: FakeProvider(
        market, fail={sym: boom for sym in market}))
    code, _ = _cli(tmp_path, *args)
    assert code == 2 and "could not be reached" in capsys.readouterr().err
    # a refused key (401) names the symbol it happened on
    monkeypatch.setattr(study, "make_provider", lambda s: FakeProvider(
        market, fail={sym: study.DataError("Tiingo HTTP 401: bad token") for sym in market}))
    code, _ = _cli(tmp_path, *args)
    err = capsys.readouterr().err
    assert code == 2 and "HTTP 401" in err and any(f"{sym}:" in err for sym in market)


def requests_connection_error():
    import requests

    return requests.exceptions.ConnectionError("no route to host")


def test_an_unknown_symbol_is_skipped_with_a_warning_and_the_cache_is_the_studys_own(
        tmp_path, monkeypatch, caplog, market):
    monkeypatch.setenv("INV_MIN_BETA", "0")
    fake = FakeProvider(market, fail={"MSFT": study.DataError("Tiingo HTTP 404: Not found.")})
    monkeypatch.setattr(study, "make_provider", lambda s: fake)
    cache = tmp_path / "cache"
    with caplog.at_level(logging.INFO):
        code, out = _cli(tmp_path, "--provider", "tiingo", "--data-dir", str(cache), *BASE_ARGS,
                         "--locked-from", "2021")
    assert code == 0 and (out / "report.md").exists()
    text = caplog.text
    assert "MSFT is not known to fake (HTTP 404): skipped" in text
    assert "INV_EXCLUDE" in text and "MSFT" in text
    assert (cache / "bars" / "AAPL.csv").exists() and not (cache / "bars" / "MSFT.csv").exists()
    assert "about" in text and "rate limit" in text and "--csv-dir" in text  # up-front ETA


def test_a_warm_cache_makes_no_request_and_the_eta_is_printed_up_front(tmp_path, monkeypatch,
                                                                       caplog, market):
    cal = market["SPY"].index
    monkeypatch.setattr(study, "_last_session", lambda today: cal[-1].date())
    # NVDA "listed" late: its history is short for good and must not be asked for again
    fake = FakeProvider({**market, "NVDA": market["NVDA"].iloc[900:]})
    monkeypatch.setattr(study, "make_provider", lambda s: fake)
    args = type("A", (), {"data_dir": str(tmp_path / "c"), "days": 1500})()
    s = settings()
    symbols = ["SPY", "AAPL", "MSFT", "NVDA"]
    with caplog.at_level(logging.INFO):
        first = study._download(args, s, symbols)
    assert sorted(fake.calls) == sorted(symbols) and set(first) == set(symbols)
    assert "4 request(s), about 0.1 h at the provider's rate limit" in caplog.text
    assert "--csv-dir" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO):
        again = study._download(args, s, symbols)
    assert len(fake.calls) == len(symbols)  # nothing requested the second time
    assert "complete and recent" in caplog.text
    assert all(len(again[x]) == len(first[x]) for x in symbols)
    # a symbol whose last bar is older than a week is the only one refreshed
    monkeypatch.setattr(study, "_last_session", lambda today: cal[-1].date() + pd.Timedelta(
        days=30))
    study._download(args, s, symbols)
    assert len(fake.calls) == 2 * len(symbols)


def test_last_session_is_the_previous_trading_day():
    from datetime import date

    assert study._last_session(date(2024, 6, 3)) == date(2024, 5, 31)  # Monday -> Friday
    assert study._last_session(date(2024, 6, 5)) == date(2024, 6, 4)
    assert study._last_session(date(2024, 7, 5)) == date(2024, 7, 3)  # July 4th holiday


def test_failed_vix_download_is_reported_and_blocks_a_recommendation_in_the_cli(
        tmp_path, monkeypatch, market):
    monkeypatch.setenv("INV_MIN_BETA", "0")

    def fred_down(*a, **k):
        raise requests_connection_error()

    monkeypatch.setattr(study, "fred_series", fred_down)
    data = _csv_dir(tmp_path, market)
    args = ["--csv-dir", str(data), "--boot", "2000", "--placebo-draws", "0", "--audit", "off",
            "--locked-from", "2021"]
    code, out = _cli(tmp_path, *args)
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["vix_status"] == "unavailable" and summary["recommended"] == []
    assert summary["verdict"].startswith(("NO CONCLUSION", "KEEP THE DEFAULTS"))
    assert any("could not be downloaded" in w for w in summary["warnings"])
    assert "could NOT be loaded" in (out / "report.md").read_text(encoding="utf-8")
    # asking for no VIX on purpose is a different, explicit state
    code, out2 = _cli(tmp_path, "--csv-dir", str(data), *BASE_ARGS, "--locked-from", "2021")
    assert code == 0
    meta = json.loads((out2 / "summary.json").read_text(encoding="utf-8"))["meta"]
    assert meta["vix_status"] == "off"


def test_a_live_broker_with_another_fee_preset_is_warned_about_in_the_cli(tmp_path, monkeypatch,
                                                                        market):
    monkeypatch.setenv("INV_MIN_BETA", "0")
    monkeypatch.setenv("INV_BROKER", "ibkr")
    monkeypatch.delenv("INV_FEE_PRESET", raising=False)
    data = _csv_dir(tmp_path, market)
    code, out = _cli(tmp_path, "--csv-dir", str(data), *BASE_ARGS, "--locked-from", "2021")
    assert code == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["fee_preset"] == "sim" and summary["meta"]["live_fee_preset"] == "ibkr"
    assert any("INV_FEE_PRESET=ibkr" in w for w in summary["warnings"])


def test_invalid_parameters_are_refused_through_the_api_too(market, s_base):
    for bad in ({"horizon": 1}, {"design_frac": 1.0}, {"design_frac": 0.0}, {"ref_equity": 0.0},
                {"n_boot": 0}, {"end": "notadate"}, {"start": "2019-13-45"}):
        with pytest.raises(StudyError, match="invalid study parameters"):
            study.run_study(market, s_base, params=dataclasses.replace(FASTER, **bad))
