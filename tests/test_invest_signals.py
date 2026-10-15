import numpy as np
import pandas as pd
import pytest

from invest.signals import (
    CONDITIONS,
    build_panel,
    etp_entry_ok,
    etp_exit_reason,
    evaluate,
    evaluate_day,
    regime_at,
    symbol_features,
    synthetic_etp,
)
from tests.invest_helpers import SYMS, make_market, settings, signal_case

EMPTY = pd.Series(dtype=float)


def test_seven_conditions_are_named():
    assert len(CONDITIONS) == 7


def test_momentum_features_definitions():
    s = settings()
    bars = make_market()
    f = symbol_features(bars["AAPL"], bars["SPY"]["close"], s)
    c = bars["AAPL"]["close"]
    i = 400
    assert f["r12_1"].iloc[i] == pytest.approx(c.iloc[i - 21] / c.iloc[i - 252] - 1)
    assert f["r6_1"].iloc[i] == pytest.approx(c.iloc[i - 21] / c.iloc[i - 126] - 1)
    assert f["r3"].iloc[i] == pytest.approx(c.iloc[i] / c.iloc[i - 63] - 1)
    assert f["h52"].iloc[i] == pytest.approx(
        c.iloc[i] / bars["AAPL"]["high"].iloc[i - 251:i + 1].max())
    assert f["hi_prev"].iloc[i] == pytest.approx(c.iloc[i - 20:i].max())  # excludes today
    assert 0.5 < f["beta"].iloc[i] < 3  # synthetic betas are 1.2..2.1


def test_panel_ranks_and_eligibility():
    s = settings()
    bars = make_market()
    panel = build_panel(bars, SYMS, s)
    day = bars["SPY"].index[450]
    ranks = {sym: f.loc[day, "pct_rank"] for sym, f in panel.items()}
    scores = {sym: f.loc[day, "score"] for sym, f in panel.items()}
    best = max(scores, key=scores.get)
    assert ranks[best] == pytest.approx(1.0)
    assert all(0 < r <= 1 for r in ranks.values())
    # Too little history -> not eligible.
    assert not panel["AAPL"]["eligible"].iloc[100]
    # Price filter.
    pricey = settings(min_price=1e6)
    assert not build_panel(bars, SYMS, pricey)["AAPL"]["eligible"].any()


def test_low_beta_stocks_filtered_out():
    s = settings(min_beta=5.0)
    bars = make_market()
    panel = build_panel(bars, SYMS, s)
    assert not any(f["eligible"].iloc[-1] for f in panel.values())


def test_regime_rules():
    s = settings()
    bars = make_market()
    day = bars["SPY"].index[-1]
    up = bars["SPY"].copy()
    up["close"] = np.linspace(100, 200, len(up))
    b = {**bars, "SPY": up}
    assert regime_at(day, b, EMPTY, EMPTY, s).on
    vix = pd.Series([36.0], index=[day])
    r = regime_at(day, b, vix, EMPTY, s)
    assert not r.on and r.off  # VIX above 30 blocks entries, above 35 is risk-off
    inverted = regime_at(day, b, pd.Series([25.0], index=[day]),
                         pd.Series([20.0], index=[day]), s)
    assert not inverted.on and not inverted.off
    stale = regime_at(day, b, pd.Series([50.0], index=[day - pd.Timedelta(days=30)]),
                      EMPTY, s)
    assert stale.on and stale.vix is None  # old macro data is ignored
    down = up.copy()
    down["close"] = np.linspace(200, 100, len(up))
    r = regime_at(day, {**bars, "SPY": down}, EMPTY, EMPTY, s)
    assert not r.on and r.off


def test_signal_requires_all_seven():
    bars, day, hits = signal_case()
    s = settings()
    panel = build_panel(bars, SYMS, s)
    regime = regime_at(day, bars, EMPTY, EMPTY, s)
    ev = evaluate_day(panel, day, regime, s)[hits[0]]
    assert ev.signal and ev.n_ok == 7
    row = panel[hits[0]].loc[day].copy()
    row["vol_ratio"] = 1.0  # no volume confirmation
    assert not evaluate(hits[0], row, regime, s).signal
    regime.on = False
    assert not evaluate(hits[0], panel[hits[0]].loc[day], regime, s).signal
    d = ev.to_dict()
    assert d["signal"] == "LONG" and set(d["long"]) == set(CONDITIONS) and d["short"] == {}


def test_synthetic_etp_is_daily_reset_leverage():
    bars = make_market()
    q = bars["QQQ"]
    etp = synthetic_etp(q, 3.0, 0.0)
    r_under = q["close"].pct_change().iloc[1:]
    r_etp = etp["close"].pct_change().iloc[1:]
    assert np.allclose(r_etp, 3 * r_under)
    assert (etp["high"] >= etp[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (etp["low"] <= etp[["open", "close"]].min(axis=1) + 1e-9).all()
    # Volatility drag: a flat but choppy index loses value at 3x.
    chop = q.iloc[:200].copy()
    chop["close"] = 100 * np.where(np.arange(200) % 2 == 0, 1.0, 1.03)
    chop["open"] = chop["high"] = chop["low"] = chop["close"]
    assert synthetic_etp(chop, 3.0, 0.0)["close"].iloc[-1] < 100


def test_etp_gates():
    s = settings(etp_enabled=True)
    calm = pd.Series({"close": 110.0, "sma200": 100.0, "sma50": 105.0, "rv20": 0.12,
                      "vol126": 0.15, "r3": 0.08})
    from invest.signals import Regime

    on = Regime(True, False, 1, 1, 15.0, 0.9)
    ok, why = etp_entry_ok(calm, on, s)
    assert ok, why
    wild = calm.copy()
    wild["rv20"] = 0.35
    assert not etp_entry_ok(wild, on, s)[0]
    weak = calm.copy()
    weak["r3"] = 0.0
    assert "too weak" in etp_entry_ok(weak, on, s)[1]
    assert etp_exit_reason(wild, on, s).startswith("realised vol")
    assert etp_exit_reason(calm, Regime(False, True, 1, 1, 40.0, 1.2), s) == "regime off"
    assert etp_exit_reason(calm, on, s) is None
