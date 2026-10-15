import pytest

from invest.config import FEE_PRESETS, FeeModel
from invest.portfolio import InvestPosition
from invest.risk import (
    breakeven_stop,
    chandelier_stop,
    pyramid_ok,
    risk_budget,
    size_position,
    stop_distance_pct,
)
from tests.invest_helpers import settings

T212 = FEE_PRESETS["trading212"]


def test_drawdown_ladder():
    s = settings(etp_enabled=True)
    assert risk_budget(100, 0.05, s).risk_per_trade == 0.04
    assert risk_budget(100, 0.05, s).slots == 3
    assert risk_budget(50, 0.05, s).slots == 2  # small equity
    lvl1 = risk_budget(100, 0.25, s)
    assert lvl1.risk_per_trade == 0.03 and lvl1.slots == 2 and lvl1.etp_allowed
    lvl2 = risk_budget(100, 0.40, s)
    assert lvl2.risk_per_trade == 0.02 and lvl2.slots == 1 and not lvl2.etp_allowed


def test_stop_distance_clamped():
    s = settings()
    assert stop_distance_pct(100, 2.0, s) == pytest.approx(0.06)
    assert stop_distance_pct(100, 0.1, s) == pytest.approx(0.04)
    assert stop_distance_pct(100, 50.0, s) == pytest.approx(0.20)


def test_sizing_aggressive_but_bounded():
    s = settings()
    budget = risk_budget(100, 0.0, s)
    sized, why = size_position(price=110.0, atr=2.2, equity=100, cash=100, open_risk_eur=0,
                               budget=budget, usd_per_eur=1.1, fees=T212, s=s)
    assert why == "ok"
    # stop 6 %: weight = min(4 % / 6 %, 1/3, 40 %) = 33 %
    assert sized.weight == pytest.approx(1 / 3, abs=0.002)
    assert sized.risk_eur == pytest.approx(sized.notional_eur * 0.06)
    assert sized.risk_eur <= 100 * s.risk_per_trade + 1e-9
    assert sized.stop_price == pytest.approx(110 * 0.94)
    assert sized.qty == pytest.approx(round(sized.qty, 4))


def test_sizing_respects_open_risk_and_cash():
    s = settings()
    budget = risk_budget(100, 0.0, s)
    sized, _ = size_position(110.0, 2.2, 100, 100, open_risk_eur=11.0, budget=budget,
                             usd_per_eur=1.1, fees=T212, s=s)
    assert sized.risk_eur <= 1.0 + 1e-6  # only 1 EUR of the 12 % budget left
    none, why = size_position(110.0, 2.2, 100, 100, open_risk_eur=12.0, budget=budget,
                              usd_per_eur=1.1, fees=T212, s=s)
    assert none is None and "open risk" in why
    poor, _ = size_position(110.0, 2.2, 100, cash=10, open_risk_eur=0, budget=budget,
                            usd_per_eur=1.1, fees=T212, s=s)
    assert poor.notional_eur <= 10


def test_whole_shares_and_fee_guard():
    s = settings(fractional=False)
    budget = risk_budget(100, 0.0, s)
    none, why = size_position(500.0, 10.0, 100, 100, 0, budget, 1.1, T212, s)
    assert none is None and "more than the slot" in why
    ibkr = FEE_PRESETS["ibkr"]
    sized, why = size_position(20.0, 0.6, 100, 100, 0, budget, 1.1, ibkr, s)
    # 1 USD minimum each way on a ~33 EUR position with a 9 % stop is too expensive
    assert sized is None and "costs" in why
    cheap = FeeModel()
    assert size_position(20.0, 0.6, 100, 100, 0, budget, 1.1, cheap, s)[0] is not None


def test_fee_model():
    ibkr = FEE_PRESETS["ibkr"]
    assert ibkr.cost(10, 30.0, foreign=True) == pytest.approx(1.0)  # min 1 USD
    assert ibkr.cost(1, 50.0, foreign=True) == pytest.approx(0.5)  # 1 % cap wins
    assert T212.cost(1, 100.0, foreign=True) == pytest.approx(0.15)
    assert T212.cost(1, 100.0, foreign=False) == 0.0


def _pos(**kw):
    base = dict(symbol="NVDA", qty=1.0, entry_price=100.0, cost_eur=91.0, initial_stop=94.0,
                stop_loss=94.0, atr=2.0, risk_amount=5.5, opened_at=0, entry_day="2026-01-02")
    base.update(kw)
    return InvestPosition(**base)


def test_chandelier_only_ratchets_up():
    s = settings()
    pos = _pos(highest=120.0)
    assert chandelier_stop(pos, 2.0, s) == pytest.approx(114.0)
    pos.stop_loss = 116.0
    assert chandelier_stop(pos, 2.0, s) == 116.0
    # Risk-off uses 2 x ATR, but 2 x 2.0 = 3.3 % is below the 4 % floor.
    assert chandelier_stop(_pos(highest=120.0), 2.0, s, mult=2.0) == pytest.approx(115.2)
    assert chandelier_stop(_pos(highest=120.0), 3.0, s, mult=2.0) == pytest.approx(114.0)
    # Never closer than the 4 % floor, even with a tiny ATR.
    assert chandelier_stop(_pos(highest=120.0), 0.1, s) == pytest.approx(120 * 0.96)


def test_pyramid_rules():
    s = settings()
    pos = _pos()
    assert pyramid_ok(pos, 109.5, True, 0.95, s)  # +1.58R, new high, top 10 %
    assert not pyramid_ok(pos, 105.0, True, 0.95, s)  # not enough profit
    assert not pyramid_ok(pos, 109.5, False, 0.95, s)  # no new high
    assert not pyramid_ok(pos, 109.5, True, 0.80, s)  # rank dropped
    pos.adds = 1
    assert not pyramid_ok(pos, 109.5, True, 0.95, s)  # only one add
    assert breakeven_stop(_pos(), 0.0015) == pytest.approx(100.3)
