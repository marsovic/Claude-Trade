import pytest

from bot.config import Settings
from bot.risk import (
    Position,
    check_exit,
    liquidation_safe_leverage,
    plan_trade,
    realize,
    update_stops,
    volatility_leverage,
)
from tests.conftest import symbol_info

INFO = symbol_info("AAAUSDT", step="0.001", tick="0.01")


def test_leverage_adapts_to_volatility_and_is_capped():
    s = Settings()  # target 3.0, max x5
    assert volatility_leverage(0.2, s) == 5  # calm market: max leverage
    assert volatility_leverage(1.0, s) == 3
    assert volatility_leverage(2.5, s) == 1  # agitated market: no leverage
    assert volatility_leverage(0.0, s) == 1
    for atr_pct in (0.01, 0.1, 0.5, 1, 3, 10):
        assert 1 <= volatility_leverage(atr_pct, s) <= s.max_leverage


def test_liquidation_guard_lowers_leverage():
    s = Settings()
    # 9 % stop at x5: liquidation ~19 % away, half of it is 9.5 % -> ok
    assert liquidation_safe_leverage(5, 0.09, s) == 5
    # 15 % stop: x5 too close to liquidation, x3 is fine
    assert liquidation_safe_leverage(5, 0.15, s) == 3


def test_position_risks_two_percent_including_fees():
    s = Settings(max_margin_per_trade=1.0, max_total_margin=1.0)
    plan, reason = plan_trade("AAAUSDT", "LONG", price=100.0, atr=0.5, atr_pct=0.5,
                              equity=50.0, used_margin=0.0, info=INFO, s=s)
    assert reason == "ok"
    assert plan.stop_loss == pytest.approx(100 - 0.75)
    assert plan.take_profit == pytest.approx(100 + 0.75 * 2)
    loss = plan.qty * (plan.price - plan.stop_loss) + plan.qty * s.fee_rate * (
        plan.price + plan.stop_loss)
    assert loss <= 50 * 0.02 + 1e-9
    assert plan.risk_amount == pytest.approx(1.0, rel=0.01)
    assert plan.leverage == 5
    assert plan.margin == pytest.approx(plan.notional / plan.leverage)


def test_short_plan_mirrors_long():
    s = Settings(max_margin_per_trade=1.0, max_total_margin=1.0)
    plan, _ = plan_trade("AAAUSDT", "SHORT", 100.0, 0.5, 0.5, 50.0, 0.0, INFO, s)
    assert plan.stop_loss > plan.price > plan.take_profit


def test_margin_cap_reduces_size_never_increases_risk():
    s = Settings(max_margin_per_trade=0.2)
    plan, _ = plan_trade("AAAUSDT", "LONG", 100.0, 0.5, 0.5, 50.0, 0.0, INFO, s)
    assert plan.margin <= 50 * 0.2 + 1e-9
    assert plan.risk_amount < 1.0


def test_refuses_when_no_free_margin_or_below_minimum():
    s = Settings()
    plan, reason = plan_trade("AAAUSDT", "LONG", 100.0, 0.5, 0.5, 50.0, 45.0, INFO, s)
    assert plan is None and "margin" in reason
    big_min = symbol_info("BTCUSDT", step="0.001", min_notional="100")
    plan, reason = plan_trade("BTCUSDT", "LONG", 100.0, 0.5, 0.5, 50.0, 0.0, big_min,
                              Settings(max_leverage=1))
    assert plan is None and "minimum" in reason


def test_quantity_rounded_down_to_step():
    s = Settings(max_margin_per_trade=1.0, max_total_margin=1.0)
    info = symbol_info("AAAUSDT", step="1")
    plan, _ = plan_trade("AAAUSDT", "LONG", 3.0, 0.02, 0.6, 50.0, 0.0, info, s)
    assert plan.qty == int(plan.qty)


def _long(**kw):
    base = dict(symbol="AAAUSDT", side="LONG", qty=1.0, entry_price=100.0, leverage=5,
                stop_loss=99.0, take_profit=102.0, initial_stop=99.0, atr=0.5, margin=20.0,
                risk_amount=1.0, opened_at=0)
    base.update(kw)
    return Position(**base)


def test_breakeven_then_trailing():
    s = Settings()
    pos = _long()
    assert not update_stops(pos, 100.5, s)  # +0.5R: nothing
    assert update_stops(pos, 101.0, s)  # +1R: break-even (+ fees)
    assert pos.stop_loss == pytest.approx(100 * (1 + 2 * s.fee_rate))
    assert update_stops(pos, 101.8, s)  # +1.8R: trailing at best - 1.5 ATR
    assert pos.stop_loss == pytest.approx(101.8 - 0.75)
    assert not update_stops(pos, 101.2, s)  # a pull-back never loosens the stop
    assert pos.stop_loss == pytest.approx(101.05)


def test_exit_rules_stop_wins_and_gaps_fill_at_open():
    pos = _long()
    assert check_exit(pos, high=101.0, low=99.5) is None
    assert check_exit(pos, high=102.5, low=98.5) == (99.0, "stop_loss")  # both: stop first
    assert check_exit(pos, high=102.5, low=100.0) == (102.0, "take_profit")
    assert check_exit(pos, high=98.0, low=97.0, open_=97.5) == (97.5, "stop_loss")  # gap
    short = _long(side="SHORT", stop_loss=101.0, initial_stop=101.0, take_profit=98.0)
    assert check_exit(short, high=101.5, low=100.0) == (101.0, "stop_loss")
    assert check_exit(short, high=99.0, low=97.0) == (98.0, "take_profit")


def test_realize_includes_fees_and_isolated_margin_cap():
    s = Settings(slippage_bps=0.0)
    pos = _long(entry_fee=100 * 0.0005)
    trade, cash = realize(pos, 102.0, 1, "take_profit", s)
    assert trade.pnl == pytest.approx(2.0 - 0.05 - 102 * 0.0005)
    assert cash == pytest.approx(2.0 - 102 * 0.0005)
    assert trade.r_multiple == pytest.approx(trade.pnl / 1.0)
    crash, _ = realize(pos, 50.0, 1, "stop_loss", s)
    assert crash.pnl >= -pos.margin - crash.fees - 1e-9
