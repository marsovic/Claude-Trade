from bot.config import Settings
from bot.indicators import add_indicators
from bot.strategy import CONDITIONS, conditions_frame, evaluate_last
from tests.conftest import flat_frame, make_frame, signal_frame


def test_seven_conditions():
    assert len(CONDITIONS) == 7


def test_long_signal_requires_all_seven():
    s = Settings()
    ev = evaluate_last("AAAUSDT", add_indicators(signal_frame("LONG"), s), s)
    assert ev.signal == "LONG"
    assert all(ev.long_checks.values())
    assert ev.long_score == 7
    assert ev.short_score < 7
    assert ev.atr > 0 and ev.price > 0


def test_short_signal_requires_all_seven():
    s = Settings()
    ev = evaluate_last("BBBUSDT", add_indicators(signal_frame("SHORT"), s), s)
    assert ev.signal == "SHORT"
    assert all(ev.short_checks.values())


def test_one_failing_condition_blocks_the_trade():
    s = Settings()
    df = signal_frame("LONG")
    df.loc[df.index[-1], "volume"] = 900.0  # no volume confirmation any more
    ev = evaluate_last("AAAUSDT", add_indicators(df, s), s)
    assert ev.signal is None
    assert ev.long_checks["volume"] is False
    assert ev.long_score == 6


def test_atr_filter_blocks_wild_markets():
    s = Settings(atr_max_pct=0.01)
    ev = evaluate_last("AAAUSDT", add_indicators(signal_frame("LONG"), s), s)
    assert ev.signal is None
    assert ev.long_checks["atr"] is False


def test_flat_market_gives_no_signal():
    s = Settings()
    conds = conditions_frame(add_indicators(flat_frame(), s), s)
    assert conds["signal"].isna().all()


def test_no_signal_during_warmup():
    s = Settings()
    df = make_frame(n=150, seed=0)
    df["volume"] = 5000.0
    conds = conditions_frame(add_indicators(df, s), s)
    assert conds["signal"].isna().all()


def test_evaluation_serialises():
    s = Settings()
    ev = evaluate_last("AAAUSDT", add_indicators(signal_frame("LONG"), s), s)
    d = ev.to_dict()
    assert d["signal"] == "LONG" and d["long_score"] == 7
    assert set(d["long"]) == set(CONDITIONS)
