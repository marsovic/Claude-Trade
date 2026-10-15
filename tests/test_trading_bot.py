import json

import pytest

from bot.config import ConfigError
from bot.trading_bot import EXIT_DEAD, BotIsDead, TradingBot, main
from tests.conftest import BAR_MS, Clock, FakeMarketClient, FakeSentiment, flat_frame, signal_frame


def make_bot(settings, frames, sentiment_score=None):
    client = FakeMarketClient(frames)
    last_close = max(int(df["close_time"].iloc[-1]) for df in frames.values())
    clock = Clock((last_close + 10_000) / 1000)
    bot = TradingBot(settings, client=client, sentiment=FakeSentiment(sentiment_score),
                     clock=clock)
    return bot, client, clock


def test_cycle_opens_position_on_full_alignment(settings):
    frames = {"AAAUSDT": signal_frame("LONG"), "FLATUSDT": flat_frame()}
    bot, client, _ = make_bot(settings, frames)
    bot.setup()
    assert bot.cycle(force_scan=True)
    pos = bot.state.positions["AAAUSDT"]
    assert pos.side == "LONG"
    assert 1 <= pos.leverage <= 5
    assert pos.stop_loss < pos.entry_price < pos.take_profit
    assert pos.risk_amount <= settings.initial_capital * settings.risk_per_trade + 1e-9
    assert "FLATUSDT" not in bot.state.positions
    kinds = [d["kind"] for d in bot.store.recent_decisions()]
    assert "scan" in kinds and "signal" in kinds and "open" in kinds
    snapshot = json.loads(bot.store.state_path.read_text())
    assert snapshot["last_scan"]["evaluated"] == 2
    assert snapshot["positions"]["AAAUSDT"]["side"] == "LONG"


def test_short_signal_opens_short(settings):
    bot, _, _ = make_bot(settings, {"BBBUSDT": signal_frame("SHORT")})
    bot.setup()
    bot.cycle(force_scan=True)
    assert bot.state.positions["BBBUSDT"].side == "SHORT"


def test_sentiment_veto_blocks_trade(settings):
    bot, _, _ = make_bot(settings, {"AAAUSDT": signal_frame("LONG")}, sentiment_score=-0.8)
    bot.setup()
    bot.cycle(force_scan=True)
    assert bot.state.positions == {}
    assert any(d["kind"] == "veto" for d in bot.store.recent_decisions())


def test_take_profit_closes_and_books_profit(settings):
    bot, client, clock = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    bot.setup()
    bot.cycle(force_scan=True)
    pos = bot.state.positions["AAAUSDT"]
    client.prices["AAAUSDT"] = pos.take_profit * 1.001
    clock.t += 30
    bot.cycle()
    assert bot.state.positions == {}
    trades = bot.store.recent_trades()
    assert len(trades) == 1 and trades[0]["reason"] == "take_profit"
    assert trades[0]["pnl"] > 0
    assert bot.state.balance > settings.initial_capital
    assert bot.state.stats["wins"] == 1
    # Cooldown: the same signal right after the exit is not traded again.
    bot.cycle(force_scan=True)
    assert "AAAUSDT" not in bot.state.positions


def test_stop_loss_loses_about_two_percent(settings):
    bot, client, clock = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    bot.setup()
    bot.cycle(force_scan=True)
    pos = bot.state.positions["AAAUSDT"]
    client.prices["AAAUSDT"] = pos.stop_loss
    clock.t += 30
    bot.cycle()
    trade = bot.store.recent_trades()[-1]
    assert trade["reason"] == "stop_loss"
    assert -0.022 * settings.initial_capital <= trade["pnl"] < 0


def test_max_hold_time_exit(settings):
    settings.max_hold_bars = 2
    bot, _, clock = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    bot.setup()
    bot.cycle(force_scan=True)
    clock.t += 3 * BAR_MS / 1000
    bot.cycle()
    assert bot.store.recent_trades()[-1]["reason"] == "time_exit"


def test_survival_rule_kills_the_bot_forever(settings):
    bot, client, clock = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    bot.setup()
    bot.cycle(force_scan=True)
    bot.state.balance = settings.survival_floor - 1  # the capital is gone
    clock.t += 30
    assert bot.cycle() is False
    assert bot.state.status == "dead"
    assert bot.state.positions == {}
    tomb = bot.store.tombstone()
    assert tomb and tomb["equity"] <= settings.survival_floor
    assert any(d["kind"] == "death" for d in bot.store.recent_decisions())

    reborn, _, _ = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    with pytest.raises(BotIsDead):
        reborn.setup()


def test_cli_refuses_to_start_a_dead_bot(settings, tmp_path, monkeypatch):
    from bot.state import StateStore

    StateStore(settings.state_dir).write_tombstone({"died_at": 1, "equity": 3.0, "trades": 9})
    monkeypatch.delenv("MODE", raising=False)
    code = main(["--state-dir", settings.state_dir, "--env-file", str(tmp_path / "none")])
    assert code == EXIT_DEAD


def test_daily_loss_limit_pauses_entries(settings):
    bot, _, _ = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    bot.setup()
    bot.state.day = "today-is-different"
    bot.check_survival({}, bot.now_ms())  # sets the day start equity
    bot.state.balance -= settings.initial_capital * 0.07
    bot.check_survival({}, bot.now_ms())
    assert bot.state.daily_halt
    bot.scan(bot.now_ms())
    assert bot.state.positions == {}


def test_state_is_resumed_and_mode_mismatch_refused(settings):
    bot, _, _ = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    bot.setup()
    bot.cycle(force_scan=True)
    resumed, _, _ = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    resumed.setup()
    assert "AAAUSDT" in resumed.state.positions

    settings.mode = "testnet"
    other, _, _ = make_bot(settings, {"AAAUSDT": signal_frame("LONG")})
    with pytest.raises(ConfigError):
        other.setup()


def test_universe_keeps_most_liquid_and_excludes(settings):
    settings.universe_size = 2
    settings.exclude_symbols = ["BBBUSDT"]
    frames = {"AAAUSDT": flat_frame(), "BBBUSDT": flat_frame(), "CCCUSDT": flat_frame(),
              "DDDUSDT": flat_frame()}
    bot, _, _ = make_bot(settings, frames)
    bot.setup()
    assert bot.state.universe == ["AAAUSDT", "CCCUSDT"]


def test_unclosed_candle_is_ignored(settings):
    frames = {"AAAUSDT": signal_frame("LONG")}
    bot, _, clock = make_bot(settings, frames)
    # The clock sits inside the last candle: it is not closed yet, so it is dropped.
    clock.t = (int(frames["AAAUSDT"]["close_time"].iloc[-1]) - 60_000) / 1000
    bot.setup()
    bot.cycle(force_scan=True)
    assert bot.state.positions == {}
