import pytest

from bot.config import LIVE_CONFIRMATION, ConfigError, Settings
from bot.risk import ClosedTrade
from bot.state import BotState, StateStore
from dashboard.dashboard import create_app, downsample


def test_dashboard_serves_page_and_state(tmp_path):
    store = StateStore(tmp_path)
    state = BotState.fresh("paper", 50.0)
    state.equity_history = [[i * 60_000, 50 + i * 0.01] for i in range(5000)]
    state.income_seen = [1, 2, 3]
    store.save(state)
    store.log_decision("open", "LONG x3", "BTCUSDT", "LONG")
    store.log_trade(ClosedTrade("BTCUSDT", "LONG", 1, 100, 102, 3, 33, 0, 1, 1.9, 0.1, 1.9,
                                "take_profit"))
    client = create_app(tmp_path).test_client()

    page = client.get("/")
    assert page.status_code == 200 and b"Survival" in page.data
    assert "Content-Security-Policy" in page.headers

    data = client.get("/api/state").get_json()
    assert data["state"]["mode"] == "paper"
    assert len(data["state"]["equity_history"]) <= 1500
    assert "income_seen" not in data["state"]
    assert data["decisions"][0]["kind"] == "open"
    assert data["trades"][0]["reason"] == "take_profit"
    assert data["tombstone"] is None


def test_dashboard_without_state(tmp_path):
    data = create_app(tmp_path / "empty").test_client().get("/api/state").get_json()
    assert data["state"] is None and data["decisions"] == []


def test_downsample_keeps_last_point():
    pts = [[i, i] for i in range(10_000)]
    out = downsample(pts, 100)
    assert len(out) == 100 and out[-1] == pts[-1] and out[0] == pts[0]


def test_live_mode_is_locked_by_default(monkeypatch, tmp_path):
    for key in ("MODE", "LIVE_TRADING", "BINANCE_API_KEY", "BINANCE_API_SECRET"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigError, match="API_KEY"):
        Settings.from_env(str(tmp_path / "none"), mode="live")
    with pytest.raises(ConfigError, match="locked"):
        Settings.from_env(str(tmp_path / "none"), mode="live", binance_api_key="k",
                          binance_api_secret="s")
    s = Settings.from_env(str(tmp_path / "none"), mode="live", binance_api_key="k",
                          binance_api_secret="s", live_trading=LIVE_CONFIRMATION)
    assert s.mode == "live" and s.base_url == "https://fapi.binance.com"


def test_env_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv("MAX_LEVERAGE", "3")
    monkeypatch.setenv("RISK_PER_TRADE", "0.01")
    monkeypatch.setenv("TRAILING", "false")
    monkeypatch.setenv("EXCLUDE_SYMBOLS", "aaausdt, bbbusdt")
    s = Settings.from_env(str(tmp_path / "none"))
    assert s.max_leverage == 3 and s.risk_per_trade == 0.01 and s.trailing is False
    assert s.exclude_symbols == ["AAAUSDT", "BBBUSDT"]
    assert s.mode == "paper"


def test_unsafe_values_rejected():
    for kwargs in ({"risk_per_trade": 0.2}, {"max_leverage": 50}, {"mode": "yolo"},
                   {"survival_floor": 60.0}, {"ema_fast": 60}):
        with pytest.raises(ConfigError):
            Settings(**kwargs).validate()


def test_public_dict_hides_secrets():
    s = Settings(binance_api_secret="very-secret", x_bearer_token="tok")
    public = s.public_dict()
    assert public["binance_api_secret"] is True and public["x_bearer_token"] is True
    assert "very-secret" not in str(public)


def test_journal_tail_reads_last_entries_in_order(tmp_path):
    store = StateStore(tmp_path)
    for i in range(3000):
        store.log_decision("scan", f"message {i} " + "x" * 50)
    last = store.recent_decisions(10)
    assert [d["message"].split()[1] for d in last] == [str(i) for i in range(2990, 3000)]
    assert len(store.recent_decisions(5000)) == 3000


def test_dashboard_token_and_cors_for_a_phone_app(tmp_path):
    from dashboard.dashboard import create_app

    client = create_app(tmp_path, token="s3cret", cors_origins="https://app.example").test_client()
    assert client.get("/api/state").status_code == 401
    assert client.get("/api/state?token=nope").status_code == 401
    ok = client.get("/api/state", headers={"Authorization": "Bearer s3cret",
                                           "Origin": "https://app.example"})
    assert ok.status_code == 200
    assert ok.headers["Access-Control-Allow-Origin"] == "https://app.example"
    pre = client.options("/api/state", headers={"Origin": "https://app.example"})
    assert pre.status_code == 204 and "Authorization" in pre.headers[
        "Access-Control-Allow-Headers"]
    other = client.get("/api/state?token=s3cret", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in other.headers
    assert client.get("/").status_code == 200  # the page loads; it sends the token itself
