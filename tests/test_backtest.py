import json

import numpy as np
import pandas as pd
import pytest

from bot.backtest import load_csv, main, run_backtest
from bot.config import Settings
from tests.conftest import BAR_MS, T0


def regime_frame(seed: int, n: int = 3000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1, 0, 1], n // 300 + 1), 300)[:n]
    r = regime * 0.0005 + rng.normal(0, 0.004, n)
    close = 50 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    t = T0 + np.arange(n) * BAR_MS
    return pd.DataFrame({"open_time": t, "open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.lognormal(7, 0.4, n),
                         "close_time": t + BAR_MS - 1})


@pytest.fixture(scope="module")
def frames():
    return {f"S{i}USDT": regime_frame(i) for i in range(3)}


def test_backtest_trades_and_respects_risk(frames):
    s = Settings()
    res = run_backtest(frames, s)
    assert res.trades, "expected some trades on trending synthetic data"
    for t in res.trades:
        assert 1 <= t.leverage <= s.max_leverage
        assert t.closed_at >= t.opened_at
        # A stop costs at most ~1R (slippage aside); never a blow-up.
        assert t.r_multiple > -1.3
    assert len({(t.symbol, t.opened_at) for t in res.trades}) == len(res.trades)
    stats = res.stats()
    assert stats["trades"] == len(res.trades)
    assert stats["max_drawdown_pct"] >= 0
    assert not res.died


def test_entries_happen_at_next_bar_open(frames):
    s = Settings(slippage_bps=0.0)
    res = run_backtest(frames, s)
    for t in res.trades[:20]:
        df = frames[t.symbol]
        row = df[df["open_time"] == t.opened_at].iloc[0]
        assert t.entry_price == pytest.approx(row["open"])


def test_max_open_positions_never_exceeded(frames):
    s = Settings(max_open_positions=1)
    res = run_backtest(frames, s)
    spans = sorted((t.opened_at, t.closed_at) for t in res.trades)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert start >= end


def test_survival_rule_in_backtest(frames):
    s = Settings(survival_floor=49.9)  # first losing trade kills the bot
    res = run_backtest(frames, s)
    assert res.died
    assert res.final_equity <= 49.9
    assert res.stats()["died_at"] is not None


def test_fear_greed_veto(frames):
    s = Settings()
    extreme_fear = pd.Series({T0 - 86_400_000: -0.9})
    res = run_backtest(frames, s, fng=extreme_fear)
    assert all(t.side == "SHORT" for t in res.trades)
    assert res.vetoes > 0


def test_cli_offline_csv(tmp_path, frames):
    data_dir = tmp_path / "csv"
    data_dir.mkdir()
    for sym, df in frames.items():
        df.drop(columns=["close_time"]).to_csv(data_dir / f"{sym}.csv", index=False)
    loaded = load_csv(data_dir / "S0USDT.csv")
    assert (loaded["close_time"] - loaded["open_time"]).iloc[0] == BAR_MS - 1
    out = tmp_path / "out"
    code = main(["--csv-dir", str(data_dir), "--out", str(out),
                 "--env-file", str(tmp_path / "none")])
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())
    assert summary["trades"] > 0
    assert (out / "trades.csv").read_text().startswith("symbol,side")
    assert (out / "equity.csv").exists()
