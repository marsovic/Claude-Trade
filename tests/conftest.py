from __future__ import annotations

import sys
from decimal import Decimal
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.binance_client import SymbolInfo  # noqa: E402
from bot.config import Settings  # noqa: E402
from bot.indicators import add_indicators  # noqa: E402
from bot.sentiment import Reading, SentimentResult  # noqa: E402
from bot.strategy import conditions_frame  # noqa: E402

BAR_MS = 900_000
T0 = 1_888_888 * BAR_MS  # aligned on a 15m boundary


def make_frame(n: int = 400, seed: int = 0, sign: int = 1, drift: float = 0.0006,
               vol: float = 0.004, start: float = 100.0, t0: int = T0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = sign * drift + rng.normal(0, vol, n)
    close = start * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    volume = rng.uniform(800, 1200, n)
    t = t0 + np.arange(n) * BAR_MS
    return pd.DataFrame({"open_time": t, "open": open_, "high": high, "low": low,
                         "close": close, "volume": volume, "close_time": t + BAR_MS - 1})


@cache
def _signal_seed(side: str) -> int:
    s = Settings()
    sign = 1 if side == "LONG" else -1
    for seed in range(500):
        df = make_frame(seed=seed, sign=sign)
        df.loc[df.index[-1], "volume"] = 3000.0
        if conditions_frame(add_indicators(df, s), s)["signal"].iloc[-1] == side:
            return seed
    raise RuntimeError("no seed produces a signal")


def signal_frame(side: str = "LONG", start: float = 100.0) -> pd.DataFrame:
    """A frame whose last bar has all 7 conditions aligned for ``side``."""
    df = make_frame(seed=_signal_seed(side), sign=1 if side == "LONG" else -1, start=start)
    df.loc[df.index[-1], "volume"] = 3000.0
    return df


def flat_frame(n: int = 400, price: float = 100.0) -> pd.DataFrame:
    df = make_frame(n=n, seed=99, drift=0.0, vol=0.002, start=price)
    return df


def frame_to_rows(df: pd.DataFrame) -> list[list]:
    return [[int(r.open_time), str(r.open), str(r.high), str(r.low), str(r.close),
             str(r.volume), int(r.close_time), "0", 0, "0", "0", "0"]
            for r in df.itertuples()]


def symbol_info(symbol: str, step: str = "0.001", tick: str = "0.0001",
                min_notional: str = "5") -> SymbolInfo:
    return SymbolInfo(symbol=symbol, base_asset=symbol.removesuffix("USDT"),
                      quote_asset="USDT", tick_size=Decimal(tick),
                      step_size=Decimal(step), min_qty=Decimal(step),
                      min_notional=Decimal(min_notional))


class FakeMarketClient:
    """Public endpoints of BinanceFuturesClient, backed by synthetic frames."""

    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.frames = frames
        self.prices = {sym: float(df["close"].iloc[-1]) for sym, df in frames.items()}
        self.kline_calls = 0

    def symbol_infos(self):
        return {sym: symbol_info(sym) for sym in self.frames}

    def ticker_24h(self):
        return [{"symbol": sym, "quoteVolume": str(1e9 - i)}
                for i, sym in enumerate(self.frames)]

    def ticker_prices(self):
        return dict(self.prices)

    def klines(self, symbol, interval, limit=300, start_time=None, end_time=None):
        self.kline_calls += 1
        return frame_to_rows(self.frames[symbol].tail(limit))


class FakeSentiment:
    def __init__(self, score: float | None = None):
        self.score = score
        self.calls = []

    def evaluate(self, asset):
        self.calls.append(asset)
        return SentimentResult(asset, self.score,
                               [Reading("fake", self.score, 1.0, "test")])

    def allows(self, side, result):
        if result.score is None:
            return True, "no data"
        if side == "LONG" and result.score < -0.25:
            return False, "bearish veto"
        if side == "SHORT" and result.score > 0.25:
            return False, "bullish veto"
        return True, "ok"

    def market_snapshot(self):
        return {"enabled": True, "fear_greed": {"value": 55, "classification": "Greed"},
                "trending": ["BTC"], "sources": {}}


class Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(state_dir=str(tmp_path / "state"))
    s.validate()
    return s
