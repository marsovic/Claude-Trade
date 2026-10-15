"""Synthetic markets and fakes for the investment bot tests (no network)."""

from __future__ import annotations

from functools import cache

import numpy as np
import pandas as pd

from invest.config import InvestSettings
from invest.data import MarketData
from invest.signals import build_panel, evaluate_day, regime_at

SYMS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "NFLX", "AMD",
        "ADBE", "CRM"]


def ohlcv(rets: np.ndarray, days: pd.DatetimeIndex, start: float, vol_base: float,
          rng: np.random.Generator) -> pd.DataFrame:
    close = start * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, len(days)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, len(days))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, len(days))))
    volume = rng.lognormal(np.log(vol_base), 0.35, len(days))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": volume}, index=days)


def make_market(n_days: int = 520, seed: int = 3, n_syms: int = 12,
                start: str = "2024-01-02") -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    mkt = rng.normal(0.0006, 0.009, n_days)
    bars = {"SPY": ohlcv(mkt, days, 450, 8e7, rng),
            "QQQ": ohlcv(1.15 * mkt + rng.normal(0, 0.002, n_days), days, 400, 5e7, rng)}
    for i, sym in enumerate(SYMS[:n_syms]):
        beta = 1.2 + 0.08 * i
        drift = 0.0012 if i % 3 == 0 else (-0.0004 if i % 3 == 1 else 0.0003)
        r = beta * mkt + drift + rng.normal(0, 0.015, n_days)
        bars[sym] = ohlcv(r, days, 50 + 20 * i, 3e6, rng)
    return bars


def settings(**kw) -> InvestSettings:
    base = {"min_beta": 0.0, "etp_enabled": False, "sentiment_enabled": False,
            "state_dir": kw.pop("state_dir", "state_invest_test")}
    base.update(kw)
    s = InvestSettings(**base)
    s.validate()
    return s


def market_data(bars, vix=None) -> MarketData:
    return MarketData(bars=bars, vix=vix if vix is not None else pd.Series(dtype=float))


@cache
def _signal_case(seed: int):
    bars = make_market(seed=seed)
    s = settings()
    panel = build_panel(bars, SYMS, s)
    for day in bars["SPY"].index[300:-5]:
        regime = regime_at(day, bars, pd.Series(dtype=float), pd.Series(dtype=float), s)
        if not regime.on:
            continue
        evs = evaluate_day(panel, day, regime, s)
        hits = [e.symbol for e in evs.values() if e.signal]
        if hits:
            return day, hits
    return None, []


def signal_case(seeds=range(1, 40)):
    """(bars, day, symbols) where at least one stock has 7/7 on ``day``."""
    for seed in seeds:
        day, hits = _signal_case(seed)
        if day is not None:
            return make_market(seed=seed), day, hits
    raise RuntimeError("no synthetic signal found")
