"""Momentum signals for stocks: ranking, market regime and the 7 conditions.

Ranking (cross-sectional z-scores, winsorised at +/-3):

    SCORE = 0.35 z(r12_1 / vol126) + 0.25 z(r6_1 / vol126)
          + 0.15 z(r3 / vol126)    + 0.25 z(close / 52-week high)

A stock is bought only when the 7 conditions below are ALL true on the
last closed daily bar (long only):

1. ``regime``        S&P 500 (SPY) above its 200-day SMA, VIX <= 30, VIX/VIX3M <= 1
2. ``top_rank``      in the top 10 % of the ranking and SCORE > 0
3. ``abs_momentum``  12-1 and 6-1 month returns both positive
4. ``trend``         close > SMA200 and EMA20 > EMA50
5. ``near_high``     close >= 85 % of the 52-week high
6. ``breakout``      close >= highest close of the previous 20 days, volume >= 1.3x
7. ``not_extended``  close <= 25 % above SMA50 and RSI(14) <= 80

Everything is computed on CLOSED daily bars; orders go to the next open.
The same panel functions feed the live bot and the backtester.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.indicators import atr as wilder_atr
from bot.indicators import ema, rsi

CONDITIONS = ("regime", "top_rank", "abs_momentum", "trend", "near_high", "breakout",
              "not_extended")
CONDITION_LABELS = ("Regime (SPY>SMA200, VIX)", "Top 10% momentum", "Abs. momentum 12-1/6-1",
                    "Trend (SMA200, EMA20>50)", "Near 52w high", "Breakout + volume",
                    "Not overextended")
WEIGHTS = {"m12": 0.35, "m6": 0.25, "m3": 0.15, "h52": 0.25}


def symbol_features(df: pd.DataFrame, bench_close: pd.Series | None, s) -> pd.DataFrame:
    """Per-symbol indicator frame (same index as ``df``)."""
    c, h, v = df["close"], df["high"], df["volume"]
    out = pd.DataFrame(index=df.index)
    out["open"], out["high"], out["low"], out["close"] = df["open"], h, df["low"], c
    out["sma200"] = c.rolling(200).mean()
    out["sma50"] = c.rolling(50).mean()
    out["ema20"] = ema(c, 20)
    out["ema50"] = ema(c, 50)
    out["rsi14"] = rsi(c, 14)
    out["atr"] = wilder_atr(df, s.atr_period)
    out["atr_pct"] = out["atr"] / c * 100
    logret = np.log(c / c.shift(1))
    out["vol126"] = logret.rolling(126).std() * np.sqrt(252)
    out["rv20"] = logret.rolling(20).std() * np.sqrt(252)
    out["r12_1"] = c.shift(21) / c.shift(252) - 1
    out["r6_1"] = c.shift(21) / c.shift(126) - 1
    out["r3"] = c / c.shift(63) - 1
    out["h52"] = c / h.rolling(252, min_periods=200).max()
    out["hi_prev"] = c.shift(1).rolling(s.breakout_lookback).max()
    out["vol_ratio"] = v / v.shift(1).rolling(20).mean()
    out["dollar_vol"] = (c * v).rolling(20).median()
    out["max_jump"] = c.pct_change().rolling(21).max()
    out["ext50"] = c / out["sma50"] - 1
    out["bars"] = np.arange(1, len(df) + 1)
    if bench_close is not None:
        bench = np.log(bench_close / bench_close.shift(1)).reindex(df.index)
        cov = logret.rolling(252, min_periods=120).cov(bench)
        out["beta"] = cov / bench.rolling(252, min_periods=120).var()
    else:
        out["beta"] = np.nan
    return out


def build_panel(bars: dict[str, pd.DataFrame], universe: list[str], s) -> dict[str, pd.DataFrame]:
    """Features for every universe symbol, plus cross-sectional score and rank."""
    bench = bars.get(s.benchmark)
    bench_close = bench["close"] if bench is not None and not bench.empty else None
    feats = {sym: symbol_features(bars[sym], bench_close, s)
             for sym in universe if sym in bars and len(bars[sym]) > 30}
    if not feats:
        return {}
    dates = sorted(set().union(*(f.index for f in feats.values())))
    wide = {k: pd.DataFrame({sym: f[k] for sym, f in feats.items()}).reindex(dates)
            for k in ("close", "r12_1", "r6_1", "r3", "h52", "vol126", "dollar_vol",
                      "bars", "atr_pct", "max_jump", "beta")}
    # Base universe: enough history, price and liquidity. Used to rank HELD stocks,
    # so that a winner is not sold just because a big up-day made it "ineligible".
    base = (
        (wide["close"] >= s.min_price)
        & (wide["dollar_vol"] >= s.min_dollar_volume)
        & (wide["bars"] >= s.min_history)
    )
    # Entry universe: also a tradable volatility, no lottery jump, high beta.
    eligible = (
        base
        & (wide["atr_pct"] >= s.atr_min_pct) & (wide["atr_pct"] <= s.atr_max_pct)
        & (wide["max_jump"] <= s.max_daily_jump)
        & ((wide["beta"] >= s.min_beta) | (s.min_beta <= 0))
    )
    vol = wide["vol126"].where(wide["vol126"] > 0)
    parts = {
        "m12": wide["r12_1"] / vol,
        "m6": wide["r6_1"] / vol,
        "m3": wide["r3"] / vol,
        "h52": wide["h52"],
    }
    score = sum(WEIGHTS[k] * _zscore(p.where(eligible)) for k, p in parts.items())
    score = score.where(eligible)
    pct = score.rank(axis=1, pct=True)
    hold_score = sum(WEIGHTS[k] * _zscore(p.where(base)) for k, p in parts.items())
    hold_pct = hold_score.where(base).rank(axis=1, pct=True)
    for sym, f in feats.items():
        f["eligible"] = eligible[sym].reindex(f.index).fillna(False).astype(bool)
        f["score"] = score[sym].reindex(f.index)
        f["pct_rank"] = pct[sym].reindex(f.index)
        f["pct_rank_hold"] = hold_pct[sym].reindex(f.index)
    return feats


def _zscore(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).clip(-3, 3).fillna(0.0).where(frame.notna())


# ---------------------------------------------------------------- regime
@dataclass
class Regime:
    on: bool
    off: bool  # "risk-off": exits / tighter stops (stronger than not-on)
    spy_close: float | None
    spy_sma: float | None
    vix: float | None
    vix_term: float | None
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"on": self.on, "off": self.off, "spy_close": self.spy_close,
                "spy_sma": self.spy_sma, "vix": self.vix, "vix_term": self.vix_term,
                "reasons": self.reasons}


def regime_at(day: pd.Timestamp, bars: dict[str, pd.DataFrame], vix: pd.Series,
              vix3m: pd.Series, s) -> Regime:
    reasons = []
    spy = bars.get(s.benchmark)
    spy_close = spy_sma = None
    above = above_prev = False
    if spy is not None and not spy.empty:
        closes = spy["close"][spy.index <= day]
        sma = closes.rolling(s.regime_sma).mean()
        if len(closes) >= s.regime_sma + 1 and not np.isnan(sma.iloc[-1]):
            spy_close, spy_sma = float(closes.iloc[-1]), float(sma.iloc[-1])
            above = bool(closes.iloc[-1] > sma.iloc[-1])
            above_prev = bool(closes.iloc[-2] > sma.iloc[-2])
        else:
            reasons.append(f"{s.benchmark} history too short")
    else:
        reasons.append(f"no {s.benchmark} data")
    v = _last(vix, day)
    v3 = _last(vix3m, day)
    term = v / v3 if v is not None and v3 else None
    on = above
    if not above:
        reasons.append(f"{s.benchmark} below its {s.regime_sma}-day SMA")
    if v is not None and v > s.vix_max_entry:
        on = False
        reasons.append(f"VIX {v:.1f} > {s.vix_max_entry:g}")
    if term is not None and term > s.vix_term_max:
        on = False
        reasons.append(f"VIX term structure inverted ({term:.2f})")
    off = (spy_close is not None and not above and not above_prev) or (
        v is not None and v > s.vix_exit)
    return Regime(on=bool(on), off=bool(off), spy_close=spy_close, spy_sma=spy_sma, vix=v,
                  vix_term=term, reasons=reasons)


def _last(series: pd.Series, day: pd.Timestamp) -> float | None:
    if series is None or series.empty:
        return None
    past = series[series.index <= day]
    # Ignore stale macro data (more than 7 days old).
    if past.empty or (day - past.index[-1]).days > 7:
        return None
    return float(past.iloc[-1])


# -------------------------------------------------------------- conditions
@dataclass
class StockEval:
    symbol: str
    day: pd.Timestamp
    price: float
    atr: float
    atr_pct: float
    score: float | None
    pct_rank: float | None
    eligible: bool
    checks: dict[str, bool]
    values: dict[str, float]

    @property
    def signal(self) -> bool:
        return self.eligible and all(self.checks.values())

    @property
    def n_ok(self) -> int:
        return sum(self.checks.values())

    def to_dict(self) -> dict:
        """Shape understood by the dashboard scanner."""
        return {
            "symbol": self.symbol,
            "signal": "LONG" if self.signal else None,
            "price": self.price,
            "atr_pct": round(self.atr_pct, 2),
            "long": self.checks,
            "short": {},
            "long_score": self.n_ok,
            "short_score": 0,
            "score": None if self.score is None else round(self.score, 3),
            "values": {k: round(v, 4) for k, v in self.values.items()},
            "eligible": self.eligible,
        }


def evaluate(sym: str, row: pd.Series, regime: Regime, s) -> StockEval:
    def val(key, default=np.nan):
        x = row.get(key, default)
        return default if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)

    close = val("close")
    score = val("score", np.nan)
    pct = val("pct_rank", np.nan)
    checks = {
        "regime": bool(regime.on),
        "top_rank": bool(pct >= 1 - s.top_pct_entry and score > 0),
        "abs_momentum": bool(val("r12_1") > 0 and val("r6_1") > 0),
        "trend": bool(close > val("sma200") and val("ema20") > val("ema50")),
        "near_high": bool(val("h52") >= s.near_high_min),
        "breakout": bool(close >= val("hi_prev") and val("vol_ratio") >= s.volume_factor),
        "not_extended": bool(val("ext50") <= s.max_extension and val("rsi14") <= s.rsi_max),
    }
    values = {"rsi": val("rsi14", 0.0), "vol_ratio": val("vol_ratio", 0.0),
              "r12_1": val("r12_1", 0.0), "r6_1": val("r6_1", 0.0), "h52": val("h52", 0.0),
              "ext50": val("ext50", 0.0), "pct_rank": 0.0 if np.isnan(pct) else pct}
    return StockEval(symbol=sym, day=row.name, price=close, atr=val("atr", 0.0),
                     atr_pct=val("atr_pct", 0.0), score=None if np.isnan(score) else score,
                     pct_rank=None if np.isnan(pct) else pct,
                     eligible=bool(row.get("eligible", False)), checks=checks, values=values)


def evaluate_day(panel: dict[str, pd.DataFrame], day: pd.Timestamp, regime: Regime, s
                 ) -> dict[str, StockEval]:
    out = {}
    for sym, feats in panel.items():
        if day in feats.index:
            out[sym] = evaluate(sym, feats.loc[day], regime, s)
    return out


# ------------------------------------------------------- leveraged ETP sleeve
def synthetic_etp(underlying: pd.DataFrame, leverage: float, annual_cost: float,
                  start_price: float = 100.0) -> pd.DataFrame:
    """Daily-reset leveraged product simulated from its underlying.

    Used by the simulator/backtester (real brokers trade the real ETP).
    Captures the volatility drag: log(LETF) ~ L log(index) - (L^2-L) sigma^2 T / 2.
    """
    c = underlying["close"]
    prev = c.shift(1)
    daily_cost = annual_cost / 252
    ret = leverage * (c / prev - 1) - daily_cost
    close = start_price * (1 + ret.fillna(0)).cumprod()
    pclose = close.shift(1).fillna(start_price)
    out = pd.DataFrame(index=underlying.index)
    for col in ("open", "high", "low"):
        move = (underlying[col] / prev - 1).fillna(0)
        out[col] = pclose * (1 + leverage * move)
    out["close"] = close
    out["high"] = out[["high", "close", "open"]].max(axis=1)
    out["low"] = out[["low", "close", "open"]].min(axis=1).clip(lower=0.01)
    out["volume"] = underlying["volume"]
    return out


def etp_entry_ok(under: pd.Series, regime: Regime, s) -> tuple[bool, str]:
    """Leveraged ETP only in calm, strong uptrends (variance drag otherwise)."""
    lev = s.etp_leverage
    rv_max = s.etp_rv_max if lev >= 3 else s.etp_rv_max + 0.05
    if not regime.on:
        return False, "regime off"
    if not (under["close"] > under["sma200"] and under["close"] > under["sma50"]):
        return False, "underlying below SMA50/SMA200"
    if not under["rv20"] < rv_max:
        return False, f"realised vol {under['rv20']:.0%} >= {rv_max:.0%}"
    sigma2 = under["vol126"] ** 2
    trend = np.log(1 + under["r3"]) * 4  # annualised 3-month log trend
    needed = (1.5 * sigma2 + 0.03) if lev >= 3 else (sigma2 + 0.02)
    if not trend >= needed:
        return False, f"trend {trend:.0%} too weak for x{lev:g} (needs {needed:.0%})"
    if regime.vix is not None and regime.vix >= s.etp_vix_max:
        return False, f"VIX {regime.vix:.1f} >= {s.etp_vix_max:g}"
    return True, "calm uptrend"


def etp_exit_reason(under: pd.Series, regime: Regime, s) -> str | None:
    if regime.off:
        return "regime off"
    if under["rv20"] > s.etp_rv_exit:
        return f"realised vol {under['rv20']:.0%} > {s.etp_rv_exit:.0%}"
    if under["close"] < under["sma50"]:
        return "underlying below SMA50"
    if regime.vix is not None and regime.vix > s.etp_vix_exit:
        return f"VIX {regime.vix:.1f} > {s.etp_vix_exit:g}"
    return None
