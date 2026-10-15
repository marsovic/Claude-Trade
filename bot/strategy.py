"""The 7-condition entry filter.

A position is only considered when **all seven** conditions agree, either all
bullish (LONG) or all bearish (SHORT):

1. ``ema_cross``  fast EMA above (below) slow EMA            -> short-term trend
2. ``ema_trend``  close above (below) the long EMA           -> long-term trend
3. ``rsi``        RSI in the momentum zone, not overextended
4. ``macd``       MACD above (below) its signal, histogram growing in that direction
5. ``bollinger``  close between the middle and the upper (lower) band
6. ``volume``     volume of the bar above its recent average  -> participation
7. ``atr``        volatility inside a tradable range          -> neither dead nor chaotic

The same vectorised function feeds the live loop (last closed bar) and the
backtester (every bar), so both always use identical rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

CONDITIONS = ("ema_cross", "ema_trend", "rsi", "macd", "bollinger", "volume", "atr")
LONG, SHORT = "LONG", "SHORT"


def conditions_frame(df: pd.DataFrame, s) -> pd.DataFrame:
    """Boolean columns ``long_<cond>`` / ``short_<cond>`` + ``signal`` column.

    ``df`` must already contain the columns from :func:`indicators.add_indicators`.
    """
    c = df["close"]
    hist = df["macd_hist"]
    hist_prev = hist.shift(1)
    atr_ok = (df["atr_pct"] >= s.atr_min_pct) & (df["atr_pct"] <= s.atr_max_pct)
    vol_ok = df["vol_ratio"] >= s.volume_factor

    out = pd.DataFrame(index=df.index)
    out["long_ema_cross"] = df["ema_fast"] > df["ema_slow"]
    out["long_ema_trend"] = c > df["ema_trend"]
    out["long_rsi"] = (df["rsi"] > s.rsi_long_min) & (df["rsi"] < s.rsi_long_max)
    out["long_macd"] = (df["macd"] > df["macd_signal"]) & (hist > hist_prev)
    out["long_bollinger"] = (c > df["bb_mid"]) & (c < df["bb_upper"])
    out["long_volume"] = vol_ok
    out["long_atr"] = atr_ok

    out["short_ema_cross"] = df["ema_fast"] < df["ema_slow"]
    out["short_ema_trend"] = c < df["ema_trend"]
    out["short_rsi"] = (df["rsi"] > s.rsi_short_min) & (df["rsi"] < s.rsi_short_max)
    out["short_macd"] = (df["macd"] < df["macd_signal"]) & (hist < hist_prev)
    out["short_bollinger"] = (c < df["bb_mid"]) & (c > df["bb_lower"])
    out["short_volume"] = vol_ok
    out["short_atr"] = atr_ok

    out = out.astype(bool)
    long_all = out[[f"long_{k}" for k in CONDITIONS]].all(axis=1)
    short_all = out[[f"short_{k}" for k in CONDITIONS]].all(axis=1)
    out["long_score"] = out[[f"long_{k}" for k in CONDITIONS]].sum(axis=1)
    out["short_score"] = out[[f"short_{k}" for k in CONDITIONS]].sum(axis=1)
    signal = pd.Series(None, index=df.index, dtype=object)
    signal[long_all] = LONG
    signal[short_all] = SHORT
    # Not enough history for the long EMA: never trade on a warm-up value.
    warm = pd.Series(range(len(df)), index=df.index) >= s.ema_trend
    signal[~warm] = None
    out["signal"] = signal
    return out


@dataclass
class Evaluation:
    """Result of evaluating the last closed bar of one symbol."""

    symbol: str
    signal: str | None
    price: float
    atr: float
    atr_pct: float
    long_checks: dict[str, bool] = field(default_factory=dict)
    short_checks: dict[str, bool] = field(default_factory=dict)
    values: dict[str, float] = field(default_factory=dict)
    bar_time: int = 0

    @property
    def long_score(self) -> int:
        return sum(self.long_checks.values())

    @property
    def short_score(self) -> int:
        return sum(self.short_checks.values())

    @property
    def strength(self) -> float:
        """Tie-breaker between simultaneous signals: volume participation."""
        return float(self.values.get("vol_ratio", 0.0))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "signal": self.signal,
            "price": self.price,
            "atr_pct": round(self.atr_pct, 3),
            "long": self.long_checks,
            "short": self.short_checks,
            "long_score": self.long_score,
            "short_score": self.short_score,
            "values": {k: round(v, 6) for k, v in self.values.items()},
            "bar_time": self.bar_time,
        }


def evaluate_last(symbol: str, df: pd.DataFrame, s) -> Evaluation:
    """Evaluate the last row of an indicator frame (must be a *closed* bar)."""
    conds = conditions_frame(df, s)
    row = df.iloc[-1]
    crow = conds.iloc[-1]
    values = {
        "close": float(row["close"]),
        "ema_fast": float(row["ema_fast"]),
        "ema_slow": float(row["ema_slow"]),
        "ema_trend": float(row["ema_trend"]),
        "rsi": float(row["rsi"]) if pd.notna(row["rsi"]) else 0.0,
        "macd_hist": float(row["macd_hist"]),
        "bb_pct": _bb_pct(row),
        "vol_ratio": float(row["vol_ratio"]) if pd.notna(row["vol_ratio"]) else 0.0,
        "atr_pct": float(row["atr_pct"]) if pd.notna(row["atr_pct"]) else 0.0,
    }
    return Evaluation(
        symbol=symbol,
        signal=crow["signal"] if isinstance(crow["signal"], str) else None,
        price=float(row["close"]),
        atr=float(row["atr"]) if pd.notna(row["atr"]) else 0.0,
        atr_pct=values["atr_pct"],
        long_checks={k: bool(crow[f"long_{k}"]) for k in CONDITIONS},
        short_checks={k: bool(crow[f"short_{k}"]) for k in CONDITIONS},
        values=values,
        bar_time=int(row["open_time"]) if "open_time" in row else 0,
    )


def _bb_pct(row) -> float:
    width = row["bb_upper"] - row["bb_lower"]
    if pd.isna(width) or width == 0:
        return 0.5
    return float((row["close"] - row["bb_lower"]) / width)
