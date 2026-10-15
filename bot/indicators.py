"""Technical indicators on OHLCV DataFrames (pandas only, no TA-Lib)."""

from __future__ import annotations

import numpy as np
import pandas as pd

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
]


def klines_to_frame(rows: list[list]) -> pd.DataFrame:
    """Binance kline rows -> DataFrame with float OHLCV and int timestamps."""
    df = pd.DataFrame(rows, columns=KLINE_COLUMNS[: len(rows[0])] if rows else KLINE_COLUMNS)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = df["open_time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses at all -> RSI 100; flat market -> 50.
    out = out.where(avg_loss != 0, 100.0)
    out = out.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return out.where(avg_gain.notna())


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    return mid - num_std * std, mid, mid + num_std * std


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range."""
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def add_indicators(df: pd.DataFrame, s) -> pd.DataFrame:
    """Return a copy of ``df`` with every column the strategy reads.

    ``s`` is a :class:`bot.config.Settings` (or anything with the same fields).
    """
    out = df.copy()
    close = out["close"]
    out["ema_fast"] = ema(close, s.ema_fast)
    out["ema_slow"] = ema(close, s.ema_slow)
    out["ema_trend"] = ema(close, s.ema_trend)
    out["rsi"] = rsi(close, s.rsi_period)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(
        close, s.macd_fast, s.macd_slow, s.macd_signal)
    out["bb_lower"], out["bb_mid"], out["bb_upper"] = bollinger(close, s.bb_period, s.bb_std)
    out["vol_sma"] = out["volume"].rolling(s.volume_period).mean().shift(1)
    out["vol_ratio"] = out["volume"] / out["vol_sma"]
    out["atr"] = atr(out, s.atr_period)
    out["atr_pct"] = out["atr"] / close * 100
    return out
