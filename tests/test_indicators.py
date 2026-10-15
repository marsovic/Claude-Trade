import numpy as np
import pandas as pd
import pytest

from bot.config import Settings
from bot.indicators import add_indicators, atr, bollinger, ema, klines_to_frame, macd, rsi
from tests.conftest import frame_to_rows, make_frame


def test_ema_matches_recursive_definition():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(values, 3)
    alpha = 2 / (3 + 1)
    expected = [1.0]
    for v in values[1:]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])
    assert np.allclose(out, expected)


def test_rsi_extremes_and_bounds():
    up = pd.Series(np.arange(1.0, 60.0))
    assert rsi(up, 14).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(60.0, 1.0, -1))
    assert rsi(down, 14).iloc[-1] == pytest.approx(0.0)
    flat = pd.Series([10.0] * 40)
    assert rsi(flat, 14).iloc[-1] == pytest.approx(50.0)
    noisy = make_frame(seed=3)["close"]
    values = rsi(noisy, 14).dropna()
    assert values.between(0, 100).all()
    assert rsi(noisy, 14).iloc[:13].isna().all()


def test_macd_positive_in_uptrend():
    close = pd.Series(np.linspace(100, 150, 200))
    line, signal, hist = macd(close)
    assert line.iloc[-1] > 0
    assert hist.iloc[-1] == pytest.approx(line.iloc[-1] - signal.iloc[-1])


def test_bollinger_band_order():
    close = make_frame(seed=1)["close"]
    lower, mid, upper = bollinger(close, 20, 2.0)
    valid = mid.notna()
    assert (lower[valid] <= mid[valid]).all() and (mid[valid] <= upper[valid]).all()
    assert mid.iloc[-1] == pytest.approx(close.iloc[-20:].mean())


def test_atr_constant_range():
    n = 50
    df = pd.DataFrame({"open": [100.0] * n, "high": [101.0] * n, "low": [99.0] * n,
                       "close": [100.0] * n})
    assert atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_klines_to_frame_and_add_indicators():
    df = make_frame(n=260)
    frame = klines_to_frame(frame_to_rows(df))
    assert list(frame.columns) == ["open_time", "open", "high", "low", "close", "volume",
                                   "close_time"]
    assert frame["close"].dtype == float
    out = add_indicators(frame, Settings())
    for col in ("ema_fast", "ema_slow", "ema_trend", "rsi", "macd", "macd_signal",
                "macd_hist", "bb_lower", "bb_mid", "bb_upper", "vol_ratio", "atr", "atr_pct"):
        assert col in out
    # The volume average excludes the current bar (no self-reference).
    expected = frame["volume"].iloc[-21:-1].mean()
    assert out["vol_ratio"].iloc[-1] == pytest.approx(frame["volume"].iloc[-1] / expected)
