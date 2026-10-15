"""v1 - the original survival bot: BTC/USDT spot only, no leverage.

Kept for reference and as the simplest possible version to read first.
One pair, one position at a time, long only:

* buy when EMA 9 crosses above EMA 21, RSI(14) < 70 and close > EMA 50
* sell on the opposite cross, on a 2 x ATR stop, or on a 3 x ATR target
* risk 2 % of the capital per trade; never more than the capital itself
* starts with 50 USDT; if the capital falls to the floor it stops forever

Paper trading by default. Live spot orders need SPOT_LIVE=I_UNDERSTAND_THE_RISKS
plus BINANCE_API_KEY / BINANCE_API_SECRET (a key WITHOUT withdrawal rights).

    python legacy/trading_bot_v1_spot.py
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd
import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

SYMBOL = "BTCUSDT"
INTERVAL = os.environ.get("V1_INTERVAL", "1h")
BASE_URL = "https://api.binance.com"
INITIAL_CAPITAL = float(os.environ.get("INITIAL_CAPITAL", "50"))
SURVIVAL_FLOOR = float(os.environ.get("SURVIVAL_FLOOR", "5"))
RISK_PER_TRADE = 0.02
FEE_RATE = 0.001  # spot taker
POLL_SECONDS = 60
STATE_FILE = Path(os.environ.get("V1_STATE_FILE", "state_v1.json"))
LIVE = os.environ.get("SPOT_LIVE") == "I_UNDERSTAND_THE_RISKS"
API_KEY = os.environ.get("BINANCE_API_KEY", "")
API_SECRET = os.environ.get("BINANCE_API_SECRET", "")

log = logging.getLogger("v1")


# ------------------------------------------------------------------ indicators
def indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-12))
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def signal(df: pd.DataFrame) -> str | None:
    prev, last = df.iloc[-2], df.iloc[-1]
    crossed_up = prev["ema9"] <= prev["ema21"] and last["ema9"] > last["ema21"]
    crossed_down = prev["ema9"] >= prev["ema21"] and last["ema9"] < last["ema21"]
    if crossed_up and last["rsi"] < 70 and last["close"] > last["ema50"]:
        return "BUY"
    if crossed_down:
        return "SELL"
    return None


# ----------------------------------------------------------------------- API
def klines() -> pd.DataFrame:
    resp = requests.get(f"{BASE_URL}/api/v3/klines",
                        params={"symbol": SYMBOL, "interval": INTERVAL, "limit": 200},
                        timeout=10)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame([r[:7] for r in rows],
                      columns=["open_time", "open", "high", "low", "close", "volume",
                               "close_time"])
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    return df[df["close_time"] < time.time() * 1000]  # closed candles only


def price() -> float:
    resp = requests.get(f"{BASE_URL}/api/v3/ticker/price", params={"symbol": SYMBOL},
                        timeout=10)
    resp.raise_for_status()
    return float(resp.json()["price"])


def signed_post(path: str, params: dict) -> dict:
    params = dict(params, timestamp=int(time.time() * 1000), recvWindow=5000)
    query = urlencode(params)
    sig = hmac.new(API_SECRET.encode(), query.encode(), hashlib.sha256).hexdigest()
    resp = requests.post(f"{BASE_URL}{path}?{query}&signature={sig}",
                         headers={"X-MBX-APIKEY": API_KEY}, timeout=10)
    resp.raise_for_status()
    return resp.json()


def market_buy(quote_amount: float) -> tuple[float, float]:
    """Returns (qty, average price)."""
    if not LIVE:
        p = price() * 1.0005
        return quote_amount * (1 - FEE_RATE) / p, p
    order = signed_post("/api/v3/order", {"symbol": SYMBOL, "side": "BUY", "type": "MARKET",
                                          "quoteOrderQty": f"{quote_amount:.2f}"})
    qty = float(order["executedQty"])
    return qty, float(order["cummulativeQuoteQty"]) / qty


def market_sell(qty: float) -> float:
    """Returns the USDT received."""
    if not LIVE:
        return qty * price() * 0.9995 * (1 - FEE_RATE)
    order = signed_post("/api/v3/order", {"symbol": SYMBOL, "side": "SELL", "type": "MARKET",
                                          "quantity": f"{qty:.5f}"})
    return float(order["cummulativeQuoteQty"]) * (1 - FEE_RATE)


# --------------------------------------------------------------------- state
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"cash": INITIAL_CAPITAL, "qty": 0.0, "entry": 0.0, "cost": 0.0, "stop": 0.0,
            "target": 0.0, "last_bar": None, "dead": False, "trades": []}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


# ---------------------------------------------------------------------- loop
def step(state: dict) -> None:
    df = indicators(klines())
    last = df.iloc[-1]
    # Signals are read once per closed candle; stops/targets every poll.
    new_bar = int(last["close_time"]) != state.get("last_bar")
    state["last_bar"] = int(last["close_time"])
    sig = signal(df) if new_bar else None
    now_price = price()
    if state["qty"] > 0:
        reason = None
        if now_price <= state["stop"]:
            reason = "stop"
        elif now_price >= state["target"]:
            reason = "target"
        elif sig == "SELL":
            reason = "cross down"
        if reason:
            received = market_sell(state["qty"])
            pnl = received - state["cost"]
            state["cash"] += received
            state["trades"].append({"time": int(time.time()), "pnl": round(pnl, 4),
                                    "reason": reason})
            log.info("SELL %.6f BTC (%s) pnl %+.2f, cash %.2f", state["qty"], reason, pnl,
                     state["cash"])
            state["qty"] = 0.0
    elif sig == "BUY":
        stop_dist = 2 * last["atr"]
        risk = state["cash"] * RISK_PER_TRADE
        budget = min(state["cash"] * 0.99, risk / stop_dist * now_price)
        if budget >= 10:  # Binance spot minimum order
            qty, entry = market_buy(budget)
            state["cash"] -= budget
            state.update(qty=qty, entry=entry, cost=budget, stop=entry - stop_dist,
                         target=entry + 3 * last["atr"])
            log.info("BUY %.6f BTC @ %.2f stop %.2f target %.2f", qty, entry, state["stop"],
                     state["target"])
    equity = state["cash"] + state["qty"] * now_price
    log.info("equity %.2f USDT", equity)
    if equity <= SURVIVAL_FLOOR:
        if state["qty"] > 0:
            state["cash"] += market_sell(state["qty"])
            state["qty"] = 0.0
        state["dead"] = True
        log.critical("Capital lost (%.2f USDT). The v1 bot stops forever.", equity)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if LIVE and not (API_KEY and API_SECRET):
        log.error("SPOT_LIVE is set but BINANCE_API_KEY / BINANCE_API_SECRET are missing")
        return 1
    state = load_state()
    if state.get("dead"):
        log.critical("This bot already lost its capital. Delete %s to start over.", STATE_FILE)
        return 2
    log.info("v1 spot bot on %s %s, %s mode, cash %.2f", SYMBOL, INTERVAL,
             "LIVE" if LIVE else "paper", state["cash"])
    while not state["dead"]:
        try:
            step(state)
        except requests.RequestException as exc:
            log.error("network error: %s", exc)
        save_state(state)
        if not state["dead"]:
            time.sleep(POLL_SECONDS)
    return 2


if __name__ == "__main__":
    sys.exit(main())
