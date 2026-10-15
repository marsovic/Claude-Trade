"""Backtester: replays the bot's exact rules on historical klines.

* Signals are computed on a closed bar and executed at the **next bar's open**
  (no look-ahead).
* Stops and targets are checked against each bar's high/low; when both are
  touched in the same bar the stop wins (pessimistic). Gaps fill at the open.
* Fees, slippage, the 2 % risk sizing, volatility leverage, max positions,
  cooldowns, the daily loss limit and the survival rule all apply.
* Only the Fear & Greed index has public history (``--fng``); the other
  sentiment sources cannot be replayed and are treated as neutral.

Examples::

    python -m bot.backtest --top 10 --days 30
    python -m bot.backtest --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 90 --fng
    python -m bot.backtest --csv-dir data/my_klines --interval 1h
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.binance_client import BinanceFuturesClient, SymbolInfo
from bot.config import Settings
from bot.indicators import add_indicators, klines_to_frame
from bot.risk import (
    ClosedTrade,
    Position,
    apply_slippage,
    check_exit,
    direction,
    plan_trade,
    realize,
    update_stops,
)
from bot.strategy import CONDITIONS, LONG, conditions_frame

log = logging.getLogger("backtest")

FNG_HISTORY_URL = "https://api.alternative.me/fng/"


def default_symbol_info(symbol: str) -> SymbolInfo:
    """Used offline: fine-grained steps, Binance's usual 5 USDT min notional."""
    return SymbolInfo(symbol=symbol, base_asset=symbol.removesuffix("USDT"),
                      quote_asset="USDT", tick_size=Decimal("0.00000001"),
                      step_size=Decimal("0.00000001"), min_qty=Decimal("0"),
                      min_notional=Decimal("5"))


# ======================================================================= data
def download_klines(client: BinanceFuturesClient, symbol: str, interval: str,
                    start_ms: int, end_ms: int) -> pd.DataFrame:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        batch = client.klines(symbol, interval, limit=1500, start_time=cursor, end_time=end_ms)
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open + 1 <= cursor or len(batch) < 1500:
            break
        cursor = last_open + 1
        time.sleep(0.2)  # be gentle with the API
    if not rows:
        return pd.DataFrame()
    df = klines_to_frame(rows).drop_duplicates("open_time").reset_index(drop=True)
    return df[df["close_time"] < min(end_ms, int(time.time() * 1000))]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"open_time", "open", "high", "low", "close", "volume"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if "close_time" not in df.columns:
        step = int(df["open_time"].diff().median())
        df["close_time"] = df["open_time"] + step - 1
    return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]


def fear_greed_history(session: requests.Session | None = None) -> pd.Series:
    """Daily Fear & Greed values as scores in [-1, 1], indexed by ms timestamp."""
    session = session or requests.Session()
    resp = session.get(FNG_HISTORY_URL, params={"limit": 0}, timeout=15)
    resp.raise_for_status()
    data = resp.json()["data"]
    series = pd.Series({int(d["timestamp"]) * 1000: (int(d["value"]) - 50) / 50 for d in data})
    return series.sort_index()


# ===================================================================== engine
@dataclass
class BacktestResult:
    trades: list[ClosedTrade]
    equity_curve: list[tuple[int, float]]
    initial_capital: float
    final_equity: float
    died: bool
    died_at: int | None = None
    signals: int = 0
    vetoes: int = 0
    skipped: dict[str, int] = field(default_factory=dict)

    def stats(self) -> dict:
        pnls = [t.pnl for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p <= 0]
        eq = np.array([e for _, e in self.equity_curve]) if self.equity_curve else np.array([])
        max_dd = 0.0
        if eq.size:
            peaks = np.maximum.accumulate(eq)
            max_dd = float(np.max((peaks - eq) / peaks))
        start = self.equity_curve[0][0] if self.equity_curve else 0
        end = self.equity_curve[-1][0] if self.equity_curve else 0
        return {
            "initial_capital": round(self.initial_capital, 2),
            "final_equity": round(self.final_equity, 2),
            "return_pct": round((self.final_equity / self.initial_capital - 1) * 100, 2),
            "trades": len(self.trades),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
            "profit_factor": round(sum(wins) / sum(losses), 2) if losses and sum(losses) > 0
            else (math.inf if wins else 0.0),
            "avg_r": round(sum(t.r_multiple for t in self.trades) / len(self.trades), 2)
            if self.trades else 0.0,
            "best_trade": round(max(pnls), 2) if pnls else 0.0,
            "worst_trade": round(min(pnls), 2) if pnls else 0.0,
            "fees_paid": round(sum(t.fees for t in self.trades), 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "days": round((end - start) / 86_400_000, 1),
            "signals": self.signals,
            "sentiment_vetoes": self.vetoes,
            "skipped": self.skipped,
            "died": self.died,
            "died_at": _iso(self.died_at) if self.died_at else None,
        }


def run_backtest(frames: dict[str, pd.DataFrame], s: Settings,
                 fng: pd.Series | None = None,
                 symbol_infos: dict[str, SymbolInfo] | None = None) -> BacktestResult:
    interval_ms = s.interval_seconds * 1000
    data: dict[str, dict] = {}
    for symbol, df in frames.items():
        if len(df) < s.ema_trend + 5:
            log.warning("%s: only %d bars, skipped", symbol, len(df))
            continue
        df = df.sort_values("open_time").reset_index(drop=True)
        ind = add_indicators(df, s)
        conds = conditions_frame(ind, s)
        data[symbol] = {
            "t": ind["open_time"].to_numpy(),
            "open": ind["open"].to_numpy(), "high": ind["high"].to_numpy(),
            "low": ind["low"].to_numpy(), "close": ind["close"].to_numpy(),
            "atr": ind["atr"].to_numpy(), "atr_pct": ind["atr_pct"].to_numpy(),
            "vol_ratio": ind["vol_ratio"].fillna(0).to_numpy(),
            "signal": conds["signal"].to_numpy(),
            "index": {int(t): i for i, t in enumerate(ind["open_time"])},
        }
    infos = symbol_infos or {}
    timeline = sorted({int(t) for d in data.values() for t in d["t"]})

    balance = s.initial_capital
    positions: dict[str, Position] = {}
    pending: list[dict] = []
    cooldowns: dict[str, int] = {}
    trades: list[ClosedTrade] = []
    curve: list[tuple[int, float]] = []
    last_close: dict[str, float] = {}
    day, day_start_equity, halted = "", s.initial_capital, False
    signals = vetoes = 0
    skipped: dict[str, int] = {}
    died_at = None

    def skip(reason: str):
        skipped[reason] = skipped.get(reason, 0) + 1

    def close(pos: Position, price: float, t: int, reason: str):
        nonlocal balance
        trade, cash = realize(pos, price, t, reason, s)
        balance += cash
        trades.append(trade)
        positions.pop(pos.symbol, None)
        cooldowns[pos.symbol] = t + s.cooldown_bars * interval_ms

    def equity() -> float:
        return balance + sum(p.unrealized(last_close.get(sym, p.entry_price))
                             for sym, p in positions.items())

    for t in timeline:
        # 1) Fill the orders decided on the previous closed bar, at this open.
        still_pending = []
        for order in pending:
            d = data[order["symbol"]]
            i = d["index"].get(t)
            if i is None:
                if order["expires"] > t:
                    still_pending.append(order)
                continue
            price = float(d["open"][i])
            used = sum(p.margin for p in positions.values())
            info = infos.get(order["symbol"]) or default_symbol_info(order["symbol"])
            plan, reason = plan_trade(order["symbol"], order["side"], price, order["atr"],
                                      order["atr_pct"], equity(), used, info, s)
            if plan is None:
                skip(f"sizing: {reason}")
                continue
            fill = apply_slippage(price, plan.side, opening=True, s=s)
            dd = direction(plan.side)
            stop_dist = abs(plan.price - plan.stop_loss)
            fee = fill * plan.qty * s.fee_rate
            balance -= fee
            positions[plan.symbol] = Position(
                symbol=plan.symbol, side=plan.side, qty=plan.qty, entry_price=fill,
                leverage=plan.leverage, stop_loss=fill - dd * stop_dist,
                take_profit=fill + dd * stop_dist * s.take_profit_r,
                initial_stop=fill - dd * stop_dist, atr=plan.atr,
                margin=plan.notional / plan.leverage, risk_amount=plan.risk_amount,
                opened_at=t, entry_fee=fee,
            )
        pending = still_pending

        # 2) Manage open positions on this bar.
        for symbol, pos in list(positions.items()):
            d = data[symbol]
            i = d["index"].get(t)
            if i is None:
                continue
            hit = check_exit(pos, d["high"][i], d["low"][i], d["open"][i])
            if hit:
                close(pos, hit[0], t, hit[1])
                continue
            update_stops(pos, d["high"][i] if pos.side == LONG else d["low"][i], s)
            if (t - pos.opened_at) // interval_ms + 1 >= s.max_hold_bars:
                close(pos, float(d["close"][i]), t + interval_ms - 1, "time_exit")

        for symbol, d in data.items():
            i = d["index"].get(t)
            if i is not None:
                last_close[symbol] = float(d["close"][i])

        # 3) Equity, daily limit, survival.
        eq = equity()
        curve.append((t + interval_ms - 1, eq))
        today = _iso(t)[:10]
        if today != day:
            day, day_start_equity, halted = today, eq, False
        if eq < day_start_equity * (1 - s.max_daily_loss):
            halted = True
        if eq <= s.survival_floor:
            for pos in list(positions.values()):
                close(pos, last_close.get(pos.symbol, pos.entry_price),
                      t + interval_ms - 1, "death")
            died_at = t + interval_ms
            curve.append((died_at, balance))
            break

        # 4) New signals on this closed bar -> orders for the next open.
        candidates = []
        for symbol, d in data.items():
            i = d["index"].get(t)
            if i is not None and isinstance(d["signal"][i], str):
                candidates.append((d["vol_ratio"][i], symbol, i))
        for _, symbol, i in sorted(candidates, reverse=True):
            d = data[symbol]
            side = d["signal"][i]
            signals += 1
            if symbol in positions or any(o["symbol"] == symbol for o in pending):
                skip("already in position")
                continue
            if cooldowns.get(symbol, 0) > t:
                skip("cooldown")
                continue
            if len(positions) + len(pending) >= s.max_open_positions:
                skip("max positions")
                continue
            if halted:
                skip("daily loss limit")
                continue
            if fng is not None and s.sentiment_enabled:
                past = fng[fng.index <= t]
                if len(past):
                    score = float(past.iloc[-1])
                    if (side == LONG and score < -s.sentiment_veto) or \
                            (side != LONG and score > s.sentiment_veto):
                        vetoes += 1
                        continue
            pending.append({"symbol": symbol, "side": side, "atr": float(d["atr"][i]),
                            "atr_pct": float(d["atr_pct"][i]),
                            "expires": t + 2 * interval_ms})

    final = curve[-1][1] if curve else s.initial_capital
    if died_at is None:
        # Mark-to-market what is still open at the end of the data.
        final = equity()
    return BacktestResult(trades=trades, equity_curve=curve,
                          initial_capital=s.initial_capital, final_equity=final,
                          died=died_at is not None, died_at=died_at, signals=signals,
                          vetoes=vetoes, skipped=skipped)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


# ======================================================================== CLI
def write_report(result: BacktestResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(result.stats(), indent=2, default=str))
    with (out_dir / "trades.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["symbol", "side", "opened", "closed", "entry", "exit", "qty",
                         "leverage", "pnl", "fees", "r", "reason"])
        for t in result.trades:
            writer.writerow([t.symbol, t.side, _iso(t.opened_at), _iso(t.closed_at),
                             f"{t.entry_price:.8g}", f"{t.exit_price:.8g}", f"{t.qty:.8g}",
                             t.leverage, f"{t.pnl:.4f}", f"{t.fees:.4f}",
                             f"{t.r_multiple:.2f}", t.reason])
    with (out_dir / "equity.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["time", "equity"])
        for t, e in result.equity_curve:
            writer.writerow([_iso(t), f"{e:.4f}"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Backtest the survival trading bot")
    p.add_argument("--symbols", help="comma-separated list, e.g. BTCUSDT,ETHUSDT")
    p.add_argument("--top", type=int, default=10, help="top-N liquid pairs (if no --symbols)")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--start", help="YYYY-MM-DD (overrides --days)")
    p.add_argument("--end", help="YYYY-MM-DD (default: now)")
    p.add_argument("--interval", help="kline interval (default: INTERVAL setting)")
    p.add_argument("--csv-dir", help="offline mode: read <SYMBOL>.csv files from here")
    p.add_argument("--fng", action="store_true", help="apply historical Fear & Greed veto")
    p.add_argument("--out", default="backtest_results")
    p.add_argument("--env-file", default=".env")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = {"mode": "paper"}
    if args.interval:
        overrides["interval"] = args.interval
    s = Settings.from_env(args.env_file, **overrides)

    frames: dict[str, pd.DataFrame] = {}
    infos: dict[str, SymbolInfo] = {}
    if args.csv_dir:
        for path in sorted(Path(args.csv_dir).glob("*.csv")):
            frames[path.stem.upper()] = load_csv(path)
    else:
        client = BinanceFuturesClient(s.base_url)
        infos = client.symbol_infos()
        if args.symbols:
            symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
        else:
            tickers = [t for t in client.ticker_24h()
                       if t["symbol"] in infos and t["symbol"] not in s.exclude_symbols
                       and infos[t["symbol"]].quote_asset == s.quote_asset]
            tickers.sort(key=lambda t: float(t["quoteVolume"]), reverse=True)
            symbols = [t["symbol"] for t in tickers[: args.top]]
        end = _parse_day(args.end) if args.end else int(time.time() * 1000)
        start = _parse_day(args.start) if args.start else end - args.days * 86_400_000
        # Extra history so that the long EMA is warmed up on the first day.
        warmup = (s.ema_trend + 50) * s.interval_seconds * 1000
        for symbol in symbols:
            log.info("downloading %s %s ...", symbol, s.interval)
            frames[symbol] = download_klines(client, symbol, s.interval, start - warmup, end)
    if not frames:
        log.error("no data")
        return 1

    fng = None
    if args.fng:
        try:
            fng = fear_greed_history()
        except Exception as exc:
            log.warning("Fear & Greed history unavailable (%s): running without it", exc)

    result = run_backtest(frames, s, fng=fng, symbol_infos=infos)
    out = Path(args.out)
    write_report(result, out)
    stats = result.stats()
    width = max(len(k) for k in stats)
    print("\n=== Backtest", ", ".join(sorted(frames)), "===")
    for key, value in stats.items():
        print(f"  {key:<{width}}  {value}")
    print(f"\nConditions: {', '.join(CONDITIONS)} (all 7 must align)")
    print(f"Reports written to {out}/ (summary.json, trades.csv, equity.csv)")
    print("A backtest is not a promise: past data, simulated fills, no funding fees.")
    return 0


def _parse_day(text: str) -> int:
    dt = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


if __name__ == "__main__":
    sys.exit(main())
