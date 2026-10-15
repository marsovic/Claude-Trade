"""Survival trading bot - main loop.

Binance USDⓈ-M futures, the 40 most liquid perpetual pairs, 7 aligned
indicators, volatility-adapted leverage (max x5), 2 % risk per trade,
sentiment veto. It starts with 50 USDT; if it loses them it stops forever.

Usage::

    python -m bot.trading_bot                 # paper trading (default)
    python -m bot.trading_bot --once          # one cycle then exit
    python -m bot.trading_bot --mode testnet  # Binance futures testnet
    python -m bot.trading_bot --mode live --i-understand-the-risks
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

if __package__ in (None, ""):  # allow `python bot/trading_bot.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot import __version__
from bot.binance_client import BinanceAPIError, BinanceFuturesClient, SymbolInfo
from bot.broker import BinanceBroker, PaperBroker
from bot.config import LIVE_CONFIRMATION, ConfigError, Settings
from bot.indicators import add_indicators, klines_to_frame
from bot.risk import ClosedTrade, plan_trade
from bot.sentiment import SentimentEngine
from bot.state import BotState, StateStore
from bot.strategy import Evaluation, evaluate_last

log = logging.getLogger("bot")

EXIT_DEAD = 2


class TradingBot:
    def __init__(self, settings: Settings, client: BinanceFuturesClient | None = None,
                 sentiment: SentimentEngine | None = None, clock=time.time):
        self.s = settings
        self.clock = clock
        self.client = client or BinanceFuturesClient(
            settings.base_url, settings.binance_api_key, settings.binance_api_secret,
            spot_base_url=settings.spot_base_url)
        self.sentiment = sentiment or SentimentEngine(settings, clock=clock)
        self.store = StateStore(settings.state_dir)
        self.state: BotState | None = None
        self.broker = None
        self.symbols: dict[str, SymbolInfo] = {}
        self.universe_at = 0.0
        self.running = True

    # ================================================================ setup
    def now_ms(self) -> int:
        return int(self.clock() * 1000)

    def setup(self) -> None:
        tomb = self.store.tombstone()
        if tomb:
            raise BotIsDead(tomb)
        state = self.store.load()
        if state is None:
            state = BotState.fresh(self.s.mode, self.s.initial_capital)
            state.started_at = state.income_cursor = self.now_ms()
            self.store.log_decision(
                "start", f"new life: {self.s.initial_capital:.2f} USDT in {self.s.mode} mode",
                ts=state.started_at)
        elif state.mode != self.s.mode:
            raise ConfigError(
                f"{self.store.dir} holds a {state.mode!r} state but MODE={self.s.mode!r}. "
                "Use a different STATE_DIR for each mode.")
        else:
            self.store.log_decision("start", f"resumed (v{__version__}), "
                                    f"{len(state.positions)} open position(s)")
        state.status = "running"
        state.settings = self.s.public_dict()
        self.state = state

        self.refresh_universe(force=True)
        if self.s.is_paper:
            self.broker = PaperBroker(self.s, state, self.store)
        else:
            self.broker = BinanceBroker(self.s, state, self.store, self.client, self.symbols)
            for warning in self.broker.preflight():
                log.warning(warning)
                self.store.log_decision("warning", warning)
        self.store.save(state)

    def refresh_universe(self, force: bool = False) -> None:
        if not force and self.clock() - self.universe_at < self.s.universe_refresh_minutes * 60:
            return
        self.symbols = self.client.symbol_infos()
        if isinstance(self.broker, BinanceBroker):
            self.broker.symbols = self.symbols
        tickers = self.client.ticker_24h()
        rows = [t for t in tickers
                if t["symbol"] in self.symbols
                and self.symbols[t["symbol"]].quote_asset == self.s.quote_asset
                and t["symbol"] not in self.s.exclude_symbols]
        rows.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
        universe = [t["symbol"] for t in rows[: self.s.universe_size]]
        if universe != self.state.universe:
            self.store.log_decision("universe", f"{len(universe)} most liquid pairs selected",
                                    data={"symbols": universe})
        self.state.universe = universe
        self.universe_at = self.clock()

    # ============================================================ main loop
    def run_forever(self) -> int:
        self.setup()
        log.info("bot started in %s mode, equity %.2f USDT, universe %d pairs",
                 self.s.mode, self.state.balance, len(self.state.universe))
        while self.running:
            try:
                alive = self.cycle()
            except Exception as exc:  # network hiccup, API error: log, wait, retry
                log.exception("cycle failed: %s", exc)
                self.store.log_decision("error", f"cycle failed: {exc}")
                alive = True
            if not alive:
                return EXIT_DEAD
            self._sleep(self.s.poll_seconds)
        self.state.status = "stopped"
        self.store.log_decision("stop", "bot stopped by operator; open positions are kept")
        self.store.save(self.state)
        return 0

    def run_once(self) -> int:
        self.setup()
        alive = self.cycle(force_scan=True)
        if alive:
            self.state.status = "stopped"
            self.store.save(self.state)
        return 0 if alive else EXIT_DEAD

    def _sleep(self, seconds: float) -> None:
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(min(1.0, end - time.time()))

    def cycle(self, force_scan: bool = False) -> bool:
        """One iteration: manage positions, maybe scan, enforce survival.

        Returns False once the bot is dead.
        """
        now = self.now_ms()
        self.refresh_universe()
        prices = self.client.ticker_prices()
        self.manage(prices, now)
        self._update_marks(prices)
        if not self.check_survival(prices, now):
            return False
        if force_scan or now >= self.state.next_scan_at:
            self.scan(now)
            prices = self.client.ticker_prices()
            interval_ms = self.s.interval_seconds * 1000
            self.state.next_scan_at = ((now // interval_ms) + 1) * interval_ms \
                + self.s.scan_delay_seconds * 1000
        self._update_marks(prices)
        alive = self.check_survival(prices, now)
        self.state.heartbeat = now
        self.store.save(self.state)
        return alive

    def _update_marks(self, prices: dict[str, float]) -> None:
        self.state.marks = {sym: prices[sym] for sym in self.state.positions if sym in prices}

    # ========================================================== positions
    def manage(self, prices: dict[str, float], now: int) -> None:
        for trade in self.broker.sync(prices, now):
            self._on_closed(trade)
        max_hold_ms = self.s.max_hold_bars * self.s.interval_seconds * 1000
        for symbol, pos in list(self.state.positions.items()):
            if now - pos.opened_at >= max_hold_ms and symbol in prices:
                self._on_closed(self.broker.close(pos, prices[symbol], "time_exit", now))

    def _on_closed(self, trade: ClosedTrade) -> None:
        self.state.positions.pop(trade.symbol, None)
        self.state.record_trade(trade)
        self.state.cooldowns[trade.symbol] = (
            trade.closed_at + self.s.cooldown_bars * self.s.interval_seconds * 1000)
        self.store.log_trade(trade)
        self.store.log_decision(
            "close", f"{trade.reason}: {trade.pnl:+.2f} USDT ({trade.r_multiple:+.2f}R)",
            trade.symbol, trade.side, trade.to_dict(), ts=trade.closed_at)
        log.info("CLOSE %s %s %s pnl=%+.2f", trade.symbol, trade.side, trade.reason, trade.pnl)

    def check_survival(self, prices: dict[str, float], now: int) -> bool:
        st = self.state
        equity = self.broker.equity(prices)
        st.record_equity(now, equity)
        day = datetime.fromtimestamp(now / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
        if st.day != day:
            st.day, st.day_start_equity, st.daily_halt = day, equity, False
        if not st.daily_halt and equity < st.day_start_equity * (1 - self.s.max_daily_loss):
            st.daily_halt = True
            self.store.log_decision(
                "halt", f"daily loss limit reached ({equity:.2f} < "
                f"{st.day_start_equity:.2f} -{self.s.max_daily_loss:.0%}): "
                "no new trade until 00:00 UTC", ts=now)
        if equity > self.s.survival_floor:
            return True

        # ---- The survival rule: the capital is lost, the bot dies for good.
        for symbol, pos in list(st.positions.items()):
            try:
                price = prices.get(symbol, pos.entry_price)
                self._on_closed(self.broker.close(pos, price, "death", now))
            except Exception as exc:  # keep closing the others
                log.error("could not close %s on death: %s", symbol, exc)
        equity = self.broker.equity(prices)
        st.record_equity(now, equity)
        st.status = "dead"
        info = {
            "died_at": now,
            "died_at_iso": datetime.fromtimestamp(now / 1000, tz=timezone.utc).isoformat(),
            "equity": round(equity, 4),
            "initial_capital": st.initial_capital,
            "survival_floor": self.s.survival_floor,
            "trades": st.stats["trades"],
            "mode": st.mode,
        }
        self.store.write_tombstone(info)
        self.store.log_decision("death", f"capital lost ({equity:.2f} USDT <= floor "
                                f"{self.s.survival_floor:.2f}). The bot stops forever.",
                                data=info, ts=now)
        self.store.save(st)
        log.critical("CAPITAL LOST - equity %.2f USDT. The bot stops permanently.", equity)
        return False

    # =============================================================== scan
    def fetch_evaluation(self, symbol: str, now: int) -> Evaluation | None:
        rows = self.client.klines(symbol, self.s.interval, self.s.kline_limit)
        if not rows:
            return None
        df = klines_to_frame(rows)
        df = df[df["close_time"] < now]  # closed candles only: no repainting
        if len(df) < self.s.ema_trend + 2:
            return None
        return evaluate_last(symbol, add_indicators(df, self.s), self.s)

    def scan(self, now: int) -> list[Evaluation]:
        evaluations: list[Evaluation] = []
        errors = 0

        def job(symbol):
            try:
                return self.fetch_evaluation(symbol, now)
            except Exception as exc:  # one bad symbol must not stop the scan
                log.warning("%s: evaluation failed: %s", symbol, exc)
                return None

        with ThreadPoolExecutor(max_workers=self.s.max_workers) as pool:
            for ev in pool.map(job, self.state.universe):
                if ev is None:
                    errors += 1
                else:
                    evaluations.append(ev)

        candidates = sorted((e for e in evaluations if e.signal),
                            key=lambda e: e.strength, reverse=True)
        market = self.sentiment.market_snapshot()
        self.state.sentiment = market
        self.state.last_scan = {
            "time": now,
            "interval": self.s.interval,
            "evaluated": len(evaluations),
            "errors": errors,
            "signals": [e.symbol for e in candidates],
            "evaluations": sorted(
                (e.to_dict() for e in evaluations),
                key=lambda d: max(d["long_score"], d["short_score"]), reverse=True),
        }
        self.store.log_decision(
            "scan", f"{len(evaluations)} pairs analysed, {len(candidates)} with 7/7 aligned",
            data={"signals": [e.symbol for e in candidates], "errors": errors}, ts=now)
        for ev in candidates:
            self.try_enter(ev, now)
        return evaluations

    def try_enter(self, ev: Evaluation, now: int) -> bool:
        st = self.state
        checks = ev.long_checks if ev.signal == "LONG" else ev.short_checks
        self.store.log_decision("signal", f"7/7 conditions aligned {ev.signal}",
                                ev.symbol, ev.signal,
                                {"checks": checks, "values": ev.values}, ts=now)

        def skip(reason: str, kind: str = "skip") -> bool:
            self.store.log_decision(kind, reason, ev.symbol, ev.signal, ts=now)
            return False

        if st.status != "running":
            return skip("bot not running")
        if ev.symbol in st.positions:
            return skip("already in a position on this pair")
        if st.cooldowns.get(ev.symbol, 0) > now:
            return skip("cooldown after a recent trade")
        if len(st.positions) >= self.s.max_open_positions:
            return skip(f"max {self.s.max_open_positions} open positions reached")
        if st.daily_halt:
            return skip("daily loss limit reached")
        info = self.symbols.get(ev.symbol)
        if info is None:
            return skip("symbol metadata unavailable")

        sentiment = self.sentiment.evaluate(info.sentiment_asset)
        allowed, why = self.sentiment.allows(ev.signal, sentiment)
        if not allowed:
            self.store.log_decision("veto", why, ev.symbol, ev.signal,
                                    sentiment.to_dict(), ts=now)
            return False

        plan, reason = plan_trade(ev.symbol, ev.signal, ev.price, ev.atr, ev.atr_pct,
                                  st.equity, st.used_margin(), info, self.s)
        if plan is None:
            return skip(f"sizing refused: {reason}")
        try:
            pos = self.broker.open(plan, ev.price, now)
        except (BinanceAPIError, RuntimeError) as exc:
            self.store.log_decision("error", f"order failed: {exc}", ev.symbol, ev.signal,
                                    ts=now)
            log.error("%s: order failed: %s", ev.symbol, exc)
            return False
        pos.sentiment = sentiment.score
        pos.signal = {"checks": checks, "values": ev.values, "sentiment": sentiment.to_dict()}
        st.positions[ev.symbol] = pos
        self.store.log_decision(
            "open",
            f"{pos.side} x{pos.leverage} qty {pos.qty:g} @ {pos.entry_price:.6g} "
            f"SL {pos.stop_loss:.6g} TP {pos.take_profit:.6g} "
            f"risk {pos.risk_amount:.2f} USDT ({why})",
            ev.symbol, ev.signal, pos.to_dict(), ts=now)
        log.info("OPEN %s %s x%d qty=%g entry=%.6g sl=%.6g tp=%.6g", pos.symbol, pos.side,
                 pos.leverage, pos.qty, pos.entry_price, pos.stop_loss, pos.take_profit)
        return True


class BotIsDead(RuntimeError):
    def __init__(self, tombstone: dict):
        super().__init__("capital lost")
        self.tombstone = tombstone


# ====================================================================== CLI
def setup_logging(state_dir: str, verbose: bool = False) -> None:
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = RotatingFileHandler(Path(state_dir) / "bot.log", maxBytes=5_000_000,
                                       backupCount=3)
    file_handler.setFormatter(fmt)
    root.handlers[:] = [console, file_handler]
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Survival trading bot (Binance futures)")
    p.add_argument("--mode", choices=["paper", "testnet", "live"], help="overrides MODE")
    p.add_argument("--state-dir", help="overrides STATE_DIR")
    p.add_argument("--once", action="store_true", help="run a single cycle and exit")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--i-understand-the-risks", action="store_true",
                   help="required (or LIVE_TRADING env) to trade real money")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = {"mode": args.mode, "state_dir": args.state_dir}
    if args.i_understand_the_risks:
        overrides["live_trading"] = LIVE_CONFIRMATION
    try:
        settings = Settings.from_env(args.env_file, **overrides)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    setup_logging(settings.state_dir, args.verbose)
    bot = TradingBot(settings)

    def stop(signum, _frame):
        log.info("signal %s received, stopping after this cycle", signum)
        bot.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if settings.mode == "live":
        log.warning("LIVE MODE: real money at risk. Capital %.2f USDT, risk %.1f%%/trade, "
                    "max leverage x%d.", settings.initial_capital,
                    settings.risk_per_trade * 100, settings.max_leverage)
    try:
        return bot.run_once() if args.once else bot.run_forever()
    except BotIsDead as dead:
        t = dead.tombstone
        log.critical("This bot lost its capital on %s (equity %.2f USDT after %s trades). "
                     "It will never trade again. To start a brand new life, archive or "
                     "delete the %s directory yourself.", t.get("died_at_iso"),
                     t.get("equity", 0), t.get("trades"), settings.state_dir)
        return EXIT_DEAD
    except ConfigError as exc:
        log.error("%s", exc)
        return 1
    except (BinanceAPIError, RuntimeError) as exc:  # unreachable API, failed preflight
        log.error("cannot start: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
