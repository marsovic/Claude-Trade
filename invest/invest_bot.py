"""Survival investment bot - main loop (US stocks + leveraged ETP, no crypto).

Starts with 100 EUR. If equity falls to the survival floor (10 EUR by
default) it liquidates, writes a tombstone and never trades again.

    python -m invest.invest_bot                       # simulator on real data
    python -m invest.invest_bot --once                # process the last close, exit
    python -m invest.invest_bot --broker trading212   # Trading 212 DEMO account
    python -m invest.invest_bot --broker ibkr         # IBKR paper (IB Gateway 4002)
    python -m invest.invest_bot --broker trading212 --environment live \\
        --i-understand-the-risks                      # real money (read the README)

Daily rhythm (New York time): after the 16:00 close the bot downloads the
day's bars, decides, and plans orders; real brokers receive them at the next
09:30 open. During the session it reconciles fills, watches stops and marks
the portfolio every ``INV_POLL_SECONDS``.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from datetime import time as dtime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.state import StateStore
from invest import __version__
from invest.brokers.base import BrokerError
from invest.config import LIVE_CONFIRMATION, ConfigError, InvestSettings
from invest.data import ET, load_market_data
from invest.engine import Engine
from invest.execution import LiveExecution, SimExecution
from invest.portfolio import InvestState
from invest.sentiment import Research
from invest.tax import TaxLedger
from invest.universe import build_universe, data_symbols

log = logging.getLogger("invest")
EXIT_DEAD = 2
SESSION_OPEN, SESSION_CLOSE = dtime(9, 30), dtime(16, 0)


class BotIsDead(RuntimeError):
    def __init__(self, tombstone: dict):
        super().__init__("capital lost")
        self.tombstone = tombstone


def make_broker(s: InvestSettings, fx):
    if s.broker == "trading212":
        from invest.brokers.trading212 import Trading212Broker

        return Trading212Broker(s.t212_api_key, s.t212_api_secret, s.environment,
                                etp_ticker=s.etp_broker_ticker, fx=fx)
    if s.broker == "alpaca":
        from invest.brokers.alpaca import AlpacaBroker

        return AlpacaBroker(s.alpaca_api_key, s.alpaca_api_secret, s.environment, fx=fx)
    if s.broker == "ibkr":
        from invest.brokers.ibkr import IbkrBroker

        port = s.ibkr_port or (4001 if s.environment == "live" else 4002)
        return IbkrBroker(s.ibkr_host, port, s.ibkr_client_id, etp_contract=s.etp_broker_ticker,
                          fx=fx)
    raise ConfigError(f"no live broker for {s.broker!r}")


class InvestBot:
    def __init__(self, s: InvestSettings, broker=None, research=None, data_loader=None,
                 clock=time.time):
        self.s = s
        self.clock = clock
        self.store = StateStore(s.state_dir, InvestState)
        self.ledger = TaxLedger(s.state_dir)
        self._broker = broker
        self.research = research
        self.data_loader = data_loader or load_market_data
        self.state: InvestState | None = None
        self.engine: Engine | None = None
        self.execution = None
        self.running = True
        self._next_daily_try = 0.0

    # ================================================================ setup
    def now_et(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), tz=ET)

    def setup(self) -> None:
        tomb = self.store.tombstone()
        if tomb:
            raise BotIsDead(tomb)
        s = self.s
        state = self.store.load()
        if state is None:
            state = InvestState.fresh(s.mode, s.initial_capital)
            self.store.log_decision("start", f"new life: {s.initial_capital:.2f} EUR, "
                                    f"{s.mode} mode (v{__version__})")
        elif state.mode != s.mode:
            raise ConfigError(f"{self.store.dir} holds a {state.mode!r} state but the bot "
                              f"runs as {s.mode!r}: use another INV_STATE_DIR.")
        else:
            self.store.log_decision("start", f"resumed, {len(state.positions)} position(s)")
        state.status = "running"
        state.universe = build_universe(s.universe or None, s.exclude)
        state.settings = s.public_dict()
        self.state = state
        if s.broker == "sim":
            self.execution = SimExecution(s, state, s.fees)
        else:
            broker = self._broker or make_broker(s, fx=lambda: self.engine.fx_now
                                                 if self.engine else s.eurusd_fallback)
            self.execution = LiveExecution(s, state, self.store, broker, clock=self.clock)
            for warning in broker.preflight():
                log.warning(warning)
                self.store.log_decision("warning", warning)
            self.execution.refresh_account()
        if self.research is None and s.sentiment_enabled:
            self.research = Research(s, clock=self.clock)
        self.engine = Engine(s, state, self.store, self.execution, research=self.research,
                             ledger=self.ledger)
        self.store.save(state)

    # ============================================================ daily job
    def daily_job(self) -> bool:
        """Process every new closed day (simulator) or the latest one (real broker)."""
        s, st = self.s, self.state
        today = self.now_et().date()
        symbols = data_symbols(st.universe, s.benchmark,
                               s.etp_underlying if s.etp_enabled else None)
        data = self.data_loader(s, symbols, today)
        bench = data.bars.get(s.benchmark)
        if bench is None or bench.empty:
            raise RuntimeError(f"no {s.benchmark} bars from the data provider")
        last = pd.Timestamp(st.last_day) if st.last_day else None
        days = [d for d in bench.index if last is None or d > last]
        if not days:
            return False
        if last is None or s.broker != "sim":
            days = days[-1:]  # never replay history against a real account
        for day in days:
            self.engine.process_day(day, data)
            if st.status == "dead":
                break
        nxt = (pd.Timestamp(days[-1]) + pd.offsets.BDay(1)).date()
        st.next_scan_at = int(datetime.combine(nxt, SESSION_CLOSE, tzinfo=ET).timestamp()
                              * 1000) + s.eod_delay_minutes * 60_000
        return True

    def in_session(self, now: datetime) -> bool:
        return now.weekday() < 5 and SESSION_OPEN <= now.time() < SESSION_CLOSE

    def daily_due(self, now: datetime) -> bool:
        if self.clock() < self._next_daily_try:
            return False
        if self.state.next_scan_at and self.clock() * 1000 < self.state.next_scan_at:
            return False  # the last close is processed; wait for the next one
        after_close = (now + timedelta(minutes=-self.s.eod_delay_minutes)).time() >= SESSION_CLOSE
        before_open = now.time() < SESSION_OPEN
        return after_close or before_open or now.weekday() >= 5

    # ============================================================= intraday
    def intraday(self, now: datetime) -> None:
        if not isinstance(self.execution, LiveExecution) or not self.in_session(now):
            return
        ex, st = self.execution, self.state
        if st.pending:
            ex.refresh_account()
            ex.submit_planned(self.engine)
        ex.sync(self.engine)
        if ex.refresh_account() is None:
            return
        self.engine.mark(ex.broker_prices())
        ex.check_monitored_stops(self.engine, st.marks)
        equity = self.engine.equity(ex.broker_equity())
        st.record_equity(int(self.clock() * 1000), equity)
        if equity <= self.s.survival_floor:
            self.engine.die(pd.Timestamp(now.date()), equity, int(self.clock() * 1000))

    # ================================================================= loop
    def cycle(self) -> bool:
        now = self.now_et()
        try:
            self.intraday(now)
        except BrokerError as exc:
            log.error("broker error: %s", exc)
            self.store.log_decision("error", f"broker: {exc}")
        if self.state.status != "dead" and self.daily_due(now):
            processed = self.daily_job()
            # Retry in 15 minutes when the provider has no new bar yet.
            self._next_daily_try = self.clock() + (3600 if processed else 900)
        self.state.heartbeat = int(self.clock() * 1000)
        self.store.save(self.state)
        return self.state.status != "dead"

    def run_forever(self) -> int:
        self.setup()
        log.info("invest bot started: %s, equity %.2f EUR, %d stocks", self.s.mode,
                 self.state.equity, len(self.state.universe))
        while self.running:
            try:
                alive = self.cycle()
            except Exception as exc:  # data/network hiccups must not kill the process
                log.exception("cycle failed: %s", exc)
                self.store.log_decision("error", f"cycle failed: {exc}")
                self._next_daily_try = self.clock() + 900
                alive = True
            if not alive:
                return EXIT_DEAD
            end = time.time() + self.s.poll_seconds
            while self.running and time.time() < end:
                time.sleep(min(1.0, end - time.time()))
        self.state.status = "stopped"
        self.store.log_decision("stop", "stopped by operator; positions and stops are kept")
        self.store.save(self.state)
        return 0

    def run_once(self) -> int:
        self.setup()
        if isinstance(self.execution, LiveExecution):
            self.execution.sync(self.engine)
        self.daily_job()
        alive = self.state.status != "dead"
        if alive:
            self.state.status = "stopped"
        self.store.save(self.state)
        return 0 if alive else EXIT_DEAD


# ======================================================================= CLI
def setup_logging(state_dir: str, verbose: bool) -> None:
    Path(state_dir).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    handler = RotatingFileHandler(Path(state_dir) / "bot.log", maxBytes=5_000_000,
                                  backupCount=3)
    handler.setFormatter(fmt)
    root.handlers[:] = [console, handler]
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Survival investment bot (stocks, no crypto)")
    p.add_argument("--broker", choices=["sim", "trading212", "ibkr", "alpaca"])
    p.add_argument("--environment", choices=["paper", "live"])
    p.add_argument("--state-dir")
    p.add_argument("--once", action="store_true", help="process the last close and exit")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--i-understand-the-risks", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    overrides = {"broker": args.broker, "environment": args.environment,
                 "state_dir": args.state_dir}
    if args.i_understand_the_risks:
        overrides["live_trading"] = LIVE_CONFIRMATION
    try:
        s = InvestSettings.from_env(args.env_file, **overrides)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1
    setup_logging(s.state_dir, args.verbose)
    bot = InvestBot(s)

    def stop(signum, _frame):
        log.info("signal %s received, stopping", signum)
        bot.running = False

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    if s.is_live:
        log.warning("REAL MONEY on %s: capital %.2f EUR, %.0f%% risk per trade, %d slots.",
                    s.broker, s.initial_capital, s.risk_per_trade * 100, s.slots)
    try:
        return bot.run_once() if args.once else bot.run_forever()
    except BotIsDead as dead:
        t = dead.tombstone
        log.critical("This bot lost its capital on %s (%.2f EUR after %s trades). It will "
                     "never trade again. Archive or delete %s yourself to start a new life.",
                     t.get("died_at_iso"), t.get("equity", 0), t.get("trades"), s.state_dir)
        return EXIT_DEAD
    except (ConfigError, BrokerError, RuntimeError) as exc:
        log.error("cannot start: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
