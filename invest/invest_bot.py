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
from datetime import date, datetime, timedelta
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
from invest.edgar import Edgar, FeedNotReady, InsiderFeed
from invest.engine import Engine
from invest.execution import LiveExecution, SimExecution
from invest.markets import is_trading_day, next_trading_day
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
        self.data_loader = data_loader or self._load_market_data
        self.state: InvestState | None = None
        self.engine: Engine | None = None
        self.execution = None
        self.running = True
        self.winding_down = False
        self._next_daily_try = 0.0
        self._failures = 0

    # ================================================================ setup
    def now_et(self) -> datetime:
        return datetime.fromtimestamp(self.clock(), tz=ET)

    def setup(self) -> None:
        s = self.s
        tomb = self.store.tombstone()
        state = self.store.load()
        if tomb:
            # A dead bot never trades again, but a real account must still be
            # brought flat: finish the liquidation started at death.
            if s.broker == "sim" or state is None or not (
                    state.positions or state.pending or state.open_orders):
                raise BotIsDead(tomb)
            self.winding_down = True
        if state is None:
            state = InvestState.fresh(s.mode, s.initial_capital)
            self.store.log_decision("start", f"new life: {s.initial_capital:.2f} EUR, "
                                    f"{s.mode} mode (v{__version__})")
        elif state.mode != s.mode:
            raise ConfigError(f"{self.store.dir} holds a {state.mode!r} state but the bot "
                              f"runs as {s.mode!r}: use another INV_STATE_DIR.")
        elif (state.strategy or "momentum") != s.strategy and (state.positions or
                                                               state.stats.get("trades")):
            raise ConfigError(f"{self.store.dir} belongs to the {state.strategy or 'momentum'} "
                              f"strategy, not {s.strategy}: use another INV_STATE_DIR.")
        else:
            self.store.log_decision("start", f"resumed, {len(state.positions)} position(s)")
        state.status = "dead" if self.winding_down else "running"
        state.strategy = s.strategy
        state.universe = ([] if s.strategy == "insider"
                          else build_universe(s.universe or None, s.exclude))
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
        if s.strategy == "insider":
            from invest.insider import InsiderEngine

            edgar = Edgar(s.sec_user_agent, cache_dir=Path(s.data_dir) / "sec")
            self.engine = InsiderEngine(s, state, self.store, self.execution,
                                        InsiderFeed(edgar), edgar, research=None,
                                        ledger=self.ledger)
        else:
            self.engine = Engine(s, state, self.store, self.execution,
                                 research=self.research, ledger=self.ledger)
        self.store.save(state)

    # ============================================================ daily job
    def _load_market_data(self, s, symbols, end):
        st = self.state
        return load_market_data(
            s, symbols, end, fx_fallback=st.fx_rate or None,
            last_day=date.fromisoformat(st.last_day) if st.last_day else None)

    def eod_delay(self) -> int:
        """Minutes after the close before the daily job: the insider strategy waits for
        the SEC's daily index of the session (published in the evening, New York time)."""
        if self.s.strategy == "insider":
            return max(self.s.eod_delay_minutes, 360)
        return self.s.eod_delay_minutes

    def insider_daily_job(self) -> bool:
        """The latest closed session, once its Form 4 filings are published."""
        s, st = self.s, self.state
        now = self.now_et()
        today = now.date()
        bench = self.data_loader(s, [s.benchmark], today).bars.get(s.benchmark)
        if bench is None or bench.empty:
            raise RuntimeError(f"no {s.benchmark} bars from the data provider")
        last = pd.Timestamp(st.last_day) if st.last_day else None
        days = [d for d in bench.index if last is None or d > last]
        ready = datetime.combine(today, SESSION_CLOSE, tzinfo=ET) + timedelta(
            minutes=self.eod_delay())
        if now < ready:
            days = [d for d in days if d.date() < today]
        if not days:
            return False
        day = days[-1]  # filings of skipped days are still read (days_to_scan)
        try:
            symbols = self.engine.prepare(day.date(), now)
        except FeedNotReady as exc:
            if st.last_scan.get("waiting") != str(day.date()):
                self.store.log_decision("info", f"{exc}: retrying every 15 minutes")
                st.last_scan["waiting"] = str(day.date())
            return False
        data = self.data_loader(s, list(dict.fromkeys([s.benchmark] + symbols)), today)
        self.engine.process_day(day, data)
        nxt = next_trading_day(pd.Timestamp(day).date())
        st.next_scan_at = int(datetime.combine(nxt, SESSION_CLOSE, tzinfo=ET).timestamp()
                              * 1000) + self.eod_delay() * 60_000
        return True

    def daily_job(self) -> bool:
        """Process every new closed day (simulator) or the latest one (real broker)."""
        s, st = self.s, self.state
        if s.strategy == "insider":
            return self.insider_daily_job()
        now = self.now_et()
        today = now.date()
        symbols = data_symbols(st.universe, s.benchmark,
                               s.etp_underlying if s.etp_enabled else None)
        data = self.data_loader(s, symbols, today)
        bench = data.bars.get(s.benchmark)
        if bench is None or bench.empty:
            raise RuntimeError(f"no {s.benchmark} bars from the data provider")
        last = pd.Timestamp(st.last_day) if st.last_day else None
        days = [d for d in bench.index if last is None or d > last]
        ready = datetime.combine(today, SESSION_CLOSE, tzinfo=ET) \
            + timedelta(minutes=s.eod_delay_minutes)
        if now < ready:
            days = [d for d in days if d.date() < today]  # today's bar is still forming
        if not days:
            return False
        if last is None or s.broker != "sim":
            days = days[-1:]  # never replay history against a real account
        try:
            for i, day in enumerate(days):
                # Today's news says nothing about past days: no research on catch-up.
                self.engine.research = self.research if i == len(days) - 1 else None
                self.engine.process_day(day, data)
                if st.status == "dead":
                    break
        finally:
            self.engine.research = self.research
        nxt = next_trading_day(pd.Timestamp(days[-1]).date())
        st.next_scan_at = int(datetime.combine(nxt, SESSION_CLOSE, tzinfo=ET).timestamp()
                              * 1000) + s.eod_delay_minutes * 60_000
        return True

    def in_session(self, now: datetime) -> bool:
        return is_trading_day(now.date()) and SESSION_OPEN <= now.time() < SESSION_CLOSE

    def daily_due(self, now: datetime) -> bool:
        if self.clock() < self._next_daily_try:
            return False
        if self.state.next_scan_at and self.clock() * 1000 < self.state.next_scan_at:
            return False  # the last close is processed; wait for the next one
        if not is_trading_day(now.date()):
            return True  # weekend or NYSE holiday: catch up any missed close
        after_close = (now + timedelta(minutes=-self.eod_delay())).time() >= SESSION_CLOSE
        return after_close or now.time() < SESSION_OPEN

    # ============================================================= intraday
    def intraday(self, now: datetime) -> None:
        if not isinstance(self.execution, LiveExecution):
            return
        ex, st = self.execution, self.state
        if not self.in_session(now):
            # Outside the session: confirm cancellations and re-place moved stops
            # (brokers accept resting orders while the market is closed).
            if ex.busy():
                ex.sync(self.engine)
            return
        if st.pending:
            ex.refresh_account()
            ex.submit_planned(self.engine)
        ex.sync(self.engine)
        if ex.refresh_account() is None:
            return
        self.engine.mark(ex.broker_prices())
        ex.check_monitored_stops(self.engine, st.marks)
        equity = self.engine.equity(ex.broker_equity())
        ts = int(self.clock() * 1000)
        st.record_equity(ts, equity)
        if equity > self.s.survival_floor:
            st.floor_breaches = 0
            return
        st.floor_breaches += 1  # a single bad price print must not kill the bot
        if st.floor_breaches >= 2:
            self.engine.die(pd.Timestamp(now.date()), equity, ts)
        else:
            self.store.log_decision("warning", f"equity {equity:.2f} EUR at the survival "
                                    "floor: confirming on the next check")

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

    def wind_down(self, max_days: float = 5.0) -> int:
        """After death: send and reconcile the liquidation until the account is flat."""
        ex = self.execution
        if not isinstance(ex, LiveExecution):
            return EXIT_DEAD
        deadline = self.clock() + max_days * 86400
        log.critical("capital lost: liquidating, no new trade will ever be made")
        while True:  # at least one step, even with max_days=0
            try:
                today = pd.Timestamp(self.now_et().date())
                ex.sync(self.engine)  # learn first which sales were rejected or expired
                ex.cancel_entries(self.engine)
                for pos in list(self.state.positions.values()):
                    if not ex.has_open_sell(pos.symbol):
                        # The previous sale was rejected, cancelled or expired.
                        ex.liquidate(self.engine, pos, today)
                ex.submit_planned(self.engine)
                if ex.refresh_account() is not None:
                    ex.reconcile_positions(self.engine, today)  # sold by hand meanwhile
            except Exception as exc:  # keep trying: the account must end flat
                log.exception("wind-down step failed: %s", exc)
            self.state.heartbeat = int(self.clock() * 1000)
            self.store.save(self.state)
            if not (self.state.positions or self.state.pending or self.state.open_orders):
                self.store.log_decision("stop", "liquidation complete: the bot is flat and "
                                        "stops forever")
                break
            if not self.running or self.clock() >= deadline:
                break
            end = time.time() + self.s.poll_seconds
            while self.running and time.time() < end:
                time.sleep(min(1.0, end - time.time()))
        self.store.save(self.state)
        return EXIT_DEAD

    def run_forever(self) -> int:
        self.setup()
        if self.winding_down:
            return self.wind_down()
        log.info("invest bot started: %s, %s strategy, equity %.2f EUR", self.s.mode,
                 self.s.strategy, self.state.equity)
        while self.running:
            try:
                alive = self.cycle()
                self._failures = 0
            except Exception as exc:  # data/network hiccups must not kill the process
                log.exception("cycle failed: %s", exc)
                self.store.log_decision("error", f"cycle failed: {exc}")
                self._failures += 1
                # 15 min, 30 min, 1 h ... up to 4 h between retries of the daily job.
                self._next_daily_try = self.clock() + min(900 * 2 ** (self._failures - 1),
                                                          4 * 3600)
                alive = True
            if not alive:
                return self.wind_down()
            end = time.time() + self.s.poll_seconds
            while self.running and time.time() < end:
                time.sleep(min(1.0, end - time.time()))
        self.state.status = "stopped"
        self.store.log_decision("stop", "stopped by operator; positions and stops are kept")
        self.store.save(self.state)
        return 0

    def run_once(self) -> int:
        self.setup()
        if self.winding_down:
            return self.wind_down(max_days=0)  # one liquidation step per run
        if isinstance(self.execution, LiveExecution):
            # Orders planned after the close are sent at the next open and stops are
            # watched during the session: a one-shot run would leave them unsent.
            raise ConfigError("--once only works with the simulator: run the bot "
                              "continuously with a real broker (see README_INVEST.md)")
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
    p.add_argument("--strategy", choices=["momentum", "insider"],
                   help="insider = buy after SEC insider purchases (README_INVEST.md s.7)")
    p.add_argument("--state-dir")
    p.add_argument("--once", action="store_true", help="process the last close and exit")
    p.add_argument("--env-file", default=".env")
    p.add_argument("--i-understand-the-risks", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    overrides = {"broker": args.broker, "environment": args.environment,
                 "state_dir": args.state_dir, "strategy": args.strategy}
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
    if s.is_live and s.strategy == "insider":
        log.warning("REAL MONEY on %s: capital %.2f EUR, %d equal slots, insider strategy.",
                    s.broker, s.initial_capital, s.insider_slots)
        log.warning("The insider rule was validated on 2 years of data only (README_INVEST.md, "
                    "section 7): most single trades end near zero. Paper trading first.")
    elif s.is_live:
        log.warning("REAL MONEY on %s: capital %.2f EUR, %.0f%% risk per trade, %d slots.",
                    s.broker, s.initial_capital, s.risk_per_trade * 100, s.slots)
        log.warning("No edge has been demonstrated: on 2019-2026 data this strategy returned "
                    "about -10 %% to -28 %% while buying SPY returned +190 %% (README_INVEST.md, "
                    "section 6). Paper trading first.")
    try:
        return bot.run_once() if args.once else bot.run_forever()
    except BotIsDead as dead:
        t = dead.tombstone
        log.critical("This bot lost its capital on %s (%.2f EUR after %s trades). It will "
                     "never trade again. Archive or delete %s yourself to start a new life.",
                     t.get("died_at_iso"), t.get("equity", 0), t.get("trades"), s.state_dir)
        return 0  # dead and flat: nothing left to do (a background service stays stopped)
    except (ConfigError, BrokerError, RuntimeError) as exc:
        log.error("cannot start: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
