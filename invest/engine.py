"""The daily decision engine, shared by the backtester, paper and live trading.

``process_day(day, data)`` is called once per closed US trading day:

1. settle what happened since the last close (simulated fills, or the real
   broker's fills), then mark positions to the close;
2. survival rule, weekly loss pause, fee halt;
3. market regime, momentum ranking and the 7 conditions;
4. manage open positions (chandelier stop, rank/time/earnings exits,
   pyramiding, leveraged ETP exits);
5. new entries with vetoes (earnings, news, retail mania), aggressive sizing;
6. publish a snapshot for the dashboard.

Orders decided on day D execute on day D+1 (open or intraday for real brokers).
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timedelta, timezone
from datetime import time as dtime

import numpy as np
import pandas as pd

from invest.brokers.base import Fill, OrderRequest
from invest.brokers.sim import session_ts
from invest.config import etp_symbol
from invest.markets import next_trading_day, sessions_until
from invest.portfolio import InvestPosition, InvestState, InvestTrade
from invest.risk import (
    breakeven_stop,
    chandelier_stop,
    pyramid_ok,
    risk_budget,
    size_position,
)
from invest.signals import (
    CONDITION_LABELS,
    CONDITIONS,
    build_panel,
    etp_entry_ok,
    etp_exit_reason,
    evaluate_day,
    regime_at,
    symbol_features,
    synthetic_etp,
)
from invest.universe import theme_of

log = logging.getLogger("invest")


class Engine:
    def __init__(self, s, state: InvestState, store, execution, research=None, ledger=None):
        self.s = s
        self.state = state
        self.store = store
        self.exec = execution
        self.research = research
        self.picker = None  # backtest placebo: callable(day, evals) -> candidate evals
        self.ledger = ledger
        self.fx_now = state.fx_rate or s.eurusd_fallback  # persisted across restarts
        self.day: pd.Timestamp | None = None

    # ================================================================ fills
    def _rate(self, currency: str, fx: float) -> float:
        return fx if currency == "USD" else 1.0

    def apply_fill(self, f: Fill) -> None:
        st = self.state
        if f.fill_id:
            if f.fill_id in st.seen_fills:
                self._log("warning", f"fill {f.fill_id} already booked: ignored", f.symbol)
                return
            st.seen_fills = (st.seen_fills + [f.fill_id])[-2000:]
        self._book(f)
        self.store.save(st)  # journals and state stay consistent if the process dies

    def _book(self, f: Fill) -> None:
        st = self.state
        order = OrderRequest.from_dict(f.order) if f.order else None
        if f.side == "BUY":
            currency = order.currency if order else "USD"
            gross = f.qty * f.price / self._rate(currency, f.usd_per_eur)
            cost = gross + f.fees_eur
            pos = st.positions.get(f.symbol)
            if pos is None:
                stop_pct = order.stop_pct if order and order.stop_pct else self.s.stop_max_pct
                stop = f.price * (1 - stop_pct)
                lev = self.s.etp_leverage if order and order.sleeve == "etp" else 1.0
                pos = InvestPosition(
                    symbol=f.symbol, qty=f.qty, entry_price=f.price, cost_eur=cost,
                    initial_stop=stop, stop_loss=stop, atr=order.atr if order else 0.0,
                    risk_amount=gross * stop_pct, opened_at=f.ts, entry_day=f.day,
                    sleeve=order.sleeve if order else "stock", currency=currency,
                    broker_ticker=order.broker_ticker if order else "", leverage=lev)
                st.positions[f.symbol] = pos
                self._log("open", f"BUY {f.qty:g} @ {f.price:.4g} ({cost:.2f} EUR) stop "
                          f"{stop:.4g} (-{stop_pct:.1%}), risk {pos.risk_amount:.2f} EUR",
                          f.symbol, pos.to_dict(), ts=f.ts)
            else:
                total = pos.qty + f.qty
                pos.entry_price = (pos.entry_price * pos.qty + f.price * f.qty) / total
                pos.qty = total
                pos.cost_eur += cost
                if order is not None and order.reason == "add":
                    pos.adds += 1
                    # R is measured against everything put at risk, the add included.
                    pos.risk_amount += gross * (order.stop_pct or 0.0)
                    pos.stop_loss = max(pos.stop_loss,
                                        breakeven_stop(pos, self.s.fees.fx_fee_pct))
                    self._log("add", f"pyramid +{f.qty:g} @ {f.price:.4g}, stop to "
                              f"{pos.stop_loss:.4g} (break-even)", f.symbol, ts=f.ts)
                else:  # another partial fill of the same entry order
                    pos.initial_qty += f.qty
                    pos.risk_amount += gross * (order.stop_pct if order and order.stop_pct
                                                else 0.0)
                    self._log("open", f"partial fill +{f.qty:g} @ {f.price:.4g}", f.symbol,
                              ts=f.ts)
            st.cash -= cost
            self._fees(f)
            if self.ledger:
                self.ledger.record(day=f.day, symbol=f.symbol, side="BUY", qty=f.qty,
                                   price=f.price, currency=currency, usd_per_eur=f.usd_per_eur,
                                   gross_eur=gross, fees_eur=f.fees_eur, fill_id=f.fill_id)
            self.exec.on_stop_change(self, pos)
            return

        pos = st.positions.get(f.symbol)
        if pos is None:
            self._log("warning", f"sell fill without a position ({f.qty:g})", f.symbol)
            return
        qty = min(f.qty, pos.qty)
        frac = qty / pos.qty if pos.qty else 1.0
        gross = qty * f.price / self._rate(pos.currency, f.usd_per_eur)
        proceeds = gross - f.fees_eur
        cost_part = pos.cost_eur * frac
        pnl = proceeds - cost_part
        risk = pos.risk_amount * frac
        trade = InvestTrade(
            symbol=f.symbol, qty=qty, entry_price=pos.entry_price, exit_price=f.price,
            cost_eur=cost_part, proceeds_eur=proceeds, pnl=pnl, fees=f.fees_eur,
            r_multiple=pnl / risk if risk > 0 else 0.0, reason=f.reason,
            opened_at=pos.opened_at, closed_at=f.ts, sleeve=pos.sleeve, leverage=pos.leverage)
        st.cash += proceeds
        st.realized_eur += pnl
        self._fees(f)
        if frac >= 0.9999:
            del st.positions[f.symbol]
            self.exec.on_position_closed(self, pos, f)
        else:
            pos.qty -= qty
            pos.cost_eur -= cost_part
            pos.risk_amount -= risk
            self.exec.on_stop_change(self, pos)
        st.record_trade(trade)
        self.store.log_trade(trade)
        if self.ledger:
            self.ledger.record(day=f.day, symbol=f.symbol, side="SELL", qty=qty, price=f.price,
                               currency=pos.currency, usd_per_eur=f.usd_per_eur,
                               gross_eur=gross, fees_eur=f.fees_eur, cost_basis_eur=cost_part,
                               gain_eur=pnl, fill_id=f.fill_id)
        self._log("close", f"{f.reason}: SELL {qty:g} @ {f.price:.4g}, {pnl:+.2f} EUR "
                  f"({trade.r_multiple:+.2f}R)", f.symbol, trade.to_dict(), ts=f.ts)

    def _fees(self, f: Fill) -> None:
        self.state.fees_eur += f.fees_eur
        if f.fees_eur:
            self.state.fee_log.append([f.day, round(f.fees_eur, 6)])

    def on_order_dead(self, order: OrderRequest, why: str) -> None:
        self._log("skip", f"{order.reason} order {order.side} {order.qty:g} not filled ({why})",
                  order.symbol)

    # =============================================================== equity
    def equity(self, broker_equity: float | None = None) -> float:
        st = self.state
        value = sum(p.value_eur(p.last_price, self.fx_now) for p in st.positions.values())
        eq = st.cash + value
        if broker_equity is not None:
            # Only ever count the bot's own money, never more than the broker holds.
            eq = min(eq, broker_equity)
        return eq

    def mark(self, prices: dict[str, float]) -> None:
        for sym, pos in self.state.positions.items():
            if sym in prices and prices[sym] > 0:
                pos.last_price = prices[sym]
            pos.unrealized_eur = pos.value_eur(pos.last_price, self.fx_now) - pos.cost_eur
        self.state.marks = {s: p.last_price for s, p in self.state.positions.items()}

    # ================================================================== day
    def process_day(self, day: pd.Timestamp, data, panel: dict | None = None) -> dict:
        s, st = self.s, self.state
        day = pd.Timestamp(day).normalize()
        equity, close_ts, dead = self._start_day(day, data)
        if dead:
            return {"dead": True, "equity": equity}

        # 3) signals
        panel = panel if panel is not None else build_panel(data.bars, st.universe, s)
        regime = regime_at(day, data.bars, data.vix, data.vix3m, s)
        was_on = st.regime.get("on") if st.regime else None
        st.regime = regime.to_dict()
        evals = evaluate_day(panel, day, regime, s)
        budget = risk_budget(equity, st.drawdown(), s, previous_level=st.dd_level)
        if budget.level != st.dd_level:
            self._log("risk", f"drawdown {st.drawdown():.0%}: ladder level {st.dd_level} -> "
                      f"{budget.level}", ts=close_ts)
            st.dd_level = budget.level
        research_ok = self.research is not None and s.sentiment_enabled
        earnings = self.research.earnings_dates(day.date()) if research_ok else {}

        # 4) manage positions
        exiting = set()
        for sym, pos in list(st.positions.items()):
            if pos.sleeve == "etp":
                reason = self._manage_etp(day, data, pos, regime)
            else:
                reason = self._manage_stock(day, pos, panel.get(sym), evals.get(sym), regime,
                                            earnings, equity)
            if reason:
                exiting.add(sym)
                self._exit(sym, pos, reason, day)

        # 5) entries
        entered = []
        if regime.on and not st.daily_halt:
            entered = self._enter_stocks(day, evals, panel, budget, equity, earnings, exiting)
        if regime.on != was_on:  # log regime changes, not every day
            self._log("regime", "risk-on: new entries allowed" if regime.on else
                      "risk-off, no new stock entry: " + "; ".join(regime.reasons), ts=close_ts)
        if s.etp_enabled and budget.etp_allowed and not st.daily_halt:
            self._enter_etp(day, data, regime, budget, equity)

        # 6) snapshot
        self._snapshot(day, evals, regime, close_ts)
        st.heartbeat = close_ts
        return {"equity": equity, "regime_on": regime.on, "entries": entered,
                "exits": sorted(exiting)}

    def _start_day(self, day: pd.Timestamp, data) -> tuple[float, int, bool]:
        """Steps 1-2, shared by every strategy: settle, mark, survival rule, pauses.
        Returns (equity, close timestamp, dead)."""
        s, st = self.s, self.state
        day = pd.Timestamp(day).normalize()
        self.day = day
        self.fx_now = data.fx(day)
        st.fx_rate = self.fx_now
        close_ts = session_ts(day, dtime(16, 0))

        if s.etp_enabled and s.broker == "sim":
            self._synthetic_etp(data)
        self._apply_splits(day, data)

        # 1) settle the previous decisions
        self.exec.settle(self, day, data)
        prices, highs = {}, {}
        for sym in st.positions:
            df = data.bars.get(sym)
            if df is not None and day in df.index:
                prices[sym] = float(df.loc[day, "close"])
                highs[sym] = float(df.loc[day, "high"])
        for sym, pos in st.positions.items():
            if sym in highs:
                pos.highest = max(pos.highest, self._usable_high(pos, data, day, highs[sym],
                                                                 prices[sym]))
        self.mark(prices)
        etp = data.bars.get(etp_symbol(s))
        if s.etp_enabled and s.broker == "sim" and etp is not None and day in etp.index:
            st.etp_anchor = [str(day.date()), float(etp.loc[day, "close"])]
        own = self.equity()
        equity = self.equity(self.exec.broker_equity())
        suspicious = equity <= s.survival_floor < own
        if suspicious:
            # Only the broker's reading is under the floor: confirm before dying.
            st.floor_breaches += 1
            if st.floor_breaches < 2:
                self._log("warning", f"broker equity {equity:.2f} EUR vs {own:.2f} EUR in "
                          "the bot's books: entries paused, confirming next check",
                          ts=close_ts)
                equity = own
        elif equity > s.survival_floor:
            st.floor_breaches = 0
        st.record_equity(close_ts, equity)
        st.last_day = str(day.date())

        # 2) survival rule and pauses
        if equity <= s.survival_floor:
            self.die(day, equity, close_ts)
            return equity, close_ts, True
        week = f"{day.isocalendar().year}-{day.isocalendar().week}"
        if week != st.week_id:
            # Reference = last close of the previous week, so Monday gaps count.
            st.week_id = week
            st.week_start_equity = st.prev_close_equity or equity
        st.prev_close_equity = equity
        if equity < st.week_start_equity * (1 - s.weekly_loss_pause) and not self._paused(day):
            until = (day + pd.offsets.BDay(s.pause_days)).date()
            st.pause_until = str(until)
            self._log("halt", f"weekly loss over {s.weekly_loss_pause:.0%}: no new entry "
                      f"until {until}", ts=close_ts)
        cutoff = str((day - timedelta(days=92)).date())
        st.fee_log = [x for x in st.fee_log if x[0] >= cutoff]
        fee_halt = bool(sum(x[1] for x in st.fee_log) > 0.05 * equity)
        if fee_halt:
            self._log("halt", "fees over the last 3 months exceed 5 % of equity: entries "
                      "paused", ts=close_ts)
        st.daily_halt = self._paused(day) or fee_halt or suspicious

        return equity, close_ts, False

    # -------------------------------------------------------------- stocks
    def _manage_stock(self, day, pos, feats, ev, regime, earnings, equity) -> str | None:
        s = self.s
        if feats is None or day not in feats.index:
            return None  # no bar today (halt/holiday): keep the stop working
        row = feats.loc[day]
        atr = float(row["atr"]) if not np.isnan(row["atr"]) else pos.atr
        mult = s.regime_off_stop_mult if regime.off else (s.trail_atr_mult or s.stop_atr_mult)
        new_stop = chandelier_stop(pos, atr, s, mult)
        if new_stop > pos.stop_loss * 1.0005:
            old = pos.stop_loss
            pos.stop_loss = new_stop
            self.exec.on_stop_change(self, pos)
            self._log("stop_moved", f"stop {old:.4g} -> {new_stop:.4g}", pos.symbol)
        close = float(row["close"])
        pct = None if ev is None else ev.pct_rank  # entry rank (pyramiding)
        hold = row.get("pct_rank_hold", np.nan)
        hold = None if hold is None or np.isnan(hold) else float(hold)
        held = int(((feats.index > pd.Timestamp(pos.entry_day)) & (feats.index <= day)).sum())
        if self._rebalance_day(day):
            if hold is None or hold < 1 - s.top_pct_hold:
                return f"rank exit (fell out of the top {s.top_pct_hold:.0%})"
            if row["r6_1"] < 0:
                return "rank exit (6-month momentum turned negative)"
        if held >= s.time_stop_days and pos.r_multiple(close) < s.time_stop_min_r and (
                hold is None or hold < 1 - s.top_pct_entry):
            return f"time stop ({held} sessions, {pos.r_multiple(close):+.2f}R)"
        next_report = earnings.get(pos.symbol)
        if next_report and self._sessions_until(day, next_report) <= s.earnings_blackout_days \
                and pos.r_multiple(close) < s.earnings_exit_below_r:
            return f"earnings on {next_report} with only {pos.r_multiple(close):+.2f}R cushion"
        new_high = close >= float(row["hi_prev"]) if not np.isnan(row["hi_prev"]) else False
        if pyramid_ok(pos, close, new_high, pct, s) and not regime.off:
            self._pyramid(day, pos, close, equity)
        return None

    def _usable_high(self, pos, data, day, high: float, close: float) -> float:
        """The day's high for the chandelier. A buy limit filled AFTER a gap-up (entry below
        the open) must not inherit a high that came before the fill: that would create a
        stop above prices the position never lived through."""
        if pos.sleeve != "stock" or pos.entry_day != str(day.date()):
            return high
        df = data.bars.get(pos.symbol)
        if df is None or day not in df.index:
            return high
        if pos.entry_price < float(df.loc[day, "open"]):
            return min(high, max(pos.entry_price, close))
        return high

    def _rebalance_day(self, day) -> bool:
        """Friday close, or the last session of the week when Friday is a market holiday."""
        if self.s.rebalance_weekday != 4:
            return day.weekday() == self.s.rebalance_weekday
        return next_trading_day(day.date()).isocalendar()[:2] != day.date().isocalendar()[:2]

    def _pyramid(self, day, pos, close, equity) -> None:
        qty = pos.initial_qty * self.s.pyramid_fraction
        rate = self._rate(pos.currency, self.fx_now)
        room = self.s.max_weight * equity - pos.value_eur(close, self.fx_now)
        qty = min(qty, max(room, 0.0) * rate / close)  # never above the per-name cap
        if not self.exec.fractional:
            qty = float(int(qty))
        notional = qty * close / rate
        if qty <= 0 or notional < self.s.min_order_eur:
            return
        if notional > self._free_cash():
            self._log("skip", "pyramid skipped: not enough free cash", pos.symbol)
            return
        stop_pct = max((close - pos.stop_loss) / close, self.s.stop_min_pct)
        order = OrderRequest(symbol=pos.symbol, side="BUY", qty=qty, kind="LIMIT",
                             limit_price=close * (1 + self.s.max_entry_gap), reason="add",
                             sleeve="stock", currency=pos.currency, stop_pct=stop_pct,
                             atr=pos.atr, broker_ticker=pos.broker_ticker,
                             day=str(day.date()))
        if self.exec.queue(self, order):
            self._log("order", f"pyramid BUY {qty:g} (+{self.s.pyramid_fraction:.0%}) at "
                      f"+{pos.r_multiple(close):.1f}R", pos.symbol)

    def _enter_stocks(self, day, evals, panel, budget, equity, earnings, exiting) -> list[str]:
        s, st = self.s, self.state
        held_stocks = [p for p in st.positions.values() if p.sleeve == "stock"
                       and p.symbol not in exiting]
        pending = {o["symbol"] for o in st.pending}
        free = budget.slots - len(held_stocks) - sum(
            1 for o in st.pending if o["side"] == "BUY" and o["reason"] == "entry")
        if free <= 0:
            return []
        candidates = [e for e in evals.values() if e.signal and e.symbol not in st.positions
                      and e.symbol not in pending]
        if self.picker is not None:  # placebo research: random entries instead of signals
            candidates = [e for e in self.picker(day, evals) if e.symbol not in st.positions
                          and e.symbol not in pending]
        themes = Counter(theme_of(p.symbol) for p in held_stocks)
        themes.update(theme_of(o["symbol"]) for o in st.pending
                      if o["side"] == "BUY" and o["reason"] == "entry")
        bonus = {}
        if self.research is not None and s.sentiment_enabled:
            for e in sorted(candidates, key=lambda x: x.score or 0, reverse=True)[:8]:
                buys = self.research.insider_buys(e.symbol, day.date())
                if buys and buys >= 2:
                    bonus[e.symbol] = s.insider_bonus
        candidates.sort(key=lambda e: (e.score or 0) + bonus.get(e.symbol, 0), reverse=True)
        open_risk = self._open_risk()
        cash = self._free_cash()
        entered = []
        for ev in candidates:
            if free <= 0:
                break
            self._log("signal", f"7/7 conditions aligned, score {ev.score:+.2f}"
                      + (" + insider buying" if ev.symbol in bonus else ""), ev.symbol,
                      {"checks": ev.checks, "values": ev.values})
            theme = theme_of(ev.symbol)
            if s.max_per_theme and theme and themes[theme] >= s.max_per_theme:
                self._log("skip", f"already {themes[theme]} position(s) in the {theme} theme: "
                          "the slots must not become one bet", ev.symbol)
                continue
            veto = self._veto(day, ev, earnings)
            if veto:
                self._log("veto", veto, ev.symbol)
                continue
            ticker = self.exec.resolve(ev.symbol)
            if ticker is None:
                self._log("skip", "not tradable at this broker", ev.symbol)
                continue
            limit = ev.price * (1 + s.max_entry_gap)
            sized, why = size_position(limit, ev.atr, equity, cash, open_risk, budget,
                                       self.fx_now, s.fees, s)
            if sized is None:
                self._log("skip", f"sizing refused: {why}", ev.symbol)
                continue
            order = OrderRequest(symbol=ev.symbol, side="BUY", qty=sized.qty, kind="LIMIT",
                                 limit_price=limit, reason="entry", sleeve="stock",
                                 currency="USD", broker_ticker=ticker,
                                 stop_pct=sized.stop_pct, atr=ev.atr, risk_eur=sized.risk_eur,
                                 day=str(day.date()))
            if not self.exec.queue(self, order):
                continue
            self._log("order", f"BUY {sized.qty:g} limit {limit:.4g} ({sized.notional_eur:.2f} "
                      f"EUR, {sized.weight:.0%} of equity), stop -{sized.stop_pct:.1%}, "
                      f"risk {sized.risk_eur:.2f} EUR", ev.symbol)
            free -= 1
            cash -= sized.notional_eur
            open_risk += sized.risk_eur
            themes[theme] += 1
            entered.append(ev.symbol)
        return entered

    def _veto(self, day, ev, earnings) -> str | None:
        s = self.s
        report = earnings.get(ev.symbol)
        # One session wider than the exit window, or a fresh entry is sold next day.
        if report and self._sessions_until(day, report) <= s.earnings_blackout_days + 1:
            return f"earnings on {report}: no entry right before results"
        if self.research is None or not s.sentiment_enabled:
            return None
        score, source = self.research.news_score(ev.symbol, day.date())
        if score is not None and score < s.news_veto:
            return f"negative news tone {score:+.2f} ({source})"
        mania = self.research.mania(ev.symbol, ev.values.get("rsi", 0.0))
        return mania

    # ----------------------------------------------------------------- ETP
    def _etp_frames(self, data):
        s = self.s
        under = data.bars.get(s.etp_underlying)
        etp = data.bars.get(etp_symbol(s))
        if under is None or under.empty:
            return None, None
        if etp is not None and etp.empty:
            etp = None
        key = (id(under), id(etp), len(under), 0 if etp is None else len(etp))
        if getattr(self, "_etp_cache_key", None) != key:
            bench = data.bars.get(s.benchmark)
            bench_close = bench["close"] if bench is not None else None
            self._etp_cache = (symbol_features(under, bench_close, s),
                               None if etp is None else symbol_features(etp, bench_close, s))
            self._etp_cache_key = key
        return self._etp_cache

    def _manage_etp(self, day, data, pos, regime) -> str | None:
        under_f, etp_f = self._etp_frames(data)
        if under_f is None or day not in under_f.index:
            return None
        reason = etp_exit_reason(under_f.loc[day], regime, self.s)
        if reason:
            return f"ETP exit: {reason}"
        if etp_f is None:
            self._log("warning", "ETP price series missing today: exit rules checked on the "
                      "underlying, stop not moved", pos.symbol)
        if etp_f is not None and day in etp_f.index:
            atr = float(etp_f.loc[day, "atr"])
            new_stop = chandelier_stop(pos, atr, self.s)
            if new_stop > pos.stop_loss * 1.0005:
                old = pos.stop_loss
                pos.stop_loss = new_stop
                self.exec.on_stop_change(self, pos)
                self._log("stop_moved", f"stop {old:.4g} -> {new_stop:.4g}", pos.symbol)
        return None

    def _enter_etp(self, day, data, regime, budget, equity) -> None:
        s, st = self.s, self.state
        sym = etp_symbol(s)
        if sym in st.positions or any(o["symbol"] == sym for o in st.pending):
            return
        under_f, etp_f = self._etp_frames(data)
        if under_f is None or etp_f is None or day not in under_f.index \
                or day not in etp_f.index:
            return
        ok, why = etp_entry_ok(under_f.loc[day], regime, s)
        if not ok:
            return
        ticker = self.exec.resolve(sym)
        if ticker is None:
            self._log("skip", "leveraged ETP not configured for this broker", sym)
            return
        row = etp_f.loc[day]
        price, atr = float(row["close"]), float(row["atr"])
        limit = price * (1 + s.max_entry_gap)
        currency = "USD" if s.broker == "sim" else s.etp_currency
        sized, reason = size_position(limit, atr, equity, self._free_cash(), self._open_risk(),
                                      budget, self.fx_now, s.fees, s, max_weight=s.etp_weight,
                                      foreign=currency != "EUR")
        if sized is None:
            self._log("skip", f"ETP sizing refused: {reason}", sym)
            return
        order = OrderRequest(symbol=sym, side="BUY", qty=sized.qty, kind="LIMIT",
                             limit_price=limit, reason="entry", sleeve="etp",
                             currency=currency,
                             broker_ticker=ticker, stop_pct=sized.stop_pct, atr=atr,
                             risk_eur=sized.risk_eur, day=str(day.date()))
        if self.exec.queue(self, order):
            self._log("order", f"x{s.etp_leverage:g} ETP BUY {sized.qty:g} ({why}), "
                      f"{sized.notional_eur:.2f} EUR", sym)

    # --------------------------------------------------------------- exits
    def _exit(self, sym, pos, reason, day) -> None:
        order = OrderRequest(symbol=sym, side="SELL", qty=pos.qty, kind="MARKET",
                             reason=reason, sleeve=pos.sleeve, currency=pos.currency,
                             broker_ticker=pos.broker_ticker, day=str(day.date()))
        if self.exec.queue(self, order):
            self._log("order", f"SELL {pos.qty:g} at next open: {reason}", sym)

    def die(self, day, equity: float, ts: int) -> None:
        st = self.state
        self.exec.cancel_entries(self)  # never add exposure after death
        for pos in list(st.positions.values()):
            self.exec.liquidate(self, pos, day)
        st.status = "dead"
        info = {"died_at": ts,
                "died_at_iso": datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(),
                "equity": round(equity, 4), "initial_capital": st.initial_capital,
                "survival_floor": self.s.survival_floor, "trades": st.stats["trades"],
                "mode": st.mode}
        self.store.write_tombstone(info)
        self._log("death", f"capital lost ({equity:.2f} EUR <= floor "
                  f"{self.s.survival_floor:.2f}). The bot stops forever.", data=info, ts=ts)

    # ------------------------------------------------------------- helpers
    def _synthetic_etp(self, data) -> None:
        """Build the simulated 3x ETP, anchored so that past prices never move.

        Paper trading reloads a moving data window every day; without the anchor
        the series would be re-based and stops/P&L of an open ETP would jump.
        """
        s, st = self.s, self.state
        sym = etp_symbol(s)
        under = data.bars.get(s.etp_underlying)
        etp = data.bars.get(sym)
        if under is None or under.empty or (etp is not None and not etp.empty
                                            and etp.index[-1] >= under.index[-1]):
            return
        etp = synthetic_etp(under, s.etp_leverage, s.etp_annual_cost)
        if st.etp_anchor:
            anchor_day, anchor_price = pd.Timestamp(st.etp_anchor[0]), st.etp_anchor[1]
            if anchor_day in etp.index and etp.loc[anchor_day, "close"] > 0:
                scale = anchor_price / float(etp.loc[anchor_day, "close"])
                etp[["open", "high", "low", "close"]] *= scale
        data.bars[sym] = etp

    def _apply_splits(self, day, data) -> None:
        """Rescale open positions and planned orders on a split's ex-date."""
        st = self.state
        for sym, (ex_date, ratio) in getattr(data, "splits", {}).items():
            key = f"{sym}:{ex_date}"
            if key in st.applied_splits or pd.Timestamp(ex_date) > day:
                continue
            st.applied_splits.append(key)
            pos = st.positions.get(sym)
            if pos is not None and pd.Timestamp(pos.entry_day) < pd.Timestamp(ex_date):
                pos.apply_split(ratio)
                self.exec.on_stop_change(self, pos)
                self._log("split", f"{ratio:g}-for-1 split on {ex_date}: position now "
                          f"{pos.qty:g} shares, stop {pos.stop_loss:.4g}", sym)
            for o in st.pending:
                if o["symbol"] == sym:
                    o["qty"] *= ratio
                    for k in ("limit_price", "stop_price"):
                        if o.get(k):
                            o[k] /= ratio

    def _paused(self, day) -> bool:
        return bool(self.state.pause_until) and str(day.date()) < self.state.pause_until

    @staticmethod
    def _sessions_until(day, report) -> int:
        return sessions_until(pd.Timestamp(day).date(), report)

    def _free_cash(self) -> float:
        reserved = 0.0
        for o in self.state.pending:
            if o["side"] == "BUY":
                rate = self._rate(o.get("currency", "USD"), self.fx_now)
                reserved += o["qty"] * (o.get("limit_price") or 0) / rate
        cash = self.state.cash - reserved
        broker_cash = self.exec.broker_cash()
        if broker_cash is not None:
            cash = min(cash, broker_cash - reserved)
        return max(cash, 0.0)

    def _open_risk(self) -> float:
        risk = 0.0
        for p in self.state.positions.values():
            dist = max(p.last_price - p.stop_loss, 0.0)
            risk += p.qty * dist / self._rate(p.currency, self.fx_now)
        risk += sum(o.get("risk_eur", 0.0) for o in self.state.pending if o["side"] == "BUY")
        return risk

    def _snapshot(self, day, evals, regime, ts) -> None:
        st = self.state
        rows = sorted((e.to_dict() for e in evals.values()),
                      key=lambda d: (d["signal"] is not None, d["long_score"],
                                     d["score"] if d["score"] is not None else -9),
                      reverse=True)
        st.last_scan = {
            "time": ts, "interval": "1d", "day": str(day.date()),
            "evaluated": len(evals), "errors": 0,
            "eligible": sum(1 for e in evals.values() if e.eligible),
            "signals": [e.symbol for e in evals.values() if e.signal],
            "conditions": list(CONDITIONS), "condition_labels": list(CONDITION_LABELS),
            "long_only": True, "regime": regime.to_dict(), "evaluations": rows[:60],
        }
        if self.research is not None:
            st.sentiment = self.research.market_snapshot(regime.to_dict())
        else:
            gauge = None if regime.vix is None else {
                "value": regime.vix, "decimals": 1, "label": "VIX",
                "higher_is_bullish": False, "bull_below": 16, "bear_above": 25}
            st.sentiment = {
                "enabled": True, "gauge": gauge, "trending": [],
                "trending_label": "Reddit attention (research sources off)",
                "source_list": [{"name": "VIX (FRED)", "ok": gauge is not None,
                                 "note": "unavailable"},
                                {"name": "News / Reddit / SEC", "ok": False,
                                 "note": "off in this mode"}],
                "regime": regime.to_dict()}
        st.settings = self.s.public_dict()
        self.store.save(st)

    def _log(self, kind, message, symbol=None, data=None, ts=None) -> None:
        if ts is None and self.day is not None:
            ts = session_ts(self.day, dtime(16, 5))
        self.store.log_decision(kind, message, symbol, "LONG" if symbol else None, data, ts=ts)
