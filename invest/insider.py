"""Insider strategy: buy after an officer's open-market purchase, sell 20 sessions later.

It trades the rule validated by ``python -m invest.events --prices massive`` on every US
stock, delisted ones included (README_INVEST.md, section 7).  Over 2024-10 to 2026-10, a
CEO or CFO buying at least 50 k$ of shares was followed by +2.3 % / +2.9 % over 20
sessions against the same stocks' ordinary days, net of costs, in both test periods.

For each US session D, processed once the SEC has published D's daily index (late
evening, New York time):

1. read every Form 4 filed on D, and on any business day missed since the last run;
2. keep the open-market purchases that match INV_INSIDER_RULES, the best rule first
   (ceo_cfo, then conviction, then buy), one per stock;
3. check each one: last close >= 2 $, mean dollar volume of the last 20 sessions >= 1 M$,
   insider price close to the market price (a reused ticker or a typo otherwise), no
   8-K item 1.01 (contract or financing) and no earnings drop of 5 % in the last 20
   sessions (both validated vetoes), tradable at the broker, not already held, and a
   free slot;
4. buy at the next open with a limit at the last close + 8 %, equal weight across
   INV_INSIDER_SLOTS, with a disaster stop at -35 % (the study had no stop);
5. sell at the open after 20 sessions.

The survival rule, the weekly loss pause and the fee halt of the momentum bot still apply.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from datetime import time as dtime

import numpy as np
import pandas as pd

from invest.brokers.base import OrderRequest
from invest.edgar import EdgarError, FeedNotReady
from invest.engine import Engine
from invest.events import (
    BUY_MIN,
    CEO_CFO_MIN,
    CONVICTION_DELTA,
    CONVICTION_MIN,
    JUMP,
    first_open_after,
    reaction_session,
)

log = logging.getLogger("invest")

RULE_ORDER = ("ceo_cfo", "conviction", "buy")
RULE_LABEL = {"ceo_cfo": "CEO/CFO buy", "conviction": "large buy (+20 % holding)",
              "buy": "officer/director buy"}
CHECKS = ("rule", "price", "liquidity", "price_check", "no_veto", "tradable", "slot")
CHECK_LABELS = ("insider rule", "price >= $2", "volume >= $1M/day", "insider price ~ market",
                "no 8-K 1.01 / earnings drop", "tradable", "free slot")
MAX_CATCH_UP = 10  # business days of filings read at most after a long stop
# the SEC publishes a day's index at about 22:02 New York time; still missing 8 hours later
# (before the open), or missing for an older day: a holiday, no filings
INDEX_GRACE = timedelta(hours=8)


def matching_rules(row, allowed) -> list[str]:
    """Rules (best first) an open-market purchase meets; same thresholds as the study."""
    if bool(row.get("plan")) or not (bool(row.get("officer")) or bool(row.get("director"))):
        return []
    value, delta = float(row.get("value") or 0), float(row.get("delta_own") or 0)
    hits = []
    if row.get("role") in ("ceo", "cfo") and value >= CEO_CFO_MIN:
        hits.append("ceo_cfo")
    if value >= CONVICTION_MIN and delta >= CONVICTION_DELTA:
        hits.append("conviction")
    if value >= BUY_MIN:
        hits.append("buy")
    return [r for r in RULE_ORDER if r in hits and r in allowed]


class InsiderEngine(Engine):
    def __init__(self, s, state, store, execution, feed, edgar, research=None, ledger=None,
                 clock=None):
        super().__init__(s, state, store, execution, research=research, ledger=ledger)
        self.feed, self.edgar = feed, edgar
        self.clock = clock or (lambda: datetime.now())
        self.candidates = pd.DataFrame()
        self.scanned_through = ""

    # --------------------------------------------------------------- filings
    def days_to_scan(self, day: date) -> list[date]:
        last = self.state.insider_scanned
        start = date.fromisoformat(last) + timedelta(days=1) if last else day
        days = [start + timedelta(days=i) for i in range((day - start).days + 1)]
        days = [d for d in days if d.weekday() < 5]
        return days[-MAX_CATCH_UP:]

    def prepare(self, day: date, now_et: datetime) -> list[str]:
        """Read the filings up to ``day`` and return the symbols that need prices.
        Raises FeedNotReady while the SEC has not published ``day`` yet."""
        frames = []
        for d in self.days_to_scan(day):
            try:
                frames.append(self.feed.purchases_on(d))
            except FeedNotReady:
                end_of_day = datetime.combine(d, dtime(22, 0))
                if d == day and now_et.replace(tzinfo=None) < end_of_day + INDEX_GRACE:
                    raise
                log.info("no SEC daily index for %s (federal holiday?): no filings", d)
        buys = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        rows = []
        for r in buys.to_dict("records"):
            rules = matching_rules(r, self.s.insider_rules)
            if rules:
                rows.append({**r, "rule": rules[0], "rank": RULE_ORDER.index(rules[0])})
        cands = pd.DataFrame(rows)
        if not cands.empty:  # best purchase per stock: best rule, then the largest
            cands = (cands.sort_values(["rank", "value"], ascending=[True, False])
                     .drop_duplicates("symbol").reset_index(drop=True))
        self.candidates = cands
        self.scanned_through = str(day)
        held = set(self.state.positions) | {o["symbol"] for o in self.state.pending}
        return sorted(held | set(cands.get("symbol", [])))

    # ------------------------------------------------------------- the day
    def process_day(self, day: pd.Timestamp, data, panel: dict | None = None) -> dict:
        s, st = self.s, self.state
        day = pd.Timestamp(day).normalize()
        equity, close_ts, dead = self._start_day(day, data)
        if dead:
            return {"dead": True, "equity": equity}
        bench = data.bars.get(s.benchmark)
        sessions = bench.index[bench.index <= day] if bench is not None else pd.DatetimeIndex([])

        exiting = set()
        for sym, pos in list(st.positions.items()):
            held = int(((sessions > pd.Timestamp(pos.entry_day)) & (sessions <= day)).sum())
            if held >= s.insider_hold_days - 1:  # sold at the next open: 20 sessions held
                exiting.add(sym)
                self._exit(sym, pos, f"time exit ({held + 1} sessions held)", day)

        rows, entered = [], []
        if not st.daily_halt:
            rows, entered = self._enter(day, data, sessions, equity, exiting)
        if self.scanned_through:
            st.insider_scanned = self.scanned_through
        self._insider_snapshot(day, rows, close_ts)
        st.heartbeat = close_ts
        return {"equity": equity, "entries": entered, "exits": sorted(exiting)}

    def _enter(self, day, data, sessions, equity, exiting) -> tuple[list[dict], list[str]]:
        s, st = self.s, self.state
        pending = {o["symbol"] for o in st.pending}
        free = s.insider_slots - len([p for p in st.positions if p not in exiting]) - sum(
            1 for o in st.pending if o["side"] == "BUY" and o["reason"] == "entry")
        cash = self._free_cash()
        rows, entered = [], []
        for c in self.candidates.to_dict("records") if len(self.candidates) else []:
            sym = c["symbol"]
            checks = dict.fromkeys(CHECKS, False)
            checks["rule"] = True
            values = {"value_k": round(float(c["value"]) / 1000, 1)}
            why = self._check(c, day, data, sessions, checks, values)
            if why is None and len(sessions) >= 2 and str(c["filed"]) <= str(
                    sessions[-2].date()):
                why = f"filed {c['filed']}, before the last session (missed run): too late"
            if why is None and (sym in st.positions or sym in pending):
                why = "already held or ordered"
            if why is None:
                checks["slot"] = free > 0
                if not checks["slot"]:
                    why = f"no free slot ({s.insider_slots} positions)"
            close = values.get("close")
            rows.append({"symbol": sym, "signal": "LONG" if why is None else None,
                         "price": close, "atr_pct": 0.0, "long": checks, "short": {},
                         "long_score": sum(checks.values()), "short_score": 0,
                         "score": None, "values": values, "eligible": True})
            label = (f"{RULE_LABEL[c['rule']]}: {c['owner']} ({c['role']}) bought "
                     f"${float(c['value']):,.0f} at {float(c['price']):.2f}, filed {c['filed']}")
            if why is not None:
                self._log("skip", f"{label}. Not bought: {why}", sym)
                continue
            # sized at the last close (the expected fill); the cash must also cover a
            # fill at the limit price
            notional = min(equity / s.insider_slots, cash / (1 + s.insider_entry_gap))
            limit = close * (1 + s.insider_entry_gap)
            qty = notional * self.fx_now / close
            if not s.fractional:
                qty = float(np.floor(qty))
            if notional < s.min_order_eur or qty <= 0:
                self._log("skip", f"{label}. Not bought: {notional:.2f} EUR free, under the "
                          f"{s.min_order_eur:.0f} EUR minimum order", sym)
                continue
            order = OrderRequest(symbol=sym, side="BUY", qty=qty, kind="LIMIT",
                                 limit_price=limit, reason="entry", sleeve="stock",
                                 currency="USD", broker_ticker=values["ticker"],
                                 stop_pct=s.insider_stop_pct, atr=close * 0.05,
                                 risk_eur=notional * s.insider_stop_pct, day=str(day.date()))
            if not self.exec.queue(self, order):
                continue
            self._log("order", f"{label}. BUY {qty:g} at the next open, limit {limit:.4g} "
                      f"({notional:.2f} EUR), disaster stop -{s.insider_stop_pct:.0%}, sold "
                      f"after {s.insider_hold_days} sessions", sym)
            free -= 1
            cash -= notional * (1 + s.insider_entry_gap)
            entered.append(sym)
        return rows, entered

    def _check(self, c, day, data, sessions, checks, values) -> str | None:
        """Fill ``checks`` and ``values``; the reason a candidate is refused, or None."""
        s = self.s
        df = data.bars.get(c["symbol"])
        if df is None or df.empty or day not in df.index:
            return "no price for the last session"
        df = df[df.index <= day]
        close = float(df["close"].iloc[-1])
        values["close"] = close
        checks["price"] = close >= s.insider_min_price
        if not checks["price"]:
            return f"price {close:.2f} $ under {s.insider_min_price:.0f} $"
        dv = float((df["close"] * df["volume"]).tail(20).mean())
        values["dv_musd"] = round(dv / 1e6, 2)
        checks["liquidity"] = len(df) >= 10 and dv >= s.insider_min_dollar_volume
        if not checks["liquidity"]:
            return f"{dv / 1e6:.2f} M$ a day, under {s.insider_min_dollar_volume / 1e6:g} M$"
        then = df[df.index <= pd.Timestamp(c.get("traded") or c["filed"])]
        ref = float(then["close"].iloc[-1]) if len(then) else close
        ratio = float(c["price"]) / ref
        checks["price_check"] = 0.8 <= ratio <= 1.25
        if not checks["price_check"]:
            return (f"insider price {float(c['price']):.2f} far from the market "
                    f"({ref:.2f}): reused ticker or typo")
        veto = self._veto_8k(int(c["cik"]), day, data, sessions)
        checks["no_veto"] = veto is None
        if veto:
            return veto
        ticker = self.exec.resolve(c["symbol"])
        checks["tradable"] = ticker is not None
        if ticker is None:
            return "not tradable at this broker"
        values["ticker"] = ticker
        return None

    def _veto_8k(self, cik: int, day, data, sessions) -> str | None:
        """Validated vetoes: an 8-K item 1.01, or results followed by a 5 % drop against
        the market, in the last INV_INSIDER_VETO_DAYS sessions."""
        s = self.s
        n = s.insider_veto_days
        if len(sessions) < 2:
            return None
        since = sessions[-min(n + 1, len(sessions))].date() - timedelta(days=7)
        try:
            filings = self.edgar.eight_ks(cik, since, until=day.date(), max_age_days=0)
        except (EdgarError, OSError, ValueError) as exc:
            return f"8-K check failed ({exc}): skipped to be safe"
        recent = sessions[-n:]
        bench = data.bars.get(s.benchmark)
        for f in filings.to_dict("records"):
            when = f.get("accepted_et")
            if when is None or pd.isna(when):
                continue
            items = [x.strip() for x in str(f.get("items", "")).split(",")]
            if "1.01" in items and "2.02" not in items:
                t0 = first_open_after(sessions, when)
                if t0 > len(sessions) - n:  # its 20-session window reaches our entry
                    return f"8-K 1.01 (contract or financing) on {pd.Timestamp(when).date()}"
            if "2.02" in items and bench is not None:
                r = reaction_session(sessions, when)
                if r < len(sessions) and sessions[r] in recent and r > 0:
                    move = self._abnormal_move(f, sessions[r - 1], sessions[r], data, cik)
                    if move is not None and move <= -JUMP:
                        return (f"results on {sessions[r].date()} followed by {move:+.1%} "
                                "against the market")
        return None

    def _abnormal_move(self, f, prev, d, data, cik) -> float | None:
        sym = next((c["symbol"] for c in self.candidates.to_dict("records")
                    if int(c["cik"]) == cik), None)
        df, bench = data.bars.get(sym), data.bars.get(self.s.benchmark)
        try:
            r = df.loc[d, "close"] / df.loc[prev, "close"] - 1
            m = bench.loc[d, "close"] / bench.loc[prev, "close"] - 1
        except (KeyError, AttributeError, TypeError):
            return None
        return float(r - m)

    # ------------------------------------------------------------ dashboard
    def _insider_snapshot(self, day, rows, ts) -> None:
        st = self.state
        rows = sorted(rows, key=lambda r: (r["signal"] is not None, r["long_score"]),
                      reverse=True)
        regime = {"on": True, "off": False, "reasons": ["insider strategy: no market filter"]}
        st.regime = regime
        st.last_scan = {
            "time": ts, "interval": "1d", "day": str(day.date()),
            "evaluated": len(rows), "errors": 0, "eligible": len(rows),
            "signals": [r["symbol"] for r in rows if r["signal"]],
            "conditions": list(CHECKS), "condition_labels": list(CHECK_LABELS),
            "long_only": True, "regime": regime, "evaluations": rows[:60],
        }
        st.sentiment = {
            "enabled": True, "gauge": None, "trending": [r["symbol"] for r in rows[:12]],
            "trending_label": "Insider purchases filed with the SEC (last session)",
            "source_list": [{"name": "SEC Form 4 (daily index)", "ok": True, "note": ""},
                            {"name": "SEC 8-K vetoes", "ok": True, "note": ""}],
            "regime": regime}
        st.settings = self.s.public_dict()
        self.store.save(st)
