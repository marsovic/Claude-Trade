"""Event study: do announcements and attention spikes predict the next weeks' returns?

The idea behind it: public news about smaller companies (insiders buying their own stock,
results that surprise, contracts, a burst of attention on Reddit or Wikipedia) might find
the "pearls" a price-only bot cannot see.  Every event definition below is fixed in
advance; the study only measures it, so a good-looking result is not tuned in hindsight.

How an event is measured, with no look-ahead:

* the event becomes PUBLIC at a known time (SEC acceptance time, insider filing date,
  end of the UTC day for pageviews, time of a Reddit snapshot);
* the trade BUYS at the open of the first session after that (t0) and SELLS at the close
  of session t0 + h - 1, for h = 1, 5, 20 and 60 sessions.  20 is the decision horizon;
* the return is compared with stocks of similar liquidity: those in the same
  dollar-volume group on the same day, over the same window.  The abnormal return is the
  event return minus the group's mean.  This removes the market's move and most of the
  small-cap effect;
* net = abnormal - round-trip costs: Trading 212's FX fee of 0.15 % each way plus a
  spread that grows as liquidity falls;
* a stock that stops trading inside the window keeps its last close (delisting).

Statistics:

* events cluster in time, so each mean gets a confidence interval and a p-value from a
  block bootstrap that resamples whole weeks;
* Holm correction across all the definitions of a run;
* a placebo buys the same stocks on random days near the real ones.  The event must beat
  that, not just zero, because a sample of stocks that went up anyway (survivors) has a
  positive mean on any day;
* DESIGN / LOCKED: the rule is judged on the early period and must then hold on the
  later one, which it has not seen.

    python -m invest.events --prices csv --csv-dir data_study/bars
    python -m invest.events --prices massive --days 730            # every US stock, 2 years
    python -m invest.events --prices tiingo --days 3000 --families insider
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

from invest.study import _boot_means, effective_n, holm_adjust

log = logging.getLogger(__name__)

HORIZONS = (1, 5, 20, 60)
PRIMARY = 20  # sessions: the decision horizon, fixed before any result was seen
CLIP = (-0.95, 3.0)  # a return outside -95 % / +300 % is clipped (data errors, squeezes)
MIN_PRICE = 2.0  # USD, previous close: no penny stocks
MIN_DV = 1_000_000.0  # USD, mean daily dollar volume of the 20 sessions before t0
FX_COST = 0.003  # Trading 212: 0.15 % FX fee on the buy and on the sell
SPREADS = ((50e6, 0.0005), (10e6, 0.0015), (2e6, 0.004), (0.0, 0.010))  # by dollar volume
JUMP = 0.05  # reaction-day threshold for the jump definitions
MIN_DESIGN = 100
MIN_LOCKED = 50
ALPHA = 0.05
PLACEBO_WINDOW = 126  # sessions on each side of a real event where its placebo is drawn
TOP_K = 10  # an edge must survive without the stocks that contribute most to it


@dataclass(frozen=True)
class EventDef:
    name: str
    family: str
    rule: str
    why: str


DEFINITIONS = (
    EventDef("insider_buy", "insider",
             "an officer or director buys >= $25k of shares on the open market (Form 4 code "
             "P, not under a 10b5-1 plan)",
             "insider purchases have predicted returns, mostly in small firms (Lakonishok & "
             "Lee 2001; Cohen, Malloy & Pomorski 2012)"),
    EventDef("insider_cluster", "insider",
             ">= 3 different officers/directors each buy >= $10k within 30 days",
             "several insiders agreeing is a stronger signal than one"),
    EventDef("insider_ceo_cfo", "insider", "the CEO or the CFO buys >= $50k",
             "the best-informed insiders"),
    EventDef("insider_conviction", "insider",
             "an officer/director buys >= $100k AND raises their holding by >= 20 %",
             "a purchase that is large relative to what the insider already holds"),
    EventDef("earnings_jump_up", "earnings",
             "results (8-K item 2.02) and the stock beats its liquidity group by >= 5 % on "
             "the reaction day; bought the next morning",
             "post-earnings drift: prices under-react to good news (weaker since the 2000s)"),
    EventDef("earnings_jump_down", "earnings",
             "results (8-K item 2.02) and the stock lags its group by >= 5 % on the "
             "reaction day",
             "drift after bad news: a candidate veto, not a buy"),
    EventDef("agreement_8k", "filings",
             "8-K item 1.01 (material definitive agreement: contract, deal, financing)",
             "an 'announcement' in the plain sense, with no price filter"),
    EventDef("news_jump_8k", "filings",
             "8-K item 1.01, 7.01 or 8.01 (not results) and the stock beats its group by "
             ">= 5 % on the reaction day",
             "a news-driven jump outside earnings"),
    EventDef("wiki_spike", "attention",
             "Wikipedia pageviews >= 3x their 60-day median and >= 1000 in a day",
             "attention brings buying pressure that later reverses (Da, Engelberg & Gao "
             "2011)"),
    EventDef("reddit_spike", "attention",
             "Reddit mentions (ApeWisdom) >= 3x the mean of the 7 previous snapshots and "
             ">= 20",
             "retail buzz; only testable once enough daily snapshots are collected"),
)
DEF_BY_NAME = {d.name: d for d in DEFINITIONS}
FAMILIES = ("insider", "earnings", "filings", "attention")

DECISION_RULE = f"""\
Fixed before the data was seen:

* decision metric: the EDGE = mean abnormal return over {PRIMARY} sessions (bought at the
  open after the event was public, compared with stocks of the same liquidity group)
  MINUS what the same stocks earn on their ordinary days (median of the placebo: same
  stocks, random days within +-{PLACEBO_WINDOW} sessions, away from their events).  This
  takes out the drift of stocks that rose anyway, i.e. survivorship;
* DESIGN: at least {MIN_DESIGN} events, Holm-adjusted two-sided p < {ALPHA} on the edge, and the
  abnormal mean beyond its placebo (above the 95th percentile of random-day means for a
  buy, below the 5th for a veto);
* without the {TOP_K} stocks that contribute most to it, a buy must still be positive NET of
  costs and a veto still negative (in both periods): a result carried by a few stars is
  not an effect;
* LOCKED (the later period): at least {MIN_LOCKED} events, same sign, one-sided p < {ALPHA};
* a BUY signal must also be positive NET of costs (edge - FX - spread) in both periods;
* verdicts: VALIDATED BUY, VALIDATED VETO, DESIGN ONLY (did not hold in LOCKED), NOT PROVEN.
"""


class EventStudyError(RuntimeError):
    pass


# ==================================================================== price panel
@dataclass
class Panel:
    days: pd.DatetimeIndex
    symbols: list[str]
    open: np.ndarray  # split-adjusted within the panel: returns are measured on these
    close: np.ndarray
    cff: np.ndarray  # close carried forward (a delisted stock keeps its last close)
    prev: np.ndarray  # previous session's close (carried forward)
    dv: np.ndarray  # mean dollar volume of the 20 sessions BEFORE each session
    last: np.ndarray  # per symbol: index of its last close
    col: dict[str, int] = field(default_factory=dict)
    # prices as they traded that day (no later split folded in): the $2 filter and the
    # insider price check must not know about a split that had not happened yet
    prev_raw: np.ndarray | None = None
    cff_raw: np.ndarray | None = None

    def __post_init__(self):
        if self.prev_raw is None:
            self.prev_raw = self.prev
        if self.cff_raw is None:
            self.cff_raw = self.cff

    @property
    def T(self) -> int:
        return len(self.days)


def make_panel(opens: pd.DataFrame, closes: pd.DataFrame, volumes: pd.DataFrame,
               scale: np.ndarray | None = None) -> Panel:
    """``scale`` (same shape, aligned on the weekday rows) turns as-traded prices into prices
    adjusted for the splits inside the panel; None when the prices are adjusted already."""
    opens = opens.sort_index()
    keep = opens.index.dayofweek < 5
    opens = opens[keep]
    if scale is not None:
        scale = np.asarray(scale, dtype=float)[np.asarray(keep)]
    closes = closes.reindex(index=opens.index, columns=opens.columns)
    volumes = volumes.reindex(index=opens.index, columns=opens.columns)
    opn = opens.to_numpy(dtype=float, copy=True)
    C = closes.to_numpy(dtype=float, copy=True)
    bad = ~(opn > 0) | ~(C > 0)
    opn[bad], C[bad] = np.nan, np.nan
    raw = pd.DataFrame(C, index=opens.index, columns=opens.columns)
    # as-traded close x as-traded volume: a split leaves the dollar volume unchanged
    dv = (raw * volumes.to_numpy(dtype=float)).rolling(20, min_periods=10).mean().shift(1)
    cff_raw = raw.ffill().to_numpy()
    nan_row = np.full((1, C.shape[1]), np.nan)
    prev_raw = np.vstack([nan_row, cff_raw[:-1]])
    if scale is not None:
        opn, C = opn * scale, C * scale
        cff = pd.DataFrame(C).ffill().to_numpy()
        prev = np.vstack([nan_row, cff[:-1]])
    else:
        cff, prev = cff_raw, prev_raw
    finite = np.isfinite(C)
    last = np.where(finite.any(0), len(C) - 1 - np.argmax(finite[::-1], axis=0), -1)
    syms = [str(c) for c in opens.columns]
    return Panel(days=pd.DatetimeIndex(opens.index), symbols=syms, open=opn, close=C, cff=cff,
                 prev=prev, dv=dv.to_numpy(), last=last, col={s: i for i, s in enumerate(syms)},
                 prev_raw=prev_raw, cff_raw=cff_raw)


def split_scale(days: pd.DatetimeIndex, symbols, splits: list[dict]) -> np.ndarray:
    """Factor turning as-traded prices into split-adjusted ones: every day before a split's
    execution (ex) date is multiplied by split_from / split_to (a 2-for-1 halves them)."""
    col = {s: i for i, s in enumerate(symbols)}
    scale = np.ones((len(days), len(col)))
    for sp in splits:
        j = col.get(str(sp.get("ticker", "")))
        try:
            factor = float(sp["split_from"]) / float(sp["split_to"])
            ex = pd.Timestamp(sp["execution_date"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
        if j is None or not factor > 0:
            continue
        scale[days < ex, j] *= factor
    return scale


def panel_from_frames(frames: dict[str, pd.DataFrame]) -> Panel:
    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
    if not frames:
        raise EventStudyError("no price data")
    opens = pd.DataFrame({k: v["open"] for k, v in frames.items()})
    closes = pd.DataFrame({k: v["close"] for k, v in frames.items()})
    vols = pd.DataFrame({k: v["volume"] for k, v in frames.items()})
    return make_panel(opens, closes, vols)


def panel_from_long(df: pd.DataFrame, splits: list[dict] | None = None) -> Panel:
    """``df`` columns: date, symbol, open, close, volume (one row per stock and day).  With
    ``splits`` the prices are as traded and get adjusted here (see ``split_scale``)."""
    if df.empty:
        raise EventStudyError("no price data")
    df = df.drop_duplicates(["date", "symbol"], keep="last")
    wide = {c: df.pivot(index="date", columns="symbol", values=c).sort_index()
            for c in ("open", "close", "volume")}
    scale = (split_scale(pd.DatetimeIndex(wide["open"].index), wide["open"].columns, splits)
             if splits is not None else None)
    return make_panel(wide["open"], wide["close"], wide["volume"], scale)


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, *CLIP)


def cost_of(dv: np.ndarray | float) -> np.ndarray:
    dv = np.asarray(dv, dtype=float)
    spread = np.select([dv >= lim for lim, _ in SPREADS], [s for _, s in SPREADS],
                       default=SPREADS[-1][1])
    return FX_COST + spread


# =================================================================== benchmarks
@dataclass
class Bench:
    """Per session and liquidity group: sum and count of the clipped returns, so a stock is
    always compared with the OTHER stocks of its group (leave-one-out)."""

    K: int
    group: np.ndarray  # (T, N) liquidity group of each tradable stock, -1 otherwise
    fwd_sum: dict[int, np.ndarray]  # h -> (T, K) forward returns, open t -> close t + h - 1
    fwd_cnt: dict[int, np.ndarray]
    react_sum: np.ndarray  # (T, K) close-to-close returns of session t
    react_cnt: np.ndarray
    tradable_count: np.ndarray

    def fwd_excess(self, r: np.ndarray, t: np.ndarray, j: np.ndarray, h: int) -> np.ndarray:
        g = np.maximum(self.group[t, j], 0)
        return _excess(r, self.fwd_sum[h][t, g], self.fwd_cnt[h][t, g])


def _excess(r: np.ndarray, s: np.ndarray, c: np.ndarray) -> np.ndarray:
    """r minus the mean of the other members of its group (NaN when it has none)."""
    r = np.asarray(r, dtype=float)
    k = np.isfinite(r)
    with np.errstate(invalid="ignore", divide="ignore"):
        other = (s - np.where(k, r, 0.0)) / (c - k)
    return np.where((c - k) > 0, r - other, np.nan)


def tradable_mask(p: Panel) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.isfinite(p.open) & (p.prev_raw >= MIN_PRICE) & (p.dv >= MIN_DV)


def forward(p: Panel, t: np.ndarray | int, j: np.ndarray | int, h: int) -> np.ndarray:
    """Clipped return from the open of session t to the close of session t + h - 1."""
    t = np.asarray(t)
    j = np.asarray(j)
    e = t + h - 1
    ok = e < p.T
    out = np.full(np.broadcast(t, j).shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = p.cff[np.minimum(e, p.T - 1), j] / p.open[t, j] - 1
    out[...] = np.where(ok, _clip(r), np.nan)
    return out


def one_day(p: Panel, t: np.ndarray | int, j: np.ndarray | int) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return _clip(p.close[t, j] / p.prev[t, j] - 1)


def build_bench(p: Panel) -> Bench:
    trade = tradable_mask(p)
    counts = trade.sum(1)
    typical = float(np.median(counts[counts > 0])) if (counts > 0).any() else 0.0
    K = 10 if typical >= 200 else 5 if typical >= 40 else 1
    group = np.full(p.open.shape, -1, dtype=np.int16)
    fwd_sum = {h: np.zeros((p.T, K)) for h in HORIZONS}
    fwd_cnt = {h: np.zeros((p.T, K)) for h in HORIZONS}
    react_sum, react_cnt = np.zeros((p.T, K)), np.zeros((p.T, K))
    for t in range(p.T):
        idx = np.flatnonzero(trade[t])
        n = len(idx)
        if n < max(2 * K, 5):
            continue
        order = np.argsort(p.dv[t, idx], kind="stable")
        g = np.empty(n, dtype=np.int64)
        g[order] = (np.arange(n) * K) // n
        group[t, idx] = g
        for h in HORIZONS:
            if t + h - 1 < p.T:
                fwd_sum[h][t], fwd_cnt[h][t] = _group_sums(g, forward(p, t, idx, h), K)
        react_sum[t], react_cnt[t] = _group_sums(g, one_day(p, t, idx), K)
    return Bench(K=K, group=group, fwd_sum=fwd_sum, fwd_cnt=fwd_cnt, react_sum=react_sum,
                 react_cnt=react_cnt, tradable_count=counts)


def _group_sums(g: np.ndarray, r: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(r)
    return (np.bincount(g[ok], weights=r[ok], minlength=K),
            np.bincount(g[ok], minlength=K).astype(float))


def reaction(p: Panel, b: Bench, t: np.ndarray, j: np.ndarray) -> np.ndarray:
    """Close-to-close return of session t minus the rest of its liquidity group's (NaN if the
    stock was not tradable that session)."""
    t, j = np.asarray(t), np.asarray(j)
    g = b.group[t, j]
    gi = np.maximum(g, 0)
    ex = _excess(one_day(p, t, j), b.react_sum[t, gi], b.react_cnt[t, gi])
    return np.where(g >= 0, ex, np.nan)


# ======================================================================= timing
def session_after(days: pd.DatetimeIndex, day) -> int:
    """First session strictly after a calendar day."""
    return int(days.searchsorted(pd.Timestamp(day).normalize(), side="right"))


def first_open_after(days: pd.DatetimeIndex, when_et: datetime) -> int:
    """First session whose 09:30 New York open comes after ``when_et`` (naive ET)."""
    d = pd.Timestamp(when_et.date())
    i = int(days.searchsorted(d, side="left"))
    on_session = i < len(days) and days[i] == d
    if on_session and when_et.time() >= dtime(9, 30):
        return i + 1
    return i


def reaction_session(days: pd.DatetimeIndex, when_et: datetime) -> int:
    """Session whose close first reflects news published at ``when_et`` (naive ET)."""
    d = pd.Timestamp(when_et.date())
    i = int(days.searchsorted(d, side="left"))
    on_session = i < len(days) and days[i] == d
    if on_session and when_et.time() >= dtime(16, 0):
        return i + 1
    return i


# =================================================================== extraction
EVENT_COLUMNS = ["definition", "family", "symbol", "j", "t0", "public", "detail"]


@dataclass
class Funnel:
    """Where events were lost, per definition (shown in the report)."""

    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, name: str, stage: str, n: int) -> None:
        self.counts.setdefault(name, {})
        self.counts[name][stage] = self.counts[name].get(stage, 0) + int(n)


SPLIT_FACTORS = np.array([2, 3, 4, 5, 8, 10, 15, 20, 25, 30, 40, 50, 100], float)


def price_matches(reported: np.ndarray, close: np.ndarray) -> np.ndarray:
    """The insider's reported price is close to the market price of that day, allowing for a
    later split (adjusted prices): catches a reused ticker or a typo in the filing."""
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.asarray(reported, float) / np.asarray(close, float)
        ok = (ratio >= 0.8) & (ratio <= 1.25)
        for k in SPLIT_FACTORS:
            for f in (k, 1.0 / k):
                ok |= np.abs(ratio / f - 1) <= 0.06
    return ok & np.isfinite(ratio)


def insider_events(buys: pd.DataFrame, p: Panel, funnel: Funnel) -> pd.DataFrame:
    if buys.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    b = buys.copy()
    b = b[~b["plan"].astype(bool) & (b["officer"].astype(bool) | b["director"].astype(bool))]
    b["j"] = b["symbol"].map(p.col)
    b["t0"] = [session_after(p.days, d) for d in b["filed"]]
    tday = np.array([max(0, int(p.days.searchsorted(pd.Timestamp(d), side="right")) - 1)
                     for d in b["traded"].fillna(b["filed"])], dtype=int)
    has = b["j"].notna().to_numpy()
    jj = b["j"].fillna(0).astype(int).to_numpy()
    close_then = np.where(has, p.cff_raw[tday, jj], np.nan)
    b["price_ok"] = price_matches(b["price"].to_numpy(float), close_then)
    b["has"] = has
    v = b["value"].astype(float)
    d = b["delta_own"].astype(float)
    picks = {
        "insider_buy": b[v >= 25_000],
        "insider_ceo_cfo": b[b["role"].isin(["ceo", "cfo"]) & (v >= 50_000)],
        "insider_conviction": b[(v >= 100_000) & (d >= 0.20)],
        "insider_cluster": _clusters(b[v >= 10_000]),
    }
    out = []
    for name, df in picks.items():
        funnel.add(name, "raw", len(df))
        funnel.add(name, "no price history for the ticker", int((~df["has"]).sum()))
        df = df[df["has"]]
        funnel.add(name, "reported price far from the market price", int((~df["price_ok"]).sum()))
        df = df[df["price_ok"]]
        detail = [f"{r.owner} ({r.role}) ${r.value:,.0f}" for r in df.itertuples()]
        out.append(pd.DataFrame({"definition": name, "family": "insider",
                                 "symbol": df["symbol"].to_numpy(),
                                 "j": df["j"].astype(int).to_numpy(),
                                 "t0": df["t0"].to_numpy(int), "public": df["filed"].to_numpy(),
                                 "detail": detail}))
    return pd.concat(out, ignore_index=True)


def _clusters(b: pd.DataFrame, need: int = 3, days: int = 30) -> pd.DataFrame:
    """Rows on which a company reaches ``need`` distinct insider buyers within ``days``."""
    rows = []
    for _, g in b.sort_values("filed").groupby("cik"):
        filed = pd.to_datetime(g["filed"]).to_numpy()
        owners = g["owner_cik"].astype(str).to_numpy()
        active = False
        for i in range(len(g)):
            lo = filed[i] - np.timedelta64(days, "D")
            n = len(set(owners[(filed > lo) & (filed <= filed[i])]))
            if n >= need and not active:
                rows.append(g.index[i])
            active = n >= need
    return b.loc[rows]


def eightk_events(filings: pd.DataFrame, resolve, p: Panel, b: Bench, funnel: Funnel,
                  families: tuple[str, ...] = ("earnings", "filings")) -> pd.DataFrame:
    """8-K filings (cik, accession, filed, items, accepted_et) -> events."""
    if filings.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    f = filings.dropna(subset=["accepted_et"]).copy()
    f["symbol"] = [resolve(c, d) for c, d in zip(f["cik"], f["filed"], strict=True)]
    f["j"] = f["symbol"].map(p.col)
    items = f["items"].fillna("").astype(str)
    has = lambda code: items.str.split(",").apply(lambda xs: code in [x.strip() for x in xs])  # noqa: E731
    is_results = has("2.02")
    f["first_open"] = [first_open_after(p.days, t) for t in f["accepted_et"]]
    f["react_t"] = [reaction_session(p.days, t) for t in f["accepted_et"]]
    groups = {
        "earnings_jump_up": f[is_results],
        "earnings_jump_down": f[is_results],
        "agreement_8k": f[has("1.01") & ~is_results],
        "news_jump_8k": f[(has("1.01") | has("7.01") | has("8.01")) & ~is_results],
    }
    out = []
    for name, df in groups.items():
        if DEF_BY_NAME[name].family not in families:
            continue
        funnel.add(name, "raw", len(df))
        funnel.add(name, "no price history for the ticker", int(df["j"].isna().sum()))
        df = df[df["j"].notna()]
        j = df["j"].astype(int).to_numpy()
        if name == "agreement_8k":
            t0 = df["first_open"].to_numpy(int)
            keep = np.ones(len(df), dtype=bool)
            det = [f"items {x}" for x in df["items"]]
        else:
            rt = df["react_t"].to_numpy(int)
            inside = rt < p.T
            ra = np.full(len(df), np.nan)
            ra[inside] = reaction(p, b, rt[inside], j[inside])
            if name == "earnings_jump_down":
                keep = ra <= -JUMP
            else:
                keep = ra >= JUMP
            funnel.add(name, "reaction below the threshold or not measurable",
                       int((~keep).sum()))
            t0 = rt + 1
            det = [f"items {x}, reaction {r:+.1%}" for x, r in zip(df["items"], ra, strict=True)]
        out.append(pd.DataFrame({"definition": name, "family": DEF_BY_NAME[name].family,
                                 "symbol": df["symbol"].to_numpy()[keep], "j": j[keep],
                                 "t0": t0[keep],
                                 "public": [str(x) for x in df["accepted_et"].to_numpy()[keep]],
                                 "detail": np.asarray(det, dtype=object)[keep]}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=EVENT_COLUMNS)


def wiki_events(views: dict[str, pd.Series], p: Panel, funnel: Funnel,
                ratio: float = 3.0, floor: float = 1000.0) -> pd.DataFrame:
    rows = []
    n_raw = n_missing = n_early = 0
    for sym, s in views.items():
        if s.empty:
            continue
        s = s.sort_index().astype(float)
        med = s.shift(1).rolling(60, min_periods=30).median()
        hit = s[(s >= floor) & (s >= ratio * med.clip(lower=1.0))]
        early = hit.index < p.days[0] - pd.Timedelta(days=1)
        n_early += int(early.sum())
        hit = hit[~early]
        n_raw += len(hit)
        if sym not in p.col:
            n_missing += len(hit)
            continue
        for d, v in hit.items():
            rows.append({"definition": "wiki_spike", "family": "attention", "symbol": sym,
                         "j": p.col[sym], "t0": session_after(p.days, d),
                         "public": f"{d.date()} (UTC day)",
                         "detail": f"{v:,.0f} views, x{v / max(med.get(d, 1.0), 1.0):.1f}"})
    funnel.add("wiki_spike", "raw", n_raw)
    funnel.add("wiki_spike", "no price history for the ticker", n_missing)
    if n_early:
        funnel.add("wiki_spike", "before the first price day (not counted in raw)", n_early)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def reddit_events(snaps: pd.DataFrame, p: Panel, funnel: Funnel, ratio: float = 3.0,
                  floor: float = 20.0, lookback: int = 7) -> pd.DataFrame:
    if snaps.empty:
        funnel.add("reddit_spike", "raw", 0)
        return pd.DataFrame(columns=EVENT_COLUMNS)
    wide = snaps.pivot_table(index="taken_utc", columns="ticker", values="mentions",
                             aggfunc="max").sort_index().fillna(0.0)
    base = wide.shift(1).rolling(lookback, min_periods=lookback).mean()
    hits = (wide >= floor) & (wide >= ratio * base.clip(lower=1.0)) & base.notna()
    rows = []
    n_missing = 0
    stacked = hits.stack()
    for taken, sym in stacked[stacked].index:
        if sym not in p.col:
            n_missing += 1
            continue
        et = taken.tz_convert("America/New_York").tz_localize(None).to_pydatetime()
        rows.append({"definition": "reddit_spike", "family": "attention", "symbol": sym,
                     "j": p.col[sym], "t0": first_open_after(p.days, et),
                     "public": str(taken), "detail": f"{wide.loc[taken, sym]:.0f} mentions"})
    funnel.add("reddit_spike", "raw", int(stacked.sum()))
    funnel.add("reddit_spike", "no price history for the ticker", n_missing)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


# ===================================================================== outcomes
def measure(events: pd.DataFrame, p: Panel, b: Bench, funnel: Funnel) -> pd.DataFrame:
    """Keep tradable, non-overlapping events and attach their returns."""
    ev = events.copy()
    ev["t0"] = ev["t0"].astype(int)
    ev["j"] = ev["j"].astype(int)
    late = ev["t0"] >= p.T
    for name, n in ev[late].groupby("definition").size().items():
        funnel.add(name, "after the last price", n)
    ev = ev[~late]
    g = b.group[ev["t0"].to_numpy(), ev["j"].to_numpy()]
    bad = g < 0
    for name, n in ev[bad].groupby("definition").size().items():
        funnel.add(name, f"not tradable at t0 (no bar, price < ${MIN_PRICE:.0f} or volume "
                         f"< ${MIN_DV / 1e6:.0f}M/day)", n)
    ev = ev[~bad].sort_values(["definition", "symbol", "t0"]).reset_index(drop=True)
    keep = np.zeros(len(ev), dtype=bool)
    last: dict[tuple, int] = {}
    for i, (name, sym, t0) in enumerate(zip(ev["definition"], ev["symbol"], ev["t0"],
                                            strict=True)):
        k = (name, sym)
        if t0 >= last.get(k, -10**9) + PRIMARY:
            keep[i] = True
            last[k] = t0
    for name, n in ev[~keep].groupby("definition").size().items():
        funnel.add(name, f"repeat within {PRIMARY} sessions of a counted one", n)
    ev = ev[keep].reset_index(drop=True)
    t0, j = ev["t0"].to_numpy(), ev["j"].to_numpy()
    grp = b.group[t0, j].astype(int)
    ev["day"] = p.days[t0]
    ev["group"] = grp
    ev["dv"] = p.dv[t0, j]
    ev["cost"] = cost_of(ev["dv"].to_numpy())
    for h in HORIZONS:
        r = forward(p, t0, j, h)
        abn = b.fwd_excess(r, t0, j, h)
        ev[f"ret_{h}"] = r
        ev[f"abn_{h}"] = abn
        ev[f"net_{h}"] = abn - ev["cost"].to_numpy()
        ev[f"delisted_{h}"] = p.last[j] < t0 + h - 1
    for name, n in ev.groupby("definition").size().items():
        funnel.add(name, "kept", n)
    return ev


# =================================================================== statistics
def week_ids(days: pd.DatetimeIndex) -> np.ndarray:
    iso = days.isocalendar()
    return (iso["year"].to_numpy(int) * 100 + iso["week"].to_numpy(int))


def boot_stats(values: np.ndarray, weeks: np.ndarray, n_boot: int, seed: int) -> dict:
    """Mean with a week-block bootstrap: CI and p-values (two-sided, mean>0, mean<0)."""
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    v, weeks = v[ok], np.asarray(weeks)[ok]
    n = len(v)
    out = {"n": n, "mean": np.nan, "median": np.nan, "hit": np.nan, "lo": np.nan,
           "hi": np.nan, "p": np.nan, "p_pos": np.nan, "p_neg": np.nan}
    if n == 0:
        return out
    out.update(mean=float(v.mean()), median=float(np.median(v)), hit=float((v > 0).mean()))
    if n < 5:
        return out
    codes, uniq = pd.factorize(weeks)
    S = np.bincount(codes, weights=v, minlength=len(uniq))[:, None]
    C = np.bincount(codes, minlength=len(uniq)).astype(float)[:, None]
    d = _boot_means(S, C, n_boot, seed)[:, 0]
    d = d[np.isfinite(d)]
    B = len(d)
    lo_p = (np.sum(d <= 0) + 1) / (B + 1)
    hi_p = (np.sum(d >= 0) + 1) / (B + 1)
    out.update(lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
               p=float(min(1.0, 2 * min(lo_p, hi_p))), p_pos=float(lo_p), p_neg=float(hi_p))
    return out


def placebo(ev: pd.DataFrame, all_t0: dict[tuple, np.ndarray], p: Panel, b: Bench,
            draws: int, seed: int) -> np.ndarray:
    """Mean abnormal return of ``draws`` random-day replicas of ``ev``: each event is moved to
    a random tradable session of the SAME stock within +-PLACEBO_WINDOW sessions, away from
    that stock's real events."""
    if ev.empty or draws <= 0:
        return np.array([])
    flat_t, flat_j, starts, sizes = [], [], [], []
    pos = 0
    trade = b.group >= 0
    for name, sym, j, t0 in zip(ev["definition"], ev["symbol"], ev["j"], ev["t0"], strict=True):
        lo, hi = max(0, t0 - PLACEBO_WINDOW), min(p.T - PRIMARY, t0 + PLACEBO_WINDOW)
        if hi <= lo:
            continue
        cand = lo + np.flatnonzero(trade[lo:hi, j])
        real = all_t0.get((name, sym), np.array([]))
        if len(real):
            near = np.abs(cand[:, None] - real[None, :]).min(1) <= PRIMARY
            cand = cand[~near]
        if len(cand) == 0:
            continue
        flat_t.append(cand)
        flat_j.append(np.full(len(cand), j))
        starts.append(pos)
        sizes.append(len(cand))
        pos += len(cand)
    if not sizes:
        return np.array([])
    ft, fj = np.concatenate(flat_t), np.concatenate(flat_j)
    vals = b.fwd_excess(forward(p, ft, fj, PRIMARY), ft, fj, PRIMARY)
    starts_a, sizes_a = np.array(starts), np.array(sizes)
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for k in range(draws):
        pick = starts_a + (rng.random(len(sizes_a)) * sizes_a).astype(int)
        out[k] = np.nanmean(vals[pick])
    return out


@dataclass
class EventResult:
    table: pd.DataFrame  # one row per definition
    events: pd.DataFrame
    funnel: Funnel
    meta: dict
    per_year: pd.DataFrame
    horizons: pd.DataFrame
    by_liquidity: pd.DataFrame


def analyse(ev: pd.DataFrame, p: Panel, b: Bench, funnel: Funnel, locked_from: pd.Timestamp,
            n_boot: int = 5000, draws: int = 1000, seed: int = 7, meta: dict | None = None
            ) -> EventResult:
    weeks_of = dict(zip(p.days, week_ids(p.days), strict=True))
    ev = ev.copy()
    ev["segment"] = np.where(ev["day"] < locked_from, "design", "locked") if len(ev) else []
    ev["week"] = [weeks_of[d] for d in ev["day"]] if len(ev) else []
    all_t0 = {k: np.sort(g["t0"].to_numpy()) for k, g in ev.groupby(["definition", "symbol"])}
    rows = []
    for d in DEFINITIONS:
        sub = ev[ev["definition"] == d.name]
        if d.name not in funnel.counts:
            continue
        row = {"definition": d.name, "family": d.family}
        for seg in ("design", "locked"):
            s = sub[(sub["segment"] == seg) & np.isfinite(sub[f"abn_{PRIMARY}"])]
            abn = s[f"abn_{PRIMARY}"].to_numpy(float)
            pl = placebo(s, all_t0, p, b, draws, seed + 1)
            # the same stocks on ordinary days: what a stock earns anyway (survivors drift up)
            usual = float(np.median(pl)) if len(pl) else 0.0
            raw = boot_stats(abn, s["week"].to_numpy(), n_boot, seed)
            st = boot_stats(abn - usual, s["week"].to_numpy(), n_boot, seed)
            row.update({f"{seg}_{k}": v for k, v in st.items()})
            row[f"{seg}_abn"] = raw["mean"]
            row[f"{seg}_usual"] = usual if len(pl) else np.nan
            row[f"{seg}_net"] = (st["mean"] - float(s["cost"].mean())) if len(s) else np.nan
            row[f"{seg}_n_eff"] = (effective_n(abn, s["symbol"].to_numpy())[0]
                                   if len(abn) >= 3 else float(len(abn)))
            row[f"{seg}_placebo_p5"] = float(np.percentile(pl, 5)) if len(pl) else np.nan
            row[f"{seg}_placebo_p95"] = float(np.percentile(pl, 95)) if len(pl) else np.nan
            row[f"{seg}_placebo_rank"] = (float((pl < raw["mean"]).mean()) if len(pl)
                                          and np.isfinite(raw["mean"]) else np.nan)
            row[f"{seg}_edge_wo_top"], row[f"{seg}_net_wo_top"] = _edge_without_top(
                s, abn - usual, st["mean"], all_t0, p, b, draws, seed)
        rows.append(row)
    table = pd.DataFrame(rows)
    if not table.empty:
        pv = table["design_p"].to_numpy(float)
        adj = np.full(len(pv), np.nan)
        ok = np.isfinite(pv)
        if ok.any():
            adj[ok] = holm_adjust(pv[ok])
        table["design_p_holm"] = adj
        table["verdict"] = [verdict(r) for r in table.to_dict("records")]
        if (meta or {}).get("hand_picked"):
            table["verdict"] = [v + " - on a hand-picked list: confirm with --prices massive"
                                if v.startswith("VALIDATED") else v for v in table["verdict"]]
    return EventResult(table=table, events=ev, funnel=funnel, meta=meta or {},
                       per_year=_per_year(ev), horizons=_horizons(ev),
                       by_liquidity=_by_liquidity(ev))


def _edge_without_top(s: pd.DataFrame, edge: np.ndarray, mean: float, all_t0, p: Panel,
                      b: Bench, draws: int, seed: int) -> tuple[float, float]:
    """(edge, net edge) once the TOP_K stocks that push it most in its own direction are
    removed, measured against THEIR usual days again.  A hand-picked list of today's
    winners can owe a 'post-earnings drift' to a few stars that became winners through
    those very rallies."""
    if len(s) < 3 * TOP_K or not np.isfinite(mean):
        return np.nan, np.nan
    contrib = pd.Series(edge, index=s.index).groupby(s["symbol"]).sum()
    top = (contrib.nlargest(TOP_K) if mean > 0 else contrib.nsmallest(TOP_K)).index
    rest = s[~s["symbol"].isin(top)]
    if rest.empty:
        return np.nan, np.nan
    pl = placebo(rest, all_t0, p, b, max(draws // 2, 50), seed + 2)
    usual = float(np.median(pl)) if len(pl) else 0.0
    e = float(rest[f"abn_{PRIMARY}"].mean() - usual)
    return e, e - float(rest["cost"].mean())


def verdict(r: dict) -> str:
    n_d, n_l = r.get("design_n", 0) or 0, r.get("locked_n", 0) or 0
    if n_d < MIN_DESIGN:
        return f"NOT PROVEN (only {n_d} design events, {MIN_DESIGN} needed)"
    m = r["design_mean"]
    if not (r.get("design_p_holm", 1.0) < ALPHA):
        return "NOT PROVEN (design effect not significant after Holm)"
    if m > 0:
        if not r["design_abn"] > r["design_placebo_p95"]:
            return "NOT PROVEN (no better than buying the same stocks on random days)"
        if not r["design_net"] > 0:
            return "NOT PROVEN (positive but eaten by costs)"
        if not r.get("design_net_wo_top", np.nan) > 0:
            return (f"NOT PROVEN (carried by a handful of stocks: without the top {TOP_K} it "
                    "does not pay its costs)")
        if n_l < MIN_LOCKED:
            return f"DESIGN ONLY (locked period has {n_l} events, {MIN_LOCKED} needed)"
        if not (r["locked_mean"] > 0 and r["locked_p_pos"] < ALPHA and r["locked_net"] > 0):
            return "DESIGN ONLY (did not hold in the locked period)"
        if not r.get("locked_net_wo_top", np.nan) > 0:
            return f"DESIGN ONLY (locked result carried by its top {TOP_K} stocks)"
        return "VALIDATED BUY"
    if not r["design_abn"] < r["design_placebo_p5"]:
        return "NOT PROVEN (no worse than the same stocks on random days)"
    if not r.get("design_edge_wo_top", np.nan) < 0:
        return f"NOT PROVEN (carried by a handful of stocks: gone without the top {TOP_K})"
    if n_l < MIN_LOCKED:
        return f"DESIGN ONLY veto (locked period has {n_l} events, {MIN_LOCKED} needed)"
    if not (r["locked_mean"] < 0 and r["locked_p_neg"] < ALPHA):
        return "DESIGN ONLY veto (did not hold in the locked period)"
    if not r.get("locked_edge_wo_top", np.nan) < 0:
        return f"DESIGN ONLY veto (locked result carried by its top {TOP_K} stocks)"
    return "VALIDATED VETO"


def _per_year(ev: pd.DataFrame) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame()
    e = ev.assign(year=pd.DatetimeIndex(ev["day"]).year)
    return (e.groupby(["definition", "year"])[f"abn_{PRIMARY}"]
            .agg(["count", "mean", "median"]).reset_index())


def _horizons(ev: pd.DataFrame) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame()
    cols = {f"abn_{h}": "mean" for h in HORIZONS} | {f"net_{h}": "mean" for h in HORIZONS}
    out = ev.groupby("definition").agg(cols)
    out["delisted_in_window_pct"] = ev.groupby("definition")[f"delisted_{PRIMARY}"].mean() * 100
    return out.reset_index()


def _by_liquidity(ev: pd.DataFrame) -> pd.DataFrame:
    if ev.empty:
        return pd.DataFrame()
    bins = [0, 5e6, 50e6, np.inf]
    labels = ["$1-5M/day", "$5-50M/day", ">$50M/day"]
    e = ev.assign(liquidity=pd.cut(ev["dv"], bins=bins, labels=labels))
    return (e.groupby(["definition", "liquidity"], observed=True)[f"abn_{PRIMARY}"]
            .agg(["count", "mean"]).reset_index())


def default_locked_from(days: pd.DatetimeIndex, frac: float = 0.6) -> pd.Timestamp:
    d = days[min(len(days) - 1, int(len(days) * frac))]
    return (d + pd.offsets.MonthBegin(1)).normalize()


# ======================================================================= report
def _pct(x, nd=2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:+.{nd}f} %"


def _p(x) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.3f}"


def render_report(res: EventResult) -> str:
    m = res.meta
    out = ["# Event study: announcements and attention", "",
           f"Prices: **{m.get('prices', '?')}**, {m.get('first_day')} .. {m.get('last_day')}, "
           f"{m.get('symbols', 0)} stocks ({m.get('tradable_median', 0):.0f} tradable on a "
           f"typical day, {m.get('groups', 0)} liquidity groups).",
           f"DESIGN before **{m.get('locked_from')}**, LOCKED from it.", ""]
    if m.get("survivorship"):
        out += [f"> **Survivorship:** {m['survivorship']}", ""]
    out += ["## Verdicts", "",
            f"Decision horizon: {PRIMARY} sessions, bought at the open after the event was "
            "public. Abnormal = vs stocks of the same liquidity group. Edge = abnormal minus "
            "what the same stocks earn on ordinary days. Net = edge minus FX and spread.", "",
            "| definition | verdict | design n | abnormal | same stocks, usual days | edge | "
            f"net edge | 95 % CI of edge | Holm p | net w/o top {TOP_K} | locked n | "
            f"locked edge | net | p | net w/o top {TOP_K} |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res.table.to_dict("records"):
        lp = r["locked_p_pos"] if (r["locked_mean"] or 0) >= 0 else r["locked_p_neg"]
        out.append(
            f"| {r['definition']} | {r['verdict']} | {r['design_n']} | "
            f"{_pct(r['design_abn'])} | {_pct(r['design_usual'])} | "
            f"{_pct(r['design_mean'])} | {_pct(r['design_net'])} | "
            f"{_pct(r['design_lo'])} .. {_pct(r['design_hi'])} | {_p(r['design_p_holm'])} | "
            f"{_pct(r['design_net_wo_top'])} | "
            f"{r['locked_n']} | {_pct(r['locked_mean'])} | {_pct(r['locked_net'])} | "
            f"{_p(lp)} | {_pct(r['locked_net_wo_top'])} |")
    out += ["", "## Rule", "", DECISION_RULE, "## Definitions", ""]
    for d in DEFINITIONS:
        if d.name in res.funnel.counts:
            out.append(f"* **{d.name}**: {d.rule}. _Why: {d.why}._")
    out += ["", "## Where events were lost", ""]
    for name, stages in res.funnel.counts.items():
        parts = ", ".join(f"{k}: {v}" for k, v in stages.items())
        out.append(f"* {name}: {parts}")
    if not res.horizons.empty:
        out += ["", "## All horizons (all periods, descriptive only)", "",
                "| definition | " + " | ".join(f"abn {h}d" for h in HORIZONS) + " | "
                + " | ".join(f"net {h}d" for h in HORIZONS) + " | delisted in 20d |",
                "|---|" + "---|" * (2 * len(HORIZONS) + 1)]
        for r in res.horizons.to_dict("records"):
            out.append(f"| {r['definition']} | "
                       + " | ".join(_pct(r[f'abn_{h}']) for h in HORIZONS) + " | "
                       + " | ".join(_pct(r[f'net_{h}']) for h in HORIZONS)
                       + f" | {r['delisted_in_window_pct']:.1f} % |")
    if not res.by_liquidity.empty:
        out += ["", "## By liquidity (abnormal, 20 sessions, descriptive)", "",
                "| definition | liquidity | n | mean |", "|---|---|---|---|"]
        for r in res.by_liquidity.to_dict("records"):
            out.append(f"| {r['definition']} | {r['liquidity']} | {r['count']} | "
                       f"{_pct(r['mean'])} |")
    if not res.per_year.empty:
        out += ["", "## Per year (abnormal, 20 sessions)", "",
                "| definition | year | n | mean | median |", "|---|---|---|---|---|"]
        for r in res.per_year.to_dict("records"):
            out.append(f"| {r['definition']} | {r['year']} | {r['count']} | "
                       f"{_pct(r['mean'])} | {_pct(r['median'])} |")
    out += ["", "## How to read this", "",
            "* A VALIDATED verdict is a reason to paper-trade the rule, not to bet on it: "
            "effects like these fade once published and costs on small stocks are uncertain.",
            "* NOT PROVEN means the data cannot tell the effect from luck; it does not prove "
            "there is none.",
            "* The placebo answers 'is the DAY special, or just the stock?'. A sample of "
            "stocks that rose anyway beats zero on any day, but not its own random days.", ""]
    return "\n".join(out)


def write_outputs(res: EventResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(res))
    cols = [c for c in res.events.columns if c not in ("j",)]
    res.events[cols].to_csv(out / "events.csv", index=False)
    res.table.to_csv(out / "definitions.csv", index=False)
    summary = {"meta": res.meta, "funnel": res.funnel.counts,
               "verdicts": dict(zip(res.table.get("definition", []),
                                    res.table.get("verdict", []), strict=True))}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))


# ===================================================================== sources
class GroupedStore:
    """Whole-market daily bars from one grouped call per session (Massive), cached per day.
    Point-in-time: a stock is there on the days it traded, delisted or not.  With
    ``adjusted=False`` the files hold as-traded prices, which never change afterwards, so a
    resume weeks later cannot mix two split adjustments."""

    MAX_REFUSED = 15  # oldest days refused in a row before giving up

    def __init__(self, folder: str | Path, provider, adjusted: bool = True):
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.p = provider
        self.adjusted = adjusted

    def path(self, day: date) -> Path:
        return self.dir / f"{day}.csv.gz"

    def refused_path(self, day: date) -> Path:
        return self.dir / f"{day}.refused"

    def update(self, start: date, end: date) -> int:
        """Download the missing sessions, oldest first.  The provider counts its plan's
        history window (2 years on Massive's free plan) from its own clock, so the oldest
        day or two of a request can fall just outside it.  Such days are skipped when they
        are older than every day served, and remembered so a re-run does not ask again."""
        from invest.data import DataError, DataNotReady
        from invest.markets import is_trading_day

        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        todo = [d for d in days if is_trading_day(d) and not self.path(d).exists()
                and not self.refused_path(d).exists()]
        if todo:
            eta = len(todo) * getattr(getattr(self.p, "limiter", None), "interval", 0.0) / 60
            log.info("grouped bars: %d session(s) to download (~%.0f min at the provider's "
                     "rate limit); each day is saved, a new run resumes", len(todo), eta)
        cached = sorted(f.name[:10] for f in self.dir.glob("*.csv.gz"))
        served = False
        refused: list[date] = []
        for n, d in enumerate(todo, 1):
            try:
                rows = self.p.grouped(d, adjusted=self.adjusted) or {}
            except DataNotReady:
                break
            except DataError as exc:
                text = str(exc).lower()
                outside = "entitlement" in text or "not_authorized" in text
                older = not cached or str(d) < cached[0]
                if not outside or served or not older or len(refused) >= self.MAX_REFUSED:
                    raise
                refused.append(d)
                continue
            if refused and not served:
                log.warning("grouped bars: the provider's plan starts after %s; %d older "
                            "session(s) skipped", refused[-1], len(refused))
                for r in refused:  # a later day was served: these are the plan's edge
                    self.refused_path(r).write_text("outside the plan's history\n")
            served = True
            df = pd.DataFrame.from_dict(rows, orient="index")
            df.index.name = "symbol"
            tmp = self.path(d).with_suffix(".tmp")
            df.to_csv(tmp, compression="gzip")
            tmp.replace(self.path(d))
            if n % 20 == 0:
                log.info("grouped bars: %d / %d sessions", n, len(todo))
        return len(todo)

    def load(self, start: date, end: date, universe: set[str] | None = None) -> pd.DataFrame:
        frames = []
        for f in sorted(self.dir.glob("*.csv.gz")):
            d = f.name[:10]
            if not (str(start) <= d <= str(end)):
                continue
            df = pd.read_csv(f, compression="gzip", keep_default_na=False,
                             na_values=[""], dtype={"symbol": str})
            if df.empty:
                continue
            if universe is not None:
                df = df[df["symbol"].isin(universe)]
            frames.append(df.assign(date=pd.Timestamp(d)))
        if not frames:
            return pd.DataFrame(columns=["date", "symbol", "open", "close", "volume"])
        return pd.concat(frames, ignore_index=True)


def migrate_adjusted(old: GroupedStore, new: GroupedStore, splits: list[dict]) -> int:
    """Turn day files saved with ``adjusted=true`` (adjusted for the splits known on the day
    they were downloaded: the file's date) into as-traded files, so an earlier download is
    not thrown away.  Prices are divided, volumes multiplied, by split_from / split_to of
    every split executed after that session and up to the download day."""
    by_ticker: dict[str, list[tuple[str, float]]] = {}
    for sp in splits:
        try:
            f = float(sp["split_from"]) / float(sp["split_to"])
            by_ticker.setdefault(str(sp["ticker"]), []).append((str(sp["execution_date"]), f))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            continue
    moved = 0
    for path in sorted(old.dir.glob("*.csv.gz")):
        day = path.name[:10]
        target = new.dir / path.name
        if target.exists():
            continue
        fetched = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        df = pd.read_csv(path, compression="gzip", keep_default_na=False, na_values=[""],
                         dtype={"symbol": str})
        if not df.empty and "symbol" in df.columns:
            factor = np.array([np.prod([f for ex, f in by_ticker.get(s, ())
                                        if day < ex <= fetched]) for s in df["symbol"]])
            for c in ("open", "high", "low", "close"):
                if c in df.columns:
                    df[c] = df[c] / factor
            if "volume" in df.columns:
                df["volume"] = df["volume"] * factor
        tmp = target.with_suffix(".tmp")
        df.to_csv(tmp, compression="gzip", index=False)
        tmp.replace(target)
        moved += 1
    return moved


def massive_splits(provider, path: Path, start: date, today: date) -> list[dict] | None:
    """The split list from ``start`` to today, cached for 12 hours (new splits re-adjust
    every cached day, so it is fetched again rather than kept).  None when the plan
    refuses it."""
    from invest.data import DataError

    with contextlib.suppress(OSError, ValueError, KeyError):
        cached = json.loads(path.read_text())
        age = datetime.now().timestamp() - float(cached["fetched_ts"])
        if cached["start"] <= str(start) and age < 12 * 3600:
            return cached["splits"]
    try:
        splits = provider.splits(start, today)
    except DataError as exc:
        if "not in your plan" in str(exc) or "not_authorized" in str(exc).lower():
            log.warning("Massive refused the split list (%s): falling back to adjusted bars",
                        exc)
            return None
        raise
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"start": str(start), "fetched_ts": datetime.now().timestamp(),
                                "splits": splits}))
    log.info("Massive: %d splits since %s", len(splits), start)
    return splits


def _csv_frames(folder: str | Path) -> dict[str, pd.DataFrame]:
    from invest.data import read_bars_csv

    out = {}
    for f in sorted(Path(folder).glob("*.csv")):
        with contextlib.suppress(Exception):
            out[f.stem.upper()] = read_bars_csv(f)
    if not out:
        raise EventStudyError(f"no <SYMBOL>.csv price files in {folder}")
    return out


# ========================================================================= run
def run(p: Panel, insiders: pd.DataFrame | None = None, filings: pd.DataFrame | None = None,
        resolve=None, wiki: dict | None = None, reddit: pd.DataFrame | None = None,
        locked_from: pd.Timestamp | None = None, n_boot: int = 5000, draws: int = 1000,
        seed: int = 7, meta: dict | None = None,
        filing_families: tuple[str, ...] = ("earnings", "filings")) -> EventResult:
    b = build_bench(p)
    funnel = Funnel()
    parts = []
    if insiders is not None:
        parts.append(insider_events(insiders, p, funnel))
    if filings is not None:
        parts.append(eightk_events(filings, resolve, p, b, funnel, filing_families))
    if wiki is not None:
        parts.append(wiki_events(wiki, p, funnel))
    if reddit is not None:
        parts.append(reddit_events(reddit, p, funnel))
    parts = [x for x in parts if not x.empty]
    raw = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=EVENT_COLUMNS)
    ev = measure(raw, p, b, funnel)
    locked_from = locked_from or default_locked_from(p.days)
    counts = b.tradable_count
    meta = dict(meta or {})
    meta.update(first_day=str(p.days[0].date()), last_day=str(p.days[-1].date()),
                symbols=len(p.symbols), groups=b.K, locked_from=str(locked_from.date()),
                tradable_median=float(np.median(counts[counts > 0])) if (counts > 0).any()
                else 0.0)
    return analyse(ev, p, b, funnel, locked_from, n_boot=n_boot, draws=draws, seed=seed,
                   meta=meta)


def _stop(message: str) -> int:
    print(f"EVENT STUDY STOPPED: {message}", file=sys.stderr)
    return 2


def main(argv=None) -> int:
    from invest.config import ConfigError, InvestSettings
    from invest.data import BarStore, DataError, MassiveProvider, TiingoProvider
    from invest.edgar import Edgar, EdgarError, symbol_resolver
    from invest.universe import build_universe

    ap = argparse.ArgumentParser(description="Event study: insider buys, 8-K announcements "
                                             "and attention spikes")
    ap.add_argument("--prices", choices=["csv", "massive", "tiingo"], required=True,
                    help="csv: files in --csv-dir; massive: every US stock from one grouped "
                         "call per day (free plan: 2 years, ~100 min the first time); tiingo: "
                         "the curated universe plus stocks with insider events")
    ap.add_argument("--csv-dir", help="folder of <SYMBOL>.csv (date,open,high,low,close,volume)")
    ap.add_argument("--days", type=int, default=None,
                    help="calendar days of history (default: 730 massive, 3000 tiingo)")
    ap.add_argument("--families", default="insider,earnings,filings",
                    help=f"comma list among {', '.join(FAMILIES)}")
    ap.add_argument("--attention-dir", default="data_attention",
                    help="where `python -m invest.attention` saved its data")
    ap.add_argument("--data-dir", default="data_events", help="cache (SEC files, prices)")
    ap.add_argument("--max-symbols", type=int, default=450,
                    help="tiingo: cap on symbols downloaded (free plan: 500 new ones a month)")
    ap.add_argument("--locked-from", help="first day of the LOCKED period (YYYY-MM-DD)")
    ap.add_argument("--boot", type=int, default=5000)
    ap.add_argument("--placebo-draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="study_events")
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    unknown = [f for f in fams if f not in FAMILIES]
    if unknown:
        ap.error(f"unknown famil(y/ies): {', '.join(unknown)}")
    if args.prices == "csv" and not args.csv_dir:
        ap.error("--prices csv needs --csv-dir")
    try:
        s = InvestSettings.from_env(args.env_file, broker="sim")
    except (ConfigError, ValueError) as exc:
        return _stop(f"invalid settings: {exc}")
    data = Path(args.data_dir)
    today = date.today()
    end = today - timedelta(days=1)
    days = args.days or (730 if args.prices == "massive" else 3000)
    start = end - timedelta(days=days)
    needs_sec = any(f in fams for f in ("insider", "earnings", "filings")) or args.prices != "csv"
    try:
        frames = None
        if args.prices == "csv":
            frames = _csv_frames(args.csv_dir)
            first = min(v.index[0] for v in frames.values()).date()
            start = max(start, first) if args.days else first
            end = min(end, max(v.index[-1] for v in frames.values()).date())
        edgar = Edgar(s.sec_user_agent, cache_dir=data / "sec") if needs_sec else None
        insiders = issuers = None
        if edgar is not None:
            insiders, issuers = edgar.insiders(start, end)
            log.info("SEC: %d open-market purchases, %d issuer tickers", len(insiders),
                     len(issuers))
        survivorship = ""
        if frames is not None:
            p = panel_from_frames({k: v.loc[str(start):] for k, v in frames.items()})
            survivorship = ("the price files hold the stocks someone chose TODAY; companies "
                            "that failed or were delisted are missing, which flatters any "
                            "rule. Trust only effects that also beat the placebo.")
        elif args.prices == "massive":
            prov = MassiveProvider(s.massive_api_key)
            survivorship = ("none by construction: every stock that traded each day is in "
                            "the data, including those delisted since. Only 2 years on the "
                            "free plan, so the samples are small.")
            # as-traded bars + the split list: the $2 filter and the insider price check
            # then see the price of the day, not one rescaled by a later (reverse) split
            splits = massive_splits(prov, data / "massive_splits.json", start, today)
            if splits is None:
                store = GroupedStore(data / "grouped", prov, adjusted=True)
                survivorship += (" The split list was refused, so prices are adjusted for "
                                 "splits known at download time: a later reverse split can "
                                 "let a sub-$2 stock through the price filter.")
            else:
                store = GroupedStore(data / "grouped_raw", prov, adjusted=False)
                old = data / "grouped"
                if old.is_dir() and any(old.glob("*.csv.gz")):
                    n = migrate_adjusted(GroupedStore(old, prov), store, splits)
                    if n:
                        log.info("reused %d day(s) downloaded earlier as adjusted bars "
                                 "(converted back to as-traded prices)", n)
            try:
                store.update(start, end)
            except DataError as exc:
                text = str(exc).lower()
                hint = (" The free plan covers about 2 years: try --days 700."
                        if "entitlement" in text or "not_authorized" in text else
                        " Run the same command again: every day already downloaded is kept.")
                raise EventStudyError(f"Massive refused a day ({exc}).{hint}") from exc
            uni = set(issuers["symbol"]) if issuers is not None and not issuers.empty else None
            p = panel_from_long(store.load(start, end, uni), splits=splits)
        else:
            want = list(build_universe(s.universe or None, s.exclude)) + [s.benchmark]
            if insiders is not None and not insiders.empty:
                ins = insiders[(insiders["value"] >= 25_000) & ~insiders["plan"].astype(bool)]
                ranked = ins["symbol"].value_counts().index.tolist()
                want += [x for x in ranked if x not in want]
            want = list(dict.fromkeys(want))[:args.max_symbols]
            store = BarStore(data / "tiingo", TiingoProvider(s.tiingo_api_key), days)
            frames = store.update(want, end)
            p = panel_from_frames(frames)
            survivorship = ("Tiingo knows many delisted tickers but a reused ticker can point "
                            "to the wrong company (the insider price check drops most). "
                            "Stocks beyond --max-symbols are left out.")
        filings = resolve = None
        if edgar is not None and any(f in fams for f in ("earnings", "filings")):
            current = edgar.tickers()
            resolve = symbol_resolver(issuers if issuers is not None else
                                      pd.DataFrame(columns=["cik", "symbol", "first", "last"]),
                                      current)
            ciks = sorted({current[x] for x in p.symbols if x in current}
                          | ({int(c) for c, sy in zip(issuers["cik"], issuers["symbol"],
                                                       strict=True) if sy in p.col}
                             if issuers is not None else set()))
            log.info("8-K filings of %d companies (one or two SEC requests each; cached)",
                     len(ciks))
            got, failed = [], []
            for n, cik in enumerate(ciks, 1):
                try:
                    got.append(edgar.eight_ks(cik, start, end))
                except (EdgarError, OSError, ValueError, KeyError) as exc:
                    # one bad company must not end a 40-minute loop; a re-run retries it
                    failed.append(cik)
                    log.warning("8-K of CIK %s skipped: %s", cik, exc)
                if n % 100 == 0:
                    log.info("8-K: %d / %d companies", n, len(ciks))
            if failed:
                log.warning("8-K: %d of %d companies failed (network?); run the same "
                            "command again to fetch them", len(failed), len(ciks))
                survivorship += (f" 8-K filings of {len(failed)} of {len(ciks)} companies "
                                 "could not be downloaded.")
            filings = pd.concat(got, ignore_index=True) if got else None
        wiki = reddit = None
        if "attention" in fams:
            from invest.attention import load_reddit, load_wiki

            wiki = load_wiki(args.attention_dir)
            reddit = load_reddit(args.attention_dir)
            log.info("attention: %d Wikipedia series, %d Reddit snapshot rows", len(wiki),
                     len(reddit))
        locked = pd.Timestamp(args.locked_from) if args.locked_from else None
        res = run(p, insiders=insiders if "insider" in fams else None,
                  filings=filings, resolve=resolve, wiki=wiki, reddit=reddit,
                  locked_from=locked, n_boot=args.boot, draws=args.placebo_draws,
                  seed=args.seed, meta={"prices": args.prices, "survivorship": survivorship,
                                        "families": fams,
                                        "hand_picked": args.prices != "massive"},
                  filing_families=tuple(f for f in fams if f in ("earnings", "filings")))
        write_outputs(res, args.out)
    except (EdgarError, DataError, EventStudyError) as exc:
        return _stop(str(exc))
    except OSError as exc:
        return _stop(f"file or network error: {exc}")
    print(render_report(res).split("## Rule")[0])
    print(f"full report: {Path(args.out) / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
