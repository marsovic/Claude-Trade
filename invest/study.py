"""Signal-level walk-forward study: which parameters of the momentum bot are robust?

WHY.  A real backtest of the bot has a few dozen trades (2-3 slots).  Any parameter chosen
on 26 trades is curve fitting.  Here EVERY 7/7 signal of every day counts as one event (no
slot, no cash limit), so ten years of data give thousands of events, which can honestly
rank parameters.  The study is walk-forward: variants are chosen on the DESIGN years only,
then the chosen ones are checked ONCE on the LOCKED years.

    python -m invest.study --provider tiingo --days 3000 --out study_invest
    python -m invest.study --csv-dir data/us_daily --out study_invest      (offline)
    python -m invest.study ... --locked-from 2022 --start 2017-01-01 --no-vix

Outputs in ``--out``: report.md (plain English), events.csv (the baseline events),
variants.csv (one row per declared variant), summary.json.  The console prints the verdict.

WHAT IS REAL.  ``build_panel``, ``regime_at`` and ``evaluate_day`` are the bot's own
functions.  The baseline events come from ``evaluate_day`` itself (every day with the
regime on).  The other variants only move thresholds, so their signals are re-evaluated
with a vectorised mirror of ``signals.evaluate`` (a few numpy comparisons); that mirror
is audited against the real ``evaluate_day`` (every day for the baseline, a sample of days
for every variant) and the study stops if one cell disagrees.  Stops use
``risk.stop_distance_pct`` and ``risk.stop_reason``; fills follow ``brokers.sim``.

HOW AN EVENT IS TRADED (mirrors ``Engine.process_day`` + ``SimBroker``):
  * signal on the close of day D; buy limit = close * (1 + max_entry_gap) on D+1:
    open <= limit -> fill at min(open * (1 + slippage), limit); open > limit but
    low <= limit -> fill at the limit; otherwise no trade (an unfilled event);
  * initial stop = fill * (1 - stop_distance_pct(limit, ATR)): 3 ATR, clamped 4%-20%;
    R = net P&L / (fill * stop%), net of the fee model (``s.fees``) and slippage;
  * a resting stop-market fills at min(open, stop) * (1 - slip) as soon as the low touches
    it, also on the entry day (gap through the stop = fill at the open, ``stop (gap)``),
    where slip = max(slippage_bps, stop_slippage_atr * entry ATR / stop): fast markets
    fill worse than the flat 5 bp (read with getattr, 0 if the setting does not exist);
  * at every close: highest high since entry, chandelier stop = highest - max(mult * ATR,
    stop_min_pct * highest) with mult = trail_atr_mult or stop_atr_mult, or
    regime_off_stop_mult when the regime is "off"; the stop only moves up, and only when
    it rises by more than 0.05 % (as in the engine);
  * entry-day high: after a limit fill following a gap-up the day's high happened BEFORE
    the fill, so it is not inherited: ``highest`` starts at max(fill, close), a price the
    position did live through (the engine's ``_usable_high``); after an open fill the whole
    day is ours and the day's high counts;
  * exits decided at a close are market sells at the next open (carried to the next session
    when the symbol has no bar): the weekly rank exit on the last session of the ISO week
    (Friday, or Thursday before a Friday holiday; pct_rank_hold below 1 - top_pct_hold, or
    6-month momentum negative) and the time stop (time_stop_days sessions, below
    time_stop_min_r, no longer in the entry top);
  * a position is closed at the close of session ``horizon`` (default 250) and a position
    still open at the end of the data is marked at the last close, both with exit costs.

SIMPLIFICATIONS versus the live engine (honesty over pretence):
  1. No slot, cash, theme or drawdown limits: every signal is traded alone.  The bot takes
     only the best-scored few; this study measures the SIGNAL, not the portfolio.
  2. One position per symbol (the bot cannot buy what it already holds): a later signal
     on a symbol still open in the SAME variant is dropped (``--allow-overlap`` lifts it).
     events.csv lists every raw 7/7 signal with a ``counted`` flag.
  3. No pyramiding (+50 % at +1.5 R): R is the R of the first entry only.
  4. No earnings blackout, news, mania or insider data (no free history; the backtest
     has them off too).  Weekly-loss pause, fee halt and survival rule are not modelled.
  5. Costs are evaluated for a reference position: ``--ref-equity`` (default
     initial_capital) times the bot's weight min(risk/stop%, 1/slots, max_weight);
     EUR/USD is constant so FX P&L cancels; no share rounding, no minimum order size.
     Entries refused by ``max_cost_r`` are skipped.
  6. The 250-session horizon cap truncates the biggest winners, identically for all
     variants; a position still open at the end of the data is marked to market.
  7. Daily bars only: when the low touches the stop the stop fills, even on the entry
     day, whatever happened first inside the bar (SimBroker does the same).
  8. Regime comes from ``regime_at`` with VIX lagged one observation, as the backtest does.
  9. Survivorship bias of today's universe and split-adjusted prices (no dividends), as
     in the backtest.
 10. Threshold variants that touch panel fields (min_price, atr_period, ...) or regime
     fields are refused: they need their own panel.

STATISTICS.  Metric = R measured against the BASELINE initial risk of the same event
("R_ref"), so that a wider stop is compared fairly with the default one (for every other
variant R_ref is exactly the R of the trade).  Paired block bootstrap over ISO weeks
(whole weeks are resampled, baseline and variant together) for the difference of mean R,
95 % interval; Holm-Bonferroni correction over the adoptable variants (declared "better";
the "worse" diagnostics, which can never be adopted, are corrected among themselves),
repeated with 4-week blocks because trades last weeks; effective sample size from the
intra-week correlation (warning when events cluster); the same difference de-clustered to
the best-scored event per week and per day; sign stability per calendar year and
leave-one-year-out; plateau check on neighbouring thresholds; the average equity
contribution (R x the bot's real position weight) as a money check; random-entry placebo
(same exits and costs).  Information-discreteness cut-offs come from design-period
signals only; design trades that run into the locked years are purged.

DECISION RULE (fixed before looking at any data; see ``DECISION_RULE``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from invest.backtest import load_csv_dir
from invest.config import InvestSettings
from invest.data import BarStore, fred_series, make_provider
from invest.risk import stop_distance_pct, stop_reason
from invest.signals import Regime, build_panel, evaluate_day, regime_at
from invest.universe import build_universe, data_symbols

log = logging.getLogger("invest.study")

BASELINE = "baseline"

DECISION_RULE = """\
A variant is adopted only if ALL of these hold.
On the DESIGN years (chosen with no access to the LOCKED years):
  1. direction: it was declared 'better' in advance and its mean R is higher than the baseline;
  2. significance: the Holm-corrected p-value of the paired block bootstrap (whole ISO weeks)
     is <= 0.05 over the family of all adoptable (declared 'better') variants (so the 95%
     interval of the difference excludes 0 after correction), and so is the same test with
     4-week blocks (trades last weeks and overlap, which a 1-week block would understate);
     the 'worse' diagnostics are corrected among themselves and can never be adopted;
  3. stability: the sign of the difference is the same in at least 70% of the design
     calendar years that have enough events, AND dropping any single design year leaves at
     least half of the gain (no single year carries the effect);
  4. plateau: neighbouring thresholds point the same way (no spike; at least one
     neighbour must exist and none may reverse or collapse to under half of the gain);
  5. coverage: it keeps at least 60% of the baseline events, or the events it keeps are
     still at least 2 times what the slots can absorb in a year;
  6. de-clustered check: the difference is also positive when only the best-scored event
     of each week is kept;
  7. money check: the average equity contribution per trade (R x the bot's real position
     weight) is also higher.
On the LOCKED years, evaluated ONCE and only for variants that passed 1-7:
  8. the difference has the same sign and at least half of the design magnitude.
Otherwise: keep the defaults.  A higher mean alone never counts."""

# ----------------------------------------------------------------------------------
# Variants, declared up front.  ``expect`` is the direction of the hypothesis:
#   "better" = we expect a higher mean R than the baseline (the only adoptable ones);
#   "worse"  = a looser setting that should dilute quality: it tells whether the default
#              threshold earns its keep; it is never adopted.
# ``role="control"``: must reproduce the baseline exactly (self-test of the machinery),
# excluded from the multiple-testing family.
# ----------------------------------------------------------------------------------


@dataclass
class Variant:
    name: str
    family: str
    overrides: dict = field(default_factory=dict)  # InvestSettings field -> value
    flt: str = ""  # entry-quality filter computed from the panel ("" = none)
    x: float = 0.0  # position on the family's axis (setting value / threshold / window)
    expect: str = "better"
    why: str = ""
    role: str = "test"


def _sv(name: str, value: float, expect: str, why: str, role: str = "test") -> Variant:
    return Variant(f"{name}={value:g}", name, {name: float(value)}, "", float(value), expect,
                   why, role)


def _fv(flt: str, x: float, expect: str, why: str, label: str) -> Variant:
    return Variant(label, flt, {}, flt, float(x), expect, why)


VARIANTS: list[Variant] = [
    # --- exits ---------------------------------------------------------------------
    _sv("stop_atr_mult", 2.5, "worse", "tighter stop: more whipsaws"),
    _sv("stop_atr_mult", 3.5, "better", "wider stop: fewer premature exits"),
    _sv("stop_atr_mult", 4.0, "better", "wider stop: fewer premature exits"),
    _sv("trail_atr_mult", 3.0, "better", "identity (stop 3, trail 3 = default): self-test",
        role="control"),
    _sv("trail_atr_mult", 4.0, "better", "looser trail lets winners run (stop stays 3)"),
    _sv("trail_atr_mult", 5.0, "better", "looser trail lets winners run (stop stays 3)"),
    # --- entry thresholds ----------------------------------------------------------
    _sv("top_pct_entry", 0.05, "better", "only the very best momentum names"),
    _sv("top_pct_entry", 0.075, "better", "only the very best momentum names"),
    _sv("top_pct_entry", 0.15, "worse", "looser ranking dilutes quality"),
    _sv("top_pct_entry", 0.20, "worse", "looser ranking dilutes quality"),
    _sv("near_high_min", 0.80, "worse", "farther from the 52-week high dilutes quality"),
    _sv("near_high_min", 0.90, "better", "closer to the 52-week high"),
    _sv("near_high_min", 0.95, "better", "closer to the 52-week high"),
    _sv("volume_factor", 1.0, "worse", "no volume confirmation dilutes quality"),
    _sv("volume_factor", 1.6, "better", "stronger volume confirmation"),
    _sv("volume_factor", 2.0, "better", "stronger volume confirmation"),
    _sv("max_extension", 0.15, "better", "refuse stocks too far above their SMA50"),
    _sv("max_extension", 0.20, "better", "refuse stocks too far above their SMA50"),
    _sv("max_extension", 0.35, "worse", "allowing more extension dilutes quality"),
    _sv("rsi_max", 70.0, "better", "refuse overbought entries"),
    _sv("rsi_max", 75.0, "better", "refuse overbought entries"),
    # --- entry-quality filters computed from the panel (the engine is not touched) -----
    _fv("gap_atr", 0.25, "better", "do not chase a gap-up at the open", "gap_atr<=0.25"),
    _fv("gap_atr", 0.5, "better", "do not chase a gap-up at the open", "gap_atr<=0.5"),
    _fv("gap_atr", 1.0, "better", "do not chase a gap-up at the open", "gap_atr<=1"),
    _fv("ext_ema20_atr", 2.5, "better", "not too far above the EMA20", "ext_ema20_atr<=2.5"),
    _fv("ext_ema20_atr", 3.5, "better", "not too far above the EMA20", "ext_ema20_atr<=3.5"),
    _fv("ext_ema20_atr", 4.5, "better", "not too far above the EMA20", "ext_ema20_atr<=4.5"),
    _fv("event_bar", 1, "better", "no news-shock bar (gap >= 2 ATR, volume >= 2.5x) in the "
        "last 1 session", "no_event_bar(1)"),
    _fv("event_bar", 3, "better", "no news-shock bar (gap >= 2 ATR, volume >= 2.5x) in the "
        "last 3 sessions", "no_event_bar(3)"),
    _fv("event_bar", 5, "better", "no news-shock bar (gap >= 2 ATR, volume >= 2.5x) in the "
        "last 5 sessions", "no_event_bar(5)"),
    _fv("id_smooth", 0.80, "better", "smooth (continuous) momentum: keep the 80% smoothest "
        "signals (information discreteness; cut-off from design-period signals only)",
        "ID_smoothest_80%"),
    _fv("id_smooth", 0.65, "better", "smooth momentum: keep the 65% smoothest signals",
        "ID_smoothest_65%"),
    _fv("id_smooth", 0.50, "better", "smooth momentum: keep the 50% smoothest signals",
        "ID_smoothest_50%"),
]

# Where the baseline sits on each family's axis (for the plateau check).  A setting family
# reads the baseline value from the settings; a filter family starts from "no filter".
FAMILY_BASE: dict[str, tuple[str, float | str]] = {
    "stop_atr_mult": ("attr", "stop_atr_mult"),
    "trail_atr_mult": ("trail", ""),
    "top_pct_entry": ("attr", "top_pct_entry"),
    "near_high_min": ("attr", "near_high_min"),
    "volume_factor": ("attr", "volume_factor"),
    "max_extension": ("attr", "max_extension"),
    "rsi_max": ("attr", "rsi_max"),
    "gap_atr": ("value", math.inf),
    "ext_ema20_atr": ("value", math.inf),
    "event_bar": ("value", 0.0),
    "id_smooth": ("value", 1.0),  # keep 100% of the signals
}

# Settings that move the panel or the regime: a variant may not change them.
PANEL_FIELDS = frozenset({
    "min_price", "min_dollar_volume", "min_history", "atr_min_pct", "atr_max_pct",
    "max_daily_jump", "min_beta", "atr_period", "breakout_lookback", "benchmark",
    "regime_sma", "vix_max_entry", "vix_term_max", "vix_exit"})
ENTRY_FIELDS = ("top_pct_entry", "near_high_min", "volume_factor", "max_extension", "rsi_max")
EXIT_FIELDS = ("stop_atr_mult", "regime_off_stop_mult", "stop_min_pct",
               "stop_max_pct", "time_stop_days", "time_stop_min_r", "top_pct_hold",
               "top_pct_entry", "rebalance_weekday", "max_entry_gap", "slippage_bps",
               "max_cost_r", "risk_per_trade", "slots", "slots_small_equity", "small_equity",
               "max_weight", "eurusd_fallback")
GRID_COLS = ("open", "high", "low", "close", "atr", "atr_pct", "sma200", "ema20", "ema50",
             "rsi14", "r12_1", "r6_1", "h52", "hi_prev", "vol_ratio", "ext50", "score",
             "pct_rank", "pct_rank_hold")


class StudyError(RuntimeError):
    pass


class LeakError(AssertionError):
    """Raised when the DESIGN selection would see a LOCKED event."""


# ==================================================================================
# parameters
# ==================================================================================
@dataclass
class Params:
    n_boot: int = 10000
    seed: int = 12345
    alpha: float = 0.05
    horizon: int = 250
    design_frac: float = 0.60
    locked_from: int | None = None
    min_locked_events: int = 150
    min_design_events: int = 40
    placebo_draws: int = 2000
    placebo_pool: int = 100_000
    allow_overlap: bool = False
    audit: str = "sample"  # "sample" | "full" | "off"
    audit_days: int = 40
    ref_equity: float | None = None
    year_share_min: float = 0.70
    block_weeks: int = 4  # robustness block length (the main test uses 1 ISO week)
    keep_min: float = 0.60
    min_year_events: int = 8
    capacity_mult: float = 2.0
    magnitude_min: float = 0.5
    plateau_outer_min: float = 0.5
    start: str | None = None
    end: str | None = None


# ==================================================================================
# grid: the panel as (sessions x symbols) arrays
# ==================================================================================
@dataclass
class Grid:
    cal: pd.DatetimeIndex
    syms: list[str]
    A: dict[str, np.ndarray]
    eligible: np.ndarray
    weekday: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def __post_init__(self):
        if self.weekday.size == 0:
            self.weekday = np.asarray(self.cal.weekday)

    _rebal: dict = field(default_factory=dict, repr=False)

    @property
    def T(self) -> int:
        return len(self.cal)

    @property
    def N(self) -> int:
        return len(self.syms)

    def col(self, name: str) -> np.ndarray:
        if name in self.A:
            return self.A[name]
        return np.full((self.T, self.N), np.nan)

    def rebalance(self, weekday: int) -> np.ndarray:
        """Sessions on which the engine runs its weekly rank exit (``Engine._rebalance_day``):
        the last session of the ISO week for the Friday setting (Friday, or the day before a
        Friday holiday), otherwise the configured weekday."""
        if weekday not in self._rebal:
            flags = self.weekday == weekday
            if weekday == 4:
                try:
                    from invest.markets import next_trading_day
                except ImportError:  # older engine: plain weekday
                    next_trading_day = None
                if next_trading_day is not None:
                    flags = np.array([
                        next_trading_day(d).isocalendar()[:2] != d.isocalendar()[:2]
                        for d in (x.date() for x in self.cal)])
            self._rebal[weekday] = flags
        return self._rebal[weekday]


def build_grid(panel: Mapping[str, pd.DataFrame], cal: pd.DatetimeIndex) -> Grid:
    syms = list(panel)
    A = {col: np.column_stack([panel[sym][col].reindex(cal).to_numpy(dtype=float)
                               for sym in syms]) for col in GRID_COLS}
    elig = np.column_stack([panel[sym]["eligible"].reindex(cal, fill_value=False)
                            .to_numpy(dtype=bool) for sym in syms])
    return Grid(cal=cal, syms=syms, A=A, eligible=elig)


def lag_series(series: pd.Series | None) -> pd.Series | None:
    """Value known at a session's close: the previous observation (as backtest._lag)."""
    if series is None or series.empty:
        return series
    return series.shift(1).dropna()


def regime_list(bars, vix, vix3m, s, cal) -> list[Regime | None]:
    """The bot's own ``regime_at`` for every session (None while history is too short)."""
    out: list[Regime | None] = []
    for i, day in enumerate(cal):
        out.append(regime_at(day, bars, vix, vix3m, s) if i >= s.regime_sma else None)
    return out


def regime_arrays(regs: Sequence[Regime | None]) -> tuple[np.ndarray, np.ndarray]:
    on = np.array([bool(r.on) if r is not None else False for r in regs])
    off = np.array([bool(r.off) if r is not None else False for r in regs])
    return on, off


# ==================================================================================
# signals: vectorised mirror of signals.evaluate + audit against the real function
# ==================================================================================
def signal_matrix(g: Grid, on: np.ndarray, s) -> np.ndarray:
    """(T, N) bool: the 7 conditions of ``signals.evaluate`` for every session and stock."""
    A = g.A
    with np.errstate(invalid="ignore"):
        c = A["close"]
        ok = g.eligible & on[:, None]
        ok &= (A["pct_rank"] >= 1 - s.top_pct_entry) & (A["score"] > 0)
        ok &= (A["r12_1"] > 0) & (A["r6_1"] > 0)
        ok &= (c > A["sma200"]) & (A["ema20"] > A["ema50"])
        ok &= A["h52"] >= s.near_high_min
        ok &= (c >= A["hi_prev"]) & (A["vol_ratio"] >= s.volume_factor)
        ok &= (A["ext50"] <= s.max_extension) & (A["rsi14"] <= s.rsi_max)
    return ok


def real_signals(panel, g: Grid, regs, s, days: Sequence[int]) -> np.ndarray:
    """The bot's own ``evaluate_day`` on the given sessions (regime-off days cannot signal:
    the first of the 7 conditions fails)."""
    out = np.zeros((g.T, g.N), dtype=bool)
    pos = {sym: k for k, sym in enumerate(g.syms)}
    for i in days:
        reg = regs[i]
        if reg is None or not reg.on:
            continue
        for sym, ev in evaluate_day(panel, g.cal[i], reg, s).items():
            if ev.signal:
                out[i, pos[sym]] = True
    return out


def audit_signals(panel, g: Grid, regs, s, mirror: np.ndarray, days: Sequence[int],
                  label: str) -> int:
    """Compare mirror and real signals on ``days``; raise StudyError on any mismatch."""
    real = real_signals(panel, g, regs, s, days)
    idx = np.asarray(list(days), dtype=int)
    bad = np.argwhere(real[idx] != mirror[idx])
    if len(bad):
        i, k = bad[0]
        raise StudyError(
            f"signal mirror disagrees with signals.evaluate_day for '{label}' "
            f"({len(bad)} cell(s); first: {g.syms[k]} on {g.cal[idx[i]].date()}, real="
            f"{bool(real[idx[i], k])}). invest/signals.py changed: update signal_matrix().")
    return int(len(idx) * g.N)


# ==================================================================================
# candidate features
# ==================================================================================
def information_discreteness(C: np.ndarray, t0: np.ndarray, j: np.ndarray) -> np.ndarray:
    """ID = sign(PRET) * (%negative days - %positive days) over sessions t-252..t-21
    (smooth momentum has a LOW, negative ID).  NaN when fewer than 100 returns."""
    with np.errstate(invalid="ignore", divide="ignore"):
        lr = np.vstack([np.full((1, C.shape[1]), np.nan), np.log(C[1:] / C[:-1])])
    valid = np.isfinite(lr)
    pos = np.where(valid, lr > 0, 0).astype(float)
    neg = np.where(valid, lr < 0, 0).astype(float)
    cums = []
    for arr in (valid.astype(float), pos, neg, np.where(valid, lr, 0.0)):
        cums.append(np.vstack([np.zeros((1, C.shape[1])), np.cumsum(arr, axis=0)]))
    b = t0 - 21
    a = np.maximum(t0 - 252, 0)
    ok = b >= a
    bb = np.where(ok, b, 0)

    def win(cum):
        return cum[bb + 1, j] - cum[a, j]

    n, p, q, tot = (win(c) for c in cums)
    with np.errstate(invalid="ignore", divide="ignore"):
        idv = np.sign(tot) * (q / n - p / n)
    return np.where(ok & (n > 100), idv, np.nan)


def candidate_features(g: Grid, t0: np.ndarray, j: np.ndarray) -> dict[str, np.ndarray]:
    A = g.A
    T = g.T
    te = np.minimum(t0 + 1, T - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap_atr = np.where(t0 + 1 < T, (A["open"][te, j] - A["close"][t0, j]) / A["atr"][t0, j],
                           np.nan)
        ext = (A["close"][t0, j] - A["ema20"][t0, j]) / A["atr"][t0, j]
        # news-shock bar: |open / previous close - 1| >= 2 ATR(previous day) and volume >= 2.5x
        gap_prev = np.full((T, g.N), np.nan)
        gap_prev[1:] = np.abs(A["open"][1:] - A["close"][:-1]) / A["atr"][:-1]
        shock = (gap_prev >= 2.0) & (A["vol_ratio"] >= 2.5)
    out = {
        "gap_atr": gap_atr,
        "ext_ema20_atr": ext,
        "id_smooth": information_discreteness(A["close"], t0, j),
        "score": A["score"][t0, j],
        "pct_rank": A["pct_rank"][t0, j],
        "rsi": A["rsi14"][t0, j],
        "vol_ratio": A["vol_ratio"][t0, j],
        "h52": A["h52"][t0, j],
        "ext50": A["ext50"][t0, j],
        "atr_pct": A["atr_pct"][t0, j],
        "price": A["close"][t0, j],
    }
    for w in (1, 3, 5):
        flag = np.zeros(len(t0), dtype=bool)
        for k in range(w):
            flag |= shock[np.maximum(t0 - k, 0), j]
        out[f"event_bar_{w}"] = flag
    return out


def filter_mask(v: Variant, feats: Mapping[str, np.ndarray],
                id_cutoffs: Mapping[float, float] | None = None) -> np.ndarray | None:
    """Boolean keep-mask of an entry-quality filter (None = no filter).  NaN never passes.

    ``id_smooth`` keeps the smoothest ``v.x`` share of the signals: the information-
    discreteness cut-off comes from ``id_cutoffs`` (computed on design-period signals only,
    so that no later data sets a threshold)."""
    if not v.flt:
        return None
    with np.errstate(invalid="ignore"):
        if v.flt == "gap_atr":
            return feats["gap_atr"] <= v.x
        if v.flt == "ext_ema20_atr":
            return feats["ext_ema20_atr"] <= v.x
        if v.flt == "id_smooth":
            if not id_cutoffs or v.x not in id_cutoffs:
                raise StudyError("id_smooth needs a design-period cut-off")
            return feats["id_smooth"] <= id_cutoffs[v.x]
        if v.flt == "event_bar":
            return ~feats[f"event_bar_{int(v.x)}"]
    raise StudyError(f"unknown filter {v.flt!r}")


# ==================================================================================
# trade simulator, vectorised over events (mirrors Engine.process_day + SimBroker)
# ==================================================================================
@dataclass
class Sim:
    filled: np.ndarray
    kind: np.ndarray  # 0 open fill, 1 limit fill after a gap-up
    fill: np.ndarray
    sp: np.ndarray  # initial stop distance (fraction of the fill)
    exit_t: np.ndarray
    exit_px: np.ndarray
    reason: np.ndarray
    held: np.ndarray
    net_ret: np.ndarray  # net return of the trade per $ of entry notional, after all costs
    weight: np.ndarray  # the bot's position weight for this stop distance

    @property
    def n(self) -> int:
        return len(self.filled)


def _trail_mult(s) -> float:
    return getattr(s, "trail_atr_mult", 0.0) or s.stop_atr_mult


def _fee_fractions(s, fill, x, sp, ref_equity):
    """Entry and exit fees as fractions of the entry notional, at the reference size."""
    fees = s.fees
    if fees.commission_min == 0 and fees.commission_per_share == 0:
        rate = min(fees.commission_pct, fees.commission_max_pct) + fees.fx_fee_pct
        return np.full_like(fill, rate), x / fill * rate
    slots = s.slots_small_equity if ref_equity < s.small_equity else s.slots
    f_in = np.empty_like(fill)
    f_out = np.empty_like(fill)
    for i in range(len(fill)):
        w = min(s.risk_per_trade / sp[i], 1.0 / slots, s.max_weight)
        qty = ref_equity * w * s.eurusd_fallback / fill[i]
        f_in[i] = fees.cost(qty, fill[i], True) / (qty * fill[i])
        f_out[i] = fees.cost(qty, x[i], True) / (qty * fill[i])
    return f_in, f_out


def position_weight(sp: np.ndarray, s, ref_equity: float) -> np.ndarray:
    slots = s.slots_small_equity if ref_equity < s.small_equity else s.slots
    return np.minimum(np.minimum(s.risk_per_trade / sp, 1.0 / slots), s.max_weight)


def simulate(g: Grid, t0: np.ndarray, j: np.ndarray, s, off: np.ndarray, horizon: int = 250,
             ref_equity: float | None = None) -> Sim:
    """Trade every event (signal on session t0, stock j) with the bot's exit logic."""
    T = g.T
    n = len(t0)
    ref_equity = s.initial_capital if ref_equity is None else ref_equity
    OP, H, L, C, ATR = (g.col(k) for k in ("open", "high", "low", "close", "atr"))
    PH, R6 = g.col("pct_rank_hold"), g.col("r6_1")
    slip = s.slippage_bps / 1e4
    stop_slip_atr = getattr(s, "stop_slippage_atr", 0.0)
    trail = _trail_mult(s)
    smin = s.stop_min_pct
    rebal = g.rebalance(s.rebalance_weekday)
    te = t0 + 1
    tec = np.minimum(te, T - 1)
    limit = C[t0, j] * (1 + s.max_entry_gap)
    sig_atr = ATR[t0, j]
    o1, l1 = OP[tec, j], L[tec, j]
    with np.errstate(invalid="ignore"):
        open_fill = (te < T) & (o1 <= limit)
        lim_fill = (te < T) & ~open_fill & (l1 <= limit)
    fill = np.where(open_fill, np.minimum(o1 * (1 + slip), limit), limit)
    sp = np.array([stop_distance_pct(float(p), float(a), s) for p, a in zip(limit, sig_atr)],
                  dtype=float)
    filled = (open_fill | lim_fill) & np.isfinite(sp) & np.isfinite(fill)
    # max_cost_r: the engine refuses an entry whose round trip costs too much (in R)
    f_lim, _ = _fee_fractions(s, np.where(filled, limit, 1.0), np.where(filled, limit, 1.0),
                              np.where(filled, sp, 0.1), ref_equity)
    with np.errstate(invalid="ignore", divide="ignore"):
        filled &= (2 * f_lim / sp) <= s.max_cost_r

    res = Sim(filled=filled, kind=np.where(open_fill, 0, 1).astype(np.int8),
              fill=np.where(filled, fill, np.nan), sp=np.where(filled, sp, np.nan),
              exit_t=np.full(n, -1, dtype=np.int64), exit_px=np.full(n, np.nan),
              reason=np.full(n, "", dtype=object), held=np.zeros(n, dtype=np.int64),
              net_ret=np.full(n, np.nan), weight=np.full(n, np.nan))

    last_valid = np.full(g.N, np.nan)
    for k in range(g.N):
        ok = np.nonzero(np.isfinite(C[:, k]))[0]
        last_valid[k] = C[ok[-1], k] if len(ok) else np.nan

    ids = np.nonzero(filled)[0]
    jj, tt = j[ids], te[ids]
    fl, spp, a0 = fill[ids], sp[ids], sig_atr[ids]
    kind = res.kind[ids]
    stop = fl * (1 - spp)
    high = fl.copy()
    held = np.zeros(len(ids), dtype=np.int64)
    pend = np.zeros(len(ids), dtype=np.int8)
    rank_top = f"rank exit (fell out of the top {s.top_pct_hold:.0%})"
    pend_text = np.array(["", rank_top, "rank exit (6-month momentum turned negative)",
                          "time stop"], dtype=object)

    d = 0
    while ids.size:
        kk = np.minimum(tt + d, T - 1)
        o, h, lo, c, a = OP[kk, jj], H[kk, jj], L[kk, jj], C[kk, jj], ATR[kk, jj]
        has = ~np.isnan(o)
        n_act = ids.size
        done = np.zeros(n_act, dtype=bool)
        px = np.full(n_act, np.nan)
        why = np.full(n_act, "", dtype=object)
        with np.errstate(invalid="ignore"):
            # (a) an exit decided at the previous close: market sell at this open
            if d > 0:
                m = (pend > 0) & has
                px[m] = o[m] * (1 - slip)
                why[m] = pend_text[pend[m]]
                done |= m
                # a sell queued for a symbol with no bar today is carried to the next session
            # (b) resting stop-market: fills at the open when the bar gaps through it, and
            # slips by max(slippage, stop_slippage_atr * entry ATR / stop) in fast markets
            m = ~done & has & (lo <= stop)
            if m.any():
                st_slip = np.maximum(slip, stop_slip_atr * a0 / np.maximum(stop, 1e-9))
                px[m] = np.minimum(o[m], stop[m]) * (1 - st_slip[m])
                for q in np.nonzero(m)[0]:
                    why[q] = stop_reason(float(px[q]), float(stop[q]), float(fl[q]),
                                         float(st_slip[q]))
                done |= m
            alive = ~done & has
            # horizon cap: sold at the close of session ``horizon``
            hz = alive & (d >= horizon)
            px[hz] = c[hz] * (1 - slip)
            why[hz] = "horizon cap"
            done |= hz
            alive &= ~hz
            # (c) end-of-day management of the surviving positions
            if d == 0:
                # entry-day high: after a limit fill following a gap-up, the morning high came
                # BEFORE the fill; the position only lived through max(fill, close)
                usable = np.where(kind == 1, np.minimum(h, np.maximum(fl, c)), h)
                high = np.where(alive, np.maximum(fl, usable), high)
            else:
                high = np.where(alive, np.maximum(high, h), high)
            atr_eff = np.where(np.isnan(a), a0, a)
            mult = np.where(off[kk], s.regime_off_stop_mult, trail)
            dist = np.maximum(mult * atr_eff, smin * high)
            new_stop = np.maximum(stop, np.where(atr_eff > 0, high - dist, stop))
            stop = np.where(alive & (new_stop > stop * 1.0005), new_stop, stop)
            if d > 0:
                held = held + alive
            rmult = (c - fl) / (fl * spp)
            hold, r6 = PH[kk, jj], R6[kk, jj]
            fri = rebal[kk]
            out_top = np.isnan(hold) | (hold < 1 - s.top_pct_hold)
            rank1 = alive & fri & out_top
            rank2 = alive & fri & ~out_top & (r6 < 0)
            tstop = (alive & ~(rank1 | rank2) & (held >= s.time_stop_days)
                     & (rmult < s.time_stop_min_r)
                     & (np.isnan(hold) | (hold < 1 - s.top_pct_entry)))
            pend = np.where(rank1, 1, np.where(rank2, 2, np.where(tstop, 3, pend))
                            ).astype(np.int8)
        # end of the data: mark to the last close
        eod = ~done & (tt + d >= T - 1)
        if eod.any():
            last = np.where(has, c, last_valid[jj])
            px[eod] = last[eod] * (1 - slip)
            why[eod] = "open (end of data)"
            done |= eod
        if done.any():
            sel = ids[done]
            res.exit_t[sel] = kk[done]
            res.exit_px[sel] = px[done]
            res.reason[sel] = why[done]
            res.held[sel] = held[done]
            keep = ~done
            ids, jj, tt, fl, spp, a0 = ids[keep], jj[keep], tt[keep], fl[keep], spp[keep], a0[keep]
            kind, stop, high, held, pend = (kind[keep], stop[keep], high[keep], held[keep],
                                            pend[keep])
        d += 1

    f = np.nonzero(filled)[0]
    f_in, f_out = _fee_fractions(s, res.fill[f], res.exit_px[f], res.sp[f], ref_equity)
    res.net_ret[f] = res.exit_px[f] / res.fill[f] - 1 - f_in - f_out
    res.weight[f] = position_weight(res.sp[f], s, ref_equity)
    return res


# ==================================================================================
# event sets
# ==================================================================================
@dataclass
class EventSet:
    """The trades one variant takes: after its signals, filters, fills and the
    one-position-per-symbol rule."""

    name: str
    t0: np.ndarray  # signal session index
    j: np.ndarray
    exit_t: np.ndarray  # session index of the exit fill
    R: np.ndarray  # R against the variant's own initial risk (net of costs)
    Rref: np.ndarray  # R against the BASELINE initial risk (the decision metric)
    eq: np.ndarray  # equity contribution in % at the bot's position weight
    score: np.ndarray
    held: np.ndarray
    reason: np.ndarray

    def sel(self, mask: np.ndarray) -> EventSet:
        return EventSet(self.name, *(getattr(self, f)[mask] for f in (
            "t0", "j", "exit_t", "R", "Rref", "eq", "score", "held", "reason")))

    @property
    def n(self) -> int:
        return len(self.t0)


def one_position_per_symbol(order: np.ndarray, j, t0, exit_t, member) -> np.ndarray:
    """Greedy chain per symbol: a signal is taken only if the previous position of the same
    symbol was closed on or before the signal session (the bot cannot add to or double up)."""
    keep = np.zeros(len(j), dtype=bool)
    free_from: dict[int, int] = {}
    for i in order[member[order]]:
        k = int(j[i])
        if t0[i] >= free_from.get(k, -1):
            keep[i] = True
            free_from[k] = int(exit_t[i])
    return keep


# ==================================================================================
# statistics
# ==================================================================================
def holm_adjust(p: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise error control)."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def effective_n(values: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    """(n_eff, intra-group correlation, mean events per active group) for clustered events
    (one-way ANOVA estimator; the design effect is 1 + (m - 1) * rho)."""
    n = len(values)
    if n < 3:
        return float(n), 0.0, 1.0
    codes, uniq = pd.factorize(groups)
    k = len(uniq)
    cnt = np.bincount(codes).astype(float)
    mean_all = values.mean()
    gm = np.bincount(codes, weights=values) / cnt
    ssb = float((cnt * (gm - mean_all) ** 2).sum())
    ssw = float(((values - gm[codes]) ** 2).sum())
    m_bar = float((cnt**2).sum() / n)
    if k < 2 or n <= k or ssw <= 0:
        return float(n), 0.0, m_bar
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - float((cnt**2).sum()) / n) / (k - 1)
    rho = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    rho = min(max(rho, 0.0), 1.0)
    deff = 1 + (m_bar - 1) * rho
    return n / deff, rho, m_bar


def _boot_means(S: np.ndarray, Cn: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap means (B, K) resampling whole weeks (rows of S / Cn) with replacement."""
    W = S.shape[0]
    rng = np.random.default_rng(seed)
    mult = rng.multinomial(W, np.full(W, 1.0 / W), size=n_boot).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (mult @ S) / (mult @ Cn)


def _p_two_sided(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan")
    b = len(d)
    lo = (np.sum(d <= 0) + 1) / (b + 1)
    hi = (np.sum(d >= 0) + 1) / (b + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def _ci(d: np.ndarray, level: float) -> tuple[float, float]:
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return float("nan"), float("nan")
    q = (1 - level) / 2 * 100
    return float(np.percentile(d, q)), float(np.percentile(d, 100 - q))


def block_boot_diff(weeks: np.ndarray, values: np.ndarray, mask: np.ndarray,
                    base_mask: np.ndarray | None = None, n_boot: int = 2000, seed: int = 0,
                    level: float = 0.95) -> dict:
    """Paired block bootstrap by week: mean(values | mask) - mean(values | base_mask)
    (base = all events when omitted).  Whole weeks are resampled, so the two means move
    together.  Deterministic for a given seed.  Returns diff, lo, hi, p (two-sided)."""
    weeks = np.asarray(weeks)
    values = np.asarray(values, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    base_mask = np.ones(len(values), dtype=bool) if base_mask is None else np.asarray(base_mask)
    codes, uniq = pd.factorize(weeks)
    W = len(uniq)
    cols = []
    for m in (mask, base_mask):
        cols.append((np.bincount(codes[m], weights=values[m], minlength=W),
                     np.bincount(codes[m], minlength=W).astype(float)))
    S = np.column_stack([c[0] for c in cols])
    Cn = np.column_stack([c[1] for c in cols])
    means = _boot_means(S, Cn, n_boot, seed)
    d = means[:, 0] - means[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        point = S[:, 0].sum() / Cn[:, 0].sum() - S[:, 1].sum() / Cn[:, 1].sum()
    lo, hi = _ci(d, level)
    return {"diff": float(point), "lo": lo, "hi": hi, "p": _p_two_sided(d)}


def _first_per_key(key: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Indices of the best-scored event of each key (ties: first)."""
    order = np.lexsort((-np.nan_to_num(score, nan=-1e9), key))
    k = key[order]
    first = np.ones(len(k), dtype=bool)
    first[1:] = k[1:] != k[:-1]
    return order[first]


@dataclass
class Calendar:
    cal: pd.DatetimeIndex
    wk: np.ndarray  # compact ISO-week id per session (non-decreasing)
    yr: np.ndarray  # calendar year per session


def make_calendar(cal: pd.DatetimeIndex) -> Calendar:
    iso = cal.isocalendar()
    key = (iso["year"].to_numpy(dtype=int) * 100 + iso["week"].to_numpy(dtype=int))
    wk = np.concatenate([[0], np.cumsum(key[1:] != key[:-1])]).astype(int)
    return Calendar(cal=cal, wk=wk, yr=np.asarray(cal.year))


def _perf(R: np.ndarray) -> dict:
    n = len(R)
    if n == 0:
        return {"n": 0, "win": np.nan, "avg": np.nan, "payoff": np.nan, "pf": np.nan}
    w, lo = R[R > 0], R[R <= 0]
    return {"n": n, "win": float((R > 0).mean()), "avg": float(R.mean()),
            "payoff": float(w.mean() / -lo.mean()) if len(w) and len(lo) and lo.mean() < 0
            else np.nan,
            "pf": float(w.sum() / -lo.sum()) if len(lo) and lo.sum() < 0 else np.nan}


def _leave_one_year_out(ev: EventSet, base: EventSet, yv, yb, years) -> float:
    """Smallest difference of mean R (variant minus baseline) over the samples that leave
    out one calendar year; an effect that lives in a single year collapses here."""
    if ev.n == 0 or base.n == 0:
        return float("nan")
    out = []
    for y in years:
        mv, mb = yv != y, yb != y
        if mv.any() and mb.any():
            out.append(float(ev.Rref[mv].mean() - base.Rref[mb].mean()))
    return min(out) if out else float("nan")


def _groups(family) -> list[list[str]]:
    """``family`` = one list of variant names, or a list of such lists (one per family)."""
    if not family:
        return []
    return [list(g) for g in family] if isinstance(family[0], (list, tuple)) else [list(family)]


def analyse_segment(sets: Mapping[str, EventSet], cal: Calendar, a: int, b: int,
                    family, params: Params, seed: int) -> pd.DataFrame:
    """Statistics of every set against the baseline on the sessions [a, b).

    ``family`` = the variants counted together for the multiple-testing correction: Holm over
    their two-sided bootstrap p-values, with a Bonferroni-widened interval next to it.  Give
    a list of lists for several independent families (each is corrected on its own).
    ``None`` = no correction (confirmation on the locked segment)."""
    names = [BASELINE] + [n for n in sets if n != BASELINE]
    W = int(cal.wk[b - 1] - cal.wk[a] + 1)
    cols_S, cols_C, where = [], [], {}
    for nm in names:
        ev = sets[nm]
        wk = cal.wk[ev.t0] - cal.wk[a]
        dec = _first_per_key(wk, ev.score) if ev.n else np.zeros(0, dtype=int)
        for kind, vals, idx in (("R", ev.Rref, slice(None)), ("eq", ev.eq, slice(None)),
                                ("dec", ev.Rref, dec)):
            w_, v_ = wk[idx], vals[idx]
            cols_S.append(np.bincount(w_, weights=v_, minlength=W))
            cols_C.append(np.bincount(w_, minlength=W).astype(float))
            where[(nm, kind)] = len(cols_S) - 1
    S, Cn = np.column_stack(cols_S), np.column_stack(cols_C)
    means = _boot_means(S, Cn, params.n_boot, seed)
    # robustness: blocks of several weeks (overlapping holds make neighbouring weeks alike)
    bw = max(int(params.block_weeks), 1)
    Wb = (W + bw - 1) // bw
    blk = {nm: (cal.wk[sets[nm].t0] - cal.wk[a]) // bw for nm in names}
    Sb = np.column_stack([np.bincount(blk[nm], weights=sets[nm].Rref, minlength=Wb)
                          for nm in names])
    Cb = np.column_stack([np.bincount(blk[nm], minlength=Wb).astype(float) for nm in names])
    means_b = _boot_means(Sb, Cb, params.n_boot, seed + 1000)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = S.sum(axis=0) / Cn.sum(axis=0)
    groups = _groups(family)
    size_of = {nm: len(grp) for grp in groups for nm in grp}
    base = sets[BASELINE]
    years = np.unique(cal.yr[a:b])
    rows = []
    for nm in names:
        ev = sets[nm]
        row = {"variant": nm, "n": ev.n}
        perf = _perf(ev.R)
        row.update({"keep": ev.n / base.n if base.n else np.nan, "win_rate": perf["win"],
                    "avg_R": perf["avg"], "median_R": float(np.median(ev.R)) if ev.n else np.nan,
                    "payoff": perf["payoff"], "profit_factor": perf["pf"],
                    "avg_R_ref": float(ev.Rref.mean()) if ev.n else np.nan,
                    "avg_eq_pct": float(ev.eq.mean()) if ev.n else np.nan,
                    "median_hold": float(np.median(ev.held)) if ev.n else np.nan})
        for kind, tag in (("R", "dR"), ("eq", "d_eq"), ("dec", "d_dec")):
            k, kb = where[(nm, kind)], where[(BASELINE, kind)]
            d = means[:, k] - means[:, kb]
            lo, hi = _ci(d, 1 - params.alpha)
            row[tag] = float(point[k] - point[kb]) if nm != BASELINE else 0.0
            row[f"{tag}_lo"], row[f"{tag}_hi"] = lo, hi
            if kind == "R":
                row["p_raw"] = _p_two_sided(d) if nm != BASELINE else 1.0
                row["ci_adj_lo"], row["ci_adj_hi"] = _ci(
                    d, 1 - params.alpha / max(size_of.get(nm, 1), 1))
        d_blk = means_b[:, names.index(nm)] - means_b[:, 0]
        row["dR_lo_blk"], row["dR_hi_blk"] = _ci(d_blk, 1 - params.alpha)
        row["p_raw_blk"] = _p_two_sided(d_blk) if nm != BASELINE else 1.0
        row["n_dec"] = int(Cn[:, where[(nm, "dec")]].sum())
        day_key = ev.t0
        dd = _first_per_key(day_key, ev.score) if ev.n else np.zeros(0, dtype=int)
        bd = _first_per_key(base.t0, base.score) if base.n else np.zeros(0, dtype=int)
        row["d_day"] = (float(ev.Rref[dd].mean() - base.Rref[bd].mean())
                        if len(dd) and len(bd) else np.nan)
        # sign stability per calendar year
        yv = cal.yr[ev.t0]
        yb = cal.yr[base.t0]
        used = 0
        diffs = []
        for y in years:
            mv, mb = yv == y, yb == y
            if mv.sum() >= params.min_year_events and mb.sum() >= params.min_year_events:
                used += 1
                diffs.append(float(ev.Rref[mv].mean() - base.Rref[mb].mean()))
        row["years_used"] = used
        row["year_diffs"] = diffs
        row["loyo_min"] = _leave_one_year_out(ev, base, yv, yb, years)
        wk_ev = cal.wk[ev.t0]
        n_eff, rho, m_bar = effective_n(ev.Rref, wk_ev) if ev.n else (0.0, 0.0, 1.0)
        row.update({"n_eff": n_eff, "icc": rho, "events_per_week": m_bar})
        rows.append(row)
    df = pd.DataFrame(rows).set_index("variant")
    # Holm correction over the declared family
    df["p_holm"] = np.nan
    df["p_holm_blk"] = np.nan
    for grp in groups:
        fam = [n for n in grp if n in df.index]
        if fam:
            df.loc[fam, "p_holm"] = holm_adjust(df.loc[fam, "p_raw"].to_numpy())
            df.loc[fam, "p_holm_blk"] = holm_adjust(df.loc[fam, "p_raw_blk"].to_numpy())
    return df


# ==================================================================================
# decision rule
# ==================================================================================
def axis_values(variants: Sequence[Variant], s) -> dict[str, dict[str, float]]:
    """family -> {variant name or BASELINE: axis position}."""
    fam: dict[str, dict[str, float]] = {}
    for v in variants:
        if v.role != "test":
            continue
        kind, ref = FAMILY_BASE.get(v.family, ("value", math.inf))
        if kind == "attr":
            base = float(getattr(s, str(ref)))
        elif kind == "trail":
            base = float(_trail_mult(s))
        else:
            base = float(ref)
        fam.setdefault(v.family, {BASELINE: base})[v.name] = v.x
    return fam


def plateau_check(v: Variant, table: pd.DataFrame, axis: Mapping[str, float],
                  variants: Mapping[str, Variant], params: Params) -> tuple[str, str]:
    """('ok' | 'fail' | 'untestable', note).  Neighbours are the adjacent points of the
    family axis (the baseline is a point with difference 0 and is not evidence)."""
    pts = sorted(axis.items(), key=lambda kv: kv[1])
    names = [k for k, _ in pts]
    if v.name not in names:
        return "untestable", "not on an axis"
    pos = names.index(v.name)
    pb = names.index(BASELINE)
    step = 1 if pos > pb else -1
    sign = 1.0 if table.at[v.name, "dR"] > 0 else -1.0
    gain = abs(table.at[v.name, "dR"])
    seen = 0
    for nb in (pos - 1, pos + 1):
        if nb < 0 or nb >= len(names) or names[nb] == BASELINE:
            continue
        other = names[nb]
        if other not in table.index or variants[other].role != "test":
            continue
        seen += 1
        dn = table.at[other, "dR"] * sign
        inner = (nb - pos) == -step  # lies between the baseline and the candidate
        if not np.isfinite(dn) or dn <= 0:
            return "fail", f"neighbour {other} points the other way ({table.at[other, 'dR']:+.3f})"
        if not inner and dn < params.plateau_outer_min * gain:
            return "fail", f"neighbour {other} keeps only {dn / gain:.0%} of the gain (spike)"
    if seen == 0:
        return "untestable", "no neighbouring threshold declared on that side"
    return "ok", f"{seen} neighbour(s) agree"


def select_on_design(design_sets: Mapping[str, EventSet], table: pd.DataFrame,
                     variants: Mapping[str, Variant], axis: Mapping[str, Mapping[str, float]],
                     params: Params, s, locked_start: int, n_design_years: float
                     ) -> pd.DataFrame:
    """Apply the decision rule on DESIGN data only.  Returns ``table`` with the criteria.

    It refuses to run when any event it is given belongs to the LOCKED years."""
    for nm, ev in design_sets.items():
        if ev.n and (ev.t0.max() >= locked_start or ev.exit_t.max() >= locked_start):
            raise LeakError(f"LOCKED data reached the selection ({nm})")
    out = table.copy()
    flags = {k: [] for k in ("c_sign", "c_holm", "c_years", "c_plateau", "c_keep", "c_decl",
                             "c_eq", "design_pass", "plateau", "plateau_note", "justified",
                             "years_same_sign", "year_share")}
    untested = {"plateau": "", "plateau_note": "", "years_same_sign": "", "year_share": np.nan}
    base = design_sets[BASELINE]
    hold = float(np.median(base.held)) if base.n else 30.0
    capacity = s.slots * 252.0 / max(hold, 1.0)
    for nm in out.index:
        v = variants.get(nm)
        r = out.loc[nm]
        if v is None or v.role != "test":
            for k in flags:
                flags[k].append(untested.get(k, False))
            continue
        better = v.expect == "better"
        c_sign = bool(better and r["dR"] > 0)
        c_holm = bool(r["p_holm"] <= params.alpha and r["p_holm_blk"] <= params.alpha)
        diffs = np.asarray(r["year_diffs"], dtype=float)
        n_same = int((diffs > 0).sum())
        share = n_same / len(diffs) if len(diffs) else np.nan
        c_years = bool(len(diffs) >= 3 and share >= params.year_share_min
                       and r["loyo_min"] >= params.magnitude_min * r["dR"] > 0)
        plat, note = plateau_check(v, out, axis[v.family], variants, params) if better and \
            r["dR"] > 0 else ("fail", "no gain to confirm")
        per_year = r["n"] / max(n_design_years, 1e-9)
        justified = bool(per_year >= params.capacity_mult * capacity)
        c_keep = bool(r["keep"] >= params.keep_min or justified)
        c_decl = bool(r["d_dec"] > 0)
        c_eq = bool(r["d_eq"] > 0)
        ok = c_sign and c_holm and c_years and plat == "ok" and c_keep and c_decl and c_eq
        for k, val in (("c_sign", c_sign), ("c_holm", c_holm), ("c_years", c_years),
                       ("c_plateau", plat == "ok"), ("c_keep", c_keep), ("c_decl", c_decl),
                       ("c_eq", c_eq), ("design_pass", ok), ("plateau", plat),
                       ("plateau_note", note), ("justified", justified),
                       ("years_same_sign", f"{n_same}/{len(diffs)}"), ("year_share", share)):
            flags[k].append(val)
    for k, vals in flags.items():
        out[k] = vals
    return out


# ==================================================================================
# placebo: random entries with the same exits and costs
# ==================================================================================
def placebo_summary(pool: EventSet | None, base: EventSet, cal: Calendar, draws: int,
                    seed: int, by_year: bool) -> dict:
    """Mean R of ``draws`` random samples of the pool, as many events as the baseline
    (per calendar year when ``by_year``), against the baseline mean R."""
    if pool is None or pool.n == 0 or base.n == 0:
        return {"draws": 0}
    rng = np.random.default_rng(seed)
    base_mean = float(base.R.mean())
    yrs_pool = cal.yr[pool.t0]
    tot = np.zeros(draws)
    n_tot = 0
    if by_year:
        yb = cal.yr[base.t0]
        matched = np.zeros(base.n, dtype=bool)
        for y in np.unique(yb):
            ny = int((yb == y).sum())
            pv = pool.R[yrs_pool == y]
            if len(pv) == 0:
                continue  # no random entry available that year: compare the other years
            idx = rng.integers(0, len(pv), size=(draws, ny))
            tot += pv[idx].sum(axis=1)
            n_tot += ny
            matched |= yb == y
        base_mean = float(base.R[matched].mean()) if matched.any() else base_mean
    else:
        n_tot = base.n
        for lo in range(0, draws, 250):
            hi = min(draws, lo + 250)
            idx = rng.integers(0, pool.n, size=(hi - lo, n_tot))
            tot[lo:hi] = pool.R[idx].sum(axis=1)
    means = tot / max(n_tot, 1)
    return {"draws": draws, "events": int(n_tot), "baseline_avg_R": base_mean,
            "random_avg_R_mean": float(means.mean()),
            "random_avg_R_p5": float(np.percentile(means, 5)),
            "random_avg_R_p95": float(np.percentile(means, 95)),
            "baseline_percentile": float((means < base_mean).mean() * 100),
            "p_value_one_sided": float((np.sum(means >= base_mean) + 1) / (draws + 1)),
            "matched_by_year": bool(by_year), "pool_events": int(pool.n)}


# ==================================================================================
# the study
# ==================================================================================
@dataclass
class StudyResult:
    meta: dict
    baseline: dict
    per_year: pd.DataFrame
    placebo: dict
    design: pd.DataFrame
    locked: pd.DataFrame
    selected: list[str]
    final: list[str]
    recommended: list[str]
    env: dict
    verdict: str
    verdict_fr: str
    warnings: list[str]
    events: pd.DataFrame
    variants_table: pd.DataFrame
    audit: dict
    checks: dict


def _check_variants(variants: Sequence[Variant], s) -> list[Variant]:
    names = {f.name for f in fields(s)}
    keep = []
    for v in variants:
        bad = [k for k in v.overrides if k not in names]
        if bad:
            log.warning("variant %s skipped: settings field(s) %s do not exist in this "
                        "version", v.name, bad)
            continue
        touched = PANEL_FIELDS & set(v.overrides)
        if touched:
            raise StudyError(f"variant {v.name} changes {sorted(touched)}: it needs its own "
                             "panel, which this study does not support")
        if v.expect not in ("better", "worse"):
            raise StudyError(f"variant {v.name}: expect must be 'better' or 'worse'")
        keep.append(v)
    if len({v.name for v in keep}) != len(keep) or any(v.name == BASELINE for v in keep):
        raise StudyError("variant names must be unique and not 'baseline'")
    return keep


def _exit_key(s):
    """Everything that changes how a trade is exited or costed (the EFFECTIVE trail multiple:
    trail_atr_mult = 0 means 'same as the stop')."""
    return (tuple(getattr(s, f, None) for f in EXIT_FIELDS)
            + (_trail_mult(s), getattr(s, "stop_slippage_atr", 0.0), s.fees))


def _entry_key(s):
    return tuple(getattr(s, f) for f in ENTRY_FIELDS)


def _env_value(x) -> str:
    if isinstance(x, float) and x == int(x) and abs(x) >= 1:
        return str(int(x)) if abs(x) >= 20 else f"{x:.1f}"
    return f"{x:g}" if isinstance(x, float) else str(x)


def env_for(v: Variant) -> dict[str, str]:
    return {f"INV_{k.upper()}": _env_value(val) for k, val in v.overrides.items()}


def run_study(bars: dict[str, pd.DataFrame], s, vix: pd.Series | None = None,
              vix3m: pd.Series | None = None, variants: Sequence[Variant] | None = None,
              params: Params | None = None, progress=None) -> StudyResult:
    """Run the whole walk-forward study on in-memory bars (see the module docstring)."""
    p = params or Params()
    say = progress or (lambda msg: log.info("%s", msg))
    variants = _check_variants(list(VARIANTS if variants is None else variants), s)
    skipped_controls = []
    for v in list(variants):
        if v.role == "control":
            sv = replace(s, **v.overrides)
            if (_exit_key(sv), _entry_key(sv)) != (_exit_key(s), _entry_key(s)):
                variants.remove(v)  # not an identity for this configuration
                skipped_controls.append(v.name)
    vmap = {v.name: v for v in variants}
    names = [BASELINE] + [v.name for v in variants]
    tests = [v.name for v in variants if v.role == "test"]
    adoptable = [v.name for v in variants if v.role == "test" and v.expect == "better"]
    diagnostics = [v.name for v in variants if v.role == "test" and v.expect == "worse"]
    ref_equity = s.initial_capital if p.ref_equity is None else p.ref_equity
    warnings: list[str] = []

    # ---- data, panel, regime -----------------------------------------------------
    if p.end:
        bars = {k: df.loc[:pd.Timestamp(p.end)] for k, df in bars.items()}
    universe = [u for u in build_universe(s.universe or None, s.exclude) if u in bars]
    if not universe or s.benchmark not in bars:
        raise StudyError("no usable universe or no benchmark bars")
    say(f"building the panel for {len(universe)} stocks...")
    panel = build_panel(bars, universe, s)
    cal = bars[s.benchmark].index
    g = build_grid(panel, cal)
    cald = make_calendar(cal)
    vix_l = lag_series(vix) if vix is not None else pd.Series(dtype=float)
    vix3_l = lag_series(vix3m) if vix3m is not None else pd.Series(dtype=float)
    regs = regime_list(bars, vix_l, vix3_l, s, cal)
    on, off = regime_arrays(regs)
    a = s.regime_sma + 60  # the engine's warm-up
    if p.start:
        a = max(a, int(cal.searchsorted(pd.Timestamp(p.start))))
    b = g.T
    if a >= b - 2:
        raise StudyError("not enough history after the warm-up to run a study")
    days = list(range(a, b))

    # ---- walk-forward split, fixed before any result exists ------------------------------
    years = sorted({int(y) for y in cald.yr[a:b]})
    if p.locked_from is not None:
        if not years[0] < p.locked_from <= years[-1]:
            raise StudyError(f"--locked-from must be in ({years[0]}, {years[-1]}]")
        locked_year = int(p.locked_from)
    else:
        if len(years) < 2:
            raise StudyError("need at least two calendar years of signals to split")
        locked_year = years[min(max(1, round(p.design_frac * len(years))), len(years) - 1)]
    L = max(int(np.searchsorted(cald.yr, locked_year, side="left")), a + 1)
    design_years = [y for y in years if y < locked_year]
    locked_years = [y for y in years if y >= locked_year]
    n_design_years = (L - a) / 252.0
    if len(design_years) < 3:
        warnings.append(f"only {len(design_years)} design year(s): the stability test is weak")

    # ---- signals: real baseline, mirror for the variants, audit ---------------------
    base_sig = signal_matrix(g, on, s)
    mask_days = np.zeros(g.T, dtype=bool)
    mask_days[a:b] = True
    base_sig &= mask_days[:, None]
    audit = {"baseline_cells_checked": 0, "baseline_mismatches": 0,
             "variant_cells_checked": 0, "variant_mismatches": 0, "mode": p.audit}
    if p.audit == "off":
        warnings.append("signal audit is OFF: signals come from the vectorised mirror of "
                        "signals.evaluate and were not compared with the real function")
    else:
        say("baseline signals with the bot's own evaluate_day...")
        real_base = real_signals(panel, g, regs, s, days)
        if not np.array_equal(real_base[a:b], base_sig[a:b]):
            bad = np.argwhere(real_base[a:b] != base_sig[a:b])[0]
            raise StudyError(
                f"signal mirror disagrees with signals.evaluate_day for the baseline "
                f"({g.syms[bad[1]]} on {cal[a + bad[0]].date()}): update signal_matrix()")
        audit["baseline_cells_checked"] = int((b - a) * g.N)

    svs: dict[str, np.ndarray] = {BASELINE: base_sig}
    keyed: dict[tuple, np.ndarray] = {_entry_key(s): base_sig}
    rng_a = np.random.default_rng(p.seed + 7)
    s_of: dict[str, object] = {BASELINE: s}
    for v in variants:
        sv = replace(s, **v.overrides) if v.overrides else s
        s_of[v.name] = sv
        key = _entry_key(sv)
        if key not in keyed:
            m = signal_matrix(g, on, sv) & mask_days[:, None]
            keyed[key] = m
            if p.audit != "off":
                sample = days if p.audit == "full" else _audit_days(m, on, days, rng_a, p)
                audit["variant_cells_checked"] += audit_signals(panel, g, regs, sv, m, sample,
                                                                v.name)
        svs[v.name] = keyed[key]
    say(f"signals ready: {int(base_sig.sum())} raw baseline signals")

    # ---- candidates ---------------------------------------------------------------
    union = np.zeros_like(base_sig)
    for m in svs.values():
        union |= m
    union[g.T - 1] = False  # a signal on the last session has no next open
    t0, j = np.nonzero(union)
    t0, j = t0.astype(np.int64), j.astype(np.int64)
    nc = len(t0)
    cand_of = {name: m[t0, j] for name, m in svs.items()}
    feats = candidate_features(g, t0, j)
    # information-discreteness cut-offs: design-period signals only (no later data)
    design_ids = feats["id_smooth"][cand_of[BASELINE] & (t0 < L)]
    design_ids = design_ids[np.isfinite(design_ids)]
    id_cutoffs = {v.x: float(np.quantile(design_ids, v.x)) if len(design_ids) else math.nan
                  for v in variants if v.flt == "id_smooth"}

    # ---- simulations, one per distinct set of exit settings ---------------------------
    groups: dict[tuple, dict] = {}
    for name in names:
        grp = groups.setdefault(_exit_key(s_of[name]),
                                {"s": s_of[name], "mask": np.zeros(nc, dtype=bool)})
        grp["mask"] |= cand_of[name]
    sims: dict[tuple, Sim] = {}
    for key, grp in groups.items():
        sel = np.nonzero(grp["mask"])[0]
        sm = simulate(g, t0[sel], j[sel], grp["s"], off, p.horizon, ref_equity)
        sims[key] = _scatter(sm, sel, nc)
    say(f"simulated {len(groups)} exit configuration(s) on {nc} candidate events")

    limit = g.A["close"][t0, j] * (1 + s.max_entry_gap)
    sp_base = np.array([stop_distance_pct(float(pp), float(aa), s)
                        for pp, aa in zip(limit, g.A["atr"][t0, j])])  # R_ref risk unit
    order = np.lexsort((t0, j))

    def make_set(name: str) -> EventSet:
        sm = sims[_exit_key(s_of[name])]
        member = cand_of[name] & sm.filled
        v = vmap.get(name)
        fm = filter_mask(v, feats, id_cutoffs) if v is not None else None
        if fm is not None:
            member &= fm
        if not p.allow_overlap:
            member = one_position_per_symbol(order, j, t0, sm.exit_t, member)
        ix = np.nonzero(member)[0]
        return EventSet(name=name, t0=t0[ix], j=j[ix], exit_t=sm.exit_t[ix],
                        R=sm.net_ret[ix] / sm.sp[ix], Rref=sm.net_ret[ix] / sp_base[ix],
                        eq=sm.weight[ix] * sm.net_ret[ix] * 100.0, score=feats["score"][ix],
                        held=sm.held[ix], reason=sm.reason[ix])

    sets = {name: make_set(name) for name in names}
    base = sets[BASELINE]
    min_p = 2.0 / (p.n_boot + 1) * max(len(adoptable), 1)
    if min_p > p.alpha:
        warnings.append(
            f"--boot {p.n_boot} is too small for {len(adoptable)} tests: the smallest corrected "
            f"p-value possible is {min_p:.3f} > {p.alpha:g}, so nothing can be selected "
            f"(use at least {int(2 * len(adoptable) / p.alpha) + 1} resamples)")
    checks = {f"control {n} skipped (not an identity for this configuration)": True
              for n in skipped_controls}
    for v in variants:
        if v.role == "control":
            ev = sets[v.name]
            same = (ev.n == base.n and np.array_equal(ev.t0, base.t0)
                    and np.allclose(ev.R, base.R, equal_nan=True))
            checks[f"control {v.name} reproduces the baseline exactly"] = bool(same)
            if not same:
                raise StudyError(f"control variant {v.name} does not reproduce the baseline: "
                                 "the machinery is broken")

    # ---- segments (a variant's design events must also END before the split) -------------
    design_sets = {nm: ev.sel((ev.t0 < L) & (ev.exit_t < L)) for nm, ev in sets.items()}
    locked_all = {nm: ev.sel(ev.t0 >= L) for nm, ev in sets.items()}
    purged = base.n - design_sets[BASELINE].n - locked_all[BASELINE].n
    n_design, n_locked = design_sets[BASELINE].n, locked_all[BASELINE].n
    if base.n < 30 or n_design < p.min_design_events:
        warnings.append(f"only {n_design} design events ({base.n} in total): too few for any "
                        "decision, nothing is selected")
    if n_locked < p.min_locked_events:
        warnings.append(
            f"the LOCKED segment has only {n_locked} baseline events (< {p.min_locked_events}): "
            "a confirmation there is weak, treat any adoption as provisional")

    # ---- DESIGN: statistics, then selection ---------------------------------------------
    say("design segment: paired block bootstrap...")
    table = analyse_segment(design_sets, cald, a, L, [adoptable, diagnostics], p, p.seed)
    axis = axis_values(variants, s)
    table = select_on_design(design_sets, table, vmap, axis, p, s, L, n_design_years)
    selected = [n for n in tests if bool(table.at[n, "design_pass"])]
    if n_design < p.min_design_events:
        selected = []
        table["design_pass"] = False

    # ---- LOCKED: only the selected variants, once ---------------------------------------
    locked_tbl = pd.DataFrame()
    final: list[str] = []
    if selected:
        say(f"locked segment: evaluating {len(selected)} selected variant(s) once...")
        lsets = {nm: locked_all[nm] for nm in [BASELINE] + selected}
        locked_tbl = analyse_segment(lsets, cald, L, b, None, p, p.seed + 1)
        ratios, oks = {}, {}
        for nm in selected:
            r = locked_tbl.loc[nm]
            dd = table.at[nm, "dR"]
            oks[nm] = bool(r["dR"] > 0 and r["dR"] >= p.magnitude_min * dd)
            ratios[nm] = float(r["dR"] / dd) if dd else float("nan")
            if oks[nm]:
                final.append(nm)
        locked_tbl["locked_ratio"] = pd.Series(ratios, dtype=float)
        locked_tbl["locked_pass"] = pd.Series(oks, dtype=object)

    # ---- placebo --------------------------------------------------------------------------
    say("random-entry placebo...")
    placebo = _run_placebo(g, s, on, off, mask_days, base, design_sets[BASELINE], cald, L, p,
                           ref_equity)

    # ---- baseline numbers, events table, meta ----------------------------------------------
    per_year = _per_year(base, cald)
    bperf = _perf(base.R)
    dperf = _perf(design_sets[BASELINE].R)
    eff = effective_n(base.R, cald.wk[base.t0]) if base.n else (0.0, 0.0, 1.0)
    d_eff = (effective_n(design_sets[BASELINE].R, cald.wk[design_sets[BASELINE].t0])
             if n_design else (0.0, 0.0, 1.0))
    baseline = {
        "raw_signals": int(base_sig.sum()), "events": base.n, "win_rate": bperf["win"],
        "avg_R": bperf["avg"], "median_R": float(np.median(base.R)) if base.n else np.nan,
        "payoff": bperf["payoff"], "profit_factor": bperf["pf"],
        "median_hold_sessions": float(np.median(base.held)) if base.n else np.nan,
        "avg_eq_pct": float(base.eq.mean()) if base.n else np.nan,
        "exit_mix": {k: int(v) for k, v in
                     pd.Series([_reason_group(x) for x in base.reason]).value_counts().items()},
        "design_events": n_design, "locked_events": n_locked, "purged_events": int(purged),
        "design_avg_R": dperf["avg"], "design_win_rate": dperf["win"],
        "design_n_eff": d_eff[0], "n_eff": eff[0], "icc": eff[1], "events_per_week": eff[2],
    }
    if base.n and eff[0] < 0.5 * base.n:
        warnings.append(
            f"events cluster: {base.n} baseline events are worth only about {eff[0]:.0f} "
            f"independent ones ({eff[2]:.1f} events per active week, intra-week correlation "
            f"{eff[1]:.2f})")
    events_df = _events_frame(g, t0, j, feats, sims[_exit_key(s)], cand_of[BASELINE], base, L)
    meta = {
        "universe": len(universe), "sessions": int(b - a),
        "first_day": str(cal[a].date()), "last_day": str(cal[b - 1].date()),
        "design_years": design_years, "locked_years": locked_years,
        "locked_from": locked_year, "design_frac_target": p.design_frac,
        "variants_declared": len(variants), "family_size": len(adoptable),
        "diagnostics_size": len(diagnostics),
        "correction": (f"Holm-Bonferroni over the {len(adoptable)} adoptable variants (declared "
                       f"'better'); the {len(diagnostics)} 'worse' diagnostics can never be "
                       "adopted and are corrected among themselves"),
        "n_boot": p.n_boot, "seed": p.seed, "horizon": p.horizon,
        "one_position_per_symbol": not p.allow_overlap, "ref_equity_eur": ref_equity,
        "fee_preset": s.fee_preset or s.broker, "alpha": p.alpha,
        "id_cutoffs": {f"{k:.0%}": v for k, v in id_cutoffs.items()},
        "overridden_settings": _overridden(s),
    }
    final_sorted = sorted(final, key=lambda n: -table.at[n, "dR"])
    recommended = final_sorted[:1]
    env = env_for(vmap[recommended[0]]) if recommended else {}
    verdict, verdict_fr = _verdict(meta, baseline, placebo, table, locked_tbl, selected,
                                   final_sorted, recommended, vmap, n_locked, p)
    vt = _variants_frame(variants, table, locked_tbl, selected, final, placebo)
    return StudyResult(meta=meta, baseline=baseline, per_year=per_year, placebo=placebo,
                       design=table, locked=locked_tbl, selected=selected, final=final_sorted,
                       recommended=recommended, env=env, verdict=verdict,
                       verdict_fr=verdict_fr, warnings=warnings, events=events_df,
                       variants_table=vt, audit=audit, checks=checks)


def _audit_days(m: np.ndarray, on: np.ndarray, days: list[int], rng, p: Params) -> list[int]:
    """Sessions on which a variant's signals are compared with the real evaluate_day: mostly
    sessions where the variant fires, plus random regime-on sessions."""
    hit = [i for i in days if m[i].any()]
    rest = [i for i in days if on[i]]
    pick: list[int] = []
    if hit:
        pick += list(rng.choice(hit, size=min(len(hit), p.audit_days * 2 // 3), replace=False))
    if rest:
        pick += list(rng.choice(rest, size=min(len(rest), p.audit_days // 3), replace=False))
    return sorted({int(x) for x in pick})


def _scatter(sm: Sim, sel: np.ndarray, n: int) -> Sim:
    """Place the results of a simulation of the candidates ``sel`` in arrays of length n."""
    full = Sim(filled=np.zeros(n, dtype=bool), kind=np.zeros(n, dtype=np.int8),
               fill=np.full(n, np.nan), sp=np.full(n, np.nan),
               exit_t=np.full(n, -1, dtype=np.int64), exit_px=np.full(n, np.nan),
               reason=np.full(n, "", dtype=object), held=np.zeros(n, dtype=np.int64),
               net_ret=np.full(n, np.nan), weight=np.full(n, np.nan))
    for f_ in fields(Sim):
        getattr(full, f_.name)[sel] = getattr(sm, f_.name)
    return full


def _run_placebo(g, s, on, off, mask_days, base, design_base, cald, L, p, ref_equity) -> dict:
    if p.placebo_draws <= 0:
        return {}
    pool_ok = g.eligible & on[:, None] & mask_days[:, None]
    pool_ok[g.T - 1] = False
    pt, pj = np.nonzero(pool_ok)
    if len(pt) > p.placebo_pool:
        pick = np.random.default_rng(p.seed + 3).choice(len(pt), p.placebo_pool, replace=False)
        pt, pj = pt[pick], pj[pick]
    if not len(pt):
        return {}
    psm = simulate(g, pt.astype(np.int64), pj.astype(np.int64), s, off, p.horizon, ref_equity)
    fi = np.nonzero(psm.filled)[0]
    r_own = psm.net_ret[fi] / psm.sp[fi]
    pool_all = EventSet("placebo", pt[fi], pj[fi], psm.exit_t[fi], r_own, r_own,
                        psm.weight[fi] * psm.net_ret[fi] * 100, np.zeros(len(fi)),
                        psm.held[fi], psm.reason[fi])
    pool_design = pool_all.sel((pool_all.t0 < L) & (pool_all.exit_t < L))
    return {
        "all": placebo_summary(pool_all, base, cald, p.placebo_draws, p.seed + 11, False),
        "all_by_year": placebo_summary(pool_all, base, cald, p.placebo_draws, p.seed + 12,
                                       True),
        "design": placebo_summary(pool_design, design_base, cald, p.placebo_draws,
                                  p.seed + 13, False),
        "design_by_year": placebo_summary(pool_design, design_base, cald, p.placebo_draws,
                                          p.seed + 14, True),
    }


def _reason_group(reason: str) -> str:
    return "rank exit" if reason.startswith("rank exit") else reason


def _overridden(s) -> dict:
    """Settings that differ from the code defaults (the baseline is YOUR configuration)."""
    ref = InvestSettings()
    skip = {"universe", "state_dir", "data_dir", "csv_dir", "data_provider", "history_days",
            "broker"}
    out = {}
    for f_ in fields(s):
        if f_.name in skip or f_.name.endswith(("_key", "_secret", "_agent")):
            continue
        if getattr(s, f_.name) != getattr(ref, f_.name):
            out[f_.name] = getattr(s, f_.name)
    return out


def _per_year(ev: EventSet, cal: Calendar) -> pd.DataFrame:
    if ev.n == 0:
        return pd.DataFrame(columns=["events", "avg_R", "win_rate"])
    df = pd.DataFrame({"year": cal.yr[ev.t0], "R": ev.R})
    out = df.groupby("year")["R"].agg(events="size", avg_R="mean",
                                       win_rate=lambda x: float((x > 0).mean()))
    return out


def _events_frame(g: Grid, t0, j, feats, sm: Sim, in_base, base: EventSet, L: int
                  ) -> pd.DataFrame:
    """Every raw baseline 7/7 signal with its trade (events.csv).  ``counted`` = the signal
    is a trade of the study (filled, and no earlier position of the same symbol still open)."""
    idx = np.nonzero(in_base)[0]
    counted = np.isin(t0 * g.N + j, base.t0 * g.N + base.j)
    seg = np.where(t0 < L, "design", "locked")
    seg = np.where((t0 < L) & (sm.exit_t >= L), "purged", seg)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_own = sm.net_ret / sm.sp
    df = pd.DataFrame({
        "signal_day": g.cal[t0[idx]].strftime("%Y-%m-%d"),
        "symbol": np.array(g.syms)[j[idx]],
        "segment": seg[idx], "filled": sm.filled[idx], "counted": counted[idx],
        "fill_kind": np.where(sm.kind[idx] == 1, "limit_after_gap", "open"),
        "fill_price": sm.fill[idx], "stop_pct": sm.sp[idx],
        "exit_day": [g.cal[x].strftime("%Y-%m-%d") if x >= 0 else "" for x in sm.exit_t[idx]],
        "exit_price": sm.exit_px[idx], "exit_reason": sm.reason[idx],
        "held_sessions": sm.held[idx], "R": r_own[idx],
        "net_return_pct": sm.net_ret[idx] * 100,
        "equity_contribution_pct": sm.weight[idx] * sm.net_ret[idx] * 100,
    })
    for k in ("score", "pct_rank", "rsi", "vol_ratio", "h52", "ext50", "atr_pct",
              "ext_ema20_atr", "gap_atr", "id_smooth", "event_bar_1", "event_bar_3",
              "event_bar_5"):
        df[k] = feats[k][idx]
    return df.reset_index(drop=True)


_FIRST_COLS = [
    "family", "expect", "role", "overrides", "filter", "n", "keep", "win_rate", "avg_R",
    "median_R", "avg_R_ref", "avg_eq_pct", "payoff", "profit_factor", "median_hold", "dR",
    "dR_lo", "dR_hi", "p_raw", "p_holm", "p_raw_blk", "p_holm_blk", "dR_lo_blk", "dR_hi_blk",
    "ci_adj_lo", "ci_adj_hi", "d_eq", "d_eq_lo", "d_eq_hi", "d_dec", "d_dec_lo", "d_dec_hi",
    "n_dec", "d_day", "years_used", "years_same_sign", "year_share", "year_diffs", "loyo_min",
    "plateau", "plateau_note", "n_eff", "icc", "events_per_week", "justified", "c_sign",
    "c_holm", "c_years", "c_plateau", "c_keep", "c_decl", "c_eq", "design_pass",
    "selected_on_design"]


def _variants_frame(variants, table, locked, selected, final, placebo) -> pd.DataFrame:
    """variants.csv: DESIGN statistics for every variant; LOCKED statistics only for the
    variants selected on the design years."""
    vmap = {v.name: v for v in variants}
    rows = []
    for nm in table.index:
        r = table.loc[nm].to_dict()
        r["variant"] = nm
        v = vmap.get(nm)
        r["family"] = v.family if v else ""
        r["expect"] = v.expect if v else ""
        r["role"] = v.role if v else "baseline"
        r["overrides"] = json.dumps(v.overrides) if v else ""
        r["filter"] = f"{v.flt} {v.x:g}" if v and v.flt else ""
        r["why"] = v.why if v else ""
        r["selected_on_design"] = nm in selected
        if nm in selected and nm in locked.index:
            lr = locked.loc[nm]
            r.update({"locked_n": lr["n"], "locked_dR": lr["dR"], "locked_ci_lo": lr["dR_lo"],
                      "locked_ci_hi": lr["dR_hi"], "locked_ratio": lr["locked_ratio"],
                      "locked_pass": bool(lr["locked_pass"])})
        r["final_pass"] = nm in final
        r["year_diffs"] = ";".join(f"{x:+.3f}" for x in r["year_diffs"])
        rows.append(r)
    pl = placebo.get("design_by_year") or placebo.get("design") or {}
    if pl.get("draws"):
        base_row = table.loc[BASELINE]
        rows.append({
            "variant": "random_entries (placebo)", "family": "placebo", "expect": "worse",
            "role": "placebo", "n": pl["events"], "avg_R": pl["random_avg_R_mean"],
            "dR": pl["random_avg_R_mean"] - pl["baseline_avg_R"],
            "dR_lo": pl["random_avg_R_p5"] - pl["baseline_avg_R"],
            "dR_hi": pl["random_avg_R_p95"] - pl["baseline_avg_R"],
            "p_raw": pl["p_value_one_sided"], "keep": pl["events"] / base_row["n"],
            "why": "random eligible stock-days, same count, exits and costs; dR = random "
                   "minus baseline (5%-95% of the draws); p_raw = share of draws at least as "
                   "good as the baseline",
            "selected_on_design": False, "final_pass": False})
    df = pd.DataFrame(rows).set_index("variant")
    rest = [c for c in df.columns if c not in _FIRST_COLS]
    return df[[c for c in _FIRST_COLS if c in df.columns] + rest]


# ==================================================================================
# verdict and report
# ==================================================================================
def _fmt_env(env: Mapping[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in env.items())


def _years_text(years: Sequence[int]) -> str:
    if not years:
        return "-"
    return str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"


def _verdict(meta, baseline, placebo, table, locked, selected, final, recommended, vmap,
             n_locked, p: Params) -> tuple[str, str]:
    """(English paragraph, French sentence)."""
    fam = meta["family_size"]
    dy, ly = _years_text(meta["design_years"]), _years_text(meta["locked_years"])
    pl = placebo.get("design_by_year") or placebo.get("design") or {}
    notes = []
    if pl.get("draws"):
        pct = pl["baseline_percentile"]
        notes.append(f"On the design years the 7/7 signal beat {pct:.0f}% of the random-entry "
                     "draws (same exits and costs)"
                     + ("." if pct >= 90 else
                        ": that is not clearly better than picking stocks at random, so the "
                        "entry signal itself is not proven."))
    if np.isfinite(baseline.get("design_avg_R", np.nan)) and baseline["design_avg_R"] <= 0:
        notes.append("The baseline loses money (net of costs) on the design years.")
    if n_locked < p.min_locked_events:
        notes.append("The locked segment is small, so any conclusion is provisional.")
    tail = (" " + " ".join(notes)) if notes else ""
    if baseline["design_events"] < p.min_design_events:
        text = (f"NO CONCLUSION: the design years {dy} hold only {baseline['design_events']} "
                f"events (at least {p.min_design_events} are needed), so nothing can be "
                f"selected. Keep the defaults and use more history or more stocks.{tail}")
        fr = ("PAS DE CONCLUSION : trop peu d'événements sur les années de conception. "
              "Garder les réglages par défaut.")
        return text, fr
    if recommended:
        nm = recommended[0]
        env = env_for(vmap[nm])
        how = (f"Set {_fmt_env(env)}." if env else
               f"{nm} is a filter that the engine does not have: it needs a code change, there "
               "is no INV_ setting for it.")
        text = (f"ONE CHANGE IS SUPPORTED: {nm}. {how} It passed every test on the design "
                f"years {dy} (mean R {table.at[nm, 'dR']:+.2f} above the baseline, corrected "
                f"p-value {table.at[nm, 'p_holm']:.3f}, same sign in "
                f"{table.at[nm, 'years_same_sign']} years) and it held on the locked years "
                f"{ly} ({locked.at[nm, 'dR']:+.2f} R). Change only this one parameter and "
                f"re-run the study: combinations of changes were not tested.{tail}")
        fr = (f"UN SEUL CHANGEMENT EST SOUTENU : {nm}. "
              + (f"Régler {_fmt_env(env)}. " if env else
                 "C'est un filtre absent du moteur : il faut modifier le code. ")
              + "Il passe tous les tests sur les années de conception et tient sur les années "
                "verrouillées. Ne changer que ce paramètre.")
        return text, fr
    if selected:
        text = (f"KEEP THE DEFAULTS. {len(selected)} change(s) looked good on the design years "
                f"{dy} ({', '.join(selected)}) but none held on the locked years {ly}: the "
                f"gain vanished or fell under half of its size. That is what luck, or a "
                f"change of market, looks like.{tail}")
        fr = ("GARDER LES RÉGLAGES PAR DÉFAUT. Certaines variantes semblaient meilleures sur "
              "les années de conception, mais aucune ne tient sur les années verrouillées.")
        return text, fr
    text = (f"KEEP THE DEFAULTS. {meta['variants_declared']} variants were written down before "
            f"looking at any result, {fam} of them as possible improvements. None of those "
            f"passed all the tests on the design years {dy} once we account for having tried "
            f"{fam} ideas at once, so there is no honest reason to change a parameter. No "
            f"variant was looked at on the locked years {ly}.{tail}")
    fr = ("GARDER LES RÉGLAGES PAR DÉFAUT. Aucune des variantes déclarées à l'avance ne passe "
          "tous les tests sur les années de conception : rien ne justifie de changer un "
          "paramètre.")
    return text, fr


def _f(x, nd=2, sign=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _why_not(r, alpha: float) -> str:
    """The first test a variant failed, in words (or PASS)."""
    if r["role"] == "control":
        return "control: identical to the baseline" if abs(r["dR"]) < 1e-12 else "CONTROL FAILED"
    if abs(r["dR"]) < 1e-12 and abs(r["keep"] - 1) < 1e-12:
        return "same events as the baseline (threshold not binding)"
    if r["expect"] == "worse":
        sig = bool(r["p_holm"] <= alpha)
        if r["dR"] < 0 and sig:
            return "default confirmed (looser is worse)"
        if r["dR"] > 0 and sig:
            return "looser is clearly better: not adopted (declared 'worse')"
        return "no clear difference: default not shown to help"
    if r["design_pass"]:
        return "PASS on design"
    for key, text in (("c_sign", "not higher"), ("c_holm", f"p(Holm) > {alpha:g}"),
                      ("c_years", "unstable across years (or one year carries it)"),
                      ("c_plateau", f"plateau: {r['plateau']}"), ("c_keep", "keeps too few events"),
                      ("c_decl", "fails when de-clustered"),
                      ("c_eq", "equity contribution not higher")):
        if not r[key]:
            return text
    return "-"


def render_report(res: StudyResult) -> str:
    m, b = res.meta, res.baseline
    dy, ly = _years_text(m["design_years"]), _years_text(m["locked_years"])
    out = [f"# Walk-forward study of the momentum bot, {m['first_day']} to {m['last_day']}", ""]
    out += ["## Verdict", "", res.verdict, "", f"*En français :* {res.verdict_fr}", ""]
    if res.env:
        out += ["Environment variables to set:", "", "```"] + \
            [f"{k}={v}" for k, v in res.env.items()] + ["```", ""]
    if res.warnings:
        out += ["### Warnings", ""] + [f"- {w}" for w in res.warnings] + [""]
    out += ["## How to read this", "",
            "- One event = one stock on one day when all 7 conditions are true. There are no "
            "slot or cash limits, so thousands of events are compared instead of a few "
            "dozen real trades.",
            "- R = profit divided by the money risked at entry (entry price minus first stop), "
            "after fees and slippage. +1 R means a gain equal to the initial risk.",
            "- 'dR' = the average R of a variant minus the average R of the baseline. A bigger "
            "average alone proves nothing: a variant counts only if it passes every test of "
            "the decision rule at the bottom of this page.",
            "- The 95% interval comes from resampling whole weeks (events of the same week "
            "move together). The p-value is corrected for having tried "
            f"{m['family_size']} ideas at once ({m['correction']}).", ""]
    out += ["## What was done", "",
            f"- Data: {m['universe']} stocks, {m['sessions']} sessions of signals "
            f"({m['first_day']} to {m['last_day']}).",
            f"- {b['raw_signals']} raw 7/7 signals gave {b['events']} counted events (one open "
            f"position per symbol: {m['one_position_per_symbol']}).",
            f"- DESIGN years {dy} ({b['design_events']} events), LOCKED years {ly} "
            f"({b['locked_events']} events). Variants are chosen on the design years only. "
            f"{b['purged_events']} design events whose trade ran into the locked years were "
            "removed, so that no locked price can influence the choice.",
            f"- {m['variants_declared']} variants were written down before looking: "
            f"{m['family_size']} possible improvements, {m['diagnostics_size']} looser "
            "settings declared 'worse' (they only show whether a default earns its keep and "
            "can never be adopted), and "
            f"{m['variants_declared'] - m['family_size'] - m['diagnostics_size']} self-test(s). "
            f"Bootstrap: {m['n_boot']} resamples of whole weeks, seed {m['seed']}.",
            "- For a variant with another stop distance, R is measured against the "
            "BASELINE initial risk, so that a wider stop does not look better only because "
            "its unit is bigger."]
    if m["overridden_settings"]:
        out += [f"- The baseline is YOUR configuration. Settings that differ from the code "
                f"defaults: {m['overridden_settings']}"]
    out += [""]
    out += ["## Baseline: all 7/7 events with the current settings", "",
            "| segment | events | win rate | avg R | payoff | profit factor | median hold "
            "(sessions) | effective n |", "|---|---|---|---|---|---|---|---|",
            f"| all years | {b['events']} | {_f(b['win_rate'], 3)} | {_f(b['avg_R'], 3, True)} "
            f"| {_f(b['payoff'])} | {_f(b['profit_factor'])} | "
            f"{_f(b['median_hold_sessions'], 0)} | {_f(b['n_eff'], 0)} |",
            f"| design years | {b['design_events']} | {_f(b['design_win_rate'], 3)} | "
            f"{_f(b['design_avg_R'], 3, True)} | | | | {_f(b['design_n_eff'], 0)} |", "",
            f"Effective n is smaller than the number of events when events of the same week "
            f"are correlated (intra-week correlation {_f(b['icc'], 2)}, "
            f"{_f(b['events_per_week'], 1)} events per active week).", "",
            "Average R per calendar year. The locked years are shown for information only; "
            "nothing was tuned on them:", "",
            "| year | events | avg R | win rate | part |", "|---|---|---|---|---|"]
    for y, r in res.per_year.iterrows():
        part = "locked" if y >= m["locked_from"] else "design"
        out.append(f"| {y} | {int(r['events'])} | {_f(r['avg_R'], 3, True)} | "
                   f"{_f(r['win_rate'], 2)} | {part} |")
    out += ["", "Exits of the baseline events: " + ", ".join(
        f"{k} {v}" for k, v in b["exit_mix"].items()), ""]
    out += ["## Random-entry placebo", ""]
    for key, label in (("design_by_year", "design years, same number of entries per year"),
                       ("design", "design years"),
                       ("all_by_year", "all years, same number of entries per year"),
                       ("all", "all years")):
        pl = res.placebo.get(key)
        if pl and pl.get("draws"):
            out.append(f"- {label}: baseline average R {_f(pl['baseline_avg_R'], 3, True)}, "
                       f"random entries {_f(pl['random_avg_R_mean'], 3, True)} (5% to 95% of "
                       f"the draws: {_f(pl['random_avg_R_p5'], 3, True)} to "
                       f"{_f(pl['random_avg_R_p95'], 3, True)}). The baseline beats "
                       f"{pl['baseline_percentile']:.0f}% of {pl['draws']} random draws.")
    out += ["", "Random entries are drawn among eligible stocks while the regime is on, with "
            "the same exits and costs. Around 50% would mean no better than chance.", ""]
    out += ["## Variants on the design years", "",
            "'keep' = share of the baseline events that remain. 'years' = design years where "
            "the difference has the right sign. 'decl.' = the difference when only the best "
            "scored event of each week is kept.", "",
            "| variant | hypothesis | n | keep | avg R | dR | 95% interval | p (Holm) | "
            "p (Holm, 4-week blocks) | years | decl. | result |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    tb = res.variants_table
    for nm, r in tb.iterrows():
        if r["role"] not in ("test", "control"):
            continue
        tested = r["role"] == "test"
        out.append(
            f"| {nm} | {r['expect']} | {int(r['n'])} | {_f(r['keep'], 2)} | "
            f"{_f(r['avg_R'], 3, True)} | {_f(r['dR'], 3, True)} | "
            f"[{_f(r['dR_lo'], 3, True)}, {_f(r['dR_hi'], 3, True)}] | "
            f"{_f(r['p_holm'], 3) if tested else '-'} | "
            f"{_f(r['p_holm_blk'], 3) if tested else '-'} | "
            f"{r['years_same_sign'] if tested else '-'} | {_f(r['d_dec'], 3, True)} | "
            f"{_why_not(r, m['alpha'])} |")
    out += ["", "## Locked years: selected variants only", ""]
    if len(res.locked):
        out += ["| variant | design dR | locked events | locked dR | locked 95% interval | "
                "locked / design | held? |", "|---|---|---|---|---|---|---|"]
        for nm in res.selected:
            r = res.locked.loc[nm]
            out.append(f"| {nm} | {_f(res.design.at[nm, 'dR'], 3, True)} | {int(r['n'])} | "
                       f"{_f(r['dR'], 3, True)} | [{_f(r['dR_lo'], 3, True)}, "
                       f"{_f(r['dR_hi'], 3, True)}] | {_f(r['locked_ratio'], 2)} | "
                       f"{'yes' if r['locked_pass'] else 'no'} |")
        out += ["", "A variant holds if its locked difference has the same sign and is at "
                "least half of the design difference."]
    else:
        out.append("No variant passed on the design years, so no variant was evaluated on "
                   "the locked years.")
    out += ["", "## Honesty checks", "", f"- Signal audit: {m_audit(res.audit)}"]
    out += [f"- {k}: {'ok' if v else 'FAILED'}" for k, v in res.checks.items()]
    out += ["", "## Decision rule, fixed in advance", "", "```", DECISION_RULE, "```", "",
            "## Limits", "",
            "- No slot, cash, theme or drawdown limits: this measures the signal and its "
            "exits, not the portfolio. The bot takes only the best-scored few events.",
            "- One position per symbol; no pyramiding; no earnings, news or insider data; "
            "constant EUR/USD; costs computed for a reference position size.",
            f"- A trade is closed after {m['horizon']} sessions at most (the bot has no such "
            "cap), which cuts the biggest winners the same way for every variant.",
            "- Today's universe (survivorship bias) and split-adjusted prices without "
            "dividends.",
            "- Combinations of changes were not tested; adopt at most one change and run the "
            "study again.", ""]
    return "\n".join(out)


def m_audit(a: dict) -> str:
    if a.get("mode") == "off":
        return "OFF (signals come from the vectorised mirror only)."
    return (f"{a['baseline_cells_checked']} baseline cells and {a['variant_cells_checked']} "
            f"variant cells compared with the bot's evaluate_day, "
            f"{a['baseline_mismatches'] + a['variant_mismatches']} mismatch(es).")


def _clean(o):
    """Plain JSON types; NaN and infinity become null (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    return o


def write_outputs(res: StudyResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(render_report(res), encoding="utf-8")
    res.events.to_csv(out / "events.csv", index=False)
    res.variants_table.to_csv(out / "variants.csv")
    summary = {
        "verdict": res.verdict, "verdict_fr": res.verdict_fr, "recommended": res.recommended,
        "env": res.env, "selected_on_design": res.selected, "passed_locked": res.final,
        "meta": res.meta, "baseline": res.baseline, "placebo": res.placebo,
        "per_year": json.loads(res.per_year.to_json(orient="index")),
        "warnings": res.warnings, "audit": res.audit, "checks": res.checks,
        "decision_rule": DECISION_RULE,
        "variants": json.loads(res.variants_table.to_json(orient="index")),
    }
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=2), encoding="utf-8")


# ==================================================================================
# command line
# ==================================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Signal-level walk-forward study: which parameters of the bot are robust?")
    ap.add_argument("--csv-dir", help="offline: <SYMBOL>.csv files (date,open,high,low,close,"
                                      "volume)")
    ap.add_argument("--provider", choices=["massive", "alpaca", "tiingo"],
                    help="download bars with this provider (keys from .env)")
    ap.add_argument("--days", type=int, default=3000, help="calendar days of history to fetch")
    ap.add_argument("--start", help="first signal day (YYYY-MM-DD)")
    ap.add_argument("--end", help="last day of data used (YYYY-MM-DD)")
    ap.add_argument("--no-vix", action="store_true", help="do not fetch VIX from FRED")
    ap.add_argument("--locked-from", type=int, metavar="YYYY",
                    help="first calendar year of the LOCKED segment (default: last ~40%%)")
    ap.add_argument("--design-frac", type=float, default=0.60,
                    help="share of the calendar years used as DESIGN (default 0.60); the rest "
                         "is LOCKED")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap draws")
    ap.add_argument("--placebo-draws", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=250, help="max sessions a trade is held")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--allow-overlap", action="store_true",
                    help="count every raw signal even while the symbol is already held")
    ap.add_argument("--audit", choices=["sample", "full", "off"], default="sample",
                    help="check the vectorised signals against evaluate_day")
    ap.add_argument("--ref-equity", type=float, help="EUR used to size the cost model "
                                                       "(default: initial capital)")
    ap.add_argument("--out", default="study_invest")
    ap.add_argument("--env-file", default=".env")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides: dict = {"broker": "sim"}
    if args.provider:
        overrides["data_provider"] = args.provider
        overrides["history_days"] = args.days
    s = InvestSettings.from_env(args.env_file, **overrides)
    universe = build_universe(s.universe or None, s.exclude)
    symbols = data_symbols(universe, s.benchmark, None)
    if args.csv_dir:
        bars = load_csv_dir(args.csv_dir, symbols)
    elif args.provider:
        store = BarStore(s.data_dir, make_provider(s), args.days)
        bars = store.update(symbols, date.today())
    else:
        ap.error("give --csv-dir or --provider")
    bars = {k: v for k, v in bars.items() if v is not None and not v.empty}
    if s.benchmark not in bars:
        raise SystemExit(f"{s.benchmark} bars are required (market regime)")
    first = min(df.index[0] for df in bars.values()).date()
    vix = vix3m = None
    if not args.no_vix:
        try:
            vix = fred_series("VIXCLS", first)
            vix3m = fred_series("VXVCLS", first)
        except Exception as exc:
            log.warning("macro series unavailable (%s): regime uses SPY only", exc)
    params = Params(n_boot=args.boot, seed=args.seed, horizon=args.horizon,
                    design_frac=args.design_frac,
                    locked_from=args.locked_from, placebo_draws=args.placebo_draws,
                    allow_overlap=args.allow_overlap, audit=args.audit,
                    ref_equity=args.ref_equity, start=args.start, end=args.end)
    try:
        res = run_study(bars, s, vix=vix, vix3m=vix3m, params=params)
    except StudyError as exc:
        print(f"STUDY STOPPED: {exc}", file=sys.stderr)
        return 2
    write_outputs(res, args.out)
    print_console(res, args.out)
    return 0


def print_console(res: StudyResult, out_dir) -> None:
    b = res.baseline
    print("\n=== Signal-level walk-forward study ===")
    print(f"  period            {res.meta['first_day']} .. {res.meta['last_day']}  "
          f"({res.meta['universe']} stocks)")
    print(f"  design / locked   {res.meta['design_years']} / {res.meta['locked_years']}")
    print(f"  baseline events   {b['events']}  (raw signals {b['raw_signals']}, "
          f"effective n ~{b['n_eff']:.0f})")
    print(f"  win rate          {_f(b['win_rate'], 3)}")
    print(f"  avg R (net)       {_f(b['avg_R'], 3, True)}")
    print(f"  payoff            {_f(b['payoff'])}")
    print(f"  profit factor     {_f(b['profit_factor'])}")
    yearly = ", ".join(f"{y}: {_f(r['avg_R'], 2, True)} (n={int(r['events'])})"
                       for y, r in res.per_year.iterrows())
    print(f"  avg R per year    {yearly}")
    for key in ("design_by_year", "all_by_year"):
        pl = res.placebo.get(key)
        if pl and pl.get("draws"):
            print(f"  placebo ({key}): baseline {_f(pl['baseline_avg_R'], 3, True)} vs random "
                  f"{_f(pl['random_avg_R_mean'], 3, True)}; beats "
                  f"{pl['baseline_percentile']:.0f}% of {pl['draws']} draws")
    for w in res.warnings:
        print(f"  WARNING: {w}")
    print(f"\nVERDICT: {res.verdict}")
    print(f"\n(FR) {res.verdict_fr}")
    if res.env:
        print("\n  Environment variables: " + _fmt_env(res.env))
    print(f"\nReport, events, variants and summary in {out_dir}/")
    print("A signal-level study is not a forecast: see the limits in report.md.")


if __name__ == "__main__":
    sys.exit(main())
