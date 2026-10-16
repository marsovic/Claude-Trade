"""Backtest of the investment bot with the exact live engine.

The engine runs day by day on historical bars; orders fill on the next bar
through the simulator (open fills, limit caps, stops on the low, gaps at the
open, Trading 212-like costs). News, Reddit, earnings and insider data have
no free history and are off; VIX can be replayed from FRED.

Caveats: survivorship bias (today's universe), no dividends (split-adjusted
prices), EUR/USD constant unless an ECB series is given.

    python -m invest.backtest --csv-dir data/us_daily            # offline
    python -m invest.backtest --provider massive --days 700      # downloads
    python -m invest.backtest --provider massive --days 700 --placebo 20
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bot.state import StateStore
from invest.config import InvestSettings
from invest.data import (
    BarStore,
    MarketData,
    ecb_eurusd,
    fred_series,
    make_provider,
    read_bars_csv,
)
from invest.engine import Engine
from invest.execution import SimExecution
from invest.portfolio import InvestState
from invest.signals import build_panel
from invest.tax import TaxLedger
from invest.universe import build_universe, data_symbols

log = logging.getLogger("invest.backtest")


@dataclass
class InvestBacktest:
    state: InvestState
    trades: list[dict]
    curve: pd.Series  # EUR equity by close
    benchmark: pd.Series  # SPY buy & hold, rebased to the initial capital
    days: int
    equal_weight: pd.Series | None = None  # whole universe, equal weights, buy & hold

    def stats(self) -> dict:
        s, curve = self.state, self.curve
        pnls = [t["pnl"] for t in self.trades]
        wins = [p for p in pnls if p > 0]
        losses = [-p for p in pnls if p <= 0]
        years = max(self.days / 252, 1e-9)
        final = float(curve.iloc[-1]) if len(curve) else s.initial_capital
        peak = curve.cummax()
        dd = float(((peak - curve) / peak).max()) if len(curve) else 0.0
        rets = curve.pct_change().dropna()
        sharpe = float(rets.mean() / rets.std() * math.sqrt(252)) if rets.std() > 0 else 0.0
        bench_final = float(self.benchmark.iloc[-1]) if len(self.benchmark) else float("nan")
        ew = self.equal_weight
        ew_final = float(ew.iloc[-1]) if ew is not None and len(ew) else float("nan")
        # Each year runs from the previous year's last close (the start for year one).
        yearly = {}
        base = s.initial_capital
        for year, part in curve.groupby(curve.index.year):
            yearly[int(year)] = round((float(part.iloc[-1]) / base - 1) * 100, 1)
            base = float(part.iloc[-1])
        return {
            "initial_eur": s.initial_capital,
            "final_eur": round(final, 2),
            "return_pct": round((final / s.initial_capital - 1) * 100, 2),
            "cagr_pct": round(((final / s.initial_capital) ** (1 / years) - 1) * 100, 2)
            if final > 0 else -100.0,
            "max_drawdown_pct": round(dd * 100, 2),
            "sharpe": round(sharpe, 2),
            "trades": len(pnls),
            "win_rate_pct": round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0,
            "profit_factor": round(sum(wins) / sum(losses), 2) if sum(losses) > 0 else None,
            "avg_r": round(float(np.mean([t["r_multiple"] for t in self.trades])), 2)
            if self.trades else 0.0,
            "fees_eur": round(s.fees_eur, 2),
            "trading_days": self.days,
            "died": s.status == "dead",
            "benchmark_spy_final_eur": round(bench_final, 2),
            "benchmark_equal_weight_universe_final_eur": round(ew_final, 2),
            "yearly_return_pct": yearly,
        }


def _equal_weight(bars, universe, data, curve: pd.Series, s) -> pd.Series:
    """Buy & hold of every stock of the universe at equal weights, in EUR."""
    if curve.empty:
        return pd.Series(dtype=float)
    cols = {sym: bars[sym]["close"].reindex(curve.index).ffill() for sym in universe
            if sym in bars}
    closes = pd.DataFrame(cols).dropna(axis=1, how="any")
    if closes.empty:
        return pd.Series(dtype=float)
    fx = pd.Series([data.fx(d) for d in curve.index], index=curve.index)
    rebased = (closes / closes.iloc[0]).mean(axis=1) / (fx / fx.iloc[0])
    return rebased * s.initial_capital


def run_placebo(bars: dict[str, pd.DataFrame], s: InvestSettings, n_entries: int,
                runs: int = 20, seed: int = 0, **kw) -> dict:
    """Same exits, sizing, costs and regime filter, but RANDOM entries.

    Entries are drawn among the eligible stocks, with about the same frequency as the
    real run (``n_entries``). The question it answers: do the 7 conditions beat picking
    stocks at random, on this very data? Win rate, exit mix and profit factor below 1
    are what noise produces too; only the comparison tells.
    """
    universe = [u for u in build_universe(s.universe or None, s.exclude) if u in bars]
    panel = build_panel(bars, universe, s)
    spy = bars[s.benchmark]
    n_days = max(len(spy.index[s.regime_sma + 60:]), 1)
    p_entry = min(1.0, n_entries / n_days)
    results = []
    for i in range(runs):
        rng = np.random.default_rng(seed + i)

        def pick(day, evals, rng=rng):
            pool = [e for e in evals.values() if e.eligible]
            if not pool or rng.random() > p_entry:
                return []
            return [pool[int(rng.integers(len(pool)))]]

        bt = run_backtest(bars, s, panel=panel, picker=pick, **kw)
        st = bt.stats()
        results.append({"return_pct": st["return_pct"], "avg_r": st["avg_r"],
                        "trades": st["trades"], "profit_factor": st["profit_factor"]})
    return {"runs": runs, "entry_probability_per_day": round(p_entry, 3), "results": results}


def placebo_summary(real: dict, placebo: dict) -> dict:
    rets = np.array([r["return_pct"] for r in placebo["results"]], dtype=float)
    avg_r = np.array([r["avg_r"] for r in placebo["results"]], dtype=float)
    return {
        "runs": placebo["runs"],
        "random_return_pct": {"median": round(float(np.median(rets)), 1),
                              "p10": round(float(np.percentile(rets, 10)), 1),
                              "p90": round(float(np.percentile(rets, 90)), 1)},
        "random_avg_r_median": round(float(np.median(avg_r)), 2),
        "real_return_pct": real["return_pct"],
        "real_avg_r": real["avg_r"],
        # share of random runs the real strategy beat: ~50 = no better than chance
        "real_beats_pct_of_random_runs": round(float((rets < real["return_pct"]).mean() * 100),
                                               0),
    }


def _lag(series: pd.Series | None) -> pd.Series | None:
    """Value known at a session's close: the previous observation."""
    if series is None or series.empty:
        return series
    return series.shift(1).dropna()


def run_backtest(bars: dict[str, pd.DataFrame], s: InvestSettings, start=None, end=None,
                 vix: pd.Series | None = None, vix3m: pd.Series | None = None,
                 eurusd: pd.Series | None = None, out_dir: str | Path | None = None,
                 panel: dict | None = None, picker=None) -> InvestBacktest:
    """``panel`` lets repeated runs share the (slow) feature computation; ``picker`` replaces
    the signals by another entry rule (see :func:`run_placebo`)."""
    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="invest_bt_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("state.json", "decisions.jsonl", "trades.jsonl", "tax_ledger.jsonl",
                 "TOMBSTONE.json"):
        (out_dir / name).unlink(missing_ok=True)  # a re-run must not append to old results
    store = StateStore(out_dir, InvestState)
    state = InvestState.fresh("sim", s.initial_capital)
    universe = [u for u in build_universe(s.universe or None, s.exclude) if u in bars]
    state.universe = universe
    execution = SimExecution(s, state, s.fees)
    engine = Engine(s, state, store, execution, research=None, ledger=TaxLedger(out_dir))
    # FRED publishes a day's VIX close the next morning: live, the decision at the
    # close uses the previous session's value, so the backtest does the same.
    vix, vix3m = (_lag(x) for x in (vix, vix3m))
    data = MarketData(bars=dict(bars), vix=vix if vix is not None else pd.Series(dtype=float),
                      vix3m=vix3m if vix3m is not None else pd.Series(dtype=float),
                      eurusd=eurusd if eurusd is not None else pd.Series(dtype=float),
                      eurusd_fallback=s.eurusd_fallback)
    engine.picker = picker
    panel = panel if panel is not None else build_panel(data.bars, universe, s)
    spy = data.bars[s.benchmark]
    days = spy.index[s.regime_sma + 60:]  # warm-up for SMA200 / 12-month momentum
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    curve = {}
    for day in days:
        engine.process_day(day, data, panel=panel)
        curve[day] = state.equity
        if state.status == "dead":
            break
    curve_s = pd.Series(curve, dtype=float)
    # Buy & hold SPY for a EUR investor: USD prices converted at each day's rate.
    bench = pd.Series([spy.loc[d, "close"] / data.fx(d) for d in curve_s.index],
                      index=curve_s.index, dtype=float)
    bench = bench / bench.iloc[0] * s.initial_capital if len(bench) else bench
    trades = store.recent_trades(100_000)
    return InvestBacktest(state=state, trades=trades, curve=curve_s, benchmark=bench,
                          days=len(curve_s), equal_weight=_equal_weight(bars, universe, data,
                                                                        curve_s, s))


def load_csv_dir(directory: str, symbols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for sym in symbols:
        path = Path(directory) / f"{sym}.csv"
        if path.exists():
            out[sym] = read_bars_csv(path)
    if not out:
        raise SystemExit(f"no <SYMBOL>.csv file found in {directory}")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Backtest the aggressive momentum invest bot")
    p.add_argument("--csv-dir",
                   help="offline: <SYMBOL>.csv files (date,open,high,low,close,volume)")
    p.add_argument("--provider", choices=["massive", "alpaca", "tiingo"],
                   help="download bars with this provider (keys from .env)")
    p.add_argument("--days", type=int, default=700, help="calendar days of history to fetch")
    p.add_argument("--start", help="first trading day to simulate (YYYY-MM-DD)")
    p.add_argument("--no-vix", action="store_true", help="do not fetch VIX from FRED")
    p.add_argument("--etp", action="store_true",
                   help="enable the simulated 3x ETP sleeve (off by default)")
    p.add_argument("--no-etp", action="store_true", help="(kept for compatibility: ETP is off)")
    p.add_argument("--placebo", type=int, default=0, metavar="N",
                   help="also run N random-entry placebo backtests (same exits, sizing, costs) "
                        "and say how the real result ranks among them")
    p.add_argument("--out", default="backtest_invest")
    p.add_argument("--env-file", default=".env")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = {"broker": "sim"}
    if args.etp:
        overrides["etp_enabled"] = True
    if args.no_etp:
        overrides["etp_enabled"] = False
    if args.provider:
        overrides["data_provider"] = args.provider
        overrides["history_days"] = args.days
    s = InvestSettings.from_env(args.env_file, **overrides)
    universe = build_universe(s.universe or None, s.exclude)
    symbols = data_symbols(universe, s.benchmark, s.etp_underlying if s.etp_enabled else None)
    if args.csv_dir:
        bars = load_csv_dir(args.csv_dir, symbols)
    elif args.provider:
        store = BarStore(s.data_dir, make_provider(s), args.days)
        bars = store.update(symbols, date.today())
    else:
        p.error("give --csv-dir or --provider")
    bars = {k: v for k, v in bars.items() if v is not None and not v.empty}
    if s.benchmark not in bars:
        raise SystemExit(f"{s.benchmark} bars are required (market regime)")
    first = min(df.index[0] for df in bars.values()).date()
    vix = vix3m = eurusd = None
    if not args.no_vix:
        try:
            vix = fred_series("VIXCLS", first)
            vix3m = fred_series("VXVCLS", first)
            eurusd = ecb_eurusd(first)
        except Exception as exc:
            log.warning("macro series unavailable (%s): regime uses SPY only", exc)
    result = run_backtest(bars, s, start=args.start, vix=vix, vix3m=vix3m, eurusd=eurusd,
                          out_dir=args.out)
    stats = result.stats()
    placebo = None
    if args.placebo > 0:
        log.info("placebo: %d random-entry backtests (a few minutes)...", args.placebo)
        raw = run_placebo(bars, s, n_entries=max(stats["trades"], 1), runs=args.placebo,
                          start=args.start, vix=vix, vix3m=vix3m, eurusd=eurusd,
                          out_dir=Path(args.out) / "placebo")
        placebo = placebo_summary(stats, raw)
        stats["placebo"] = placebo
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "summary.json").write_text(json.dumps(stats, indent=2))
    result.curve.rename("equity_eur").to_csv(Path(args.out) / "equity.csv")
    width = max(len(k) for k in stats)
    print(f"\n=== Invest backtest: {len(result.state.universe)} stocks, "
          f"{result.days} trading days ===")
    for key, value in stats.items():
        print(f"  {key:<{width}}  {value}")
    if placebo:
        print(f"\nPlacebo ({placebo['runs']} random-entry runs): real {placebo['real_return_pct']}%"
              f" vs random median {placebo['random_return_pct']['median']}% (p10 "
              f"{placebo['random_return_pct']['p10']}, p90 {placebo['random_return_pct']['p90']});"
              f" the real strategy beat {placebo['real_beats_pct_of_random_runs']:.0f}% of them.")
        print("Around 50% = no better than picking at random; well above 80-90% on many trades"
              " is what a real edge looks like.")
    print(f"\nJournal, trades and tax ledger in {args.out}/")
    print("Past results, survivorship bias and simulated fills: this is not a forecast.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
