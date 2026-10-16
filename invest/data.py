"""Daily market data: bar providers, local cache, VIX and EUR/USD.

Official APIs only (no Yahoo scraping):

* Massive (ex-Polygon)  free: 5 calls/min, 2 years of EOD history. The
  grouped-daily endpoint returns the whole US market for one day in ONE call.
* Alpaca market data    free with any Alpaca account (paper included).
* Tiingo                free: 50 calls/hour, 1000/day, adjusted EOD bars.
* CSV                   offline files, for backtests and tests.
* FRED                  VIX (VIXCLS) and VIX3M (VXVCLS), keyless CSV export.
* ECB                   EUR/USD reference rate, keyless.

Bars are cached per symbol in ``data_dir/bars/<SYMBOL>.csv`` and only the
missing days are downloaded afterwards.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from invest.markets import next_trading_day, split_ratio

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")
COLUMNS = ["open", "high", "low", "close", "volume"]


class DataError(RuntimeError):
    pass


def empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name="date"), dtype=float)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Sorted, de-duplicated, float OHLCV indexed by naive midnight dates."""
    if df.empty:
        return empty_frame()
    out = df[COLUMNS].astype(float)
    out.index = pd.DatetimeIndex(pd.to_datetime(out.index)).normalize()
    out.index.name = "date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["open", "high", "low", "close"])


class RateLimiter:
    def __init__(self, per_minute: float, sleep=time.sleep, clock=time.monotonic):
        self.interval = 60.0 / per_minute if per_minute > 0 else 0.0
        self.sleep, self.clock = sleep, clock
        self.last = -1e18

    def wait(self) -> None:
        delay = self.last + self.interval - self.clock()
        if delay > 0:
            self.sleep(delay)
        self.last = self.clock()


# =================================================================== providers
class BarProvider:
    name = "base"
    supports_grouped = False
    batch_size = 1  # symbols per history_many() call (the cache is saved after each batch)

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        raise NotImplementedError

    def history_many(self, symbols: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
        return {s: self.history(s, start, end) for s in symbols}

    def grouped(self, day: date) -> dict[str, dict] | None:
        return None


class MassiveProvider(BarProvider):
    """https://api.massive.com (Polygon renamed on 2025-10-30)."""

    name = "massive"
    supports_grouped = True
    BASE = "https://api.massive.com"

    def __init__(self, api_key: str, session: requests.Session | None = None,
                 per_minute: float = 5, sleep=time.sleep):
        if not api_key:
            raise DataError("MASSIVE_API_KEY is required (free key at massive.com)")
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.limiter = RateLimiter(per_minute, sleep=sleep)
        self.sleep = sleep

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(4):
            self.limiter.wait()
            resp = self.session.get(self.BASE + path, params=params, timeout=30)
            text = resp.text[:300]
            if resp.status_code == 429 or "maximum requests" in text.lower():
                self.sleep(15 * (attempt + 1))
                continue
            if resp.status_code == 403:
                raise DataError(f"Massive: endpoint not in your plan ({text})")
            if resp.status_code >= 400:
                raise DataError(f"Massive HTTP {resp.status_code}: {text}")
            return resp.json()
        raise DataError("Massive: rate limited too many times")

    @staticmethod
    def _rows_to_frame(rows: list[dict], key_ts: str = "t") -> pd.DataFrame:
        if not rows:
            return empty_frame()
        idx = [datetime.fromtimestamp(r[key_ts] / 1000, tz=ET).date() for r in rows]
        df = pd.DataFrame({"open": [r["o"] for r in rows], "high": [r["h"] for r in rows],
                           "low": [r["l"] for r in rows], "close": [r["c"] for r in rows],
                           "volume": [r.get("v", 0.0) for r in rows]},
                          index=pd.to_datetime(idx))
        return normalise(df)

    def history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        body = self._get(f"/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}",
                         {"adjusted": "true", "sort": "asc", "limit": 50000})
        return self._rows_to_frame(body.get("results") or [])

    def grouped(self, day: date) -> dict[str, dict] | None:
        body = self._get(f"/v2/aggs/grouped/locale/us/market/stocks/{day}", {"adjusted": "true"})
        rows = body.get("results") or []
        return {r["T"]: {"open": r["o"], "high": r["h"], "low": r["l"], "close": r["c"],
                         "volume": r.get("v", 0.0)} for r in rows if "T" in r}


class AlpacaDataProvider(BarProvider):
    """https://data.alpaca.markets - free with any Alpaca account (paper too)."""

    name = "alpaca"
    batch_size = 50
    BASE = "https://data.alpaca.markets/v2/stocks/bars"

    def __init__(self, api_key: str, api_secret: str, session: requests.Session | None = None,
                 feed: str = "sip", per_minute: float = 180, sleep=time.sleep):
        if not (api_key and api_secret):
            raise DataError("Alpaca data needs ALPACA_API_KEY and ALPACA_API_SECRET")
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": api_key,
                                     "APCA-API-SECRET-KEY": api_secret})
        self.feed = feed
        self.limiter = RateLimiter(per_minute, sleep=sleep)

    def history_many(self, symbols, start, end):
        out: dict[str, list] = {s: [] for s in symbols}
        for i in range(0, len(symbols), 50):
            chunk = symbols[i:i + 50]
            token = None
            while True:
                self.limiter.wait()
                # The free plan serves SIP data older than 15 minutes only.
                latest = datetime.now(timezone.utc) - timedelta(minutes=16)
                stop = min(datetime.combine(end, dtime(23, 59, 59), tzinfo=timezone.utc),
                           latest)
                params = {"symbols": ",".join(chunk), "timeframe": "1Day",
                          "start": f"{start}T00:00:00Z",
                          "end": stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
                          "adjustment": "split", "feed": self.feed, "limit": 10000}
                if token:
                    params["page_token"] = token
                resp = self.session.get(self.BASE, params=params, timeout=30)
                if resp.status_code == 403 and self.feed == "sip":
                    # IEX alone is ~2 % of US volume: its bars would distort the
                    # volume and breakout rules, so never switch silently.
                    raise DataError("Alpaca refused the SIP feed (403). Use another "
                                    "INV_DATA_PROVIDER (massive, tiingo) or a plan with SIP.")
                if resp.status_code >= 400:
                    raise DataError(f"Alpaca data HTTP {resp.status_code}: {resp.text[:200]}")
                body = resp.json()
                for sym, bars in (body.get("bars") or {}).items():
                    out.setdefault(sym, []).extend(bars)
                token = body.get("next_page_token")
                if not token:
                    break
        frames = {}
        for sym, bars in out.items():
            if not bars:
                frames[sym] = empty_frame()
                continue
            idx = [pd.Timestamp(b["t"]).tz_convert(ET).date() for b in bars]
            frames[sym] = normalise(pd.DataFrame(
                {"open": [b["o"] for b in bars], "high": [b["h"] for b in bars],
                 "low": [b["l"] for b in bars], "close": [b["c"] for b in bars],
                 "volume": [b.get("v", 0) for b in bars]}, index=pd.to_datetime(idx)))
        return frames

    def history(self, symbol, start, end):
        return self.history_many([symbol], start, end)[symbol]


class TiingoProvider(BarProvider):
    name = "tiingo"
    BASE = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key: str, session: requests.Session | None = None,
                 per_minute: float = 0.8, sleep=time.sleep):
        if not api_key:
            raise DataError("TIINGO_API_KEY is required")
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Token {api_key}"})
        self.limiter = RateLimiter(per_minute, sleep=sleep)  # 50/hour on the free plan

    def history(self, symbol, start, end):
        self.limiter.wait()
        resp = self.session.get(f"{self.BASE}/{symbol.replace('.', '-')}/prices",
                                params={"startDate": str(start), "endDate": str(end)},
                                timeout=30)
        if resp.status_code >= 400:
            raise DataError(f"Tiingo HTTP {resp.status_code}: {resp.text[:200]}")
        rows = resp.json()
        if not rows:
            return empty_frame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["date"].str[:10])
        df = df.rename(columns={"adjOpen": "o", "adjHigh": "h", "adjLow": "l",
                                "adjClose": "c", "adjVolume": "v"})
        return normalise(pd.DataFrame({"open": df["o"], "high": df["h"], "low": df["l"],
                                       "close": df["c"], "volume": df["v"]}))


class CsvProvider(BarProvider):
    """``<dir>/<SYMBOL>.csv`` with a date (or ms open_time) column and OHLCV."""

    name = "csv"

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)

    def history(self, symbol, start, end):
        path = self.dir / f"{symbol}.csv"
        if not path.exists():
            return empty_frame()
        return read_bars_csv(path).loc[pd.Timestamp(start):pd.Timestamp(end)]


def read_bars_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        idx = pd.to_datetime(df["date"])
    elif "open_time" in df.columns:
        idx = pd.to_datetime(df["open_time"], unit="ms")
    else:
        raise DataError(f"{path}: needs a 'date' or 'open_time' column")
    df.index = idx
    return normalise(df)


def make_provider(s) -> BarProvider:
    if s.data_provider == "massive":
        return MassiveProvider(s.massive_api_key)
    if s.data_provider == "alpaca":
        return AlpacaDataProvider(s.alpaca_api_key, s.alpaca_api_secret)
    if s.data_provider == "tiingo":
        return TiingoProvider(s.tiingo_api_key)
    return CsvProvider(s.csv_dir)


# ===================================================================== cache
class BarStore:
    """Per-symbol CSV cache, refreshed incrementally."""

    def __init__(self, data_dir: str | Path, provider: BarProvider, history_days: int = 460):
        self.dir = Path(data_dir) / "bars"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.provider = provider
        self.history_days = history_days

    def path(self, symbol: str) -> Path:
        return self.dir / f"{symbol}.csv"

    def load(self, symbol: str) -> pd.DataFrame:
        path = self.path(symbol)
        if not path.exists():
            return empty_frame()
        return read_bars_csv(path)

    def save(self, symbol: str, df: pd.DataFrame) -> None:
        out = df.copy()
        out.insert(0, "date", out.index.strftime("%Y-%m-%d"))
        tmp = self.path(symbol).with_suffix(".tmp")
        out.to_csv(tmp, index=False)
        tmp.replace(self.path(symbol))

    def update(self, symbols: list[str], end: date) -> dict[str, pd.DataFrame]:
        """Refresh the cache up to ``end``.

        Providers return prices adjusted for splits *as of today*, so after a
        split the cached history no longer matches. The last cached bar is
        compared with the provider's current value; on a mismatch the symbol's
        history is downloaded again, and a split-like ratio (e.g. 10 for a
        10-for-1 split) is reported in ``self.splits`` so open positions can be
        rescaled.
        """
        self.splits: dict[str, tuple[str, float]] = {}
        start_full = end - timedelta(days=self.history_days)
        frames = {s: self.load(s) for s in symbols}
        need_full = [s for s, f in frames.items() if f.empty or f.index[0].date() > start_full
                     + timedelta(days=10)]
        if need_full:
            log.info("downloading %d symbols of history from %s (saved as it goes: an "
                     "interrupted run resumes where it stopped)", len(need_full),
                     self.provider.name)
            step = max(1, self.provider.batch_size)
            for i in range(0, len(need_full), step):
                chunk = need_full[i:i + step]
                for sym, df in self.provider.history_many(chunk, start_full, end).items():
                    frames[sym] = normalise(pd.concat([frames.get(sym, empty_frame()), df]))
                    if not frames[sym].empty:
                        self.save(sym, frames[sym])
                done = min(i + step, len(need_full))
                if done == len(need_full) or done % 10 < step:
                    log.info("history: %d/%d symbols downloaded (last: %s)", done,
                             len(need_full), chunk[-1])
        stale = [s for s, f in frames.items() if s not in need_full and not f.empty
                 and f.index[-1].date() < end]
        if stale:
            last = {sym: frames[sym].index[-1] for sym in stale}
            mismatch: dict[str, float] = {}
            # Grouped daily files (one call per day) only for symbols that are a
            # few days behind: one long-stale symbol must not slow everything down.
            recent = [sym for sym in stale if (end - last[sym].date()).days <= 21]
            if not self.provider.supports_grouped:
                recent = []
            if recent:
                oldest = min(last[sym].date() for sym in recent)
                days = [oldest + timedelta(days=i) for i in range(1, (end - oldest).days + 1)]
                days = [d for d in days if d.weekday() < 5]
                for check in sorted({last[sym].date() for sym in recent}):
                    rows = self.provider.grouped(check) or {}
                    for sym in recent:
                        if last[sym].date() == check and sym in rows:
                            self._compare(sym, frames[sym], rows[sym]["close"], mismatch)
                for day in days:
                    rows = self.provider.grouped(day) or {}
                    for sym in recent:
                        if sym in rows and sym not in mismatch \
                                and frames[sym].index[-1].date() < day:
                            new = pd.DataFrame([rows[sym]], index=pd.to_datetime([day]))
                            frames[sym] = normalise(pd.concat([frames[sym], new]))
            behind = [sym for sym in stale if sym not in recent]
            if behind:
                oldest = min(last[sym].date() for sym in behind)
                for sym, df in self.provider.history_many(behind, oldest, end).items():
                    if last[sym] in df.index:
                        self._compare(sym, frames[sym], float(df.loc[last[sym], "close"]),
                                      mismatch)
                    if sym not in mismatch:
                        frames[sym] = normalise(pd.concat([frames[sym], df]))
            if mismatch:
                log.warning("price history changed for %s: downloading it again",
                            ", ".join(sorted(mismatch)))
                fresh = self.provider.history_many(sorted(mismatch), start_full, end)
                for sym, df in fresh.items():
                    if df.empty:
                        continue
                    ratio = split_ratio(mismatch[sym])
                    if ratio:
                        new_days = df.index[df.index > last[sym]]
                        # No new bar yet (weekend): the split is effective next session.
                        ex_date = new_days[0].date() if len(new_days) \
                            else next_trading_day(last[sym].date())
                        self.splits[sym] = (str(ex_date), ratio)
                    frames[sym] = df
        self.splits = self._remember_splits(self.splits, end)
        cutoff = pd.Timestamp(start_full)
        for sym in symbols:
            frames[sym] = frames[sym][frames[sym].index >= cutoff]
            if not frames[sym].empty:
                self.save(sym, frames[sym])
        return frames

    def _remember_splits(self, found: dict[str, tuple[str, float]],
                         end: date) -> dict[str, tuple[str, float]]:
        """Splits survive restarts: once the history is reloaded nothing shows them.

        The engine applies each (symbol, ex-date) once; entries older than 60
        days are forgotten.
        """
        path = self.dir / "splits.json"
        known: dict = {}
        with contextlib.suppress(OSError, ValueError):
            known = json.loads(path.read_text())
        known = {k: tuple(v) for k, v in known.items() if isinstance(v, list) and len(v) == 2}
        known.update(found)
        cutoff = end - timedelta(days=60)
        known = {k: v for k, v in known.items() if date.fromisoformat(v[0]) >= cutoff}
        if known or path.exists():
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps({k: list(v) for k, v in known.items()}, indent=1))
            tmp.replace(path)
        return known

    @staticmethod
    def _compare(sym: str, cached: pd.DataFrame, provider_close: float,
                 mismatch: dict[str, float]) -> None:
        old = float(cached["close"].iloc[-1])
        if provider_close > 0 and abs(old / provider_close - 1) > 0.02:
            mismatch[sym] = old / provider_close  # > 1: forward split, < 1: reverse


# ============================================================ macro series
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API = "https://api.stlouisfed.org/fred/series/observations"
ECB_EURUSD = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"


def fred_series(series_id: str, start: date, api_key: str = "",
                session: requests.Session | None = None) -> pd.Series:
    """FRED daily series (e.g. VIXCLS, VXVCLS). Missing values are dropped."""
    session = session or requests.Session()
    if api_key:
        resp = session.get(FRED_API, params={"series_id": series_id, "api_key": api_key,
                                             "file_type": "json",
                                             "observation_start": str(start)}, timeout=30)
        resp.raise_for_status()
        obs = resp.json().get("observations", [])
        values = {o["date"]: o["value"] for o in obs}
    else:
        resp = session.get(FRED_CSV, params={"id": series_id, "cosd": str(start)}, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        values = dict(zip(df.iloc[:, 0].astype(str), df.iloc[:, 1].astype(str), strict=False))
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def ecb_eurusd(start: date, session: requests.Session | None = None) -> pd.Series:
    """USD per 1 EUR, ECB reference rate."""
    session = session or requests.Session()
    resp = session.get(ECB_EURUSD, params={"startPeriod": str(start), "format": "csvdata"},
                       timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    series = pd.to_numeric(pd.Series(df["OBS_VALUE"].values,
                                     index=pd.to_datetime(df["TIME_PERIOD"])), errors="coerce")
    return series.dropna().sort_index()


AV_URL = "https://www.alphavantage.co/query"


def alphavantage_daily(symbol: str, api_key: str,
                       session: requests.Session | None = None) -> pd.DataFrame:
    """Last ~100 daily bars of any Alpha Vantage symbol (used for EU-listed ETPs)."""
    session = session or requests.Session()
    resp = session.get(AV_URL, params={"function": "TIME_SERIES_DAILY", "symbol": symbol,
                                       "outputsize": "compact", "apikey": api_key},
                       timeout=30)
    resp.raise_for_status()
    body = resp.json()
    series = body.get("Time Series (Daily)")
    if not series:
        raise DataError(f"Alpha Vantage {symbol}: {str(body)[:150]}")
    df = pd.DataFrame.from_dict(series, orient="index")
    df = df.rename(columns={"1. open": "open", "2. high": "high", "3. low": "low",
                            "4. close": "close", "5. volume": "volume"})
    return normalise(df.astype(float))


@dataclass
class MarketData:
    """Everything the strategy reads for one decision (daily bars + macro)."""

    bars: dict[str, pd.DataFrame]
    splits: dict[str, tuple[str, float]] = field(default_factory=dict)
    vix: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    vix3m: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    eurusd: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    eurusd_fallback: float = 1.10

    def fx(self, day: pd.Timestamp) -> float:
        """USD per EUR on ``day`` (last known value)."""
        if self.eurusd.empty:
            return self.eurusd_fallback
        past = self.eurusd[self.eurusd.index <= day]
        return float(past.iloc[-1]) if len(past) else float(self.eurusd.iloc[0])

    def last_date(self, symbol: str) -> pd.Timestamp | None:
        df = self.bars.get(symbol)
        return None if df is None or df.empty else df.index[-1]


def load_market_data(s, symbols: list[str], end: date, store: BarStore | None = None,
                     session: requests.Session | None = None, fx_fallback: float | None = None,
                     last_day: date | None = None) -> MarketData:
    """``fx_fallback``: last EUR/USD rate known to the bot (the ECB may be down).
    ``last_day``: last processed close; the ETP prices (Alpha Vantage, 25 calls a
    day on the free plan) are fetched only once a newer close exists."""
    store = store or BarStore(s.data_dir, make_provider(s), s.history_days)
    bars = store.update(symbols, end)
    splits = dict(getattr(store, "splits", {}))
    bench = bars.get(s.benchmark)
    new_close = bench is not None and not bench.empty and (
        last_day is None or bench.index[-1].date() > last_day)
    if s.broker != "sim" and s.etp_enabled and s.etp_price_symbol and s.alphavantage_api_key \
            and new_close:
        from invest.config import etp_symbol

        try:
            bars[etp_symbol(s)] = alphavantage_daily(s.etp_price_symbol, s.alphavantage_api_key,
                                                     session)
        except Exception as exc:
            log.warning("ETP prices unavailable (%s): no ETP decision today", exc)
    start = end - timedelta(days=s.history_days)
    macro = {}
    for name, loader in (("vix", lambda: fred_series("VIXCLS", start, s.fred_api_key, session)),
                         ("vix3m", lambda: fred_series("VXVCLS", start, s.fred_api_key, session)),
                         ("eurusd", lambda: ecb_eurusd(start, session))):
        try:
            macro[name] = loader()
        except Exception as exc:  # macro data is optional; the bot degrades gracefully
            log.warning("%s unavailable: %s", name, exc)
            macro[name] = pd.Series(dtype=float)
    return MarketData(bars=bars, splits=splits,
                      eurusd_fallback=fx_fallback or s.eurusd_fallback, **macro)
