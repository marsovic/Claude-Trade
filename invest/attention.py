"""Retail attention: what people talk about and look up, collected every day.

There is no free HISTORY of Reddit or X activity: ApeWisdom only shows the last 24 hours
and the X API is paid.  So this module

1. saves a dated ApeWisdom snapshot (Reddit mentions of every stock) each day.  After a
   few months the snapshots ARE a history, and ``python -m invest.events --families
   attention`` tests whether a mention spike predicted anything;
2. downloads Wikipedia pageviews, the one free attention series with years of history
   (July 2015 onwards), so attention spikes can be tested today.  Articles are found
   through Wikidata, which records each company's SEC CIK number.

    python -m invest.attention collect                  # once a day, after the US close
    python -m invest.attention wiki                     # curated universe
    python -m invest.attention wiki --symbols-file my_list.txt
    python -m invest.attention wiki --all-sec           # every SEC-listed company
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

log = logging.getLogger(__name__)

APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
PAGEVIEWS_URL = ("https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/"
                 "all-access/user/{title}/daily/{start}00/{end}00")
WIKI_START = date(2015, 7, 1)  # first day of the pageviews API
REDDIT_COLUMNS = ["taken_utc", "rank", "ticker", "name", "mentions", "upvotes",
                  "rank_24h_ago", "mentions_24h_ago"]


class AttentionError(RuntimeError):
    pass


def _get(session: requests.Session, url: str, params: dict | None = None, tries: int = 5,
         sleep=time.sleep, headers: dict | None = None) -> requests.Response | None:
    """GET with backoff on 429/5xx; None on 404."""
    for attempt in range(tries):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=60)
        except requests.exceptions.RequestException as exc:
            if attempt == tries - 1:
                raise AttentionError(f"{url}: {exc}") from exc
            sleep(2 ** attempt)
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code in (429, 500, 502, 503, 504):
            sleep(min(60, 3 * 2 ** attempt))
            continue
        if resp.status_code >= 400:
            raise AttentionError(f"HTTP {resp.status_code} for {url}: {resp.text[:200]}")
        return resp
    raise AttentionError(f"{url}: still rate limited after {tries} tries")


# ===================================================================== ApeWisdom
def snapshot_reddit(session: requests.Session | None = None, sleep=time.sleep,
                    now: datetime | None = None, max_pages: int = 30) -> pd.DataFrame:
    """Every stock ApeWisdom tracks right now (all pages)."""
    session = session or requests.Session()
    taken = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows, page, pages = [], 1, 1
    while page <= min(pages, max_pages):
        resp = _get(session, APEWISDOM_URL.format(page=page), sleep=sleep)
        if resp is None:
            break
        body = resp.json()
        pages = int(body.get("pages") or 1)
        for r in body.get("results") or []:
            rows.append({"taken_utc": taken, "rank": _num(r.get("rank")),
                         "ticker": str(r.get("ticker", "")).upper().strip(),
                         "name": str(r.get("name", "")).strip(),
                         "mentions": _num(r.get("mentions")), "upvotes": _num(r.get("upvotes")),
                         "rank_24h_ago": _num(r.get("rank_24h_ago")),
                         "mentions_24h_ago": _num(r.get("mentions_24h_ago"))})
        page += 1
        sleep(1.0)
    return pd.DataFrame(rows, columns=REDDIT_COLUMNS)


def collect(out_dir: str | Path, session=None, now: datetime | None = None, force=False,
            sleep=time.sleep) -> Path | None:
    """Save today's snapshot as ``<out>/apewisdom/YYYY-MM-DD.csv`` (UTC date, once a day)."""
    now = now or datetime.now(timezone.utc)
    path = Path(out_dir) / "apewisdom" / f"{now.strftime('%Y-%m-%d')}.csv"
    if path.exists() and not force:
        log.info("snapshot %s already taken today", path)
        return None
    df = snapshot_reddit(session, sleep=sleep, now=now)
    if df.empty:
        raise AttentionError("ApeWisdom returned no stocks")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    log.info("saved %d stocks to %s", len(df), path)
    return path


def load_reddit(out_dir: str | Path) -> pd.DataFrame:
    """All saved snapshots, oldest first."""
    files = sorted((Path(out_dir) / "apewisdom").glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=REDDIT_COLUMNS)
    df = pd.concat([pd.read_csv(f, dtype={"ticker": str}) for f in files], ignore_index=True)
    df["taken_utc"] = pd.to_datetime(df["taken_utc"], utc=True)
    return df.sort_values(["taken_utc", "rank"]).reset_index(drop=True)


# ===================================================================== Wikipedia
def wiki_titles(ciks: dict[str, int], session: requests.Session, user_agent: str,
                sleep=time.sleep, batch: int = 150) -> dict[str, str | None]:
    """symbol -> English Wikipedia article of the company with that SEC CIK (Wikidata P5531)."""
    out: dict[str, str | None] = {s: None for s in ciks}
    by_cik: dict[str, list[str]] = {}
    for sym, cik in ciks.items():
        by_cik.setdefault(f"{int(cik):010d}", []).append(sym)
    keys = list(by_cik)
    for i in range(0, len(keys), batch):
        values = " ".join(f'"{k}"' for k in keys[i:i + batch])
        query = ("SELECT ?cik ?article WHERE { VALUES ?cik { " + values + " } "
                 "?item wdt:P5531 ?cik . ?article schema:about ?item ; "
                 "schema:isPartOf <https://en.wikipedia.org/> . }")
        resp = _get(session, WIKIDATA_SPARQL, params={"query": query, "format": "json"},
                    headers={"User-Agent": user_agent,
                             "Accept": "application/sparql-results+json"}, sleep=sleep)
        if resp is None:
            continue
        for b in resp.json().get("results", {}).get("bindings", []):
            cik = b["cik"]["value"]
            title = b["article"]["value"].rsplit("/wiki/", 1)[-1]
            for sym in by_cik.get(cik, []):
                out[sym] = out[sym] or requests.utils.unquote(title)
        sleep(1.0)
    return out


def pageviews(title: str, start: date, end: date, session: requests.Session,
              user_agent: str, sleep=time.sleep) -> pd.Series:
    """Daily human pageviews of one article (missing days = 0)."""
    url = PAGEVIEWS_URL.format(title=quote(title.replace(" ", "_"), safe=""),
                               start=start.strftime("%Y%m%d"), end=end.strftime("%Y%m%d"))
    resp = _get(session, url, headers={"User-Agent": user_agent, "Api-User-Agent": user_agent},
                sleep=sleep)
    idx = pd.date_range(start, end, freq="D")
    if resp is None:
        return pd.Series(0.0, index=idx, name="views")
    items = resp.json().get("items") or []
    s = pd.Series({pd.Timestamp(datetime.strptime(it["timestamp"][:8], "%Y%m%d")):
                   float(it.get("views", 0)) for it in items}, dtype=float)
    return s.reindex(idx, fill_value=0.0).rename("views")


class WikiStore:
    """``<dir>/wiki/titles.json`` and ``<dir>/wiki/views/<SYMBOL>.csv``, updated incrementally."""

    def __init__(self, out_dir: str | Path, user_agent: str, session=None, sleep=time.sleep,
                 per_second: float = 5.0):
        if "@" not in (user_agent or ""):
            raise AttentionError("Wikimedia asks for a User-Agent with a contact email: set "
                                 'SEC_USER_AGENT="Jane Doe jane@example.com" in .env')
        self.dir = Path(out_dir) / "wiki"
        (self.dir / "views").mkdir(parents=True, exist_ok=True)
        self.ua = user_agent
        self.session = session or requests.Session()
        self.sleep = sleep
        self.pause = 1.0 / per_second

    def titles(self, ciks: dict[str, int], refresh_days: int = 30) -> dict[str, str | None]:
        path = self.dir / "titles.json"
        known: dict = {}
        with contextlib.suppress(OSError, ValueError):
            known = json.loads(path.read_text())
        fresh = str(date.today() - timedelta(days=refresh_days))
        ask = {s: c for s, c in ciks.items()
               if s not in known or (known[s].get("title") is None
                                     and known[s].get("checked", "") < fresh)}
        if ask:
            log.info("looking up %d Wikipedia article(s) on Wikidata", len(ask))
            got = wiki_titles(ask, self.session, self.ua, sleep=self.sleep)
            for s, t in got.items():
                known[s] = {"title": t, "cik": int(ciks[s]), "checked": str(date.today())}
            path.write_text(json.dumps(known, indent=1, sort_keys=True))
        return {s: known.get(s, {}).get("title") for s in ciks}

    def update(self, ciks: dict[str, int], end: date | None = None) -> dict[str, int]:
        """Download what is missing for every symbol with an article; returns days added."""
        end = end or date.today() - timedelta(days=1)
        added = {}
        titles = {s: t for s, t in self.titles(ciks).items() if t}
        log.info("%d of %d symbols have an English Wikipedia article", len(titles), len(ciks))
        for n, (sym, title) in enumerate(sorted(titles.items()), 1):
            path = self.dir / "views" / f"{sym}.csv"
            old = load_views_file(path)
            start = WIKI_START if old.empty else (old.index[-1] + timedelta(days=1)).date()
            if start > end:
                continue
            new = pageviews(title, start, end, self.session, self.ua, sleep=self.sleep)
            merged = pd.concat([old, new])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            merged.rename_axis("date").to_frame("views").to_csv(path)
            added[sym] = len(new)
            if n % 50 == 0:
                log.info("pageviews: %d / %d articles", n, len(titles))
            self.sleep(self.pause)
        return added


def load_views_file(path: Path) -> pd.Series:
    if not path.exists():
        return pd.Series(dtype=float, name="views")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df["views"].astype(float)


def load_wiki(out_dir: str | Path) -> dict[str, pd.Series]:
    folder = Path(out_dir) / "wiki" / "views"
    return {p.stem: load_views_file(p) for p in sorted(folder.glob("*.csv"))}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


# =========================================================================== CLI
def _symbols(args) -> list[str]:
    from invest.universe import build_universe

    syms: list[str] = []
    if args.symbols:
        syms += [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    if args.symbols_file:
        text = Path(args.symbols_file).read_text()
        syms += [x.strip().upper() for x in text.replace(",", "\n").splitlines() if x.strip()]
    if not syms:
        syms = list(build_universe(None, []))
    return sorted(set(syms))


def main(argv=None) -> int:
    from invest.config import InvestSettings
    from invest.edgar import Edgar, EdgarError

    ap = argparse.ArgumentParser(description="Collect retail-attention data (Reddit via "
                                             "ApeWisdom, Wikipedia pageviews)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="save today's ApeWisdom snapshot (run once a day)")
    c.add_argument("--force", action="store_true", help="replace today's snapshot")
    w = sub.add_parser("wiki", help="download / update Wikipedia pageview history")
    w.add_argument("--symbols", help="comma-separated tickers (default: curated universe)")
    w.add_argument("--symbols-file", help="file with one ticker per line")
    w.add_argument("--all-sec", action="store_true",
                   help="every company in the SEC ticker list (~10 000; about 4 000 have an "
                        "English article): small caps included, ~30 min the first time")
    for p in (c, w):
        p.add_argument("--out", default="data_attention")
        p.add_argument("--env-file", default=".env")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.cmd == "collect":
            path = collect(args.out, force=args.force)
            print(f"snapshot: {path or 'already taken today'}")
            return 0
        s = InvestSettings.from_env(args.env_file, broker="sim")
        edgar = Edgar(s.sec_user_agent, cache_dir=Path(args.out) / "sec")
        current = edgar.tickers()
        syms = sorted(current) if args.all_sec else _symbols(args)
        ciks = {x: current[x] for x in syms if x in current}
        unknown = [x for x in syms if x not in current]
        if unknown:
            log.warning("%d ticker(s) not in the SEC list, skipped: %s", len(unknown),
                        ", ".join(unknown[:20]))
        added = WikiStore(args.out, s.sec_user_agent).update(ciks)
        print(f"pageviews updated for {len(added)} article(s) in {args.out}/wiki/views")
        return 0
    except (AttentionError, EdgarError) as exc:
        print(f"ATTENTION STOPPED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
