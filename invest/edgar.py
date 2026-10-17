"""SEC EDGAR: the free, official record of what US companies and their insiders announce.

* Insider transactions data sets: one zip per quarter with every Form 3/4/5 filed
  (all companies, since 2006).  Each filing carries the issuer's ticker AS REPORTED AT
  THE TIME, so the sample is point-in-time: companies delisted or renamed since are in it.
* Submissions JSON (one per company): every filing with its form, its 8-K items and its
  acceptance time.

SEC fair-access rules: at most 10 requests per second and a User-Agent that names you
and gives an email (SEC_USER_AGENT="Firstname Lastname you@example.com").

Pitfall handled here: the ``acceptanceDateTime`` of the submissions JSON is New York
wall-clock time shifted TWICE by the UTC offset (a filing accepted 06:30 ET reads
"14:30Z" in summer, not "10:30Z").  ``accepted_et`` undoes it, and ``calibrate`` checks
it against the filing header on a few filings before a study relies on it.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import re
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

log = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

INSIDER_ZIP = ("https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/"
               "{y}q{q}_form345.zip",
               "https://www.sec.gov/files/datastandardsinnovation/data/"
               "insider-transactions-data-sets/{y}q{q}_form345.zip")
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
FILING_TXT = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_plain}/{acc}.txt"

PURCHASE_COLUMNS = ["accession", "filed", "traded", "cik", "symbol", "owner_cik", "owner",
                    "role", "officer", "director", "owner10", "shares", "value", "price",
                    "owned_after", "delta_own", "plan"]
ISSUER_COLUMNS = ["cik", "symbol", "first", "last"]
EIGHTK_COLUMNS = ["cik", "accession", "filed", "items", "raw_time"]
_NOT_SHARES = ("preferred", "note", "warrant", "debenture", "right", "unit", "option",
               "bond", "swap")
_UA = re.compile(r"\S+@\S+\.\S+")


class EdgarError(RuntimeError):
    pass


def _check_agent(user_agent: str) -> str:
    ua = (user_agent or "").strip()
    if not _UA.search(ua):
        raise EdgarError("SEC_USER_AGENT must name you and give an email, e.g. "
                         '"Jane Doe jane@example.com" (SEC fair-access rule). Set it in .env')
    return ua


# ================================================================== time helpers
def accepted_et(text: str, mode: str = "double") -> datetime | None:
    """New York time of an ``acceptanceDateTime`` from the submissions JSON.

    ``double`` (observed 2025-2026): the value is ET wall time plus twice the UTC offset.
    ``single``: the value is true UTC (in case the SEC fixes it)."""
    if not text:
        return None
    try:
        t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    if t.tzinfo is None:
        t = t.replace(tzinfo=timezone.utc)
    t = t.astimezone(ET)
    if mode == "double":
        t = t.replace(tzinfo=timezone.utc).astimezone(ET)
    return t.replace(tzinfo=None)


def _sec_date(text) -> str | None:
    """'31-JAN-2024' or '2024-01-31' -> '2024-01-31'."""
    if not isinstance(text, str) or not text.strip():
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        with contextlib.suppress(ValueError):
            return datetime.strptime(text.strip(), fmt).strftime("%Y-%m-%d")
    return None


def clean_symbol(text) -> str | None:
    """'nasdaq: abc', 'ABC, ABCW', 'brk-b' -> 'ABC', 'ABC', 'BRK.B'; junk -> None."""
    if not isinstance(text, str):
        return None
    s = text.strip().upper()
    if ":" in s:
        s = s.split(":")[-1].strip()
    s = re.split(r"[,;/\s]+", s)[0] if s else ""
    s = s.replace("-", ".")
    if s in ("", "NONE", "N.A", "NA", "N/A") or not re.fullmatch(r"[A-Z]{1,5}(\.[A-Z]{1,2})?", s):
        return None
    return s


def role_of(relationship: str, title: str) -> str:
    rel = (relationship or "").lower()
    t = f" {(title or '').lower()} "
    if "chief executive" in t or re.search(r"\bceo\b", t):
        return "ceo"
    if "chief financial" in t or re.search(r"\bcfo\b", t):
        return "cfo"
    if "officer" in rel:
        return "officer"
    if "director" in rel:
        return "director"
    if "tenpercent" in rel.replace(" ", "").replace("%", "percent"):
        return "owner10"
    return "other"


# ======================================================================== client
class Edgar:
    """Polite EDGAR client with a disk cache under ``cache_dir``."""

    def __init__(self, user_agent: str, cache_dir: str | Path = "data_events/sec",
                 session: requests.Session | None = None, per_second: float = 8.0,
                 sleep=time.sleep, clock=time.monotonic):
        self.ua = _check_agent(user_agent)
        self.dir = Path(cache_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": self.ua, "Accept-Encoding": "gzip, deflate"})
        self.interval = 1.0 / per_second
        self.sleep, self.clock = sleep, clock
        self._last = -1e18
        self.time_mode = "double"

    # ----------------------------------------------------------- plumbing
    def get(self, url: str, allow_404: bool = False) -> requests.Response | None:
        for attempt in range(5):
            wait = self._last + self.interval - self.clock()
            if wait > 0:
                self.sleep(wait)
            self._last = self.clock()
            try:
                resp = self.session.get(url, timeout=60)
            except requests.exceptions.RequestException as exc:
                if attempt == 4:
                    raise EdgarError(f"SEC unreachable: {exc}") from exc
                self.sleep(2 ** attempt)
                continue
            if resp.status_code == 404 and allow_404:
                return None
            # the SEC answers 403 "Request Rate Threshold Exceeded" or 429 when pushed
            if resp.status_code in (403, 429, 500, 502, 503, 504):
                if resp.status_code == 403 and "threshold" not in resp.text.lower():
                    raise EdgarError(f"SEC refused {url} (HTTP 403). Is SEC_USER_AGENT a real "
                                     "name and email?")
                self.sleep(min(60, 5 * 2 ** attempt))
                continue
            if resp.status_code >= 400:
                raise EdgarError(f"SEC HTTP {resp.status_code} for {url}")
            return resp
        raise EdgarError(f"SEC kept refusing {url} (rate limit)")

    # ------------------------------------------------------ insider data sets
    def insider_quarter(self, year: int, quarter: int, today: date | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        """(open-market purchases, issuer tickers) of one quarter, parsed once and cached.
        None when the SEC has not published that quarter yet."""
        today = today or date.today()
        tag = f"{year}q{quarter}"
        buys_path = self.dir / "form345" / f"{tag}_purchases.csv"
        iss_path = self.dir / "form345" / f"{tag}_issuers.csv"
        if buys_path.exists() and iss_path.exists():
            return (pd.read_csv(buys_path, dtype={"symbol": str, "owner_cik": str}),
                    pd.read_csv(iss_path, dtype={"symbol": str}))
        end_of_quarter = date(year, 3 * quarter, 1) + timedelta(days=31)
        if end_of_quarter > today:
            return None  # quarter not over: not published
        resp = None
        for pattern in INSIDER_ZIP:
            resp = self.get(pattern.format(y=year, q=quarter), allow_404=True)
            if resp is not None:
                break
        if resp is None:
            log.warning("SEC insider data set %s not found (not published yet?)", tag)
            return None
        buys, issuers = parse_form345(resp.content)
        buys_path.parent.mkdir(parents=True, exist_ok=True)
        buys.to_csv(buys_path, index=False)
        issuers.to_csv(iss_path, index=False)
        log.info("SEC insider data set %s: %d open-market purchases, %d issuers", tag,
                 len(buys), len(issuers))
        return buys, issuers

    def insiders(self, start: date, end: date, today: date | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
        buys, issuers = [], []
        y, q = start.year, (start.month - 1) // 3 + 1
        while (y, q) <= (end.year, (end.month - 1) // 3 + 1):
            got = self.insider_quarter(y, q, today)
            if got is not None:
                buys.append(got[0])
                issuers.append(got[1])
            y, q = (y + 1, 1) if q == 4 else (y, q + 1)
        b = pd.concat(buys, ignore_index=True) if buys else pd.DataFrame(columns=PURCHASE_COLUMNS)
        i = (pd.concat(issuers, ignore_index=True) if issuers
             else pd.DataFrame(columns=ISSUER_COLUMNS))
        if not i.empty:
            i = i.groupby(["cik", "symbol"], as_index=False).agg(first=("first", "min"),
                                                                 last=("last", "max"))
        b = b[(b["filed"] >= str(start)) & (b["filed"] <= str(end))] if not b.empty else b
        return b.reset_index(drop=True), i

    # ------------------------------------------------------------- companies
    def tickers(self) -> dict[str, int]:
        """Current ticker -> CIK (cached one day)."""
        path = self.dir / "company_tickers.json"
        fresh = path.exists() and time.time() - path.stat().st_mtime < 86400
        if not fresh:
            path.write_bytes(self.get(TICKERS_URL).content)
        data = json.loads(path.read_text())
        out = {}
        for v in data.values():
            sym = clean_symbol(str(v.get("ticker", "")))
            if sym and sym not in out:
                out[sym] = int(v["cik_str"])
        return out

    def eight_ks(self, cik: int, since: date, until: date | None = None) -> pd.DataFrame:
        """Every 8-K of a company since ``since`` with items and New York acceptance time.
        Cached per company; fetched again only when the cache stops more than a week
        before ``until`` or starts after ``since``."""
        path = self.dir / "8k" / f"{int(cik)}.csv"
        marker = path.parent / f"{int(cik)}.json"
        until = until or date.today()
        df = None
        with contextlib.suppress(OSError, ValueError, KeyError):
            meta = json.loads(marker.read_text())
            if meta["since"] <= str(since) and meta["fetched"] >= str(until - timedelta(days=7)):
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if df is None:
            resp = self.get(SUBMISSIONS_URL.format(name=f"CIK{int(cik):010d}.json"),
                            allow_404=True)
            rows: list[dict] = []
            if resp is not None:
                filings = resp.json().get("filings", {})
                rows += _eightk_rows(int(cik), filings.get("recent", {}))
                for extra in filings.get("files", []):
                    if str(extra.get("filingTo", "9999")) < str(since):
                        continue
                    more = self.get(SUBMISSIONS_URL.format(name=extra["name"]), allow_404=True)
                    if more is not None:
                        rows += _eightk_rows(int(cik), more.json())
            df = pd.DataFrame(rows, columns=EIGHTK_COLUMNS).drop_duplicates("accession")
            df = df.sort_values("filed").reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            marker.write_text(json.dumps({"since": str(since), "fetched": str(date.today())}))
        df = df[df["filed"] >= str(since)].reset_index(drop=True)
        df["cik"] = int(cik)
        # read the raw time with the calibrated convention, never a cached interpretation
        df["accepted_et"] = [accepted_et(x, self.time_mode) for x in df["raw_time"]]
        return df

    def header_acceptance(self, cik: int, accession: str) -> datetime | None:
        """Acceptance time printed in the filing header (New York time)."""
        resp = self.get(FILING_TXT.format(cik=int(cik), acc_plain=accession.replace("-", ""),
                                          acc=accession), allow_404=True)
        if resp is None:
            return None
        m = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", resp.text[:5000])
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None

    def calibrate(self, samples: list[tuple[int, str, str]]) -> str:
        """Check how the JSON acceptance times must be read on up to 3 (cik, accession,
        raw JSON time) samples against their filing headers; sets and returns the mode."""
        votes = {"double": 0, "single": 0}
        for cik, acc, raw in samples[:3]:
            truth = self.header_acceptance(cik, acc)
            if truth is None:
                continue
            for mode in votes:
                got = accepted_et(raw, mode)
                if got is not None and abs((got - truth).total_seconds()) < 120:
                    votes[mode] += 1
        if not any(votes.values()):
            raise EdgarError("cannot calibrate the 8-K acceptance times against the filing "
                             "headers; the SEC may have changed its format")
        self.time_mode = max(votes, key=votes.get)
        log.info("8-K acceptance times read in '%s' mode (%s)", self.time_mode, votes)
        return self.time_mode


def _eightk_rows(cik: int, block: dict) -> list[dict]:
    out = []
    keys = ("form", "filingDate", "acceptanceDateTime", "accessionNumber", "items")
    n = len(block.get("form") or [])
    cols = [list(block.get(k) or []) + [""] * (n - len(block.get(k) or [])) for k in keys]
    for form, filed, acc_time, acc, items in zip(*cols, strict=True):
        if form == "8-K":
            out.append({"cik": cik, "accession": acc, "filed": filed,
                        "items": str(items or ""), "raw_time": acc_time})
    return out


# ===================================================================== parsing
def _read_tsv(z: zipfile.ZipFile, name: str, wanted: list[str]) -> pd.DataFrame:
    member = next((n for n in z.namelist() if n.upper().endswith(name)), None)
    if member is None:
        raise EdgarError(f"insider data set without {name}")
    with z.open(member) as fh:
        df = pd.read_csv(io.TextIOWrapper(fh, encoding="latin-1"), sep="\t", dtype=str,
                         usecols=lambda c: c in wanted, quoting=3, on_bad_lines="skip")
    for c in wanted:
        if c not in df.columns:
            df[c] = None
    return df


def parse_form345(content: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open-market purchases (Form 4, code P, shares acquired) and the issuers' tickers."""
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        sub = _read_tsv(z, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE",
                                              "DOCUMENT_TYPE", "ISSUERCIK",
                                              "ISSUERTRADINGSYMBOL", "AFF10B5ONE"])
        tx = _read_tsv(z, "NONDERIV_TRANS.TSV", [
            "ACCESSION_NUMBER", "SECURITY_TITLE", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
            "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
            "DIRECT_INDIRECT_OWNERSHIP"])
        own = _read_tsv(z, "REPORTINGOWNER.TSV", ["ACCESSION_NUMBER", "RPTOWNERCIK",
                                                  "RPTOWNERNAME", "RPTOWNER_RELATIONSHIP",
                                                  "RPTOWNER_TITLE"])
    sub["filed"] = sub["FILING_DATE"].map(_sec_date)
    sub["symbol"] = sub["ISSUERTRADINGSYMBOL"].map(clean_symbol)
    sub["cik"] = pd.to_numeric(sub["ISSUERCIK"], errors="coerce")
    sub = sub.dropna(subset=["filed", "cik"])
    issuers = (sub.dropna(subset=["symbol"])
               .groupby(["cik", "symbol"], as_index=False)
               .agg(first=("filed", "min"), last=("filed", "max")))
    issuers["cik"] = issuers["cik"].astype(int)

    forms = sub[sub["DOCUMENT_TYPE"].astype(str).str.strip() == "4"].dropna(subset=["symbol"])
    tx = tx[(tx["TRANS_CODE"].astype(str).str.strip() == "P")
            & (tx["TRANS_ACQUIRED_DISP_CD"].astype(str).str.strip().isin(["A", "nan", ""]))]
    title = tx["SECURITY_TITLE"].fillna("").str.lower()
    tx = tx[~title.apply(lambda t: any(w in t for w in _NOT_SHARES))].copy()
    tx["shares"] = pd.to_numeric(tx["TRANS_SHARES"], errors="coerce")
    tx["price"] = pd.to_numeric(tx["TRANS_PRICEPERSHARE"], errors="coerce")
    tx["after"] = pd.to_numeric(tx["SHRS_OWND_FOLWNG_TRANS"], errors="coerce")
    tx = tx[(tx["shares"] > 0) & (tx["price"] > 0)]
    tx["value"] = tx["shares"] * tx["price"]
    tx["traded"] = tx["TRANS_DATE"].map(_sec_date)
    direct = tx["DIRECT_INDIRECT_OWNERSHIP"].astype(str).str.strip().str.upper() == "D"
    tx["after_direct"] = tx["after"].where(direct)
    agg = tx.groupby("ACCESSION_NUMBER").agg(
        shares=("shares", "sum"), value=("value", "sum"), traded=("traded", "max"),
        owned_after=("after_direct", "max"), owned_any=("after", "max"))
    agg["owned_after"] = agg["owned_after"].fillna(agg["owned_any"])
    agg["price"] = agg["value"] / agg["shares"]

    own = own.sort_values("ACCESSION_NUMBER").drop_duplicates("ACCESSION_NUMBER")
    own = own.set_index("ACCESSION_NUMBER")
    df = forms.set_index("ACCESSION_NUMBER").join(agg, how="inner").join(own, how="left")
    if df.empty:
        return pd.DataFrame(columns=PURCHASE_COLUMNS), issuers
    df["role"] = [role_of(r, t) for r, t in zip(df["RPTOWNER_RELATIONSHIP"].fillna(""),
                                                df["RPTOWNER_TITLE"].fillna(""), strict=True)]
    rel = df["RPTOWNER_RELATIONSHIP"].fillna("").str.lower()
    df["officer"] = rel.str.contains("officer") | df["role"].isin(["ceo", "cfo"])
    df["director"] = rel.str.contains("director")
    df["owner10"] = rel.str.replace(" ", "").str.contains("tenpercent")
    before = (df["owned_after"] - df["shares"]).clip(lower=0)
    df["delta_own"] = (df["shares"] / before.where(before > 0)).fillna(10.0).clip(upper=10.0)
    flag = df["AFF10B5ONE"].astype(str).str.strip().str.lower()
    df["plan"] = flag.isin(["1", "true", "y", "yes"])
    df["owner_cik"] = df["RPTOWNERCIK"].astype(str).str.lstrip("0")
    df["owner"] = df["RPTOWNERNAME"].fillna("")
    df["cik"] = df["cik"].astype(int)
    out = df.reset_index().rename(columns={"ACCESSION_NUMBER": "accession"})
    out = out[PURCHASE_COLUMNS].sort_values(["filed", "accession"]).reset_index(drop=True)
    return out, issuers


def symbol_resolver(issuers: pd.DataFrame, current: dict[str, int] | None = None):
    """``f(cik, day) -> ticker`` used by the company on that day (insider filings record the
    ticker at the time; the current SEC list is the fallback)."""
    by_cik: dict[int, list[tuple[str, str, str]]] = {}
    for r in issuers.itertuples(index=False):
        by_cik.setdefault(int(r.cik), []).append((str(r.first), str(r.last), str(r.symbol)))
    latest = {}
    for sym, cik in (current or {}).items():
        latest.setdefault(int(cik), sym)

    def resolve(cik: int, day: str) -> str | None:
        spans = by_cik.get(int(cik))
        if spans:
            inside = [s for f, lst, s in spans if f <= day <= lst]
            if inside:
                return inside[0]
            # nearest span in time: tickers change rarely, the closest record is the best bet
            best = min(spans, key=lambda x: min(abs(_days(day, x[0])), abs(_days(day, x[1]))))
            return best[2]
        return latest.get(int(cik))

    return resolve


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(a) - date.fromisoformat(b)).days
