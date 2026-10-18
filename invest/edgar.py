"""SEC EDGAR: the free, official record of what US companies and their insiders announce.

* Insider transactions data sets: one zip per quarter with every Form 3/4/5 filed
  (all companies, since 2006).  Each filing carries the issuer's ticker AS REPORTED AT
  THE TIME, so the sample is point-in-time: companies delisted or renamed since are in it.
* Submissions JSON (one per company): every filing with its form, its 8-K items and its
  acceptance time.

SEC fair-access rules: at most 10 requests per second and a User-Agent that names you
and gives an email (SEC_USER_AGENT="Firstname Lastname you@example.com").

Pitfall handled here: the ``acceptanceDateTime`` of the submissions JSON comes in two
formats depending on the company file.  Some files hold true UTC (Abbott, AAR); others
hold New York wall-clock time shifted TWICE by the UTC offset (Apple, JPMorgan, Nvidia: a
filing accepted 06:30 ET reads "14:30Z" in summer, not "10:30Z").  ``block_mode`` tells
them apart for each file (the wrong reading puts filings outside EDGAR's 06:00-22:00
hours), the filing header settles a tie, and when nothing decides, the later reading is
used: an entry can then come late, never before the news was public.
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
from datetime import time as dtime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
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
DAILY_INDEX = "https://www.sec.gov/Archives/edgar/daily-index/{y}/QTR{q}/form.{ymd}.idx"
ARCHIVE = "https://www.sec.gov/Archives/{path}"

PURCHASE_COLUMNS = ["accession", "filed", "traded", "cik", "symbol", "owner_cik", "owner",
                    "role", "officer", "director", "owner10", "shares", "value", "price",
                    "owned_after", "delta_own", "plan"]
ISSUER_COLUMNS = ["cik", "symbol", "first", "last"]
EIGHTK_COLUMNS = ["cik", "accession", "filed", "items", "raw_time", "mode"]
EDGAR_HOURS = (dtime(5, 55), dtime(22, 5))  # EDGAR accepts filings 06:00-22:00 New York
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

    ``double``: the value is ET wall time plus twice the UTC offset (Apple, JPM, Nvidia).
    ``single``: the value is true UTC (Abbott, AAR).  See ``block_mode``."""
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


def block_mode(times, dates) -> str | None:
    """How one submissions block writes its acceptance times: the wrong reading puts many
    filings outside EDGAR's hours or after their filing date (measured on real files: 0-9
    misplaced filings for the right reading, 55-4400 for the wrong one).  None = undecided."""
    bad = {"double": 0, "single": 0}
    for raw, filed in zip(times, dates, strict=False):
        try:
            day = date.fromisoformat(str(filed))
        except ValueError:
            continue
        for mode in bad:
            t = accepted_et(raw, mode)
            if t is None:
                continue
            lag = (day - t.date()).days
            if not (EDGAR_HOURS[0] <= t.time() <= EDGAR_HOURS[1] and 0 <= lag <= 4):
                bad[mode] += 1
    lo, hi = sorted(bad.values())
    if hi >= 3 and lo <= 0.2 * hi:
        return min(bad, key=bad.get)
    return None


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
    if s.rstrip(".") in ("", "NONE", "N.A", "NA", "N/A"):
        return None
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
            if allow_404 and (resp.status_code == 404 or (
                    resp.status_code == 403 and "<Code>AccessDenied</Code>" in resp.text)):
                return None  # the Archives are on S3: a missing file is a 403 AccessDenied
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

    def eight_ks(self, cik: int, since: date, until: date | None = None,
                 max_age_days: int = 7) -> pd.DataFrame:
        """Every 8-K of a company since ``since`` with items and New York acceptance time.
        Cached per company; fetched again when the cache was taken more than
        ``max_age_days`` before ``until`` (0 for live decisions) or starts after ``since``."""
        path = self.dir / "8k" / f"{int(cik)}.csv"
        marker = path.parent / f"{int(cik)}.json"
        until = until or date.today()
        df = None
        with contextlib.suppress(OSError, ValueError, KeyError):
            meta = json.loads(marker.read_text())
            fresh_from = str(until - timedelta(days=max_age_days))
            # max_age_days=0 (live decisions) always refetches: the cache date is the
            # machine's local date, a day ahead of New York on a Mac in Europe at night
            if (max_age_days > 0 and meta["since"] <= str(since) and meta["fetched"] >= fresh_from
                    and meta.get("version") == 2):
                df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if df is None:
            resp = self.get(SUBMISSIONS_URL.format(name=f"CIK{int(cik):010d}.json"),
                            allow_404=True)
            rows: list[dict] = []
            if resp is not None:
                filings = resp.json().get("filings", {})
                rows += self._block_rows(int(cik), filings.get("recent", {}))
                for extra in filings.get("files", []):
                    if str(extra.get("filingTo", "9999")) < str(since):
                        continue
                    more = self.get(SUBMISSIONS_URL.format(name=extra["name"]), allow_404=True)
                    if more is not None:
                        rows += self._block_rows(int(cik), more.json())
            df = pd.DataFrame(rows, columns=EIGHTK_COLUMNS).drop_duplicates("accession")
            df = df.sort_values("filed").reset_index(drop=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            marker.write_text(json.dumps({"since": str(since), "fetched": str(date.today()),
                                          "version": 2}))
        df = df[df["filed"] >= str(since)].reset_index(drop=True)
        df["cik"] = int(cik)
        df["accepted_et"] = [accepted_et(x, m) for x, m in zip(df["raw_time"], df["mode"],
                                                               strict=True)]
        return df

    def _block_rows(self, cik: int, block: dict) -> list[dict]:
        """8-K rows of one submissions block, each tagged with the block's time format."""
        rows = _eightk_rows(cik, block)
        if not rows:
            return rows
        mode = block_mode(block.get("acceptanceDateTime") or [], block.get("filingDate") or [])
        if mode is None:
            mode = self._mode_from_header(cik, rows[-1]["accession"], rows[-1]["raw_time"])
        for r in rows:
            r["mode"] = mode
        return rows

    def _mode_from_header(self, cik: int, accession: str, raw: str) -> str:
        """Settle an undecided block with one filing header; 'single' (the later reading)
        when even that fails, so a mistake delays an entry instead of anticipating news."""
        with contextlib.suppress(EdgarError):
            truth = self.header_acceptance(cik, accession)
            if truth is not None:
                for mode in ("double", "single"):
                    got = accepted_et(raw, mode)
                    if got is not None and abs((got - truth).total_seconds()) < 120:
                        return mode
        return "single"

    def header_acceptance(self, cik: int, accession: str) -> datetime | None:
        """Acceptance time printed in the filing header (New York time)."""
        resp = self.get(FILING_TXT.format(cik=int(cik), acc_plain=accession.replace("-", ""),
                                          acc=accession), allow_404=True)
        if resp is None:
            return None
        m = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", resp.text[:5000])
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S") if m else None


def _eightk_rows(cik: int, block: dict) -> list[dict]:
    out = []
    keys = ("form", "filingDate", "acceptanceDateTime", "accessionNumber", "items")
    n = len(block.get("form") or [])
    cols = [list(block.get(k) or []) + [""] * (n - len(block.get(k) or [])) for k in keys]
    for form, filed, acc_time, acc, items in zip(*cols, strict=True):
        if form == "8-K":
            out.append({"cik": cik, "accession": acc, "filed": filed,
                        "items": str(items or ""), "raw_time": acc_time, "mode": ""})
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


# ============================================================== live Form 4 feed
_IDX_LINE = re.compile(r"^(?P<form>\S+(?: \S+)*)\s{2,}(?P<company>.+?)\s{2,}(?P<cik>\d+)\s{2,}"
                       r"(?P<date>\d{8})\s{2,}(?P<path>edgar/\S+)\s*$")


def parse_daily_index(text: str, forms: tuple[str, ...] = ("4",)) -> list[dict]:
    """Rows of an EDGAR daily form index for the given form types, one per filing (the
    index lists a Form 4 once under the issuer and once under each reporting owner)."""
    out, seen = [], set()
    for line in text.splitlines():
        m = _IDX_LINE.match(line)
        if not m or m["form"] not in forms:
            continue
        acc = m["path"].rsplit("/", 1)[-1].removesuffix(".txt")
        if acc in seen:  # same filing under the issuer's and the owner's folder
            continue
        seen.add(acc)
        out.append({"form": m["form"], "path": m["path"], "accession": acc,
                    "filed": f"{m['date'][:4]}-{m['date'][4:6]}-{m['date'][6:]}"})
    return out


def _x(node, path: str) -> str:
    """Text of ``path/value`` or ``path`` under an XML node ('' when missing)."""
    if node is None:
        return ""
    el = node.find(path + "/value")
    if el is None:
        el = node.find(path)
    return (el.text or "").strip() if el is not None and el.text else ""


def _true(text: str) -> bool:
    return text.strip().lower() in ("1", "true", "y", "yes")


def parse_form4(text: str, accession: str, filed: str) -> dict | None:
    """One open-market purchase row (PURCHASE_COLUMNS) from a Form 4 submission, or None
    when the filing holds no purchase of shares.  Same rules as ``parse_form345``."""
    import xml.etree.ElementTree as ElementTree

    start, end = text.find("<ownershipDocument"), text.find("</ownershipDocument>")
    if start < 0 or end < 0:
        return None
    try:
        root = ElementTree.fromstring(text[start:end + len("</ownershipDocument>")])
    except ElementTree.ParseError:
        return None
    if _x(root, "documentType") != "4":
        return None
    symbol = clean_symbol(_x(root, "issuer/issuerTradingSymbol"))
    cik = _x(root, "issuer/issuerCik")
    if not symbol or not cik.isdigit():
        return None
    shares = value = 0.0
    traded, owned = "", []
    for tx in root.iter("nonDerivativeTransaction"):
        title = _x(tx, "securityTitle").lower()
        if any(w in title for w in _NOT_SHARES):
            continue
        if _x(tx, "transactionCoding/transactionCode") != "P":
            continue
        if _x(tx, "transactionAmounts/transactionAcquiredDisposedCode") not in ("A", ""):
            continue
        try:
            q = float(_x(tx, "transactionAmounts/transactionShares"))
            px = float(_x(tx, "transactionAmounts/transactionPricePerShare"))
        except ValueError:
            continue
        if q <= 0 or px <= 0:
            continue
        shares += q
        value += q * px
        traded = max(traded, _sec_date(_x(tx, "transactionDate")) or "")
        with contextlib.suppress(ValueError):
            after = float(_x(tx, "postTransactionAmounts/sharesOwnedFollowingTransaction"))
            direct = _x(tx, "ownershipNature/directOrIndirectOwnership").upper() == "D"
            owned.append((direct, after))
    if shares <= 0:
        return None
    owner = root.find("reportingOwner")
    rel = owner.find("reportingOwnerRelationship") if owner is not None else None
    flags = {k: _true(_x(rel, k)) for k in ("isDirector", "isOfficer", "isTenPercentOwner")}
    relationship = ",".join(n for k, n in (("isDirector", "Director"), ("isOfficer", "Officer"),
                                           ("isTenPercentOwner", "TenPercentOwner"))
                            if flags[k])
    title = _x(rel, "officerTitle")
    role = role_of(relationship, title)
    direct = [a for d, a in owned if d]
    owned_after = max(direct) if direct else (max(a for _, a in owned) if owned else np.nan)
    before = owned_after - shares if np.isfinite(owned_after) else np.nan
    delta = shares / before if np.isfinite(before) and before > 0 else 10.0
    return {"accession": accession, "filed": filed, "traded": traded or filed,
            "cik": int(cik), "symbol": symbol,
            "owner_cik": _x(owner, "reportingOwnerId/rptOwnerCik").lstrip("0"),
            "owner": _x(owner, "reportingOwnerId/rptOwnerName"), "role": role,
            "officer": flags["isOfficer"] or role in ("ceo", "cfo"),
            "director": flags["isDirector"], "owner10": flags["isTenPercentOwner"],
            "shares": shares, "value": value, "price": value / shares,
            "owned_after": owned_after, "delta_own": min(delta, 10.0),
            "plan": _true(_x(root, "aff10b5One"))}


class FeedNotReady(EdgarError):
    """The SEC has not published the daily index of that day yet."""


class InsiderFeed:
    """Open-market purchases filed on one business day, from the EDGAR daily index and each
    Form 4 (about 500 filings a day: 1-2 minutes at the SEC's rate limit), cached."""

    def __init__(self, edgar: Edgar):
        self.edgar = edgar
        self.dir = edgar.dir / "form4_daily"
        self.dir.mkdir(parents=True, exist_ok=True)

    def purchases_on(self, day: date) -> pd.DataFrame:
        path = self.dir / f"{day}.csv"
        if path.exists():
            return pd.read_csv(path, dtype={"symbol": str, "owner_cik": str,
                                            "accession": str})
        q = (day.month - 1) // 3 + 1
        resp = self.edgar.get(DAILY_INDEX.format(y=day.year, q=q, ymd=day.strftime("%Y%m%d")),
                              allow_404=True)
        if resp is None:
            raise FeedNotReady(f"SEC daily index of {day} not published yet")
        rows = []
        for f in parse_daily_index(resp.text):
            doc = self.edgar.get(ARCHIVE.format(path=f["path"]), allow_404=True)
            if doc is None:
                continue
            row = parse_form4(doc.text, f["accession"], f["filed"])
            if row is not None:
                rows.append(row)
        df = pd.DataFrame(rows, columns=PURCHASE_COLUMNS)
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)
        log.info("SEC Form 4 of %s: %d open-market purchase(s)", day, len(df))
        return df


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
