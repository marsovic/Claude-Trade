"""Research & sentiment sources for stocks (vetoes and bonuses, never triggers).

* News (per stock): Alpha Vantage NEWS_SENTIMENT scores when a key is set
  (25 calls/day on the free plan), otherwise Finnhub company-news headlines
  scored with a finance lexicon. A clearly negative tone vetoes an entry.
* Retail attention: ApeWisdom (Reddit / WallStreetBets mention counts, free,
  no key). A mention explosion on an already overbought stock = mania veto.
* Earnings calendar: Alpha Vantage EARNINGS_CALENDAR (one call/day) or
  Finnhub. No new entry right before results; weak positions exit before them.
* Insider buying: SEC EDGAR Form 4 open-market purchases (code "P") in the
  last 30 days add a bonus to the momentum score. Needs SEC_USER_AGENT
  ("Name email@example.com"), as required by the SEC.

Every source fails safely to "no opinion" and is cached.
"""

from __future__ import annotations

import io
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta

import pandas as pd
import requests

from bot.sentiment import BEARISH, BULLISH, NEGATIONS

log = logging.getLogger(__name__)

STOCK_BULLISH = BULLISH | {"beat", "beats", "upgrade", "upgraded", "raises", "raised",
                           "buyback", "outperform", "overweight", "strong", "growth",
                           "expands", "wins", "contract", "soars", "jumps", "surges"}
STOCK_BEARISH = BEARISH | {"miss", "misses", "downgrade", "downgraded", "cuts", "cut",
                           "probe", "recall", "layoffs", "delay", "delays", "halt",
                           "halted", "slump", "slumps", "underperform", "underweight",
                           "weak", "guidance-cut", "subpoena", "short-seller"}
_TOKEN = re.compile(r"[a-z][a-z'-]*")

AV_URL = "https://www.alphavantage.co/query"
FINNHUB_URL = "https://finnhub.io/api/v1"
APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/{page}"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SEC_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


def headline_score(texts: list[str], min_hits: int = 3) -> float | None:
    bull = bear = 0
    for text in texts:
        negate = False
        for tok in _TOKEN.findall(text.lower()):
            if tok in NEGATIONS:
                negate = True
                continue
            if tok in STOCK_BULLISH:
                bull, bear = (bull, bear + 1) if negate else (bull + 1, bear)
            elif tok in STOCK_BEARISH:
                bull, bear = (bull + 1, bear) if negate else (bull, bear + 1)
            negate = False
    if bull + bear < min_hits:
        return None
    return (bull - bear) / (bull + bear)


class Research:
    FAILURE_TTL = 600

    def __init__(self, s, session: requests.Session | None = None, clock=time.time):
        self.s = s
        self.session = session or requests.Session()
        self.clock = clock
        self._cache: dict[str, tuple[float, object]] = {}
        self._sec_last = 0.0

    # ----------------------------------------------------------- plumbing
    def _cached(self, key: str, ttl: float, loader):
        now = self.clock()
        hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
        try:
            value = loader()
        except Exception as exc:
            log.warning("research source %s unavailable: %s", key, exc)
            value, ttl = None, self.FAILURE_TTL
        self._cache[key] = (now + ttl, value)
        return value

    def _get(self, url: str, params: dict | None = None, headers: dict | None = None):
        resp = self.session.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        return resp

    # ---------------------------------------------------------------- news
    def news_score(self, symbol: str, today: date) -> tuple[float | None, str]:
        if not self.s.sentiment_enabled:
            return None, "disabled"
        if self.s.alphavantage_api_key:
            score = self._cached(f"av-news:{symbol}:{today}", 6 * 3600,
                                 lambda: self._av_news(symbol, today))
            if score is not None:
                return score, "Alpha Vantage news sentiment"
        if self.s.finnhub_api_key:
            score = self._cached(f"fh-news:{symbol}:{today}", 6 * 3600,
                                 lambda: self._finnhub_news(symbol, today))
            if score is not None:
                return score, "Finnhub headlines (lexicon)"
        return None, "no news source"

    def _av_news(self, symbol: str, today: date) -> float | None:
        since = (today - timedelta(days=self.s.news_days)).strftime("%Y%m%dT0000")
        body = self._get(AV_URL, {"function": "NEWS_SENTIMENT", "tickers": symbol,
                                  "time_from": since, "sort": "LATEST", "limit": 200,
                                  "apikey": self.s.alphavantage_api_key}).json()
        if any(k in body for k in ("Information", "Note", "Error Message")):
            raise RuntimeError(str(body)[:150])  # throttled or premium-only
        weighted = total = 0.0
        for item in body.get("feed", []):
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker") == symbol:
                    rel = float(ts.get("relevance_score", 0) or 0)
                    weighted += rel * float(ts.get("ticker_sentiment_score", 0) or 0)
                    total += rel
        if total < 0.5:
            return None
        # AV scores live roughly in [-0.35, 0.35]: rescale to [-1, 1].
        return max(-1.0, min(1.0, weighted / total / 0.35))

    def _finnhub_news(self, symbol: str, today: date) -> float | None:
        rows = self._get(f"{FINNHUB_URL}/company-news",
                         {"symbol": symbol, "from": str(today - timedelta(days=self.s.news_days)),
                          "to": str(today), "token": self.s.finnhub_api_key}).json()
        texts = [f"{r.get('headline', '')}. {r.get('summary', '')}" for r in rows[:60]]
        return headline_score(texts)

    # ------------------------------------------------------------ attention
    def attention(self) -> dict[str, dict]:
        """ApeWisdom mentions: ticker -> {rank, mentions, mentions_24h_ago}."""
        def load():
            out = {}
            for page in (1, 2):
                body = self._get(APEWISDOM_URL.format(page=page)).json()
                for r in body.get("results", []):
                    out[str(r.get("ticker", "")).upper()] = {
                        "rank": _int(r.get("rank")),
                        "mentions": _int(r.get("mentions")),
                        "mentions_24h_ago": _int(r.get("mentions_24h_ago")),
                    }
            return out
        if not self.s.sentiment_enabled:
            return {}
        return self._cached("apewisdom", 1800, load) or {}

    def mania(self, symbol: str, rsi14: float) -> str | None:
        info = self.attention().get(symbol)
        if not info or info["mentions"] < 20:
            return None
        before = max(info["mentions_24h_ago"], 1)
        ratio = info["mentions"] / before
        if ratio >= self.s.mania_ratio and rsi14 >= self.s.mania_rsi:
            return (f"retail mania: Reddit mentions x{ratio:.1f} in 24h with RSI "
                    f"{rsi14:.0f}")
        return None

    # ------------------------------------------------------------- earnings
    def earnings_dates(self, today: date) -> dict[str, date]:
        def load_av():
            text = self._get(AV_URL, {"function": "EARNINGS_CALENDAR", "horizon": "3month",
                                      "apikey": self.s.alphavantage_api_key}).text
            df = pd.read_csv(io.StringIO(text))
            if "symbol" not in df.columns:
                raise RuntimeError(text[:150])
            out = {}
            for sym, day in zip(df["symbol"], df["reportDate"], strict=False):
                d = pd.to_datetime(day).date()
                if d >= today and (sym not in out or d < out[sym]):
                    out[str(sym).upper()] = d
            return out

        def load_fh():
            body = self._get(f"{FINNHUB_URL}/calendar/earnings",
                             {"from": str(today), "to": str(today + timedelta(days=60)),
                              "token": self.s.finnhub_api_key}).json()
            out = {}
            for row in body.get("earningsCalendar", []):
                d = pd.to_datetime(row["date"]).date()
                sym = str(row.get("symbol", "")).upper()
                if sym and (sym not in out or d < out[sym]):
                    out[sym] = d
            return out

        if self.s.alphavantage_api_key:
            got = self._cached(f"earn-av:{today}", 12 * 3600, load_av)
            if got is not None:
                return got
        if self.s.finnhub_api_key:
            return self._cached(f"earn-fh:{today}", 12 * 3600, load_fh) or {}
        return {}

    # --------------------------------------------------------------- insiders
    def _sec(self, url: str):
        if not self.s.sec_user_agent:
            raise RuntimeError("SEC_USER_AGENT not set")
        wait = self._sec_last + 0.15 - time.monotonic()  # SEC limit: 10 requests/s
        if wait > 0:
            time.sleep(wait)
        self._sec_last = time.monotonic()
        return self._get(url, headers={"User-Agent": self.s.sec_user_agent})

    def insider_buys(self, symbol: str, today: date, days: int = 30) -> int | None:
        """Number of open-market purchases (Form 4, code P) in the last ``days``."""
        if not (self.s.sentiment_enabled and self.s.sec_user_agent):
            return None

        def cik_map():
            data = self._sec(SEC_TICKERS).json()
            return {str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()}

        def load():
            ciks = self._cached("sec-ciks", 7 * 86400, cik_map) or {}
            cik = ciks.get(symbol.replace(".", "-")) or ciks.get(symbol)
            if not cik:
                return 0
            recent = self._sec(SEC_SUBMISSIONS.format(cik=cik)).json()["filings"]["recent"]
            since = today - timedelta(days=days)
            count = 0
            for form, filed, acc, doc in zip(recent["form"], recent["filingDate"],
                                             recent["accessionNumber"],
                                             recent["primaryDocument"], strict=False):
                if form != "4" or datetime.strptime(filed, "%Y-%m-%d").date() < since:
                    continue
                raw_doc = doc.split("/")[-1]  # strip the xslF345X0x/ rendering prefix
                xml = self._sec(SEC_ARCHIVE.format(cik=cik, acc=acc.replace("-", ""),
                                                   doc=raw_doc)).text
                count += count_open_market_buys(xml)
                if count >= 3:
                    break
            return count
        return self._cached(f"insider:{symbol}:{today}", 24 * 3600, load)

    # ---------------------------------------------------------------- summary
    def market_snapshot(self, regime: dict) -> dict:
        att = self.attention() if self.s.sentiment_enabled else {}
        top = sorted(att.items(), key=lambda kv: kv[1]["rank"] or 999)[:12]
        vix = regime.get("vix")
        return {
            "enabled": self.s.sentiment_enabled,
            "gauge": None if vix is None else {
                "value": vix, "decimals": 1, "label": "VIX",
                "classification": _vix_label(vix), "higher_is_bullish": False,
                "bull_below": 16, "bear_above": 25,
            },
            "trending": [t for t, _ in top],
            "trending_label": "Most discussed on Reddit (ApeWisdom)",
            "source_list": [
                {"name": "VIX (FRED)", "ok": vix is not None, "note": "unavailable"},
                {"name": "Reddit (ApeWisdom)", "ok": bool(att), "note": "unavailable"},
                {"name": "Alpha Vantage news", "ok": bool(self.s.alphavantage_api_key),
                 "note": "no key"},
                {"name": "Finnhub news", "ok": bool(self.s.finnhub_api_key), "note": "no key"},
                {"name": "SEC insiders", "ok": bool(self.s.sec_user_agent),
                 "note": "no SEC_USER_AGENT"},
            ],
            "regime": regime,
        }


def count_open_market_buys(xml_text: str) -> int:
    """Count non-derivative transactions with code P (open-market purchase)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return 0
    count = 0
    for tx in root.iter("nonDerivativeTransaction"):
        code = tx.findtext("transactionCoding/transactionCode", default="").strip()
        acq = tx.findtext("transactionAmounts/transactionAcquiredDisposedCode/value",
                          default="").strip()
        if code == "P" and acq in ("A", ""):
            count += 1
    return count


def _vix_label(v: float) -> str:
    if v < 16:
        return "calm"
    if v < 20:
        return "normal"
    if v < 25:
        return "nervous"
    if v < 30:
        return "fearful"
    return "panic"


def _int(x) -> int:
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0
