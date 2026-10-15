"""Market sentiment used as a veto on technical signals.

Sources (each optional, each failing safely to "no opinion"):

* Fear & Greed index (alternative.me, free, no key) - market-wide mood
* CoinGecko trending coins (free, optional demo key)  - attention on an asset
* CryptoPanic news (needs ``CRYPTOPANIC_TOKEN``)      - community votes on news
* X / Twitter recent search (needs ``X_BEARER_TOKEN``) - lexicon score of posts

Every source yields a score in [-1, 1]. The weighted mean of the available
scores is compared to ``SENTIMENT_VETO``: a LONG is refused when sentiment is
clearly bearish, a SHORT when it is clearly bullish. Sentiment never opens a
trade on its own; it only blocks trades that go against the crowd's mood.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

import requests

from bot.strategy import LONG

log = logging.getLogger(__name__)

FEAR_GREED_URL = "https://api.alternative.me/fng/"
COINGECKO_TRENDING_URL = "https://api.coingecko.com/api/v3/search/trending"

BULLISH = {
    "bull", "bullish", "moon", "mooning", "pump", "pumping", "breakout", "rally",
    "rallies", "buy", "buying", "long", "longs", "ath", "surge", "surges", "soar",
    "soars", "gain", "gains", "green", "accumulate", "accumulating", "uptrend",
    "upgrade", "adoption", "approval", "approved", "partnership", "listing", "launch",
    "record", "higher", "rebound", "recovery", "support", "inflows", "outperform",
    "bullrun", "undervalued", "rocket", "parabolic", "etf",
}
BEARISH = {
    "bear", "bearish", "dump", "dumping", "crash", "crashes", "sell", "selling",
    "short", "shorts", "rekt", "scam", "hack", "hacked", "exploit", "plunge", "plunges",
    "drop", "drops", "fall", "falls", "red", "liquidation", "liquidations", "downtrend",
    "rug", "rugpull", "lawsuit", "sued", "delist", "delisting", "ban", "banned", "outage",
    "fud", "collapse", "bankrupt", "bankruptcy", "lower", "outflows", "overvalued",
    "fraud", "investigation", "warning", "selloff", "capitulation", "tank", "tanks",
}
NEGATIONS = {"not", "no", "never", "isnt", "isn't", "dont", "don't", "wont", "won't",
             "without", "aint", "ain't"}
_TOKEN = re.compile(r"[a-z']+")


def lexicon_counts(text: str) -> tuple[int, int]:
    """Count bullish / bearish words; a preceding negation flips the word."""
    bull = bear = 0
    negate = False
    for token in _TOKEN.findall(text.lower()):
        if token in NEGATIONS:
            negate = True
            continue
        if token in BULLISH:
            if negate:
                bear += 1
            else:
                bull += 1
        elif token in BEARISH:
            if negate:
                bull += 1
            else:
                bear += 1
        negate = False
    return bull, bear


def lexicon_score(texts: list[str], min_hits: int = 3) -> float | None:
    bull = bear = 0
    for text in texts:
        b, s = lexicon_counts(text)
        bull += b
        bear += s
    if bull + bear < min_hits:
        return None
    return (bull - bear) / (bull + bear)


@dataclass
class Reading:
    source: str
    score: float | None
    weight: float
    detail: str = ""

    def to_dict(self) -> dict:
        return {"source": self.source,
                "score": None if self.score is None else round(self.score, 3),
                "weight": self.weight, "detail": self.detail}


@dataclass
class SentimentResult:
    asset: str
    score: float | None
    readings: list[Reading] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.score is None:
            return "n/a"
        if self.score >= 0.25:
            return "bullish"
        if self.score <= -0.25:
            return "bearish"
        return "neutral"

    def to_dict(self) -> dict:
        return {"asset": self.asset,
                "score": None if self.score is None else round(self.score, 3),
                "label": self.label,
                "readings": [r.to_dict() for r in self.readings]}


def combine(readings: list[Reading]) -> float | None:
    usable = [r for r in readings if r.score is not None and r.weight > 0]
    total = sum(r.weight for r in usable)
    if not usable or total <= 0:
        return None
    return sum(r.score * r.weight for r in usable) / total


class SentimentEngine:
    FAILURE_TTL = 300  # seconds before retrying a source that failed

    def __init__(self, settings, session: requests.Session | None = None, clock=time.time):
        self.s = settings
        self.session = session or requests.Session()
        self.clock = clock
        self._cache: dict[str, tuple[float, object]] = {}

    # ------------------------------------------------------------- plumbing
    def _cached(self, key: str, loader):
        now = self.clock()
        hit = self._cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
        try:
            value = loader()
            ttl = self.s.sentiment_cache_minutes * 60
        except Exception as exc:  # network, JSON, HTTP errors: no opinion
            log.warning("sentiment source %s unavailable: %s", key, exc)
            value = None
            ttl = self.FAILURE_TTL
        self._cache[key] = (now + ttl, value)
        return value

    def _get_json(self, url: str, params: dict | None = None, headers: dict | None = None):
        resp = self.session.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # --------------------------------------------------------------- sources
    def fear_greed_raw(self) -> dict | None:
        def load():
            data = self._get_json(FEAR_GREED_URL, {"limit": 1})["data"][0]
            return {"value": int(data["value"]),
                    "classification": data.get("value_classification", "")}
        return self._cached("fear_greed", load)

    def trending_raw(self) -> list[dict] | None:
        def load():
            headers = {"x-cg-demo-api-key": self.s.coingecko_api_key} \
                if self.s.coingecko_api_key else None
            coins = self._get_json(COINGECKO_TRENDING_URL, headers=headers).get("coins", [])
            out = []
            for coin in coins:
                item = coin.get("item", {})
                change = (item.get("data") or {}).get("price_change_percentage_24h", {})
                out.append({
                    "symbol": str(item.get("symbol", "")).upper(),
                    "name": str(item.get("name", "")),
                    "change_24h": change.get("usd") if isinstance(change, dict) else None,
                })
            return out
        return self._cached("trending", load)

    def fear_greed(self) -> Reading:
        raw = self.fear_greed_raw()
        if raw is None:
            return Reading("fear_greed", None, self.s.weight_fear_greed, "unavailable")
        score = (raw["value"] - 50) / 50
        return Reading("fear_greed", score, self.s.weight_fear_greed,
                       f"{raw['value']} ({raw['classification']})")

    def trending(self, asset: str) -> Reading:
        coins = self.trending_raw()
        if coins is None:
            return Reading("coingecko_trending", None, self.s.weight_trending, "unavailable")
        for coin in coins:
            if coin["symbol"] == asset:
                change = coin.get("change_24h")
                if change is None:
                    return Reading("coingecko_trending", 0.25, self.s.weight_trending,
                                   "trending")
                score = 0.5 if change >= 0 else -0.5
                return Reading("coingecko_trending", score, self.s.weight_trending,
                               f"trending, {change:+.1f}% 24h")
        return Reading("coingecko_trending", None, self.s.weight_trending, "not trending")

    def cryptopanic(self, asset: str) -> Reading:
        weight = self.s.weight_cryptopanic
        if not self.s.cryptopanic_token:
            return Reading("cryptopanic", None, weight, "no token")

        def load():
            body = self._get_json(self.s.cryptopanic_url, {
                "auth_token": self.s.cryptopanic_token, "currencies": asset,
                "kind": "news", "public": "true",
            })
            return score_cryptopanic_posts(body.get("results", [])[:30])
        result = self._cached(f"cryptopanic:{asset}", load)
        if result is None:
            return Reading("cryptopanic", None, weight, "unavailable")
        score, n = result
        return Reading("cryptopanic", score, weight, f"{n} posts scored")

    def x_social(self, asset: str) -> Reading:
        weight = self.s.weight_x
        if not self.s.x_bearer_token:
            return Reading("x", None, weight, "no token")

        def load():
            body = self._get_json(
                self.s.x_api_url,
                {"query": f"(${asset} OR #{asset}) -is:retweet lang:en", "max_results": 50},
                {"Authorization": f"Bearer {self.s.x_bearer_token}"},
            )
            texts = [t.get("text", "") for t in body.get("data", []) or []]
            return lexicon_score(texts, min_hits=5), len(texts)
        result = self._cached(f"x:{asset}", load)
        if result is None:
            return Reading("x", None, weight, "unavailable")
        score, n = result
        return Reading("x", score, weight, f"{n} posts")

    # ---------------------------------------------------------------- public
    def evaluate(self, asset: str) -> SentimentResult:
        readings = [self.fear_greed(), self.trending(asset),
                    self.cryptopanic(asset), self.x_social(asset)]
        return SentimentResult(asset=asset, score=combine(readings), readings=readings)

    def allows(self, side: str, result: SentimentResult) -> tuple[bool, str]:
        if not self.s.sentiment_enabled:
            return True, "sentiment disabled"
        if result.score is None:
            return True, "no sentiment data (neutral)"
        if side == LONG and result.score < -self.s.sentiment_veto:
            return False, f"bearish sentiment {result.score:+.2f} vetoes LONG"
        if side != LONG and result.score > self.s.sentiment_veto:
            return False, f"bullish sentiment {result.score:+.2f} vetoes SHORT"
        return True, f"sentiment {result.score:+.2f} compatible"

    def market_snapshot(self) -> dict:
        """Market-wide view for the dashboard."""
        if not self.s.sentiment_enabled:
            return {"enabled": False}
        fng = self.fear_greed_raw()
        trending = self.trending_raw() or []
        return {
            "enabled": True,
            "fear_greed": fng,
            "trending": [c["symbol"] for c in trending[:15]],
            "sources": {
                "cryptopanic": bool(self.s.cryptopanic_token),
                "x": bool(self.s.x_bearer_token),
            },
        }


def score_cryptopanic_posts(posts: list[dict]) -> tuple[float | None, int]:
    """Votes when available, otherwise a lexicon score of the headline."""
    scores = []
    for post in posts:
        sentiment = str(post.get("sentiment", "")).lower()
        if sentiment in ("bullish", "positive"):
            scores.append(1.0)
            continue
        if sentiment in ("bearish", "negative"):
            scores.append(-1.0)
            continue
        votes = post.get("votes") or {}
        pos = votes.get("positive", 0) + votes.get("liked", 0)
        neg = votes.get("negative", 0) + votes.get("disliked", 0) + votes.get("toxic", 0)
        if pos + neg > 0:
            scores.append((pos - neg) / (pos + neg))
            continue
        bull, bear = lexicon_counts(post.get("title", ""))
        if bull + bear:
            scores.append((bull - bear) / (bull + bear))
    if len(scores) < 2:
        return None, len(scores)
    return sum(scores) / len(scores), len(scores)
