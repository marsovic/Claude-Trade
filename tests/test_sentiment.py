import pytest
import requests

from bot.config import Settings
from bot.sentiment import (
    Reading,
    SentimentEngine,
    combine,
    lexicon_counts,
    lexicon_score,
    score_cryptopanic_posts,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        for prefix, payload in self.routes.items():
            if url.startswith(prefix):
                if isinstance(payload, Exception):
                    raise payload
                return FakeResponse(payload)
        return FakeResponse({}, 404)


FNG = {"data": [{"value": "20", "value_classification": "Extreme Fear"}]}
TRENDING = {"coins": [
    {"item": {"symbol": "sol", "name": "Solana",
              "data": {"price_change_percentage_24h": {"usd": 6.5}}}},
    {"item": {"symbol": "pepe", "name": "Pepe",
              "data": {"price_change_percentage_24h": {"usd": -4.0}}}},
]}


def test_lexicon_with_negation():
    assert lexicon_counts("BTC breakout, very bullish!") == (2, 0)
    assert lexicon_counts("this is not bullish") == (0, 1)
    assert lexicon_counts("exchange hacked, massive dump") == (0, 2)
    assert lexicon_score(["moon pump rally", "crash"]) == pytest.approx(0.5)
    assert lexicon_score(["nothing to see"]) is None


def test_combine_weighted_mean_ignores_missing():
    readings = [Reading("a", 1.0, 1.0), Reading("b", -0.5, 2.0), Reading("c", None, 5.0)]
    assert combine(readings) == pytest.approx(0.0)
    assert combine([Reading("a", None, 1.0)]) is None


def test_fear_greed_and_trending():
    s = Settings()
    session = FakeSession({"https://api.alternative.me": FNG,
                           "https://api.coingecko.com": TRENDING})
    engine = SentimentEngine(s, session=session)
    res = engine.evaluate("SOL")
    by_source = {r.source: r for r in res.readings}
    assert by_source["fear_greed"].score == pytest.approx(-0.6)
    assert by_source["coingecko_trending"].score == pytest.approx(0.5)
    assert by_source["cryptopanic"].score is None  # no token configured
    assert by_source["x"].score is None
    # (-0.6 * 1 + 0.5 * 0.5) / 1.5
    assert res.score == pytest.approx(-0.35 / 1.5)
    assert engine.evaluate("PEPE").readings[1].score == pytest.approx(-0.5)
    assert engine.evaluate("BTC").readings[1].score is None


def test_results_are_cached():
    s = Settings()
    session = FakeSession({"https://api.alternative.me": FNG,
                           "https://api.coingecko.com": TRENDING})
    engine = SentimentEngine(s, session=session, clock=lambda: 1000.0)
    engine.evaluate("SOL")
    engine.evaluate("ETH")
    assert len(session.calls) == 2  # one call per source, then cache


def test_failures_fall_back_to_no_opinion():
    s = Settings()
    session = FakeSession({"https://api.alternative.me": requests.ConnectionError("down"),
                           "https://api.coingecko.com": requests.Timeout("slow")})
    engine = SentimentEngine(s, session=session)
    res = engine.evaluate("SOL")
    assert res.score is None
    assert engine.allows("LONG", res) == (True, "no sentiment data (neutral)")
    assert engine.market_snapshot()["fear_greed"] is None


def test_veto_logic():
    s = Settings(sentiment_veto=0.25)
    engine = SentimentEngine(s, session=FakeSession({}))
    from bot.sentiment import SentimentResult

    bearish = SentimentResult("X", -0.5)
    bullish = SentimentResult("X", 0.5)
    assert engine.allows("LONG", bearish)[0] is False
    assert engine.allows("SHORT", bearish)[0] is True
    assert engine.allows("SHORT", bullish)[0] is False
    assert engine.allows("LONG", bullish)[0] is True
    s.sentiment_enabled = False
    assert engine.allows("LONG", bearish)[0] is True


def test_cryptopanic_and_x_with_tokens():
    s = Settings(cryptopanic_token="tok", x_bearer_token="bearer")
    posts = {"results": [
        {"title": "SOL rally continues", "votes": {"positive": 8, "negative": 2}},
        {"title": "Solana network outage", "votes": {}},
        {"title": "Neutral headline", "votes": {"liked": 3}},
    ]}
    tweets = {"data": [{"text": "SOL breakout, bullish"}, {"text": "buying more SOL, moon"},
                       {"text": "might dump"}]}
    session = FakeSession({"https://api.alternative.me": FNG,
                           "https://api.coingecko.com": TRENDING,
                           s.cryptopanic_url: posts, s.x_api_url: tweets})
    engine = SentimentEngine(s, session=session)
    res = engine.evaluate("SOL")
    by_source = {r.source: r for r in res.readings}
    assert by_source["cryptopanic"].score == pytest.approx((0.6 - 1.0 + 1.0) / 3)
    assert by_source["x"].score == pytest.approx((4 - 1) / 5)
    x_call = [c for c in session.calls if c[0] == s.x_api_url][0]
    assert x_call[2]["Authorization"] == "Bearer bearer"
    assert "$SOL" in x_call[1]["query"]


def test_cryptopanic_explicit_sentiment_field():
    score, n = score_cryptopanic_posts([{"sentiment": "bullish"}, {"sentiment": "bearish"},
                                        {"sentiment": "bullish"}])
    assert n == 3 and score == pytest.approx(1 / 3)
