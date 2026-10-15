import json
from datetime import date

import pandas as pd
import pytest

from invest.config import LIVE_CONFIRMATION, ConfigError, InvestSettings
from invest.data import (
    AlpacaDataProvider,
    BarProvider,
    BarStore,
    MarketData,
    MassiveProvider,
    TiingoProvider,
    ecb_eurusd,
    fred_series,
)
from invest.sentiment import Research, count_open_market_buys, headline_score
from invest.tax import TaxLedger
from invest.universe import build_universe
from tests.invest_helpers import settings


class Resp:
    def __init__(self, payload=None, status=200, text=None):
        self.payload = payload
        self.status_code = status
        self.text = text if text is not None else json.dumps(payload)

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class Session:
    def __init__(self, handler):
        self.handler = handler
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, params, headers))
        return self.handler(url, params or {})


# ---------------------------------------------------------------- providers
def test_massive_history_and_grouped():
    ts = int(pd.Timestamp("2026-10-14 04:00", tz="America/New_York").timestamp() * 1000)

    def handler(url, params):
        if "/grouped/" in url:
            return Resp({"results": [{"T": "AMD", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 9,
                                      "t": ts}]})
        return Resp({"results": [{"o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 9, "t": ts}]})

    sleeps = []
    p = MassiveProvider("KEY", session=Session(handler), sleep=sleeps.append)
    assert p.session.headers["Authorization"] == "Bearer KEY"
    df = p.history("AMD", date(2026, 10, 1), date(2026, 10, 14))
    assert df.index[0] == pd.Timestamp("2026-10-14") and df["close"].iloc[0] == 1.5
    assert p.grouped(date(2026, 10, 14))["AMD"]["close"] == 1.5
    assert sleeps  # the free plan's 5 calls/min are respected
    url = p.session.calls[0][0]
    assert url.endswith("/v2/aggs/ticker/AMD/range/1/day/2026-10-01/2026-10-14")


def test_massive_rate_limit_retry():
    responses = [Resp(status=429, text="You've exceeded the maximum requests per minute"),
                 Resp({"results": []})]
    p = MassiveProvider("K", session=Session(lambda u, q: responses.pop(0)),
                        sleep=lambda _: None)
    assert p.history("AMD", date(2026, 1, 1), date(2026, 1, 2)).empty


def test_alpaca_data_pagination_and_iex_fallback():
    pages = [Resp(status=403, text="sip not allowed"),
             Resp({"bars": {"AMD": [{"t": "2026-10-13T04:00:00Z", "o": 1, "h": 2, "l": 1,
                                     "c": 1.5, "v": 10}]}, "next_page_token": "x"}),
             Resp({"bars": {"AMD": [{"t": "2026-10-14T04:00:00Z", "o": 1, "h": 2, "l": 1,
                                     "c": 1.6, "v": 10}]}, "next_page_token": None})]
    p = AlpacaDataProvider("k", "s", session=Session(lambda u, q: pages.pop(0)),
                           sleep=lambda _: None)
    df = p.history("AMD", date(2026, 10, 1), date(2026, 10, 14))
    assert list(df["close"]) == [1.5, 1.6] and p.feed == "iex"


def test_tiingo_adjusted_prices():
    rows = [{"date": "2026-10-14T00:00:00.000Z", "open": 10, "high": 11, "low": 9, "close": 10,
             "volume": 5, "adjOpen": 5, "adjHigh": 5.5, "adjLow": 4.5, "adjClose": 5,
             "adjVolume": 10}]
    p = TiingoProvider("k", session=Session(lambda u, q: Resp(rows)), sleep=lambda _: None)
    df = p.history("BRK.B", date(2026, 10, 1), date(2026, 10, 14))
    assert df["close"].iloc[0] == 5 and "BRK-B" in p.session.calls[0][0]


class FakeProvider(BarProvider):
    """Close 1.0 until 2026-10-12, then 2.0; ``split`` divides the past by 10."""

    name = "fake"
    supports_grouped = True

    def __init__(self, split=False):
        self.history_calls, self.grouped_calls = [], []
        self.split = split

    def _price(self, day):
        past = 1.0 / (10 if self.split else 1)
        return past if pd.Timestamp(day) <= pd.Timestamp("2026-10-12") else 2.0

    def history(self, symbol, start, end):
        self.history_calls.append(symbol)
        idx = pd.bdate_range(start, end)
        closes = [self._price(d) for d in idx]
        return pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes,
                             "volume": 1.0}, index=idx)

    def grouped(self, day):
        self.grouped_calls.append(day)
        c = self._price(day)
        return {"AMD": {"open": c, "high": c, "low": c, "close": c, "volume": 2}}


def test_barstore_full_then_incremental(tmp_path):
    p = FakeProvider()
    store = BarStore(tmp_path, p, history_days=60)
    frames = store.update(["AMD"], date(2026, 10, 12))
    assert p.history_calls == ["AMD"] and (tmp_path / "bars" / "AMD.csv").exists()
    frames = store.update(["AMD"], date(2026, 10, 14))
    assert p.history_calls == ["AMD"]  # no second full download
    # one overlap check on the last cached day, then the two new days
    assert p.grouped_calls == [date(2026, 10, 12), date(2026, 10, 13), date(2026, 10, 14)]
    assert frames["AMD"]["close"].iloc[-1] == 2 and store.splits == {}


def test_barstore_detects_split_and_reloads_history(tmp_path):
    p = FakeProvider()
    store = BarStore(tmp_path, p, history_days=60)
    store.update(["AMD"], date(2026, 10, 12))
    p.split = True  # the provider now reports past prices divided by 10
    frames = store.update(["AMD"], date(2026, 10, 14))
    assert p.history_calls == ["AMD", "AMD"]  # history downloaded again
    assert store.splits == {"AMD": ("2026-10-13", 10.0)}
    assert frames["AMD"]["close"].loc["2026-10-12"] == pytest.approx(0.1)


def test_fred_csv_both_header_styles_and_ecb():
    old = "DATE,VIXCLS\n2026-10-13,18.5\n2026-10-14,.\n"
    new = "observation_date,VIXCLS\n2026-10-13,18.5\n2026-10-14,19.0\n"
    s1 = fred_series("VIXCLS", date(2026, 10, 1), session=Session(lambda u, q: Resp(text=old)))
    s2 = fred_series("VIXCLS", date(2026, 10, 1), session=Session(lambda u, q: Resp(text=new)))
    assert list(s1) == [18.5] and list(s2) == [18.5, 19.0]
    csv = "KEY,FREQ,TIME_PERIOD,OBS_VALUE\nEXR.D,D,2026-10-13,1.16\nEXR.D,D,2026-10-14,1.17\n"
    fx = ecb_eurusd(date(2026, 10, 1), session=Session(lambda u, q: Resp(text=csv)))
    md = MarketData(bars={}, eurusd=fx)
    assert md.fx(pd.Timestamp("2026-10-15")) == 1.17
    assert MarketData(bars={}).fx(pd.Timestamp("2026-10-15")) == 1.10  # fallback


# ----------------------------------------------------------------- research
def test_headline_score():
    assert headline_score(["AMD beats estimates, raises guidance, upgrade"]) == 1.0
    assert headline_score(["Probe and lawsuit after earnings miss"]) == -1.0
    assert headline_score(["nothing"]) is None


FORM4 = """<ownershipDocument><nonDerivativeTable>
<nonDerivativeTransaction><transactionCoding><transactionCode>P</transactionCode>
</transactionCoding><transactionAmounts><transactionAcquiredDisposedCode><value>A</value>
</transactionAcquiredDisposedCode></transactionAmounts></nonDerivativeTransaction>
<nonDerivativeTransaction><transactionCoding><transactionCode>S</transactionCode>
</transactionCoding></nonDerivativeTransaction>
</nonDerivativeTable></ownershipDocument>"""


def test_form4_parsing():
    assert count_open_market_buys(FORM4) == 1
    assert count_open_market_buys("not xml") == 0


def research(handler, **kw):
    s = settings(sentiment_enabled=True, **kw)
    return Research(s, session=Session(handler), clock=lambda: 1000.0)


def test_news_av_then_finnhub_fallback():
    def handler(url, params):
        if "alphavantage" in url:
            return Resp({"Information": "rate limit"})
        return Resp([{"headline": "AMD downgrade after miss", "summary": "shares slump"}])

    r = research(handler, alphavantage_api_key="a", finnhub_api_key="f")
    score, source = r.news_score("AMD", date(2026, 10, 14))
    assert score == -1.0 and "Finnhub" in source


def test_av_news_scores():
    feed = {"feed": [{"ticker_sentiment": [
        {"ticker": "AMD", "relevance_score": "0.9", "ticker_sentiment_score": "0.35"},
        {"ticker": "NVDA", "relevance_score": "0.9", "ticker_sentiment_score": "-0.9"}]}]}
    r = research(lambda u, q: Resp(feed), alphavantage_api_key="a")
    assert r.news_score("AMD", date(2026, 10, 14))[0] == pytest.approx(1.0)


def test_mania_and_earnings():
    ape = {"results": [{"ticker": "GME", "rank": 1, "mentions": "500",
                        "mentions_24h_ago": "50"}]}
    csv = "symbol,name,reportDate,fiscalDateEnding,estimate,currency\n" \
          "AMD,AMD,2026-10-16,2026-09-30,1.0,USD\nAMD,AMD,2027-01-20,2026-12-31,1.1,USD\n"

    def handler(url, params):
        if "apewisdom" in url:
            return Resp(ape if "page/1" in url else {"results": []})
        return Resp(text=csv)

    r = research(handler, alphavantage_api_key="a")
    assert "mania" in r.mania("GME", 82.0)
    assert r.mania("GME", 60.0) is None and r.mania("AMD", 90.0) is None
    assert r.earnings_dates(date(2026, 10, 14))["AMD"] == date(2026, 10, 16)
    snap = r.market_snapshot({"vix": 14.0})
    assert snap["gauge"]["label"] == "VIX" and snap["trending"] == ["GME"]


def test_insider_buys_via_sec(monkeypatch):
    monkeypatch.setattr("invest.sentiment.time.sleep", lambda _: None)

    def handler(url, params):
        if url.endswith("company_tickers.json"):
            return Resp({"0": {"cik_str": 2488, "ticker": "AMD", "title": "AMD"}})
        if "submissions" in url:
            return Resp({"filings": {"recent": {
                "form": ["4", "4", "10-Q"], "filingDate": ["2026-10-10", "2026-10-01",
                                                          "2026-10-02"],
                "accessionNumber": ["0001-26-1", "0001-26-2", "0001-26-3"],
                "primaryDocument": ["xslF345X05/doc1.xml", "doc2.xml", "q.htm"]}}})
        return Resp(text=FORM4)

    r = research(handler, sec_user_agent="Test test@example.com")
    assert r.insider_buys("AMD", date(2026, 10, 14)) == 2
    urls = [c[0] for c in r.session.calls]
    assert any(u.endswith("/2488/0001261/doc1.xml") for u in urls)  # xsl prefix removed
    assert all(c[2] == {"User-Agent": "Test test@example.com"} for c in r.session.calls)


def test_research_failures_are_neutral():
    def boom(url, params):
        raise ConnectionError("down")

    r = research(boom, alphavantage_api_key="a", finnhub_api_key="f")
    assert r.news_score("AMD", date(2026, 10, 14))[0] is None
    assert r.attention() == {} and r.earnings_dates(date(2026, 10, 14)) == {}


# ------------------------------------------------------------- config & tax
def test_universe_excludes_crypto_and_us_etfs():
    u = build_universe()
    assert "MSTR" not in u and "COIN" not in u and "SPY" not in u
    assert len(u) == len(set(u)) > 100
    assert build_universe(["amd", "MSTR", "AMD"]) == ["AMD"]


def test_config_env_names_and_locks(monkeypatch, tmp_path):
    monkeypatch.setenv("INV_RISK_PER_TRADE", "0.05")
    monkeypatch.setenv("INV_SLOTS", "2")
    monkeypatch.setenv("FINNHUB_API_KEY", "fh")
    monkeypatch.delenv("LIVE_TRADING", raising=False)
    s = InvestSettings.from_env(str(tmp_path / "none"))
    assert s.risk_per_trade == 0.05 and s.slots == 2 and s.finnhub_api_key == "fh"
    assert s.public_dict()["finnhub_api_key"] is True
    with pytest.raises(ConfigError, match="T212_API_KEY"):
        InvestSettings(broker="trading212").validate()
    with pytest.raises(ConfigError, match="locked"):
        InvestSettings(broker="alpaca", environment="live", alpaca_api_key="k",
                       alpaca_api_secret="s").validate()
    with pytest.raises(ConfigError, match="written OK"):
        InvestSettings(broker="trading212", environment="live", t212_api_key="k",
                       t212_api_secret="s", live_trading=LIVE_CONFIRMATION).validate()
    ok = InvestSettings(broker="trading212", environment="live", t212_api_key="k",
                        t212_api_secret="s", live_trading=LIVE_CONFIRMATION,
                        t212_terms_acknowledged=True)
    ok.validate()
    ib = InvestSettings(broker="ibkr")
    ib.validate()
    assert ib.fractional is False and ib.fees.commission_min == 1.0
    with pytest.raises(ConfigError):
        InvestSettings(max_gross=1.5).validate()  # no margin, ever


def test_tax_summary(tmp_path):
    ledger = TaxLedger(tmp_path)
    ledger.record(day="2026-03-01", symbol="AMD", side="BUY", qty=1, price=110,
                  currency="USD", usd_per_eur=1.1, gross_eur=100, fees_eur=0.15)
    ledger.record(day="2026-04-01", symbol="AMD", side="SELL", qty=1, price=132,
                  currency="USD", usd_per_eur=1.1, gross_eur=120, fees_eur=0.18,
                  cost_basis_eur=100.15, gain_eur=19.67)
    ledger.record(day="2026-05-01", symbol="NVDA", side="SELL", qty=1, price=90,
                  currency="USD", usd_per_eur=1.1, gross_eur=50, fees_eur=0.1,
                  cost_basis_eur=55, gain_eur=-5.1)
    summary = ledger.summary(2026)
    assert summary["net_eur"] == pytest.approx(14.57)
    assert summary["box"] == "3VG"
    assert summary["estimated_pfu_eur"] == pytest.approx(14.57 * 0.314, abs=0.01)
    assert ledger.summary(2025)["sales"] == 0
    out = ledger.export_csv(2026, tmp_path / "t.csv")
    assert out.read_text().startswith("day,")
