"""The insider strategy: live Form 4 feed, entries, vetoes, time exits, bot loop."""

from datetime import date, datetime

import pandas as pd
import pytest

from bot.state import StateStore
from invest.config import ConfigError
from invest.data import ET
from invest.edgar import FeedNotReady, InsiderFeed, parse_daily_index, parse_form4
from invest.execution import SimExecution
from invest.insider import InsiderEngine, matching_rules
from invest.invest_bot import InvestBot
from invest.portfolio import InvestState
from invest.tax import TaxLedger
from tests.invest_helpers import make_market, market_data, settings
from tests.test_invest_live import Clock, fake_loader

UA = "Jane Doe jane@example.com"

def _idx(form, name, cik, acc):
    return f"{form:<17}{name:<62}{cik:<12}20261016    edgar/data/{cik}/{acc}.txt"


INDEX = "\n".join([
    "Description:           Daily Index of EDGAR Dissemination Feed by Form Type",
    "Form Type   Company Name                    CIK         Date Filed  File Name",
    "-" * 100,
    _idx("3", "SOMEONE", 1111, "0001-26-000001"),
    _idx("4", "DOE JANE", 2222, "0002-26-000002"),
    _idx("4", "ACME CORP", 3333, "0002-26-000002"),
    _idx("4/A", "ACME CORP", 3333, "0002-26-000003"),
    _idx("SC 13G", "ACME CORP", 3333, "0002-26-000004"),
]) + "\n"


def form4(code="P", shares="1000", price="60", title="Common Stock", officer="1",
          director="0", officer_title="Chief Executive Officer", plan="0", after="5000",
          symbol="AAA"):
    return f"""<SEC-DOCUMENT>header
<XML>
<?xml version="1.0"?>
<ownershipDocument>
  <documentType>4</documentType>
  <aff10b5One>{plan}</aff10b5One>
  <issuer><issuerCik>0000003333</issuerCik>
    <issuerTradingSymbol>{symbol}</issuerTradingSymbol></issuer>
  <reportingOwner>
    <reportingOwnerId><rptOwnerCik>0000002222</rptOwnerCik>
      <rptOwnerName>Doe Jane</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><isDirector>{director}</isDirector><isOfficer>{officer}</isOfficer>
      <officerTitle>{officer_title}</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable><nonDerivativeTransaction>
    <securityTitle><value>{title}</value></securityTitle>
    <transactionDate><value>2026-10-15</value></transactionDate>
    <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
    <transactionAmounts><transactionShares><value>{shares}</value></transactionShares>
      <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
      <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode></transactionAmounts>
    <postTransactionAmounts><sharesOwnedFollowingTransaction><value>{after}</value></sharesOwnedFollowingTransaction></postTransactionAmounts>
    <ownershipNature><directOrIndirectOwnership><value>D</value></directOrIndirectOwnership></ownershipNature>
  </nonDerivativeTransaction></nonDerivativeTable>
</ownershipDocument>
</XML>"""


# ========================================================================= feed
def test_daily_index_keeps_each_form4_once():
    rows = parse_daily_index(INDEX)
    assert [r["accession"] for r in rows] == ["0002-26-000002"]  # no 3, 4/A or 13G; no dup
    assert rows[0]["filed"] == "2026-10-16"


def test_form4_purchase_is_parsed_like_the_bulk_data():
    row = parse_form4(form4(), "acc", "2026-10-16")
    assert row["symbol"] == "AAA" and row["cik"] == 3333 and row["role"] == "ceo"
    assert row["value"] == 60_000 and row["price"] == 60 and row["traded"] == "2026-10-15"
    assert row["delta_own"] == pytest.approx(0.25) and not row["plan"]
    assert parse_form4(form4(code="S"), "a", "2026-10-16") is None  # a sale
    assert parse_form4(form4(title="Series A Preferred"), "a", "2026-10-16") is None
    assert parse_form4(form4(plan="1"), "a", "2026-10-16")["plan"]
    assert parse_form4("no xml here", "a", "2026-10-16") is None


def test_rules_match_the_study():
    base = {"plan": False, "officer": True, "director": False, "role": "ceo",
            "value": 60_000, "delta_own": 0.05}
    assert matching_rules(base, ["ceo_cfo", "conviction", "buy"]) == ["ceo_cfo", "buy"]
    assert matching_rules({**base, "value": 150_000, "delta_own": 0.3},
                          ["conviction", "buy"]) == ["conviction", "buy"]
    assert matching_rules({**base, "plan": True}, ["ceo_cfo", "buy"]) == []
    assert matching_rules({**base, "officer": False, "role": "owner10"}, ["buy"]) == []
    assert matching_rules({**base, "value": 20_000}, ["ceo_cfo", "buy"]) == []


class FakeEdgar:
    def __init__(self, tmp_path, pages=None, eight_ks=None):
        self.dir = tmp_path
        self.pages = pages or {}
        self.calls = []
        self.filings = eight_ks if eight_ks is not None else pd.DataFrame(
            columns=["cik", "accession", "filed", "items", "raw_time", "accepted_et"])

    def get(self, url, allow_404=False):
        self.calls.append(url)
        for key, text in self.pages.items():
            if key in url:
                return type("R", (), {"text": text, "status_code": 200})()
        return None

    def eight_ks(self, cik, since, until=None, max_age_days=7):
        return self.filings


def test_feed_reads_a_day_once_and_waits_for_the_index(tmp_path):
    e = FakeEdgar(tmp_path, {"form.20261016.idx": INDEX, "0002-26-000002.txt": form4()})
    feed = InsiderFeed(e)
    df = feed.purchases_on(date(2026, 10, 16))
    assert list(df["symbol"]) == ["AAA"]
    n = len(e.calls)
    feed.purchases_on(date(2026, 10, 16))
    assert len(e.calls) == n  # cached
    with pytest.raises(FeedNotReady):
        feed.purchases_on(date(2026, 10, 19))


# ======================================================================= engine
def market(n_days=120):
    bars = make_market(n_days=n_days, n_syms=4)
    for sym in list(bars):
        if sym not in ("SPY", "QQQ"):
            bars[sym] = bars[sym].assign(volume=1e6)  # ~ $50-100M a day
    return bars


class StubFeed:
    def __init__(self, rows):
        self.rows = rows

    def purchases_on(self, day):
        return pd.DataFrame([r for r in self.rows if r["filed"] == str(day)])


def buy(sym, day, bars, **kw):
    px = float(bars[sym].loc[day, "close"])
    row = {"accession": f"{sym}-{day.date()}", "filed": str(day.date()),
           "traded": str(day.date()), "cik": 7, "symbol": sym, "owner_cik": "9",
           "owner": "Doe Jane", "role": "ceo", "officer": True, "director": False,
           "owner10": False, "shares": 1000, "value": 80_000.0, "price": px,
           "owned_after": 5000, "delta_own": 0.25, "plan": False}
    row.update(kw)
    return row


def insider_engine(tmp_path, rows, eight_ks=None, **kw):
    s = settings(state_dir=str(tmp_path), strategy="insider", sec_user_agent=UA, **kw)
    state = InvestState.fresh("sim", s.initial_capital)
    store = StateStore(tmp_path, InvestState)
    engine = InsiderEngine(s, state, store, SimExecution(s, state, s.fees), StubFeed(rows),
                           FakeEdgar(tmp_path, eight_ks=eight_ks), ledger=TaxLedger(tmp_path))
    return engine, state


def run_day(engine, bars, day):
    engine.prepare(day.date(), datetime.combine(day.date(), datetime.min.time()).replace(
        hour=23, tzinfo=ET))
    return engine.process_day(day, market_data(bars))


def test_insider_buy_is_bought_next_open_and_sold_after_20_sessions(tmp_path):
    bars = market()
    sym = [k for k in bars if k not in ("SPY", "QQQ")][0]
    idx = bars["SPY"].index
    d0 = idx[60]
    engine, state = insider_engine(tmp_path, [buy(sym, d0, bars)])
    out = run_day(engine, bars, d0)
    assert out["entries"] == [sym]
    (order,) = state.pending
    assert order["limit_price"] == pytest.approx(bars[sym].loc[d0, "close"] * 1.08)
    run_day(engine, bars, idx[61])  # filled at the open of the next session
    pos = state.positions[sym]
    assert pos.entry_day == str(idx[61].date())
    assert pos.stop_loss == pytest.approx(pos.entry_price * 0.65)
    assert pos.cost_eur == pytest.approx(25.0, rel=0.05)  # 100 EUR / 4 slots
    for k in range(62, 80):
        run_day(engine, bars, idx[k])
    assert sym in state.positions and not state.pending  # 19 sessions: still held
    out = run_day(engine, bars, idx[80])
    assert out["exits"] == [sym]  # sold at the open after 20 sessions
    run_day(engine, bars, idx[81])
    assert sym not in state.positions
    assert state.insider_scanned == str(idx[81].date())


def test_checks_refuse_bad_candidates(tmp_path):
    bars = market()
    a, b, c, d = [k for k in bars if k not in ("SPY", "QQQ")][:4]
    day = bars["SPY"].index[60]
    bars[b] = bars[b].assign(volume=100.0)  # illiquid
    rows = [buy(a, day, bars, price=float(bars[a].loc[day, "close"]) * 3),  # reused ticker
            buy(b, day, bars), buy(c, day, bars, plan=True),  # 10b5-1: no rule
            buy(d, day, bars, value=30_000.0, role="director", officer=False, director=True)]
    engine, state = insider_engine(tmp_path, rows)
    out = run_day(engine, bars, day)
    assert out["entries"] == [d]  # only the director's 30 k$ buy passes ("buy" rule)
    skips = [x["message"] for x in engine.store.recent_decisions() if x["kind"] == "skip"]
    assert any("far from the market" in m for m in skips)
    assert any("M$ a day" in m for m in skips)
    scan = state.last_scan
    assert scan["conditions"][0] == "rule" and scan["signals"] == [d]


def test_validated_vetoes_block_an_entry(tmp_path):
    bars = market()
    sym = [k for k in bars if k not in ("SPY", "QQQ")][0]
    idx = bars["SPY"].index
    day = idx[60]
    contract = pd.DataFrame({"cik": [7], "accession": ["x"], "filed": [str(idx[55].date())],
                             "items": ["1.01,9.01"], "raw_time": [""],
                             "accepted_et": [datetime.combine(idx[55].date(),
                                                              datetime.min.time()).replace(
                                 hour=17)]})
    engine, state = insider_engine(tmp_path, [buy(sym, day, bars)], eight_ks=contract)
    assert run_day(engine, bars, day)["entries"] == []
    vetoes = [x["message"] for x in engine.store.recent_decisions() if x["kind"] == "skip"]
    assert any("8-K 1.01" in m for m in vetoes)
    # an old contract (30 sessions ago) no longer vetoes
    contract.loc[0, "accepted_et"] = datetime.combine(idx[25].date(),
                                                      datetime.min.time()).replace(hour=17)
    engine2, _ = insider_engine(tmp_path / "b", [buy(sym, day, bars)], eight_ks=contract)
    assert run_day(engine2, bars, day)["entries"] == [sym]


def test_earnings_drop_vetoes(tmp_path):
    bars = market()
    sym = [k for k in bars if k not in ("SPY", "QQQ")][0]
    idx = bars["SPY"].index
    df = bars[sym].copy()
    df.loc[idx[56]:, ["open", "high", "low", "close"]] *= 0.85  # -15 % on results day
    bars[sym] = df
    results = pd.DataFrame({"cik": [7], "accession": ["r"], "filed": [str(idx[56].date())],
                            "items": ["2.02,9.01"], "raw_time": [""],
                            "accepted_et": [datetime.combine(idx[56].date(),
                                                             datetime.min.time()).replace(
                                hour=7)]})
    engine, _ = insider_engine(tmp_path, [buy(sym, idx[60], bars)], eight_ks=results)
    assert run_day(engine, bars, idx[60])["entries"] == []


def test_slots_are_limited(tmp_path):
    bars = market()
    syms = [k for k in bars if k not in ("SPY", "QQQ")][:4]
    day = bars["SPY"].index[60]
    engine, state = insider_engine(tmp_path, [buy(x, day, bars) for x in syms],
                                   insider_slots=2)
    assert len(run_day(engine, bars, day)["entries"]) == 2


def test_insider_strategy_needs_the_sec_contact():
    with pytest.raises(ConfigError, match="SEC_USER_AGENT"):
        settings(strategy="insider", sec_user_agent="")


# ========================================================================== bot
def test_bot_waits_for_the_sec_index_then_trades(tmp_path, monkeypatch):
    bars = market()
    sym = [k for k in bars if k not in ("SPY", "QQQ")][0]
    day = bars["SPY"].index[60]
    rows = [buy(sym, day, bars)]
    ready = {"ok": False}

    class Feed(StubFeed):
        def purchases_on(self, d):
            if not ready["ok"]:
                raise FeedNotReady(f"SEC daily index of {d} not published yet")
            return super().purchases_on(d)

    monkeypatch.setattr("invest.invest_bot.InsiderFeed", lambda e: Feed(rows))
    s = settings(state_dir=str(tmp_path), strategy="insider", sec_user_agent=UA,
                 data_dir=str(tmp_path / "data"))
    clock = Clock(datetime.combine(day.date(), datetime.min.time()).replace(hour=23,
                                                                            tzinfo=ET))
    bot = InvestBot(s, data_loader=fake_loader(bars), clock=clock)
    bot.setup()
    bot.engine.edgar = FakeEdgar(tmp_path)
    assert bot.daily_job() is False  # index not published: retry later, no crash
    assert sum(d["kind"] == "info" for d in bot.store.recent_decisions()) == 1
    assert bot.daily_job() is False
    assert sum(d["kind"] == "info" for d in bot.store.recent_decisions()) == 1  # once
    ready["ok"] = True
    assert bot.daily_job() is True
    assert [o["symbol"] for o in bot.state.pending] == [sym]
    assert bot.state.strategy == "insider" and bot.state.last_day == str(day.date())
