"""SEC EDGAR parsing, attention collection and the announcement event study."""

import io
import json
import zipfile
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from invest import attention, edgar, events
from invest.edgar import (
    Edgar,
    EdgarError,
    accepted_et,
    clean_symbol,
    parse_form345,
    role_of,
    symbol_resolver,
)
from invest.events import (
    Funnel,
    build_bench,
    eightk_events,
    first_open_after,
    insider_events,
    measure,
    panel_from_frames,
    price_matches,
    reaction_session,
    session_after,
)

UA = "Jane Doe jane@example.com"


# ====================================================================== helpers
class Resp:
    def __init__(self, status=200, body=None, content=b"", text=""):
        self.status_code = status
        self._body = body
        self.content = content
        self.text = text or (json.dumps(body) if body is not None else "")

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, routes):
        self.routes = routes  # list of (substring, response or callable)
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        for key, resp in self.routes:
            if key in url:
                return resp(url, params) if callable(resp) else resp
        return Resp(404)


def market(n_days=320, n_stocks=60, seed=0, start="2022-01-03"):
    """Random-walk stocks, liquid and above $2."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    frames = {}
    for k in range(n_stocks):
        r = rng.normal(0.0003, 0.02, n_days)
        close = 20 * np.exp(np.cumsum(r))
        opn = close * np.exp(rng.normal(0, 0.003, n_days))
        frames[f"S{k:02d}"] = pd.DataFrame(
            {"open": opn, "high": np.maximum(opn, close) * 1.01,
             "low": np.minimum(opn, close) * 0.99, "close": close,
             "volume": np.full(n_days, 2e5 * (1 + k))}, index=idx)
    return frames


def boost(frames, sym, t0, gain=0.10, days=20):
    """From the CLOSE of session t0 the stock drifts up by ``gain`` over ``days`` sessions;
    the open of t0 is untouched (the buyer pays the pre-news price)."""
    df = frames[sym].copy()
    n = len(df)
    factor = np.ones(n)
    for k in range(t0, n):
        factor[k] = 1 + gain * min(k - t0 + 1, days) / days
    for c in ("high", "low", "close"):
        df[c] = df[c] * factor
    nxt = np.ones(n)
    nxt[t0 + 1:] = factor[t0:-1]
    df["open"] = df["open"] * nxt
    frames[sym] = df
    return frames


def form345_zip(sub_rows, tx_rows, own_rows):
    def tsv(cols, rows):
        return "\t".join(cols) + "\n" + "".join("\t".join(str(r.get(c, "")) for c in cols)
                                                 + "\n" for r in rows)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("SUBMISSION.tsv", tsv(["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE",
                                          "ISSUERCIK", "ISSUERTRADINGSYMBOL", "AFF10B5ONE"],
                                         sub_rows))
        z.writestr("NONDERIV_TRANS.tsv", tsv(
            ["ACCESSION_NUMBER", "SECURITY_TITLE", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES",
             "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD", "SHRS_OWND_FOLWNG_TRANS",
             "DIRECT_INDIRECT_OWNERSHIP"], tx_rows))
        z.writestr("REPORTINGOWNER.tsv", tsv(["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME",
                                              "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"],
                                             own_rows))
    return buf.getvalue()


def sample_zip():
    sub = [
        {"ACCESSION_NUMBER": "A1", "FILING_DATE": "05-FEB-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "0000000101", "ISSUERTRADINGSYMBOL": "abc", "AFF10B5ONE": "0"},
        {"ACCESSION_NUMBER": "A2", "FILING_DATE": "06-FEB-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "101", "ISSUERTRADINGSYMBOL": "ABC", "AFF10B5ONE": "1"},
        {"ACCESSION_NUMBER": "A3", "FILING_DATE": "07-FEB-2024", "DOCUMENT_TYPE": "4/A",
         "ISSUERCIK": "101", "ISSUERTRADINGSYMBOL": "ABC"},
        {"ACCESSION_NUMBER": "A4", "FILING_DATE": "08-FEB-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "202", "ISSUERTRADINGSYMBOL": "NASDAQ: xyz"},
        {"ACCESSION_NUMBER": "A5", "FILING_DATE": "09-FEB-2024", "DOCUMENT_TYPE": "4",
         "ISSUERCIK": "202", "ISSUERTRADINGSYMBOL": "XYZ"},
    ]
    tx = [
        # CEO buys twice in one filing: aggregated
        {"ACCESSION_NUMBER": "A1", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "01-FEB-2024",
         "TRANS_CODE": "P", "TRANS_SHARES": "1000", "TRANS_PRICEPERSHARE": "10",
         "TRANS_ACQUIRED_DISP_CD": "A", "SHRS_OWND_FOLWNG_TRANS": "3000",
         "DIRECT_INDIRECT_OWNERSHIP": "D"},
        {"ACCESSION_NUMBER": "A1", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "02-FEB-2024",
         "TRANS_CODE": "P", "TRANS_SHARES": "1000", "TRANS_PRICEPERSHARE": "12",
         "TRANS_ACQUIRED_DISP_CD": "A", "SHRS_OWND_FOLWNG_TRANS": "4000",
         "DIRECT_INDIRECT_OWNERSHIP": "D"},
        {"ACCESSION_NUMBER": "A2", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "05-FEB-2024",
         "TRANS_CODE": "P", "TRANS_SHARES": "500", "TRANS_PRICEPERSHARE": "11",
         "TRANS_ACQUIRED_DISP_CD": "A", "DIRECT_INDIRECT_OWNERSHIP": "D"},
        {"ACCESSION_NUMBER": "A3", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "05-FEB-2024",
         "TRANS_CODE": "P", "TRANS_SHARES": "500", "TRANS_PRICEPERSHARE": "11",
         "TRANS_ACQUIRED_DISP_CD": "A", "DIRECT_INDIRECT_OWNERSHIP": "D"},
        {"ACCESSION_NUMBER": "A4", "SECURITY_TITLE": "Series A Preferred Stock",
         "TRANS_DATE": "07-FEB-2024", "TRANS_CODE": "P", "TRANS_SHARES": "100",
         "TRANS_PRICEPERSHARE": "25", "TRANS_ACQUIRED_DISP_CD": "A"},
        {"ACCESSION_NUMBER": "A5", "SECURITY_TITLE": "Common Stock", "TRANS_DATE": "08-FEB-2024",
         "TRANS_CODE": "S", "TRANS_SHARES": "100", "TRANS_PRICEPERSHARE": "25",
         "TRANS_ACQUIRED_DISP_CD": "D"},
    ]
    own = [
        {"ACCESSION_NUMBER": "A1", "RPTOWNERCIK": "0000009001", "RPTOWNERNAME": "Doe Jane",
         "RPTOWNER_RELATIONSHIP": "Director,Officer", "RPTOWNER_TITLE": "President and CEO"},
        {"ACCESSION_NUMBER": "A2", "RPTOWNERCIK": "9002", "RPTOWNERNAME": "Roe Rick",
         "RPTOWNER_RELATIONSHIP": "Officer", "RPTOWNER_TITLE": "EVP"},
        {"ACCESSION_NUMBER": "A4", "RPTOWNERCIK": "9003", "RPTOWNERNAME": "Fund LP",
         "RPTOWNER_RELATIONSHIP": "TenPercentOwner"},
    ]
    return form345_zip(sub, tx, own)


# ======================================================================== EDGAR
def test_acceptance_time_is_read_back_to_new_york_time():
    # verified on real filings: JPM 06:30:38 ET (summer), NVDA 16:31:25 ET (winter)
    assert accepted_et("2026-10-13T14:30:38.000Z") == datetime(2026, 10, 13, 6, 30, 38)
    assert accepted_et("2026-02-26T02:31:25.000Z") == datetime(2026, 2, 25, 16, 31, 25)
    assert accepted_et("2026-10-13T10:30:38.000Z", "single") == datetime(2026, 10, 13, 6, 30, 38)
    assert accepted_et("") is None and accepted_et("garbage") is None


def test_symbols_and_roles_are_normalised():
    assert clean_symbol(" nasdaq: abc ") == "ABC"
    assert clean_symbol("ABC, ABCW") == "ABC"
    assert clean_symbol("brk-b") == "BRK.B"
    assert clean_symbol("NONE") is None and clean_symbol("12AB") is None
    assert clean_symbol(None) is None
    assert role_of("Director,Officer", "President & Chief Executive Officer") == "ceo"
    assert role_of("Officer", "EVP, CFO") == "cfo"
    assert role_of("Officer", "General Counsel") == "officer"
    assert role_of("Director", "") == "director"
    assert role_of("TenPercentOwner", "") == "owner10"


def test_form345_keeps_open_market_purchases_of_shares_only():
    buys, issuers = parse_form345(sample_zip())
    assert list(buys["accession"]) == ["A1", "A2"]  # no 4/A, no preferred, no sale
    a1 = buys.iloc[0]
    assert a1["symbol"] == "ABC" and a1["cik"] == 101 and a1["role"] == "ceo"
    assert a1["shares"] == 2000 and a1["value"] == 22000 and a1["price"] == 11
    assert a1["owned_after"] == 4000 and a1["delta_own"] == pytest.approx(1.0)
    assert a1["traded"] == "2024-02-02" and a1["filed"] == "2024-02-05"
    assert bool(a1["officer"]) and bool(a1["director"]) and not bool(a1["plan"])
    assert bool(buys.iloc[1]["plan"])  # 10b5-1 flag kept for the filter
    assert set(issuers["symbol"]) == {"ABC", "XYZ"}


def test_edgar_needs_a_contact_and_backs_off_on_rate_limits():
    with pytest.raises(EdgarError, match="email"):
        Edgar("anonymous", cache_dir="/tmp/x")
    slept = []
    sess = FakeSession([])
    answers = iter([Resp(403, text="Request Rate Threshold Exceeded"), Resp(200, body={})])
    sess.get = lambda url, timeout=None: next(answers)
    e = Edgar(UA, cache_dir="/tmp/edgar_test", session=sess, sleep=slept.append)
    assert e.get("https://x").status_code == 200 and max(slept) >= 5
    sess.get = lambda url, timeout=None: Resp(403, text="Forbidden")
    with pytest.raises(EdgarError, match="SEC_USER_AGENT"):
        e.get("https://y")


def test_insider_quarter_is_downloaded_once(tmp_path):
    sess = FakeSession([("2024q1_form345.zip", Resp(200, content=sample_zip()))])
    e = Edgar(UA, cache_dir=tmp_path, session=sess, sleep=lambda s: None)
    buys, _ = e.insider_quarter(2024, 1, today=date(2024, 6, 1))
    again, _ = Edgar(UA, cache_dir=tmp_path, session=sess,
                     sleep=lambda s: None).insider_quarter(2024, 1, today=date(2024, 6, 1))
    assert len(sess.calls) == 1 and len(buys) == len(again) == 2
    assert e.insider_quarter(2024, 3, today=date(2024, 8, 1)) is None  # quarter not over


def test_eight_ks_with_older_files_and_calibration(tmp_path):
    recent = {"form": ["8-K", "4", "8-K/A", "8-K"],
              "filingDate": ["2026-07-30", "2026-07-01", "2026-06-01", "2026-04-30"],
              "acceptanceDateTime": ["2026-07-31T00:30:28.000Z", "", "",
                                     "2026-05-01T00:30:41.000Z"],
              "accessionNumber": ["0001-26-000018", "x", "y", "0001-26-000011"],
              "items": ["2.02,9.01", "", "", "2.02,9.01"]}
    old = {"form": ["8-K"], "filingDate": ["2019-01-29"],
           "acceptanceDateTime": ["2019-01-30T02:30:00.000Z"],
           "accessionNumber": ["0001-19-000001"], "items": ["1.01"]}
    sess = FakeSession([
        ("CIK0000000007.json", Resp(200, body={"filings": {
            "recent": recent, "files": [{"name": "CIK0000000007-submissions-001.json",
                                         "filingTo": "2020-01-01"}]}})),
        ("submissions-001.json", Resp(200, body=old)),
        ("0001-26-000018.txt", Resp(200, text="<ACCEPTANCE-DATETIME>20260730163028\n")),
        ("0001-26-000011.txt", Resp(200, text="<ACCEPTANCE-DATETIME>20260430163041\n")),
    ])
    e = Edgar(UA, cache_dir=tmp_path, session=sess, sleep=lambda s: None)
    df = e.eight_ks(7, date(2018, 1, 1), until=date(2026, 8, 1))
    assert list(df["accession"]) == ["0001-19-000001", "0001-26-000011", "0001-26-000018"]
    assert df["accepted_et"].iloc[-1] == datetime(2026, 7, 30, 16, 30, 28)
    n = len(sess.calls)
    e.eight_ks(7, date(2018, 1, 1), until=date(2026, 8, 1))
    assert len(sess.calls) == n  # cached
    samples = list(zip([7, 7], df["accession"].tail(2), df["raw_time"].tail(2), strict=True))
    assert e.calibrate(samples) == "double"


def test_point_in_time_ticker_resolution():
    issuers = pd.DataFrame({"cik": [5, 5], "symbol": ["OLD", "NEW"],
                            "first": ["2019-01-01", "2022-03-01"],
                            "last": ["2021-12-31", "2026-01-01"]})
    resolve = symbol_resolver(issuers, {"NEW": 5, "OTHER": 9})
    assert resolve(5, "2020-06-01") == "OLD" and resolve(5, "2024-06-01") == "NEW"
    assert resolve(5, "2022-01-15") == "OLD"  # gap: closest record
    assert resolve(9, "2024-01-01") == "OTHER" and resolve(77, "2024-01-01") is None


# ======================================================================= timing
def test_event_timing_never_trades_before_the_news_is_public():
    days = pd.bdate_range("2026-10-05", periods=10)  # Mon 5 .. Fri 16
    assert days[session_after(days, "2026-10-07")] == pd.Timestamp("2026-10-08")
    assert days[session_after(days, "2026-10-10")] == pd.Timestamp("2026-10-12")  # Saturday
    pre = datetime(2026, 10, 7, 7, 0)
    intra = datetime(2026, 10, 7, 11, 0)
    after = datetime(2026, 10, 7, 16, 5)
    sat = datetime(2026, 10, 10, 9, 0)
    assert days[first_open_after(days, pre)] == pd.Timestamp("2026-10-07")
    assert days[first_open_after(days, intra)] == pd.Timestamp("2026-10-08")
    assert days[first_open_after(days, after)] == pd.Timestamp("2026-10-08")
    assert days[first_open_after(days, sat)] == pd.Timestamp("2026-10-12")
    assert days[reaction_session(days, pre)] == pd.Timestamp("2026-10-07")
    assert days[reaction_session(days, intra)] == pd.Timestamp("2026-10-07")
    assert days[reaction_session(days, after)] == pd.Timestamp("2026-10-08")


# ======================================================================== panel
def ev_frame(rows):
    return pd.DataFrame(rows, columns=events.EVENT_COLUMNS)


def one_event(frames, sym, t0):
    p = panel_from_frames(frames)
    b = build_bench(p)
    return measure(ev_frame([{"definition": "insider_buy", "family": "insider", "symbol": sym,
                              "j": p.col[sym], "t0": t0, "public": "", "detail": ""}]),
                   p, b, Funnel())


def test_abnormal_return_counts_only_what_happens_after_the_entry_open():
    plain = one_event(market(), "S10", 200)
    ev = one_event(boost(market(), "S10", 200, gain=0.10), "S10", 200)
    # the boost multiplies the 20-session outcome by 1.10, the entry open is unchanged
    growth = (1 + ev["ret_20"].iloc[0]) / (1 + plain["ret_20"].iloc[0])
    assert growth == pytest.approx(1.10, abs=1e-9)
    assert ev["abn_20"].iloc[0] - plain["abn_20"].iloc[0] == pytest.approx(
        0.10 * (1 + plain["ret_20"].iloc[0]), abs=0.01)
    assert ev["net_20"].iloc[0] == pytest.approx(ev["abn_20"].iloc[0] - ev["cost"].iloc[0])
    # buying one session later pays a price that already holds part of the move
    late = one_event(boost(market(), "S10", 200, gain=0.10), "S10", 201)
    plain_late = one_event(market(), "S10", 201)
    assert (1 + late["ret_20"].iloc[0]) / (1 + plain_late["ret_20"].iloc[0]) < 1.10


def test_untradable_repeated_and_late_events_are_dropped_and_counted():
    frames = market()
    frames["S03"] = frames["S03"].assign(close=frames["S03"]["close"] / 20,
                                         open=frames["S03"]["open"] / 20)  # about $1
    p = panel_from_frames(frames)
    b = build_bench(p)
    f = Funnel()
    rows = [{"definition": "insider_buy", "family": "insider", "symbol": s, "j": p.col[s],
             "t0": t, "public": "", "detail": ""}
            for s, t in [("S03", 100), ("S20", 100), ("S20", 110), ("S20", 125), ("S20", 9999)]]
    ev = measure(ev_frame(rows), p, b, f)
    assert list(zip(ev["symbol"], ev["t0"], strict=True)) == [("S20", 100), ("S20", 125)]
    stages = f.counts["insider_buy"]
    assert stages["kept"] == 2 and stages["after the last price"] == 1
    assert sum(v for k, v in stages.items() if k.startswith("not tradable")) == 1
    assert stages["repeat within 20 sessions of a counted one"] == 1


def test_delisted_stock_keeps_its_last_close():
    frames = market()
    df = frames["S30"].copy()
    df.iloc[205:] = np.nan
    frames["S30"] = df.dropna()
    p = panel_from_frames(frames)
    b = build_bench(p)
    ev = measure(ev_frame([{"definition": "insider_buy", "family": "insider", "symbol": "S30",
                            "j": p.col["S30"], "t0": 200, "public": "", "detail": ""}]),
                 p, b, Funnel())
    expected = frames["S30"]["close"].iloc[204] / frames["S30"]["open"].iloc[200] - 1
    assert ev["ret_20"].iloc[0] == pytest.approx(expected)
    assert bool(ev["delisted_20"].iloc[0]) and not bool(ev["delisted_1"].iloc[0])


def test_insider_events_check_the_reported_price_and_find_clusters():
    frames = market()
    p = panel_from_frames(frames)
    day = lambda i: str(p.days[i].date())  # noqa: E731
    px = float(frames["S05"]["close"].iloc[150])

    def buy(acc, owner, i, value=30_000, price=px, sym="S05", role="director", plan=False):
        return {"accession": acc, "filed": day(i), "traded": day(i - 1), "cik": 55,
                "symbol": sym, "owner_cik": owner, "owner": owner, "role": role,
                "officer": role in ("ceo", "cfo", "officer"), "director": role == "director",
                "owner10": False, "shares": value / price, "value": value, "price": price,
                "owned_after": 1e6, "delta_own": 0.01, "plan": plan}

    buys = pd.DataFrame([
        buy("a", "o1", 150), buy("b", "o2", 152), buy("c", "o3", 155),  # cluster on c
        buy("d", "o4", 156, price=px * 7.3),  # wrong price (reused ticker / typo)
        buy("e", "o5", 157, plan=True),  # 10b5-1 plan: not a decision
        buy("f", "o6", 158, sym="ZZZ"),  # no prices
        buy("g", "o7", 160, value=80_000, role="ceo"),
    ])
    f = Funnel()
    ev = insider_events(buys, p, f)
    by = {k: g for k, g in ev.groupby("definition")}
    assert list(by["insider_cluster"]["public"]) == [day(155)]
    assert by["insider_cluster"]["t0"].iloc[0] == 156  # the session after the filing day
    assert set(by["insider_ceo_cfo"]["public"]) == {day(160)}
    assert f.counts["insider_buy"]["reported price far from the market price"] == 1
    assert f.counts["insider_buy"]["no price history for the ticker"] == 1
    assert "e" not in set(buys.loc[buys["plan"], "accession"]) - {"e"}
    assert len(by["insider_buy"]) == 4  # a, b, c, g


def test_price_check_allows_later_splits():
    assert list(price_matches(np.array([10.0, 40.0, 1.0, 37.0]),
                              np.array([10.5, 10.0, 10.0, 10.0]))) == [True, True, True, False]


def test_earnings_reaction_is_measured_on_the_right_day():
    frames = market()
    p = panel_from_frames(frames)
    # results after the close of session 150: the jump shows in session 151
    df = frames["S40"].copy()
    df.iloc[151:, :4] = df.iloc[151:, :4] * 1.12
    frames["S40"] = df
    p = panel_from_frames(frames)
    b = build_bench(p)
    accepted = datetime.combine(p.days[150].date(), datetime.min.time()).replace(hour=16, minute=5)
    filings = pd.DataFrame({"cik": [1, 1], "accession": ["x", "y"],
                            "filed": [str(p.days[150].date())] * 2, "items": ["2.02,9.01", "1.01"],
                            "accepted_et": [accepted, accepted], "raw_time": ["", ""]})
    f = Funnel()
    ev = eightk_events(filings, lambda cik, d: "S40", p, b, f)
    up = ev[ev["definition"] == "earnings_jump_up"]
    assert list(up["t0"]) == [152]  # bought the morning after the reaction day
    assert "reaction +1" in up["detail"].iloc[0]  # about +12 % vs the group
    assert ev[ev["definition"] == "earnings_jump_down"].empty
    assert list(ev.loc[ev["definition"] == "agreement_8k", "t0"]) == [151]
    only_filings = eightk_events(filings, lambda cik, d: "S40", p, b, Funnel(), ("filings",))
    assert set(only_filings["family"]) == {"filings"}


# =================================================================== statistics
def test_bootstrap_p_values_follow_the_sign():
    rng = np.random.default_rng(1)
    weeks = np.repeat(np.arange(100), 3)
    pos = events.boot_stats(rng.normal(0.02, 0.05, 300), weeks, 2000, 3)
    flat = events.boot_stats(rng.normal(0.0, 0.05, 300), weeks, 2000, 3)
    assert pos["p_pos"] < 0.01 and pos["lo"] > 0
    assert flat["p"] > 0.05 and flat["lo"] < 0 < flat["hi"]


def _many_events(frames, syms, t0s, gain):
    for s, t in zip(syms, t0s, strict=True):
        boost(frames, s, t, gain=gain)
    return frames


def test_a_real_event_effect_is_validated():
    rng = np.random.default_rng(5)
    frames = market(n_days=700, n_stocks=80, seed=2)
    picks = [(f"S{k:02d}", int(t)) for k in range(0, 80, 2)
             for t in range(60 + int(rng.integers(0, 60)), 660, 60)]
    frames = _many_events(frames, [s for s, _ in picks], [t for _, t in picks], 0.06)
    p = panel_from_frames(frames)
    rows = [{"definition": "insider_buy", "family": "insider", "symbol": s, "j": p.col[s],
             "t0": t, "public": "", "detail": ""} for s, t in picks]
    f = Funnel()
    f.add("insider_buy", "raw", len(rows))
    b = build_bench(p)
    ev = measure(ev_frame(rows), p, b, f)
    res = events.analyse(ev, p, b, f, locked_from=p.days[420], n_boot=2000, draws=300)
    row = res.table.set_index("definition").loc["insider_buy"]
    assert row["design_mean"] > row["design_placebo_p95"]
    assert row["verdict"] == "VALIDATED BUY", row.to_dict()


def test_stocks_that_rise_anyway_do_not_pass_the_placebo():
    """Survivorship: event stocks drift up every day, but the event day is not special."""
    frames = market(n_days=700, n_stocks=80, seed=3)
    winners = [f"S{k:02d}" for k in range(0, 80, 4)]
    for s in winners:
        df = frames[s]
        drift = np.exp(np.arange(len(df)) * 0.004)
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * drift
    p = panel_from_frames(frames)
    rng = np.random.default_rng(9)
    rows = [{"definition": "insider_buy", "family": "insider", "symbol": s, "j": p.col[s],
             "t0": int(t), "public": "", "detail": ""}
            for s in winners for t in rng.choice(np.arange(60, 660, 25), 12, replace=False)]
    f = Funnel()
    f.add("insider_buy", "raw", len(rows))
    b = build_bench(p)
    ev = measure(ev_frame(rows), p, b, f)
    res = events.analyse(ev, p, b, f, locked_from=p.days[420], n_boot=2000, draws=300)
    row = res.table.set_index("definition").loc["insider_buy"]
    assert row["design_abn"] > 0.02  # beats the other stocks easily...
    assert abs(row["design_mean"]) < row["design_abn"] / 2  # ...not its own usual days
    assert row["verdict"].startswith("NOT PROVEN")


def test_verdicts():
    base = {"design_n": 300, "locked_n": 200, "design_mean": 0.02, "design_net": 0.01,
            "design_abn": 0.025,
            "design_p_holm": 0.001, "design_placebo_p95": 0.005, "design_placebo_p5": -0.01,
            "locked_mean": 0.015, "locked_p_pos": 0.01, "locked_p_neg": 0.99,
            "locked_net": 0.005}
    assert events.verdict(base) == "VALIDATED BUY"
    assert events.verdict({**base, "design_n": 50}).startswith("NOT PROVEN (only 50")
    assert "costs" in events.verdict({**base, "design_net": -0.001})
    assert events.verdict({**base, "locked_mean": -0.01}).startswith("DESIGN ONLY")
    assert "random days" in events.verdict({**base, "design_placebo_p95": 0.03})
    veto = {**base, "design_mean": -0.03, "design_abn": -0.02, "design_placebo_p5": -0.01,
            "locked_mean": -0.02,
            "locked_p_neg": 0.01}
    assert events.verdict(veto) == "VALIDATED VETO"


def test_run_writes_a_report(tmp_path):
    frames = market(n_days=400, n_stocks=50, seed=4)
    p = panel_from_frames(frames)
    views = {"S07": pd.Series(np.r_[np.full(200, 100.0), [5000.0], np.full(199, 100.0)],
                              index=pd.date_range("2022-01-01", periods=400, freq="D"))}
    res = events.run(p, wiki=views, n_boot=500, draws=50,
                     reddit=pd.DataFrame(columns=attention.REDDIT_COLUMNS),
                     meta={"prices": "csv", "survivorship": "test"})
    events.write_outputs(res, tmp_path)
    text = (tmp_path / "report.md").read_text()
    assert "wiki_spike" in text and "Survivorship" in text and "reddit_spike" in text
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["funnel"]["wiki_spike"]["raw"] == 1
    assert (tmp_path / "events.csv").exists()


def test_cli_with_csv_prices_and_attention_only(tmp_path, monkeypatch):
    frames = market(n_days=300, n_stocks=30, seed=6)
    prices = tmp_path / "bars"
    prices.mkdir()
    for s, df in frames.items():
        df.rename_axis("date").to_csv(prices / f"{s}.csv")
    att = tmp_path / "att"
    (att / "wiki" / "views").mkdir(parents=True)
    pd.DataFrame({"date": pd.date_range("2022-01-01", periods=300, freq="D"),
                  "views": 100.0}).to_csv(att / "wiki" / "views" / "S01.csv", index=False)
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    code = events.main(["--prices", "csv", "--csv-dir", str(prices), "--families", "attention",
                        "--attention-dir", str(att), "--out", str(tmp_path / "out"),
                        "--boot", "200", "--placebo-draws", "20", "--env-file", "none"])
    assert code == 0 and (tmp_path / "out" / "report.md").exists()


# ===================================================================== grouped
def test_grouped_store_resumes_and_is_point_in_time(tmp_path):
    from invest.data import DataNotReady

    class Grouped:
        calls = []

        def grouped(self, day):
            self.calls.append(day)
            if day >= date(2026, 10, 16):
                raise DataNotReady("not yet")
            rows = {"AAA": {"open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1e6}}
            if day <= date(2026, 10, 13):
                rows["GONE"] = {"open": 5, "high": 5, "low": 5, "close": 5, "volume": 1e6}
            return rows

    prov = Grouped()
    store = events.GroupedStore(tmp_path, prov)
    store.update(date(2026, 10, 12), date(2026, 10, 16))
    n = len(prov.calls)
    store.update(date(2026, 10, 12), date(2026, 10, 15))
    assert len(prov.calls) == n  # nothing downloaded twice
    long = store.load(date(2026, 10, 1), date(2026, 10, 31), {"AAA", "GONE"})
    p = events.panel_from_long(long)
    assert p.T == 4 and p.symbols == ["AAA", "GONE"]
    assert p.last[p.col["GONE"]] == 1  # delisted after the 13th, still in the sample


# ==================================================================== attention
def apewisdom_page(page, pages=2):
    return Resp(200, body={"pages": pages, "currentPage": page, "results": [
        {"rank": page, "ticker": f"t{page}", "name": "x", "mentions": "50", "upvotes": 3,
         "rank_24h_ago": 9, "mentions_24h_ago": "10"}]})


def test_reddit_snapshot_is_saved_once_a_day(tmp_path):
    sess = FakeSession([("page/1", apewisdom_page(1)), ("page/2", apewisdom_page(2))])
    now = datetime(2026, 10, 16, 21, 0, tzinfo=timezone.utc)
    path = attention.collect(tmp_path, session=sess, now=now, sleep=lambda s: None)
    assert path.name == "2026-10-16.csv"
    assert attention.collect(tmp_path, session=sess, now=now, sleep=lambda s: None) is None
    df = attention.load_reddit(tmp_path)
    assert list(df["ticker"]) == ["T1", "T2"] and df["mentions"].tolist() == [50.0, 50.0]


def test_reddit_spike_needs_a_history_and_waits_for_the_next_open():
    days = pd.bdate_range("2026-09-01", periods=40)
    frames = {"AAA": pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                                   "volume": 1e6}, index=days)}
    p = panel_from_frames(frames)
    snaps = []
    for k in range(10):
        taken = pd.Timestamp("2026-09-07 21:00", tz="UTC") + pd.Timedelta(days=k)
        snaps.append({"taken_utc": taken, "ticker": "AAA", "rank": 5,
                      "mentions": 300.0 if k == 8 else 20.0})
    f = Funnel()
    ev = events.reddit_events(pd.DataFrame(snaps), p, f)
    assert len(ev) == 1
    # snapshot at 21:00 UTC = 17:00 New York, after the close: bought next session
    assert p.days[ev["t0"].iloc[0]] == pd.Timestamp("2026-09-16")


def test_wikipedia_titles_and_pageviews(tmp_path):
    sparql = Resp(200, body={"results": {"bindings": [
        {"cik": {"value": "0000320193"},
         "article": {"value": "https://en.wikipedia.org/wiki/Apple_Inc."}}]}})
    views = Resp(200, body={"items": [{"timestamp": "2024010100", "views": 10},
                                      {"timestamp": "2024010300", "views": 30}]})
    sess = FakeSession([("wikidata", sparql), ("pageviews", views)])
    store = attention.WikiStore(tmp_path, UA, session=sess, sleep=lambda s: None)
    added = store.update({"AAPL": 320193, "NOPE": 1}, end=date(2024, 1, 3))
    assert added == {"AAPL": (date(2024, 1, 3) - attention.WIKI_START).days + 1}
    s = attention.load_wiki(tmp_path)["AAPL"]
    assert s.iloc[-3:].tolist() == [10.0, 0.0, 30.0]  # a missing day is a zero
    assert any("Apple_Inc." in c for c in sess.calls)
    calls = len(sess.calls)
    store.update({"AAPL": 320193, "NOPE": 1}, end=date(2024, 1, 3))
    assert len(sess.calls) == calls  # titles remembered, nothing new to fetch
    with pytest.raises(attention.AttentionError, match="email"):
        attention.WikiStore(tmp_path, "anonymous")


def test_modules_document_their_sources():
    assert "point-in-time" in edgar.__doc__ and "Wikipedia" in attention.__doc__
    assert "placebo" in events.__doc__


def test_cli_point_in_time_run_with_massive(tmp_path, monkeypatch):
    """The whole --prices massive path with fake SEC and Massive services."""
    import invest.data
    import invest.edgar

    frames = market(n_days=260, n_stocks=30, seed=8, start="2025-09-01")
    days = frames["S00"].index

    class FakeMassive:
        name = "massive"

        def __init__(self, key):
            self.limiter = None

        def grouped(self, day):
            ts = pd.Timestamp(day)
            if ts not in days:
                return {}
            return {s: df.loc[ts].to_dict() for s, df in frames.items()}

    px = float(frames["S04"]["close"].iloc[100])
    buys = pd.DataFrame([{
        "accession": "a", "filed": str(days[100].date()), "traded": str(days[99].date()),
        "cik": 4, "symbol": "S04", "owner_cik": "1", "owner": "Doe", "role": "ceo",
        "officer": True, "director": False, "owner10": False, "shares": 1e4, "value": 1e4 * px,
        "price": px, "owned_after": 2e4, "delta_own": 1.0, "plan": False}])
    issuers = pd.DataFrame({"cik": list(range(30)), "symbol": list(frames),
                            "first": "2015-01-01", "last": "2030-01-01"})

    class FakeEdgar:
        time_mode = "double"

        def __init__(self, ua, cache_dir):
            pass

        def insiders(self, start, end):
            return buys, issuers

        def tickers(self):
            return {s: k for k, s in enumerate(frames)}

        def eight_ks(self, cik, since, until=None):
            when = datetime.combine(days[150].date(), datetime.min.time()).replace(hour=7)
            return pd.DataFrame({"cik": [cik], "accession": [f"x{cik}"],
                                 "filed": [str(days[150].date())], "items": ["2.02"],
                                 "raw_time": [""], "accepted_et": [when]})

        def calibrate(self, samples):
            return "double"

    monkeypatch.setattr(invest.data, "MassiveProvider", FakeMassive)
    monkeypatch.setattr(invest.edgar, "Edgar", FakeEdgar)
    monkeypatch.setattr(events, "date", type("D", (), {"today": staticmethod(
        lambda: date(2026, 5, 20))}))
    code = events.main(["--prices", "massive", "--days", "400", "--data-dir",
                        str(tmp_path / "data"), "--out", str(tmp_path / "out"), "--boot", "200",
                        "--placebo-draws", "20", "--env-file", "none"])
    assert code == 0
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["funnel"]["insider_ceo_cfo"]["kept"] == 1
    assert summary["funnel"]["earnings_jump_up"]["raw"] == 30
    assert len(list((tmp_path / "data" / "grouped").glob("*.csv.gz"))) >= 200
    assert "none by construction" in (tmp_path / "out" / "report.md").read_text()
