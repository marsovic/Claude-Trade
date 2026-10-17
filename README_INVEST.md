# Survival Investment Bot — stocks & ETPs, no crypto

The same idea as the crypto bot, applied to **investments**:

> **It starts with 100 €. If it loses them, it stops — permanently.**

It ranks ~130 liquid, high-beta **US stocks** every evening by momentum and buys
only when **7 conditions align**. It trades aggressively: 4 % risk per trade,
2–3 concentrated positions, pyramiding on winners, and an optional
**3x leveraged ETP** sleeve. Crypto and crypto proxies (MSTR, COIN, miners) are
excluded. It cross-checks every candidate with **research sources**: news tone,
Reddit mania, earnings calendar and insider buying. A **web dashboard** shows
every decision in real time.

> [!WARNING]
> **Not financial advice. Very aggressive by design.** A concentrated momentum
> strategy can lose 50–70 % in a bad year, and leveraged ETPs can lose most of
> their value in a volatile market. Momentum's historical edge has shrunk a lot
> since 2000. Run backtests, paper trade for weeks, and only fund it with money
> you accept to lose.

---

## 1. Which platform allows a bot in France?

Research done in October 2026. Fees and terms change, so re-check before funding.

| Platform (entity for France) | Official order API | 100 € viable? | Notes |
|---|---|---|---|
| **Trading 212** (Invest account / CTO, Trading 212 EU GmbH, BaFin) | **Yes**: Public API v0 (beta); live + demo; market, limit, stop orders since Feb 2026 | **Yes**: 0 commission, fractional shares from ~1 €, 0.15 % FX fee on USD stocks | ⚠️ Its own GitHub tool says *"As required by MiFID II, you are prohibited from using this tool for algorithmic trading"*. **Get Trading 212's written agreement before going live.** No market data in the API, no French IFU, no PEA. |
| **Interactive Brokers** (IBKR Ireland) | **Yes**, built for automation (TWS API / IB Gateway) | **Poorly**: the API trades whole shares only; US stocks cost 0.005 $/share (min 1 $, max 1 %) | The safest legally. Realistic from ~1 000–2 000 €. Needs IB Gateway running (weekly 2FA login). Real-time data needs ~500 $ of equity. |
| **Alpaca** (Alpaca Securities LLC, USA) | **Yes**, the best API; free paper account for anyone | Trading is free and fractional | Live eligibility for French residents **not confirmed** by an official source. Account is in USD (wire funding, 35 $ outbound wire). US protection (SIPC), no EU scheme. US stocks only. |
| Saxo (PEA possible), eToro, Freedom24 | APIs exist | No: ~2 €/order minimum, or 0.75 % FX | Saxo PEA via API is undocumented. |
| Capital.com | API, **CFDs only** | — | Leveraged CFDs: excluded on purpose. |
| Boursorama, Fortuneo, Bourse Direct, DEGIRO, Trade Republic, Revolut, Lightyear | **No** official order API | — | Scrapers or unofficial APIs break their terms: not supported. |
| XTB | API **shut down** (March 2025) | — | |
| Scalable Capital | MCP/CLI (Aug 2026), but every order needs your manual approval | — | Not autonomous. |

**Recommendation**

- **To actually start with 100 €:** use **Trading 212**. Start on the **demo**
  account, then email support and get their written OK for automated orders
  through the API before you switch to live. The bot refuses live Trading 212
  until you set `INV_T212_TERMS_ACKNOWLEDGED=true`.
- **If Trading 212 says no, or once you have ≥ 1–2 k€:** use **Interactive
  Brokers**. The bot's fee guard skips trades whose costs exceed 0.15 R, so
  expect very few trades at 100 €.
- **To test against a real broker simulator right away:** open a free
  **Alpaca paper** account (email only).
- **PEA:** no French PEA provider offers a documented order API, so the bot
  runs on a CTO (ordinary securities account).

## 2. Strategy (`invest/signals.py`, `invest/engine.py`)

The bot decides once per day, after the US close (around 22:30 Paris time). Orders execute at the
**next open**. Real brokers get them at 09:30 New York (15:30 Paris).

**Ranking.** Each stock gets a cross-sectional z-score, winsorised at ±3:
`0.35·z(r12-1/vol) + 0.25·z(r6-1/vol) + 0.15·z(r3/vol) + 0.25·z(close/52w high)`.

**The 7 conditions (all required, long only):**

| # | Condition | Rule |
|---|---|---|
| 1 | `regime` | SPY above its 200-day SMA, VIX ≤ 30, VIX/VIX3M ≤ 1 |
| 2 | `top_rank` | top 10 % of the ranking and score > 0 |
| 3 | `abs_momentum` | 12-1 and 6-1 month returns > 0 |
| 4 | `trend` | close > SMA200 and EMA20 > EMA50 |
| 5 | `near_high` | close ≥ 85 % of the 52-week high |
| 6 | `breakout` | close ≥ highest close of the previous 20 days, volume ≥ 1.3× average |
| 7 | `not_extended` | close ≤ 25 % above SMA50 and RSI(14) ≤ 80 |

**Universe filters:** price ≥ 10 $, median daily dollar volume ≥ 50 M$, ≥ 300
bars of history, ATR between 1 and 7 %, no daily jump > 15 % in the last 21
days, beta ≥ 1.

**Vetoes (never triggers):**
- earnings within 2 sessions;
- clearly negative news tone (Alpha Vantage scores, or Finnhub headlines scored with a finance lexicon);
- retail mania: Reddit mentions ×4 in 24 h while RSI ≥ 75.

**Bonus:** ≥ 2 insider open-market purchases (SEC Form 4) in 30 days adds +0.25 to the score.

**Sizing (aggressive):**
- Initial stop = 3×ATR(22), clamped to 4–20 %.
- Weight = min(4 % / stop distance, 1/slots, 40 %).
- 3 slots, 2 below 60 €.
- Total open risk ≤ 12 %.
- No margin.
- Buys are limit orders capped at close + 3 %, so the bot never chases a gap.

**Exits:**
- Chandelier stop (highest high − 3 ATR, ratchets up), resting at the broker.
- Weekly rank exit (drops out of the top 25 % or 6-month momentum turns negative).
- Time stop (30 sessions below +0.5 R while not in the top 10 %).
- Exit before earnings below +1 R.
- Risk-off (SPY below SMA200 for 2 closes, or VIX > 35) tightens stops to 2 ATR.

**Pyramiding:** one add of 50 % at +1.5 R on a new 20-day high, trimmed so the position never exceeds the 40 % cap. The stop then moves to break-even, and the add's own risk counts in the trade's R.

**One bet per theme:** a hand-maintained map (`invest/universe.py`, `THEMES`: AI chips and hardware, software, financials...) allows one position per theme by default (`INV_MAX_PER_THEME`). Three slots filled with AI chips are one 100 % bet that stops out on the same day (two stops fired together in the first real backtest). Set it to 0 to disable.

**3x ETP sleeve** (off by default: `INV_ETP_ENABLED=true` to turn it on; it only added exposure to the same Nasdaq bet and has not passed its own out-of-sample test):
- At most one position, 33 % weight.
- Only in calm uptrends: underlying above SMA50 and SMA200, realised volatility < 20 %, VIX < 22, and trend strong enough to beat the volatility drag.
- In simulation it is synthesised from QQQ with daily reset and 0.75 %/yr costs.
- With a real broker, set the instrument to trade (`INV_ETP_BROKER_TICKER`). EU retail cannot buy US-domiciled ETFs (PRIIPs), so use an EU-listed ETP after the broker's appropriateness test.

**Survival:**
- Floor 10 €: below it the bot liquidates, writes `TOMBSTONE.json` and refuses to restart.
- Drawdown ladder: below −20 %, 3 % risk and one slot less; below −35 %, 2 % risk, one slot and no ETP.
- A −8 % week (about two stops) pauses entries for 5 sessions.
- Orders under 8 € are refused: a 2 € "dust" position would occupy a slot for weeks.
- Fees above 5 % of equity over 3 months pause entries.

## 3. Data and research sources (official APIs only)

| Need | Source | Key | Free limits |
|---|---|---|---|
| Daily US bars | **Massive** (ex-Polygon): grouped daily = whole market in 1 call | `MASSIVE_API_KEY` | 5 calls/min, 2 years |
| Daily US bars (alternative) | Alpaca market data / Tiingo | `ALPACA_*` / `TIINGO_API_KEY` | 200/min / 50 per hour |
| VIX, VIX3M | FRED (`VIXCLS`, `VXVCLS`) | none (keyless CSV) | — |
| EUR/USD | ECB reference rate | none | — |
| News tone | Alpha Vantage NEWS_SENTIMENT, Finnhub company-news | `ALPHAVANTAGE_API_KEY`, `FINNHUB_API_KEY` | 25 calls/day, 60 calls/min |
| Retail attention | ApeWisdom (Reddit / WallStreetBets) | none | — |
| Earnings dates | Alpha Vantage EARNINGS_CALENDAR or Finnhub | same keys | 1 call/day |
| Insider buying | SEC EDGAR Form 4 | `SEC_USER_AGENT="Name email"` | 10 req/s |

Not used:
- Yahoo / yfinance: unofficial, against Yahoo's terms.
- X API: paid per post.
- StockTwits: closed to new developers.
- Reddit official API: requires approval.

The first run downloads ~1.5 years of history (≈ 30 min on Massive's free plan). After that, it needs one call per day.

## 4. Getting started

```bash
pip install -r requirements.txt          # + `pip install ib_async` for IBKR only
cp .env.example .env                     # fill the INVEST section (MASSIVE_API_KEY at least)

# Simulator on real market data, Trading 212 costs (no broker account needed)
python -m invest.invest_bot              # runs forever (decides after each US close)
python -m invest.invest_bot --once       # simulator only: process the last close and exit

# Dashboard (same as the crypto bot), in another terminal
python dashboard/dashboard.py --state-dir state_invest --port 8051

# Backtest with the exact same engine
python -m invest.backtest --provider massive --days 700
python -m invest.backtest --csv-dir data/us_daily --no-vix      # offline CSV files

# Trading 212 demo account (keys created in the app: Settings > API, demo mode)
python -m invest.invest_bot --broker trading212 --state-dir state_t212_demo

# Real money (after weeks of demo, and Trading 212's written OK)
INV_T212_TERMS_ACKNOWLEDGED=true python -m invest.invest_bot --broker trading212 \
    --environment live --state-dir state_t212_live --i-understand-the-risks
```

Keep one state directory per broker and environment: the bot refuses to mix them.

With a real broker the bot must run **continuously** (systemd, Docker, `tmux`):
orders decided after the close are sent at the next open and stops are watched
during the session. `--once` is therefore refused with a real broker (it still
runs one liquidation step after a death). The NYSE calendar (weekends and
holidays) is built in; early closes (13:00 ET) are not.

### Real-money safety rules built in

- Live trading stays locked until `INV_LIVE_TRADING=I_UNDERSTAND_THE_RISKS` (or
  `--i-understand-the-risks`). Trading 212 live also needs
  `INV_T212_TERMS_ACKNOWLEDGED=true`.
- Every position gets a stop resting at the broker. The stop is cancelled in two
  phases: the bot asks for the cancellation and waits for the broker's
  confirmation before selling or placing a new stop. Brokers lock shares under a
  pending stop, and on IBKR a sell racing a stop fill could go short.
- An order whose submission timed out is looked up at the broker, never sent
  twice. Partial fills are booked and protected at once.
- Entries planned for a past session are dropped. A position counts as "closed
  by hand" only after two consecutive checks.
- Survival rule:
  - a single bad intraday reading cannot kill the bot: two are needed;
  - at death it cancels working buys and liquidates;
  - it keeps reconciling the liquidation until the account is flat, even after a
    restart, and then refuses to run.
- Stock splits are detected in the data cache (and remembered across restarts in
  `bars/splits.json`), and open positions are rescaled.
- A refused or expired liquidation order is sent again until the account is flat.

**Leveraged ETP with a real broker.** EU retail cannot buy US-domiciled ETFs, so
pick an EU-listed 2x/3x product your broker offers, and pass its appropriateness
test in the app. Then set three variables:
- `INV_ETP_BROKER_TICKER`: the instrument (Trading 212 ticker, or
  `SYMBOL:EXCHANGE:CURRENCY` for IBKR);
- `INV_ETP_PRICE_SYMBOL`: its Alpha Vantage symbol, used for the daily prices;
- `INV_ETP_CURRENCY`: its currency.

Without them the sleeve only runs in the simulator. Alpaca has no EU ETPs.

## 5. French tax: what you must do

The bot keeps `state_invest/tax_ledger.jsonl` (every fill in EUR, PMP cost basis):

```bash
python -m invest.tax --state-dir state_t212_live --year 2026
```

- **Flat tax (PFU) 2026: 31.4 %** = 12.8 % income tax + 18.6 % social levies
  (CSG raised by the LFSS 2026, loi n° 2025-1403).
- Gains go on **form 2074** and the **2042-C, box 3VG** (gain) or **3VH** (loss).
  Losses offset gains for 10 years. Foreign brokers issue **no IFU**, so the ledger
  is your working paper.
- **Form 3916-bis every year** for each foreign account (Trading 212, IBKR,
  Alpaca): the fine is **1 500 € per undeclared account**, i.e. 15 times the capital.
- US dividends: sign the W-8BEN (15 % withholding). The momentum bot rarely holds through many dividends.
- Algorithmic trading on your own account needs no authorisation (MiFID II
  art. 2(1)(d)). Market-abuse rules still apply.

This is not tax advice: check impots.gouv.fr or an adviser.

## 6. Is the strategy any good? What the audits found

An independent audit of the simulation engine (look-ahead, fills, exits, sizing) found **no look-ahead and no bias** in the backtest machinery: on zero-edge synthetic markets it loses about its costs, as it should. So the first real backtest (−26 % over 219 sessions while the S&P 500 gained 17 %) is a verdict on the **signals**, not on a bug. But 20–26 trades cannot validate or refute anything, and nothing in the entry rules has been changed because no change can be justified on that sample.

What the audit did change: the backtest could not open a stock for its first two months (`min_history`), a "trailing stop" label hid gap losses, an add to a winner ignored the 40 % cap and its own risk, the weekly loss pause was unreachable, three slots could all be AI chips, and the leveraged ETP sleeve was on by default.

How to judge the strategy properly, with 10 years of data instead of 26 trades:

```bash
# 1. The study (Tiingo, free key in TIINGO_API_KEY): ~3 h the first time, then minutes
python -m invest.study --provider tiingo --days 3000 --out study_invest
# 2. The backtest of the whole bot (slots, cash, themes...) on the same data
INV_DATA_DIR=data_invest_tiingo python -m invest.backtest --provider tiingo --days 3000 \
    --placebo 20 --out backtest_invest_tiingo
```

**`invest.study`** treats every 7/7 signal of every day as one trade (no slot or cash limit), so ten years give thousands of trades. Parameter variants are declared in advance (`VARIANTS` in `invest/study.py`): stop and trailing distance, rank cut-off, distance to the 52-week high, volume factor, extension, entry filters... They are **selected on the first ~60 % of the years** and **confirmed once on the last ~40 %**, which the tool never shows for variants that did not pass. Every result is measured against the market's own drift (a longer hold earns more in a rising market even without skill), with a block bootstrap by week, Holm correction for the number of variants tried, and a random-entry placebo with the same holding time. The verdict is one of:

- **KEEP THE DEFAULTS**: nothing beat the current rules robustly. This is the expected answer most of the time.
- **ONE CHANGE IS SUPPORTED**: it names the exact `INV_*` variable to set. Apply it, then re-run the backtest.
- **NO CONCLUSION**: not enough locked-years data or a data problem (it says which).

Useful flags: `--csv-dir DIR` (offline), `--start/--end`, `--locked-from 2022`, `--no-vix`, `--audit full`. It writes `report.md`, `events.csv`, `variants.csv` and `summary.json` in `--out`. On synthetic markets with no edge it recommended a change in under 1 % of runs, and it detects a planted effect only when that effect is real, large enough and spread over the years. That is by design: a conservative tool, because a false recommendation changes a live trading rule.

`invest.backtest --placebo N` is a quicker, rougher check of the whole bot (random entries, same exits and sizing); the study's placebo is the calibrated one. Also compare with the equal-weight buy & hold the backtest prints next to SPY.

## 7. Honest limitations

- **The edge may not exist any more.** Long-short momentum had a Sharpe of 0.12 over 2000–2025, against 0.45 since 1927. A good backtest proves nothing.
- **Concentration.** 2–3 stocks means 35–40 % annual volatility, overnight gaps through stops, and correlated losses.
- **Backtest biases.**
  - Today's curated universe brings survivorship bias.
  - Prices are split-adjusted, without dividends.
  - EUR/USD is constant unless an ECB series is loaded.
  - News, Reddit, earnings and insider history are not replayed.
  - The VIX is used with a one-session lag, as live (FRED publishes it the next morning).
- **Leveraged ETPs.** Volatility drag of roughly 3σ² per year at 3x. Real ETP prices and tickers depend on the broker.
- **Data.** Alpaca market data needs the SIP feed (15-minute delay on the free plan):
  the bot refuses the IEX-only feed, whose volumes would distort the signals.
- **Fees.** 100 € is tiny. On IBKR, a 1 $ minimum commission is ~3 % of a 33 € position, so most trades are refused by the fee guard.
- **Trading 212.** The API is still beta and its policy on bots is not explicit.
- **No network here.** The code was tested against fakes and synthetic markets only. Run it in simulation/demo first and read the decision journal.

## Résumé en français

Bot d'investissement **actions/ETP (sans crypto)** :
- Il démarre avec **100 €**. S'il les perd, il s'arrête **définitivement**.
- Il classe chaque soir ~130 actions US liquides par **momentum** et n'achète que si **7 conditions** sont alignées.
- Il est **agressif** : 4 % de risque par trade, 2-3 positions, renforcement des gagnants, poche **ETP x3** optionnelle.
- Le sentiment sert de filtre : news, manie Reddit, résultats trimestriels, achats d'initiés.

**Plateforme :**
- **Trading 212** est la seule réellement viable à 100 € : API officielle, 0 commission, fractions d'actions. **Obtenez leur accord écrit** avant le réel, car leur outil officiel interdit le trading algorithmique (MiFID II).
- **Interactive Brokers** accepte explicitement les bots, mais n'est rentable qu'à partir de ~1 000-2 000 €.
- **Alpaca** convient au paper trading.

**Fiscalité :**
- CTO, PFU **31,4 %** en 2026.
- Formulaire **2074** et case **3VG/3VH**.
- **3916-bis obligatoire** chaque année pour le compte étranger.

**Ceci n'est pas un conseil financier.**
