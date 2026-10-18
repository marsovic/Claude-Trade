# Crypto Survival Bot (Binance futures / Revolut X) — paused

An open-source crypto trading bot for **Binance USDⓈ-M futures** with one simple rule:

> **It starts with 50 USDT. If it loses them, it stops — permanently.**

It scans the 40 most liquid perpetual pairs and only opens a position when
**7 technical conditions agree** (EMA, RSI, MACD, Bollinger, volume, ATR), long or
short. It adapts its leverage to volatility (**x5 maximum**), risks **2 % of its
capital per trade**, and cross-checks every signal with **market sentiment**
(Fear & Greed index, CoinGecko trending, CryptoPanic, X). A **real-time web
dashboard** shows every decision it takes.

> **New:** an investment version for **stocks & ETPs (no crypto)**, starting with
> 100 €, with the list of platforms that accept bots in France →
> [README_INVEST.md](README_INVEST.md).
>
> **New:** the crypto bot also runs on **Revolut X** (spot, EUR pairs, long only,
> no leverage) → [Revolut X (spot, EUR)](#revolut-x-spot-eur).

> [!WARNING]
> **This is not financial advice.** It is a learning project, still imperfect.
> Leveraged trading can wipe out the whole capital very quickly. Read the code,
> run backtests, paper trade for **several weeks**, use API keys **without
> withdrawal rights**, and never commit money you cannot afford to lose.

---

## Table of contents

- [How it decides](#how-it-decides)
- [The 7 conditions](#the-7-conditions)
- [Risk management](#risk-management)
- [The survival rule](#the-survival-rule)
- [Sentiment](#sentiment)
- [Project layout](#project-layout)
- [Installation](#installation)
- [Usage](#usage) — paper trading, dashboard, backtests, testnet, live
- [Revolut X (spot, EUR)](#revolut-x-spot-eur)
- [Configuration](#configuration)
- [State files](#state-files)
- [Running 24/7](#running-247)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Résumé en français](#résumé-en-français)

---

## How it decides

```
every 20 s ─┬─ manage open positions ── break-even / trailing stop / time exit
            └─ survival check ───────── equity <= floor ? → close all, die forever

every closed 15m candle:
  top 40 USDT perpetuals by 24h volume
        │
        ▼
  7 indicators on the last CLOSED candle ──── 7/7 bullish → LONG candidate
        │                                └── 7/7 bearish → SHORT candidate
        ▼
  filters: already in position? cooldown? max positions? daily loss limit?
        ▼
  sentiment veto: LONG refused if mood is clearly bearish, SHORT if clearly bullish
        ▼
  sizing: 2 % risk at the stop, leverage = f(volatility) ≤ x5, liquidation guard
        ▼
  order + stop-loss + take-profit (placed ON the exchange in testnet/live)
        ▼
  every step is written to the decision journal → dashboard
```

## The 7 conditions

All seven must be true on the same side. Default parameters (all configurable):

| # | Condition   | LONG                                          | SHORT                                         |
|---|-------------|-----------------------------------------------|-----------------------------------------------|
| 1 | `ema_cross` | EMA 20 > EMA 50                               | EMA 20 < EMA 50                               |
| 2 | `ema_trend` | close > EMA 200                               | close < EMA 200                               |
| 3 | `rsi`       | 50 < RSI(14) < 70 (momentum, not overbought)  | 30 < RSI(14) < 50                             |
| 4 | `macd`      | MACD > signal **and** histogram rising        | MACD < signal **and** histogram falling       |
| 5 | `bollinger` | close between middle and upper band           | close between lower and middle band           |
| 6 | `volume`    | volume ≥ 1.3 × average of the 20 previous bars | same                                         |
| 7 | `atr`       | 0.15 % ≤ ATR(14)/price ≤ 3 % (tradable volatility) | same                                     |

Signals are computed on **closed candles only** (no repainting). The exact same
vectorised function (`bot/strategy.py`) drives the live loop and the backtester.

## Risk management

| Rule | Default | Where |
|------|---------|-------|
| Risk per trade | **2 %** of equity, fees and slippage included | `RISK_PER_TRADE` |
| Stop-loss | 1.5 × ATR from entry | `STOP_ATR_MULT` |
| Take-profit | 2 R (twice the stop distance) | `TAKE_PROFIT_R` |
| Break-even | stop moved to entry (+ fees) at +1 R | `BREAKEVEN_R` |
| Trailing stop | from +1.5 R, 1.5 × ATR behind the best price | `TRAILING`, `TRAIL_*` |
| Leverage | `LEVERAGE_VOL_TARGET / ATR%`, clamped to **1…5** | `MAX_LEVERAGE` |
| Liquidation guard | leverage lowered until the stop is < ½ of the liquidation distance | `risk.py` |
| Margin | isolated; ≤ 35 % of equity per trade, ≤ 90 % in total | `MAX_MARGIN_PER_TRADE`, `MAX_TOTAL_MARGIN` |
| Open positions | 3 at most, one per pair | `MAX_OPEN_POSITIONS` |
| Daily loss limit | −6 % from the day's start → no new trade until 00:00 UTC | `MAX_DAILY_LOSS` |
| Cooldown | 4 candles before re-entering a pair | `COOLDOWN_BARS` |
| Max holding time | 96 candles (24 h on 15m) | `MAX_HOLD_BARS` |

**Risk is defined by the stop, not by the leverage.** The position size is
`(equity × 2 %) / (loss per unit at the stop)`. Leverage only decides how much
margin is locked: calm market → up to x5, agitated market → down to x1. If the
margin limits would be exceeded the position is made *smaller* — the bot never
risks more than 2 % to fit a trade, and it skips trades below the exchange's
minimum size rather than over-sizing them.

## The survival rule

`INITIAL_CAPITAL` (50 USDT) is the bot's whole life. When equity falls to
`SURVIVAL_FLOOR` (5 USDT by default — about Binance's minimum order, below which
it could not trade anyway), the bot:

1. closes every open position at market,
2. writes `STATE_DIR/TOMBSTONE.json` (date, equity, number of trades),
3. logs a `death` decision and exits with code `2`.

While the tombstone exists, the bot **refuses to start**. There is no flag to
bypass it: to start a brand-new life you must archive or delete the state
directory yourself. In testnet/live mode, the bot's equity is its own ledger
(`INITIAL_CAPITAL` + realized PnL + fees + funding since it started +
unrealized PnL), capped by the real futures wallet — so it only ever trades
with its 50 USDT, even if the account holds more.

## Sentiment

Sentiment never opens a trade; it can only **veto** one that goes against the mood.

| Source | Key needed | Score in [-1, 1] |
|--------|-----------|------------------|
| [Fear & Greed index](https://alternative.me/crypto/fear-and-greed-index/) | no | `(value − 50) / 50` |
| [CoinGecko trending](https://www.coingecko.com/en/api) | no (optional demo key) | ±0.5 if the asset is trending, by its 24 h direction |
| [CryptoPanic](https://cryptopanic.com/developers/api/) | `CRYPTOPANIC_TOKEN` | community votes on recent news (headline lexicon as fallback) |
| [X recent search](https://developer.x.com/en/docs/x-api) | `X_BEARER_TOKEN` (paid tier) | bullish/bearish lexicon over recent posts (`$SOL OR #SOL`) |

The weighted mean of the available scores is compared with `SENTIMENT_VETO`
(0.25): a LONG is refused below −0.25, a SHORT above +0.25. A failing source is
simply ignored (cached 5 min), so a dead API never blocks the bot. Results are
cached 15 minutes and only fetched for pairs that already have a 7/7 signal,
which keeps API usage tiny.

## Project layout

```
.
├── bot/
│   ├── trading_bot.py     # Main bot: universe, scan, entries, management, survival
│   ├── strategy.py        # The 7 conditions (shared with the backtester)
│   ├── indicators.py      # EMA, RSI, MACD, Bollinger, ATR, volume (pandas only)
│   ├── risk.py            # Sizing, volatility leverage, stops, trailing, exits
│   ├── sentiment.py       # Fear & Greed, CoinGecko, CryptoPanic, X
│   ├── broker.py          # PaperBroker (simulation) and BinanceBroker (real orders)
│   ├── binance_client.py  # Minimal signed REST client (incl. Algo Order API)
│   ├── revolutx_client.py # Revolut X REST client (Ed25519 signatures)
│   ├── revolutx_broker.py # Revolut X spot orders: IOC entries, resting take-profit
│   ├── state.py           # state.json, decision journal, trades, tombstone
│   ├── backtest.py        # Backtester with the same rules
│   └── config.py          # Settings from environment / .env
├── dashboard/
│   ├── dashboard.py       # Real-time web interface (read-only)
│   └── static/index.html
├── invest/                # Investment bot: US stocks + 3x ETP, 100 €, no crypto
│   ├── invest_bot.py      #   main loop (simulator, Trading 212, IBKR, Alpaca)
│   ├── engine.py          #   daily decisions shared by live and backtest
│   ├── signals.py         #   momentum ranking, regime, the 7 conditions
│   ├── brokers/           #   sim, trading212, alpaca, ibkr adapters
│   └── tax.py             #   French tax ledger (PMP in EUR, 2074 / 3VG)
├── tests/                 # pytest suite (no network needed)
├── .env.example           # Template for API keys and settings
├── .gitignore             # Excludes .env and state files
├── LICENSE                # MIT
├── README.md              # project front page
├── README_CRYPTO.md       # this file
└── requirements.txt
```

## Installation

Python 3.10+.

```bash
git clone https://github.com/marsovic/Claude-Trade.git
cd Claude-Trade
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then edit it
```

## Usage

### 1. Paper trading (default — no API key needed)

Real Binance market data, simulated fills with taker fees (0.05 %) and adverse
slippage. No order ever leaves your machine.

```bash
python -m bot.trading_bot             # runs forever, Ctrl+C to stop
python -m bot.trading_bot --once      # a single cycle (scan + management), then exit
```

Stopping the bot keeps its state; restarting resumes the open positions.

### 2. Dashboard

In a second terminal:

```bash
python dashboard/dashboard.py         # http://127.0.0.1:8050
```

It shows equity and its curve against the survival floor, open positions with
unrealized PnL, the scanner (7 dots per side for each of the 40 pairs), market
sentiment, the decision journal (signals, vetoes, skips, opens, closes, stop
moves…) and closed trades. It refreshes every 3 seconds, only reads the state
files and cannot send any order. It listens on `127.0.0.1` by default; do not
expose it to the internet without putting authentication in front of it.

### 3. Backtests

```bash
python -m bot.backtest --top 10 --days 30            # 10 most liquid pairs, 30 days
python -m bot.backtest --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 90 --fng
python -m bot.backtest --start 2025-01-01 --end 2025-06-30 --interval 1h
python -m bot.backtest --csv-dir data/my_klines      # offline, <SYMBOL>.csv files
```

Entries are filled at the **next candle's open**, stops/targets are checked on
each candle's high/low (stop first when both are touched), fees and slippage
apply, and the survival rule can kill the run. Results go to
`backtest_results/` (`summary.json`, `trades.csv`, `equity.csv`). `--fng`
replays the historical Fear & Greed veto; other sentiment sources have no
public history and are neutral in backtests.

### 4. Binance futures testnet

Create keys on <https://testnet.binancefuture.com>, then:

```bash
MODE=testnet STATE_DIR=state_testnet BINANCE_API_KEY=... BINANCE_API_SECRET=... \
  python -m bot.trading_bot
```

### 5. Live trading (real money)

Live mode is locked until you explicitly unlock it. Checklist first:

- [ ] Backtested over several months and several market regimes
- [ ] Paper traded and testnet traded for **several weeks**
- [ ] Read `bot/risk.py`, `bot/broker.py` and `bot/trading_bot.py`
- [ ] Dedicated **sub-account** holding only what you accept to lose
- [ ] API key with **futures trading only — no withdrawal**, IP-restricted
- [ ] Futures account in **one-way** position mode

```bash
MODE=live STATE_DIR=state_live python -m bot.trading_bot --i-understand-the-risks
```

Before trading, the bot checks (and refuses to start otherwise) that the key
**cannot withdraw** and has futures permission, and that the account is in
one-way mode. Every position gets a `STOP_MARKET` and a `TAKE_PROFIT_MARKET`
(`closePosition=true`, mark price) placed **on the exchange** through the Algo
Order API, so it stays protected even if the bot crashes. If the stop cannot be
placed, the position is closed immediately.

## Revolut X (spot, EUR)

`EXCHANGE=revolutx` runs the same bot on [Revolut X](https://exchange.revolut.com),
Revolut's crypto exchange, through its official REST API. It is a **spot**
exchange, so the rules change:

| | Binance futures | Revolut X spot |
|---|---|---|
| Direction | long and short | **long only** (SHORT signals are logged and skipped) |
| Leverage | x1 to x5 | **x1** (no leverage, no liquidation) |
| Universe | 40 most liquid USDT perpetuals | liquid **EUR** pairs (≥ 50 000 €/24 h, about 25), stablecoins excluded |
| Fees | 0.05 % taker | 0.09 % taker, 0 % maker |
| Stop-loss | on the exchange (Algo Order API) | **watched by the bot** every `POLL_SECONDS` (the API has no stop orders) |
| Take-profit | on the exchange | **on the exchange**: GTC limit sell (maker, 0 % fee) |
| Testnet | yes | no: use paper mode |

The 7 conditions, the 2 % risk sizing, break-even, trailing stop, daily loss
limit, sentiment veto and the survival rule (50 € → stops forever if lost) are
unchanged.

### Keys

Revolut X signs every request with an **Ed25519** key pair. Candles and tickers
also need the key, so **paper mode needs it too**.

```bash
openssl genpkey -algorithm ed25519 -out revx_private.pem
openssl pkey -in revx_private.pem -pubout -out revx_public.pem
chmod 600 revx_private.pem          # the bot refuses a key other users can read
```

Register `revx_public.pem` at **exchange.revolut.com → Profile → API Keys**,
create the API key (whitelist your server's IP), then in `.env`:

```bash
EXCHANGE=revolutx
REVX_API_KEY=<the 64-character key>
REVX_PRIVATE_KEY_PATH=/path/to/revx_private.pem
STATE_DIR=state_revx_paper
```

If these two variables are empty, the bot reads the configuration of the
official `revx` CLI / Revolut X MCP server (`~/.config/revolut-x/config.json`
and `private.pem`, or `REVOLUTX_CONFIG_DIR`). Revolut X API keys cannot
withdraw: the API has no withdrawal endpoint.

### Run

```bash
python -m bot.trading_bot --exchange revolutx                    # paper
python -m bot.trading_bot --exchange revolutx --mode live \
    --state-dir state_revx_live --i-understand-the-risks           # real money
python -m bot.backtest --exchange revolutx --symbols BTC-EUR,ETH-EUR,SOL-EUR --days 30
```

Before trading real money the bot checks the key (a `401` usually means the
IP is not whitelisted) and **refuses to start on an empty account**. It never
sells more than the coins it bought itself: coins you already hold on Revolut X
are left alone, and so are orders you placed by hand.

How orders are sent:

- **Entry**: IOC limit buy capped at `ask × (1 + REVX_ENTRY_SLIPPAGE)` (0.3 %).
  Not filled → no trade.
- **Take-profit**: GTC limit sell resting on the exchange; it executes even when
  the bot is offline.
- **Stop-loss / trailing stop**: the bot compares the **bid** with the stop at
  every poll and sells at market when it is touched (the take-profit is
  cancelled first so the coins are free). **While the bot is stopped, nothing
  watches the stop**: run it under systemd / Docker with automatic restart.
- **Lost replies**: every order has a `client_order_id`, saved before sending,
  that the exchange refuses twice. A buy whose reply is lost is looked up by
  that id on the next cycles (never sent again). A sell is capped by the coins
  really available, so it can never be doubled.

### Monitoring with the Revolut X connector

The Revolut X connector for Claude (MCP) is **read-only**: balances, open and
past orders, fills, candles, tickers. It cannot place orders, so the bot talks
to the REST API itself — but you can ask Claude, through the connector, to check
what the bot did (open orders, fills, balances) and compare it with the
dashboard.

### What the data says

Backtest on **real Revolut X 15-minute candles** (10 most traded EUR pairs, 17
Sep → 16 Oct 2026, downloaded through the connector), default settings, long
only: **−22 %** (198 trades, 26 % winners, profit factor 0.52, 5 € of fees on
50 €), in a month when simply holding those coins returned +3 % to +103 %. Wider
stops (2.5–3.5 ATR) or 1-hour candles lost less (−5 % to −12 %) but **every
variant tested lost money**. One month proves little either way, but do not go
live on this strategy without a long paper-trading period.

## Configuration

All settings live in `bot/config.py` and can be overridden by environment
variables or `.env` (upper-case names). The most useful ones:

| Variable | Default | Meaning |
|----------|---------|---------|
| `MODE` | `paper` | `paper`, `testnet` or `live` |
| `INITIAL_CAPITAL` | `50` | the bot's whole life, in USDT |
| `SURVIVAL_FLOOR` | `5` | equity at which the capital is considered lost |
| `UNIVERSE_SIZE` | `40` | number of most liquid pairs scanned |
| `INTERVAL` | `15m` | candle interval |
| `RISK_PER_TRADE` | `0.02` | max loss per trade (fraction of equity, ≤ 0.05) |
| `MAX_LEVERAGE` | `5` | leverage cap |
| `LEVERAGE_VOL_TARGET` | `3.0` | leverage = this / ATR % |
| `STOP_ATR_MULT` / `TAKE_PROFIT_R` | `1.5` / `2.0` | stop distance and reward/risk |
| `MAX_OPEN_POSITIONS` | `3` | simultaneous positions |
| `MAX_DAILY_LOSS` | `0.06` | daily loss that pauses new entries |
| `VOLUME_FACTOR` | `1.3` | volume confirmation threshold |
| `SENTIMENT_VETO` | `0.25` | sentiment threshold for the veto |
| `STATE_DIR` | `state` | where state, journals and logs are written |

See [`.env.example`](.env.example) for the full list.

## State files

| File | Content |
|------|---------|
| `state/state.json` | equity, positions, last scan, sentiment, stats (rewritten atomically) |
| `state/decisions.jsonl` | append-only journal: `scan`, `signal`, `veto`, `skip`, `open`, `close`, `stop_moved`, `halt`, `error`, `death`… |
| `state/trades.jsonl` | every closed trade with PnL, fees, R multiple and exit reason |
| `state/bot.log` | rotating log |
| `state/TOMBSTONE.json` | exists only once the bot has died |

## Running 24/7

A minimal systemd unit (adapt paths and user):

```ini
# /etc/systemd/system/survival-bot.service
[Unit]
Description=Survival trading bot
After=network-online.target

[Service]
WorkingDirectory=/opt/Claude-Trade
ExecStart=/opt/Claude-Trade/.venv/bin/python -m bot.trading_bot
Restart=on-failure
RestartPreventExitStatus=2
User=bot

[Install]
WantedBy=multi-user.target
```

`RestartPreventExitStatus=2` matters: exit code 2 means the bot is dead and
must not be restarted.

## Tests

```bash
pip install -r requirements-dev.txt
pytest          # indicators, strategy, risk, sentiment, bot loop, survival, brokers, backtest, dashboard
ruff check .
```

The suite needs no network: the exchange and the sentiment APIs are replaced
by in-memory fakes, and the request signing is checked against Binance's
documented example.

## Known limitations

- **A good backtest or a good day proves nothing.** Synthetic or past data,
  simulated fills, and a strategy that may simply be fitted to noise.
- Paper and backtest fills ignore order-book depth and **funding fees**
  (live mode does account for real funding through the income history).
- Paper trading checks stops every `POLL_SECONDS` on the last price, so a fast
  wick may be missed or filled worse than in reality.
- The sentiment lexicon is deliberately simple; sarcasm and context are lost.
  The X API recent-search endpoint requires a paid tier.
- Requiring 7/7 conditions makes signals rare — by design. Expect days without
  any trade.
- Exchange APIs change (e.g. Binance moved conditional orders to the Algo Order
  API in December 2025). Re-test on testnet after any upgrade.
- One-way mode only; one position per pair; USDT-margined perpetuals only.
- Revolut X: the stop-loss lives in the bot, not on the exchange (the REST API
  has no stop orders). A crash or a network outage leaves positions without a
  stop until the bot is back; the take-profit keeps resting on the exchange.

Feedback, issues and contributions are welcome: test it, break it, improve it.

## Résumé en français

Bot de trading open source pour **Binance Futures (USDⓈ-M)**. Règle simple :
il démarre avec **50 USDT**, et s'il les perd, **il s'arrête définitivement**
(fichier `TOMBSTONE.json`, refus de redémarrer).

- Analyse les **40 paires les plus liquides** et ne prend position que si les
  **7 conditions** (EMA ×2, RSI, MACD, Bollinger, volume, ATR) sont alignées, à
  la hausse comme à la baisse.
- Levier adapté à la volatilité, **x5 maximum** ; **2 % de risque** par trade
  (frais inclus), stop à 1,5 ATR, objectif à 2R, break-even puis trailing stop.
- Croise l'analyse technique avec le **sentiment** (Fear & Greed, CoinGecko
  trending, CryptoPanic, X) qui peut **bloquer** un trade à contre-courant.
- **Interface web temps réel** : chaque décision est journalisée et visible.
- Modes **paper** (par défaut), **testnet** et **live** (verrouillé, refuse les
  clés API autorisant les retraits).
- **Backtester** avec exactement les mêmes règles.

**Revolut X** (`EXCHANGE=revolutx`) : même bot en **spot**, paires en **euros**,
**achat uniquement** (pas de vente à découvert), **sans levier**. Le
take-profit est un ordre limite posé chez Revolut ; le stop-loss est surveillé
par le bot toutes les 20 s (l'API n'a pas d'ordre stop), donc le bot doit
tourner en continu. Clé Ed25519 obligatoire, même en simulation. Sur les vraies
bougies Revolut X du dernier mois, la stratégie a **perdu 22 %** alors que le
marché montait : simulation longue obligatoire avant tout argent réel.

Avant toute utilisation : configurez vos propres clés API **sans droit de
retrait**, lancez des backtests, faites tourner le bot en simulation pendant
plusieurs semaines et lisez le code. **Ceci n'est pas un conseil financier.**
Le trading avec effet de levier peut faire perdre la totalité du capital, très vite.

## License

[MIT](LICENSE)
