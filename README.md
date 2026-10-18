# Claude-Trade — V1

Small-capital trading bots with a **survival rule**: each bot starts with a fixed amount
(100 € for stocks) and stops for good if it loses it. The project also ships the
research tools that decide which rules deserve to trade. No rule goes live without
passing them.

| Part | What it does | Status |
|---|---|---|
| **Insider bot** (`invest/insider.py`) | Buys a US stock after its CEO, CFO or another officer bought shares on the open market (SEC Form 4); sells 20 sessions later. Avoids stocks that just signed a financing/contract 8-K or fell on their results | **the only validated rule**: run it on the Trading 212 demo first |
| Event study (`invest/events.py`) | Tests announcement and attention rules on every US stock, delisted ones included, without look-ahead | research |
| Attention data (`invest/attention.py`) | Daily Reddit snapshots (ApeWisdom), Wikipedia pageview history | research |
| Momentum bot (`invest/engine.py`) | The original 7-condition breakout bot | no edge found: paper only |
| Crypto bot (`bot/`) | Binance futures / Revolut X spot | paused, see [README_CRYPTO.md](README_CRYPTO.md) |

Full details, results and limits: [README_INVEST.md](README_INVEST.md).

## What the data says (README_INVEST.md, sections 6-7)

The test universe is every US stock, delisted ones included, from 2024-10 to 2026-10.

| Rule | Edge over the same stocks' usual days, 20 sessions, net of costs |
|---|---|
| CEO or CFO buys ≥ 50 k$ of shares | **+1.7 %** (2024-25) then **+2.3 %** (2026) |
| Officer buys ≥ 100 k$, raising the holding by 20 % | +1.3 % then +2.6 % |
| 8-K contract/financing, results with a 5 % drop | **−2 %**: used as vetoes |
| Momentum breakout (7 conditions), on 2019-2026 | −10 % to −28 % while SPY made +190 % |

These are averages over thousands of trades, and the median trade is close to zero.
With 4 positions the result will swing a lot. Two years of data make one market regime.

## Quick start on a Mac

```bash
git clone https://github.com/marsovic/Claude-Trade.git ~/Claude-Trade && cd ~/Claude-Trade
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env    # then fill at least: MASSIVE_API_KEY, SEC_USER_AGENT,
                        # T212_API_KEY and T212_API_SECRET (demo keys from the app)

# one run in the foreground, to check everything (simulator, no broker)
.venv/bin/python -m invest.invest_bot --strategy insider --broker sim --once

# in the background, on the Trading 212 DEMO account, with its dashboard
scripts/macos/install.sh            # or: scripts/macos/install.sh sim
open http://127.0.0.1:8051          # the dashboard
tail -f ~/Library/Logs/claude-trade/bot.log
scripts/macos/uninstall.sh          # stop both
```

How the background mode works:
- The bot starts at login and restarts if it crashes.
- It keeps the Mac awake while it runs. A closed laptop lid still sleeps the Mac.
- It reads the SEC filings of each session after 22:00 New York time (about 04:00 in
  Paris) and places its orders for the next open (15:30 in Paris). Leave the Mac on
  and plugged in overnight.
- Keep the project outside Documents, Desktop or Downloads. macOS blocks background
  programs there.

Real money stays locked until all of these hold:
- weeks of demo trading;
- `INV_LIVE_TRADING=I_UNDERSTAND_THE_RISKS`;
- for Trading 212, its written OK for a bot (`INV_T212_TERMS_ACKNOWLEDGED=true`).

## Research commands

```bash
python -m invest.events --prices massive --days 730 --out study_events_massive  # rules
python -m invest.attention collect          # daily Reddit snapshot (cron)
python -m invest.attention wiki --all-sec   # Wikipedia pageview history
python -m invest.study --provider tiingo --days 3000   # momentum parameters
python -m invest.backtest --provider massive --days 700 # momentum backtest
python -m pytest                            # 353 tests, no network needed
```

## Layout

```
invest/      stock bots: insider.py, engine.py (momentum), invest_bot.py (loop),
             events.py / edgar.py / attention.py (research), brokers/, tax.py
bot/         crypto bot (paused)
dashboard/   read-only web dashboard (both bots)
scripts/     macOS background services
tests/       pytest suite
study_*/     research outputs kept as evidence (reports, events)
data_*/      cached market and SEC data
```

## Résumé en français

- **V1.** Le bot recommandé achète une action US après un achat d'actions par son PDG, son directeur financier ou un dirigeant (déclaration SEC « Form 4 »). Il revend 20 séances plus tard.
- **C'est la seule règle validée** sur toutes les actions US, disparues comprises (2024-2026) : +1,7 % puis +2,3 % de mieux que la normale par trade, après frais.
- **Lancement en arrière-plan sur Mac** : `scripts/macos/install.sh`, puis le tableau de bord sur http://127.0.0.1:8051.
- **Toujours en démo d'abord.** L'argent réel reste verrouillé : il faut l'accord écrit de Trading 212.

MIT license. This is not investment advice.
