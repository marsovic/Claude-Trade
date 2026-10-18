# Changelog

## 1.0.0 — 2026-10-18

First release.

### Bots
- **Insider bot (`INV_STRATEGY=insider`)**:
  - buys after SEC Form 4 open-market purchases by the CEO, the CFO or officers, and sells 20 sessions later;
  - vetoes 8-K contract/financing filings and 5 % earnings drops;
  - equal-weight slots, disaster stop;
  - runs on the simulator, the Trading 212 demo and Trading 212 live (locked).
- **Momentum bot**: the original 7-condition breakout strategy, kept for paper trading. No edge was found on 2019-2026.
- **Crypto bot** (Binance futures, Revolut X spot): paused, unchanged.
- **macOS background services** (`scripts/macos/install.sh`): bot and dashboard run under launchd, start at login, restart after a crash, and keep the Mac awake.

### Research tools
- `invest.events`: event study of announcements and attention, with:
  - a point-in-time universe (Massive grouped bars, as traded plus the split list);
  - entry at the open after publication;
  - liquidity-matched benchmarks, and a placebo on the same stocks' ordinary days;
  - week block bootstrap, Holm correction and design/locked periods;
  - a check that each result survives without its 10 biggest contributors.
- `invest.edgar`:
  - SEC insider data sets (point-in-time tickers);
  - 8-K items with acceptance times, read in each company file's own time format;
  - a daily Form 4 feed.
- `invest.attention`: daily ApeWisdom snapshots and the Wikipedia pageview history found via Wikidata.
- `invest.study`: walk-forward study of the momentum parameters.

### Clean-up
- The root README is now the project front page; the crypto bot's guide moved to `README_CRYPTO.md`.
- Removed the legacy v1 spot bot, two backtest outputs made with pre-audit code, and the redundant adjusted copy of the Massive bars (`data_events/grouped`, superseded by `grouped_raw`).
