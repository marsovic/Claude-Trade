# Event study: announcements and attention

Prices: **csv**, 2018-07-30 .. 2026-10-16, 133 stocks (127 tradable on a typical day, 5 liquidity groups).
DESIGN before **2023-08-01**, LOCKED from it.

> **Survivorship:** the price files hold the stocks someone chose TODAY; companies that failed or were delisted are missing, which flatters any rule. Trust only effects that also beat the placebo.

## Verdicts

Decision horizon: 20 sessions, bought at the open after the event was public. Abnormal = vs stocks of the same liquidity group. Edge = abnormal minus what the same stocks earn on ordinary days. Net = edge minus FX and spread.

| definition | verdict | design n | abnormal | same stocks, usual days | edge | net edge | 95 % CI of edge | Holm p | net w/o top 10 | locked n | locked edge | net | p | net w/o top 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| wiki_spike | NOT PROVEN (design effect not significant after Holm) | 279 | -0.25 % | +0.58 % | -0.83 % | -1.18 % | -2.39 % .. +0.83 % | 0.329 | +0.59 % | 236 | +0.15 % | -0.20 % | 0.436 | -2.00 % |
| reddit_spike | NOT PROVEN (only 0 design events, 100 needed) | 0 | n/a | n/a | n/a | n/a | n/a .. n/a | n/a | n/a | 0 | n/a | n/a | n/a | n/a |

## Rule

Fixed before the data was seen:

* decision metric: the EDGE = mean abnormal return over 20 sessions (bought at the
  open after the event was public, compared with stocks of the same liquidity group)
  MINUS what the same stocks earn on their ordinary days (median of the placebo: same
  stocks, random days within +-126 sessions, away from their events).  This
  takes out the drift of stocks that rose anyway, i.e. survivorship;
* DESIGN: at least 100 events, Holm-adjusted two-sided p < 0.05 on the edge, and the
  abnormal mean beyond its placebo (above the 95th percentile of random-day means for a
  buy, below the 5th for a veto);
* without the 10 stocks that contribute most to it, a buy must still be positive NET of
  costs and a veto still negative (in both periods): a result carried by a few stars is
  not an effect;
* LOCKED (the later period): at least 50 events, same sign, one-sided p < 0.05;
* a BUY signal must also be positive NET of costs (edge - FX - spread) in both periods;
* verdicts: VALIDATED BUY, VALIDATED VETO, DESIGN ONLY (did not hold in LOCKED), NOT PROVEN.

## Definitions

* **wiki_spike**: Wikipedia pageviews >= 3x their 60-day median and >= 1000 in a day. _Why: attention brings buying pressure that later reverses (Da, Engelberg & Gao 2011)._
* **reddit_spike**: Reddit mentions (ApeWisdom) >= 3x the mean of the 7 previous snapshots and >= 20. _Why: retail buzz; only testable once enough daily snapshots are collected._

## Where events were lost

* wiki_spike: raw: 13362, no price history for the ticker: 11646, before the first price day (not counted in raw): 2668, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 83, repeat within 20 sessions of a counted one: 1113, kept: 520
* reddit_spike: raw: 0

## All horizons (all periods, descriptive only)

| definition | abn 1d | abn 5d | abn 20d | abn 60d | net 1d | net 5d | net 20d | net 60d | delisted in 20d |
|---|---|---|---|---|---|---|---|---|---|
| wiki_spike | -0.06 % | +0.02 % | -0.59 % | -0.31 % | -0.41 % | -0.33 % | -0.94 % | -0.66 % | 1.0 % |

## By liquidity (abnormal, 20 sessions, descriptive)

| definition | liquidity | n | mean |
|---|---|---|---|
| wiki_spike | $5-50M/day | 3 | +6.47 % |
| wiki_spike | >$50M/day | 512 | -0.63 % |

## Per year (abnormal, 20 sessions)

| definition | year | n | mean | median |
|---|---|---|---|---|
| wiki_spike | 2018 | 17 | +1.78 % | +0.36 % |
| wiki_spike | 2019 | 56 | +0.39 % | -0.47 % |
| wiki_spike | 2020 | 65 | -2.17 % | -5.70 % |
| wiki_spike | 2021 | 52 | +0.26 % | -0.51 % |
| wiki_spike | 2022 | 54 | +1.52 % | +2.64 % |
| wiki_spike | 2023 | 56 | -2.25 % | -2.04 % |
| wiki_spike | 2024 | 72 | +0.83 % | -1.20 % |
| wiki_spike | 2025 | 84 | -1.23 % | -0.63 % |
| wiki_spike | 2026 | 59 | -2.42 % | -4.43 % |

## How to read this

* A VALIDATED verdict is a reason to paper-trade the rule, not to bet on it: effects like these fade once published and costs on small stocks are uncertain.
* NOT PROVEN means the data cannot tell the effect from luck; it does not prove there is none.
* The placebo answers 'is the DAY special, or just the stock?'. A sample of stocks that rose anyway beats zero on any day, but not its own random days.
