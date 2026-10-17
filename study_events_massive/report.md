# Event study: announcements and attention

Prices: **massive**, 2024-10-17 .. 2026-10-16, 6480 stocks (3596 tradable on a typical day, 10 liquidity groups).
DESIGN before **2026-01-01**, LOCKED from it.

> **Survivorship:** none by construction: every stock that traded each day is in the data, including those delisted since. Only 2 years on the free plan, so the samples are small.

## Verdicts

Decision horizon: 20 sessions, bought at the open after the event was public. Abnormal = vs stocks of the same liquidity group. Edge = abnormal minus what the same stocks earn on ordinary days. Net = edge minus FX and spread.

| definition | verdict | design n | abnormal | same stocks, usual days | edge | net edge | 95 % CI of edge | Holm p | net w/o top 10 | locked n | locked edge | net | p | net w/o top 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| insider_buy | VALIDATED BUY | 2161 | +0.93 % | -0.44 % | +1.36 % | +0.81 % | +0.63 % .. +2.04 % | 0.006 | +0.31 % | 1410 | +1.81 % | +1.26 % | 0.000 | +0.62 % |
| insider_cluster | NOT PROVEN (design effect not significant after Holm) | 354 | +0.57 % | -1.21 % | +1.78 % | +1.21 % | +0.16 % .. +3.48 % | 0.102 | -0.05 % | 217 | +3.46 % | +2.91 % | 0.000 | +1.20 % |
| insider_ceo_cfo | VALIDATED BUY | 685 | +1.45 % | -0.81 % | +2.26 % | +1.71 % | +1.22 % .. +3.27 % | 0.004 | +0.71 % | 436 | +2.89 % | +2.35 % | 0.000 | +1.11 % |
| insider_conviction | VALIDATED BUY | 862 | +0.81 % | -0.97 % | +1.78 % | +1.26 % | +0.75 % .. +2.75 % | 0.008 | +0.35 % | 582 | +3.08 % | +2.57 % | 0.000 | +1.32 % |
| earnings_jump_up | DESIGN ONLY (did not hold in the locked period) | 3096 | +0.78 % | -0.94 % | +1.72 % | +1.20 % | +0.99 % .. +2.31 % | 0.005 | +0.69 % | 2117 | +0.16 % | -0.34 % | 0.294 | -0.88 % |
| earnings_jump_down | VALIDATED VETO | 3262 | -0.10 % | +1.57 % | -1.67 % | -2.20 % | -2.33 % .. -1.03 % | 0.004 | -1.96 % | 2149 | -2.29 % | -2.79 % | 0.000 | -2.21 % |
| agreement_8k | VALIDATED VETO | 4350 | -2.11 % | -0.39 % | -1.72 % | -2.26 % | -2.34 % .. -1.14 % | 0.004 | -2.02 % | 2668 | -1.90 % | -2.42 % | 0.000 | -2.21 % |
| news_jump_8k | NOT PROVEN (design effect not significant after Holm) | 1446 | -2.44 % | -1.99 % | -0.44 % | -0.98 % | -2.16 % .. +1.33 % | 1.000 | -0.59 % | 954 | -0.07 % | -0.61 % | 0.466 | -0.05 % |
| wiki_spike | NOT PROVEN (design effect not significant after Holm) | 477 | -0.69 % | -0.32 % | -0.38 % | -0.76 % | -1.67 % .. +0.95 % | 1.000 | +0.16 % | 303 | +0.14 % | -0.23 % | 0.424 | -1.65 % |
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

* **insider_buy**: an officer or director buys >= $25k of shares on the open market (Form 4 code P, not under a 10b5-1 plan). _Why: insider purchases have predicted returns, mostly in small firms (Lakonishok & Lee 2001; Cohen, Malloy & Pomorski 2012)._
* **insider_cluster**: >= 3 different officers/directors each buy >= $10k within 30 days. _Why: several insiders agreeing is a stronger signal than one._
* **insider_ceo_cfo**: the CEO or the CFO buys >= $50k. _Why: the best-informed insiders._
* **insider_conviction**: an officer/director buys >= $100k AND raises their holding by >= 20 %. _Why: a purchase that is large relative to what the insider already holds._
* **earnings_jump_up**: results (8-K item 2.02) and the stock beats its liquidity group by >= 5 % on the reaction day; bought the next morning. _Why: post-earnings drift: prices under-react to good news (weaker since the 2000s)._
* **earnings_jump_down**: results (8-K item 2.02) and the stock lags its group by >= 5 % on the reaction day. _Why: drift after bad news: a candidate veto, not a buy._
* **agreement_8k**: 8-K item 1.01 (material definitive agreement: contract, deal, financing). _Why: an 'announcement' in the plain sense, with no price filter._
* **news_jump_8k**: 8-K item 1.01, 7.01 or 8.01 (not results) and the stock beats its group by >= 5 % on the reaction day. _Why: a news-driven jump outside earnings._
* **wiki_spike**: Wikipedia pageviews >= 3x their 60-day median and >= 1000 in a day. _Why: attention brings buying pressure that later reverses (Da, Engelberg & Gao 2011)._
* **reddit_spike**: Reddit mentions (ApeWisdom) >= 3x the mean of the 7 previous snapshots and >= 20. _Why: retail buzz; only testable once enough daily snapshots are collected._

## Where events were lost

* insider_buy: raw: 10119, no price history for the ticker: 380, reported price far from the market price: 488, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 2660, repeat within 20 sessions of a counted one: 2980, kept: 3611
* insider_ceo_cfo: raw: 2294, no price history for the ticker: 81, reported price far from the market price: 111, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 617, repeat within 20 sessions of a counted one: 348, kept: 1137
* insider_conviction: raw: 2919, no price history for the ticker: 112, reported price far from the market price: 242, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 596, repeat within 20 sessions of a counted one: 507, kept: 1462
* insider_cluster: raw: 924, no price history for the ticker: 32, reported price far from the market price: 50, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 263, kept: 579
* earnings_jump_up: raw: 31827, no price history for the ticker: 70, reaction below the threshold or not measurable: 26486, after the last price: 3, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 4, repeat within 20 sessions of a counted one: 35, kept: 5229
* earnings_jump_down: raw: 31827, no price history for the ticker: 70, reaction below the threshold or not measurable: 26163, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 102, repeat within 20 sessions of a counted one: 55, kept: 5437
* agreement_8k: raw: 16647, no price history for the ticker: 74, after the last price: 16, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 7671, repeat within 20 sessions of a counted one: 1584, kept: 7302
* news_jump_8k: raw: 53002, no price history for the ticker: 233, reaction below the threshold or not measurable: 49833, after the last price: 1, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 11, repeat within 20 sessions of a counted one: 452, kept: 2472
* wiki_spike: raw: 3583, no price history for the ticker: 1371, before the first price day (not counted in raw): 12447, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 179, repeat within 20 sessions of a counted one: 1230, kept: 803
* reddit_spike: raw: 0

## All horizons (all periods, descriptive only)

| definition | abn 1d | abn 5d | abn 20d | abn 60d | net 1d | net 5d | net 20d | net 60d | delisted in 20d |
|---|---|---|---|---|---|---|---|---|---|
| agreement_8k | -0.43 % | -1.01 % | -2.08 % | -3.52 % | -0.96 % | -1.55 % | -2.62 % | -4.06 % | 4.7 % |
| earnings_jump_down | -0.11 % | -0.03 % | -0.03 % | +0.64 % | -0.63 % | -0.55 % | -0.54 % | +0.12 % | 0.6 % |
| earnings_jump_up | -0.00 % | +0.06 % | +0.28 % | +0.85 % | -0.51 % | -0.45 % | -0.24 % | +0.34 % | 0.5 % |
| insider_buy | +0.45 % | +0.67 % | +0.93 % | +1.16 % | -0.10 % | +0.12 % | +0.38 % | +0.60 % | 1.2 % |
| insider_ceo_cfo | +0.67 % | +1.24 % | +1.53 % | +1.78 % | +0.13 % | +0.70 % | +0.99 % | +1.24 % | 1.6 % |
| insider_cluster | +0.31 % | +0.64 % | +0.81 % | +0.95 % | -0.25 % | +0.08 % | +0.25 % | +0.39 % | 1.6 % |
| insider_conviction | +0.32 % | +0.59 % | +1.09 % | +0.59 % | -0.20 % | +0.07 % | +0.58 % | +0.07 % | 1.4 % |
| news_jump_8k | -0.22 % | -1.10 % | -1.96 % | -2.40 % | -0.76 % | -1.64 % | -2.50 % | -2.94 % | 4.4 % |
| wiki_spike | -0.12 % | -0.39 % | -0.43 % | -2.12 % | -0.49 % | -0.77 % | -0.81 % | -2.50 % | 2.9 % |

## By liquidity (abnormal, 20 sessions, descriptive)

| definition | liquidity | n | mean |
|---|---|---|---|
| agreement_8k | $1-5M/day | 1424 | -2.81 % |
| agreement_8k | $5-50M/day | 2802 | -2.44 % |
| agreement_8k | >$50M/day | 2792 | -1.35 % |
| earnings_jump_down | $1-5M/day | 977 | +0.63 % |
| earnings_jump_down | $5-50M/day | 2231 | -0.35 % |
| earnings_jump_down | >$50M/day | 2203 | +0.00 % |
| earnings_jump_up | $1-5M/day | 913 | +0.84 % |
| earnings_jump_up | $5-50M/day | 2225 | +0.24 % |
| earnings_jump_up | >$50M/day | 2075 | +0.07 % |
| insider_buy | $1-5M/day | 838 | +1.96 % |
| insider_buy | $5-50M/day | 1436 | +0.47 % |
| insider_buy | >$50M/day | 1297 | +0.79 % |
| insider_ceo_cfo | $1-5M/day | 244 | +3.58 % |
| insider_ceo_cfo | $5-50M/day | 479 | +1.04 % |
| insider_ceo_cfo | >$50M/day | 398 | +0.86 % |
| insider_cluster | $1-5M/day | 143 | +1.49 % |
| insider_cluster | $5-50M/day | 240 | +0.34 % |
| insider_cluster | >$50M/day | 188 | +0.88 % |
| insider_conviction | $1-5M/day | 266 | +2.95 % |
| insider_conviction | $5-50M/day | 555 | +0.11 % |
| insider_conviction | >$50M/day | 623 | +1.17 % |
| news_jump_8k | $1-5M/day | 501 | -0.78 % |
| news_jump_8k | $5-50M/day | 1065 | -2.90 % |
| news_jump_8k | >$50M/day | 834 | -1.45 % |
| wiki_spike | $1-5M/day | 10 | -5.25 % |
| wiki_spike | $5-50M/day | 109 | -1.18 % |
| wiki_spike | >$50M/day | 661 | -0.24 % |

## Per year (abnormal, 20 sessions)

| definition | year | n | mean | median |
|---|---|---|---|---|
| agreement_8k | 2024 | 658 | -1.22 % | -1.99 % |
| agreement_8k | 2025 | 3692 | -2.27 % | -2.07 % |
| agreement_8k | 2026 | 2668 | -2.03 % | -1.82 % |
| earnings_jump_down | 2024 | 485 | -1.07 % | -1.92 % |
| earnings_jump_down | 2025 | 2777 | +0.07 % | -0.51 % |
| earnings_jump_down | 2026 | 2149 | +0.09 % | -1.15 % |
| earnings_jump_up | 2024 | 488 | +1.01 % | -0.72 % |
| earnings_jump_up | 2025 | 2608 | +0.74 % | -0.68 % |
| earnings_jump_up | 2026 | 2117 | -0.46 % | -1.88 % |
| insider_buy | 2024 | 323 | -0.73 % | -0.89 % |
| insider_buy | 2025 | 1838 | +1.22 % | -0.25 % |
| insider_buy | 2026 | 1410 | +0.94 % | +0.27 % |
| insider_ceo_cfo | 2024 | 98 | +0.86 % | +1.00 % |
| insider_ceo_cfo | 2025 | 587 | +1.55 % | -0.42 % |
| insider_ceo_cfo | 2026 | 436 | +1.65 % | +0.54 % |
| insider_cluster | 2024 | 47 | +1.27 % | +0.72 % |
| insider_cluster | 2025 | 307 | +0.46 % | -0.49 % |
| insider_cluster | 2026 | 217 | +1.20 % | +0.45 % |
| insider_conviction | 2024 | 129 | -1.24 % | -1.75 % |
| insider_conviction | 2025 | 733 | +1.16 % | -0.63 % |
| insider_conviction | 2026 | 582 | +1.52 % | +0.77 % |
| news_jump_8k | 2024 | 177 | -1.45 % | -1.76 % |
| news_jump_8k | 2025 | 1269 | -2.58 % | -3.10 % |
| news_jump_8k | 2026 | 954 | -1.23 % | -2.35 % |
| wiki_spike | 2024 | 66 | +1.17 % | -1.72 % |
| wiki_spike | 2025 | 411 | -0.99 % | -1.83 % |
| wiki_spike | 2026 | 303 | -0.02 % | -0.94 % |

## How to read this

* A VALIDATED verdict is a reason to paper-trade the rule, not to bet on it: effects like these fade once published and costs on small stocks are uncertain.
* NOT PROVEN means the data cannot tell the effect from luck; it does not prove there is none.
* The placebo answers 'is the DAY special, or just the stock?'. A sample of stocks that rose anyway beats zero on any day, but not its own random days.
