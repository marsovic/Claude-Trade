# Event study: announcements and attention

Prices: **csv**, 2018-07-30 .. 2026-10-16, 133 stocks (127 tradable on a typical day, 5 liquidity groups).
DESIGN before **2023-08-01**, LOCKED from it.

> **Survivorship:** the price files hold the stocks someone chose TODAY; companies that failed or were delisted are missing, which flatters any rule. Trust only effects that also beat the placebo.

## Verdicts

Decision horizon: 20 sessions, bought at the open after the event was public. Abnormal = vs stocks of the same liquidity group. Edge = abnormal minus what the same stocks earn on ordinary days. Net = edge minus FX and spread.

| definition | verdict | design n | abnormal | same stocks, usual days | edge | net edge | 95 % CI of edge | Holm p | net w/o top 10 | locked n | locked edge | net | p | net w/o top 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| insider_buy | NOT PROVEN (design effect not significant after Holm) | 276 | -0.30 % | -0.16 % | -0.14 % | -0.49 % | -1.41 % .. +1.11 % | 1.000 | +0.98 % | 173 | +0.56 % | +0.21 % | 0.251 | -1.78 % |
| insider_cluster | NOT PROVEN (only 23 design events, 100 needed) | 23 | -1.02 % | -2.38 % | +1.36 % | +1.01 % | -2.97 % .. +5.83 % | 1.000 | n/a | 10 | +0.18 % | -0.17 % | 0.482 | n/a |
| insider_ceo_cfo | NOT PROVEN (only 70 design events, 100 needed) | 70 | -0.90 % | -0.46 % | -0.44 % | -0.80 % | -2.80 % .. +1.86 % | 1.000 | +1.91 % | 52 | -0.04 % | -0.39 % | 0.499 | +3.23 % |
| insider_conviction | NOT PROVEN (design effect not significant after Holm) | 154 | -1.51 % | -0.77 % | -0.74 % | -1.10 % | -2.32 % .. +0.77 % | 1.000 | +0.42 % | 70 | +2.80 % | +2.45 % | 0.020 | -2.06 % |
| earnings_jump_up | NOT PROVEN (carried by a handful of stocks: without the top 10 it does not pay its costs) | 496 | +0.40 % | -1.02 % | +1.42 % | +1.06 % | +0.42 % .. +2.47 % | 0.031 | -0.30 % | 422 | +1.90 % | +1.55 % | 0.003 | -0.07 % |
| earnings_jump_down | NOT PROVEN (design effect not significant after Holm) | 394 | -0.18 % | +0.79 % | -0.97 % | -1.33 % | -2.30 % .. +0.34 % | 0.914 | +0.06 % | 431 | -2.85 % | -3.21 % | 0.000 | -1.85 % |
| agreement_8k | DESIGN ONLY veto (did not hold in the locked period) | 562 | -1.46 % | +0.36 % | -1.83 % | -2.19 % | -2.91 % .. -0.74 % | 0.029 | -0.73 % | 369 | -1.12 % | -1.47 % | 0.064 | -0.17 % |
| news_jump_8k | NOT PROVEN (design effect not significant after Holm) | 122 | +0.06 % | -0.38 % | +0.44 % | +0.08 % | -2.79 % .. +3.73 % | 1.000 | -3.84 % | 92 | -1.81 % | -2.17 % | 0.146 | +0.51 % |

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

## Where events were lost

* insider_buy: raw: 49010, no price history for the ticker: 48184, reported price far from the market price: 79, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 15, repeat within 20 sessions of a counted one: 282, kept: 450
* insider_ceo_cfo: raw: 10180, no price history for the ticker: 9987, reported price far from the market price: 6, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 1, repeat within 20 sessions of a counted one: 64, kept: 122
* insider_conviction: raw: 12577, no price history for the ticker: 12235, reported price far from the market price: 34, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 8, repeat within 20 sessions of a counted one: 75, kept: 225
* insider_cluster: raw: 4527, no price history for the ticker: 4483, reported price far from the market price: 9, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 1, repeat within 20 sessions of a counted one: 1, kept: 33
* earnings_jump_up: raw: 3941, no price history for the ticker: 135, reaction below the threshold or not measurable: 2880, repeat within 20 sessions of a counted one: 7, kept: 919
* earnings_jump_down: raw: 3941, no price history for the ticker: 135, reaction below the threshold or not measurable: 2967, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 2, repeat within 20 sessions of a counted one: 11, kept: 826
* agreement_8k: raw: 1279, no price history for the ticker: 69, not tradable at t0 (no bar, price < $2 or volume < $1M/day): 25, repeat within 20 sessions of a counted one: 248, kept: 937
* news_jump_8k: raw: 4855, no price history for the ticker: 260, reaction below the threshold or not measurable: 4349, repeat within 20 sessions of a counted one: 32, kept: 214

## All horizons (all periods, descriptive only)

| definition | abn 1d | abn 5d | abn 20d | abn 60d | net 1d | net 5d | net 20d | net 60d | delisted in 20d |
|---|---|---|---|---|---|---|---|---|---|
| agreement_8k | -0.24 % | -0.66 % | -1.32 % | -0.69 % | -0.59 % | -1.01 % | -1.68 % | -1.05 % | 0.6 % |
| earnings_jump_down | -0.19 % | -0.47 % | -0.60 % | -0.07 % | -0.54 % | -0.83 % | -0.95 % | -0.42 % | 0.1 % |
| earnings_jump_up | +0.29 % | +0.02 % | +0.53 % | +1.12 % | -0.07 % | -0.33 % | +0.17 % | +0.76 % | 0.1 % |
| insider_buy | +0.25 % | +0.14 % | -0.37 % | -1.28 % | -0.11 % | -0.22 % | -0.73 % | -1.63 % | 0.2 % |
| insider_ceo_cfo | +0.54 % | +0.08 % | -1.89 % | -1.98 % | +0.19 % | -0.28 % | -2.24 % | -2.34 % | 0.0 % |
| insider_cluster | +0.43 % | -0.20 % | -1.85 % | -3.49 % | +0.08 % | -0.55 % | -2.20 % | -3.84 % | 0.0 % |
| insider_conviction | +0.12 % | +0.05 % | -0.75 % | -3.03 % | -0.24 % | -0.30 % | -1.10 % | -3.38 % | 0.4 % |
| news_jump_8k | -0.20 % | -0.65 % | +0.03 % | +3.93 % | -0.56 % | -1.01 % | -0.33 % | +3.57 % | 0.0 % |

## By liquidity (abnormal, 20 sessions, descriptive)

| definition | liquidity | n | mean |
|---|---|---|---|
| agreement_8k | $1-5M/day | 2 | +6.14 % |
| agreement_8k | $5-50M/day | 37 | -3.20 % |
| agreement_8k | >$50M/day | 892 | -1.26 % |
| earnings_jump_down | $5-50M/day | 21 | -2.49 % |
| earnings_jump_down | >$50M/day | 804 | -0.55 % |
| earnings_jump_up | $1-5M/day | 1 | -1.17 % |
| earnings_jump_up | $5-50M/day | 39 | +6.00 % |
| earnings_jump_up | >$50M/day | 878 | +0.29 % |
| insider_buy | $5-50M/day | 14 | +2.05 % |
| insider_buy | >$50M/day | 435 | -0.45 % |
| insider_ceo_cfo | $5-50M/day | 6 | -0.97 % |
| insider_ceo_cfo | >$50M/day | 116 | -1.93 % |
| insider_cluster | >$50M/day | 33 | -1.85 % |
| insider_conviction | $5-50M/day | 7 | -0.19 % |
| insider_conviction | >$50M/day | 217 | -0.77 % |
| news_jump_8k | $1-5M/day | 1 | +14.54 % |
| news_jump_8k | $5-50M/day | 12 | +0.92 % |
| news_jump_8k | >$50M/day | 201 | -0.10 % |

## Per year (abnormal, 20 sessions)

| definition | year | n | mean | median |
|---|---|---|---|---|
| agreement_8k | 2018 | 35 | +0.70 % | +1.30 % |
| agreement_8k | 2019 | 100 | -1.21 % | -1.31 % |
| agreement_8k | 2020 | 147 | -1.92 % | -2.91 % |
| agreement_8k | 2021 | 138 | -0.88 % | -1.67 % |
| agreement_8k | 2022 | 96 | -1.92 % | -0.75 % |
| agreement_8k | 2023 | 87 | -2.06 % | -2.33 % |
| agreement_8k | 2024 | 108 | -0.00 % | -1.36 % |
| agreement_8k | 2025 | 129 | -0.67 % | -1.45 % |
| agreement_8k | 2026 | 91 | -3.05 % | -3.67 % |
| earnings_jump_down | 2018 | 22 | +0.19 % | +1.05 % |
| earnings_jump_down | 2019 | 70 | +0.29 % | -0.74 % |
| earnings_jump_down | 2020 | 69 | -1.32 % | -2.05 % |
| earnings_jump_down | 2021 | 73 | +0.86 % | -1.62 % |
| earnings_jump_down | 2022 | 103 | -1.06 % | -0.47 % |
| earnings_jump_down | 2023 | 98 | +0.37 % | -1.25 % |
| earnings_jump_down | 2024 | 127 | -3.15 % | -3.21 % |
| earnings_jump_down | 2025 | 149 | -0.89 % | -1.67 % |
| earnings_jump_down | 2026 | 114 | +1.03 % | -0.77 % |
| earnings_jump_up | 2018 | 31 | +0.21 % | +2.11 % |
| earnings_jump_up | 2019 | 87 | +1.18 % | -0.45 % |
| earnings_jump_up | 2020 | 91 | -0.46 % | -2.06 % |
| earnings_jump_up | 2021 | 79 | +0.98 % | +0.50 % |
| earnings_jump_up | 2022 | 132 | +0.47 % | +0.32 % |
| earnings_jump_up | 2023 | 134 | +0.88 % | +0.08 % |
| earnings_jump_up | 2024 | 146 | +2.24 % | -1.03 % |
| earnings_jump_up | 2025 | 121 | -0.64 % | -1.57 % |
| earnings_jump_up | 2026 | 97 | -0.91 % | -2.92 % |
| insider_buy | 2018 | 20 | +0.52 % | +3.89 % |
| insider_buy | 2019 | 40 | -0.80 % | -2.39 % |
| insider_buy | 2020 | 50 | -3.41 % | -5.00 % |
| insider_buy | 2021 | 52 | +0.97 % | +0.14 % |
| insider_buy | 2022 | 80 | +0.43 % | -0.67 % |
| insider_buy | 2023 | 52 | +1.20 % | +0.25 % |
| insider_buy | 2024 | 44 | +0.41 % | -2.20 % |
| insider_buy | 2025 | 61 | +0.26 % | +0.35 % |
| insider_buy | 2026 | 50 | -3.14 % | -2.25 % |
| insider_ceo_cfo | 2018 | 10 | +3.45 % | +5.32 % |
| insider_ceo_cfo | 2019 | 12 | -0.19 % | -2.60 % |
| insider_ceo_cfo | 2020 | 9 | -7.02 % | -8.14 % |
| insider_ceo_cfo | 2021 | 10 | +0.06 % | -2.80 % |
| insider_ceo_cfo | 2022 | 20 | +0.25 % | +0.75 % |
| insider_ceo_cfo | 2023 | 13 | -0.11 % | -0.48 % |
| insider_ceo_cfo | 2024 | 15 | -1.37 % | -3.45 % |
| insider_ceo_cfo | 2025 | 11 | -2.94 % | -1.17 % |
| insider_ceo_cfo | 2026 | 22 | -6.83 % | -3.70 % |
| insider_cluster | 2018 | 2 | +3.25 % | +3.25 % |
| insider_cluster | 2019 | 5 | +1.24 % | +1.99 % |
| insider_cluster | 2020 | 2 | -1.93 % | -1.93 % |
| insider_cluster | 2021 | 7 | +0.44 % | -3.22 % |
| insider_cluster | 2022 | 6 | -3.43 % | -6.54 % |
| insider_cluster | 2023 | 1 | -14.87 % | -14.87 % |
| insider_cluster | 2025 | 3 | +0.38 % | -3.70 % |
| insider_cluster | 2026 | 7 | -5.52 % | -5.66 % |
| insider_conviction | 2018 | 8 | -5.34 % | -1.61 % |
| insider_conviction | 2019 | 24 | -1.39 % | -2.39 % |
| insider_conviction | 2020 | 25 | -4.99 % | -4.81 % |
| insider_conviction | 2021 | 31 | +0.16 % | -0.28 % |
| insider_conviction | 2022 | 52 | -0.11 % | -1.57 % |
| insider_conviction | 2023 | 22 | -0.40 % | +1.22 % |
| insider_conviction | 2024 | 12 | +1.57 % | -0.32 % |
| insider_conviction | 2025 | 24 | +2.66 % | +3.19 % |
| insider_conviction | 2026 | 26 | -1.54 % | -2.13 % |
| news_jump_8k | 2018 | 5 | -9.19 % | -14.39 % |
| news_jump_8k | 2019 | 11 | +5.96 % | +5.17 % |
| news_jump_8k | 2020 | 37 | +0.38 % | -2.90 % |
| news_jump_8k | 2021 | 28 | +1.46 % | +0.94 % |
| news_jump_8k | 2022 | 26 | -1.54 % | -0.25 % |
| news_jump_8k | 2023 | 25 | -3.94 % | -5.33 % |
| news_jump_8k | 2024 | 20 | +5.01 % | +1.04 % |
| news_jump_8k | 2025 | 33 | -2.61 % | -5.20 % |
| news_jump_8k | 2026 | 29 | +1.94 % | +1.38 % |

## How to read this

* A VALIDATED verdict is a reason to paper-trade the rule, not to bet on it: effects like these fade once published and costs on small stocks are uncertain.
* NOT PROVEN means the data cannot tell the effect from luck; it does not prove there is none.
* The placebo answers 'is the DAY special, or just the stock?'. A sample of stocks that rose anyway beats zero on any day, but not its own random days.
