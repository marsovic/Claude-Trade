# Walk-forward study of the momentum bot, 2019-08-12 to 2026-10-16

## Verdict

KEEP THE DEFAULTS. 33 variants were written down before looking at any result, 26 of them as possible improvements. None of those passed all the tests on the design years 2019-2023 once we account for having tried 26 ideas at once, so there is no honest reason to change a parameter. No variant was looked at on the locked years 2024-2026. Against random entries among the same top-ranked momentum stocks (same exits, costs and market adjustment) the 7/7 signal earned -0.05 R more per trade on the design years (one-sided p 0.727): the entry signal is not shown to beat a random pick among top-ranked stocks, so it is not proven. The baseline loses money (net of costs) on the design years. The equal-weight universe grew 35% a year in this sample: every result is measured net of that drift, but a longer hold (wider stop or trail) is the first thing a bull market flatters.

*En français :* GARDER LES RÉGLAGES PAR DÉFAUT. Aucune des variantes déclarées à l'avance ne passe tous les tests sur les années de conception : rien ne justifie de changer un paramètre.

### Warnings

- the equal-weight universe grew 35% a year in this sample (a bull market, and today's universe is made of survivors): any variant that holds longer earns that drift. Every variant is judged net of beta x the universe return over its holding window, but treat a wider stop or trail with extra caution
- the LOCKED segment has only 139 baseline events (< 150): too few to confirm anything, so no variant will be evaluated there and nothing will be recommended

## How to read this

- One event = one stock on one day when all 7 conditions are true. There are no slot or cash limits, so thousands of events are compared instead of a few dozen real trades.
- R = profit divided by the money risked at entry (entry price minus first stop), after fees and slippage. +1 R means a gain equal to the initial risk.
- 'dR' = the average R of a variant minus the average R of the baseline, where every trade's R is first reduced by the market's move over its holding window (beta x the equal-weight universe return from the signal to the exit). Without that, a wider stop or trail would win in any rising market just by keeping the money invested longer. 'raw dR' is the same difference without the adjustment. A bigger average alone proves nothing: a variant counts only if it passes every test of the decision rule at the bottom of this page.
- The 95% interval comes from resampling whole weeks (events of the same week move together). The p-value is corrected for having tried 26 ideas at once (Holm-Bonferroni over the 26 adoptable variants (declared 'better'); the 6 'worse' diagnostics can never be adopted and are corrected among themselves).

## What was done

- Data: 132 stocks, 1806 sessions of signals (2019-08-12 to 2026-10-16).
- 549 raw 7/7 signals gave 293 counted events (one open position per symbol: True).
- DESIGN years 2019-2023 (151 events), LOCKED years 2024-2026 (139 events). Variants are chosen on the design years only. 3 design events whose trade ran into the locked years were removed, so that no locked price can influence the choice.
- 33 variants were written down before looking: 26 possible improvements, 6 looser settings declared 'worse' (they only show whether a default earns its keep and can never be adopted), and 1 self-test(s). Bootstrap: 10000 resamples of whole weeks, seed 12345.
- For a variant with another stop distance, R is measured against the BASELINE initial risk, so that a wider stop does not look better only because its unit is bigger.
- Decision metric: R net of beta x the equal-weight universe return over each holding window. The equal-weight universe grew 35.3% a year over the analysed years.
- Regime: SPY trend plus the VIX gates (VIX from FRED 2018-07-30 to 2026-10-15, VIX3M 2018-07-30 to 2026-10-15).
- Costs: fee preset 'sim', reference equity 100 EUR.
- A variant must also gain at least 0.05 R and change at least 30 events (a bootstrap that rests on a handful of events is unreliable), and the locked years need at least 150 events to confirm anything.
- events.csv: the entry-quality columns of the locked-year rows are left empty on purpose, so that the locked years stay a single untouched confirmation.
- The baseline is YOUR configuration. Settings that differ from the code defaults: {'etp_enabled': True}

## Baseline: all 7/7 events with the current settings

| segment | events | win rate | avg R | payoff | profit factor | median hold (sessions) | effective n |
|---|---|---|---|---|---|---|---|
| all years | 293 | 0.392 | +0.012 | 1.59 | 1.03 | 11 | 262 |
| design years | 151 | 0.364 | -0.059 | | | | 151 |

Effective n is smaller than the number of events when events of the same week are correlated (intra-week correlation 0.09, 2.4 events per active week).

Average R per calendar year. The locked years are shown for information only; nothing was tuned on them:

| year | events | avg R | win rate | part |
|---|---|---|---|---|
| 2019 | 21 | +0.282 | 0.52 | design |
| 2020 | 27 | +0.051 | 0.41 | design |
| 2021 | 50 | +0.008 | 0.38 | design |
| 2022 | 3 | -0.641 | 0.00 | design |
| 2023 | 53 | -0.296 | 0.28 | design |
| 2024 | 64 | +0.163 | 0.48 | locked |
| 2025 | 43 | +0.371 | 0.47 | locked |
| 2026 | 32 | -0.405 | 0.25 | locked |

Exits of the baseline events: stop 138, trailing stop 113, stop (gap) 33, rank exit 9

## Random-entry placebo

- design years, years weighted like the baseline: net of the market, baseline -0.238 R vs random entries -0.190 R, difference -0.047 R (95% interval -0.189 to +0.105, one-sided p 0.727); raw R -0.059 vs +0.021; mean hold 13.5 vs 14.4 sessions.
- design years: net of the market, baseline -0.238 R vs random entries -0.179 R, difference -0.059 R (95% interval -0.205 to +0.078, one-sided p 0.795); raw R -0.059 vs -0.001; mean hold 13.5 vs 14.0 sessions.
- INFORMATION ONLY, includes the locked years, years weighted like the baseline: net of the market, baseline -0.169 R vs random entries -0.090 R, difference -0.079 R (95% interval -0.200 to +0.046, one-sided p 0.896); raw R +0.012 vs +0.104; mean hold 13.3 vs 14.1 sessions.
- INFORMATION ONLY, includes the locked years: net of the market, baseline -0.169 R vs random entries -0.084 R, difference -0.085 R (95% interval -0.203 to +0.039, one-sided p 0.915); raw R +0.012 vs +0.087; mean hold 13.3 vs 13.8 sessions.

Random entries are drawn among the stocks the signal itself starts from (eligible, regime on, in the entry top of the momentum ranking), with the same exits, costs and market adjustment, so the comparison is about the 7/7 timing and not about holding time. The interval resamples whole weeks for both sides; a p-value above 0.05 means the signal is not shown to beat a random pick among top-ranked stocks.

## Variants on the design years

'keep' = share of the baseline events that remain. 'years' = design years where the difference has the right sign. 'decl.' = the difference when only the best scored event of each week is kept.

| variant | hypothesis | n | keep | avg R | dR | raw dR | 95% interval | p (Holm) | p (Holm, 4-week blocks) | years | decl. | result |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| stop_atr_mult=2.5 | worse | 163 | 1.08 | -0.041 | +0.051 | +0.028 | [-0.023, +0.123] | 1.000 | 0.892 | 4/4 | +0.027 | no clear difference: default not shown to help |
| stop_atr_mult=3.5 | better | 142 | 0.94 | -0.018 | -0.037 | +0.040 | [-0.104, +0.045] | 1.000 | 1.000 | 1/4 | +0.010 | not higher |
| stop_atr_mult=4 | better | 134 | 0.89 | -0.017 | -0.107 | +0.039 | [-0.228, +0.021] | 1.000 | 1.000 | 2/4 | -0.072 | not higher |
| trail_atr_mult=3 | better | 151 | 1.00 | -0.059 | +0.000 | +0.000 | [+0.000, +0.000] | - | - | - | +0.000 | control: identical to the baseline |
| trail_atr_mult=4 | better | 136 | 0.90 | -0.035 | -0.107 | +0.025 | [-0.224, +0.017] | 1.000 | 1.000 | 1/4 | -0.044 | not higher |
| trail_atr_mult=5 | better | 125 | 0.83 | +0.031 | -0.120 | +0.090 | [-0.267, +0.040] | 1.000 | 1.000 | 1/4 | -0.155 | not higher |
| top_pct_entry=0.05 | better | 92 | 0.61 | -0.013 | +0.020 | +0.047 | [-0.094, +0.135] | 1.000 | 1.000 | 2/4 | +0.024 | p(Holm) > 0.05 |
| top_pct_entry=0.075 | better | 120 | 0.79 | -0.030 | -0.013 | +0.029 | [-0.081, +0.054] | 1.000 | 1.000 | 1/4 | +0.009 | not higher |
| top_pct_entry=0.15 | worse | 218 | 1.44 | -0.068 | -0.033 | -0.009 | [-0.120, +0.058] | 1.000 | 1.000 | 1/4 | +0.013 | no clear difference: default not shown to help |
| top_pct_entry=0.2 | worse | 295 | 1.95 | -0.044 | -0.013 | +0.015 | [-0.124, +0.095] | 1.000 | 1.000 | 1/4 | +0.006 | no clear difference: default not shown to help |
| near_high_min=0.8 | worse | 151 | 1.00 | -0.059 | +0.000 | +0.000 | [+0.000, +0.000] | 1.000 | 1.000 | 0/4 | +0.000 | same events as the baseline (threshold not binding) |
| near_high_min=0.9 | better | 150 | 0.99 | -0.058 | -0.003 | +0.001 | [-0.017, +0.007] | 1.000 | 1.000 | 1/4 | -0.003 | not higher |
| near_high_min=0.95 | better | 146 | 0.97 | -0.039 | +0.009 | +0.020 | [-0.018, +0.036] | 1.000 | 1.000 | 1/4 | +0.003 | p(Holm) > 0.05 |
| volume_factor=1 | worse | 228 | 1.51 | -0.057 | -0.010 | +0.002 | [-0.112, +0.091] | 1.000 | 1.000 | 2/4 | -0.047 | no clear difference: default not shown to help |
| volume_factor=1.6 | better | 99 | 0.66 | -0.188 | -0.109 | -0.128 | [-0.245, +0.018] | 1.000 | 1.000 | 1/4 | -0.044 | not higher |
| volume_factor=2 | better | 61 | 0.40 | -0.120 | -0.080 | -0.061 | [-0.272, +0.109] | 1.000 | 1.000 | 2/4 | -0.067 | not higher |
| max_extension=0.15 | better | 110 | 0.73 | -0.012 | +0.058 | +0.048 | [-0.026, +0.152] | 1.000 | 1.000 | 3/4 | +0.081 | p(Holm) > 0.05 |
| max_extension=0.2 | better | 134 | 0.89 | -0.013 | +0.046 | +0.046 | [+0.006, +0.093] | 0.546 | 0.442 | 4/4 | +0.048 | p(Holm) > 0.05 |
| max_extension=0.35 | worse | 161 | 1.07 | -0.041 | +0.024 | +0.018 | [-0.015, +0.080] | 1.000 | 1.000 | 2/4 | +0.010 | no clear difference: default not shown to help |
| rsi_max=70 | better | 104 | 0.69 | -0.104 | -0.014 | -0.045 | [-0.117, +0.081] | 1.000 | 1.000 | 3/4 | +0.063 | not higher |
| rsi_max=75 | better | 143 | 0.95 | -0.068 | -0.012 | -0.009 | [-0.051, +0.027] | 1.000 | 1.000 | 2/4 | -0.010 | not higher |
| gap_atr<=0.25 | better | 132 | 0.87 | -0.099 | -0.003 | -0.040 | [-0.093, +0.085] | 1.000 | 1.000 | 2/4 | -0.008 | not higher |
| gap_atr<=0.5 | better | 147 | 0.97 | -0.050 | +0.009 | +0.010 | [-0.031, +0.054] | 1.000 | 1.000 | 1/4 | +0.033 | p(Holm) > 0.05 |
| gap_atr<=1 | better | 151 | 1.00 | -0.064 | -0.003 | -0.005 | [-0.012, +0.006] | 1.000 | 1.000 | 1/4 | -0.004 | not higher |
| ext_ema20_atr<=2.5 | better | 108 | 0.72 | -0.052 | +0.042 | +0.008 | [-0.072, +0.147] | 1.000 | 1.000 | 3/4 | +0.042 | p(Holm) > 0.05 |
| ext_ema20_atr<=3.5 | better | 147 | 0.97 | -0.091 | -0.015 | -0.032 | [-0.047, +0.012] | 1.000 | 1.000 | 1/4 | -0.001 | not higher |
| ext_ema20_atr<=4.5 | better | 151 | 1.00 | -0.059 | +0.000 | +0.000 | [+0.000, +0.000] | 1.000 | 1.000 | 0/4 | +0.000 | same events as the baseline (threshold not binding) |
| no_event_bar(1) | better | 150 | 0.99 | -0.059 | +0.000 | -0.000 | [-0.033, +0.036] | 1.000 | 1.000 | 1/4 | +0.017 | p(Holm) > 0.05 |
| no_event_bar(3) | better | 146 | 0.97 | -0.062 | +0.011 | -0.003 | [-0.021, +0.039] | 1.000 | 1.000 | 2/4 | +0.011 | p(Holm) > 0.05 |
| no_event_bar(5) | better | 146 | 0.97 | -0.062 | +0.011 | -0.003 | [-0.021, +0.039] | 1.000 | 1.000 | 2/4 | +0.011 | p(Holm) > 0.05 |
| ID_smoothest_80% | better | 118 | 0.78 | +0.007 | +0.077 | +0.066 | [-0.007, +0.162] | 1.000 | 1.000 | 3/4 | +0.027 | p(Holm) > 0.05 |
| ID_smoothest_65% | better | 99 | 0.66 | +0.012 | +0.073 | +0.071 | [-0.053, +0.203] | 1.000 | 1.000 | 2/4 | +0.007 | p(Holm) > 0.05 |
| ID_smoothest_50% | better | 75 | 0.50 | +0.025 | +0.084 | +0.084 | [-0.086, +0.265] | 1.000 | 1.000 | 2/4 | -0.004 | p(Holm) > 0.05 |

## Locked years: selected variants only

No variant passed on the design years, so no variant was evaluated on the locked years.

## Honesty checks

- Signal audit: 238392 baseline cells and 76296 variant cells compared with the bot's evaluate_day, 0 mismatch(es).
- control trail_atr_mult=3 reproduces the baseline exactly: ok

## Decision rule, fixed in advance

```
A variant is adopted only if ALL of these hold.
Metric: R against the baseline initial risk, net of the market's move over each trade's
holding window (beta x the equal-weight universe return from the signal close to the exit),
so that holding longer in a rising market earns nothing by itself.
On the DESIGN years (chosen with no access to the LOCKED years):
  1. direction: it was declared 'better' in advance and its mean R is higher than the baseline;
  2. significance: the Holm-corrected p-value of the paired block bootstrap (whole ISO weeks)
     is <= 0.05 over the family of all adoptable (declared 'better') variants (so the 95%
     interval of the difference excludes 0 after correction), and so is the same test with
     4-week blocks (trades last weeks and overlap, which a 1-week block would understate);
     the 'worse' diagnostics are corrected among themselves and can never be adopted;
  3. size: the gain is at least 0.05 R and the variant changes the outcome of at least 30
     events (a bootstrap that rests on a handful of events is anti-conservative);
  4. stability: the sign of the difference is the same in at least 70% of the design
     calendar years that have enough events, AND dropping any single design year leaves at
     least half of the gain (no single year carries the effect);
  5. plateau: neighbouring thresholds point the same way (no spike; at least one
     neighbour must exist and none may reverse or collapse to under half of the gain);
  6. coverage: it keeps at least 60% of the baseline events, or the events it keeps are
     still at least 2 times what the slots can absorb in a year (a strong filter that fails
     only this test is reported as 'real but not adoptable': the slots would sit idle);
  7. de-clustered check: the difference is also positive when only the best-scored event
     of each week is kept;
  8. money check: the average equity contribution per trade (R x the bot's real position
     weight, not adjusted) is also higher.
On the LOCKED years, evaluated ONCE, only for variants that passed 1-8, and only if the locked
years hold at least 150 events (below that nothing is evaluated and nothing is recommended):
  9. the difference has the same sign, at least half of the design magnitude, AND is
     significant: one-sided p <= 0.05 after Holm over the selected variants, with 1-week and
     4-week blocks.
At most ONE change is named: the one whose gain held in both periods (the smaller of the
design and locked differences) is largest.
Otherwise: keep the defaults.  A higher mean alone never counts.
```

## Limits

- No slot, cash, theme or drawdown limits: this measures the signal and its exits, not the portfolio. The bot takes only the best-scored few events.
- One position per symbol; no pyramiding; no earnings, news or insider data; constant EUR/USD; costs computed for a reference position size.
- A trade is closed after 250 sessions at most (the bot has no such cap), which cuts the biggest winners the same way for every variant.
- Today's universe (survivorship bias) and split-adjusted prices without dividends.
- Combinations of changes were not tested; adopt at most one change and run the study again.
