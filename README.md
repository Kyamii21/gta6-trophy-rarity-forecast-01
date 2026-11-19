# GTA VI Trophy Rarity Forecast — v3

[![Bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kyamii21/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![Reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kyamii21/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasts the Exophase earn rate of all 55 PlayStation 5 trophies of
Grand Theft Auto VI at the case target, as five quantiles per trophy
(0.05 / 0.25 / 0.5 / 0.75 / 0.95), scored by mean pinball loss on the
logit scale.

## Model

An earn rate is `earners / players_tracked`. Both counts only grow, but
at different speeds, so a rate can fall through dilution or rise through
progression. The program forecasts the two counts separately and divides.

- **Refresh epochs** — consecutive identical reads are collapsed into
  Exophase refresh epochs; all fitting uses epochs, never raw reads.
- **Players tracked** — a damped power law in log time blended with a
  saturating Weibull growth curve, weights chosen by rolling-origin
  validation on log players. A single epoch falls back to a launch
  growth prior `P ~ (age + 2h)^gamma`.
- **Earners per trophy** — a hazard/convolution model: players arrive in
  cohorts at each epoch (observed arrivals, then the forecast player
  path in five segments) and earn a trophy after a Weibull delay,
  `E(t) = a + F * sum_j dP_j * G(t - t_j)`. The delay scale is
  grid-searched per trophy, `(a, F)` is fitted by weighted least squares
  on epoch earners, and `F` is shrunk toward a class prior read from the
  trophy text (first mission, story chapter, named mission, strand
  final, activity, collectible, big collectible, final mission, 100%
  completion, platinum).
- **Blend** — the convolution median is blended on the logit scale with
  a logit-trend candidate, an earner-velocity candidate and persistence,
  with weights `1/(MAE + 0.04)^2` from rolling-origin validation.
- **Spread** — logit-normal quantiles whose sigma combines a horizon
  term, a refresh-jump term scaled by the observed epoch-to-epoch move,
  a small-earner-count term and player-count uncertainty, calibrated
  against backtest residuals. When the page is stale (one epoch over
  hours of reads), quantiles come from a two-component mixture: the
  no-refresh world around the last rate and the post-refresh diluted
  world around the convolution forecast.
- **Funnel** — story-chain medians are forced non-increasing
  (first mission >= chapter 1 >= ... >= final mission >= 100% >=
  platinum) and the platinum median never exceeds any other trophy's.

## Local validation

Rolling-origin backtests plus a synthetic launch simulator with known
answers (48 cases: 24 final-like 60h -> +108h, 12 short 30h -> +24h,
12 stale-page 3h -> +1.5h; mean pinball loss on the logit scale, lower
is better). All methods scored on the same simulated cases:

| scenario | v3 | v2 | house curve | house persistence |
|---|---|---|---|---|
| final-like | **0.2206** | 0.2536 | 0.2837 | 0.6517 |
| short | **0.1798** | 0.2247 | 0.3214 | 0.4294 |
| stale-page | **0.2241** | 0.2855 | 0.2961 | 0.2961 |
| all 48 | **0.2113** | 0.2543 | 0.2962 | 0.5072 |
| robustness (delays x2, shares x0.7) | **0.2819** | 0.3199 | 0.3685 | 0.6023 |
| robustness (lognormal delays, hyperbolic players) | **0.1621** | 0.2107 | 0.3423 | 0.4918 |

90% coverage: 0.905 final-like, 0.934 over all 48 cases.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`), writes
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).
Python standard library only, deterministic, no network.
