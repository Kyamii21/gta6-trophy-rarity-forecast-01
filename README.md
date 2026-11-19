# GTA VI Trophy Rarity Forecast — v2

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

- **Players tracked** — consecutive identical reads are collapsed into
  Exophase refresh epochs. A damped power law in log time is fitted to
  the epochs and blended with the most recent segment's slope. With a
  single epoch it falls back to a launch-growth prior.
- **Earners per trophy** — current earners plus three growth terms: the
  trophy's own earner velocity across epochs (damped as its pool
  saturates), a hazard term for already-tracked players who have not
  earned it yet, and a new-player term (arrivals earn early trophies
  quickly, late ones rarely). Class priors from the trophy text — story
  chapter, collectible with a large numeric target, one-off activity,
  100% completion, platinum — set the hazard, the new-player probability
  and an eventual-share cap. Story-chain medians respect the funnel
  order: chapter 1 ≥ chapter 2 ≥ … ≥ final mission ≥ 100% ≥ platinum.
- **Refresh timing** — the answering read shows the last refresh before
  the target, so the forecast is made at the target minus half the
  typical refresh gap, and the spread carries a refresh component.
- **Quantiles** — logit-normal around the median rate. The logit spread
  grows with the square root of the horizon and adds terms for refresh
  timing, small earner counts, player-count uncertainty, and a stale
  page (a long run of identical reads means the next refresh can jump).

Validation (rolling-origin backtests on the practice history and a
synthetic launch simulator with known answers) is summarised in
`NOTES-v2.md`.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`), writes
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`).
Python standard library only, deterministic, no network.
