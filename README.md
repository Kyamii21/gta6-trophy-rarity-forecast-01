# GTA VI Trophy Rarity Forecast

[![make.fun bounty](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kyamii21/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Kyamii21/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty
[gta6-trophy-rarity-forecast-01](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast every GTA VI PS5 trophy's Exophase earn rate at 2026-11-26T00:00Z
from the house's recorded reads, scored by mean pinball loss on the logit
scale.

## Approach

`main.py` (Python standard library only, no network, deterministic) reads
each case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes five quantiles
per trophy to `$MAKEFUN_OUTPUT_DIR` (else `/output`).

Per trophy it blends several candidate medians, weighted by a
rolling-origin validation inside the case itself:

- persistence (the last observed rate),
- logit-linear trends against `ln(1 + hours since 2026-11-19)` and against
  `sqrt(hours)`, over all reads and over the last 24 hours,
- an earners model: `ln(earners + 0.5)` is trended and divided by a global
  `ln(players_tracked)` forecast, so a surge of new tracked players can
  dilute a rate even while the earner count grows.

The quantile spread combines the trend fit's residual scale, a
random-walk term growing with the horizon, and a count-uncertainty term
that is large while only a handful of players have earned a trophy.
Spread coefficients were tuned on truncated-history backtests (next-read
forecasts on the practice cases, plus synthetic final-shaped cases with
growing player counts).

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
