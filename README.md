# gta6-trophy-forecast-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rezimtirani/gta6-trophy-forecast-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/rezimtirani/gta6-trophy-forecast-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at 2026-11-26T00:00:00Z — a week
after launch — from the house's recording of the game's trophy page up to
the deadline. An earn rate is a ratio, so the model splits it: the tracked
player count is fitted as a power law in log-time, each trophy's earner
count is fitted the same way, and the rate forecast is earners over
players. A second, direct forecast fits the logit rate against log-time;
each fit runs on all history and on the last 24 hours, and the median
averages the routes and windows in logit space, shrunk one third of the
way back toward the last observed rate (growth bends toward saturation,
and a straight trend over-reads four and a half days out). The spread
combines the house random-walk term with the fit's residual, the slope's
standard error over the extrapolation distance, and the routes'
disagreement. The platinum is capped at the smallest other data-based
median, the story chain is kept non-increasing, and a trophy nobody has
earned yet gets a tier prior with a wide spread. Standard library only,
no network, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

That reads every `<id>.in` case in `input/` (from the competition kit)
and writes one `<id>.out` forecast per case to `out/`: `{"trophies":
[{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`. On the practice
cases the history is perfectly flat, so the medians equal the last
observed rates; the final case carries two days of real dynamics, which
is where the model earns its keep.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
