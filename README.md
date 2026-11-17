# gta6-twitch-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/101parallax/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/101parallax/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

A quantile forecaster for Grand Theft Auto VI's launch-day Twitch viewers, hour by hour. Given the first hour's print of release day (2026-11-19 UTC), it forecasts hours 2–24 as five quantiles (5%, 25%, 50%, 75%, 95%) each, fitted only from the case's own comparables: 38 past big game launches' hourly Twitch curves.

The model: every past launch gives a log-ratio curve ln(viewers_h / viewers_1). Those curves fall on a ladder keyed to the size of the hour-1 print alone (log peak ratio correlates −0.94 with ln(p1)) — launches already live before UTC midnight stay flat, launches still waiting for the US midnight unlock explode 10×–1000×. Per hour, the median is an OLS fit on [1, ln(p1), max(0, 10.5 − ln(p1))], clipped to the observed range; the spread is the RMS residual of the 10 training launches nearest in ln(p1), each scored against its own leave-one-out fit, floored at 0.25 log units. Big launches get a narrow band, small launches a wide one.

Local validation: leave-one-out over the 37 usable comparables scores a pinball loss of 0.371 (house baseline 0.930), typical miss 98%, 90% coverage 0.898. On the four practice launches (answers re-fetched from SullyGnome's API as documented in the kit) the pooled loss is 0.199 versus the house baseline's 0.387. Python 3.13 standard library only, no network, deterministic.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` (23 rows, hours 2–24) to the output directory. Each case runs in ~0.1 s.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
