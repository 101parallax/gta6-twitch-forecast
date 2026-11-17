"""GTA VI launch-day Twitch viewers, hour by hour (make.fun gta6-twitch-ccu-forecast-01).

A quantile forecaster fitted only from the case's own comparables (past big
game launches' hourly Twitch curves, 14 days before to 3 days after release).

What the data says
------------------
Every past launch gives a log-ratio curve L[h] = ln(viewers_h / viewers_1)
over the UTC release day, hours 2..24. Those curves fall on a near-deterministic
ladder keyed to the size of the hour-1 print alone: the log peak ratio
correlates -0.94 with ln(p1). A launch whose big markets were already unlocked
before UTC midnight (a global unlock the evening before, early access) sits
near its hour-1 level all day (peak ratio ~1); a launch still waiting for the
US midnight unlock at 05:00Z/08:00Z explodes 10x-1000x off a tiny hour-1
print. The evening-before level, channels and viewers-per-channel add
nothing once ln(p1) is known (checked on leave-one-out), so the model uses
ln(p1) only.

Model (per hour h = 2..24, fitted from scratch on each run)
------------------------------------------------------------
Median: ordinary least squares of L[h] on [1, ln(p1), max(0, 10.5 - ln(p1))].
The hinge at ln(p1) = 10.5 (~36k viewers) is where the ladder saturates:
above it launches are already live and the curve is flat, below it the
ratio grows about linearly as the hour-1 print shrinks. Predictions are
clipped to the observed range of L[h] across the training launches, so an
hour-1 print larger than any past launch (GTA VI may be one) cannot
extrapolate the median outside what launches have actually done.

Spread: the honest "how wrong is this median for a launch this size" is
measured locally, not globally. For each training launch we fit the same
OLS without it and record its residual; the spread at hour h is the RMS
residual of the 10 training launches nearest in ln(p1) to the case, floored
at 0.25 log units. Big launches get a narrow band (their neighbours agree);
small launches get a wide one (their neighbours disagree by orders of
magnitude). Quantiles are log-normal around the median:
q_tau = p1 * exp(med_h + z_tau * sd_h), rounded half up to 2 decimals.

Validation (validate.py, outside this repo; practice answers fetched fresh
from the SullyGnome API named in the kit's input/NOTICE.txt, which
reproduces all four practice hour-1 prints exactly):
  leave-one-out over the 37 usable comparables: loss 0.371 (house 0.930),
    typical miss 98%, 90% coverage 0.898.
  practice p00-p03 pooled: loss 0.199 (house 0.412), typical miss 39%,
    90% coverage 0.967.
Nothing was fitted to the practice answers: the practice launches are not
among the comparables, and every constant above was chosen on
leave-one-out over the comparables themselves.

Contract (same as the kit starter): read every <id>.in in $MAKEFUN_INPUT_DIR
(else /input), write <id>.out to $MAKEFUN_OUTPUT_DIR (else /output) as
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}]} for h = 2..24.
Python 3.13 standard library only, no network, deterministic.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# ln(p1) where the launch-size ladder saturates (~36k viewers).
HINGE = 10.5
# Training launches nearest in ln(p1) whose residuals set the spread.
K_LOCAL = 10
# Minimum log-spread per hour (the 5%-95% band never gets narrower).
SPREAD_FLOOR = 0.25
# A launch counts when its hour 1 exists and >= 18 of hours 2..24 do.
MIN_DAY_HOURS = 18


def parse_ts(text):
    """ISO 8601 UTC timestamp -> aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_key(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    if n % 2:
        return vals[n // 2]
    return 0.5 * (vals[n // 2 - 1] + vals[n // 2])


def round2(x):
    """x rounded to 2 decimals, half up (as the kit starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def solve_linear(matrix, rhs):
    """Small dense solve by Gauss elimination with partial pivoting.

    Returns the solution, or None when the system is (near-)singular.
    """
    n = len(rhs)
    aug = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[piv][col]) < 1e-12:
            return None
        aug[col], aug[piv] = aug[piv], aug[col]
        for row in range(n):
            if row != col:
                factor = aug[row][col] / aug[col][col]
                for cc in range(col, n + 1):
                    aug[row][cc] -= factor * aug[col][cc]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def features(logp1):
    """The median model's features for a launch of hour-1 size logp1."""
    return [1.0, logp1, max(0.0, HINGE - logp1)]


def fit_hour(train, hour):
    """OLS coefficients for L[hour] on features(logp1), or None."""
    rows = [(features(c["logp1"]), c["L"][hour])
            for c in train if hour in c["L"]]
    if len(rows) < 6:
        return None
    k = 3
    matrix = [[sum(r[0][i] * r[0][j] for r in rows) for j in range(k)]
              for i in range(k)]
    rhs = [sum(r[0][i] * r[1] for r in rows) for i in range(k)]
    return solve_linear(matrix, rhs)


def predict(beta, logp1):
    feats = features(logp1)
    return sum(b * x for b, x in zip(beta, feats))


def extract_curves(case):
    """One log-ratio curve per comparable launch usable as a past launch.

    A comparable counts when its release-day hour 1 exists and at least 18
    of hours 2..24 are present. The GTA VI category's own comparable has no
    hour 1 on its premiere day, so it drops out here on its own.
    """
    curves = []
    for comp in case.get("comparables") or []:
        try:
            day_start = parse_ts(comp["day_start"])
        except (KeyError, TypeError, ValueError):
            continue
        hourly = {}
        for row in comp.get("hourly") or []:
            v = row.get("viewers")
            if v:
                hourly[row["from"]] = v
        day = [hourly.get(hour_key(day_start + timedelta(hours=h)))
               for h in range(24)]
        p1 = day[0]
        if not p1 or p1 <= 0:
            continue
        if sum(1 for v in day[1:] if v) < MIN_DAY_HOURS:
            continue
        ratios = {}
        for h in range(2, 25):
            v = day[h - 1]
            if v:
                ratios[h] = math.log(v / p1)
        curves.append({
            "name": comp.get("name", "?"),
            "logp1": math.log(p1),
            "L": ratios,
        })
    curves.sort(key=lambda c: c["name"])
    return curves


def forecast_from_curves(p1, logp1, curves):
    """Quantile rows for hours 2..24 from the fitted comparable curves."""
    if len(curves) < 6:
        return None
    # Full-train median per hour, clipped to the observed range.
    med = {}
    sd = {}
    nearest = sorted(curves, key=lambda c: (abs(c["logp1"] - logp1),
                                            c["name"]))[:K_LOCAL]
    # Each nearest launch's own leave-one-out median, for the spread:
    # how wrong is this median for a launch this size?
    loo_med = {}
    for curve in nearest:
        rest = [c for c in curves if c is not curve]
        loo_med[curve["name"]] = {}
        for h in range(2, 25):
            beta = fit_hour(rest, h)
            if beta is None:
                continue
            loo_med[curve["name"]][h] = predict(beta, curve["logp1"])
    for h in range(2, 25):
        observed = [c["L"][h] for c in curves if h in c["L"]]
        if not observed:
            med[h], sd[h] = 0.0, 0.6
            continue
        beta = fit_hour(curves, h)
        if beta is None:
            med[h] = median(observed)
        else:
            med[h] = max(min(observed),
                         min(max(observed), predict(beta, logp1)))
        residuals = []
        for c in nearest:
            own = loo_med[c["name"]].get(h)
            if own is not None and h in c["L"]:
                residuals.append(c["L"][h] - own)
        if len(residuals) >= 4:
            spread = math.sqrt(sum(r * r for r in residuals)
                               / len(residuals))
        else:
            spread = 0.6
        sd[h] = max(spread, SPREAD_FLOOR)

    rows = []
    for h in range(2, 25):
        qs = []
        prev = 0.0
        for z in Z:
            q = p1 * math.exp(med[h] + z * sd[h])
            q = min(max(q, 0.01), 1e9)
            q = round2(q)
            if q < prev:  # rounding must never break non-decreasing order
                q = prev
            qs.append(q)
            prev = q
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def house_baseline(p1):
    """Persistence fallback (the kit starter's numbers) if no curve fits."""
    rows = []
    for h in range(2, 25):
        qs = []
        prev = 0.0
        for z in Z:
            q = round2(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)))
            if q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = extract_curves(case)
    forecast = forecast_from_curves(p1, math.log(p1), curves)
    return forecast if forecast is not None else house_baseline(p1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
