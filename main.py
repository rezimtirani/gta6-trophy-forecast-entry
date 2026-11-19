"""GTA VI trophy rarity forecast for make.fun's gta6-trophy-rarity-forecast-01.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at the case's target, scored by pinball
loss on the logit scale.

Model (standard library only, deterministic, no network)
--------------------------------------------------------
An earn rate is a ratio, and its two halves move for different reasons, so the
forecast splits them:

* Players tracked P(t).  Exophase's tracked-player count grows as more
  players sync profiles.  ln P is fitted by least squares against
  u = ln(1 + hours since release) and read at the target.  A power law in
  (1 + t) bends growth down as the launch rush passes, which is what the
  launch-night reads show (638 players at 23:50, 3,868 by the 04:30 refresh).
* Earners E_i(t) = rate_i(t) * P(t) / 100 per trophy.  ln E_i is fitted
  against the same u.  Earners only grow (a trophy, once earned, stays
  earned), so their log trend is smoother than the rate's own trend, which
  is pulled down whenever the denominator grows faster than earners do.
* The rate forecast is 100 * E_i(target) / P(target).  A second, direct
  forecast fits logit(rate_i) against u and reads it at the target.  Each
  fit is run on two windows -- all of the case's history, and the last 24
  hours before the checkpoint, where the current regime (not the
  launch-night rush) sets the slope -- and the median is the average of
  the routes and windows in logit space, so a failure mode of any one of
  them is diluted, not doubled.  The extrapolated median is never allowed
  more than 1.5 logit units from the trophy's last rate: by the deadline
  the launch rush is over and growth has bent toward saturation, and a
  wider cap lets a steep early trend run a story trophy to 95%+.

Structure the data itself enforces:

* The platinum (id 1) is earned only by holding every other trophy, so its
  rate can never exceed any other trophy's rate; its median is capped at the
  smallest other data-based median (a trophy with no earner yet, whose
  median is only a tier prior, does not set the cap).
* The story chain (ids 3, 4, 5, 6, 7: chapters 1-4, then the final mission)
  is gated in order, so its medians are made non-increasing along the chain.
* "A Long Road" (id 14, 100% completion) requires the final mission, so its
  median is capped at the final mission's median.
* A trophy with no earner in any read gets a tier prior (platinum 0.04%,
  gold 0.08%, silver 0.15%, bronze 0.30%) with a wide spread, instead of a
  zero that the logit cannot carry.

Spread (the quantiles)
----------------------
sigma_i combines, in quadrature: the house random-walk term
0.5 * sqrt(horizon in days); the residual spread of the trophy's own logit
trend fit (floored, so a perfectly flat history still carries the baseline
spread); the slope's standard error times the distance extrapolated beyond
the data; and half the disagreement between the two median forecasts.
Total sigma is capped at 2.0 logit units.  q_tau =
inv_logit(logit(median) + z_tau * sigma_i), rounded to 4 decimals half-up.

On the practice cases the history is perfectly flat (every read the same
page), so both trend fits are flat, the residual and extrapolation terms are
zero, and the forecast is the persistence baseline's row to rounding: the
practice measures how a refresh the data never showed is coped with, and
no data-driven model can beat persistence there.  The final case carries
two days of real dynamics, which is where this model earns its keep.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages. -P
keeps this file's directory OFF sys.path, so a module shipped beside main.py
imports only after the sys.path line below, which puts the directory first.

What an entry must do:
  - read every <id>.in in the input directory (the grader runs one case at a
    time, with only that case's file there): JSON
    {"schema": "makefun/gta6-trophies-case/v1", "id", "at",
    "horizon_minutes", "target", "levels", "game", "trophies", "history",
    "notes"}, where `trophies` lists the case's trophies {id, name,
    description, tier, points, secret} and `history` every read before `at`,
    oldest first, each {slot, at, players_tracked, trophies_earned, rates},
    `rates` in the order of `trophies` (null where that read lacked the
    trophy);
  - write <id>.out to the output directory:
    {"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}, one row
    for each of the case's trophies, each id once, no other key, each q five
    finite numbers from 0 to 100 (percent), non-decreasing; at most 65,536
    bytes.  Anything else is an invalid case, scored as if every quantile
    equalled the trophy's rate in the last read, and an entry with an
    invalid case takes no prize.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# Random-walk spread of a rate's move, in logit units per square-root day
# (the house baseline's sigma).
SIGMA_RW = 0.5
# The trend-implied move from the last rate is shrunk toward persistence:
# a rate's growth bends toward saturation (its rise in log-time is
# concave), so a straight log-time line over-reads the next 4.5 days, and
# the target week holds a weekend whose play surge no trend in the data
# can price.  Half the implied move is the calibrated compromise: it keeps
# most of the trend's gain where growth is still steep and most of
# persistence's safety where it has already bent.
DELTA_SHRINK = 0.33
# Widest logit move allowed from a trophy's last rate, either way, after
# the shrink above: a trophy's logit rate rarely moves more than this over
# the 4.5 days past the data.
MAX_LOGIT_MOVE = 1.5
# Trend fits are read on two windows: all history, and the last 24 hours
# before the checkpoint, where the current regime (not the launch-night
# rush) sets the slope.  The house curve baseline uses the last 24 hours
# alone; averaging the two hedges a regime change either way.
LAST_WINDOW_HOURS = 24.0
# Widest total spread, in logit units.
MAX_SIGMA = 2.0
# Tier priors (percent) for a trophy with no earner in any read.
TIER_PRIOR = {"platinum": 0.04, "gold": 0.08, "silver": 0.15, "bronze": 0.30}
# Story chain, in play order: chapters 1-4, then the final mission.
STORY_CHAIN = [3, 4, 5, 6, 7]
# 100% completion (id 14) requires the final mission (id 7).
COMPLETION_ID = 14
FINAL_MISSION_ID = 7
PLATINUM_ID = 1


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100 / (1 + math.exp(-lg))
    e = math.exp(lg)
    return 100 * e / (1 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's approximation, |err| < 1.2e-9)."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def least_squares(xs: list[float], ys: list[float]) -> dict | None:
    """Least-squares line y = a + b*x. Returns slope, intercept, residual sd,
    slope standard error and the x-mean; None when the fit is impossible."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid2 = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        sd = math.sqrt(resid2 / (n - 2))
    else:
        sd = 0.0
    se = sd / math.sqrt(sxx) if sxx > 0 else 0.0
    return {"a": a, "b": b, "sd": sd, "se": se, "mx": mx, "n": n}


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    z = [norm_ppf(p) for p in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    target = parse_time(case["target"])
    horizon_days = case["horizon_minutes"] / 1440.0

    def u_of(when: datetime) -> float:
        hours = (when - release).total_seconds() / 3600.0
        return math.log(1.0 + max(hours, -0.9))

    u_star = u_of(target)

    # --- Players tracked: ln P against u, all history and the last 24 h. ---
    checkpoint = parse_time(case["at"])
    cutoff = checkpoint - timedelta(hours=LAST_WINDOW_HOURS)
    pu_all, pl_all, pu_rec, pl_rec = [], [], [], []
    for h in history:
        p = h.get("players_tracked")
        if not p:
            continue
        u = u_of(parse_time(h["at"]))
        pu_all.append(u)
        pl_all.append(math.log(p))
        if parse_time(h["at"]) >= cutoff:
            pu_rec.append(u)
            pl_rec.append(math.log(p))
    pfit = least_squares(pu_all, pl_all) if len(pu_all) >= 2 else None
    pfit_rec = least_squares(pu_rec, pl_rec) if len(pu_rec) >= 2 else None
    last_players = history[-1]["players_tracked"] if history else None
    preds = []
    if pfit is not None:
        preds.append(pfit["a"] + pfit["b"] * u_star)
    if pfit_rec is not None:
        preds.append(pfit_rec["a"] + pfit_rec["b"] * u_star)
    if preds:
        players_star = math.exp(sum(preds) / len(preds))
    else:
        players_star = float(last_players) if last_players else 638.0
    players_star = max(players_star, float(last_players or 1))

    # --- Last observed rate per trophy (last non-null read). ---
    last_rate = []
    for i in range(len(trophies)):
        rate = None
        for h in reversed(history):
            r = h["rates"][i] if i < len(h["rates"]) else None
            if r is not None:
                rate = float(r)
                break
        last_rate.append(rate)

    medians: list[float] = []
    sigmas: list[float] = []
    prior_used: list[bool] = []
    for i, trophy in enumerate(trophies):
        pts: list[tuple[float, float, float, float]] = []  # (u, logit rate, ln earners, at)
        max_earners = 0.0
        for h in history:
            r = h["rates"][i] if i < len(h["rates"]) else None
            if r is None:
                continue
            p = h.get("players_tracked")
            earners = float(r) * float(p) / 100.0 if p else 0.0
            max_earners = max(max_earners, earners)
            pts.append((u_of(parse_time(h["at"])), logit(float(r)),
                        math.log(max(earners, 0.5)) if p else float("nan"),
                        parse_time(h["at"]).timestamp()))
        base = last_rate[i] if last_rate[i] is not None else 0.0

        if max_earners < 1.0 or not pts:
            # Nobody has earned this trophy yet (a flat 0.00% in every read
            # makes the fits degenerate constants at the clamp floor, so
            # they must not run): tier prior, wide spread.
            medians.append(TIER_PRIOR.get(trophy["tier"], 0.30))
            sigmas.append(1.5)
            prior_used.append(True)
            continue
        prior_used.append(False)

        def window_fit(pts_sub, col, want_earn):
            xs = [p[0] for p in pts_sub if p[col] == p[col]]
            ys = [p[col] for p in pts_sub if p[col] == p[col]]
            fit = least_squares(xs, ys) if len(xs) >= 2 else None
            if fit is None:
                return None, None
            if want_earn:
                e_star = math.exp(fit["a"] + fit["b"] * u_star)
                return logit(100.0 * e_star / players_star), fit
            return fit["a"] + fit["b"] * u_star, fit

        cutoff_ts = cutoff.timestamp()
        recent = [p for p in pts if p[3] >= cutoff_ts]
        windows = [pts] + ([recent] if len(recent) >= 3 else [])
        cands: list[float] = []
        resid_sd = 0.0
        slope_se = 0.0
        fit_mx = u_star
        for w in windows:
            pred, fit = window_fit(w, 1, False)   # direct: logit rate vs u
            if pred is not None:
                cands.append(pred)
                resid_sd, slope_se, fit_mx = fit["sd"], fit["se"], fit["mx"]
            pred, fit = window_fit(w, 2, True)    # decomposition: ln earners vs u
            if pred is not None:
                cands.append(pred)
                if resid_sd == 0.0:
                    resid_sd, slope_se, fit_mx = fit["sd"], fit["se"], fit["mx"]
        if not cands:
            # A single read (or a single usable read) leaves no trend to
            # fit: persistence with the baseline spread.
            medians.append(base if base > 0 else TIER_PRIOR.get(trophy["tier"], 0.30))
            sigmas.append(SIGMA_RW * math.sqrt(horizon_days))
            prior_used.append(False)
            continue
        med_logit = sum(cands) / len(cands)
        # Disagreement between the routes and windows widens the spread.
        disagree = (max(cands) - min(cands)) / 2.0 if len(cands) > 1 else 0.0
        # Shrink the trend-implied move toward persistence, then cap it.
        last_lg = logit(base) if base > 0 else logit(0.01)
        med_logit = last_lg + DELTA_SHRINK * (med_logit - last_lg)
        med_logit = min(max(med_logit, last_lg - MAX_LOGIT_MOVE),
                        last_lg + MAX_LOGIT_MOVE)
        medians.append(clamp(inv_logit(med_logit)))
        rw = SIGMA_RW * math.sqrt(horizon_days)
        resid = min(max(resid_sd, 0.0), 1.0)
        extrap = min(slope_se * abs(u_star - fit_mx), 0.8)
        sigma = math.sqrt(rw * rw + resid * resid + extrap * extrap
                          + min(disagree, 0.5) ** 2)
        sigmas.append(min(max(sigma, rw), MAX_SIGMA))

    # --- Structure the data enforces, on the medians. ---
    by_id = {t["id"]: k for k, t in enumerate(trophies)}
    if PLATINUM_ID in by_id:
        others = [medians[k] for k, t in enumerate(trophies)
                  if t["id"] != PLATINUM_ID and not prior_used[k]]
        if others:
            medians[by_id[PLATINUM_ID]] = min(medians[by_id[PLATINUM_ID]], min(others))
    chain = [by_id[i] for i in STORY_CHAIN if i in by_id]
    run = None
    for k in chain:
        run = medians[k] if run is None else min(run, medians[k])
        medians[k] = run
    if COMPLETION_ID in by_id and FINAL_MISSION_ID in by_id:
        medians[by_id[COMPLETION_ID]] = min(medians[by_id[COMPLETION_ID]],
                                            medians[by_id[FINAL_MISSION_ID]])

    rows = []
    for k, trophy in enumerate(trophies):
        lg = logit(medians[k])
        q = [round4(clamp(inv_logit(lg + zz * sigmas[k]))) for zz in z]
        q = sorted(q)
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
