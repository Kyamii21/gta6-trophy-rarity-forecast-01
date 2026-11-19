#!/usr/bin/env python3
"""Forecast GTA VI (PS5) trophy earn rates on Exophase at a case's target.

For every case file (*.in) in $MAKEFUN_INPUT_DIR (else /input) this writes
<id>.out in $MAKEFUN_OUTPUT_DIR (else /output): five quantiles of each
trophy's earn rate (percent) at the case's target instant.

Approach, per trophy:
  * A global fit of ln(players_tracked) against ln(1 + hours since
    2026-11-19T00:00Z) predicts how many players Exophase will track at
    the target.
  * Several candidate medians are computed: persistence (last rate),
    logit-linear trends in ln(1 + hours) and in sqrt(hours) over all reads
    and over the last 24 hours, and an earners model that fits
    ln(earners + 0.5) and divides by the predicted player count (so a
    flood of new players can dilute a rate even while earners grow).
  * Candidates are weighted by a rolling-origin validation inside the
    case itself: fit on an early slice, score the median against the
    last observed rate on the logit scale.
  * The spread combines the trend fit's residual scale, a random-walk
    term that grows with the horizon, and a count-uncertainty term that
    is large when only a handful of players have earned the trophy.

Everything is deterministic: no randomness, no clock, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
RATE_LO = 0.01
RATE_HI = 99.99
LOGIT_LO = math.log(0.0001 / 0.9999)
LOGIT_HI = math.log(0.9999 / 0.0001)
DAY_SECONDS = 86400.0

# spread calibration (tuned on truncated-history backtests)
RW_COEF = 0.32        # random-walk logit sigma per sqrt(day)
CNT_COEF = 0.35       # count-uncertainty logit sigma at one earner
CNT_HORIZON_DAYS = 0.25  # horizon scale over which count uncertainty ramps in
RESID_FLOOR = 0.05
SIGMA_CAP = 1.6


# --------------------------------------------------------------------------
# small numeric helpers
# --------------------------------------------------------------------------

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_of_rate(rate_pct):
    p = clamp(rate_pct, RATE_LO, RATE_HI) / 100.0
    return math.log(p / (1.0 - p))


def rate_of_logit(x):
    x = clamp(x, LOGIT_LO, LOGIT_HI)
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.endswith("Z"):
        t = t[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(t)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since_epoch(dt):
    return (dt - EPOCH).total_seconds() / 3600.0


def feat_log(hours):
    return math.log1p(max(hours, 0.0))


def feat_sqrt(hours):
    return math.sqrt(max(hours, 0.0))


def fit_line(xs, ys):
    """Least-squares y = a + b*x. Returns (a, b) or None if degenerate."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def residual_std(xs, ys, line):
    if line is None or len(xs) < 3:
        return 0.0
    a, b = line
    var = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)) / max(len(xs) - 2, 1)
    return math.sqrt(var)


# --------------------------------------------------------------------------
# case preparation
# --------------------------------------------------------------------------

class Read:
    __slots__ = ("when", "hours", "players", "rates")

    def __init__(self, when, hours, players, rates):
        self.when = when
        self.hours = hours
        self.players = players
        self.rates = rates


def prepare_reads(case):
    reads = []
    for raw in case.get("history") or []:
        when = parse_time(raw.get("at")) or parse_time(raw.get("slot"))
        if when is None:
            continue
        players = raw.get("players_tracked")
        try:
            players = float(players) if players is not None else None
        except (TypeError, ValueError):
            players = None
        rates = raw.get("rates") or []
        reads.append(Read(when, hours_since_epoch(when), players, rates))
    reads.sort(key=lambda r: r.when)
    return reads


def fit_players(reads, x_target):
    """Predict players_tracked at x_target from ln(players) ~ ln(1+hours)."""
    pts = [(feat_log(r.hours), math.log(r.players))
           for r in reads if r.players and r.players > 0]
    if not pts:
        return None
    last_players = math.exp(pts[-1][1])
    if len(pts) < 3:
        return last_players
    preds = []
    line_all = fit_line([p[0] for p in pts], [p[1] for p in pts])
    if line_all is not None:
        preds.append(line_all[0] + line_all[1] * x_target)
    if reads:
        t_last = reads[-1].when.timestamp()
        recent = [(feat_log(r.hours), math.log(r.players)) for r in reads
                  if r.players and r.players > 0
                  and r.when.timestamp() >= t_last - DAY_SECONDS]
        if len(recent) >= 3:
            line_r = fit_line([p[0] for p in recent], [p[1] for p in recent])
            if line_r is not None:
                preds.append(line_r[0] + line_r[1] * x_target)
    if not preds:
        return last_players
    pred = math.exp(sum(preds) / len(preds))
    # player counts do not shrink in any meaningful way; keep a sane band
    return clamp(pred, 0.5 * last_players, 50.0 * last_players + 1000.0)


# --------------------------------------------------------------------------
# candidate medians (each returns a logit, or None when it cannot speak)
# --------------------------------------------------------------------------

def series_for(reads, idx):
    """(hours, rate, players) triples where this trophy has a rate."""
    out = []
    for r in reads:
        if idx < len(r.rates) and r.rates[idx] is not None:
            try:
                rate = float(r.rates[idx])
            except (TypeError, ValueError):
                continue
            out.append((r.hours, rate, r.players))
    return out


def cand_persist(train, ctx):
    if not train:
        return None
    return logit_of_rate(train[-1][1])


def cand_logit_trend(train, ctx, feat, x_key, window_hours=None):
    pts = train
    if window_hours is not None and train:
        t_last = train[-1][0]
        pts = [p for p in train if p[0] >= t_last - window_hours]
    if len(pts) < 3:
        return None
    xs = [feat(p[0]) for p in pts]
    if max(xs) - min(xs) < 1e-9:
        return None
    ys = [logit_of_rate(p[1]) for p in pts]
    line = fit_line(xs, ys)
    if line is None:
        return None
    return line[0] + line[1] * ctx[x_key]


def cand_earners(train, ctx, window_hours=None):
    """Fit ln(earners + 0.5) ~ ln(1+hours); rate = earners / players."""
    pts = [p for p in train if p[2] and p[2] > 0]
    if window_hours is not None and pts:
        t_last = pts[-1][0]
        pts = [p for p in pts if p[0] >= t_last - window_hours]
    p_t = ctx["players"]
    if len(pts) < 3 or not p_t or p_t <= 0:
        return None
    xs = [feat_log(p[0]) for p in pts]
    if max(xs) - min(xs) < 1e-9:
        return None
    ys = [math.log(p[1] / 100.0 * p[2] + 0.5) for p in pts]
    line = fit_line(xs, ys)
    if line is None:
        return None
    earners = math.exp(line[0] + line[1] * ctx["x_log"]) - 0.5
    earners = clamp(earners, 0.0, p_t)
    return logit_of_rate(100.0 * earners / p_t)


def make_candidates():
    """name -> function(train, ctx) -> logit or None.

    ctx carries the target features: x_log, x_sqrt, players.
    """
    def logit_log_all(train, ctx):
        return cand_logit_trend(train, ctx, feat_log, "x_log")

    def logit_log_24h(train, ctx):
        return cand_logit_trend(train, ctx, feat_log, "x_log", 24.0)

    def logit_sqrt_all(train, ctx):
        return cand_logit_trend(train, ctx, feat_sqrt, "x_sqrt")

    def earn_all(train, ctx):
        return cand_earners(train, ctx)

    def earn_24h(train, ctx):
        return cand_earners(train, ctx, 24.0)

    return [
        ("persist", cand_persist),
        ("logit_log_all", logit_log_all),
        ("logit_log_24h", logit_log_24h),
        ("logit_sqrt_all", logit_sqrt_all),
        ("earn_all", earn_all),
        ("earn_24h", earn_24h),
    ]


# --------------------------------------------------------------------------
# validation weighting
# --------------------------------------------------------------------------

def validation_errors(series, candidates):
    """Rolling-origin median errors on the logit scale, per candidate.

    Fits each candidate on an early slice of the series and scores its
    median against the last observed rate. Returns {name: mean_abs_err}
    for candidates that could be evaluated, else {}.
    """
    if len(series) < 8:
        return {}
    t0 = series[0][0]
    t1 = series[-1][0]
    span = t1 - t0
    if span < 18.0:
        return {}
    eval_hours, eval_rate, eval_players = series[-1]
    y = logit_of_rate(eval_rate)
    ctx = {
        "x_log": feat_log(eval_hours),
        "x_sqrt": feat_sqrt(eval_hours),
        "players": eval_players,
    }
    errs = {}
    for frac in (0.5, 0.65, 0.8):
        cut = t0 + frac * span
        train = [p for p in series if p[0] <= cut and p[0] < eval_hours]
        if len(train) < 3:
            continue
        persist_logit = logit_of_rate(train[-1][1])
        for name, fn in candidates:
            try:
                pred = fn(train, ctx)
            except Exception:
                pred = None
            if pred is None or not math.isfinite(pred):
                continue
            pred = clamp(pred, persist_logit - 3.5, persist_logit + 3.5)
            pred = clamp(pred, LOGIT_LO, LOGIT_HI)
            errs.setdefault(name, []).append(abs(pred - y))
    return {k: sum(v) / len(v) for k, v in errs.items() if v}


# --------------------------------------------------------------------------
# per-trophy forecast
# --------------------------------------------------------------------------

def forecast_trophy(series, ctx, days, candidates):
    """Return (mu_logit, sigma_logit) for one trophy."""
    if not series:
        # Never observed: a rare trophy by default, honestly wide.
        return logit_of_rate(0.05), 1.2

    last_rate = series[-1][1]
    persist_logit = logit_of_rate(last_rate)

    preds = {}
    for name, fn in candidates:
        try:
            v = fn(series, ctx)
        except Exception:
            v = None
        if v is None or not math.isfinite(v):
            continue
        v = clamp(v, persist_logit - 3.5, persist_logit + 3.5)
        preds[name] = clamp(v, LOGIT_LO, LOGIT_HI)
    if "persist" not in preds:
        preds["persist"] = persist_logit

    errs = validation_errors(series, candidates)
    weights = {}
    if errs:
        for name in preds:
            e = errs.get(name)
            weights[name] = 1.0 / (0.08 + e) ** 2 if e is not None else 0.25
    else:
        for name in preds:
            weights[name] = 1.5 if name == "persist" else 1.0
    wsum = sum(weights.values())
    mu = sum(preds[k] * weights[k] for k in preds) / wsum

    # --- spread ------------------------------------------------------------
    xs = [feat_log(p[0]) for p in series]
    ys = [logit_of_rate(p[1]) for p in series]
    sigma_resid = residual_std(xs, ys, fit_line(xs, ys)) if len(series) >= 4 else 0.0
    sigma_rw = RW_COEF * math.sqrt(max(days, 0.0))
    last_players = series[-1][2]
    earners_last = (last_rate / 100.0 * last_players) if last_players else 0.0
    # count uncertainty ramps in with the horizon: over minutes the earner
    # count barely moves, over days a handful of earners can multiply
    hfac = math.sqrt(days / (days + CNT_HORIZON_DAYS)) if days > 0 else 0.0
    sigma_cnt = clamp(CNT_COEF / math.sqrt(max(earners_last, 1.0)), 0.0, 0.6) * hfac
    sigma = math.sqrt(sigma_resid ** 2 + sigma_rw ** 2 + sigma_cnt ** 2)
    # a median that travels far from persistence is less certain
    sigma += 0.12 * abs(mu - persist_logit)
    sigma = clamp(sigma, RESID_FLOOR, SIGMA_CAP)
    return mu, sigma


def forecast_case(case):
    reads = prepare_reads(case)
    trophies = case.get("trophies") or []
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    zs = [norm_ppf(clamp(float(p), 0.001, 0.999)) for p in levels]

    target_dt = parse_time(case.get("target"))
    if target_dt is None:
        horizon = float(case.get("horizon_minutes") or 0.0)
        base = parse_time(case.get("at"))
        if base is not None:
            target_dt = base + timedelta(minutes=horizon)
    if target_dt is None and reads:
        target_dt = reads[-1].when
    target_hours = hours_since_epoch(target_dt) if target_dt else 0.0

    if reads and target_dt is not None:
        days = max((target_dt - reads[-1].when).total_seconds(), 0.0) / DAY_SECONDS
    else:
        days = float(case.get("horizon_minutes") or 0.0) / 1440.0

    x_log = feat_log(target_hours)
    ctx = {
        "x_log": x_log,
        "x_sqrt": feat_sqrt(target_hours),
        "players": fit_players(reads, x_log),
    }
    candidates = make_candidates()

    rows = []
    for idx, trophy in enumerate(trophies):
        tid = trophy.get("id", idx + 1)
        series = series_for(reads, idx)
        mu, sigma = forecast_trophy(series, ctx, days, candidates)
        qs = []
        for z in zs:
            q = rate_of_logit(mu + z * sigma)
            qs.append(round(clamp(q, 0.0, 100.0), 4))
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": tid, "q": qs})
    return {"trophies": rows}


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------

def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    except OSError:
        names = []
    for name in names:
        stem = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        text = json.dumps(result, separators=(",", ":"), sort_keys=False)
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
