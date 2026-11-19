#!/usr/bin/env python3
"""Forecast GTA VI PS5 trophy earn rates on Exophase (make.fun bounty
gta6-trophy-rarity-forecast-01), v2 model.

An earn rate is earners / players_tracked. Both counts only grow, but at
different speeds, so a rate can fall (dilution by new players) or rise
(existing players progressing). The model forecasts the two counts and
divides.

Players tracked: consecutive identical reads are collapsed into Exophase
refresh epochs. A damped power law in log time is fitted to the epochs
and blended with the most recent segment's slope; with a single epoch a
launch-growth prior is used instead.

Earners per trophy: a hazard/convolution model. Players arrive over
time (the observed epoch-to-epoch increases, then the forecast player
path) and each arrival earns a given trophy after a random delay with a
Weibull distribution whose scale depends on the trophy's class (story
chapter, collectible with a large numeric target, one-off activity,
100% completion, platinum). Expected earners are

    E(t) = a + F * sum_j dP_j * G(t - t_j),

where G is the delay CDF, F the eventual share of players who ever earn
the trophy, and a captures pre-launch earners. F, a and the delay scale
are fitted per trophy by least squares over the epochs (small grid over
the scale). This one mechanism produces saturation of early trophies,
slow growth of late ones and dilution of rates when new players flood
in. The convolution median is blended in logit space with a logit-trend
candidate and persistence, with weights from rolling-origin validation
on the epochs. Story-chain medians are forced to respect the funnel
order (chapter 1 >= chapter 2 >= ... >= final mission >= 100% >=
platinum).

The answering read shows the last Exophase refresh before the target,
so the forecast is made at the target minus half the typical refresh
gap. Quantiles are logit-normal around the median; the logit spread
grows with the square root of the horizon and adds terms for refresh
timing, small earner counts, player-count uncertainty and a stale page
(a long run of identical reads means the next refresh can jump).

Standard library only, deterministic, no network.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722]

# Tuned on a synthetic launch simulator (see NOTES-v2.md).
SIGMA_PER_SQRT_DAY = 0.26
SIGMA_BASE = 0.10
SIGMA_REFRESH = 0.15
SIGMA_COUNT_COEF = 0.80
SIGMA_COUNT_CAP = 0.42
SIGMA_STALE_MULT = 1.35
SIGMA_MIN = 0.07
SIGMA_MAX = 2.20

LAUNCH = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)

# Delay-scale prior (hours) and Weibull shape per trophy class.
CLASS_DELAY = {
    "first": (1.5, 1.3),
    "chapter": (22.0, 1.05),
    "final_story": (320.0, 1.0),
    "completion": (700.0, 0.9),
    "platinum": (900.0, 0.9),
    "collectible_big": (380.0, 0.85),
    "collectible": (160.0, 0.9),
    "activity": (55.0, 0.95),
}
LAMBDA_GRID = [0.3, 0.5, 0.75, 1.0, 1.4, 2.0, 3.0, 4.5]


def parse_time(text):
    if text is None:
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


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    p = clamp(x, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def classify(trophy):
    """Return (kind, eventual_share_prior, chapter_number_or_0)."""
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    tid = trophy.get("id")

    if tier == "platinum" or tid == 1:
        return ("platinum", 0.055, 0)
    if "100% completion" in desc or "100% completion" in name:
        return ("completion", 0.085, 0)
    m = re.search(r"chapter\s+(\d+)", desc)
    if m:
        n = int(m.group(1))
        return ("chapter", clamp(0.92 - 0.085 * n, 0.35, 0.90), n)
    if "final mission" in desc and tier == "gold":
        return ("final_story", 0.50, 0)
    if "gone fishing" in desc:
        return ("first", 0.98, 0)
    nums = [int(x) for x in re.findall(r"\d+", desc.replace(",", ""))]
    big = max(nums) if nums else 0
    if big >= 50 or "every" in desc or "all " in desc:
        return ("collectible_big", 0.16, 0)
    if big >= 3 or ("complete" in desc and "sets" in desc):
        return ("collectible", 0.30, 0)
    if "final mission for" in desc:
        return ("collectible", 0.32, 0)
    return ("activity", 0.38, 0)


def delay_prior(kind, chapter_n):
    lam, shape = CLASS_DELAY[kind]
    if kind == "chapter" and chapter_n:
        lam = 9.0 * (2.3 ** chapter_n)
    return lam, shape


def story_chain_rank(trophy):
    """Position in the story funnel, or None if not on the main chain."""
    desc = (trophy.get("description") or "").lower()
    tier = (trophy.get("tier") or "").lower()
    if "gone fishing" in desc:
        return 0
    m = re.search(r"chapter\s+(\d+)", desc)
    if m:
        return int(m.group(1))
    if "final mission" in desc and tier == "gold":
        return 50
    if "100% completion" in desc:
        return 60
    if tier == "platinum":
        return 70
    return None


def dedupe_epochs(history):
    """Collapse consecutive identical reads into refresh epochs."""
    epochs = []
    last_key = None
    for read in history:
        rates = read.get("rates") or []
        key = (read.get("players_tracked"),
               tuple(-1.0 if r is None else float(r) for r in rates))
        if key != last_key:
            epochs.append(read)
            last_key = key
    return epochs


def linfit(xs, ys):
    n = len(xs)
    if n < 2:
        return (0.0, ys[0] if ys else 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (0.0, my)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return (slope, my - slope * mx)


def weibull_cdf(age, lam, shape):
    if age <= 0:
        return 0.0
    return 1.0 - math.exp(-((age / lam) ** shape))


def forecast_players(epochs, t0, t_target):
    """Return (players_at_target, sigma_log_players, refresh_gap_hours)."""
    times = []
    players = []
    for e in epochs:
        dt = parse_time(e.get("at"))
        p = e.get("players_tracked")
        if dt is not None and p:
            times.append((dt - t0).total_seconds() / 3600.0)
            players.append(float(p))
    if not players:
        return (1000.0, 0.5, 3.0)
    p_now = players[-1]
    h_now = times[-1]
    h_tgt = (t_target - t0).total_seconds() / 3600.0
    horizon_h = max(h_tgt - h_now, 0.0)

    gaps = [times[i + 1] - times[i] for i in range(len(times) - 1)
            if times[i + 1] > times[i]]
    gaps_sorted = sorted(gaps)
    gap_med = gaps_sorted[len(gaps_sorted) // 2] if gaps_sorted else 3.0
    gap_med = clamp(gap_med, 0.5, 12.0)

    if len(players) >= 2 and h_now > 0.25:
        xs = [math.log1p(h + 2.0) for h in times]
        ys = [math.log(p) for p in players]
        slope, intercept = linfit(xs, ys)
        x_tgt = math.log1p(h_tgt + 2.0)
        fit_logp = intercept + slope * x_tgt
        if xs[-1] > xs[-2]:
            recent = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            recent = clamp(recent, 0.0, max(slope * 1.6, 0.05))
            recent_logp = ys[-1] + 0.72 * recent * (x_tgt - xs[-1])
            logp = 0.62 * fit_logp + 0.38 * recent_logp
        else:
            logp = fit_logp
        logp = max(logp, math.log(p_now))
        logp = min(logp, math.log(p_now)
                   + 0.55 * math.sqrt(horizon_h / 24.0) + 0.35)
        resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
        sigma = 0.10 + 0.10 * math.sqrt(horizon_h / 24.0)
        if resid:
            sigma += min(0.25, math.sqrt(sum(r * r for r in resid) / len(resid)))
        return (math.exp(logp), sigma, gap_med)

    age_days = max((h_now + 2.0) / 24.0, 0.06)
    gamma = 1.50 if age_days < 0.3 else 1.15 if age_days < 1.0 else 0.65
    ratio = ((age_days + horizon_h / 24.0) / age_days) ** gamma
    ratio = clamp(ratio, 1.0, 9.0)
    sigma = 0.35 + 0.22 * math.sqrt(horizon_h / 24.0)
    return (p_now * ratio, sigma, gap_med)


def exposure(times, deltas, t, lam, shape):
    """A(t) = sum_j dP_j * G(t - t_j) over arrivals at or before t."""
    total = 0.0
    for tj, dp in zip(times, deltas):
        if tj <= t and dp > 0:
            total += dp * weibull_cdf(t - tj, lam, shape)
    return total


def fit_convolution(times, deltas, earners, lam0, shape):
    """Grid-search the delay scale; LS for (a, F). Returns (a, F, lam)."""
    best = None
    for mult in LAMBDA_GRID:
        lam = lam0 * mult
        avals = [exposure(times, deltas, t, lam, shape) for t in times]
        # Least squares earners ~ a + F * A with a, F >= 0.
        slope, intercept = linfit(avals, earners)
        if slope < 0:
            slope, intercept = 0.0, sum(earners) / len(earners)
        if intercept < 0:
            intercept = 0.0
            denom = sum(a * a for a in avals)
            slope = (sum(a * e for a, e in zip(avals, earners)) / denom
                     if denom > 0 else 0.0)
            slope = max(slope, 0.0)
        sse = sum((e - (intercept + slope * a)) ** 2
                  for a, e in zip(avals, earners))
        if best is None or sse < best[0]:
            best = (sse, intercept, slope, lam)
    return best[1], best[2], best[3]


def future_arrivals(h_now, h_tgt, p_now, p_tgt, segments=5):
    """Lump the forecast player growth into segments (time, dP)."""
    if h_tgt <= h_now or p_tgt <= p_now:
        return []
    out = []
    x0 = math.log1p(h_now + 2.0)
    x1 = math.log1p(h_tgt + 2.0)
    prev_p = p_now
    for s in range(segments):
        frac = (s + 1) / segments
        x = x0 + (x1 - x0) * frac
        h = math.expm1(x) - 2.0
        p = p_now * (p_tgt / p_now) ** frac
        mid = h_now + (h - h_now) * 0.5 if s == 0 else (prev_h + h) * 0.5
        out.append((mid, p - prev_p))
        prev_p = p
        prev_h = h
    return out


def predict_conv(times, deltas, earners, lam0, shape, share_prior,
                 all_times, all_deltas, h_tgt, e_now):
    """Fit the convolution model and predict earners at h_tgt.

    The eventual share F is shrunk toward the class prior when the
    trophy has few earners (F and the delay scale are weakly identified
    then), and the intercept is re-anchored so the curve passes through
    the current earner count.
    """
    a, fshare, lam = fit_convolution(times, deltas, earners, lam0, shape)
    w = e_now / (e_now + 20.0)
    fshare = w * fshare + (1.0 - w) * share_prior
    a_now = exposure(times, deltas, times[-1], lam, shape)
    a = clamp(e_now - fshare * a_now, 0.0, e_now)
    a_tgt = exposure(all_times, all_deltas, h_tgt, lam, shape)
    return a + fshare * a_tgt


def trend_median(times, rates, h_tgt):
    """Logit rate vs ln(1+h) fit on the last 48 hours of epochs."""
    pairs = [(t, r) for t, r in zip(times, rates) if r is not None]
    if not pairs:
        return 1.0
    recent = [(t, r) for t, r in pairs if t >= pairs[-1][0] - 48.0]
    if len(recent) >= 3:
        pairs = recent
    xs = [math.log1p(max(t, 0.0)) for t, _r in pairs]
    ys = [logit_pct(r) for _t, r in pairs]
    slope, intercept = linfit(xs, ys)
    slope = clamp(slope, -2.5, 2.5)
    return inv_logit_pct(intercept + slope * math.log1p(max(h_tgt, 0.0)))


def forecast_case(case):
    trophies = case.get("trophies") or []
    history = case.get("history") or []
    at = parse_time(case.get("at"))
    target = parse_time(case.get("target"))
    epochs = dedupe_epochs(history)
    if not epochs:
        epochs = history[-1:] if history else []
    # Bound the fitting cost: thin very long epoch lists evenly, keeping
    # the first and last epoch. Player deltas are recomputed from the
    # kept epochs below, so arrival totals are preserved.
    if len(epochs) > 36:
        keep = sorted(set(
            round(i * (len(epochs) - 1) / 35.0) for i in range(36)))
        epochs = [epochs[i] for i in keep]

    t0 = LAUNCH
    if epochs:
        first = parse_time(epochs[0].get("at"))
        if first is not None and first < t0:
            t0 = first
    if at is None:
        at = parse_time(epochs[-1].get("at")) if epochs else target
    if target is None:
        target = at

    p_target, sigma_logp, gap_med = forecast_players(epochs, t0, target)

    lag_h = 0.5 * gap_med
    horizon_h = max((target - at).total_seconds() / 3600.0 - lag_h, 0.25)
    horizon_days = horizon_h / 24.0
    h_now = (at - t0).total_seconds() / 3600.0
    h_tgt = h_now + horizon_h

    epoch_times = []
    epoch_players = []
    for e in epochs:
        dt = parse_time(e.get("at"))
        epoch_times.append(((dt - t0).total_seconds() / 3600.0) if dt else 0.0)
        epoch_players.append(float(e.get("players_tracked") or 0))
    p_now = epoch_players[-1] if epoch_players else p_target
    if p_now <= 0:
        p_now = max(p_target, 1.0)
    deltas = []
    prev = 0.0
    for p in epoch_players:
        deltas.append(max(p - prev, 0.0))
        prev = p
    fut = future_arrivals(epoch_times[-1] if epoch_times else h_now,
                          h_tgt, p_now, p_target)
    all_times = epoch_times + [f[0] for f in fut]
    all_deltas = deltas + [f[1] for f in fut]

    stale = False
    if len(epochs) <= 1 and len(history) >= 3:
        d_first = parse_time(history[0].get("at"))
        d_last = parse_time(history[-1].get("at"))
        if d_first and d_last:
            stale = (d_last - d_first).total_seconds() / 3600.0 >= 1.0

    # Rolling-origin origins shared by all trophies.
    origins = []
    if len(epochs) >= 4:
        start = max(2, len(epochs) - 4)
        origins = list(range(start, len(epochs) - 1))

    rows = []
    medians = {}
    for idx, trophy in enumerate(trophies):
        kind, share_prior, chapter_n = classify(trophy)
        lam0, shape = delay_prior(kind, chapter_n)

        earners_hist = []
        rates_hist = []
        last_rate = None
        for e, p in zip(epochs, epoch_players):
            rates = e.get("rates") or []
            r = rates[idx] if idx < len(rates) else None
            if r is not None:
                last_rate = float(r)
            rates_hist.append(last_rate)
            earners_hist.append(
                last_rate / 100.0 * p if (last_rate is not None and p > 0)
                else None)
        if last_rate is None:
            last_rate = clamp(100.0 * share_prior * 0.25, 0.05, 60.0)
        e_now = last_rate / 100.0 * p_now

        cand = {"persist": last_rate}

        if len(epochs) >= 2 and all(v is not None for v in earners_hist):
            e_conv = predict_conv(epoch_times, deltas, earners_hist,
                                  lam0, shape, share_prior,
                                  all_times, all_deltas, h_tgt, e_now)
            e_conv = max(e_conv, e_now)
            cap = share_prior * 1.6 * p_target
            if e_conv > cap:
                e_conv = cap + (e_conv - cap) * 0.25
            cand["conv"] = clamp(100.0 * e_conv / p_target, 0.01, 99.99)
            cand["trend"] = trend_median(epoch_times, rates_hist, h_tgt)
        else:
            # Single epoch: prior convolution from launch age.
            age = max(h_now, 1.0)
            gain_old = p_now * (weibull_cdf(age + horizon_h, lam0, shape)
                                - weibull_cdf(age, lam0, shape))
            gain_new = max(p_target - p_now, 0.0) * weibull_cdf(
                horizon_h * 0.5, lam0, shape)
            e_conv = e_now + share_prior * (gain_old + gain_new)
            cand["conv"] = clamp(100.0 * e_conv / p_target, 0.01, 99.99)

        # Rolling-origin validation weights (logit MAE per candidate).
        # Each origin predicts the LAST epoch, so the weights reflect
        # medium-horizon skill rather than one-refresh-ahead skill.
        weights = {}
        if origins and "conv" in cand:
            errs = {k: [] for k in cand}
            last_i = len(epochs) - 1
            for oi in origins:
                sub_t = epoch_times[:oi + 1]
                sub_d = deltas[:oi + 1]
                sub_e = earners_hist[:oi + 1]
                sub_r = rates_hist[:oi + 1]
                y = rates_hist[last_i]
                if y is None or any(v is None for v in sub_e):
                    continue
                errs["persist"].append(
                    abs(logit_pct(y) - logit_pct(sub_r[-1])))
                try:
                    a2, f2, lam2 = fit_convolution(
                        sub_t, sub_d, sub_e, lam0, shape)
                    pred_e = a2 + f2 * exposure(
                        sub_t, sub_d, epoch_times[last_i], lam2, shape)
                    pred_r = clamp(100.0 * pred_e / epoch_players[last_i],
                                   0.01, 99.99)
                    errs["conv"].append(abs(logit_pct(y) - logit_pct(pred_r)))
                except Exception:
                    pass
                if "trend" in cand:
                    pred_t = trend_median(sub_t, sub_r, epoch_times[last_i])
                    errs["trend"].append(abs(logit_pct(y) - logit_pct(pred_t)))
            for k in cand:
                e = errs.get(k) or []
                mae = sum(e) / len(e) if e else 0.30
                weights[k] = 1.0 / (mae + 0.04) ** 2
        if not weights:
            if len(epochs) >= 2:
                weights = {"persist": 0.20, "conv": 0.50, "trend": 0.30}
            else:
                # A refresh must happen before dilution can show up.
                w_dil = 1.0 - math.exp(-horizon_h / 4.5)
                weights = {"persist": 1.0 - w_dil, "conv": w_dil}
            weights = {k: v for k, v in weights.items() if k in cand}
        wsum = sum(weights.values()) or 1.0
        med_logit = sum(weights[k] * logit_pct(cand[k])
                        for k in weights) / wsum
        median = inv_logit_pct(med_logit)
        # A trophy already at the ceiling stays there: tracked players
        # earn the first trophy almost immediately, and the scoring
        # clamp makes 99.9 vs 99.0 a large logit distance.
        if last_rate >= 99.5:
            median = max(median, 99.9)
        medians[trophy.get("id")] = median

        refresh_scale = math.sqrt(1.0 - math.exp(-horizon_h / 4.5))
        grow_scale = math.sqrt(1.0 - math.exp(-horizon_h / 12.0))
        sigma = math.sqrt(
            SIGMA_BASE ** 2
            + (SIGMA_PER_SQRT_DAY ** 2) * horizon_days
            + (SIGMA_REFRESH * refresh_scale) ** 2
            + (min(SIGMA_COUNT_CAP,
                   SIGMA_COUNT_COEF / math.sqrt(e_now + 2.0))
               * grow_scale) ** 2
            + (0.55 * sigma_logp * grow_scale) ** 2
        )
        if stale:
            # A page that has not refreshed for hours can jump a long
            # way at the next refresh, whatever the horizon.
            sigma = math.sqrt(sigma ** 2 + (2.20 * refresh_scale) ** 2)
            sigma *= SIGMA_STALE_MULT if horizon_h >= 1.0 else 1.10
        sigma = clamp(sigma, SIGMA_MIN, SIGMA_MAX)

        centre = logit_pct(median)
        qs = [inv_logit_pct(centre + z * sigma) for z in Z]
        rows.append({"id": trophy.get("id"), "q": qs, "_sigma": sigma})

    # Funnel order on the story chain: medians non-increasing with rank.
    chain = [(story_chain_rank(t), t.get("id")) for t in trophies]
    chain = sorted((r, i) for r, i in chain if r is not None)
    if len(chain) >= 2:
        vals = [medians[i] for _, i in chain]
        blocks = [[v] for v in vals]
        changed = True
        while changed:
            changed = False
            for j in range(len(blocks) - 1):
                a = sum(blocks[j]) / len(blocks[j])
                b = sum(blocks[j + 1]) / len(blocks[j + 1])
                if a < b:
                    blocks[j] = blocks[j] + blocks[j + 1]
                    del blocks[j + 1]
                    changed = True
                    break
        flat = []
        for b in blocks:
            flat.extend([sum(b) / len(b)] * len(b))
        for (_, tid), v in zip(chain, flat):
            medians[tid] = v
        for row in rows:
            if row["id"] in medians:
                centre = logit_pct(medians[row["id"]])
                row["q"] = [inv_logit_pct(centre + z * row["_sigma"])
                            for z in Z]

    out_rows = []
    for row in rows:
        qs = []
        prev = 0.01
        for q in row["q"]:
            v = clamp(round(float(q), 4), 0.01, 99.99)
            v = max(v, prev)
            qs.append(v)
            prev = v
        out_rows.append({"id": row["id"], "q": qs})
    out_rows.sort(key=lambda r: (r["id"] is None, r["id"]))
    return {"trophies": out_rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    try:
        names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    except OSError:
        names = []
    for name in names:
        path = os.path.join(in_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
            result = forecast_case(case)
        except Exception:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    case = json.load(fh)
                hist = case.get("history") or []
                last = (hist[-1].get("rates") if hist else []) or []
                rows = []
                for i, t in enumerate(case.get("trophies") or []):
                    r = last[i] if i < len(last) and last[i] is not None else 1.0
                    r = clamp(float(r), 0.01, 99.99)
                    rows.append({"id": t.get("id"), "q": [r] * 5})
                result = {"trophies": rows}
            except Exception:
                continue
        cid = case.get("id") or name[:-3]
        text = json.dumps(result, separators=(",", ":"), ensure_ascii=False)
        with open(os.path.join(out_dir, str(cid) + ".out"), "w",
                  encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
