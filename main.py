"""Forecaster kelangkaan trofi GTA VI (v3) — hazard-convolution + blend terkalibrasi.

Sebuah earn rate adalah earners / players_tracked. Kedua hitungan hanya
bertumbuh, tetapi dengan kecepatan berbeda, jadi sebuah rate bisa turun
karena dilusi pemain baru atau naik karena progres pemain lama. Program
ini meramalkan kedua hitungan secara terpisah lalu membaginya.

Ringkasan model:
  * Baca-baca identik berturut-turut dilipat menjadi epoch refresh
    Exophase; semua fitting memakai epoch, bukan read mentah.
  * Players tracked diramalkan dengan blend dua kurva (power law
    teredam dalam log-time dan kurva saturasi Weibull) yang bobotnya
    dipilih lewat rolling-origin; satu epoch memakai prior pertumbuhan
    launch.
  * Earners per trofi memakai model konvolusi hazard: pemain tiba
    dalam kohort di tiap epoch dan memperoleh trofi setelah delay
    Weibull. Skala delay di-grid-search per trofi, pasangan (offset,
    eventual share) di-fit least squares berbobot, lalu eventual share
    di-shrink ke prior kelas trofi (chapter cerita, collectible,
    aktivitas, 100%, platinum, dst.) yang dibaca dari teks trofi.
  * Median konvolusi di-blend pada skala logit dengan kandidat tren
    logit, kandidat kecepatan earner, dan persistence; bobot dari
    rolling-origin validation.
  * Spread logit-normal dikalibrasi dari residual backtest. Bila
    halaman stale (satu epoch selama berjam-jam read), kuantil diambil
    dari campuran dua logit-normal: dunia tanpa refresh dan dunia
    setelah refresh (dilusi), sehingga ekor bawah lebih panjang.
  * Median rantai cerita dipaksa tak-naik (funnel) dan platinum tidak
    melebihi median trofi mana pun.

Python standard library saja, deterministik, tanpa jaringan.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
MAX_EPOCHS = 30


# ---------------------------------------------------------------- util

def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit_pct(x):
    c = clamp_pct(x)
    return math.log(c / (100.0 - c))


def inv_logit_pct(v):
    if v >= 0:
        return 100.0 / (1.0 + math.exp(-v))
    e = math.exp(v)
    return 100.0 * e / (1.0 + e)


def round4(x):
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    s = text.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x; returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    return (my - b * mx, b)


# ------------------------------------------------------------ kelas trofi

def trophy_class(tr):
    """Return (class_key, delay_scale_prior_hours, eventual_share_prior)."""
    desc = (tr.get("description") or "").strip()
    low = desc.lower()
    tier = tr.get("tier") or "bronze"
    tid = tr.get("id") or 0
    if tier == "platinum":
        return ("platinum", 900.0, 0.012)
    if "100% completion" in low:
        return ("completion", 700.0, 0.02)
    m = re.search(r"complete chapter (\d+)", low)
    if m:
        n = int(m.group(1))
        return ("chapter%d" % n, 9.0 * (2.3 ** n), 0.78 * (0.74 ** n))
    if low == "complete the final mission.":
        return ("final", 320.0, 0.35)
    if "final mission for" in low:
        return ("strand_final", 170.0, 0.12)
    if tid == 2:
        return ("first", 1.5, 0.85)
    nums = []
    for tok in re.findall(r"\d[\d,]*", desc):
        try:
            nums.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    money = "$" in desc
    if re.match(r"complete \d", low) or (nums and not low.startswith("complete ")):
        big = money or any(v >= 100 for v in nums)
        if big:
            return ("collectible_big", 380.0, 0.028)
        return ("collectible", 160.0, 0.07)
    if low.startswith("complete "):
        k = max(0, tid - 8)
        return ("mission", 14.0 * (1.35 ** k), 0.35 * (0.92 ** k))
    return ("activity", 55.0, 0.22)


# ---------------------------------------------------------------- epoch

def build_epochs(case, origin):
    """Collapse consecutive identical reads into refresh epochs."""
    epochs = []
    for read in case["history"]:
        players = read.get("players_tracked") or 0
        rates = read.get("rates") or []
        t = (parse_time(read["at"]) - origin).total_seconds() / 3600.0
        if epochs and epochs[-1]["players"] == players and epochs[-1]["rates"] == rates:
            epochs[-1]["t_last"] = t
            epochs[-1]["reads"] += 1
            continue
        epochs.append({
            "t": t,
            "t_last": t,
            "players": float(players),
            "rates": list(rates),
            "reads": 1,
        })
    if len(epochs) > MAX_EPOCHS:
        keep = []
        step = (len(epochs) - 1) / float(MAX_EPOCHS - 1)
        idx = sorted(set(int(round(i * step)) for i in range(MAX_EPOCHS)))
        keep = [epochs[i] for i in idx]
        epochs = keep
    return epochs


def cohorts_from_epochs(epochs):
    """Arrival cohorts (time, delta_players) implied by the epochs."""
    out = []
    prev = 0.0
    for ep in epochs:
        delta = ep["players"] - prev
        if delta > 0:
            out.append((ep["t"], delta))
        prev = max(prev, ep["players"])
    return out


# ------------------------------------------------------------- pemain

def _power_fit(epochs):
    xs = [math.log(1.0 + max(ep["t"], 0.0) + 0.1) for ep in epochs]
    ys = [math.log(max(ep["players"], 1.0)) for ep in epochs]
    return wls_line(xs, ys, [1.0] * len(xs))


def _power_pred(fit, t):
    if fit is None:
        return None
    a, b = fit
    return math.exp(a + b * math.log(1.0 + max(t, 0.0) + 0.1))


def _sat_basis(t, tau, k):
    tt = max(t, 0.0)
    return 1.0 - math.exp(-((tt / tau) ** k)) if tau > 0 else 0.0


def _sat_fit(epochs):
    best = None
    ys = [ep["players"] for ep in epochs]
    for tau in (24.0, 40.0, 60.0, 90.0, 140.0, 220.0):
        for k in (0.9, 1.2, 1.6):
            xs = [_sat_basis(ep["t"], tau, k) for ep in epochs]
            fit = wls_line(xs, ys, [1.0] * len(xs))
            if fit is None:
                continue
            c0, c1 = fit
            if c1 < 0:
                continue
            sse = sum((y - (c0 + c1 * x)) ** 2 for x, y in zip(xs, ys))
            if best is None or sse < best[0]:
                best = (sse, c0, c1, tau, k)
    if best is None:
        return None
    return best[1:]


def _sat_pred(fit, t):
    if fit is None:
        return None
    c0, c1, tau, k = fit
    return max(c0 + c1 * _sat_basis(t, tau, k), 1.0)


def fit_players(epochs, t_fc, horizon_days):
    """Return (P_at_fc, sigma_logp, predict_fn)."""
    n = len(epochs)
    last = epochs[-1]
    if n == 1:
        # Prior pertumbuhan launch P ~ (age + c)^gamma dengan offset c=2 jam
        # agar epoch pertama yang sangat muda tidak meledakkan ramalan.
        age = max(last["t"], 0.25)
        age_fc = max(t_fc, age + 0.01)
        if age < 7.2:
            gamma = 1.5
        elif age < 24.0:
            gamma = 1.15
        else:
            gamma = 0.65
        c = 2.0
        p_fc = last["players"] * (((age_fc + c) / (age + c)) ** gamma)
        sigma = 0.45 + 0.10 * math.sqrt(max(horizon_days, 0.0))

        def predict(t, _p=last["players"], _a=age, _g=gamma, _c=c):
            return _p * (((max(t, _a + 0.01) + _c) / (_a + _c)) ** _g)

        return p_fc, min(sigma, 0.9), predict

    power = _power_fit(epochs)
    sat = _sat_fit(epochs)
    # Rolling-origin weights on ln P (predict the last epoch).
    errs = {"power": [], "sat": []}
    for cut in range(max(1, n - 4), n - 1):
        train = epochs[:cut + 1]
        if len(train) < 2:
            continue
        target = math.log(max(last["players"], 1.0))
        pp = _power_pred(_power_fit(train), last["t"])
        ps = _sat_pred(_sat_fit(train), last["t"])
        if pp and pp > 0:
            errs["power"].append(abs(math.log(pp) - target))
        if ps and ps > 0:
            errs["sat"].append(abs(math.log(ps) - target))
    weights = {}
    for key, vals in errs.items():
        mae = sum(vals) / len(vals) if vals else 0.25
        weights[key] = 1.0 / ((mae + 0.03) ** 2)
    if sat is None:
        weights["sat"] = 0.0
    if power is None:
        weights["power"] = 0.0
    total = weights["power"] + weights["sat"]
    if total <= 0:
        weights = {"power": 1.0, "sat": 0.0}
        total = 1.0
    wp = weights["power"] / total
    ws = weights["sat"] / total

    def predict(t):
        vals = []
        if wp > 0:
            v = _power_pred(power, t)
            if v:
                vals.append((wp, math.log(max(v, 1.0))))
        if ws > 0:
            v = _sat_pred(sat, t)
            if v:
                vals.append((ws, math.log(max(v, 1.0))))
        if not vals:
            return last["players"]
        tw = sum(w for w, _ in vals)
        return math.exp(sum(w * lv for w, lv in vals) / tw)

    p_fc = max(predict(t_fc), last["players"] * 0.98)
    # Residual scale in log space.
    resid = []
    for ep in epochs:
        pv = predict(ep["t"])
        if pv > 0 and ep["players"] > 0:
            resid.append(math.log(ep["players"]) - math.log(pv))
    if resid:
        mean_r = sum(resid) / len(resid)
        var = sum((r - mean_r) ** 2 for r in resid) / len(resid)
        sigma = math.sqrt(var + (0.08 * math.sqrt(max(horizon_days, 0.0))) ** 2)
    else:
        sigma = 0.3
    return p_fc, min(max(sigma, 0.03), 0.8), predict


# ------------------------------------------------------- konvolusi hazard

def weibull_cdf(dt, scale, shape):
    if dt <= 0 or scale <= 0:
        return 0.0
    x = dt / scale
    return 1.0 - math.exp(-(x ** shape))


def conv_basis(epoch_times, cohorts, scale, shape):
    out = []
    for t in epoch_times:
        s = 0.0
        for tc, dp in cohorts:
            if tc <= t:
                s += dp * weibull_cdf(t - tc, scale, shape)
        out.append(s)
    return out


def fit_convolution(epochs, cohorts, idx, prior_scale, prior_share):
    """Fit (a, F, scale, shape) for trophy column idx; returns dict."""
    times = []
    earners = []
    weights = []
    for ep in epochs:
        rate = ep["rates"][idx] if idx < len(ep["rates"]) else None
        if rate is None:
            continue
        times.append(ep["t"])
        earners.append(rate * ep["players"] / 100.0)
        weights.append(math.sqrt(max(ep["players"], 1.0)))
    if not times:
        return {"a": 0.0, "F": prior_share, "scale": prior_scale, "shape": 1.25,
                "e_last": 0.0}
    e_last = earners[-1]
    if len(times) == 1:
        scale, shape = prior_scale, 1.25
        s_last = conv_basis(times, cohorts, scale, shape)[-1]
        a = min(max(e_last - prior_share * s_last, 0.0), e_last)
        return {"a": a, "F": prior_share, "scale": scale, "shape": shape,
                "e_last": e_last}
    best = None
    for smult in (0.3, 0.45, 0.65, 0.9, 1.25, 1.7, 2.3, 3.1, 4.2):
        scale = prior_scale * smult
        for shape in (0.9, 1.1, 1.35, 1.7):
            basis = conv_basis(times, cohorts, scale, shape)
            fit = wls_line(basis, earners, weights)
            if fit is None:
                continue
            a, f = fit
            sse = sum(w * (e - (a + f * s)) ** 2
                      for e, s, w in zip(earners, basis, weights))
            sse *= 1.0 + 0.03 * abs(math.log(smult))
            if best is None or sse < best[0]:
                best = (sse, a, f, scale, shape, basis[-1])
    if best is None:
        return {"a": 0.0, "F": prior_share, "scale": prior_scale, "shape": 1.25,
                "e_last": e_last}
    _, a, f, scale, shape, s_last = best
    f = min(max(f, 0.0), 1.0)
    w = e_last / (e_last + 20.0)
    f = w * f + (1.0 - w) * prior_share
    a = min(max(e_last - f * s_last, 0.0), e_last)
    return {"a": a, "F": f, "scale": scale, "shape": shape, "e_last": e_last}


def conv_predict(fit, cohorts, t_fc):
    s = 0.0
    for tc, dp in cohorts:
        if tc <= t_fc:
            s += dp * weibull_cdf(t_fc - tc, fit["scale"], fit["shape"])
    return fit["a"] + fit["F"] * s


# ------------------------------------------------------------ kandidat

def trend_candidate(epochs, idx, t_fc):
    last_t = epochs[-1]["t"]
    xs, ys = [], []
    for ep in epochs:
        if ep["t"] < last_t - 48.0:
            continue
        rate = ep["rates"][idx] if idx < len(ep["rates"]) else None
        if rate is None:
            continue
        xs.append(math.log(1.0 + max(ep["t"], 0.0) + 0.1))
        ys.append(logit_pct(rate))
    if len(xs) < 3:
        return None
    fit = wls_line(xs, ys, [1.0] * len(xs))
    if fit is None:
        return None
    a, b = fit
    val = a + b * math.log(1.0 + max(t_fc, 0.0) + 0.1)
    last_rate = epochs[-1]["rates"][idx]
    if last_rate is not None:
        base = logit_pct(last_rate)
        val = min(max(val, base - 2.5), base + 2.5)
    return inv_logit_pct(val)


def velocity_candidate(epochs, idx, t_fc, p_fc, prior_share):
    if len(epochs) < 2:
        return None
    pts = []
    for ep in epochs:
        rate = ep["rates"][idx] if idx < len(ep["rates"]) else None
        if rate is not None:
            pts.append((ep["t"], rate * ep["players"] / 100.0))
    if len(pts) < 2:
        return None
    t0, e0 = pts[max(0, len(pts) - 3)]
    t1, e1 = pts[-1]
    if t1 <= t0:
        return None
    vel = max((e1 - e0) / (t1 - t0), 0.0)
    dt = max(t_fc - t1, 0.0)
    pool = max(prior_share * p_fc - e1, 0.02 * p_fc)
    if pool > 0 and vel > 0:
        e_fut = e1 + pool * (1.0 - math.exp(-vel * dt / pool))
    else:
        e_fut = e1 + 0.3 * vel * dt
    return 100.0 * min(e_fut, p_fc) / max(p_fc, 1.0)


def last_rate_of(epochs, idx):
    for ep in reversed(epochs):
        if idx < len(ep["rates"]) and ep["rates"][idx] is not None:
            return ep["rates"][idx]
    return 0.0


# ---------------------------------------------------------------- spread

def mixture_quantiles(mu0, s0, mu1, s1, w1):
    """Quantiles of (1-w1)*N(mu0,s0) + w1*N(mu1,s1) on the logit scale."""
    lo = logit_pct(0.01)
    hi = logit_pct(99.99)

    def cdf(v):
        return (1.0 - w1) * norm_cdf((v - mu0) / s0) + w1 * norm_cdf((v - mu1) / s1)

    out = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(60):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        out.append(inv_logit_pct(0.5 * (a + b)))
    return out


def pava_nonincreasing(values):
    """Pool-adjacent-violators for a non-increasing sequence."""
    blocks = [[v, 1] for v in values]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            total = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            cnt = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [total / cnt, cnt]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for val, cnt in blocks:
        out.extend([val] * cnt)
    return out


# ------------------------------------------------------------- forecast

def forecast_case(case):
    origin = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    horizon_min = float(case.get("horizon_minutes") or
                        (target - at).total_seconds() / 60.0)
    horizon_h = horizon_min / 60.0
    horizon_days = horizon_min / 1440.0
    trophies = case["trophies"]
    epochs = build_epochs(case, origin)
    if not epochs:
        raise ValueError("case has no history")
    t_target = (target - origin).total_seconds() / 3600.0
    t_at = (at - origin).total_seconds() / 3600.0

    # Refresh timing: answer shows the last refresh before the target.
    gaps = [epochs[i + 1]["t"] - epochs[i]["t"] for i in range(len(epochs) - 1)]
    mean_gap = (sum(gaps) / len(gaps)) if gaps else 4.5
    lag = min(0.5 * mean_gap, 0.4 * horizon_h)
    t_fc = max(t_target - lag, epochs[-1]["t"] + 0.01)

    p_fc, sigma_logp, predict_p = fit_players(epochs, t_fc, horizon_days)

    # Cohorts: observed arrivals + forecast arrivals in 5 segments.
    cohorts = cohorts_from_epochs(epochs)
    t_last_ep = epochs[-1]["t"]
    seg = 5
    prev_p = epochs[-1]["players"]
    for s_i in range(1, seg + 1):
        t_seg = t_last_ep + (t_fc - t_last_ep) * s_i / seg
        p_seg = predict_p(t_seg)
        delta = p_seg - prev_p
        if delta > 0:
            cohorts.append((t_seg, delta))
        prev_p = max(prev_p, p_seg)

    # Refresh scale: typical |delta logit| between epochs.
    jumps = []
    for i in range(len(epochs) - 1):
        for j in range(len(trophies)):
            r0 = epochs[i]["rates"][j] if j < len(epochs[i]["rates"]) else None
            r1 = epochs[i + 1]["rates"][j] if j < len(epochs[i + 1]["rates"]) else None
            if r0 is not None and r1 is not None:
                jumps.append(abs(logit_pct(r1) - logit_pct(r0)))
    refresh_scale = median(jumps) if jumps else 0.6
    refresh_scale = min(max(refresh_scale, 0.15), 1.5)

    read_span_h = 0.0
    if case["history"]:
        first_t = (parse_time(case["history"][0]["at"]) - origin).total_seconds() / 3600.0
        read_span_h = t_at - first_t
    stale = len(epochs) == 1 and read_span_h >= 1.0
    p_refresh = 1.0 - math.exp(-horizon_h / 4.5)

    n_ep = len(epochs)
    rows = []
    medians = []
    class_keys = []
    for idx, tr in enumerate(trophies):
        ckey, prior_scale, prior_share = trophy_class(tr)
        class_keys.append(ckey)
        last_rate = last_rate_of(epochs, idx)
        fit = fit_convolution(epochs, cohorts, idx, prior_scale, prior_share)
        e_conv = conv_predict(fit, cohorts, t_fc)
        rate_conv = 100.0 * min(e_conv, p_fc) / max(p_fc, 1.0)
        rate_trend = trend_candidate(epochs, idx, t_fc)
        rate_vel = velocity_candidate(epochs, idx, t_fc, p_fc, prior_share)
        cands = {"conv": rate_conv, "persist": last_rate}
        if rate_trend is not None:
            cands["trend"] = rate_trend
        if rate_vel is not None:
            cands["vel"] = rate_vel

        # Rolling-origin weights: origins predict the last epoch.
        back_err = []
        if n_ep >= 3:
            errs = {k: [] for k in cands}
            last_ep_rate = last_rate
            for cut in range(max(0, n_ep - 5), n_ep - 1):
                train = epochs[:cut + 1]
                if len(train) < 2:
                    continue
                t_o = epochs[-1]["t"]
                p_o = epochs[-1]["players"]
                co_o = cohorts_from_epochs(train)
                fit_o = fit_convolution(train, co_o, idx, prior_scale, prior_share)
                pred = {"conv": 100.0 * min(conv_predict(fit_o, co_o, t_o), p_o) / max(p_o, 1.0),
                        "persist": last_rate_of(train, idx)}
                tr_o = trend_candidate(train, idx, t_o)
                if tr_o is not None:
                    pred["trend"] = tr_o
                ve_o = velocity_candidate(train, idx, t_o, p_o, prior_share)
                if ve_o is not None:
                    pred["vel"] = ve_o
                for key, vals in errs.items():
                    if key in pred:
                        vals.append(abs(logit_pct(pred[key]) - logit_pct(last_ep_rate)))
            weights = {}
            for key in cands:
                vals = errs.get(key) or []
                mae = sum(vals) / len(vals) if vals else 0.6
                weights[key] = 1.0 / ((mae + 0.04) ** 2)
                if key == "conv":
                    back_err = vals
        elif n_ep == 1:
            weights = {"conv": 0.8 * p_refresh, "persist": 1.0 - 0.8 * p_refresh}
        else:
            weights = {"conv": 0.55, "persist": 0.3}
            if "vel" in cands:
                weights["vel"] = 0.15
            if "trend" in cands:
                weights["trend"] = 0.15
        tw = sum(weights.get(k, 0.0) for k in cands)
        if tw <= 0:
            weights = {"persist": 1.0}
            tw = 1.0
        mu = sum(weights.get(k, 0.0) * logit_pct(v) for k, v in cands.items()) / tw
        med = inv_logit_pct(mu)
        medians.append(med)

        # Spread.
        e_now = fit["e_last"]
        small = min(0.42, 0.8 / math.sqrt(e_now + 2.0))
        hfac = 1.0 - math.exp(-horizon_h / 12.0)
        var = (0.10 ** 2 + (0.24 * math.sqrt(max(horizon_days, 0.0))) ** 2
               + (0.5 * refresh_scale) ** 2 * (1.0 - math.exp(-horizon_h / 4.5))
               + small ** 2 * hfac + (0.55 * sigma_logp) ** 2 * hfac)
        sigma = math.sqrt(var)
        if back_err:
            mae_b = sum(back_err) / len(back_err)
            sigma_b = 1.15 * mae_b
            sigma = math.sqrt(0.5 * sigma ** 2 + 0.5 * sigma_b ** 2)
        # Horizon pendek cenderung terlalu lebar: skala moderat < 2 hari.
        sigma *= 0.85 + 0.15 * min(1.0, max(horizon_days, 0.0) / 2.0)
        sigma = min(max(sigma, 0.08), 3.2)

        if stale:
            mu0 = logit_pct(last_rate)
            s0 = 0.12 + 0.10 * math.sqrt(max(horizon_days, 0.0))
            mu1 = logit_pct(rate_conv)
            s1 = max(0.9 * sigma, 0.7)
            qs = mixture_quantiles(mu0, s0, mu1, s1, p_refresh)
        else:
            qs = [inv_logit_pct(mu + z * sigma) for z in Z]
        rows.append({"id": tr["id"], "q": qs, "mu": mu, "last": last_rate})

    # Funnel: story chain medians non-increasing.
    chain_order = ["first", "chapter1", "chapter2", "chapter3", "chapter4",
                   "final", "completion", "platinum"]
    chain_idx = []
    for key in chain_order:
        for i, ck in enumerate(class_keys):
            if ck == key:
                chain_idx.append(i)
                break
    if len(chain_idx) >= 2:
        chain_meds = [medians[i] for i in chain_idx]
        adjusted = pava_nonincreasing(chain_meds)
        for pos, i in enumerate(chain_idx):
            if abs(adjusted[pos] - medians[i]) > 1e-9:
                shift = logit_pct(adjusted[pos]) - rows[i]["mu"]
                rows[i]["q"] = [inv_logit_pct(logit_pct(q) + shift) for q in rows[i]["q"]]
                rows[i]["mu"] += shift
                medians[i] = adjusted[pos]
    # Platinum cannot exceed any other trophy's median.
    plat_pos = [i for i, ck in enumerate(class_keys) if ck == "platinum"]
    if plat_pos and len(medians) > 1:
        i = plat_pos[0]
        others_min = min(m for j, m in enumerate(medians) if j != i)
        if medians[i] > others_min:
            shift = logit_pct(others_min) - rows[i]["mu"]
            rows[i]["q"] = [inv_logit_pct(logit_pct(q) + shift) for q in rows[i]["q"]]
            rows[i]["mu"] += shift
            medians[i] = others_min
    # A trophy already ~100% stays ~100% (clamp geometry).
    for i, row in enumerate(rows):
        if row["last"] >= 99.5 and medians[i] < 99.9:
            shift = logit_pct(99.9) - row["mu"]
            row["q"] = [inv_logit_pct(logit_pct(q) + shift) for q in row["q"]]
            row["mu"] += shift
            medians[i] = 99.9

    out_rows = []
    for row in rows:
        qs = sorted(clamp_pct(q) for q in row["q"])
        out_rows.append({"id": row["id"], "q": [round4(q) for q in qs]})
    return {"trophies": out_rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError("unexpected levels in %s" % path.name)
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
