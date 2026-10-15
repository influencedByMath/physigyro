"""Pilot study on fit-half logs (THESIS.md 7.4.3, pilot steps 1-5; decision D-10).

It MEASURES and PROPOSES. Nothing is frozen here: the author approves the
proposed thresholds, which are then recorded as one decision in THESIS
Section 2 before any test-half log is processed.

Only fit-half vehicles are allowed: the script stops if a log belongs to a
test-half vehicle (data/screen/split.csv) or, before the split exists, to a
vehicle that is not forced into the fit half (D-17, D-19).

Run from the repository root after src/split.py and the pilot downloads:
    .venv/Scripts/python experiments/pilot_study/run_pilot.py
    .venv/Scripts/python experiments/pilot_study/run_pilot.py --logs data/raw_logs/b27d09e3-....ulg   (debug)
Writes results.json and arrays.npz next to this file; make_figures.py draws the figures.
"""
import argparse
import json
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import asdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_quality as dq          # noqa: E402
import fc_data as fd               # noqa: E402
import noise_data as nd            # noqa: E402
import split as sp                 # noqa: E402
from log_screen import clean_uuid  # noqa: E402
from validate_real import band_errors_db   # noqa: E402

TH = dq.CANDIDATES                 # candidates only; the pilot proposes the final values
BANDS = [(20, 60), (60, 90), (90, 130), (130, 180), (180, 240), (240, 320), (320, 400), (400, 800)]
GAP_FRACTIONS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30)
N_REP = 20
SEED = 20261014
OUT, ARR = {}, {}


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ which logs

def pilot_paths(args):
    """(log path, vehicle key) pairs; every vehicle must be in the fit half."""
    split_csv = os.path.join(ROOT, "data", "screen", "split.csv")
    split = pd.read_csv(split_csv) if os.path.exists(split_csv) else None
    if args.logs:
        pairs = []
        for p in args.logs:
            u = nd.load(p, [])
            key = "uuid:" + (clean_uuid(u.msg_info_dict.get("sys_uuid")) or "")
            pairs.append((p, key))
    else:
        pilot = pd.read_csv(os.path.join(ROOT, "data", "screen", "pilot_set.csv"))
        pairs = [(os.path.join(ROOT, "data", "raw_logs", r.log_ids.split(";")[0] + ".ulg"), r.vehicle_key)
                 for r in pilot.itertuples()]
    for p, key in pairs:
        if split is not None:
            row = split[split["vehicle_key"] == key]
            ok = len(row) == 1 and row["split"].iloc[0] == "fit"
        else:
            ok = key in sp.FORCED_FIT_KEYS
        if not ok:
            raise SystemExit(f"STOP: {os.path.basename(p)} ({key}) is not a fit-half vehicle (leakage rule D-10)")
    return pairs


# ------------------------------------------------------------------ per-log measurements

def gap_lengths_s(gs):
    """Lengths of the holes between gap-free FIFO segments [s]."""
    return [float(gs.t[a2] - gs.t[b1 - 1] - 1.0 / gs.fs_measured)
            for (a1, b1), (a2, b2) in zip(gs.segments[:-1], gs.segments[1:])]


def pick_windows(wins, W, n_max=8):
    """Non-overlapping windows that pass the CANDIDATE Q10 limits, quietest first."""
    ok = [w for w in wins if w["sp_std_dps"] <= TH.sp_std_max_dps
          and not (np.isfinite(w["thr_std"]) and w["thr_std"] > TH.thr_std_max)]
    out = []
    for w in sorted(ok, key=lambda w: w["sp_std_dps"]):
        if all(w["t1"] <= o["t0"] or w["t0"] >= o["t1"] for o in out):
            out.append(w)
        if len(out) == n_max:
            break
    return sorted(out, key=lambda w: w["t0"])


def rpm_unit_check(f, P, motor_hz):
    """Gate Q16 unit check: where is the strongest roll/pitch peak within +-25 % of f_m and of 2 f_m?"""
    fm = float(np.mean(motor_hz))
    out = {}
    for h in (1, 2):
        m = (f > 0.75 * h * fm) & (f < 1.25 * h * fm)
        if m.any():
            out[f"h{h}_peak_over_expected"] = round(float(f[m][np.argmax(P[m, :2].sum(1))] / (h * fm)), 3)
    return out


def process_log(path, key):
    t = time.perf_counter()
    u = nd.load(path)
    gs = nd.gyro_stream(u)
    stats = dq.log_stats(u, gs)
    fc = fd.fc_info(u)
    rej, flags = dq.log_gates(stats, fc, TH)
    W = TH.window_s
    wins = nd.airborne_windows(u, W=W, t_edge=TH.t_edge_s, step=1.0)
    nper = dq.nperseg_for(gs.fs_measured, TH.resolution_hz)
    picked = []
    for w in pick_windows(wins, W):
        try:
            f, P, navg, clipped = nd.window_psd(gs, w["t0"], w["t1"], nperseg=nper)
        except ValueError:
            continue
        rpm = nd.esc_rpm(u, w["t0"], w["t1"])
        wr = dict(w, n_averages=int(navg), clipped=int(clipped), rpm_valid=rpm is not None,
                  rpm_hz=None if rpm is None else [round(x, 2) for x in rpm["mean"] / 60],
                  rpm_std_hz=None if rpm is None else [round(x, 2) for x in rpm["std"] / 60],
                  rpm_rate_hz=None if rpm is None else round(rpm["rate_hz"], 1))
        wr["gates"] = dq.window_gates(wr, TH)
        if rpm is not None:
            wr["rpm_unit_check"] = rpm_unit_check(f, P, rpm["mean"] / 60)
        i = len(picked)
        ARR[f"{key}|w{i}|f"], ARR[f"{key}|w{i}|P"] = f.astype(np.float32), P.astype(np.float32)
        picked.append(wr)
    fcj = {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in fc.items()}
    return dict(path=os.path.relpath(path, ROOT).replace(os.sep, "/"), vehicle_key=key, stats=stats, fc=fcj,
                rotation=gs.rotation, is_external=gs.is_external,
                log_rejections=rej, log_flags=flags, gap_lengths_s=gap_lengths_s(gs), nperseg=nper,
                n_airborne_windows=len(wins),
                activity=[dict(sp=round(w["sp_std_dps"], 3), thr=None if not np.isfinite(w["thr_std"]) else
                               round(w["thr_std"], 5)) for w in wins],
                windows=picked, seconds=round(time.perf_counter() - t, 1)), (u, gs)


# ------------------------------------------------------------------ step 1: dropout experiment (D_max, L_min)

def cut_segments(segments, t, gaps):
    """Remove the time intervals ``gaps`` [(start, end), ...] from the gap-free segments."""
    out = []
    for a, b in segments:
        pieces = [(a, b)]
        for g0, g1 in gaps:
            nxt = []
            for c, d in pieces:
                i0, i1 = np.searchsorted(t[c:d], [g0, g1]) + c
                if i1 <= c or i0 >= d:
                    nxt.append((c, d))
                    continue
                if i0 > c:
                    nxt.append((c, int(i0)))
                if i1 < d:
                    nxt.append((int(i1), d))
            pieces = nxt
        out += pieces
    return out


def dropout_experiment(clean, pooled_gaps, rng):
    """Insert gaps drawn from the real gap lengths at increasing fractions; band errors vs the clean PSD."""
    res = {}
    for frac in GAP_FRACTIONS:
        errs, failed, few = [], 0, 0
        for key, gs, (t0, t1), nper in clean:
            f0, P0, _, _ = nd.window_psd(gs, t0, t1, nperseg=nper)
            for _ in range(N_REP):
                gaps, removed = [], 0.0
                while removed < frac * (t1 - t0):
                    L = float(rng.choice(pooled_gaps))
                    s = float(rng.uniform(t0, t1))
                    gaps.append((s, s + L))
                    removed += L
                g = nd.GyroStream(gs.t, gs.gyro, gs.raw_counts, gs.fs, gs.fs_measured, gs.device_id,
                                  cut_segments(gs.segments, gs.t, gaps), gs.gap_fraction)
                try:
                    _, P1, navg, _ = nd.window_psd(g, t0, t1, nperseg=nper)
                except ValueError:
                    failed += 1
                    continue
                few += navg < TH.min_averages
                for ax in (0, 1):
                    errs += finite_abs(band_errors_db(f0, P1[:, ax], P0[:, ax], BANDS))
        res[str(frac)] = dict(n_trials=len(clean) * N_REP, failed=failed, few_averages=int(few),
                              median_abs_db=round(float(np.median(errs)), 3) if errs else None,
                              p95_abs_db=round(float(np.percentile(errs, 95)), 3) if errs else None)
        log(f"   gaps {frac:.0%}: {res[str(frac)]}")
    return res


# ------------------------------------------------------------------ step 3: window length

def window_length_errors(u, gs, nper, acc):
    """Band powers of 5 s and 10 s sub-windows vs the 20 s window containing them (same flight);
    appends |dB| errors to acc["5"], acc["10"] (at most 3 non-overlapping 20 s windows per log)."""
    wins20 = [w for w in nd.airborne_windows(u, W=20.0, t_edge=TH.t_edge_s, step=5.0)
              if w["sp_std_dps"] <= TH.sp_std_max_dps and not (np.isfinite(w["thr_std"]) and w["thr_std"] > TH.thr_std_max)]
    used = []
    for w in wins20:
        if any(not (w["t1"] <= o[0] or w["t0"] >= o[1]) for o in used):
            continue
        try:
            f, P20, _, _ = nd.window_psd(gs, w["t0"], w["t1"], nperseg=nper)
        except ValueError:
            continue
        used.append((w["t0"], w["t1"]))
        for L in (5.0, 10.0):
            for s in np.arange(w["t0"], w["t1"] - L + 1e-9, L):
                try:
                    _, Ps, _, _ = nd.window_psd(gs, s, s + L, nperseg=nper)
                except ValueError:
                    continue
                for ax in (0, 1):
                    acc[str(int(L))] += finite_abs(band_errors_db(f, Ps[:, ax], P20[:, ax], BANDS))
        if len(used) == 3:
            break
    return len(used)


def finite_abs(x):
    x = np.abs(np.asarray(x, dtype=float))
    return list(x[np.isfinite(x)])


def sub_stream(gs, t0, t1):
    """The part of a GyroStream inside [t0, t1) (small enough to keep in memory for many logs)."""
    i0, i1 = np.searchsorted(gs.t, [t0, t1])
    segs = [(max(a, i0) - i0, min(b, i1) - i0) for a, b in gs.segments if b > i0 and a < i1]
    return nd.GyroStream(gs.t[i0:i1], gs.gyro[i0:i1], gs.raw_counts[i0:i1], gs.fs, gs.fs_measured, gs.device_id,
                         segs, gs.gap_fraction)


# ------------------------------------------------------------------ summaries and rule-based proposals

def share(x):
    return round(float(np.mean(x)), 3) if len(x) else None


def summarize(logs, dropout, wl):
    """Distributions and the THESIS rules applied to them (proposals only; the author decides)."""
    ok = [r for r in logs if "stats" in r]
    fs_meas = np.array([r["stats"]["fs_measured"] for r in ok])
    fs_nom = np.array([r["stats"]["fs_nominal"] for r in ok])
    cands = [500, 600, 700, 750, 780, 790, 800, 900, 1000, 1200, 1500, 2000, 2500]
    fmax = {str(c): dict(measured=share(0.8 * fs_meas / 2 >= c), nominal=share(0.8 * fs_nom / 2 >= c)) for c in cands}
    best = lambda key: max([c for c in cands if (fmax[str(c)][key] or 0) >= 0.9], default=None)
    rpm_logs = [r for r in ok if r["fc"]["has_rpm_topic"]]
    motor = [max(w["rpm_hz"]) for r in ok for w in r["windows"] if w.get("rpm_hz")]
    wins = [w for r in ok for w in r["windows"]]
    S = dict(
        n_logs=len(logs), n_errors=len(logs) - len(ok),
        fs=dict(measured=sorted(round(x, 1) for x in fs_meas), nominal=sorted(set(fs_nom.tolist()))),
        f_max_share_meeting=fmax, f_max_rule_90pct=dict(measured=best("measured"), nominal=best("nominal")),
        third_harmonic_max_hz=round(3 * max(motor), 1) if motor else None,
        dropout_fraction=sorted(round(r["stats"]["dropout_fraction"], 4) for r in ok),
        jitter=sorted(round(r["stats"]["jitter"], 5) for r in ok),
        logs_with_rate_change=sum(r["stats"]["n_rate_changes"] > 0 for r in ok),
        clipped_samples_per_log=sorted(r["stats"]["clipped_samples"] for r in ok),
        windows_passing_candidates_per_log=sorted(len(r["windows"]) for r in ok),
        welch_averages=sorted({(round(r["stats"]["fs_measured"]), w["n_averages"]) for r in ok for w in r["windows"]}),
        esc_rate_hz=sorted(round(r["fc"]["esc_rate_hz"], 1) for r in rpm_logs),
        rpm_unit_check=[w["rpm_unit_check"] for w in wins if w.get("rpm_unit_check")],
        shares=dict(defaults_known=share([r["fc"]["defaults_known"] for r in ok]),
                    fifo_is_control_gyro=share([r["fc"]["fifo_is_control_gyro"] for r in ok]),
                    fft_notch=share([r["fc"]["fft_notch"] for r in ok]),
                    changed_in_flight=share([bool(r["fc"]["changed_in_flight"]) for r in ok]),
                    has_rpm_topic=share([r["fc"]["has_rpm_topic"] for r in ok]),
                    has_highrate=share([r["fc"]["has_highrate"] for r in ok]),
                    has_sysid=share([r["fc"]["has_sysid"] for r in ok])),
        log_rejections=dict(Counter(c for r in ok for c in r["log_rejections"])),
        log_flags=dict(Counter(c for r in ok for c in r["log_flags"])),
        chips=dict(Counter(r["fc"]["fifo_chip"]["name"] for r in ok)),
        versions=dict(Counter(r["fc"]["version"] for r in ok)),
        board_rotation=dict(Counter(str(nd_rotation(r)) for r in ok)))
    if dropout:
        good = [float(k) for k, v in dropout.items() if v["median_abs_db"] is not None and v["median_abs_db"] < 1.0
                and v["few_averages"] <= 0.1 * max(1, v["n_trials"] - v["failed"])]
        S["d_max_rule"] = max(good, default=None)
    S["window_rule"] = min([int(L) for L in ("5", "10") if wl.get(L, {}).get("median_abs_db") is not None
                            and wl[L]["median_abs_db"] < 1.0], default=20)
    sp = np.array([w["sp"] for r in ok for w in r["activity"] if w["sp"] is not None])
    thr = np.array([w["thr"] for r in ok for w in r["activity"] if w["thr"] is not None])
    S["activity_share_kept"] = {str(x): share(sp <= x) for x in (5, 10, 15, 20, 30)}
    S["throttle_share_kept"] = {str(x): share(thr <= x) for x in (0.005, 0.01, 0.02, 0.05)}
    S["activity_quantiles"] = {str(q): round(float(np.quantile(sp, q)), 2) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(sp) else None
    S["throttle_quantiles"] = {str(q): round(float(np.quantile(thr, q)), 4) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(thr) else None
    return S


def nd_rotation(r):
    return r.get("rotation", "?")


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="*")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    t_all = time.perf_counter()
    pairs = pilot_paths(args)
    log(f"pilot: {len(pairs)} fit-half logs")
    logs, clean, wl_acc, n_wl = [], [], {"5": [], "10": []}, 0
    for path, key in pairs:
        log(f"\n=== {os.path.basename(path)[:8]} {key}")
        try:
            r, (u, gs) = process_log(path, key)
            n_wl += window_length_errors(u, gs, r["nperseg"], wl_acc)
        except Exception as e:                    # a crash on a real log is a finding, not a reason to stop
            logs.append(dict(path=path, vehicle_key=key, error=repr(e)[:300], trace=traceback.format_exc()[-1500:]))
            log(f"   ERROR {e!r}")
            continue
        logs.append(r)
        s = r["stats"]
        log(f"   fs {s['fs_measured']:.1f} Hz, dropout {s['dropout_fraction']:.3%}, segments {s['n_segments']}, "
            f"airborne {s['airborne_s']:.0f} s, windows passing candidates {len(r['windows'])}, "
            f"rejections {r['log_rejections']}, flags {r['log_flags']} ({r['seconds']} s)")
        if r["windows"] and s["dropout_fraction"] < 0.01:
            w = r["windows"][0]
            clean.append((key, sub_stream(gs, w["t0"], w["t1"]), (w["t0"], w["t1"]), r["nperseg"]))
        del u, gs
    OUT["logs"] = logs
    OUT["thresholds_used"] = asdict(TH)
    OUT["window_length"] = {L: dict(n=len(v), median_abs_db=round(float(np.median(v)), 3) if v else None,
                                    p90_abs_db=round(float(np.percentile(v, 90)), 3) if v else None)
                            for L, v in wl_acc.items()}
    OUT["window_length"]["n_20s_windows"] = n_wl
    log(f"\n=== step 3: window length (5 s and 10 s vs 20 s): {OUT['window_length']}")
    pooled = [g for r in logs if "gap_lengths_s" in r for g in r["gap_lengths_s"] if g > 0]
    OUT["gap_lengths_s"] = dict(n=len(pooled), median=float(np.median(pooled)) if pooled else None,
                                p90=float(np.percentile(pooled, 90)) if pooled else None)
    log(f"\n=== step 1: dropout experiment on {len(clean)} clean logs, {len(pooled)} real gaps")
    OUT["dropout_experiment"] = dropout_experiment(clean, pooled, rng) if clean and pooled else None
    OUT["summary"] = summarize(logs, OUT["dropout_experiment"], OUT["window_length"])
    for r in logs:                               # keep a summary of the gaps, not every gap (file size)
        g = np.array(r.pop("gap_lengths_s", []))
        r["gaps"] = dict(n=int(len(g)), median_s=float(np.median(g)) if len(g) else None,
                         p90_s=float(np.percentile(g, 90)) if len(g) else None, max_s=float(g.max()) if len(g) else None)
    log(json.dumps(OUT["summary"], indent=1, default=str)[:4000])
    OUT["meta"] = dict(date=time.strftime("%Y-%m-%d"), seconds=round(time.perf_counter() - t_all, 1), seed=SEED,
                       n_logs=len(pairs))
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(os.path.join(HERE, "arrays.npz"), **ARR)
    log(f"\nDone in {OUT['meta']['seconds']} s")


if __name__ == "__main__":
    main()
