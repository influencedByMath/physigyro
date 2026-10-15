"""Spectrum library: frozen gates (D-23) -> selected steady windows -> PSDs + metadata (THESIS 7.4.3-7.4.5, 7.5).

One log gives at most ``th.windows_per_vehicle`` windows (one screened log per vehicle, D-21). Every
candidate window is GATED FIRST (Q10 activity and throttle, Q9 clipping, Q13 Welch averages) and only then
selected: quietest first (rate-setpoint activity), non-overlapping. Selecting first and gating after would
let a gappy quiet window block an overlapping good one.

The library stores, per window and body axis (roll, pitch, yaw), the native Welch PSD (resolution
fs/nperseg, about 1.5 Hz) and the PSD on one common grid 0-f_max (log-log interpolation), plus metadata and
CC-BY provenance. Clustering features are band powers of NATIVE bins (``band_features``).
"""
import os
import time
from collections import Counter
from dataclasses import asdict

import numpy as np

import data_quality as dq
import fc_data as fd
import noise_data as nd

VERIFIED_ROTATIONS = (0, 2, 4)      # board rotations confirmed on data (pilot axes.json, 17 of 17 logs)
LICENCE = "CC-BY 4.0, PX4 Flight Review (review.px4.io), PX4 maintainers"


def axis_gate(rotation, check):
    """Gate Q7. The data check (noise_data.axis_correlation) decides when it exists; without it, only
    rotations already verified on data are accepted. An inconclusive check (diagonal_ok None: too little
    roll/pitch motion) counts as no check."""
    if check is not None and check["diagonal_ok"] is not None:
        return ([], []) if check["diagonal_ok"] else (["Q7_axes"], [])
    if int(rotation) in VERIFIED_ROTATIONS:
        return [], ["Q7_not_checked"]
    return ["Q7_unverified"], []


def window_reasons(cands, th):
    """First rejection reason of every rejected candidate window (data funnel)."""
    return Counter(r[0] for r in (dq.window_gates(w, th)[0] for w in cands) if r)


def select_windows(cands, th):
    """Candidates passing every window gate, then greedy quietest-first non-overlapping, at most the Q14 cap;
    returned in time order."""
    ok = [w for w in cands if not dq.window_gates(w, th)[0]]
    out = []
    for w in sorted(ok, key=lambda w: (w["sp_std_dps"], w["t0"])):
        if all(w["t1"] <= o["t0"] or w["t0"] >= o["t1"] for o in out):
            out.append(w)
            if len(out) == th.windows_per_vehicle:
                break
    return sorted(out, key=lambda w: w["t0"])


def common_grid(th):
    return np.arange(0.0, th.f_max + 1e-9, th.resolution_hz)


def to_grid(f, P, grid):
    """PSD (F, axes) on the native grid -> common grid, interpolating log10 power against log10 frequency
    (exact for power laws); the 0 Hz point is copied."""
    P = np.asarray(P, float)
    out = np.empty((len(grid), P.shape[1]))
    pos = grid > 0
    lf = np.log10(f[1:])
    for i in range(P.shape[1]):
        out[pos, i] = 10 ** np.interp(np.log10(grid[pos]), lf, np.log10(P[1:, i]))
    out[~pos] = P[0]
    return out


def band_edges(th, n=48):
    """n log-spaced bands over the fixed band [f_min, f_max] (THESIS 7.5 step 1)."""
    return np.geomspace(th.f_min, th.f_max, n + 1)


def band_features(f, P, edges):
    """log10 of the mean power of the native bins in each band [lo, hi); shape (n_bands, axes)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if not m.any():
            raise ValueError(f"no native bin in band {lo:.1f}-{hi:.1f} Hz")
        out.append(np.log10(P[m].mean(0)))
    return np.array(out)


def _jsonable(fc):
    return {k: (asdict(v) if hasattr(v, "__dataclass_fields__") else v) for k, v in fc.items()}


def process_log(path, vehicle_key, th):
    """One log -> (record, arrays). Applies log gates (Q4-Q8, Q11, Q16 flags), the axis gate (Q7) and the
    window gates; arrays hold the native f and P (float32) of each selected window."""
    t_start = time.perf_counter()
    u = nd.load(path)
    gs = nd.gyro_stream(u)
    stats = dq.log_stats(u, gs)
    fc = fd.fc_info(u)
    rej, flags = dq.log_gates(stats, fc, th)
    try:
        check = nd.axis_correlation(u, gs)
    except Exception:                         # no vehicle_angular_velocity
        check = None
    ar, af = axis_gate(gs.rotation, check)
    rej, flags = rej + ar, flags + af
    log_id = os.path.basename(path)[:-4]
    rec = dict(log_id=log_id, vehicle_key=vehicle_key, stats=stats, fc=_jsonable(fc), rotation=gs.rotation,
               is_external=gs.is_external, axis_check=check, log_rejections=rej, log_flags=flags,
               nperseg=None, n_candidates=0, window_reasons={}, windows=[])
    arrays = {}
    if not rej:
        nper = dq.nperseg_for(gs.fs_measured, th.resolution_hz)
        clip = np.concatenate([[0], np.cumsum((np.abs(gs.raw_counts) >= 32000).any(axis=1))])
        cands = nd.airborne_windows(u, W=th.window_s, t_edge=th.t_edge_s, step=1.0)
        for w in cands:
            w["n_averages"] = nd.welch_count(gs, w["t0"], w["t1"], nper)
            i0, i1 = np.searchsorted(gs.t, [w["t0"], w["t1"]])   # all samples of the window (stricter than PSD)
            w["clipped"] = int(clip[i1] - clip[i0])
        rec.update(nperseg=nper, n_candidates=len(cands), window_reasons=dict(window_reasons(cands, th)))
        for i, w in enumerate(select_windows(cands, th)):
            f, P, navg, _ = nd.window_psd(gs, w["t0"], w["t1"], nperseg=nper)
            if navg != w["n_averages"]:
                raise AssertionError(f"Welch average count mismatch {navg} != {w['n_averages']}")
            rpm = nd.esc_rpm(u, w["t0"], w["t1"])
            w = dict(w, rpm_valid=rpm is not None,
                     rpm_hz=None if rpm is None else [round(float(x), 3) for x in rpm["mean"] / 60],
                     rpm_std_hz=None if rpm is None else [round(float(x), 3) for x in rpm["std"] / 60],
                     rpm_rate_hz=None if rpm is None else round(float(rpm["rate_hz"]), 2))
            w["rpm_tracking"] = bool(w["rpm_valid"] and w["rpm_rate_hz"] >= th.rpm_rate_min_hz)   # D-13
            w["gates"] = dq.window_gates(w, th)
            rec["windows"].append(w)
            arrays[f"{log_id}|w{i}|f"], arrays[f"{log_id}|w{i}|P"] = f.astype(np.float32), P.astype(np.float32)
    rec["seconds"] = round(time.perf_counter() - t_start, 1)
    return rec, arrays


def process_half(rows, half, raw_dir, th, process=process_log, done=None, sink=None):
    """Run ``process`` on the screened log of every vehicle of one split half (rows of split.csv).

    Rows of the other half are never touched. A log that fails gets one record with rejection ``error``.
    ``done`` (log_id -> record) skips finished logs (resume); ``sink(record, arrays)`` is called for each
    newly processed log so the caller can save as it goes."""
    done = done or {}
    out = []
    for r in rows:
        if r["split"] != half:
            continue
        for log_id in r["log_ids"].split(";"):
            if log_id in done:
                out.append(done[log_id])
                continue
            try:
                rec, arrays = process(os.path.join(raw_dir, log_id + ".ulg"), r["vehicle_key"], th)
            except Exception as e:
                rec, arrays = dict(log_id=log_id, vehicle_key=r["vehicle_key"], log_rejections=["error"],
                                   error=repr(e)[:300], log_flags=[], windows=[]), {}
            out.append(rec)
            if sink is not None:
                sink(rec, arrays)
    return out
