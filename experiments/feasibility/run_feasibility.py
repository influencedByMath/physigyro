"""Feasibility test of the pipeline up to the physics model and the optimizer (no ML).

Day 13 (2026-10-10); audited and re-run on Day 17 (2026-10-14). Run from the repository root:
    .venv/Scripts/python experiments/feasibility/run_feasibility.py
Writes results.json and arrays.npz next to this file; make_figures.py draws the plots.

Stages
  1  data: two public PX4 logs -> gates, steady window, raw-gyro PSD, ESC RPM
  2  real-data validation of the filter model and the noise metric (E10, E11)
  3  plant identification (Tier 2) and realism check with the log's real gains
  4  optimizer: real-gain drone (reference = pilot's setting) and synthetic
     Tier 3a drones with real noise (reference = PX4 v1.16 defaults);
     modes A, B, C; brute-force check; timing
"""
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
from scipy.signal import find_peaks, welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import noise_data as nd          # noqa: E402
import optimizer as opt          # noqa: E402
import sysid                     # noqa: E402
import validate_real as vr       # noqa: E402
from loop_model import (PX4_V116_MC_DEFAULT, FilterSettings, LoopCase, Plant, Rates,  # noqa: E402
                        design_gains)

SEED = 20261010
LOGS = {
    "b27d09e3": dict(file="b27d09e3-4542-4496-a18b-2c3c5317024e.ulg", main=True),
    "a2571040": dict(file="a2571040-29ef-4b1a-b7cf-7776f1a78a1f.ulg", main=False),
}
URL = "https://review.px4.io/plot_app?log={}"
BANDS = [(20, 60), (60, 90), (90, 130), (130, 180), (180, 240), (240, 320), (320, 399)]
F_HI = 800.0
MIN_RPM_LOG_HZ = 50.0   # P1 gate (candidate, pilot study): tracking notch only with a high logged RPM rate
OUT, ARR = {}, {}


def log(*a):
    print(*a, flush=True)


def metrics_row(m):
    keys = ["PM", "GM", "Ms", "fc", "DM", "tau_F", "tau_D", "N_rel", "stable"]
    return {k: (bool(m[k]) if k == "stable" else round(float(m[k]), 4)) for k in keys}


def settings_row(s: FilterSettings):
    d = asdict(s)
    d["notches"] = [list(map(float, n)) for n in s.notches]
    return d


def top_peaks(f, P, f_lo=20.0, f_hi=F_HI, n=6, prom=0.5):
    sel = (f > f_lo) & (f < f_hi)
    lp = np.log10(P[sel])
    pk, pr = find_peaks(lp, prominence=prom)
    order = np.argsort(pr["prominences"])[::-1][:n]
    return [(float(f[sel][pk][i]), float(pr["prominences"][i])) for i in order]


# ----------------------------------------------------------------- stage 1
def stage_data():
    log("\n=== Stage 1: data")
    res = {}
    for key, info in LOGS.items():
        t = time.perf_counter()
        u = nd.load(os.path.join(ROOT, "data", "raw_logs", info["file"]))
        gs = nd.gyro_stream(u)
        fc = nd.fc_settings(u)
        try:
            sel = [int(x) for x in u.get_dataset("sensor_selection").data["gyro_device_id"]]
        except Exception:
            sel = []
        wins = nd.steady_windows(u)
        t0, t1, act = wins[0]
        f, P, navg, clipped = nd.window_psd(gs, t0, t1)
        rpm = nd.esc_rpm(u, t0, t1)
        v = fc["version"]
        r = dict(
            url=URL.format(info["file"][:-4]), licence="CC-BY (PX4 Flight Review public logs)",
            hw=fc["hw"], version=f"{(v >> 24) & 255}.{(v >> 16) & 255}.{(v >> 8) & 255}" + ("" if v & 255 == 255 else " (dev)"),
            duration_s=round((u.last_timestamp - u.start_timestamp) / 1e6, 1),
            fs_nominal=gs.fs, fs_measured=round(gs.fs_measured, 2),
            clock_error_pct=round(100 * (gs.fs_measured / gs.fs - 1), 3),
            dropouts_ms=int(sum(d.duration for d in u.dropouts)), n_segments=len(gs.segments),
            gap_fraction_pct=round(100 * gs.gap_fraction, 3),
            fifo_device=gs.device_id, selected_gyro_devices=sorted(set(sel)),
            fifo_is_control_gyro=bool(sel) and all(s == gs.device_id for s in sel),
            board_rotation=gs.rotation, fifo_is_external=gs.is_external,
            fft_notch_enabled=fc["fft_notch"], fc_params_changed_in_flight=fc["changed_in_flight"],
            airborne=[tuple(round(x, 1) for x in a) for a in nd.airborne_intervals(u)],
            n_steady_windows=len(wins), window=(round(t0, 2), round(t1, 2)), window_activity_dps=round(act, 1),
            welch_averages=navg, clipped_samples=clipped,
            rpm_hz=None if rpm is None else [round(x, 1) for x in rpm["mean"] / 60],
            rpm_std_hz=None if rpm is None else [round(x, 1) for x in rpm["std"] / 60],
            rpm_rate_hz=None if rpm is None else round(rpm["rate_hz"], 1),
            tracking_notch_ok=rpm is not None and rpm["rate_hz"] >= MIN_RPM_LOG_HZ,
            peaks_roll=top_peaks(f, P[:, 0]), peaks_pitch=top_peaks(f, P[:, 1]),
            actual=settings_row(fc["actual"]), default=settings_row(fc["default"]),
            gains_roll=asdict(fc["gains"]), gains_pitch=asdict(nd.fc_settings(u, "PITCH")["gains"]),
            f_ctrl=fc["f_ctrl"], load_s=round(time.perf_counter() - t, 1))
        if rpm is not None:   # E13-lite: motor harmonics vs spectrum peaks
            fm = float(np.mean(rpm["mean"] / 60))
            r["peak_harmonic_match"] = [(p, round(p / fm, 2)) for p, _ in r["peaks_roll"]]
        res[key] = r
        ARR[f"{key}_f"], ARR[f"{key}_P"] = f, P
        info.update(ulog=u, gs=gs, fc=fc, window=(t0, t1), f=f, P=P, rpm=rpm)
        log(json.dumps({k: r[k] for k in ("hw", "version", "fs_measured", "clock_error_pct", "dropouts_ms",
                                          "fifo_is_control_gyro", "board_rotation", "window", "welch_averages",
                                          "rpm_hz", "rpm_std_hz")}))
        log("   peaks (Hz, prominence decades):", [(round(p, 1), round(q, 2)) for p, q in r["peaks_roll"]])
    OUT["data"] = res


# ----------------------------------------------------------------- stage 2
def stage_validation():
    log("\n=== Stage 2: real-data validation (E10 filtered gyro, D-term input; E11 torque)")
    info = LOGS["b27d09e3"]
    u, gs, fc = info["ulog"], info["gs"], info["fc"]
    t0, t1 = info["window"]
    s, motor_hz = fc["actual"], info["rpm"]["mean"] / 60
    f, P = info["f"], info["P"]
    assert OUT["data"]["b27d09e3"]["tracking_notch_ok"], "logged RPM rate too low for the tracking notch (P1 gate)"
    t = time.perf_counter()
    f_tr, P_tr, _ = nd.tracking_notch_psd(u, gs, t0, t1, s.dnf_bw, s.dnf_hmc, s.dnf_min)
    t_track = time.perf_counter() - t
    res = dict(bands=BANDS, tracking_psd_seconds_2axes=round(t_track, 2))
    for axis in (0, 1):
        g = nd.fc_settings(u, ["ROLL", "PITCH"][axis])["gains"]
        meas = vr.measured_slow_psds(u, t0, t1, axis=axis)
        variants = {
            "PX4 model, fixed notch at mean RPM": (f, P[:, axis], s, {}),
            "PX4 model, tracking notch (hybrid)": (f_tr, P_tr[:, axis], replace(s, dnf_esc=False), {}),
            "wrong: no RPM notch": (f, P[:, axis], replace(s, dnf_esc=False), {}),
            "wrong: D-term 2nd-order LPF": (f_tr, P_tr[:, axis], replace(s, dnf_esc=False), dict(dterm_kind="lpf2")),
            "wrong: D before gyro LPF": (f_tr, P_tr[:, axis], replace(s, dnf_esc=False), dict(d_after_lpf=False)),
            "wrong: gyro LPF 40 Hz": (f_tr, P_tr[:, axis], replace(s, dnf_esc=False, f_gyro=40.0), {}),
        }
        ax_res = {}
        for name, (ff, SS, ss, kw) in variants.items():
            pred = vr.predicted_slow_psds(ff, SS, ss, g, gs.fs, meas["pub_rate"], meas["f"], motor_hz, **kw)
            ax_res[name] = {k: [round(x, 2) for x in vr.band_errors_db(meas["f"], pred[k], meas[k], BANDS)]
                            for k in ("rate", "accel", "torque")}
            if axis == 0 and name.startswith("PX4"):
                tag = "track" if "tracking" in name else "fixed"
                for k in ("rate", "accel", "torque"):
                    ARR[f"E10_pred_{tag}_{k}"] = pred[k]
        if axis == 0:
            ARR["E10_f"] = meas["f"]
            for k in ("rate", "accel", "torque"):
                ARR[f"E10_meas_{k}"] = meas[k]
        # full time-domain replica of the PX4 chain (tracking notches), last sample per FIFO batch
        rep = vr.replicate_px4_filters(u, s, t0, t1, warmup=2.0, axes=(axis,))
        f_rep_rate = len(rep["t_sample"]) / (t1 - t0)
        _, S_rr = welch(rep["rate"][:, 0], fs=f_rep_rate, nperseg=512)
        _, S_ra = welch(rep["accel"][:, 0], fs=f_rep_rate, nperseg=512)
        ax_res["time-domain replica (PX4 chain, tracking)"] = {
            "rate": [round(x, 2) for x in vr.band_errors_db(meas["f"], S_rr, meas["rate"], BANDS)],
            "accel": [round(x, 2) for x in vr.band_errors_db(meas["f"], S_ra, meas["accel"], BANDS)],
            "torque": [None] * len(BANDS)}
        res["xy"[axis]] = ax_res
        for name, e in ax_res.items():
            if e["torque"][0] is None:
                log(f"   axis {'xy'[axis]} {name:38s} rate {e['rate']}  accel {e['accel']}")
                continue
            log(f"   axis {'xy'[axis]} {name:38s} rate {e['rate']}  accel max|e| {max(map(abs, e['accel'])):.1f}"
                f"  torque max|e| {max(map(abs, e['torque'])):.1f}")
    OUT["validation"] = res


# ----------------------------------------------------------------- stage 3
def stage_plant():
    log("\n=== Stage 3: plant identification and realism check")
    info = LOGS["b27d09e3"]
    u, gs = info["ulog"], info["gs"]
    res = {}
    # windows with roll / pitch doublets, picked by eye for this log (the pilot study needs an automatic rule)
    for name, axis, (t0, t1) in [("roll", 0, (24.0, 53.0)), ("pitch", 1, (8.5, 22.9))]:
        r = sysid.identify(sysid.prepare(u, gs, t0, t1, axis=axis))
        # instrument check (P2): in Stabilized mode the attitude setpoint is the strictly external signal
        r2 = sysid.identify(sysid.prepare(u, gs, t0, t1, axis=axis, instrument="att_sp"))
        res[name] = dict(window=(t0, t1), b=round(r["b"], 1), tau_m=round(r["tau_m"], 4), Td=round(r["Td"], 5),
                         n_freq=r["n_freq"], f_used=[round(x, 2) for x in r["f_used"]],
                         mag_err_db=round(r["mag_err_db"], 2), phase_err_deg=round(r["phase_err_deg"], 1),
                         att_sp_instrument=dict(b=round(r2["b"], 1), tau_m=round(r2["tau_m"], 4),
                                                Td=round(r2["Td"], 5), n_freq=r2["n_freq"]))
        ARR[f"sysid_{name}_f"], ARR[f"sysid_{name}_G"] = r["f"], r["G"]
        ARR[f"sysid_{name}_coh"], ARR[f"sysid_{name}_sel"] = r["coh"], r["sel"]
        log(f"   {name}: {res[name]}")
    # realism check: real gains + identified plant + real filters (tracking notch)
    fc, rates = info["fc"], Rates(fs=gs.fs, f_ctrl=info["fc"]["f_ctrl"])
    t0, t1 = info["window"]
    s_act = fc["actual"]
    f_tr, P_tr, _ = nd.tracking_notch_psd(u, gs, t0, t1, s_act.dnf_bw, s_act.dnf_hmc, s_act.dnf_min)
    real = {}
    for name, axis in (("roll", 0), ("pitch", 1)):
        p = res[name]
        plant = Plant(p["b"], p["tau_m"], p["Td"])
        gains = nd.fc_settings(u, name.upper())["gains"]
        case = LoopCase(plant, gains, rates, info["f"], info["P"][:, axis], motor_hz=info["rpm"]["mean"] / 60,
                        f_hi=F_HI, reference=fc["default"])
        S_act = case.psd_on_noise_grid(f_tr, P_tr[:, axis])
        rows = {"PX4 default (log)": metrics_row(case.ref_metrics),
                "pilot actual": metrics_row(case.evaluate(s_act, S_eff=S_act))}
        # sensitivity to plant uncertainty (8 corners, THESIS 7.1.1 step 4)
        worst = {"PM": 1e9, "GM": 1e9, "Ms": 0.0}
        for db in (0.8, 1.2):
            for dt in (0.8, 1.2):
                for dd in (0.0, 1.0 / rates.f_ctrl):
                    c2 = LoopCase(Plant(p["b"] * db, p["tau_m"] * dt, p["Td"] + dd), gains, rates, info["f"],
                                  info["P"][:, axis], motor_hz=info["rpm"]["mean"] / 60, f_hi=F_HI,
                                  reference=fc["default"])
                    m = c2.evaluate(s_act, S_eff=c2.psd_on_noise_grid(f_tr, P_tr[:, axis]))
                    worst = dict(PM=min(worst["PM"], m["PM"]), GM=min(worst["GM"], m["GM"]), Ms=max(worst["Ms"], m["Ms"]))
        rows["pilot actual, worst of 8 plant corners"] = {k: round(v, 2) for k, v in worst.items()}
        real[name] = rows
        for k, v in rows.items():
            log(f"   {name} {k:40s} {v}")
    res["realism"] = real
    OUT["plant"] = res
    info["tracking_actual"] = (f_tr, P_tr)


# ----------------------------------------------------------------- stage 4
def tracking_cache(info, axis):
    """Time-domain ESC-notch spectra, computed once per (bw, hmc, dnf_min) and log window.

    Returns tracking(bw, hmc, dnf_min) -> (f, psd) and a dict with the time spent."""
    u, gs, (t0, t1) = info["ulog"], info["gs"], info["window"]
    cache, timing = {}, {"seconds": 0.0, "n": 0}

    def tracking(bw, h, dnf_min):
        key = (float(bw), int(h), float(dnf_min))
        if key not in cache:
            t = time.perf_counter()
            f, P, _ = nd.tracking_notch_psd(u, gs, t0, t1, bw, h, dnf_min, axes=(axis,))
            cache[key] = (f, P[:, 0])
            timing["seconds"] += time.perf_counter() - t
            timing["n"] += 1
        return cache[key]
    return tracking, timing


def run_case(label, case, base, options, brute=True, fine_n=(120, 100), modes="AB"):
    """Modes A and B, Pareto front, brute-force check. Returns a result row."""
    row = dict(label=label, reference=metrics_row(case.ref_metrics), f_lo=round(case.f_lo, 1))
    t = time.perf_counter()
    grid = opt.grid_search(case, base, options)
    row["t_grid_s"] = round(time.perf_counter() - t, 2)
    row["n_settings_grid"] = int(sum(r["PM"].size for _, r in grid))
    for mode in modes:
        r = opt.optimize(case, base, options, mode=mode, grid=grid)
        if not r["feasible"]:
            row[f"mode_{mode}"] = dict(feasible=False)
            log(f"   {label} mode {mode}: no feasible setting")
            continue
        s = r["settings"]
        mrow = dict(feasible=True, option=r["option"].label, f_gyro=round(s.f_gyro, 2), f_dterm=round(s.f_dterm, 2),
                    grid_point=[round(x, 2) for x in r["grid_point"]], metrics=metrics_row(r["metrics"]),
                    refine_evals=r["n_refine_evals"], t_refine_s=round(r["t_total"] - r["t_grid"], 2))
        if brute:   # fine brute-force grid for the chosen notch option
            rg = opt.ranges(case.rates.fs)
            fine = opt.grid_search(case, base, [r["option"]], f_gyro=np.geomspace(*rg["f_gyro"], fine_n[0]),
                                   f_dterm=np.geomspace(*rg["f_dterm"], fine_n[1]))
            b = opt.best_on_grid(fine, mode, case.ref_metrics["PM"])
            key = "PM" if mode == "A" else "N_rel"
            if b is None:   # refined point exists but no fine-grid point is feasible
                mrow["brute_force"] = dict(value=None, refined_value=round(float(r["metrics"][key]), 4))
                row[f"mode_{mode}"] = mrow
                continue
            fine_val = -b[0] if mode == "A" else b[0]
            mrow["brute_force"] = dict(value=round(float(fine_val), 4), f_gyro=round(float(b[2]), 2),
                                       f_dterm=round(float(b[3]), 2), refined_value=round(float(r["metrics"][key]), 4))
        row[f"mode_{mode}"] = mrow
        log(f"   {label} mode {mode}: {mrow['option']}, f_g {mrow['f_gyro']} f_d {mrow['f_dterm']} -> "
            f"PM {mrow['metrics']['PM']:.1f} (ref {row['reference']['PM']:.1f}), tauF {mrow['metrics']['tau_F']*1e3:.2f} ms "
            f"(ref {row['reference']['tau_F']*1e3:.2f}), N_rel {mrow['metrics']['N_rel']:.3f}, Ms {mrow['metrics']['Ms']:.2f}"
            + (f"  | brute {mrow['brute_force']}" if brute else ""))
    row["pareto"] = opt.pareto_front(grid, np.round(np.geomspace(0.25, 2.0, 13), 3))
    return row, grid


def sample_synthetic_drones(n, rates, rng, max_tries=4000):
    """Tier 3a crossover design; keep drones with unfiltered PM 45-75 and safe defaults."""
    drones, tries = [], 0
    while len(drones) < n and tries < max_tries:
        tries += 1
        plant = Plant(1.0, float(np.exp(rng.uniform(np.log(0.03), np.log(0.10)))),
                      float(np.exp(rng.uniform(np.log(0.0005), np.log(0.005)))))
        fcx, r = float(np.exp(rng.uniform(np.log(3), np.log(15)))), float(rng.uniform(0.5, 3.0))
        g, m = design_gains(fcx, r, plant, rates)
        if not (m["stable"] and 45 <= m["PM"] <= 75):
            continue
        drones.append(dict(plant=plant, gains=g, f_cross=fcx, r=r, pm_unfiltered=float(m["PM"])))
    return drones, tries


def stage_optimizer():
    log("\n=== Stage 4: optimizer")
    rng = np.random.default_rng(SEED)
    res = dict(limits=opt.LIMITS)
    info = LOGS["b27d09e3"]
    gs, fc = info["gs"], info["fc"]
    rates = Rates(fs=gs.fs, f_ctrl=fc["f_ctrl"])
    motor_hz = info["rpm"]["mean"] / 60
    bws, hmcs = (5, 7.5, 10, 15, 20, 30), (1, 2, 3)
    tracking, timing = tracking_cache(info, axis=0)

    # (a) real-gain drone, reference = what the pilot flew
    p = OUT["plant"]["roll"]
    f_tr, P_tr = info["tracking_actual"]
    case = LoopCase(Plant(p["b"], p["tau_m"], p["Td"]), fc["gains"], rates, info["f"], info["P"][:, 0],
                    motor_hz=motor_hz, f_hi=F_HI, reference=fc["actual"], reference_S_eff=(f_tr, P_tr[:, 0]))
    options = opt.scenario_b_options(case, fc["actual"], bws, hmcs, tracking)
    res["tracking_cache_seconds_18_options"] = round(timing["seconds"], 1)
    log(f"   tracking-notch spectra for {timing['n']} options: {timing['seconds']:.1f} s")
    row, grid = run_case("real-gain roll, ref = pilot", case, fc["actual"], options)
    default_m = case.evaluate(fc["default"])
    row["px4_default"] = metrics_row(default_m)
    res["real_gain"] = row
    _store_pareto("real", grid, case)

    # (b) synthetic drones x real noise (reference = PX4 v1.16 defaults, D-09)
    drones, tries = sample_synthetic_drones(8, rates, rng)
    res["synthetic_sampling"] = dict(accepted=len(drones), tries=tries)
    rows, kept, kept_idx = [], 0, []
    for i, d in enumerate(drones):
        case = LoopCase(d["plant"], d["gains"], rates, info["f"], info["P"][:, 0], motor_hz=motor_hz, f_hi=F_HI,
                        reference=PX4_V116_MC_DEFAULT)
        ref = case.ref_metrics
        safe_default = ref["stable"] and ref["PM"] >= opt.LIMITS["PM_min"] and ref["Ms"] <= opt.LIMITS["Ms_max"] \
            and ref["GM"] >= opt.LIMITS["GM_min"]
        desc = dict(tau_m=round(d["plant"].tau_m, 4), Td=round(d["plant"].Td, 5), f_cross=round(d["f_cross"], 2),
                    r=round(d["r"], 2), pm_unfiltered=round(d["pm_unfiltered"], 1), default_safe=bool(safe_default))
        if not safe_default:
            rows.append(dict(label=f"syn{i}", drone=desc, skipped="unsafe at PX4 defaults (THESIS 7.1.1 Tier 3 step 3)"))
            log(f"   syn{i} {desc} -> skipped (unsafe at defaults: PM {ref['PM']:.1f}, Ms {ref['Ms']:.2f})")
            continue
        kept += 1
        kept_idx.append(i)
        for scen in ("B", "A"):
            if scen == "B":
                options = opt.scenario_b_options(case, PX4_V116_MC_DEFAULT, bws, hmcs, tracking)
            else:
                peaks = [pk for pk, _ in OUT["data"]["b27d09e3"]["peaks_roll"] if case.f_lo <= pk <= F_HI]
                options = opt.scenario_a_options(peaks)
            row, grid = run_case(f"syn{i} scen {scen}", case, PX4_V116_MC_DEFAULT, options, brute=(kept <= 4))
            row["drone"], row["scenario"] = desc, scen
            rows.append(row)
            if kept == 1:
                _store_pareto(f"syn_{scen}", grid, case)
    res["synthetic"] = rows

    # (c) second log, Scenario A (static notches): first three drones that pass the Tier 3a
    #     acceptance rules again with this log's rates (unfiltered PM 45-75, safe at the defaults)
    info2 = LOGS["a2571040"]
    rates2 = Rates(fs=info2["gs"].fs, f_ctrl=info2["fc"]["f_ctrl"])
    rows2 = []
    for i, d in enumerate(drones):
        if len(rows2) == 3:
            break
        g, m0 = design_gains(d["f_cross"], d["r"], d["plant"], rates2)
        if not (m0["stable"] and 45 <= m0["PM"] <= 75):
            continue
        case = LoopCase(d["plant"], g, rates2, info2["f"], info2["P"][:, 0], f_hi=F_HI, reference=PX4_V116_MC_DEFAULT)
        ref = case.ref_metrics
        if not (ref["stable"] and ref["PM"] >= opt.LIMITS["PM_min"] and ref["Ms"] <= opt.LIMITS["Ms_max"]
                and ref["GM"] >= opt.LIMITS["GM_min"]):
            continue
        peaks = [pk for pk, _ in OUT["data"]["a2571040"]["peaks_roll"] if case.f_lo <= pk <= F_HI]
        row, _ = run_case(f"log2 syn{i} scen A", case, PX4_V116_MC_DEFAULT, opt.scenario_a_options(peaks), brute=False)
        rows2.append(row)
    res["second_log"] = rows2

    # (d) speed: reduced grids vs full grids, on the first drone that was kept in (b)
    d0 = drones[kept_idx[0]]
    speed = []
    for label, kw, ng, nd_ in [("full (1500 stab pts, 0.5 Hz noise grid, 40x30)", dict(), 40, 30),
                               ("fast (500 stab pts, 1 Hz noise grid, 24x18)", dict(n_stab=500, noise_df=1.0), 24, 18)]:
        case = LoopCase(d0["plant"], d0["gains"], rates, info["f"], info["P"][:, 0], motor_hz=motor_hz, f_hi=F_HI,
                        reference=PX4_V116_MC_DEFAULT, **kw)
        options = opt.scenario_b_options(case, PX4_V116_MC_DEFAULT, bws, hmcs, tracking)
        t = time.perf_counter()
        r = opt.optimize(case, PX4_V116_MC_DEFAULT, options, mode="A", n_g=ng, n_d=nd_)
        dt = time.perf_counter() - t
        speed.append(dict(label=label, seconds=round(dt, 2), PM=round(float(r["metrics"]["PM"]), 3),
                          f_gyro=round(r["settings"].f_gyro, 2), f_dterm=round(r["settings"].f_dterm, 2),
                          option=r["option"].label))
        log(f"   speed {label}: {dt:.2f} s per case (mode A), PM* {r['metrics']['PM']:.2f}")
    res["speed"] = speed
    OUT["optimizer"] = res


def _store_pareto(tag, grid, case):
    pm, n = [], []
    for _, r in grid:
        ok = opt.feasible(r, "C", n_max=np.inf)
        pm.append(r["PM"][ok])
        n.append(r["N_rel"][ok])
    ARR[f"pareto_{tag}_PM"], ARR[f"pareto_{tag}_N"] = np.concatenate(pm), np.concatenate(n)
    ARR[f"pareto_{tag}_ref"] = np.array([case.ref_metrics["PM"], 1.0])


def main():
    """Full run. With --validation-only, stages 1-2 are recomputed and merged into
    the existing results.json / arrays.npz (stages 3-4 are kept)."""
    only_val = "--validation-only" in sys.argv
    t = time.perf_counter()
    stage_data()
    stage_validation()
    if only_val:
        old = json.load(open(os.path.join(HERE, "results.json")))
        old.update(data=OUT["data"], validation=OUT["validation"])
        OUT.clear()
        OUT.update(old)
        arr = dict(np.load(os.path.join(HERE, "arrays.npz")))
        arr.update(ARR)
        ARR.clear()
        ARR.update(arr)
        OUT["meta"]["validation_rerun_seconds"] = round(time.perf_counter() - t, 1)
    else:
        stage_plant()
        stage_optimizer()
        OUT["meta"] = dict(seed=SEED, run_seconds=round(time.perf_counter() - t, 1), date=time.strftime("%Y-%m-%d"),
                           numpy=np.__version__)
    with open(os.path.join(HERE, "results.json"), "w") as fh:
        json.dump(OUT, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    np.savez_compressed(os.path.join(HERE, "arrays.npz"), **ARR)
    log(f"\nDone in {round(time.perf_counter() - t, 1)} s")


if __name__ == "__main__":
    main()
