"""Evidence for decisions A2 (PM floor), A3 (Ms_max) and A4 (uncertainty set): who passes which limits.

    .venv/Scripts/python experiments/phase4/robust_limits.py

Two populations, FIT HALF only:
1. Real-gain drones: every accepted identification of plant_variation.json (plant_variation.accept) with the log's
   own gains, the pilot's actual filters (theta_actual, D-15), the measured controller rate (D-28), the control
   IMU's raw rate as filter rate (sensor_selection + vehicle_imu_status; else the FIFO rate) and, if the ESC-RPM
   notch is on, the mean motor frequencies of the airborne time. Static notches included; the FFT notch is not
   modelled (flagged). The identified Td contains the FIFO chip's on-chip filter; it is corrected so that the loop
   contains the CONTROL chip's filter once (chip_td_correction, lower delay bounds of Task 1); the uncorrected
   result is kept as a sensitivity. Rows whose loop gains or gyro filters changed during the flight are flagged and
   left out of the pass rates. The b27d09e3 roll and pitch plants of the feasibility test are added as reference rows.
2. Synthetic Tier 3a drones (THESIS 7.1.1; D-28 priors): f_c 4-15 Hz, tau_m 20-60 ms, Td 0.5-5 ms (log-uniform),
   r in [0.5, 3], (FIFO rate, controller rate) pairs of same-IMU full-rate fit logs; kept if the unfiltered PM is
   45-75 deg; evaluated at the PX4 v1.16 default filters (their reference and safety-check fallback, D-09).
Grid: PM floor {25, 30, 35} x Ms_max {1.8, 2.0, 2.2} x width {10, 15, 20, 30 %} on b and tau_m x delay step
{0.5, 1, 2} controller samples; GM >= 6 dB always (THESIS 7.3). Worst-case limits either the same as the nominal
ones, or separate: Ms_wc in {2.5, 3.0} with the PM and GM that Ms_wc guarantees (PM >= 2 asin(1 / (2 Ms)),
GM >= Ms / (Ms - 1); Astrom and Murray, Feedback Systems, Sec. 12.2). ICM chip delay (D-25): synthetic ICM-4268x
drones also with the chip delay of either rule, as a nominal shift or inside the delay step.
Writes robust_limits.json next to this file.
"""
import json
import math
import os
import re
import sys
import time

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import noise_data as nd                     # noqa: E402
import plant_variation as pv                # noqa: E402
import priors as pr                         # noqa: E402
import robust as rb                         # noqa: E402
from loop_model import (FilterSettings, Gains, Plant, Rates, PX4_V116_MC_DEFAULT, design_gains)   # noqa: E402

PM_FLOORS = (25.0, 30.0, 35.0)
MS_MAX = (1.8, 2.0, 2.2)
GM_MIN = 6.0
WIDTHS = (0.10, 0.15, 0.20, 0.30)
DELAYS = (0.5, 1.0, 2.0)                    # controller samples
WC_MS = (2.5, 3.0)
N_SYNTH = 1000
SEED = 20261018
F_C, TAU, TD = (4.0, 15.0), (0.020, 0.060), (0.0005, 0.005)     # D-28
R_RANGE = (0.5, 3.0)                        # THESIS 7.1.1 Tier 3a
KEY_SETS = ("PM30_Ms2_same", "PM30_Ms2_wcMs3", "PM25_Ms2.2_wcMs3")   # shown by crossover bin
KEY_CELLS = ("w0.1_d0.5", "w0.15_d1", "w0.2_d1")
CHIP_CHIPS = ("ICM42688P", "ICM42686P")  # D-25: the ICM-4268x chips; delays from priors_from_logs.json chip_delay


def chip_delays(pri):
    """{(chip|rule): (low, high) delay [s]} of the ICM-4268x chips under both rules (Task 1, D-25)."""
    out = {}
    for chip in CHIP_CHIPS:
        for rule in ("bic", "rms"):
            lo, hi = pri["chip_delay"][chip][rule]["delay_ms"]
            out[f"{chip}|{rule}"] = (lo * 1e-3, hi * 1e-3)
    return out


def wc_limits(ms):
    """Worst-case limits implied by Ms: PM >= 2 asin(1 / (2 Ms)), GM >= 20 log10(Ms / (Ms - 1))."""
    return dict(PM=math.degrees(2 * math.asin(1 / (2 * ms))), Ms=ms, GM=20 * math.log10(ms / (ms - 1)))


def limit_sets():
    out = {}
    for pm in PM_FLOORS:
        for ms in MS_MAX:
            out[f"PM{pm:g}_Ms{ms:g}_same"] = dict(PM_min=pm, Ms_max=ms, GM_min=GM_MIN, wc=None)
            for w in WC_MS:
                out[f"PM{pm:g}_Ms{ms:g}_wcMs{w:g}"] = dict(PM_min=pm, Ms_max=ms, GM_min=GM_MIN, wc=wc_limits(w))
    return out


def evaluate(plant, gains, rates, s, motor_hz, extra_nominal=0.0, extra_corner=0.0):
    """robust_metrics for every (width, delay step); extra delays [s] for the chip-delay variants."""
    p = Plant(plant.b, plant.tau_m, plant.Td + extra_nominal)
    res = {}
    for w in WIDTHS:
        for d in DELAYS:
            res[f"w{w:g}_d{d:g}"] = rb.robust_metrics(p, gains, rates, s, motor_hz, w, w, d / rates.f_ctrl + extra_corner)
    return res


def pass_rates(evals, sets):
    """Share of cases passing each (limit set, width, delay); evals: list of evaluate() results."""
    out = {}
    for name, lim in sets.items():
        for key in evals[0]:
            out[f"{name}|{key}"] = round(float(np.mean([rb.passes(e[key], **lim) for e in evals])), 4)
    return out


def _nominal_ok(e, pm, ms):
    n = next(iter(e.values()))["nominal"]            # the nominal plant is the same for every width and delay
    return bool(n["stable"] and n["PM"] >= pm and n["Ms"] <= ms and n["GM"] >= GM_MIN)


def nominal_pass_rates(evals):
    """Share of cases whose NOMINAL plant meets PM >= floor, Ms <= Ms_max, GM >= 6 dB (no uncertainty set)."""
    return {f"PM{pm:g}_Ms{ms:g}": round(float(np.mean([_nominal_ok(e, pm, ms) for e in evals])), 4)
            for pm in PM_FLOORS for ms in MS_MAX}


def conditional_rates(evals, sets):
    """Among the cases that pass nominally, the share that also passes the robust check (per set, width, delay)."""
    out = {}
    for name, lim in sets.items():
        ok = [e for e in evals if _nominal_ok(e, lim["PM_min"], lim["Ms_max"])]
        for key in evals[0]:
            out[f"{name}|{key}"] = round(float(np.mean([rb.passes(e[key], **lim) for e in ok])), 4) if ok else None
    return out


def fc_bins(synth, evals, sets, keys, edges=(4.0, 6.5, 10.0, 15.0)):
    """Nominal and robust pass rates of synthetic drones by crossover-frequency bin."""
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = [i for i, d in enumerate(synth) if lo <= d["f_c"] < hi or (hi == edges[-1] and d["f_c"] == hi)]
        row = dict(n=len(idx))
        for name, lim in sets.items():
            row[f"nominal|PM{lim['PM_min']:g}_Ms{lim['Ms_max']:g}"] = round(float(np.mean(
                [_nominal_ok(evals[i], lim["PM_min"], lim["Ms_max"]) for i in idx])), 4) if idx else None
            for key in keys:
                row[f"{name}|{key}"] = round(float(np.mean([rb.passes(evals[i][key], **lim) for i in idx])), 4) if idx else None
        out[f"{lo:g}-{hi:g} Hz"] = row
    return out


def _settings(d):
    return FilterSettings(f_gyro=d["f_gyro"], f_dterm=d["f_dterm"], notches=tuple(tuple(n) for n in d["notches"]),
                          dnf_esc=d["dnf_esc"], dnf_bw=d["dnf_bw"], dnf_hmc=int(d["dnf_hmc"]), dnf_min=d["dnf_min"])


LOOP_PARAMS = re.compile(r"^(MC_(ROLL|PITCH)RATE_[PIDK]|IMU_GYRO_CUTOFF|IMU_DGYRO_CUTOFF|IMU_GYRO_NF[01]_(FRQ|BW)"
                         r"|IMU_GYRO_DNF_(EN|BW|HMC|MIN)|IMU_GYRO_RATEMAX)$")


def loop_params_changed(changed):
    """Rate-loop gains and gyro-filter parameters among the parameters changed during the flight (sorted)."""
    return sorted(p for p in changed if LOOP_PARAMS.match(p))


def control_rate(u):
    """(raw FIFO rate of the control IMU, source): sensor_selection.gyro_device_id matched to vehicle_imu_status."""
    sel = None
    try:
        sel = int(u.get_dataset("sensor_selection").data["gyro_device_id"][-1])
    except Exception:
        pass
    statuses = [(int(d.data["gyro_device_id"][-1]), float(np.median(d.data["gyro_raw_rate_hz"])))
                for d in u.data_list if d.name == "vehicle_imu_status" and "gyro_raw_rate_hz" in d.data]
    fs = rb.control_imu_raw_rate(sel, statuses)
    return (fs, "control IMU") if fs is not None else (None, "control IMU not matched")


def chip_td_correction(fifo_chip, control_chip, chip_lower):
    """Delay [s] to add to an identified Td so that the loop contains the CONTROL chip's on-chip filter once.

    The identified Td (raw FIFO gyro as the output) contains the FIFO chip's filter. chip_lower: lower delay bound
    [s] of every chip with a fitted low-pass term (Task 1, D-25); other chips count as 0 (no measured roll-off).
    None when the control chip is unknown (no correction possible); two equal chips give 0 by the subtraction."""
    if control_chip is None:
        return None
    return chip_lower.get(control_chip, 0.0) - chip_lower.get(fifo_chip, 0.0)


def case_from_log(u, rec, r, chip_lower):
    """One real-gain case from a ULog-like object ``u`` (needs sensor_selection, vehicle_imu_status and, if the
    ESC-RPM notch is on, vehicle_land_detected and esc_status), the fit_logs record and an accepted identification."""
    fc = rec["fc"]
    fs, fs_src = control_rate(u)
    if fs is None:
        fs, fs_src = float(rec["stats"]["fs_nominal"]), "FIFO nominal (control IMU not matched)"
    s = _settings(fc["actual"])
    motor_hz, flags = None, []
    if s.dnf_esc:
        air = [(a + pv.T_EDGE, b - pv.T_EDGE) for a, b in nd.airborne_intervals(u) if b - a > 2 * pv.T_EDGE]
        rpm = nd.esc_rpm(u, air[0][0], air[-1][1]) if air else None
        if rpm is None:
            flags.append("ESC notch on, no valid RPM: notch left out")
        else:
            motor_hz = np.asarray(rpm["mean"], float) / 60
    if fc.get("fft_notch"):
        flags.append("FFT notch on: not modelled")
    f = r["full"]
    corr = chip_td_correction(r["fifo_chip"], r["control_chip"], chip_lower)
    td = f["Td"]
    if corr is None:
        flags.append("control IMU unknown: Td keeps the FIFO chip's filter")
    elif corr != 0.0:
        td = f["Td"] + corr
        flags.append(f"Td corrected {1e3 * corr:+.2f} ms (FIFO chip {r['fifo_chip']} -> control chip {r['control_chip']})")
        if td < 0:
            td = 0.0
            flags.append("corrected Td clamped at 0")
    changed = loop_params_changed(fc.get("changed_in_flight") or [])
    if changed:
        flags.append("loop parameters changed in flight (" + ", ".join(changed) + "): excluded from the pass rates")
    g = fc["gains_roll" if r["axis"] == 0 else "gains_pitch"]
    return dict(log_id=r["log_id"], axis=r["axis"], cls=r["cls"], seconds=f["seconds"], hw=fc.get("hw"),
                fifo_chip=r["fifo_chip"], control_chip=r["control_chip"],
                plant=dict(b=f["b"], tau_m=f["tau_m"], Td=float(td)), td_identified=f["Td"], fit_band=f.get("f_used"),
                gains=dict(Kp=g["Kp"], Ki=g["Ki"], Kd=g["Kd"]), f_ctrl=r["f_ctrl"], fs=float(fs), fs_source=fs_src,
                settings=fc["actual"], motor_hz=None if motor_hz is None else motor_hz.tolist(), flags=flags,
                exclude=bool(changed))


def chip_lower_bounds(pri, rule="rms"):
    """Lower delay bound [s] per chip with a fitted low-pass term (priors_from_logs.json chip_delay)."""
    return {chip: v[rule]["delay_ms"][0] * 1e-3 for chip, v in pri["chip_delay"].items()}


def real_cases(pvj, logs, chip_lower):
    by_log = {r["log_id"]: r for r in logs}
    best = {}
    for r in pvj["rows"]:
        if "full" not in r:
            continue
        ok, _ = pv.accept(r["full"], r["cls"])
        if not ok:
            continue
        key = (r["log_id"], r["axis"])
        if key not in best or r["full"]["seconds"] > best[key]["full"]["seconds"]:
            best[key] = r
    cases = []
    for (log_id, axis), r in sorted(best.items()):
        u = nd.load(pr.fit_path(ROOT, "data", "raw_logs", log_id + ".ulg"), nd.TOPICS + ["vehicle_imu_status"])
        cases.append(case_from_log(u, by_log[log_id], r, chip_lower))
    return cases


def feasibility_cases():
    """b27d09e3 as identified in the feasibility test (hand-picked windows), for continuity with REPORT 5."""
    R = json.load(open(os.path.join(ROOT, "experiments", "feasibility", "results.json")))
    d = R["data"]["b27d09e3"]
    out = []
    for axis, name in ((0, "roll"), (1, "pitch")):
        p = R["plant"][name]
        out.append(dict(log_id="b27d09e3 (feasibility plant)", axis=axis, cls="att_sp (hand-picked windows)",
                        plant=dict(b=p["b"], tau_m=p["tau_m"], Td=p["Td"]), gains=d[f"gains_{name}"],
                        f_ctrl=d["f_ctrl"], fs=d["fs_nominal"], fs_source="FIFO nominal (as in the feasibility test)",
                        settings=d["actual"], motor_hz=[x for x in d["rpm_hz"]], flags=[]))
    return out


def eval_synth(d, extra_nominal=0.0, extra_corner=0.0):
    """evaluate() of one synthetic drone at the PX4 v1.16 defaults."""
    return evaluate(Plant(1.0, d["tau_m"], d["Td"]), Gains(**d["gains"]), Rates(d["fs"], d["f_ctrl"]),
                    PX4_V116_MC_DEFAULT, None, extra_nominal=extra_nominal, extra_corner=extra_corner)


def _par(jobs):
    return Parallel(n_jobs=6)(jobs)


def run_case(c):
    s = _settings(c["settings"])
    g = Gains(**c["gains"])
    rates = Rates(float(c["fs"]), float(c["f_ctrl"]))
    mh = None if c["motor_hz"] is None else np.asarray(c["motor_hz"], float)
    p = Plant(**c["plant"])
    return evaluate(p, g, rates, s, mh)


def synthetic_drones(rate_rows, n, seed, tau=TAU, td=TD, f_c=F_C, max_tries=None, pm_window=(45.0, 75.0), i_ratio=0.1):
    pairs = [(float(r["fs_nominal"]), float(r["pub_rate"])) for r in rate_rows
             if r["fifo_is_control"] and r["full_rate"] and r["pub_rate"]]
    rng = np.random.default_rng(seed)
    out, tries = [], 0
    max_tries = 200 * n if max_tries is None else max_tries
    while len(out) < n:
        tries += 1
        if tries > max_tries:
            raise RuntimeError(f"only {len(out)} of {n} drones met the unfiltered PM window in {max_tries} tries")
        fc = math.exp(rng.uniform(*np.log(f_c)))
        tau_m = math.exp(rng.uniform(*np.log(tau)))
        t_d = math.exp(rng.uniform(*np.log(td)))
        r = rng.uniform(*R_RANGE)
        fs, fctrl = pairs[int(rng.integers(len(pairs)))]
        plant, rates = Plant(1.0, tau_m, t_d), Rates(fs, fctrl)
        g, m = design_gains(fc, r, plant, rates, i_ratio=i_ratio)
        pm = float(m["PM"])
        if not (pm_window[0] <= pm <= pm_window[1]):
            continue
        out.append(dict(f_c=fc, tau_m=tau_m, Td=t_d, r=r, fs=fs, f_ctrl=fctrl, pm_unfiltered=pm,
                        gains=dict(Kp=g.Kp, Ki=g.Ki, Kd=g.Kd)))
    return out, tries, len(pairs)


def summarize_real(cases, evals):
    rows = []
    for c, e in zip(cases, evals):
        row = dict(log=c["log_id"][:8] if len(c["log_id"]) == 36 else c["log_id"], axis="roll" if c["axis"] == 0 else "pitch",
                   cls=c["cls"], fifo_chip=c.get("fifo_chip"), control_chip=c.get("control_chip"), b=round(c["plant"]["b"], 1),
                   tau_ms=round(1e3 * c["plant"]["tau_m"], 1), Td_ms=round(1e3 * c["plant"]["Td"], 2),
                   Td_identified_ms=None if c.get("td_identified") is None else round(1e3 * c["td_identified"], 2),
                   fit_band=c.get("fit_band"), f_ctrl=c["f_ctrl"], fs=round(float(c["fs"]), 1),
                   fs_source=c.get("fs_source"), exclude=bool(c.get("exclude", False)), flags=c["flags"],
                   nominal={k: round(v, 3) if isinstance(v, float) else v for k, v in e["w0.2_d1"]["nominal"].items()})
        row["worst"] = {key: {k: round(v, 3) if isinstance(v, float) else v for k, v in e[key]["worst"].items()}
                        for key in e}
        rows.append(row)
    return rows


def main():
    t0 = time.perf_counter()
    pvj = json.load(open(pr.fit_path(HERE, "plant_variation.json")))
    logs = json.load(open(pr.fit_path(ROOT, "data", "spectra", "fit_logs.json")))
    pri = json.load(open(pr.fit_path(HERE, "priors_from_logs.json")))
    sets = limit_sets()
    chip_lower = chip_lower_bounds(pri)
    real = real_cases(pvj, logs, chip_lower)
    feas = feasibility_cases()
    ev_real = [run_case(c) for c in real]
    inc = [i for i, c in enumerate(real) if not c["exclude"]]
    ev_inc = [ev_real[i] for i in inc]
    ev_unc = [run_case(dict(real[i], plant=dict(real[i]["plant"], Td=real[i]["td_identified"]))) for i in inc]
    ev_feas = [run_case(c) for c in feas]
    synth, tries, n_pairs = synthetic_drones(pri["rate_rows"], N_SYNTH, SEED)
    ev_syn = _par(delayed(eval_synth)(d) for d in synth)
    chip_delay = chip_delays(pri)
    chip = {}
    for key, (lo, hi) in chip_delay.items():
        ev_shift = _par(delayed(eval_synth)(d, lo, hi - lo) for d in synth)
        ev_corner = _par(delayed(eval_synth)(d, 0.0, hi) for d in synth)
        chip[key] = dict(delay_s=[lo, hi], nominal_shift=pass_rates(ev_shift, sets), in_delay_step=pass_rates(ev_corner, sets))
    out = dict(
        settings=dict(PM_FLOORS=PM_FLOORS, MS_MAX=MS_MAX, GM_MIN=GM_MIN, WIDTHS=WIDTHS, DELAYS=DELAYS, WC_MS=WC_MS,
                      wc_limits={str(w): wc_limits(w) for w in WC_MS}, N_SYNTH=N_SYNTH, SEED=SEED, F_C=F_C, TAU=TAU,
                      TD=TD, R_RANGE=R_RANGE, chip_delay=chip_delay),
        real=dict(n_cases=len(real), n_included=len(inc), n_vehicles=len({c["log_id"] for c in real}),
                  n_vehicles_included=len({real[i]["log_id"] for i in inc}), chip_lower_s=chip_lower,
                  cases=summarize_real(real, ev_real),
                  pass_rates=pass_rates(ev_inc, sets) if ev_inc else None,
                  nominal_pass_rates=nominal_pass_rates(ev_inc) if ev_inc else None,
                  conditional_rates=conditional_rates(ev_inc, sets) if ev_inc else None,
                  uncorrected_td=dict(pass_rates=pass_rates(ev_unc, sets), nominal_pass_rates=nominal_pass_rates(ev_unc),
                                      conditional_rates=conditional_rates(ev_unc, sets)) if ev_unc else None),
        feasibility=dict(cases=summarize_real(feas, ev_feas), pass_rates=pass_rates(ev_feas, sets)),
        synthetic=dict(n=len(synth), tries=tries, n_rate_pairs=n_pairs, pass_rates=pass_rates(ev_syn, sets),
                       nominal_pass_rates=nominal_pass_rates(ev_syn), conditional_rates=conditional_rates(ev_syn, sets),
                       fc_bins=fc_bins(synth, ev_syn, {k: sets[k] for k in KEY_SETS}, KEY_CELLS),
                       nominal_pm_default=dict(zip(("p10", "p50", "p90"), np.round(np.quantile(
                           [e["w0.2_d1"]["nominal"]["PM"] for e in ev_syn], [0.1, 0.5, 0.9]), 2).tolist())),
                       chip_delay=chip),
        seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(HERE, "robust_limits.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"real cases {len(real)} ({out['real']['n_vehicles']} vehicles; {len(inc)} in the pass rates); "
          f"synthetic {len(synth)} of {tries} tries; "
          f"{out['seconds']} s")


if __name__ == "__main__":
    main()
