"""Options for findings F1-F4 and a first estimate for A5 (author's request 2026-10-18). FIT HALF only.

    .venv/Scripts/python experiments/phase4/f_options.py        (after a4_followup.py)

F1 plant priors (tau_m, Td) of synthetic drones: D-28, the author's options (tau_m 15-60 ms; Td 0.3-8 ms; both;
   identified plants for half of the drones) and the evidence-pack proposal (tau_m 10-60 ms, Td 0.1-5 ms). Impact:
   how many of the 28 identified real plants each prior covers, and how many synthetic drones pass at their PX4
   defaults (Tier 3 step 3, nominal PM >= 30, Ms <= 2, GM >= 6 dB) and robustly (+-20 % / +1 sample, Ms_wc 3).
   Identified Td are compared NET of the chip filter (net_td): synthetic drones get the chip delay separately (D-33).
F2 Tier 3a gain design: PM window 45-75 (THESIS), 45-95 (widened), 60-95 (real p10-p90), integral corner 0.1 (THESIS),
   0.03 (real median) or log-uniform 0.01-0.08 (real range), and combinations.
F3 the crossover distribution of the drones each design keeps at the PX4 defaults, next to the real drones'.
F4 real flown settings with 5 f_c < 20 Hz: the f_lo rules on the 23 nominally passing settings (share of the noise
   power N^2 at the pilot's setting that comes from below 20 Hz, where D-23 Q5 does not define the noise metric).
A5 Scenario B (ESC RPM) counts of the fit half; the test half is not opened.
Synthetic drones are evaluated without the ICM chip delay (as in robust_limits); same seed and rate pairs everywhere.
Writes f_options.json next to this file.
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import a4_followup as af                    # noqa: E402
import priors as pr                         # noqa: E402
import robust as rb                         # noqa: E402
import robust_limits as rl                  # noqa: E402
from evidence_tables import plant_lag_deg, prior_lag_range   # noqa: E402
from loop_model import (Gains, Plant, Rates, PX4_V116_MC_DEFAULT, derivative_response, design_gains,   # noqa: E402
                        gyro_stack, integrator_response, plant_response, pt1_response, zoh_response)

FC_BINS = (4.0, 5.0, 6.5, 8.0, 10.0, 12.5, 15.0)
F_SPLIT, F_HI, NOISE_DF = 20.0, 780.0, 0.5          # D-23 Q5 band 20-780 Hz; LoopCase noise grid step
PRIORS = {
    "P0": dict(label="D-28: tau_m 20-60 ms, Td 0.5-5 ms", tau=(0.020, 0.060), td=(0.0005, 0.005)),
    "P1": dict(label="tau_m 15-60 ms (Td 0.5-5 ms)", tau=(0.015, 0.060), td=(0.0005, 0.005)),
    "P2": dict(label="Td 0.3-8 ms (tau_m 20-60 ms)", tau=(0.020, 0.060), td=(0.0003, 0.008)),
    "P3": dict(label="tau_m 15-60 ms and Td 0.3-8 ms", tau=(0.015, 0.060), td=(0.0003, 0.008)),
    "P4": dict(label="tau_m 10-60 ms, Td 0.1-5 ms (evidence-pack proposal)", tau=(0.010, 0.060), td=(0.0001, 0.005)),
    "P5": dict(label="D-28 for half the drones, an identified plant for the other half", tau=(0.020, 0.060),
               td=(0.0005, 0.005), identified_share=0.5),
}
DESIGNS = {
    "D0": dict(label="THESIS: PM window 45-75 deg, integral corner 0.1", pm_window=(45.0, 75.0), i_ratio=0.1),
    "D1": dict(label="PM window 45-95 deg (widened)", pm_window=(45.0, 95.0), i_ratio=0.1),
    "D2": dict(label="PM window 60-95 deg (real p10-p90)", pm_window=(60.0, 95.0), i_ratio=0.1),
    "D3": dict(label="integral corner 0.03 (real median)", pm_window=(45.0, 75.0), i_ratio=0.03),
    "D4": dict(label="integral corner log-uniform 0.01-0.08 (real range)", pm_window=(45.0, 75.0), i_range=(0.01, 0.08)),
    "D5": dict(label="PM window 60-95 deg and integral corner 0.03", pm_window=(60.0, 95.0), i_ratio=0.03),
}
COMBOS = {"C1": ("P4", "D5"), "C2": ("P3", "D5")}


def synthetic_population(rate_rows, n, seed, tau=rl.TAU, td=rl.TD, pm_window=(45.0, 75.0), i_ratio=0.1, i_range=None,
                         identified=None, identified_share=0.0, f_c=rl.F_C, max_tries=None):
    """robust_limits.synthetic_drones (the same draws when identified_share = 0 and i_range is None), plus: a share
    of drones taking (tau_m, Td) of an identified plant, and an integral corner drawn log-uniformly from i_range."""
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
        r = rng.uniform(*rl.R_RANGE)
        fs, fctrl = pairs[int(rng.integers(len(pairs)))]
        src = "prior"
        if identified_share > 0 and rng.uniform() < identified_share:
            tau_m, t_d = identified[int(rng.integers(len(identified)))]
            src = "identified"
        ir = i_ratio if i_range is None else math.exp(rng.uniform(*np.log(i_range)))
        g, m = design_gains(fc, r, Plant(1.0, tau_m, t_d), Rates(fs, fctrl), i_ratio=ir)
        pm = float(m["PM"])
        if not (pm_window[0] <= pm <= pm_window[1]):
            continue
        out.append(dict(f_c=fc, tau_m=tau_m, Td=t_d, r=r, fs=fs, f_ctrl=fctrl, pm_unfiltered=pm,
                        gains=dict(Kp=g.Kp, Ki=g.Ki, Kd=g.Kd), i=ir, source=src))
    return out, tries


def histogram(values, edges=FC_BINS):
    """Counts per bin [lo, hi) (the last bin closed), with under- and overflow bins."""
    out = {f"< {edges[0]:g}": 0}
    out.update({f"{lo:g}-{hi:g}": 0 for lo, hi in zip(edges[:-1], edges[1:])})
    out[f"> {edges[-1]:g}"] = 0
    for v in values:
        if v < edges[0]:
            out[f"< {edges[0]:g}"] += 1
        elif v > edges[-1]:
            out[f"> {edges[-1]:g}"] += 1
        else:
            k = min(int(np.searchsorted(edges, v, side="right")) - 1, len(edges) - 2)
            out[f"{edges[k]:g}-{edges[k + 1]:g}"] += 1
    return out


def net_td(case, chip_lower):
    """Identified Td without the chip's on-chip filter (lower delay bound of Task 1): the control chip's after the
    case_from_log correction, the FIFO chip's when the control chip is unknown; clamped at 0."""
    chip = case.get("control_chip") or case.get("fifo_chip")
    return max(case["plant"]["Td"] - chip_lower.get(chip, 0.0), 0.0)


def prior_coverage(pairs, tau, td, f=10.0):
    """Of the (tau_m, Td) pairs: how many lie in the prior box, and how many lag (motor + delay at f) less than the
    fastest or more than the slowest plant of the prior."""
    lo, hi = prior_lag_range(tau, td, f)
    lags = [plant_lag_deg(t, d, f) for t, d in pairs]
    return dict(n=len(pairs), in_box=int(sum(tau[0] <= t <= tau[1] and td[0] <= d <= td[1] for t, d in pairs)),
                faster_than_prior=int(sum(x < lo for x in lags)), slower_than_prior=int(sum(x > hi for x in lags)),
                lag_range=(lo, hi))


def noise_split(plant, gains, rates, s, motor_hz, psd_f, psd, f_lo, f_split=F_SPLIT, f_hi=F_HI, df=NOISE_DF):
    """(N^2 over [f_lo, f_hi], the part of it below f_split) with the integrand, grid and PSD interpolation of
    loop_model.LoopCase (static notches; ESC bank at the mean motor frequencies)."""
    def integrand(fn):
        A = gains.Kp + gains.Ki * integrator_response(fn, rates.f_ctrl)
        Dd = gains.Kd * derivative_response(fn, rates.fs)
        PZ = zoh_response(fn, rates.f_ctrl) * plant_response(fn, plant)
        F = gyro_stack(fn, s, rates.fs, motor_hz)
        K = A + Dd * pt1_response(fn, s.f_dterm, rates.fs)
        S = np.exp(np.interp(fn, psd_f, np.log(np.maximum(psd, 1e-30))))
        return np.abs(F * K / (1.0 + F * K * PZ)) ** 2 * S
    fn = np.arange(f_lo, f_hi + 1e-9, df)
    total = float(np.trapezoid(integrand(fn), fn))
    if f_split <= f_lo:
        return total, 0.0
    fb = np.append(fn[fn < f_split], f_split)
    return total, float(np.trapezoid(integrand(fb), fb))


def f_lo_rules(fc):
    """Lower edge of the noise band under the rules the author asked about (fc: the reference crossover)."""
    return {"5 f_c": 5.0 * fc, "3 f_c": 3.0 * fc, "10 Hz if 5 f_c < 20 Hz": 10.0 if 5.0 * fc < F_SPLIT else 5.0 * fc,
            "max(5 f_c, 20 Hz)": max(5.0 * fc, F_SPLIT)}


def scenario_b_counts(windows, fits):
    """Fit-half vehicles with at least one valid-RPM window, and generator entries with motor harmonics (the only
    possible sources of synthetic Scenario B drones: the vehicle spectrum keeps motor_hz only if every window has
    valid RPM, spectra_library.vehicle_spectra)."""
    veh = {}
    for w in windows:
        veh.setdefault(w["vehicle_key"], []).append(str(w["rpm_valid"]).lower() in ("true", "1"))
    used = {k: v for k, v in fits.items() if not v.get("dropped")}
    return dict(vehicles=len(veh), with_valid_rpm=int(sum(any(v) for v in veh.values())), generator_entries=len(used),
                generator_rpm_sources=int(sum(bool(v.get("motor_hz")) and v.get("n_harm", 0) > 0 for v in used.values())))


# ------------------------------------------------------------------ runner

def _eval(d):
    return af.synth_cells(d, cells=((af.KEY, af.WIDTH, af.DELAY),))[af.KEY]


def population_summary(synth, tries, evals):
    lim = af.separate_limits(af.MS_WC)
    nom = [af.nominal_ok(e) for e in evals]
    rob = [rb.passes(e, **lim) for e in evals]
    fc = [d["f_c"] for d in synth]
    return dict(n=len(synth), tries=tries, acceptance=len(synth) / tries,
                nominal_share=float(np.mean(nom)), robust_share=float(np.mean(rob)),
                robust_share_of_nominal=float(np.sum(rob) / max(np.sum(nom), 1)),
                hist_generated=histogram(fc), hist_nominal=histogram([x for x, k in zip(fc, nom) if k]),
                hist_robust=histogram([x for x, k in zip(fc, rob) if k]),
                share_identified=float(np.mean([d["source"] == "identified" for d in synth])),
                pm_unfiltered_q=[round(float(x), 1) for x in np.quantile([d["pm_unfiltered"] for d in synth], [0.1, 0.5, 0.9])],
                fc_nominal_q=[round(float(x), 2) for x in np.quantile([x for x, k in zip(fc, nom) if k] or [np.nan], [0.1, 0.5, 0.9])])


def run_population(rate_rows, prior, design, identified):
    kw = dict(tau=prior["tau"], td=prior["td"], pm_window=design["pm_window"],
              identified=identified, identified_share=prior.get("identified_share", 0.0))
    if "i_range" in design:
        kw["i_range"] = design["i_range"]
    else:
        kw["i_ratio"] = design["i_ratio"]
    synth, tries = synthetic_population(rate_rows, rl.N_SYNTH, rl.SEED, **kw)
    evals = Parallel(n_jobs=6)(delayed(_eval)(d) for d in synth)
    return population_summary(synth, tries, evals)


def f4_rows(cases, included, native, chip_lower):
    """Every included real case: its crossover, f_lo under each rule, and the share of N^2 below 20 Hz at the
    pilot's setting (mean over the case's library windows)."""
    out = []
    for i in included:
        c = cases[i]
        p, g, rates, mh = af._case_objects(c)
        s = rl._settings(c["settings"])
        rm = rb.robust_metrics(p, g, rates, s, mh, af.WIDTH, af.WIDTH, af.DELAY / rates.f_ctrl)
        fc = rm["nominal"]["fc"]
        keys = sorted(k[:-2] for k in native.files if k.startswith(c["log_id"]) and k.endswith("|f"))
        row = dict(case=i, log=c["log_id"][:8], axis="roll" if c["axis"] == 0 else "pitch", fc=fc,
                   nominal_ok=af.nominal_ok(rm), passes=rb.passes(rm, **af.separate_limits(af.MS_WC)),
                   fit_band=c["fit_band"], n_windows=len(keys), rules={})
        for name, f_lo in f_lo_rules(fc).items():
            shares, n2 = [], []
            for k in keys:
                f, P = native[k + "|f"].astype(float), native[k + "|P"][:, c["axis"]].astype(float)
                tot, below = noise_split(p, g, rates, s, mh, f, P, f_lo)
                shares.append(below / tot)
                n2.append(tot)
            row["rules"][name] = dict(f_lo=f_lo, share_below_20=float(np.mean(shares)) if shares else None,
                                      N2=[float(x) for x in n2])
        out.append(row)
    return out


def main():
    t0 = time.perf_counter()
    pri = json.load(open(pr.fit_path(HERE, "priors_from_logs.json")))
    afj = json.load(open(pr.fit_path(HERE, "a4_followup.json")))
    ps = json.load(open(pr.fit_path(HERE, "prior_sensitivity.json")))
    chip_lower = rl.chip_lower_bounds(pri)
    cases, inc = afj["cases"], afj["included"]
    ident = [(cases[i]["plant"]["tau_m"], net_td(cases[i], chip_lower)) for i in inc]
    cover = {k: prior_coverage(ident, v["tau"], v["td"]) for k, v in PRIORS.items()}
    pops = {}
    for k, v in PRIORS.items():
        pops[f"{k}+D0"] = run_population(pri["rate_rows"], v, DESIGNS["D0"], ident)
        print(k, round(time.perf_counter() - t0, 1), "s", flush=True)
    for k, v in DESIGNS.items():
        if k != "D0":
            pops[f"P0+{k}"] = run_population(pri["rate_rows"], PRIORS["P0"], v, ident)
            print(k, round(time.perf_counter() - t0, 1), "s", flush=True)
    for k, (pk, dk) in COMBOS.items():
        pops[f"{pk}+{dk}"] = run_population(pri["rate_rows"], PRIORS[pk], DESIGNS[dk], ident)
    if abs(pops["P0+D0"]["nominal_share"] - rl_nominal(afj)) > 1e-9:
        raise RuntimeError("P0+D0 does not reproduce the synthetic drones of robust_limits / a4_followup")
    inc_keys = {(cases[i]["log_id"][:8], "roll" if cases[i]["axis"] == 0 else "pitch") for i in inc}
    real_inc = [c for c in ps["real"]["cases"] if (c["log"], c["axis"]) in inc_keys]
    real = dict(hist_unfiltered=histogram([c["fc_unfiltered"] for c in real_inc]),
                hist_flown=histogram([c["fc_flown"] for c in real_inc]), n=len(real_inc),
                pm_unfiltered_q=[round(float(x), 1) for x in np.quantile([c["pm_unfiltered"] for c in real_inc], [0.1, 0.5, 0.9])],
                i_q=[round(float(x), 4) for x in np.quantile([c["i"] for c in real_inc], [0, 0.1, 0.5, 0.9, 1])])
    native = np.load(pr.fit_path(ROOT, "data", "spectra", "fit_psd_native.npz"))
    f4 = f4_rows(cases, inc, native, chip_lower)
    windows = list(csv.DictReader(open(pr.fit_path(ROOT, "data", "spectra", "fit_windows.csv"))))
    fits = json.load(open(os.path.join(ROOT, "experiments", "part2", "model_fits.json")))["vehicles"]
    rpm_logs = sorted({w["log_id"][:8] for w in windows if str(w["rpm_valid"]).lower() in ("true", "1")})
    identified_logs = sorted({cases[i]["log_id"][:8] for i in inc})
    a5 = dict(counts=scenario_b_counts(windows, fits), rpm_logs=rpm_logs,
              rpm_and_identified=sorted(set(rpm_logs) & set(identified_logs)))
    out = dict(settings=dict(PRIORS=PRIORS, DESIGNS=DESIGNS, COMBOS=COMBOS, FC_BINS=FC_BINS, N=rl.N_SYNTH, SEED=rl.SEED,
                             limits=dict(PM=af.PM_MIN, Ms=af.MS_MAX, GM=af.GM_MIN, Ms_wc=af.MS_WC, width=af.WIDTH,
                                         delay=af.DELAY)),
               identified=[dict(log=cases[i]["log_id"][:8], axis=cases[i]["axis"], tau_m=t, Td_net=d,
                                Td=cases[i]["plant"]["Td"], chip=cases[i].get("control_chip") or cases[i].get("fifo_chip"))
                           for i, (t, d) in zip(inc, ident)],
               coverage=cover, populations=pops, real=real, f4=f4, a5=a5, seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(HERE, "f_options.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"done in {out['seconds']} s")


def rl_nominal(afj):
    """Nominal pass share of the robust_limits synthetic drones, as a4_followup Q3 counted it (n_nominal / n_all)."""
    q = afj["q3"]["synthetic"]["3.0"]
    return q["n_nominal"] / q["n_all"]


if __name__ == "__main__":
    main()
