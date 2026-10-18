"""Which synthetic Tier 3a drones are realistic? Design sensitivity (evidence for D-28 and THESIS 7.1.1 Tier 3a).

    .venv/Scripts/python experiments/phase4/prior_sensitivity.py

The A2-A4 evidence (robust_limits.json) showed that synthetic drones with f_c >= 10 Hz never pass even the nominal
limits at the PX4 default filters, while real fit-half drones (plant_variation.json identifications with the
pilot's gains) fly at 8-15 Hz. This runner
1. measures the real drones' design in the same terms as Tier 3a: crossover, PM WITHOUT filters, the D-corner ratio
   r = (Kp / Kd) / (2 pi f_c) and the I-corner ratio i = (Ki / Kp) / (2 pi f_c) (fit half; robust_limits cases);
2. repeats the synthetic evaluation of robust_limits.py for four designs (same seed, same rate pairs, f_c 4-15 Hz):
   (a) THESIS today: D-28 plant priors, unfiltered PM window 45-75 deg, i = 0.1;
   (b) the PM window matched to the real drones (their 10-90 % range, rounded out to 5 deg);
   (c) (b) plus i matched to the real median (rounded to 0.01);
   (d) (c) plus plant priors covering the identifications: tau_m 10-60 ms, Td 0.1-5 ms.
Writes prior_sensitivity.json next to this file.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import priors as pr                         # noqa: E402
import robust as rb                         # noqa: E402
import robust_limits as rl                  # noqa: E402
from joblib import delayed                  # noqa: E402
from loop_model import FilterSettings, Gains, Plant, Rates, margins, n_integrators   # noqa: E402

NO_FILTERS = FilterSettings(f_gyro=0.0, f_dterm=0.0)      # both low-passes off (pass-through), no notches


def real_design(robust_json, logs):
    """Unfiltered PM, crossover, r and i of every real-gain case (identified plant + the log's gains)."""
    by_log = {r["log_id"][:8]: r for r in logs}
    out = []
    for c in robust_json["real"]["cases"]:
        g = by_log[c["log"]]["fc"]["gains_roll" if c["axis"] == "roll" else "gains_pitch"]
        G = Gains(g["Kp"], g["Ki"], g["Kd"])
        rates = Rates(float(c["fs"]), float(c["f_ctrl"]))
        p = Plant(c["b"], c["tau_ms"] / 1e3, c["Td_ms"] / 1e3)
        f = rb.stability_grid(rates)
        m = margins(f, rb.open_loop(f, p, G, rates, NO_FILTERS, None), n_integrators(G))
        wc = 2 * math.pi * float(m["fc"])
        out.append(dict(log=c["log"], axis=c["axis"], fc_unfiltered=float(m["fc"]), pm_unfiltered=float(m["PM"]),
                        fc_flown=c["nominal"]["fc"], pm_flown=c["nominal"]["PM"],
                        r=(g["Kp"] / g["Kd"]) / wc if g["Kd"] > 0 else None, i=(g["Ki"] / g["Kp"]) / wc))
    return out


def main():
    t0 = time.perf_counter()
    pri = json.load(open(pr.fit_path(HERE, "priors_from_logs.json")))
    rob = json.load(open(pr.fit_path(HERE, "robust_limits.json")))
    logs = json.load(open(pr.fit_path(ROOT, "data", "spectra", "fit_logs.json")))
    real = real_design(rob, logs)
    pm = np.array([x["pm_unfiltered"] for x in real])
    i_med = float(np.median([x["i"] for x in real]))
    r_vals = np.array([x["r"] for x in real if x["r"] is not None])
    window = (5 * math.floor(np.quantile(pm, 0.1) / 5), 5 * math.ceil(np.quantile(pm, 0.9) / 5))
    i_match = round(i_med, 2)
    designs = {
        "a: THESIS (D-28, PM 45-75, i 0.1)": dict(tau=rl.TAU, td=rl.TD, pm_window=(45.0, 75.0), i_ratio=0.1),
        f"b: PM window {window[0]}-{window[1]}": dict(tau=rl.TAU, td=rl.TD, pm_window=window, i_ratio=0.1),
        f"c: b + i {i_match}": dict(tau=rl.TAU, td=rl.TD, pm_window=window, i_ratio=i_match),
        "d: c + tau 10-60 ms, Td 0.1-5 ms": dict(tau=(0.010, 0.060), td=(0.0001, 0.005), pm_window=window, i_ratio=i_match),
    }
    sets = rl.limit_sets()
    key_sets = {k: sets[k] for k in rl.KEY_SETS}
    out = dict(real=dict(cases=real,
                         pm_unfiltered_q={str(q): round(float(np.quantile(pm, q)), 1) for q in (0, 0.1, 0.5, 0.9, 1)},
                         share_in_45_75=round(float(np.mean((pm >= 45) & (pm <= 75))), 3),
                         i_median=round(i_med, 4), r_q={str(q): round(float(np.quantile(r_vals, q)), 2) for q in (0, 0.1, 0.5, 0.9, 1)},
                         r_share_in_0p5_3=round(float(np.mean((r_vals >= 0.5) & (r_vals <= 3))), 3)),
               designs={}, settings=dict(N=rl.N_SYNTH, SEED=rl.SEED, F_C=rl.F_C, R_RANGE=rl.R_RANGE, window=window,
                                         i_match=i_match))
    for name, kw in designs.items():
        synth, tries, n_pairs = rl.synthetic_drones(pri["rate_rows"], rl.N_SYNTH, rl.SEED, **kw)
        ev = rl._par(delayed(rl.eval_synth)(d) for d in synth)
        out["designs"][name] = dict(params={k: list(v) if isinstance(v, tuple) else v for k, v in kw.items()},
                                    n=len(synth), tries=tries, nominal_pass_rates=rl.nominal_pass_rates(ev),
                                    conditional_rates=rl.conditional_rates(ev, key_sets), pass_rates=rl.pass_rates(ev, key_sets),
                                    fc_bins=rl.fc_bins(synth, ev, key_sets, rl.KEY_CELLS))
        print(name, "done", round(time.perf_counter() - t0, 1), "s", flush=True)
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(HERE, "prior_sensitivity.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
