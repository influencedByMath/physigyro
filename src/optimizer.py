"""Optimal filter settings for one drone and one noise spectrum (THESIS.md Section 7.3).

Decision variables: gyro LPF cutoff f_g, D-term cutoff f_d (continuous) and a
discrete notch option (Scenario A: static notches at detected peaks;
Scenario B: ESC-RPM notch bank bandwidth and harmonic count, or off).

Stability limits S (all modes): Nyquist-stable, GM >= GM_min, Ms <= Ms_max,
PM >= PM_min, all checked at every crossover.
  Mode A  maximize PM            s.t. N_rel <= 1 (no noisier than the reference) and S
  Mode B  minimize N_rel          s.t. PM >= PM(reference) and S
  Mode C  Pareto front: max PM for each noise limit N_max

Solver: vectorized coarse grid, best feasible point, then COBYLA refinement of
(log f_g, log f_d) with the notch option fixed.
"""
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

from loop_model import FilterSettings, LoopCase, notch_stack

LIMITS = dict(PM_min=30.0, GM_min=6.0, Ms_max=2.0)   # candidates; final values TBD (THESIS 7.8)


@dataclass
class NotchOption:
    label: str
    fields: dict                 # FilterSettings fields for this option
    S_eff: np.ndarray = None     # tracking-notch spectrum on the noise grid (Scenario B, real data)


def ranges(fs):
    return dict(f_gyro=(15.0, min(300.0, 0.45 * fs)), f_dterm=(10.0, min(200.0, 0.45 * fs)))


def scenario_b_options(case: LoopCase, base: FilterSettings, bws=(5, 7.5, 10, 15, 20, 30), hmcs=(1, 2, 3),
                       tracking=None):
    """'off' plus every (bandwidth, harmonics) pair of the ESC-RPM notch bank.

    tracking(bw, hmc, dnf_min) -> (f, psd) gives the time-domain notched spectrum
    of a real log, computed with the SAME notch floor as the stability path
    (base.dnf_min); without it the notch is modelled at the mean motor frequencies."""
    opts = [NotchOption("RPM notch off", dict(dnf_esc=False))]
    for h in hmcs:
        for bw in bws:
            S = None
            if tracking is not None:
                f, P = tracking(bw, h, base.dnf_min)
                S = case.psd_on_noise_grid(f, P)
            opts.append(NotchOption(f"RPM notch bw {bw:g} Hz x{h}", dict(dnf_esc=True, dnf_bw=float(bw), dnf_hmc=h,
                                                                          dnf_min=base.dnf_min), S))
    return opts


def scenario_a_options(peaks, bws=(5, 10, 20, 35, 50, 70, 100)):
    """Up to two static notches (IMU_GYRO_NF0/NF1) centred on the two largest peaks.

    Bandwidths span the PX4 parameter range of IMU_GYRO_NF0/1_BW (max 100 Hz in
    v1.16.0; decision P4: the 2026-10-10 optimum always sat on the old 50 Hz edge)."""
    p = list(peaks)[:2]
    opts = [NotchOption("no notch", dict(notches=()))]
    if not p:
        return opts
    for bw0 in bws:
        opts.append(NotchOption(f"NF0 {p[0]:.0f} Hz bw {bw0}", dict(notches=((p[0], bw0),))))
        if len(p) > 1:
            for bw1 in bws:
                opts.append(NotchOption(f"NF0 {p[0]:.0f}/{bw0} + NF1 {p[1]:.0f}/{bw1}",
                                        dict(notches=((p[0], bw0), (p[1], bw1)))))
    return opts


def feasible(r, mode, pm_ref=None, n_max=1.0, lim=LIMITS):
    ok = r["stable"] & (r["GM"] >= lim["GM_min"]) & (r["Ms"] <= lim["Ms_max"]) & (r["PM"] >= lim["PM_min"])
    if mode in ("A", "C"):
        ok = ok & (r["N_rel"] <= n_max)
    if mode == "B":
        ok = ok & (r["PM"] >= pm_ref)
    return ok


def _objective(r, mode):
    return -r["PM"] if mode in ("A", "C") else r["N_rel"]


def grid_search(case: LoopCase, base: FilterSettings, options, n_g=40, n_d=30, f_gyro=None, f_dterm=None):
    """Evaluate every option on the (f_g x f_d) grid. Returns list of (option, grid dict)."""
    rg = ranges(case.rates.fs)
    fg = np.geomspace(*rg["f_gyro"], n_g) if f_gyro is None else np.asarray(f_gyro)
    fd = np.geomspace(*rg["f_dterm"], n_d) if f_dterm is None else np.asarray(f_dterm)
    out = []
    for opt in options:
        s = replace(base, **opt.fields)
        fn = lambda f, s=s: notch_stack(f, s, case.rates.fs, case.motor_hz)
        nfn = None
        if opt.S_eff is not None:     # noise path: static notches only, ESC bank is inside S_eff
            sn = replace(s, dnf_esc=False)
            nfn = lambda f, sn=sn: notch_stack(f, sn, case.rates.fs, case.motor_hz)
        r = case.evaluate_grid(fg, fd, fn, S_eff=opt.S_eff, noise_notch_fn=nfn)
        r["f_gyro"], r["f_dterm"] = fg, fd
        out.append((opt, r))
    return out


def best_on_grid(grid, mode, pm_ref=None, n_max=1.0):
    best = None
    for opt, r in grid:
        ok = feasible(r, mode, pm_ref, n_max)
        if not ok.any():
            continue
        obj = np.where(ok, _objective(r, mode), np.inf)
        i, j = np.unravel_index(np.argmin(obj), obj.shape)
        if best is None or obj[i, j] < best[0]:
            best = (obj[i, j], opt, r["f_gyro"][i], r["f_dterm"][j])
    return best


def refine(case: LoopCase, base, opt: NotchOption, fg0, fd0, mode, pm_ref=None, n_max=1.0, lim=LIMITS):
    """COBYLA on (log f_g, log f_d), notch option fixed. Keeps the grid point if refinement fails."""
    rg = ranges(case.rates.fs)
    lo = np.log([rg["f_gyro"][0], rg["f_dterm"][0]])
    hi = np.log([rg["f_gyro"][1], rg["f_dterm"][1]])

    def ev(x):
        s = replace(base, **opt.fields, f_gyro=float(np.exp(x[0])), f_dterm=float(np.exp(x[1])))
        return case.evaluate(s, S_eff=opt.S_eff)

    cache = {}

    def get(x):
        k = tuple(np.round(x, 10))
        if k not in cache:
            cache[k] = ev(x)
        return cache[k]

    cons = [lambda x: get(x)["PM"] - lim["PM_min"],
            lambda x: min(get(x)["GM"], 60.0) - lim["GM_min"],
            lambda x: lim["Ms_max"] - get(x)["Ms"],
            lambda x: 1.0 if get(x)["stable"] else -1.0,
            lambda x: x[0] - lo[0], lambda x: hi[0] - x[0], lambda x: x[1] - lo[1], lambda x: hi[1] - x[1]]
    if mode in ("A", "C"):
        cons.append(lambda x: n_max - get(x)["N_rel"])
    else:
        cons.append(lambda x: get(x)["PM"] - pm_ref)
    sign = -1.0 if mode in ("A", "C") else 1.0
    key = "PM" if mode in ("A", "C") else "N_rel"
    x0 = np.log([fg0, fd0])
    res = minimize(lambda x: sign * get(x)[key], x0, method="COBYLA",
                   constraints=[dict(type="ineq", fun=c) for c in cons],
                   options=dict(rhobeg=0.08, maxiter=200, catol=1e-6))
    m0, m1 = get(x0), get(res.x)
    ok1 = all(c(res.x) >= -1e-6 for c in cons)
    better = sign * m1[key] < sign * m0[key]
    x = res.x if (ok1 and better) else x0
    return float(np.exp(x[0])), float(np.exp(x[1])), get(x), len(cache)


def optimize(case: LoopCase, base: FilterSettings, options, mode="A", n_max=1.0, grid=None, n_g=40, n_d=30):
    """Full solve for one mode. Returns dict with the setting, its metrics and timing."""
    t = time.perf_counter()
    if grid is None:
        grid = grid_search(case, base, options, n_g, n_d)
    t_grid = time.perf_counter() - t
    pm_ref = case.ref_metrics["PM"]
    best = best_on_grid(grid, mode, pm_ref, n_max)
    if best is None:
        return dict(feasible=False, t_grid=t_grid)
    _, opt, fg0, fd0 = best
    fg, fd, m, n_eval = refine(case, base, opt, fg0, fd0, mode, pm_ref, n_max)
    s = replace(base, **opt.fields, f_gyro=fg, f_dterm=fd)
    return dict(feasible=True, settings=s, option=opt, metrics=m, grid_point=(fg0, fd0),
                t_grid=t_grid, t_total=time.perf_counter() - t, n_refine_evals=n_eval)


def pareto_front(grid, n_max_values):
    """Mode C from the stored grid: best PM for each noise limit."""
    front = []
    for nm in n_max_values:
        b = best_on_grid(grid, "C", n_max=nm)
        front.append((nm, None if b is None else -b[0], None if b is None else b[1].label))
    return front
