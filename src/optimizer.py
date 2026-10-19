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

Task 3 (D-37): with ``robust`` the limits S also hold the separate worst-case limits (D-32, D-36) for the 8 corner
plants of the uncertainty set, so every optimum passes the safety check of THESIS 7.8 by construction; with
``coarse`` the grid search runs on a coarse LoopCase (fast grid) and every candidate, the refinement and the
reported metrics use the full LoopCase. Candidates are tried in the order of the objective (lazy check).
"""
import time
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import minimize

import robust as rb
from loop_model import FilterSettings, LoopCase, notch_stack

LIMITS = dict(PM_min=30.0, GM_min=6.0, Ms_max=2.0)   # nominal limits: D-30, D-31 (THESIS 7.8)
FAST = dict(n_stab=500, noise_df=1.0, n_g=24, n_d=18)     # coarse search grid (D-37 (4); feasibility fast grid)
FULL = dict(n_stab=1500, noise_df=0.5, n_g=40, n_d=30)    # refinement and reporting grid (LoopCase defaults)


@dataclass
class Robust:
    """Uncertainty set and worst-case limits: corners b (1 +- delta_b), tau_m (1 +- delta_tau), Td + {0, delta_d};
    wc = dict(PM=, Ms=, GM=) for every corner plant (D-32, D-36 (1))."""
    delta_b: float
    delta_tau: float
    delta_d: float
    wc: dict


def robust_of(case: LoopCase, s: FilterSettings, spec: Robust):
    """robust.robust_metrics of setting s on the stability grid of ``case`` (stability path as in LoopCase.evaluate)."""
    return rb.robust_metrics(case.plant, case.gains, case.rates, s, case.motor_hz, spec.delta_b, spec.delta_tau,
                             spec.delta_d, n_stab=len(case.f_stab))


def corners_ok(rm, wc):
    w = rm["worst_corners"]
    return bool(w["stable"] and w["PM"] >= wc["PM"] and w["GM"] >= wc["GM"] and w["Ms"] <= wc["Ms"])


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


def refine(case: LoopCase, base, opt: NotchOption, fg0, fd0, mode, pm_ref=None, n_max=1.0, lim=LIMITS, robust=None):
    """COBYLA on (log f_g, log f_d), notch option fixed. Keeps the grid point if refinement fails.
    With ``robust`` (Robust) the worst-case limits of the corner plants are constraints too."""
    rg = ranges(case.rates.fs)
    lo = np.log([rg["f_gyro"][0], rg["f_dterm"][0]])
    hi = np.log([rg["f_gyro"][1], rg["f_dterm"][1]])

    def ev(x):
        s = replace(base, **opt.fields, f_gyro=float(np.exp(x[0])), f_dterm=float(np.exp(x[1])))
        return case.evaluate(s, S_eff=opt.S_eff)

    cache, rcache = {}, {}

    def get(x):
        k = tuple(np.round(x, 10))
        if k not in cache:
            cache[k] = ev(x)
        return cache[k]

    def getr(x):
        k = tuple(np.round(x, 10))
        if k not in rcache:
            s = replace(base, **opt.fields, f_gyro=float(np.exp(x[0])), f_dterm=float(np.exp(x[1])))
            rcache[k] = robust_of(case, s, robust)["worst_corners"]
        return rcache[k]

    cons = [lambda x: get(x)["PM"] - lim["PM_min"],
            lambda x: min(get(x)["GM"], 60.0) - lim["GM_min"],
            lambda x: lim["Ms_max"] - get(x)["Ms"],
            lambda x: 1.0 if get(x)["stable"] else -1.0,
            lambda x: x[0] - lo[0], lambda x: hi[0] - x[0], lambda x: x[1] - lo[1], lambda x: hi[1] - x[1]]
    if mode in ("A", "C"):
        cons.append(lambda x: n_max - get(x)["N_rel"])
    else:
        cons.append(lambda x: get(x)["PM"] - pm_ref)
    if robust is not None:
        wc = robust.wc
        cons += [lambda x: getr(x)["PM"] - wc["PM"], lambda x: min(getr(x)["GM"], 60.0) - wc["GM"],
                 lambda x: wc["Ms"] - getr(x)["Ms"], lambda x: 1.0 if getr(x)["stable"] else -1.0]
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


def _full_check(case, s, opt, mode, pm_ref, n_max, lim, robust):
    """(passes, metrics, robust metrics) of setting s on the full LoopCase."""
    m = case.evaluate(s, S_eff=opt.S_eff)
    ok = bool(feasible(m, mode, pm_ref, n_max, lim))
    rm = None
    if ok and robust is not None:
        rm = robust_of(case, s, robust)
        ok = corners_ok(rm, robust.wc)
    return ok, m, rm


def optimize_checked(case: LoopCase, base: FilterSettings, options, mode="A", n_max=1.0, robust=None, coarse=None,
                     grid=None, n_g=FAST["n_g"], n_d=FAST["n_d"], max_candidates=3000, lim=LIMITS, options_full=None,
                     reference=None):
    """D-37 solve: grid on ``coarse`` (or ``case``); the nominal-feasible grid points in the order of the objective
    are checked on the full ``case`` (nominal limits, mode constraint, worst-case limits) until one passes; the
    refinement starts there, on the full case. Returns the setting with its full-grid metrics.
    ``options_full``: the same options built on ``case`` (their S_eff lie on its noise grid); default ``options``.
    ``reference``: the reference setting of ``case``; when it meets every limit it is a candidate and a second
    refinement starts from it, so the optimum is never worse than a feasible reference."""
    options_full = options if options_full is None else options_full
    if [o.label for o in options_full] != [o.label for o in options]:
        raise ValueError("options_full must hold the same options in the same order")
    t = time.perf_counter()
    if grid is None:
        grid = grid_search(coarse if coarse is not None else case, base, options, n_g, n_d)
    t_grid = time.perf_counter() - t
    pm_ref = case.ref_metrics["PM"]
    pm_ref_c = (coarse if coarse is not None else case).ref_metrics["PM"]
    cand = []
    for k, (opt, r) in enumerate(grid):
        ok = feasible(r, mode, pm_ref_c, n_max, lim)
        obj = _objective(r, mode)
        for i, j in zip(*np.nonzero(ok)):
            cand.append((float(obj[i, j]), k, int(i), int(j)))
    cand.sort()
    n_checked = 0
    found = None
    for _, k, i, j in cand[:max_candidates]:
        _, r = grid[k]
        opt = options_full[k]
        fg0, fd0 = float(r["f_gyro"][i]), float(r["f_dterm"][j])
        s0 = replace(base, **opt.fields, f_gyro=fg0, f_dterm=fd0)
        n_checked += 1
        ok, m0, rm0 = _full_check(case, s0, opt, mode, pm_ref, n_max, lim, robust)
        if not ok:
            continue
        fg, fd, m, n_eval = refine(case, base, opt, fg0, fd0, mode, pm_ref, n_max, lim, robust)
        s = replace(base, **opt.fields, f_gyro=fg, f_dterm=fd)
        ok1, m1, rm1 = _full_check(case, s, opt, mode, pm_ref, n_max, lim, robust)
        if not ok1:                       # refinement left the feasible set: keep the checked grid point
            s, m1, rm1 = s0, m0, rm0
        if robust is not None and rm1 is None:
            rm1 = robust_of(case, s, robust)
        found = dict(feasible=True, settings=s, option=opt, metrics=m1, robust=rm1, grid_point=(fg0, fd0),
                     n_refine_evals=n_eval, start="grid")
        break
    if reference is not None:
        rres = _from_reference(case, base, options_full, reference, mode, pm_ref, n_max, lim, robust)
        if rres is not None and (found is None or _objective(rres["metrics"], mode) < _objective(found["metrics"], mode)):
            found = rres
    timing = dict(n_candidates=n_checked, n_feasible_coarse=len(cand), t_grid=t_grid, t_total=time.perf_counter() - t)
    if found is None:
        return dict(feasible=False, **timing)
    return dict(found, **timing)


def _from_reference(case, base, options_full, reference, mode, pm_ref, n_max, lim, robust):
    """The reference as a candidate: refined inside the option that reproduces it, or kept as it is when no option
    does (e.g. flown static notches in Scenario A); None if the reference breaks a limit."""
    opt = next((o for o in options_full
                if replace(base, **o.fields, f_gyro=reference.f_gyro, f_dterm=reference.f_dterm) == reference), None)
    if opt is None:
        m = case.ref_metrics
        ok = bool(feasible(m, mode, pm_ref, n_max, lim))
        rm = robust_of(case, reference, robust) if robust is not None else None
        if not ok or (rm is not None and not corners_ok(rm, robust.wc)):
            return None
        fields = dict(notches=reference.notches, dnf_esc=reference.dnf_esc, dnf_bw=reference.dnf_bw,
                      dnf_hmc=reference.dnf_hmc, dnf_min=reference.dnf_min)
        return dict(feasible=True, settings=reference, option=NotchOption("reference setting", fields), metrics=m,
                    robust=rm, grid_point=(reference.f_gyro, reference.f_dterm), n_refine_evals=0, start="reference")
    ok, m0, rm0 = _full_check(case, reference, opt, mode, pm_ref, n_max, lim, robust)
    if not ok:
        return None
    fg, fd, _, n_eval = refine(case, base, opt, reference.f_gyro, reference.f_dterm, mode, pm_ref, n_max, lim, robust)
    s = replace(base, **opt.fields, f_gyro=fg, f_dterm=fd)
    ok1, m1, rm1 = _full_check(case, s, opt, mode, pm_ref, n_max, lim, robust)
    if not ok1:
        s, m1, rm1 = reference, m0, rm0
    if robust is not None and rm1 is None:
        rm1 = robust_of(case, s, robust)
    return dict(feasible=True, settings=s, option=opt, metrics=m1, robust=rm1,
                grid_point=(reference.f_gyro, reference.f_dterm), n_refine_evals=n_eval, start="reference")


def brute_force(case: LoopCase, base: FilterSettings, option: NotchOption, mode="A", n_max=1.0, robust=None,
                n_g=120, n_d=100, lim=LIMITS):
    """Fine grid search of one notch option on the full ``case`` (THESIS 7.3 brute-force check): the best grid point
    that meets the nominal limits, the mode constraint and (with ``robust``) the worst-case limits."""
    grid = grid_search(case, base, [option], n_g, n_d)
    _, r = grid[0]
    ok = feasible(r, mode, case.ref_metrics["PM"], n_max, lim)
    obj = _objective(r, mode)
    cand = sorted((float(obj[i, j]), int(i), int(j)) for i, j in zip(*np.nonzero(ok)))
    for n, (_, i, j) in enumerate(cand, 1):
        s = replace(base, **option.fields, f_gyro=float(r["f_gyro"][i]), f_dterm=float(r["f_dterm"][j]))
        if robust is not None and not corners_ok(robust_of(case, s, robust), robust.wc):
            continue
        return dict(feasible=True, settings=s, metrics=case.evaluate(s, S_eff=option.S_eff), f_gyro=s.f_gyro,
                    f_dterm=s.f_dterm, n_candidates=n)
    return dict(feasible=False, n_candidates=len(cand))


def optimize(case: LoopCase, base: FilterSettings, options, mode="A", n_max=1.0, grid=None, n_g=40, n_d=30,
             robust=None, coarse=None):
    """Full solve for one mode. Returns dict with the setting, its metrics and timing. With ``robust`` or ``coarse``
    the D-37 solve (optimize_checked) is used."""
    if robust is not None or coarse is not None:
        return optimize_checked(case, base, options, mode, n_max, robust, coarse, grid, n_g, n_d)
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


def pareto_front(grid, n_max_values, accept=None):
    """Mode C from the stored grid: best PM for each noise limit. ``accept(option, f_g, f_d)`` (optional) must also
    hold, e.g. the worst-case check of robust_accept; points are tried in the order of PM."""
    front = []
    for nm in n_max_values:
        if accept is None:
            b = best_on_grid(grid, "C", n_max=nm)
            front.append((nm, None if b is None else -b[0], None if b is None else b[1].label))
            continue
        cand = []
        for opt, r in grid:
            ok = feasible(r, "C", n_max=nm)
            for i, j in zip(*np.nonzero(ok)):
                cand.append((-float(r["PM"][i, j]), opt, float(r["f_gyro"][i]), float(r["f_dterm"][j])))
        cand.sort(key=lambda c: c[0])
        hit = next((c for c in cand if accept(c[1], c[2], c[3])), None)
        front.append((nm, None if hit is None else -hit[0], None if hit is None else hit[1].label))
    return front


def robust_accept(case: LoopCase, base: FilterSettings, spec: Robust):
    """accept() for pareto_front: the corner plants of ``case`` meet the worst-case limits of ``spec``."""
    def accept(opt, fg, fd):
        return corners_ok(robust_of(case, replace(base, **opt.fields, f_gyro=fg, f_dterm=fd), spec), spec.wc)
    return accept
