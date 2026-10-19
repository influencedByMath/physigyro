"""Mutation check of the Task 3 tests (S1-S4): break the code one way at a time, run the tests, restore byte-identical.

    .venv/Scripts/python experiments/phase4/mutation_check_task3.py

Never run it while an rq1_run.py part is running: its worker processes import these files from disk.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from mutation_check_priors import main  # noqa: E402

SRC = os.path.join(ROOT, "src")
TIER3 = [
    ("chip delay all in the step", "    return Plant(1.0, tau_m, td + lo), DELAY_SAMPLES / f_ctrl + (hi - lo)",
     "    return Plant(1.0, tau_m, td), DELAY_SAMPLES / f_ctrl + hi"),
    ("gains designed without the chip delay", "        g, m = design_gains(fc, r, p, Rates(fs, fctrl), i_ratio=I_RATIO)",
     "        g, m = design_gains(fc, r, Plant(1.0, tau_m, td), Rates(fs, fctrl), i_ratio=I_RATIO)"),
    ("never flagged", "    return kept, not rb.passes(rm, **limits()), rm", "    return kept, False, rm"),
    ("step 3 drops nothing", "        if ok or keep_all:", "        if True:"),
    ("no worst-case limits", "    return dict(NOMINAL, wc=worst_case_limits())", "    return dict(NOMINAL, wc=None)"),
]
LOOP = [("f_lo = 5 f_c again", 'self.f_lo = max(5.0 * self.ref_metrics["fc"], F_LO_MIN) if f_lo is None else f_lo',
         'self.f_lo = 5.0 * self.ref_metrics["fc"] if f_lo is None else f_lo')]
OPT = [
    ("corner limits ignored", '    return bool(w["stable"] and w["PM"] >= wc["PM"] and w["GM"] >= wc["GM"] and w["Ms"] <= wc["Ms"])',
     '    return bool(w["stable"])'),
    ("candidates not checked on the full grid", "        ok, m0, rm0 = _full_check(case, s0, opt, mode, pm_ref, n_max, lim, robust)",
     "        ok, m0, rm0 = True, case.evaluate(s0, S_eff=opt.S_eff), None"),
    ("refinement with the coarse option", "        opt = options_full[k]", "        opt = grid[k][0]"),
    ("refinement without the worst-case Ms constraint", '                 lambda x: wc["Ms"] - getr(x)["Ms"], lambda x: 1.0 if getr(x)["stable"] else -1.0]',
     "                 lambda x: 1.0, lambda x: 1.0]"),
    ("brute force skips the corners", "        if robust is not None and not corners_ok(robust_of(case, s, robust), robust.wc):",
     "        if False:"),
    ("reference ignored", "    if reference is not None:", "    if False:"),
    ("Pareto ignores the check", "        hit = next((c for c in cand if accept(c[1], c[2], c[3])), None)",
     "        hit = next((c for c in cand), None)"),
]
NM = [
    ("peaks below f_lo kept", "    inside = (f[pk] >= f_lo) & (f[pk] <= f_hi)", "    inside = f[pk] <= f_hi"),
    ("least prominent first", "    return [float(f[i]) for i in pk[np.argsort(prom)[::-1]][:n]]",
     "    return [float(f[i]) for i in pk[np.argsort(prom)][:n]]"),
]
CASES = [
    ("roll column for every case", '        Ps.append(np.asarray(native[w["window_id"] + "|P"], float)[:, axis])',
     '        Ps.append(np.asarray(native[w["window_id"] + "|P"], float)[:, 0])'),
    ("RPM if any window has it", '    if not all(_true(w["rpm_valid"]) for w in rows):', '    if not any(_true(w["rpm_valid"]) for w in rows):'),
    ("in-flight changes kept", '        if c.get("exclude"):', "        if False:"),
    ("loose tracking check", "- 1) <= tol)", "- 1) <= 10 * tol)"),
]
RUN = [
    ("motors without the frequency scale", '            mh = [float(x) * m["scale"] for x in e["motor_hz"]]',
     '            mh = [float(x) for x in e["motor_hz"]]'),
    ("no resume", '    todo = [it for it in items if it["id"] not in done]', "    todo = list(items)"),
    ("flagged drone with the smallest Ms", '    pick["flagged real"] = max(flagged, key=lambda r: r["worst_corner_ms"])["id"]',
     '    pick["flagged real"] = min(flagged, key=lambda r: r["worst_corner_ms"])["id"]'),
]
REPORT = [
    ("worst-only counts nominal failures", '    worst_only = sum(f["nominal_ok"] and not f["worst_ok"] for f in fl)',
     '    worst_only = sum(not f["worst_ok"] for f in fl)'),
    ("flown 10 Hz count from the unfiltered crossover", 'above_10_flown=sum((r["reference"].get("fc") or 0) > 10.0 for r in rows),',
     "above_10_flown=sum(fc_of(r) > 10.0 for r in rows),"),
    ("unsafe = nominal failures only", '    unsafe = st["ref_nominal_fail"] + st["ref_worst_only_fail"]', '    unsafe = st["ref_nominal_fail"]'),
]

if __name__ == "__main__":
    main(os.path.join(SRC, "tier3.py"), TIER3, "tests/test_tier3.py")
    main(os.path.join(SRC, "loop_model.py"), LOOP, "tests/test_loop_model.py")
    main(os.path.join(SRC, "optimizer.py"), OPT, "tests/test_optimizer_robust.py")
    main(os.path.join(SRC, "noise_model.py"), NM, "tests/test_noise_model.py")
    main(os.path.join(HERE, "rq1_cases.py"), CASES, "tests/test_rq1_cases.py")
    main(os.path.join(HERE, "rq1_run.py"), RUN, "tests/test_rq1_run.py")
    main(os.path.join(HERE, "rq1_report.py"), REPORT, "tests/test_rq1_report.py")
