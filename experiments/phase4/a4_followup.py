"""The author's questions before finalizing A4 (2026-10-18). FIT HALF only.

    .venv/Scripts/python experiments/phase4/a4_followup.py

Limits: nominal PM >= 30 deg (A2), Ms <= 2.0 (A3), GM >= 6 dB; separate worst-case limit (concept accepted): every
plant of the set Nyquist-stable with Ms <= Ms_wc and the PM / GM that Ms_wc guarantees; set +-20 % on b and tau_m,
delay step 1 controller sample (the A4 candidate).
Q1 Which real flown settings pass nominally but fail the worst case (Ms_wc 3)? Every plant's margins and critical
   frequencies (robust.corner_metrics), the identified band, and the MEASURED frequency response next to the model
   around the critical frequencies (is extrapolation the cause?).
Q2 The same identifications with a wider band (to 40 Hz), a relaxed coherence gate (0.3) and longer Welch segments
   (2.56 s), for every included case (not only the failing ones), with the provisional gates of plant_variation.accept.
Q3 Pass rates for Ms_wc 2.0-4.0, real flown settings and synthetic Tier 3a drones at the PX4 defaults.
Q4 For the cases whose flown setting fails: does ANY filter setting in the optimizer's ranges pass (THESIS 7.8 step 4.1
   "reference unsafe", D-15 "reference outside limits")? Stability limits only; noise is not part of this check.
The rebuilt cases must reproduce robust_limits.json, and the synthetic drones its pass rates (checked, else it stops).
Writes a4_followup.json next to this file. Nothing reads the test half (priors.fit_path).
"""
import json
import math
import os
import sys
import time
from dataclasses import replace

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import noise_data as nd                     # noqa: E402
import optimizer as opt                     # noqa: E402
import plant_variation as pv                # noqa: E402
import priors as pr                         # noqa: E402
import robust as rb                         # noqa: E402
import robust_limits as rl                  # noqa: E402
import sysid                                # noqa: E402
from loop_model import (Gains, Plant, Rates, PX4_V116_MC_DEFAULT, derivative_response, gyro_stack,   # noqa: E402
                        integrator_response, pt1_response)

PM_MIN, MS_MAX, GM_MIN = 30.0, 2.0, 6.0     # A2, A3 (author 2026-10-18); GM THESIS 7.3
WIDTH, DELAY = 0.20, 1.0                    # A4 candidate: +-20 % on b and tau_m, delay step in controller samples
KEY = "w0.2_d1"                             # the same cell in robust_limits.json
MS_WC = 3.0                                 # worst-case candidate (question 3)
MS_SWEEP = (2.0, 2.5, 3.0, 3.5, 4.0)
ID_VARIANTS = {
    "V0": dict(label="D-14: 0.5-20 Hz, coherence >= 0.5, 1.28 s segments", f_band=(0.5, 20.0), min_coh=0.5, seg_mult=1),
    "V1": dict(label="band to 40 Hz", f_band=(0.5, 40.0), min_coh=0.5, seg_mult=1),
    "V2": dict(label="band to 40 Hz, coherence >= 0.3", f_band=(0.5, 40.0), min_coh=0.3, seg_mult=1),
    "V3": dict(label="band to 40 Hz, 2.56 s segments", f_band=(0.5, 40.0), min_coh=0.5, seg_mult=2),
    "V4": dict(label="band to 40 Hz, coherence >= 0.3, 2.56 s segments", f_band=(0.5, 40.0), min_coh=0.3, seg_mult=2),
}
NOTCH_OPTIONS = {"notches as flown": {}, "all notches off": dict(notches=(), dnf_esc=False)}
F_DIAG = (5.0, 40.0)                        # measured-vs-model table around the critical frequencies
F_STORE = 60.0                              # measured FRF stored up to this frequency


def _r(d, nd_=4):
    """JSON-safe copy: floats rounded, NaN -> None."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (bool, np.bool_)):
            out[k] = bool(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = round(float(v), nd_) if math.isfinite(v) else None
        else:
            out[k] = v
    return out


def separate_limits(ms_wc, pm=PM_MIN, ms=MS_MAX, gm=GM_MIN):
    """Nominal (pm, ms, gm) and the separate worst-case limits implied by ms_wc (robust_limits.wc_limits)."""
    return dict(PM_min=pm, Ms_max=ms, GM_min=gm, wc=rl.wc_limits(ms_wc))


def nominal_ok(rm, pm=PM_MIN, ms=MS_MAX, gm=GM_MIN):
    n = rm["nominal"]
    return bool(n["stable"] and n["PM"] >= pm and n["Ms"] <= ms and n["GM"] >= gm)


def ms_wc_sweep(rows, ms_values=MS_SWEEP):
    """Pass counts of robust_metrics results for each Ms_wc (separate limits), overall and among the nominal passes."""
    nom = [nominal_ok(r) for r in rows]
    n_nom = int(sum(nom))
    out = {}
    for m in ms_values:
        lim = separate_limits(m)
        n_pass = int(sum(rb.passes(r, **lim) for r in rows))
        out[m] = dict(PM=lim["wc"]["PM"], GM=lim["wc"]["GM"], n_all=len(rows), n_nominal=n_nom, n_pass=n_pass,
                      share_all=n_pass / len(rows) if rows else None,
                      share_of_nominal=n_pass / n_nom if n_nom else None)
    return out


def failing_cases(rows, ms_wc=MS_WC):
    """Indices of results that pass the nominal limits but fail the separate worst-case limit."""
    lim = separate_limits(ms_wc)
    return [i for i, r in enumerate(rows) if nominal_ok(r) and not rb.passes(r, **lim)]


def loop_parts(f, gains, rates, s, motor_hz):
    """F_g [Kp + Ki I(z) + Kd D(z) F_d]: robust.open_loop without ZOH * plant."""
    A = gains.Kp + gains.Ki * integrator_response(f, rates.f_ctrl)
    Dd = gains.Kd * derivative_response(f, rates.fs)
    return gyro_stack(f, s, rates.fs, motor_hz) * (A + Dd * pt1_response(f, s.f_dterm, rates.fs))


def measured_open_loop(G, f, td_shift, parts):
    """Open loop with the MEASURED response G (torque setpoint -> body rate; ZOH and the FIFO chip inside) in place of
    ZOH * plant; td_shift [s] is the chip correction that case_from_log applied to the identified Td (a pure delay)."""
    return parts * np.asarray(G) * np.exp(-2j * np.pi * np.asarray(f) * td_shift)


def frf_rel_error(coh, n_seg):
    """Approximate 1-sigma relative random error of the instrument-variable FRF G = S_rw / S_ru: the two H1 errors
    sqrt(1 - g2) / (|g| sqrt(2 n)) (Bendat and Piersol) added in quadrature, with g2 the smaller coherence."""
    coh = np.asarray(coh, dtype=float)
    return np.sqrt((1.0 - coh) / (coh * n_seg))


def refit_case(case, fit, corr):
    """The case with another identification's plant; corr = robust_limits.chip_td_correction (None: control chip
    unknown, no correction), with the clamp at 0 of case_from_log."""
    td = fit["Td"] if corr is None else max(fit["Td"] + corr, 0.0)
    return dict(case, plant=dict(b=fit["b"], tau_m=fit["tau_m"], Td=float(td)), td_identified=fit["Td"])


def variant_fits(pieces, Tc, nseg, cls, variants=ID_VARIANTS):
    """plant_variation.fit_pieces under each identification variant; a variant that cannot run records its error."""
    out = {}
    for k, v in variants.items():
        try:
            out[k] = pv.fit_pieces(pieces, Tc, nseg * v["seg_mult"], cls, f_band=v["f_band"], min_coh=v["min_coh"])
        except ValueError as e:
            out[k] = dict(error=str(e))
    return out


def settings_with(case, **fields):
    return replace(rl._settings(case["settings"]), **fields)


def _case_objects(case):
    g = Gains(**case["gains"])
    rates = Rates(float(case["fs"]), float(case["f_ctrl"]))
    mh = None if case["motor_hz"] is None else np.asarray(case["motor_hz"], float)
    return Plant(**case["plant"]), g, rates, mh


def eval_cell(case, width=WIDTH, delay=DELAY, s=None):
    """robust_metrics of one case at one (width, delay step) -- the computation of robust_limits.evaluate."""
    p, g, rates, mh = _case_objects(case)
    return rb.robust_metrics(p, g, rates, s or rl._settings(case["settings"]), mh, width, width, delay / rates.f_ctrl)


def feasible_filters(case, n_g=24, n_d=18, width=WIDTH, delay=DELAY, ms_wc=MS_WC):
    """Grid over the optimizer's ranges of f_gyro x f_dterm (log-spaced), notches as flown or all off: how many
    settings pass the separate limits, and the best one (passing first, then nominal, then the lowest worst Ms)."""
    p, g, rates, mh = _case_objects(case)
    lim = separate_limits(ms_wc)
    rg = opt.ranges(rates.fs)
    fgs = [float(x) for x in np.geomspace(*rg["f_gyro"], n_g)]
    fds = [float(x) for x in np.geomspace(*rg["f_dterm"], n_d)]
    out = {}
    for name, fields in NOTCH_OPTIONS.items():
        n_pass = n_nom = 0
        best = None
        for fg in fgs:
            for fd in fds:
                s = settings_with(case, f_gyro=fg, f_dterm=fd, **fields)
                rm = rb.robust_metrics(p, g, rates, s, mh, width, width, delay / rates.f_ctrl)
                ok, nom = rb.passes(rm, **lim), nominal_ok(rm)
                n_pass += ok
                n_nom += nom
                key = (not ok, not nom, rm["worst"]["Ms"])
                if best is None or key < best[0]:
                    best = (key, dict(f_gyro=fg, f_dterm=fd, passes=bool(ok), nominal_ok=bool(nom),
                                      nominal=_r(rm["nominal"]), worst=_r(rm["worst"])))
        out[name] = dict(n_grid=len(fgs) * len(fds), n_pass=int(n_pass), n_nominal=int(n_nom), f_gyro=fgs,
                         f_dterm=fds, best=best[1])
    return out


# ------------------------------------------------------------------ runner parts (need the fit-half logs)

def reidentify_log(log_id, items, variants=ID_VARIANTS):
    """items: [(case index, axis, cls)] of one log. Pieces as in plant_variation (MAX_HOLD, T_EDGE, D-14 instrument);
    fits under every variant and the measured FRF of the fit instrument at both segment lengths."""
    u = nd.load(pr.fit_path(ROOT, "data", "raw_logs", log_id + ".ulg"))
    gs = nd.gyro_stream(u)
    out = {}
    for i, axis, cls in items:
        groups, Tc, nseg = pv.build_pieces(u, gs, axis)
        pieces = groups[cls]
        name = "rate_sp" if cls == "rate_sp" else "att_sp"
        frf = {}
        for mult in (1, 2):
            try:
                S = sysid.piece_spectra([dict(r=p[name], u=p["u"], w=p["w"]) for p in pieces], fs=1 / Tc,
                                        nperseg=nseg * mult)
            except ValueError as e:
                frf[str(mult)] = dict(error=str(e))
                continue
            f, G, coh = sysid.frf_from_spectra(S)
            m = f <= F_STORE
            frf[str(mult)] = dict(f=f[m].tolist(), re=G[m].real.tolist(), im=G[m].imag.tolist(), coh=coh[m].tolist(),
                                  n_seg=int(S["n_seg"]))
        out[i] = dict(fits=variant_fits(pieces, Tc, nseg, cls, variants), frf=frf, Tc=Tc, nseg=nseg)
    return out


def measured_vs_model(case, frf, Tc):
    """Rows at the Welch frequencies in F_DIAG: coherence, approximate random error, measured/model gain [dB] and
    phase [deg] (model = the identified plant with its uncorrected Td), and the sensitivity 1/|1+L| from the measured
    response and from the model loop (both with the chip correction)."""
    f = np.asarray(frf["f"])
    G = np.asarray(frf["re"]) + 1j * np.asarray(frf["im"])
    coh = np.asarray(frf["coh"])
    sel = (f >= F_DIAG[0]) & (f <= F_DIAG[1])
    f, G, coh = f[sel], G[sel], coh[sel]
    p, g, rates, mh = _case_objects(case)
    s = rl._settings(case["settings"])
    model = sysid.plant_model(f, p.b, p.tau_m, case["td_identified"], Tc)
    parts = loop_parts(f, g, rates, s, mh)
    L_meas = measured_open_loop(G, f, p.Td - case["td_identified"], parts)
    L_mod = rb.open_loop(f, p, g, rates, s, mh)
    err = frf_rel_error(coh, frf["n_seg"])
    q = G / model
    return [dict(f=round(float(f[k]), 2), coh=round(float(coh[k]), 3), err_db=round(float(20 * np.log10(1 + err[k])), 2),
                 err_deg=round(float(np.degrees(err[k])), 1), gain_db=round(float(20 * np.log10(abs(q[k]))), 2),
                 phase_deg=round(float(np.degrees(np.angle(q[k]))), 1),
                 S_meas=round(float(1 / abs(1 + L_meas[k])), 3), S_model=round(float(1 / abs(1 + L_mod[k])), 3))
            for k in range(len(f))]


def check_reproduction(cases, evals, stored):
    """The rebuilt cases must give robust_limits.json (stored rounded to 3 decimals)."""
    bad = []
    if len(cases) != len(stored):
        raise RuntimeError(f"{len(cases)} cases rebuilt, {len(stored)} stored")
    for c, e, s in zip(cases, evals, stored):
        if c["log_id"][:8] != s["log"][:8] or ("roll" if c["axis"] == 0 else "pitch") != s["axis"]:
            bad.append((c["log_id"][:8], c["axis"], "order"))
        for k in ("PM", "Ms", "GM"):
            if abs(e[KEY]["nominal"][k] - s["nominal"][k]) > 1.5e-3:
                bad.append((s["log"], s["axis"], "nominal", k))
            for key in e:
                if abs(e[key]["worst"][k] - s["worst"][key][k]) > 1.5e-3:
                    bad.append((s["log"], s["axis"], key, k))
    if bad:
        raise RuntimeError(f"rebuilt cases differ from robust_limits.json: {bad[:6]}")


def synth_cells(d, cells=(("w0.2_d1", 0.20, 1.0), ("w0.15_d1", 0.15, 1.0))):
    """robust_metrics of one synthetic drone at the PX4 v1.16 defaults (robust_limits.eval_synth, fewer cells)."""
    p, g, rates = Plant(1.0, d["tau_m"], d["Td"]), Gains(**d["gains"]), Rates(d["fs"], d["f_ctrl"])
    return {k: rb.robust_metrics(p, g, rates, PX4_V116_MC_DEFAULT, None, w, w, dd / rates.f_ctrl) for k, w, dd in cells}


def _corner_rows(case):
    p, g, rates, mh = _case_objects(case)
    return [_r(x) for x in rb.corner_metrics(p, g, rates, rl._settings(case["settings"]), mh, WIDTH, WIDTH,
                                            DELAY / rates.f_ctrl)]


def _coherent(frf, lo, hi, c):
    f, coh = np.asarray(frf["f"]), np.asarray(frf["coh"])
    return [round(float(x), 2) for x in f[(f >= lo) & (f <= hi) & (coh >= c)]]


def _case_summary(c):
    keep = ("log_id", "axis", "cls", "seconds", "hw", "fifo_chip", "control_chip", "plant", "td_identified", "fit_band",
            "gains", "f_ctrl", "fs", "fs_source", "settings", "motor_hz", "flags", "exclude")
    return {k: c.get(k) for k in keep}


def main():
    t0 = time.perf_counter()
    pvj = json.load(open(pr.fit_path(HERE, "plant_variation.json")))
    logs = json.load(open(pr.fit_path(ROOT, "data", "spectra", "fit_logs.json")))
    pri = json.load(open(pr.fit_path(HERE, "priors_from_logs.json")))
    rlj = json.load(open(pr.fit_path(HERE, "robust_limits.json")))
    chip_lower = rl.chip_lower_bounds(pri)
    cases = rl.real_cases(pvj, logs, chip_lower)
    evals = [rl.run_case(c) for c in cases]
    check_reproduction(cases, evals, rlj["real"]["cases"])
    print("cases rebuilt and reproduced", round(time.perf_counter() - t0, 1), "s", flush=True)
    inc = [i for i, c in enumerate(cases) if not c["exclude"]]
    rows = [evals[i][KEY] for i in inc]
    fail = [inc[j] for j in failing_cases(rows)]
    nomfail = [i for i in inc if not nominal_ok(evals[i][KEY])]

    # Q2 first: the re-identification also gives the measured FRFs that Q1 needs
    by_log = {}
    for i in inc:
        c = cases[i]
        by_log.setdefault(c["log_id"], []).append((i, c["axis"], c["cls"]))
    res = Parallel(n_jobs=4, verbose=5)(delayed(reidentify_log)(lid, items) for lid, items in by_log.items())
    reid = {i: v for part in res for i, v in part.items()}
    print("re-identification done", round(time.perf_counter() - t0, 1), "s", flush=True)
    q2 = []
    for i in inc:
        c, r = cases[i], reid[i]
        corr = rl.chip_td_correction(c["fifo_chip"], c["control_chip"], chip_lower)
        crit = _corner_rows(c)
        worst_ms = max(crit, key=lambda x: x["Ms"])
        row = dict(case=i, log=c["log_id"][:8], axis="roll" if c["axis"] == 0 else "pitch", cls=c["cls"],
                   group="fails worst case" if i in fail else ("fails nominal" if i in nomfail else "passes"),
                   f_Ms_nominal=crit[0]["f_Ms"], f_Ms_worst=worst_ms["f_Ms"], variants={})
        for k, fit in r["fits"].items():
            v = dict(fit={kk: vv for kk, vv in fit.items() if kk not in ("f", "G", "coh", "sel")})
            if "b" in fit:
                ok, why = pv.accept(fit, c["cls"])
                v.update(accepted=ok, reasons=why)
                if ok:
                    rm = eval_cell(refit_case(c, fit, corr))
                    v.update(nominal=_r(rm["nominal"]), worst=_r(rm["worst"]), nominal_ok=nominal_ok(rm),
                             passes=rb.passes(rm, **separate_limits(MS_WC)),
                             band_reaches_f_Ms=bool(fit["f_used"][1] >= row["f_Ms_nominal"]))
            else:
                v.update(accepted=False, reasons=["no fit"])
            row["variants"][k] = v
        v0 = r["fits"]["V0"]
        row["V0_reproduces_stored_fit"] = bool("b" in v0 and abs(v0["b"] - c["plant"]["b"]) < 1e-6 * c["plant"]["b"]
                                               and abs(v0["tau_m"] - c["plant"]["tau_m"]) < 1e-9
                                               and abs(v0["Td"] - c["td_identified"]) < 1e-9)
        q2.append(row)
    if not all(x["V0_reproduces_stored_fit"] for x in q2):
        raise RuntimeError("V0 does not reproduce the stored identifications: " +
                           str([x["log"] for x in q2 if not x["V0_reproduces_stored_fit"]]))

    # Q1: the failing cases in detail
    q1 = []
    for i in fail:
        c, r = cases[i], reid[i]
        crit = _corner_rows(c)
        fr = r["frf"]["1"]
        q1.append(dict(case=i, log=c["log_id"][:8], axis="roll" if c["axis"] == 0 else "pitch", cls=c["cls"],
                       f_ctrl=c["f_ctrl"], fit_band=c["fit_band"], n_seg=fr["n_seg"],
                       coherent_05=_coherent(fr, 0.5, 20.0, 0.5), coherent_03_to_40=_coherent(fr, 0.5, 40.0, 0.3),
                       corners=crit, measured_vs_model=measured_vs_model(c, fr, r["Tc"]),
                       plant=c["plant"], td_identified=c["td_identified"], flags=c["flags"]))

    # Q3: Ms_wc sweep, real (included) and synthetic
    rows15 = [evals[i]["w0.15_d1"] for i in inc]
    synth, tries, _ = rl.synthetic_drones(pri["rate_rows"], rl.N_SYNTH, rl.SEED)
    ev_s = Parallel(n_jobs=6)(delayed(synth_cells)(d) for d in synth)
    q3 = dict(real=ms_wc_sweep(rows), real_w015=ms_wc_sweep(rows15),
              synthetic=ms_wc_sweep([e[KEY] for e in ev_s]), synthetic_w015=ms_wc_sweep([e["w0.15_d1"] for e in ev_s]))
    for m in (2.5, 3.0):
        stored = rlj["synthetic"]["pass_rates"][f"PM30_Ms2_wcMs{m:g}|{KEY}"]
        if abs(q3["synthetic"][m]["share_all"] - stored) > 1e-4:
            raise RuntimeError(f"synthetic drones differ from robust_limits.json at Ms_wc {m}")
    print("Q3 done", round(time.perf_counter() - t0, 1), "s", flush=True)

    # Q4: can any filter setting make the failing flown settings pass?
    targets = fail + nomfail
    feas = Parallel(n_jobs=6)(delayed(feasible_filters)(cases[i]) for i in targets)
    q4 = []
    for i, fz in zip(targets, feas):
        c = cases[i]
        q4.append(dict(case=i, log=c["log_id"][:8], axis="roll" if c["axis"] == 0 else "pitch",
                       group="fails worst case" if i in fail else "fails nominal", flown=dict(
                           f_gyro=c["settings"]["f_gyro"], f_dterm=c["settings"]["f_dterm"],
                           notches=c["settings"]["notches"], dnf_esc=c["settings"]["dnf_esc"],
                           nominal=_r(evals[i][KEY]["nominal"]), worst=_r(evals[i][KEY]["worst"])),
                       flags=c["flags"], options=fz))
    out = dict(settings=dict(PM_MIN=PM_MIN, MS_MAX=MS_MAX, GM_MIN=GM_MIN, WIDTH=WIDTH, DELAY=DELAY, KEY=KEY, MS_WC=MS_WC,
                             MS_SWEEP=MS_SWEEP, ID_VARIANTS=ID_VARIANTS, NOTCH_OPTIONS={k: list(v) for k, v in
                                                                                     NOTCH_OPTIONS.items()},
                             F_DIAG=F_DIAG, optimizer_ranges="optimizer.ranges(fs)"),
               cases=[_case_summary(c) for c in cases], included=inc, failing=fail, nominal_failing=nomfail,
               q1=q1, q2=q2, q3={k: {str(m): v for m, v in d.items()} for k, d in q3.items()},
               synthetic=dict(n=len(synth), tries=tries), q4=q4, seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(HERE, "a4_followup.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"failing {len(fail)}, nominal failing {len(nomfail)}; {out['seconds']} s")


if __name__ == "__main__":
    main()
