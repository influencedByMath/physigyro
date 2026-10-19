"""Task 3 RQ1 runner (THESIS 7.3, E3; D-08, D-15, D-36, D-37).

    .venv/Scripts/python experiments/phase4/rq1_run.py synthetic        (1,000 kept Tier 3 drones, generator spectra)
    .venv/Scripts/python experiments/phase4/rq1_run.py realgain fit     (fit-half real-gain cases)
    .venv/Scripts/python experiments/phase4/rq1_run.py realnoise fit    (52 fit vehicle spectra x 10 synthetic drones)
    .venv/Scripts/python experiments/phase4/rq1_run.py brute            (brute-force check, 20 synthetic cases)
    .venv/Scripts/python experiments/phase4/rq1_run.py pareto           (Mode C fronts of the 4 Fig. 4 drones)
    .venv/Scripts/python experiments/phase4/rq1_run.py realgain test    (test half: LAST, on committed code)
    .venv/Scripts/python experiments/phase4/rq1_run.py realnoise test

Every case: the reference setting (synthetic: PX4 v1.16 default, D-09; real: the pilot's flown setting, D-15), the
optimum of Mode A (best PM with N_rel <= 1) and Mode B (least N_rel with PM >= PM_ref) under the D-37 limits
(nominal limits on the nominal plant, D-36 worst-case limits on the 8 corners), searched on the fast grid and refined
and reported on the full grid (D-37 (4)). Scenario A: static notches at the two most prominent peaks of the spectrum;
Scenario B (motor frequencies known): the ESC-RPM notch bank. Real logs with RPM use the tracking-notch spectra of
rq1_cases.py (D-13); synthetic spectra the mean-RPM notch. Parts run in resumable chunks (seed and git hash stored).
"""
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import noise_generator as ng                # noqa: E402
import noise_model as nm                    # noqa: E402
import optimizer as opt                     # noqa: E402
import priors as pr                         # noqa: E402
import rq1_cases as rc                      # noqa: E402
import tier3 as t3                          # noqa: E402
from loop_model import FilterSettings, Gains, LoopCase, Plant, PX4_V116_MC_DEFAULT, Rates   # noqa: E402

SEED = 20261018
N_SYNTH = 1000                              # D-37 (3)
N_PER_SPECTRUM = 10                         # D-37 (3)
N_BRUTE = 20                                # D-37 (4)
F_HI = 780.0                                # D-23 Q5
F_SYN = np.arange(20.0, F_HI + 1e-9, 0.5)
N_MAX_FRONT = tuple(float(x) for x in np.round(np.geomspace(0.25, 2.0, 12), 4))   # Mode C noise limits
METRIC_KEYS = ("PM", "GM", "Ms", "fc", "DM", "stable", "tau_F", "tau_D", "N_rel")


# ------------------------------------------------------------------ helpers

def _num(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    x = float(x)
    return x if np.isfinite(x) else (None if np.isnan(x) else ("inf" if x > 0 else "-inf"))


def metrics_of(m):
    return {k: _num(m[k]) for k in METRIC_KEYS if k in m}


def robust_of_dict(rm):
    return dict(nominal={k: _num(v) for k, v in rm["nominal"].items()},
                worst_corners={k: _num(v) for k, v in rm["worst_corners"].items()})


def settings_dict(s):
    d = asdict(s)
    d["notches"] = [list(n) for n in d["notches"]]
    return d


def settings_of(d):
    return FilterSettings(**{k: (tuple(tuple(x) for x in v) if k == "notches" else v) for k, v in d.items()})


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


# ------------------------------------------------------------------ one case

def solve(plant, gains, rates, psd_f, psd, motor_hz, reference, base, scenario, delta_d, reference_S_eff=None,
          tracking=None, default=None, default_S_eff=None, n_g=opt.FAST["n_g"], n_d=opt.FAST["n_d"], brute=False,
          front=False):
    """Reference (and default) metrics and the Mode A / B optima of one case under the D-37 limits."""
    mh = None if motor_hz is None else np.asarray(motor_hz, float)
    full = LoopCase(plant, gains, rates, psd_f, psd, motor_hz=mh, f_hi=F_HI, reference=reference,
                    reference_S_eff=reference_S_eff)
    fast = LoopCase(plant, gains, rates, psd_f, psd, motor_hz=mh, f_hi=F_HI, reference=reference,
                    reference_S_eff=reference_S_eff, f_lo=full.f_lo, n_stab=opt.FAST["n_stab"],
                    noise_df=opt.FAST["noise_df"])
    spec = opt.Robust(t3.WIDTH, t3.WIDTH, delta_d, t3.worst_case_limits())
    peaks = []
    if scenario == "B":
        o_fast = opt.scenario_b_options(fast, base, rc.TRACK_BWS, rc.TRACK_HMCS, tracking)
        o_full = opt.scenario_b_options(full, base, rc.TRACK_BWS, rc.TRACK_HMCS, tracking)
    else:
        peaks = nm.detect_peaks(psd_f, psd, f_lo=full.f_lo, f_hi=min(F_HI, 0.45 * rates.fs), n=2)
        o_fast = o_full = opt.scenario_a_options(peaks)
    rm_ref = opt.robust_of(full, reference, spec)
    ref = full.ref_metrics
    nominal_ok = bool(ref["stable"] and ref["PM"] >= opt.LIMITS["PM_min"] and ref["Ms"] <= opt.LIMITS["Ms_max"]
                      and ref["GM"] >= opt.LIMITS["GM_min"])
    out = dict(f_lo=float(full.f_lo), peaks=peaks, n_options=len(o_full), reference=metrics_of(ref),
               reference_robust=robust_of_dict(rm_ref),
               reference_flags=dict(nominal_ok=nominal_ok, worst_ok=opt.corners_ok(rm_ref, spec.wc)))
    if default is not None:
        md = full.evaluate(default, S_eff=None if default_S_eff is None else full.psd_on_noise_grid(*default_S_eff))
        out["default"] = metrics_of(md)
        out["default_robust"] = robust_of_dict(opt.robust_of(full, default, spec))
    t = time.perf_counter()
    grid = opt.grid_search(fast, base, o_fast, n_g, n_d)
    for mode in ("A", "B"):
        r = opt.optimize_checked(full, base, o_fast, mode, robust=spec, coarse=fast, grid=grid, options_full=o_full,
                                 reference=reference)
        res = dict(feasible=r["feasible"], n_candidates=r["n_candidates"], n_feasible_coarse=r["n_feasible_coarse"])
        if r["feasible"]:
            res.update(start=r["start"])
            res.update(settings=settings_dict(r["settings"]), option=r["option"].label, metrics=metrics_of(r["metrics"]),
                       robust=robust_of_dict(r["robust"]))
            if brute:
                bf = opt.brute_force(full, base, r["option"], mode, robust=spec)
                res["brute"] = dict(feasible=bf["feasible"], n_candidates=bf["n_candidates"],
                                    **(dict(metrics=metrics_of(bf["metrics"]), f_gyro=bf["f_gyro"], f_dterm=bf["f_dterm"])
                                       if bf["feasible"] else {}))
        out[mode] = res
    out["seconds"] = round(time.perf_counter() - t, 2)
    if front:
        g_full = opt.grid_search(full, base, o_full, opt.FULL["n_g"], opt.FULL["n_d"])
        out["front"] = [dict(n_max=nm_, PM=pm, option=lab) for nm_, pm, lab in
                        opt.pareto_front(g_full, N_MAX_FRONT, accept=opt.robust_accept(full, base, spec))]
    return out


# ------------------------------------------------------------------ spectra

def build_generator():
    """The Phase 3 generator (D-26, D-27) fitted on the fit half, with the jitter of generator_check.json."""
    import pandas as pd
    import spectra_library as sl
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(pr.fit_path(spec, "fit_windows.csv"))
    native = np.load(pr.fit_path(spec, "fit_psd_native.npz"))
    fits = json.load(open(os.path.join(ROOT, "experiments", "part2", "model_fits.json")))
    gc = json.load(open(os.path.join(ROOT, "experiments", "part2", "generator_check.json")))
    entries = []
    for v in sl.vehicle_spectra(df, native):
        e = dict(fits["vehicles"][v["vehicle_key"]])
        e["lowpass"] = None if e["lowpass"] is None else tuple(e["lowpass"])
        entries.append(e)
    return ng.fit_generator(entries, (20.0, F_HI), gc["jitter"]["freq_sd"], gc["jitter"]["height_sd"]), entries


def generator_draw(gen, entries, f, sample=ng.sample):
    """draw(rng) for tier3.sample_drones: one generator spectrum; Scenario B motor frequencies = the source vehicle's
    mean RPM / 60 times the spectrum's frequency scale (hover f_m rule of D-34)."""
    def draw(rng):
        S, meta = sample(gen, 1, f, rng)
        m = meta[0]
        e = entries[m["source"]]
        mh = None
        if e.get("motor_hz") and e.get("n_harm", 0) > 0:
            mh = [float(x) * m["scale"] for x in e["motor_hz"]]
        return dict(chip=m["chip"], motor_hz=mh, S=S[0], meta=dict(source=int(m["source"]), scale=float(m["scale"])))
    return draw


def fixed_draw(spectrum):
    def draw(rng):
        return spectrum
    return draw


# ------------------------------------------------------------------ resumable parts

def run_part(items, work, out_path, meta, chunk=48, n_jobs=6, finish=True):
    """Run ``work`` on every item (dict with 'id') in chunks; finished results go to <out>.partial.jsonl first, so a
    restart skips them. With ``finish`` the full result file is written and the partial file removed."""
    part = out_path + ".partial.jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [it for it in items if it["id"] not in done]
    for k in range(0, len(todo), chunk):
        block = todo[k:k + chunk]
        res = (Parallel(n_jobs=n_jobs)(delayed(work)(it) for it in block) if n_jobs != 1 else [work(it) for it in block])
        with open(part, "a", encoding="utf-8") as fh:
            for r in res:
                fh.write(json.dumps(r) + "\n")
                done[r["id"]] = r
        print(f"  {len(done)}/{len(items)}", flush=True)
    out = dict(meta=meta, results=[done[it["id"]] for it in items if it["id"] in done])
    if finish:
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=1)
        os.remove(part) if os.path.exists(part) else None
    return out


# ------------------------------------------------------------------ the parts

def _synthetic_work(d):
    p = Plant(1.0, d["tau_m"], d["Td_nominal"])
    r = solve(p, Gains(**d["gains"]), Rates(d["fs"], d["f_ctrl"]), d["psd_f"], d["S"], d["motor_hz"],
              PX4_V116_MC_DEFAULT, PX4_V116_MC_DEFAULT, "B" if d["motor_hz"] else "A", d["delay_step"],
              tracking=d.get("tracking_fn"), brute=d.get("brute", False), front=d.get("front", False))
    keep = ("id", "part", "half", "vehicle", "f_c", "tau_m", "Td", "Td_nominal", "delay_step", "r", "fs", "f_ctrl",
            "pm_unfiltered", "chip", "chip_delay", "motor_hz", "flagged", "meta")
    return dict({k: d.get(k) for k in keep}, scenario="B" if d["motor_hz"] else "A", **r)


def synthetic_drones(seed=SEED, n=N_SYNTH):
    pri = json.load(open(pr.fit_path(HERE, "priors_from_logs.json")))
    gen, entries = build_generator()
    delays = t3.chip_delays(pri["chip_delay"], gen["lowpass"])
    res = t3.sample_drones(pri["rate_rows"], n, seed, generator_draw(gen, entries, F_SYN), delays)
    for i, d in enumerate(res["drones"]):
        d.update(id=f"syn{i:04d}", part="synthetic", half="fit", psd_f=F_SYN)
    return res, delays


def figure_drones(synth, real, scenario_b_id="fit|b27d09e3|roll"):
    """The 4 Fig. 4 drones: the synthetic drone with the median Mode A PM gain, the same inside 8-10 Hz, the real
    Scenario B drone flown with the ESC bank on, and the real case flagged only in the worst case with the largest
    worst-corner Ms at its flown setting."""
    def median_of(rows):
        rows = [s for s in rows if s["A"]["feasible"]]
        v = np.median([s["A"]["dpm"] for s in rows])
        return min(rows, key=lambda s: (abs(s["A"]["dpm"] - v), s["id"]))["id"]
    pick = {"synthetic median": median_of(synth),
            "synthetic 8-10 Hz": median_of([s for s in synth if 8.0 <= s["f_c"] < 10.0])}
    pick["Scenario B real"] = scenario_b_id
    flagged = [r for r in real if r["flag"] == "worst case"]
    pick["flagged real"] = max(flagged, key=lambda r: r["worst_corner_ms"])["id"]
    return pick


def realgain_items(half):
    pre = "fit" if half == "fit" else "test"
    path = os.path.join(HERE, f"{pre}_rq1_cases.json")
    meta = json.load(open(pr.fit_path(path) if half == "fit" else path))
    arr = np.load(path[:-5] + ".npz")
    items = []
    for c in meta["cases"]:
        tr = {k: (arr[c["id"] + "|f"], arr[c["id"] + "|track|" + k]) for k in c["tracking_keys"]}
        items.append(dict(c, f=arr[c["id"] + "|f"], P=arr[c["id"] + "|P"], tracking=tr, part="realgain"))
    return items


def _realgain_work(c):
    tr = c["tracking"]

    def tracking(bw, hmc, fmin):
        return tr[rc.track_key(bw, hmc, fmin)]
    actual, default = settings_of(c["actual"]), settings_of(c["default"])
    ref_S = default_S = None
    if c["scenario"] == "B" and tr:
        if actual.dnf_esc:
            ref_S = tracking(actual.dnf_bw, actual.dnf_hmc, actual.dnf_min)
        if default.dnf_esc:
            k = rc.track_key(default.dnf_bw, default.dnf_hmc, default.dnf_min)
            default_S = tr.get(k)
    plant = Plant(**c["plant"])
    r = solve(plant, Gains(**c["gains"]), Rates(c["fs"], c["f_ctrl"]), c["f"], c["P"], c["motor_hz"], actual, actual,
              c["scenario"], t3.DELAY_SAMPLES / c["f_ctrl"], reference_S_eff=ref_S,
              tracking=tracking if (c["scenario"] == "B" and tr) else None,
              default=default if c["defaults_known"] else None, default_S_eff=default_S, front=c.get("front", False))
    keep = ("id", "part", "half", "log_id", "axis_name", "vehicle_key", "scenario", "rpm_tracking", "fc_unfiltered",
            "pm_unfiltered", "fifo_chip", "control_chip", "flags", "defaults_known", "f_ctrl", "fs", "plant", "actual",
            "default", "motor_hz")
    return dict({k: c.get(k) for k in keep}, **r)


def realnoise_items(half, seed=SEED):
    """Each vehicle spectrum of one half with N_PER_SPECTRUM synthetic Tier 3 drones (chip and RPM of the vehicle)."""
    import pandas as pd
    import spectra_library as sl
    spec = os.path.join(ROOT, "data", "spectra")
    win = os.path.join(spec, f"{half}_windows.csv")
    nat = os.path.join(spec, f"{half}_psd_native.npz")
    df = pd.read_csv(pr.fit_path(win) if half == "fit" else win)
    native = np.load(pr.fit_path(nat) if half == "fit" else nat)
    rows = list(csv.DictReader(open(win)))
    pri = json.load(open(pr.fit_path(HERE, "priors_from_logs.json")))
    gen, _ = build_generator()
    delays = t3.chip_delays(pri["chip_delay"], gen["lowpass"])
    items = []
    for vi, v in enumerate(sl.vehicle_spectra(df, native)):
        vrows = [w for w in rows if w["vehicle_key"] == v["vehicle_key"]]
        tracking = None
        if v["motor_hz"] is not None and all(rc._true(w["rpm_tracking"]) for w in vrows):
            tracking = checked_vehicle_tracking(vrows, v)
        draw = fixed_draw(dict(chip=v["chip"], motor_hz=v["motor_hz"], S=v["P"], meta=dict(vehicle=v["vehicle_key"])))
        res = t3.sample_drones(pri["rate_rows"], N_PER_SPECTRUM, seed + 1000 * (vi + 1) + (0 if half == "fit" else 7),
                               draw, delays)
        for j, d in enumerate(res["drones"]):
            d.update(id=f"{half}|veh{vi:03d}|{j}", part="realnoise", half=half, vehicle=v["vehicle_key"], psd_f=v["f"],
                     tracking_fn=None if tracking is None else _TrackFn(tracking))
            items.append(d)
    return items


class _TrackFn:
    """Picklable tracking(bw, hmc, fmin) lookup for joblib."""
    def __init__(self, table):
        self.table = table

    def __call__(self, bw, hmc, fmin):
        return self.table[rc.track_key(bw, hmc, fmin)]


def vehicle_tracking(vrows, motor_hz):
    """Tracking spectra of a vehicle spectrum (roll and pitch power-averaged, D-24 (2)) for every Scenario B option
    of the PX4 default (dnf_min 25 Hz); every window comes from one log."""
    import noise_data as nd
    log_id = vrows[0]["log_id"]
    u = nd.load(pr.fit_path(ROOT, "data", "raw_logs", log_id + ".ulg"))
    gs = nd.gyro_stream(u)
    nper = int(vrows[0]["nperseg"])

    def psd_fn(t0, t1, bw, hmc, fmin):
        f, P, _ = nd.tracking_notch_psd(u, gs, t0, t1, bw, hmc, fmin, axes=(0, 1), nperseg=nper)
        return f, P.mean(1)
    st = [(bw, h, PX4_V116_MC_DEFAULT.dnf_min) for h in rc.TRACK_HMCS for bw in rc.TRACK_BWS]
    return rc.tracking_spectra(vrows, st, psd_fn), float(gs.fs)


def checked_vehicle_tracking(vrows, v):
    """vehicle_tracking with the S2 consistency check against the library vehicle spectrum (same samples and axes)."""
    tr, fs = vehicle_tracking(vrows, v["motor_hz"])
    for k, (ft, Pt) in tr.items():
        bw, hmc, fmin = k.split("|")
        if not rc.tracking_consistent(v["f"], v["P"], ft, Pt, v["motor_hz"], bw, hmc, fmin, fs):
            raise RuntimeError(f"{v['vehicle_key']}: tracking spectrum {k} does not match the library vehicle spectrum")
    return tr


def main(part, half="fit"):
    t0 = time.perf_counter()
    meta = dict(part=part, half=half, seed=SEED, git=git_hash(), limits=t3.limits(), width=t3.WIDTH,
                delay_samples=t3.DELAY_SAMPLES, fast=opt.FAST, full=opt.FULL, f_hi=F_HI)
    if part == "synthetic":
        res, delays = synthetic_drones()
        meta.update(draws=res["draws"], in_window=res["in_window"], kept=res["kept"], chip_delays=delays,
                    flagged=int(sum(d["flagged"] for d in res["drones"])))
        run_part(res["drones"], _synthetic_work, os.path.join(HERE, "rq1_synthetic.json"), meta)
    elif part == "brute":
        res, _ = synthetic_drones()
        rng = np.random.default_rng(SEED + 5)
        pick = sorted(rng.choice(len(res["drones"]), N_BRUTE, replace=False).tolist())
        items = [dict(res["drones"][i], brute=True) for i in pick]
        run_part(items, _synthetic_work, os.path.join(HERE, "rq1_brute.json"), dict(meta, picked=pick), chunk=N_BRUTE)
    elif part == "realgain":
        name = ("fit" if half == "fit" else "test") + "_rq1_realgain.json"
        run_part(realgain_items(half), _realgain_work, os.path.join(HERE, name), meta, chunk=12)
    elif part == "realnoise":
        name = ("fit" if half == "fit" else "test") + "_rq1_realnoise.json"
        run_part(realnoise_items(half), _synthetic_work, os.path.join(HERE, name), meta)
    elif part == "pareto":
        syn = json.load(open(pr.fit_path(HERE, "rq1_synthetic.json")))["results"]
        rg = json.load(open(pr.fit_path(HERE, "fit_rq1_realgain.json")))["results"]
        for s in syn:
            s["A"]["dpm"] = (s["A"]["metrics"]["PM"] - s["reference"]["PM"]) if s["A"]["feasible"] else None
        real = [dict(id=r["id"], worst_corner_ms=r["reference_robust"]["worst_corners"]["Ms"],
                     flag=("nominal" if not r["reference_flags"]["nominal_ok"] else
                           ("worst case" if not r["reference_flags"]["worst_ok"] else "none"))) for r in rg]
        pick = figure_drones(syn, real)
        res, _ = synthetic_drones()
        by_id = {d["id"]: d for d in res["drones"]}
        items = [dict(by_id[pick["synthetic median"]], front=True, role="synthetic median"),
                 dict(by_id[pick["synthetic 8-10 Hz"]], front=True, role="synthetic 8-10 Hz")]
        rgi = {c["id"]: c for c in realgain_items("fit")}
        items_r = [dict(rgi[pick["Scenario B real"]], front=True, role="Scenario B real"),
                   dict(rgi[pick["flagged real"]], front=True, role="flagged real")]
        out_s = [dict(_synthetic_work(it), role=it["role"]) for it in items]
        out_r = [dict(_realgain_work(it), role=it["role"]) for it in items_r]
        with open(os.path.join(HERE, "rq1_pareto.json"), "w") as fh:
            json.dump(dict(meta=dict(meta, pick=pick, n_max=N_MAX_FRONT), results=out_s + out_r), fh, indent=1)
    else:
        raise ValueError(part)
    print(f"{part} {half}: done in {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["fit"]))
