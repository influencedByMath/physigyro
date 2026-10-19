"""Real-gain cases for Task 3 (RQ1; THESIS 7.1.1 Tier 3 (b), D-15, D-36 (6), D-37 (2)).

    .venv/Scripts/python experiments/phase4/rq1_cases.py fit
    .venv/Scripts/python experiments/phase4/rq1_cases.py test      (S2: opens the test half; frozen identification)

A case is one (log, axis) with an accepted Tier 2 identification AND steady library windows from the SAME log:
- fit half: the 30 cases rebuilt and reproduced in a4_followup.json (identification of plant_variation.py);
- test half: plant_variation.analyze on the test logs that are logged at full rate and have library windows, then
  robust_limits.real_cases (the same accept gates, frozen in D-36, the same FIFO -> control chip correction);
- cases whose loop parameters changed in flight are dropped (as in the evidence pack).
Per case: the identified plant, the flown gains and setting (the reference, D-15), the log's default setting
(D-09), the noise spectrum = power mean of the case's windows on its own axis, the window motor frequencies
(Scenario B when every window has valid RPM) and, for Scenario B windows that pass the D-13 RPM-rate gate, the
tracking-notch spectra for every ESC-bank setting the optimizer can choose and for the flown one.
Writes {half}_rq1_cases.json (+ .npz with the spectra) next to this file; test-half files start with test_.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import noise_data as nd                     # noqa: E402
import priors as pr                         # noqa: E402
from loop_model import (_stability_grid, derivative_response, integrator_response, margins,   # noqa: E402
                        n_integrators, plant_response, zoh_response, Gains, Plant, Rates)

TRACK_BWS = (5.0, 7.5, 10.0, 15.0, 20.0, 30.0)     # optimizer.scenario_b_options defaults (THESIS 7.3: 5-30 Hz)
TRACK_HMCS = (1, 2, 3)


def _true(x):
    return str(x).strip().lower() in ("true", "1")


def windows_by_log(rows):
    out = {}
    for w in rows:
        out.setdefault(w["log_id"], []).append(w)
    return out


def case_spectrum(rows, native, axis):
    """(f, power mean over the windows of the PSD on ``axis``); all windows must share one grid."""
    f = np.asarray(native[rows[0]["window_id"] + "|f"], float)
    Ps = []
    for w in rows:
        if not np.array_equal(np.asarray(native[w["window_id"] + "|f"], float), f):
            raise ValueError(f"{w['window_id']}: windows on different frequency grids")
        Ps.append(np.asarray(native[w["window_id"] + "|P"], float)[:, axis])
    return f, np.mean(Ps, 0)


def window_rpm(rows):
    """Mean motor frequencies [Hz] over the windows if every window has valid ESC RPM, else None."""
    if not all(_true(w["rpm_valid"]) for w in rows):
        return None
    return np.mean([json.loads(w["rpm_hz"]) for w in rows], 0).tolist()


def unfiltered_crossover(plant, gains, rates):
    """(crossover [Hz], PM [deg]) of the loop without any gyro or D-term filter (as the Tier 3a design)."""
    f = _stability_grid(rates)
    L = ((gains.Kp + gains.Ki * integrator_response(f, rates.f_ctrl) + gains.Kd * derivative_response(f, rates.fs))
         * zoh_response(f, rates.f_ctrl) * plant_response(f, plant))
    m = margins(f, L, n_integrators(gains))
    return float(m["fc"]), float(m["PM"])


def build_cases(cases, rows, native, logs, half):
    """RQ1 real-gain cases from identification cases; (cases, excluded with reasons)."""
    wl = windows_by_log(rows)
    recs = {r["log_id"]: r for r in logs}
    out, excluded = [], []
    for c in cases:
        key = dict(log_id=c["log_id"], axis=c["axis"])
        if c.get("exclude"):
            excluded.append(dict(key, reason="loop parameters changed in flight"))
            continue
        ws = wl.get(c["log_id"])
        if not ws:
            excluded.append(dict(key, reason="no steady window in the identified log"))
            continue
        rec = recs[c["log_id"]]
        f, P = case_spectrum(ws, native, c["axis"])
        rpm = window_rpm(ws)
        plant, g, rates = Plant(**c["plant"]), Gains(**c["gains"]), Rates(float(c["fs"]), float(c["f_ctrl"]))
        fcu, pmu = unfiltered_crossover(plant, g, rates)
        axis = "roll" if c["axis"] == 0 else "pitch"
        out.append(dict(id=f"{half}|{c['log_id'][:8]}|{axis}", half=half, log_id=c["log_id"], axis=c["axis"],
                        axis_name=axis, vehicle_key=rec["vehicle_key"], cls=c.get("cls"), plant=c["plant"],
                        gains=c["gains"], f_ctrl=float(c["f_ctrl"]), fs=float(c["fs"]), actual=c["settings"],
                        default=rec["fc"]["default"], defaults_known=bool(rec["fc"].get("defaults_known")),
                        motor_hz=rpm if rpm is not None else c.get("motor_hz"), scenario="B" if rpm is not None else "A",
                        rpm_tracking=bool(rpm is not None and all(_true(w["rpm_tracking"]) for w in ws)),
                        fifo_chip=c.get("fifo_chip"), control_chip=c.get("control_chip"), flags=list(c.get("flags") or []),
                        fc_unfiltered=fcu, pm_unfiltered=pmu, nperseg=rec.get("nperseg"),
                        window_ids=[w["window_id"] for w in ws], windows=[(float(w["t0"]), float(w["t1"])) for w in ws],
                        f=f, P=P))
    return out, excluded


def track_key(bw, hmc, fmin):
    return f"{float(bw):g}|{int(hmc)}|{float(fmin):g}"


def tracking_spectra(rows, settings, psd_fn):
    """{track_key: (f, power mean over the windows)} for every (bw, hmc, fmin); psd_fn(t0, t1, bw, hmc, fmin)
    returns (f, P) of one window after the tracking ESC-RPM notch bank (noise_data.tracking_notch_psd)."""
    out = {}
    for bw, hmc, fmin in settings:
        Ps, f = [], None
        for w in rows:
            f, P = psd_fn(float(w["t0"]), float(w["t1"]), bw, hmc, fmin)
            Ps.append(np.asarray(P, float))
        out[track_key(bw, hmc, fmin)] = (f, np.mean(Ps, 0))
    return out


def tracking_consistent(f, P, f_track, P_track, motor_hz, bw, hmc, fmin, fs, f_max=700.0, tol=0.02):
    """True if both spectra lie on the same grid (library grids are stored as float32) and, at the bins farther than 3 bandwidths from every notch centre (20 Hz to f_max), the tracking spectrum
    equals the library spectrum times the static bank response at the mean motor frequencies (median ratio within
    ``tol``): then both spectra use the same samples and axis."""
    from filters import esc_notch_frequencies, notch_response
    f = np.asarray(f, float)
    if np.shape(f_track) != f.shape or not np.allclose(f_track, f, rtol=1e-6, atol=1e-3):
        return False
    c = esc_notch_frequencies(np.asarray(motor_hz, float), int(hmc), float(bw), float(fmin))
    H2 = np.prod([np.abs(notch_response(f, x, float(bw), fs)) ** 2 for x in c], 0)
    far = (f >= 20.0) & (f <= f_max) & (np.min(np.abs(f[:, None] - c[None, :]), 1) > 3 * float(bw))
    if not far.any():
        return False
    return bool(abs(np.median(np.asarray(P_track, float)[far] / (np.asarray(P, float)[far] * H2[far])) - 1) <= tol)


def tracking_settings(case):
    """ESC-bank settings that need a tracking spectrum: every optimizer option (base dnf_min) and the flown one."""
    fmin = float(case["actual"]["dnf_min"])
    st = [(bw, h, fmin) for h in TRACK_HMCS for bw in TRACK_BWS]
    a = case["actual"]
    if a["dnf_esc"] and (float(a["dnf_bw"]), int(a["dnf_hmc"]), fmin) not in st:
        st.append((float(a["dnf_bw"]), int(a["dnf_hmc"]), fmin))
    return st


def log_tracking(case, rows):
    """Tracking spectra of one Scenario B case from its raw log (D-13), on the case axis, with the log's nperseg."""
    u = nd.load(pr.fit_path(ROOT, "data", "raw_logs", case["log_id"] + ".ulg"))
    gs = nd.gyro_stream(u)
    case["fs_fifo"] = float(gs.fs)                 # the rate the firmware notches run at

    def psd_fn(t0, t1, bw, hmc, fmin):
        f, P, _ = nd.tracking_notch_psd(u, gs, t0, t1, bw, hmc, fmin, axes=(case["axis"],), nperseg=case["nperseg"])
        return f, P[:, 0]
    return tracking_spectra(rows, tracking_settings(case), psd_fn)


def test_candidates(records, rows):
    """Test-half logs that can give a case: logged at full rate (priors.full_rate_logged) and with library windows."""
    have = set(windows_by_log(rows))
    return [r for r in records if pr.full_rate_logged(r["fc"]) and r["log_id"] in have]


def save(half, cases, excluded, extra, seconds):
    arrays, meta = {}, []
    for c in cases:
        arrays[c["id"] + "|f"], arrays[c["id"] + "|P"] = np.asarray(c["f"], float), np.asarray(c["P"], float)
        for k, (f, P) in (c.get("tracking") or {}).items():
            arrays[c["id"] + "|track|" + k] = np.asarray(P, float)
        meta.append({k: v for k, v in c.items() if k not in ("f", "P", "tracking")}
                    | dict(tracking_keys=sorted(c.get("tracking") or {})))
    np.savez_compressed(os.path.join(HERE, f"{half}_rq1_cases.npz"), **arrays)
    with open(os.path.join(HERE, f"{half}_rq1_cases.json"), "w") as fh:
        json.dump(dict(half=half, cases=meta, excluded=excluded, seconds=seconds, **extra), fh, indent=1)


def main(half):
    t0 = time.perf_counter()
    import robust_limits as rl
    spec = os.path.join(ROOT, "data", "spectra")
    if half == "fit":
        afj = json.load(open(pr.fit_path(HERE, "a4_followup.json")))
        cases = afj["cases"]
        logs = json.load(open(pr.fit_path(spec, "fit_logs.json")))
        rows = list(csv.DictReader(open(pr.fit_path(spec, "fit_windows.csv"))))
        native = np.load(pr.fit_path(spec, "fit_psd_native.npz"))
        extra = dict(source="a4_followup.json (cases rebuilt from the logs and reproduced)")
    elif half == "test":                       # evaluation only (D-37 (2)): the frozen pipeline, nothing tuned
        import plant_variation as pv
        logs = json.load(open(os.path.join(spec, "test_logs.json")))
        rows = list(csv.DictReader(open(os.path.join(spec, "test_windows.csv"))))
        native = np.load(os.path.join(spec, "test_psd_native.npz"))
        todo = test_candidates(logs, rows)
        idrows = Parallel(n_jobs=4, verbose=5)(delayed(pv.analyze)(r) for r in todo)
        idrows = [x for grp in idrows for x in grp]
        with open(os.path.join(HERE, "test_rq1_identification.json"), "w") as fh:
            json.dump(dict(n_logs=len(todo), rows=idrows), fh, indent=1)
        pri = json.load(open(pr.fit_path(HERE, "priors_from_logs.json")))
        cases = rl.real_cases(dict(rows=idrows), logs, rl.chip_lower_bounds(pri))
        extra = dict(source="plant_variation.analyze + robust_limits.real_cases on the test half",
                     n_candidate_logs=len(todo))
    else:
        raise ValueError("half must be 'fit' or 'test'")
    out, excluded = build_cases(cases, rows, native, logs, half)
    wl = windows_by_log(rows)
    for c in out:
        if c["scenario"] == "B" and c["rpm_tracking"]:
            c["tracking"] = log_tracking(c, wl[c["log_id"]])
            for k, (ft, Pt) in c["tracking"].items():
                bw, hmc, fmin = k.split("|")
                if not tracking_consistent(c["f"], c["P"], ft, Pt, c["motor_hz"], bw, hmc, fmin, c["fs_fifo"]):
                    raise RuntimeError(f"{c['id']}: tracking spectrum {k} does not match the library window spectrum")
            print("tracking spectra", c["id"], len(c["tracking"]), "consistent", flush=True)
    save(half, out, excluded, extra, round(time.perf_counter() - t0, 1))
    print(f"{half}: {len(out)} cases, {len(excluded)} excluded, {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1])
