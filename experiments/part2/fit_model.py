"""Parametric noise model on the fit-half library (plan Part 2b, Task 9; THESIS 7.5; D-24, D-25).

    .venv/Scripts/python experiments/part2/fit_model.py
One spectrum per vehicle (D-24 (2)); 4-knot log-log spline floor (D-24 (3)); motor frequencies fixed to RPM/60
where every window has valid ESC RPM; one low-pass per IMU chip model, chosen by BIC over a grid (D-24 (1),
P1), used for the MAGNITUDE only - its corner and order are not identifiable (D-25).
Floor fixed to the robust envelope, BIC gain >= 10 for a chip term, fit error > 5 dB dropped (D-26).
Writes model_fits.json next to this file: per chip the chosen low-pass and the BIC table; per vehicle the
fitted parameters (enough to rebuild the model), the fit error and the prominence sensitivity.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data_quality as dq          # noqa: E402
import noise_model as nm           # noqa: E402
import spectra_library as sl       # noqa: E402

TH = dq.FROZEN
BAND = (TH.f_min, TH.f_max)
FLOOR = "spline"
PROMINENCE = 0.5                   # THESIS 7.4.4 candidate; sensitivity at 0.3 and 1.0 is reported
MAX_FREE = 15                      # not binding (peak_cap_check.py: 2 of 52 vehicles reach 15; 42 reached 6)
FIX_FLOOR = True                   # D-26 (1): robust lower-envelope floor, peaks fitted on top
MIN_BIC_GAIN = 10.0                # D-26 (3): a chip gets a low-pass term only for a BIC gain >= 10
DROP_DB = 5.0                      # D-26 (4): spectra with a fit error above 5 dB are dropped
GRID = [(float(round(fc)), float(n)) for fc in np.geomspace(150, 1500, 10) for n in (1, 2, 3, 4, 6)]


def main():
    t0 = time.perf_counter()
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    native = np.load(os.path.join(spec, "fit_psd_native.npz"))
    vs = sl.vehicle_spectra(df, native)
    out = dict(floor=FLOOR, fix_floor=FIX_FLOOR, prominence=PROMINENCE, max_free=MAX_FREE, min_bic_gain=MIN_BIC_GAIN,
               drop_db=DROP_DB, grid=GRID, chips={}, vehicles={})
    for chip in sorted({v["chip"] for v in vs}):
        group = [v for v in vs if v["chip"] == chip]
        res = nm.fit_chip_lowpass([(v["f"], v["P"], v["motor_hz"]) for v in group], GRID, floor=FLOOR,
                                  prominence=PROMINENCE, band=BAND, max_free=MAX_FREE, fix_floor=FIX_FLOOR,
                                  min_bic_gain=MIN_BIC_GAIN)
        lp = res["lowpass"]
        ranked = sorted(res["bic"].items(), key=lambda kv: kv[1])
        out["chips"][chip] = dict(n_vehicles=len(group), lowpass=lp, bic_gain=res["bic_gain"],
                                  best_term=res["best_term"], bic_best5=ranked[:5],
                                  bic_none=res["bic"]["None"],
                                  gain_db_at=None if lp is None else {
                                      str(f): round(float(10 * np.log10(nm.lowpass_gain2(f, lp))), 2)
                                      for f in (100, 200, 300, 400, 500, 600, 780)})
        for v, r in zip(group, res["fits"]):
            sens = {}
            for prom in (0.3, 1.0):
                rr = nm.fit_spectrum(v["f"], v["P"], BAND, floor=FLOOR, motor_hz=v["motor_hz"], prominence=prom,
                                     lowpass=lp, max_free=MAX_FREE, fix_floor=FIX_FLOOR)
                sens[str(prom)] = dict(rms_db=round(rr["rms_db"], 3), n_free=rr["n_free"])
            out["vehicles"][v["vehicle_key"]] = dict(
                chip=chip, n_windows=v["n_windows"], lowpass=lp, motor_hz=v["motor_hz"], n_harm=r["n_harm"],
                n_free=r["n_free"], params=[float(x) for x in r["params"]], rms_db=r["rms_db"],
                success=r["success"], nfev=r["nfev"], peaks_hz=r["peaks_hz"], sensitivity=sens,
                dropped=bool(r["rms_db"] > DROP_DB))
        print(f"{chip:10s} n={len(group):2d} lowpass={lp} best term={res['best_term']} BIC gain={res['bic_gain']:.0f}",
              flush=True)
    rms = np.array([v["rms_db"] for v in out["vehicles"].values()])
    out["rms_db_quantiles"] = {str(q): round(float(np.quantile(rms, q)), 3) for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)}
    out["n_not_converged"] = int(sum(not v["success"] for v in out["vehicles"].values()))
    out["n_dropped"] = int(sum(v["dropped"] for v in out["vehicles"].values()))
    out["seconds"] = round(time.perf_counter() - t0, 1)
    with open(os.path.join(HERE, "model_fits.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("fit error quantiles (dB):", out["rms_db_quantiles"], "not converged:", out["n_not_converged"],
          "dropped (> 5 dB):", out["n_dropped"])


if __name__ == "__main__":
    main()
