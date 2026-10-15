"""Parametric noise-model fits on the fit-half library (plan Task 3 evidence C and Task 5).

    .venv/Scripts/python experiments/part2/fit_model.py            # all floors x prominences (evidence)
Writes fits.csv next to this file: one row per window x axis (roll, pitch) x floor x prominence, with the fit
error (RMS of log10 residual, dB), number of parameters and BIC. Motor frequencies are fixed to RPM/60
where the window has valid ESC RPM (Section 7.4.5 use 1). Nothing is chosen here (author decision D-24).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data_quality as dq          # noqa: E402
import noise_model as nm           # noqa: E402

TH = dq.FROZEN
BAND = (TH.f_min, TH.f_max)
PROMINENCES = (0.3, 0.5, 1.0)


def main():
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    z = np.load(os.path.join(spec, "fit_psd_native.npz"))
    rows = []
    for w in df.itertuples():
        f, P = z[f"{w.window_id}|f"].astype(float), z[f"{w.window_id}|P"].astype(float)
        motors = json.loads(w.rpm_hz) if w.rpm_valid else None
        for ax, name in ((0, "roll"), (1, "pitch")):
            for floor in nm.FLOORS:
                for prom in PROMINENCES:
                    try:
                        r = nm.fit_spectrum(f, P[:, ax], BAND, floor=floor, motor_hz=motors, prominence=prom)
                    except Exception as e:                # report, never hide
                        rows.append(dict(window_id=w.window_id, axis=name, floor=floor, prominence=prom,
                                         error=repr(e)[:200]))
                        continue
                    n, k = r["n_bins"], r["n_params"]
                    rss = (r["rms_db"] / 10) ** 2 * n
                    rows.append(dict(window_id=w.window_id, vehicle_key=w.vehicle_key, axis=name, floor=floor,
                                     prominence=prom, rms_db=r["rms_db"], n_params=k, n_bins=n,
                                     bic=n * np.log(rss / n) + k * np.log(n), n_free=r["n_free"],
                                     n_harm=r["n_harm"], has_rpm=motors is not None, success=r["success"],
                                     peaks_hz=json.dumps([round(x, 1) for x in r["peaks_hz"]]),
                                     params=json.dumps([round(float(x), 5) for x in r["params"]])))
        print(w.window_id[:11], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "fits.csv"), index=False)
    ok = out[out.get("error").isna()] if "error" in out else out
    print(ok.groupby(["floor", "prominence"])["rms_db"].describe(percentiles=[0.5, 0.9]).round(3).to_string())


if __name__ == "__main__":
    main()
