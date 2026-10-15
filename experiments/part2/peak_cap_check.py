"""Does the free-peak cap (max_free = 6, a design choice of fit_model.py, not from THESIS) limit the fit?

    .venv/Scripts/python experiments/part2/peak_cap_check.py
Refits every vehicle spectrum with its chip's chosen low-pass and max_free 6, 10, 15 (prominence 0.5), and reports
the fit error, the BIC per vehicle and how many fits hit the cap. Writes peak_cap_check.json.
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
import spectra_library as sl       # noqa: E402

TH = dq.FROZEN
BAND = (TH.f_min, TH.f_max)


def main():
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    vs = sl.vehicle_spectra(df, np.load(os.path.join(spec, "fit_psd_native.npz")))
    fits = json.load(open(os.path.join(HERE, "model_fits.json")))
    out = {}
    for cap in (6, 10, 15):
        rms, bic, at_cap, nfree = [], [], 0, []
        for v in vs:
            lp = fits["chips"][v["chip"]]["lowpass"]
            r = nm.fit_spectrum(v["f"], v["P"], BAND, floor="spline", motor_hz=v["motor_hz"], prominence=0.5,
                                lowpass=None if lp is None else tuple(lp), max_free=cap)
            n, k = r["n_bins"], r["n_params"]
            rss = (r["rms_db"] / 10) ** 2 * n
            rms.append(r["rms_db"])
            bic.append(n * np.log(rss / n) + k * np.log(n))
            at_cap += r["n_free"] == cap
            nfree.append(r["n_free"])
        out[str(cap)] = dict(rms_median=round(float(np.median(rms)), 3), rms_p90=round(float(np.percentile(rms, 90)), 3),
                             bic=bic, at_cap=int(at_cap), n_free_median=float(np.median(nfree)))
        print(cap, {k: v for k, v in out[str(cap)].items() if k != "bic"}, flush=True)
    for a, b in (("6", "10"), ("10", "15")):
        d = np.array(out[b]["bic"]) - np.array(out[a]["bic"])
        out[f"bic_{b}_minus_{a}"] = dict(median=round(float(np.median(d)), 1), share_better=round(float(np.mean(d < -10)), 3))
        print(f"BIC {b} - {a}: median {np.median(d):.1f}, share of vehicles better by >10: {np.mean(d < -10):.2f}")
    with open(os.path.join(HERE, "peak_cap_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
