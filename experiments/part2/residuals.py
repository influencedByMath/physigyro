"""Where does the parametric model miss? (fit error ~3 dB vs Welch noise < 1 dB; plan Part 2b, Task 9)

    .venv/Scripts/python experiments/part2/residuals.py      (after fit_model.py)
Rebuilds every vehicle's fitted model from model_fits.json, computes the log10 residual (data - model) on the
native bins and summarises it per 48-band: median |residual| in dB and the median signed residual (bias).
Also estimates the Welch noise of each vehicle spectrum (chi-squared with 2 x averages degrees of freedom), so
the misfit can be compared with pure estimation noise. Writes residuals.json.
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
    edges = sl.band_edges(TH)
    abs_b, sig_b, noise_db = [], [], []
    for v in vs:
        e = fits["vehicles"][v["vehicle_key"]]
        m = (v["f"] >= BAND[0]) & (v["f"] <= BAND[1])
        f, P = v["f"][m], v["P"][m]
        lp = None if e["lowpass"] is None else tuple(e["lowpass"])
        S = nm.model_psd(f, np.array(e["params"]), "spline", BAND, e["motor_hz"], e["n_harm"], e["n_free"], lowpass=lp)
        r = np.log10(P) - np.log10(S)
        idx = np.digitize(f, edges) - 1
        abs_b.append([10 * np.median(np.abs(r[idx == b])) for b in range(len(edges) - 1)])
        sig_b.append([10 * np.median(r[idx == b]) for b in range(len(edges) - 1)])
        n_avg = df[df.vehicle_key == v["vehicle_key"]].n_averages.sum() * 2        # roll + pitch averaged
        dof = 2 * n_avg
        sim = np.log10(np.random.default_rng(0).chisquare(dof, 20000) / dof)
        noise_db.append(10 * np.sqrt(np.mean(sim ** 2)))
    abs_b, sig_b = np.array(abs_b), np.array(sig_b)
    centres = np.sqrt(edges[:-1] * edges[1:])
    out = dict(band_centres_hz=np.round(centres, 1).tolist(),
               median_abs_db_per_band=np.round(np.median(abs_b, 0), 2).tolist(),
               median_bias_db_per_band=np.round(np.median(sig_b, 0), 2).tolist(),
               welch_noise_rms_db=dict(median=round(float(np.median(noise_db)), 3), max=round(float(max(noise_db)), 3)),
               fit_rms_db=dict(median=round(float(np.median([fits["vehicles"][v["vehicle_key"]]["rms_db"] for v in vs])), 3)))
    with open(os.path.join(HERE, "residuals.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    worst = np.argsort(out["median_abs_db_per_band"])[::-1][:8]
    print("Welch noise rms (dB):", out["welch_noise_rms_db"], " fit rms:", out["fit_rms_db"])
    print("worst bands:", [(out["band_centres_hz"][i], out["median_abs_db_per_band"][i], out["median_bias_db_per_band"][i])
                           for i in worst])


if __name__ == "__main__":
    main()
