"""Free vs fixed (robust lower-envelope) floor: fit error, peak-to-floor ratios, generator realism.

    .venv/Scripts/python experiments/part2/floor_mode_check.py     (after fit_model.py)
The free spline floor is not identified against broad Lorentzian tails (coefficients run to the -20 bound,
peak/floor ratios up to 10^20), so a generator on those parameters samples nonsense. The fixed-floor variant
takes the floor as the robust 20th-percentile envelope (as in evidence C) and fits only the peaks on top. Both
use the chip low-pass terms of model_fits.json. Writes floor_mode_check.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import data_quality as dq          # noqa: E402
import noise_generator as ng       # noqa: E402
import noise_model as nm           # noqa: E402
import spectra_library as sl       # noqa: E402
from generator_check import auc    # noqa: E402

TH = dq.FROZEN
BAND = (TH.f_min, TH.f_max)
SEED = 20261015


def main():
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    vs = sl.vehicle_spectra(df, np.load(os.path.join(spec, "fit_psd_native.npz")))
    fits = json.load(open(os.path.join(HERE, "model_fits.json")))
    edges = sl.band_edges(TH)
    real = np.array([sl.band_features(v["f"], v["P"][:, None], edges)[:, 0] for v in vs])
    out = {}
    for mode in ("free", "fixed"):
        entries, rms = [], []
        for v in vs:
            lp = fits["chips"][v["chip"]]["lowpass"]
            lp = None if lp is None else tuple(lp)
            r = nm.fit_spectrum(v["f"], v["P"], BAND, floor="spline", motor_hz=v["motor_hz"], prominence=0.5,
                                lowpass=lp, max_free=15, fix_floor=(mode == "fixed"))
            rms.append(r["rms_db"])
            entries.append(dict(chip=v["chip"], lowpass=lp, motor_hz=v["motor_hz"], n_harm=r["n_harm"],
                                n_free=r["n_free"], params=list(r["params"])))
        ratios = np.array([p[1] for e in entries for p in ng.entry_peaks(e, BAND)])
        gen = ng.fit_generator(entries, BAND)
        f = np.arange(BAND[0], BAND[1] + 1e-9, 0.5)
        rng = np.random.default_rng(SEED)
        S, _ = ng.sample(gen, 2000, f, rng)
        syn = np.array([sl.band_features(f, s[:, None], edges)[:, 0] for s in S])
        q50 = 10 * (np.percentile(syn, 50, 0) - np.percentile(real, 50, 0))
        a_abs, a_shape = [], []
        for i in range(20):
            Xs = syn[rng.choice(len(syn), len(real), replace=False)]
            a_abs.append(auc(real, Xs, SEED + i))
            a_shape.append(auc(real - real.mean(1, keepdims=True), Xs - Xs.mean(1, keepdims=True), SEED + i))
        out[mode] = dict(rms_db_median=round(float(np.median(rms)), 3), rms_db_p90=round(float(np.percentile(rms, 90)), 3),
                         peak_ratio_dec=dict(zip(("p10", "p50", "p90", "max"),
                                                 np.round(np.percentile(ratios, [10, 50, 90, 100]), 2).tolist())),
                         floor_coef_at_bound=int(sum(np.isclose(e["params"][:8], -20).any() for e in entries)),
                         median_band_diff_db=dict(median_abs=round(float(np.median(np.abs(q50))), 2),
                                                  max_abs=round(float(np.abs(q50).max()), 2)),
                         auc_absolute=round(float(np.mean(a_abs)), 3), auc_shape=round(float(np.mean(a_shape)), 3))
        print(mode, out[mode], flush=True)
    with open(os.path.join(HERE, "floor_mode_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
