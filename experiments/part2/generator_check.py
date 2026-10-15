"""Synthetic vs real spectra (plan Part 2b, Task 11; THESIS 7.5 sanity check).

    .venv/Scripts/python experiments/part2/generator_check.py     (after fit_model.py and cluster_final.py)
Fits the generator (src/noise_generator.py) to the per-vehicle model fits (model_fits.json) and compares
synthetic with real spectra on the 48 log band powers (20-780 Hz):
1. per-band quantiles (10/50/90 %) real vs synthetic, in dB;
2. a two-sample classifier test: cross-validated logistic regression real-vs-synthetic AUC (0.5 = cannot tell
   them apart), on absolute levels and on shapes (mean-removed), 20 synthetic draws of the real sample size.
The generator is fitted on all fit-half vehicles, so the check is in-sample (optimistic); stated as such.
Writes generator_check.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data_quality as dq          # noqa: E402
import noise_generator as ng       # noqa: E402
import spectra_library as sl       # noqa: E402

TH = dq.FROZEN
BAND = (TH.f_min, TH.f_max)
SEED = 20261015


def auc(Xr, Xs, seed):
    X = np.vstack([Xr, Xs])
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xs))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def main():
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    vs = sl.vehicle_spectra(df, np.load(os.path.join(spec, "fit_psd_native.npz")))
    fits = json.load(open(os.path.join(HERE, "model_fits.json")))
    entries = [dict(fits["vehicles"][v["vehicle_key"]], lowpass=None if fits["vehicles"][v["vehicle_key"]]["lowpass"]
                    is None else tuple(fits["vehicles"][v["vehicle_key"]]["lowpass"])) for v in vs]
    gen = ng.fit_generator(entries, BAND)
    edges = sl.band_edges(TH)
    real = np.array([sl.band_features(v["f"], v["P"][:, None], edges)[:, 0] for v in vs])
    f = np.arange(BAND[0], BAND[1] + 1e-9, 0.5)
    rng = np.random.default_rng(SEED)
    S, meta = ng.sample(gen, 2000, f, rng)
    syn = np.array([sl.band_features(f, s[:, None], edges)[:, 0] for s in S])
    qs = {}
    for q in (10, 50, 90):
        d = 10 * (np.percentile(syn, q, 0) - np.percentile(real, q, 0))
        qs[str(q)] = dict(median_abs_db=round(float(np.median(np.abs(d))), 2), max_abs_db=round(float(np.abs(d).max()), 2),
                          per_band_db=np.round(d, 2).tolist())
    aucs = {"absolute": [], "shape": []}
    n = len(real)
    for i in range(20):
        Xs = syn[rng.choice(len(syn), n, replace=False)]
        aucs["absolute"].append(auc(real, Xs, SEED + i))
        aucs["shape"].append(auc(real - real.mean(1, keepdims=True), Xs - Xs.mean(1, keepdims=True), SEED + i))
    out = dict(n_real=n, n_synthetic=len(syn), quantile_diff=qs,
               auc={k: dict(mean=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3)) for k, v in aucs.items()},
               chip_share=gen["chip_share"], lowpass=gen["lowpass"], peaks_per_spectrum=dict(
                   real_median=float(np.median(gen["n_peaks"])), real_max=int(max(gen["n_peaks"]))),
               in_sample=True)
    with open(os.path.join(HERE, "generator_check.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("n_real", "auc", "chip_share")}, default=str))
    print({q: (v["median_abs_db"], v["max_abs_db"]) for q, v in qs.items()})


if __name__ == "__main__":
    main()
