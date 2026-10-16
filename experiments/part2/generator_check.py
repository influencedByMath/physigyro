"""Synthetic vs real spectra for the D-26 generator (plan Part 2b, Task 11; THESIS 7.5 sanity check).

    .venv/Scripts/python experiments/part2/generator_check.py     (after fit_model.py)
1. Jitter sizes measured within vehicles (D-26 (2)): frequency = median within-vehicle sd of log10 mean motor
   frequency across windows (vehicles with valid RPM and >= 2 windows); height = median within-vehicle sd of the
   48 log10 band powers across windows.
2. In-sample: per-band quantiles (10/50/90 %) real vs synthetic in dB, and a cross-validated logistic-regression
   real-vs-synthetic AUC on absolute levels and on shapes (0.5 = cannot tell them apart), 20 draws.
3. Leave one vehicle out (generalisation): for each vehicle a generator WITHOUT it (jitter re-estimated without
   it; the chip low-pass choice of model_fits.json still includes it - stated); the distance (RMS dB over 48 bands)
   from the held-out spectrum to its nearest synthetic spectrum vs to its nearest other real vehicle, with EQUAL
   sample sizes (as many synthetic draws as other real vehicles; 20 repeats, median). Independent review
   2026-10-16: with 200 draws vs 49 real the minimum favoured the generator (share 0.76 vs 0.48 at equal size).
4. Novelty: distance of each synthetic spectrum to its own source vehicle vs the TYPICAL (median pairwise)
   distance between two real vehicles. Peak sets are near-copies of the source (frequency jitter 0.65 %); only the
   floor is a new draw - stated, not hidden.
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
F_SYN = np.arange(BAND[0], BAND[1] + 1e-9, 0.5)


def auc(Xr, Xs, seed):
    X = np.vstack([Xr, Xs])
    y = np.r_[np.zeros(len(Xr)), np.ones(len(Xs))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="roc_auc").mean())


def jitter(df, native, edges):
    freq, height = [], []
    for v, g in df.groupby("vehicle_key"):
        if len(g) < 2:
            continue
        B = np.array([sl.band_features(np.asarray(native[w + "|f"], float),
                                       np.asarray(native[w + "|P"], float)[:, :2].mean(1)[:, None], edges)[:, 0]
                      for w in g.window_id])
        height.append(float(np.median(B.std(0, ddof=1))))
        if g.rpm_valid.all():
            freq.append(float(np.std(np.log10([np.mean(json.loads(x)) for x in g.rpm_hz]), ddof=1)))
    return dict(freq_sd=float(np.median(freq)), n_freq_vehicles=len(freq), height_sd=float(np.median(height)),
                n_height_vehicles=len(height))


def bands(S, edges):
    return np.array([sl.band_features(F_SYN, s[:, None], edges)[:, 0] for s in S])


def rms_db(a, B):
    return 10 * np.sqrt(np.mean((B - a) ** 2, axis=1))


def main():
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    native = np.load(os.path.join(spec, "fit_psd_native.npz"))
    vs = sl.vehicle_spectra(df, native)
    fits = json.load(open(os.path.join(HERE, "model_fits.json")))
    edges = sl.band_edges(TH)
    entries = []
    for v in vs:
        e = dict(fits["vehicles"][v["vehicle_key"]])
        e["lowpass"] = None if e["lowpass"] is None else tuple(e["lowpass"])
        entries.append(e)
    keep = [i for i, e in enumerate(entries) if not e.get("dropped")]
    real = np.array([sl.band_features(vs[i]["f"], vs[i]["P"][:, None], edges)[:, 0] for i in keep])
    jit = jitter(df, native, edges)
    rng = np.random.default_rng(SEED)

    gen = ng.fit_generator(entries, BAND, jit["freq_sd"], jit["height_sd"])
    S, meta = ng.sample(gen, 2000, F_SYN, rng)
    syn = bands(S, edges)
    qs = {}
    for q in (10, 50, 90):
        d = 10 * (np.percentile(syn, q, 0) - np.percentile(real, q, 0))
        qs[str(q)] = dict(median_abs_db=round(float(np.median(np.abs(d))), 2),
                          max_abs_db=round(float(np.abs(d).max()), 2), per_band_db=np.round(d, 2).tolist())
    aucs = {"absolute": [], "shape": []}
    for i in range(20):
        Xs = syn[rng.choice(len(syn), len(real), replace=False)]
        aucs["absolute"].append(auc(real, Xs, SEED + i))
        aucs["shape"].append(auc(real - real.mean(1, keepdims=True), Xs - Xs.mean(1, keepdims=True), SEED + i))

    # leave one vehicle out, equal sample sizes
    lovo = []
    for n_i, i in enumerate(keep):
        others = [e for j, e in enumerate(entries) if j != i]
        jit_i = jitter(df[df.vehicle_key != vs[i]["vehicle_key"]], native, edges)
        g_i = ng.fit_generator(others, BAND, jit_i["freq_sd"], jit_i["height_sd"])
        n_other = len(real) - 1
        S_i, _ = ng.sample(g_i, 20 * n_other, F_SYN, np.random.default_rng(SEED + 1000 + i))
        D = rms_db(real[n_i], bands(S_i, edges)).reshape(20, n_other)
        d_syn = float(np.median(D.min(axis=1)))
        d_real = rms_db(real[n_i], np.delete(real, n_i, axis=0)).min()
        lovo.append((d_syn, d_real))
    lovo = np.array(lovo)
    # novelty (in-sample): synthetic to its own source vs the typical distance between two real vehicles
    pos = {i: n for n, i in enumerate(keep)}
    d_src = np.array([rms_db(real[pos[m["source"]]], syn[k:k + 1])[0] for k, m in enumerate(meta)])
    iu = np.triu_indices(len(real), 1)
    pair = np.array([rms_db(real[a], real[b:b + 1])[0] for a, b in zip(*iu)])
    out = dict(jitter=jit, n_real=len(real), n_dropped=len(entries) - len(keep), quantile_diff=qs,
               auc={k: dict(mean=round(float(np.mean(v)), 3), sd=round(float(np.std(v)), 3)) for k, v in aucs.items()},
               lovo=dict(median_nearest_synthetic_db=round(float(np.median(lovo[:, 0])), 2),
                         median_nearest_real_db=round(float(np.median(lovo[:, 1])), 2),
                         share_synthetic_at_least_as_close=round(float(np.mean(lovo[:, 0] <= lovo[:, 1])), 3),
                         per_vehicle=np.round(lovo, 2).tolist()),
               novelty=dict(median_synthetic_to_source_db=round(float(np.median(d_src)), 2),
                            median_real_pairwise_db=round(float(np.median(pair)), 2),
                            note="peak sets are near-copies of the source (frequency jitter 0.65 %); the floor is a "
                                 "new draw"),
               chip_share=gen["chip_share"], lowpass={k: v for k, v in gen["lowpass"].items()}, in_sample_auc=True)
    with open(os.path.join(HERE, "generator_check.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("jitter", "n_real", "n_dropped", "auc", "novelty")}, default=str))
    print("LOVO:", {k: v for k, v in out["lovo"].items() if k != "per_vehicle"})
    print({q: (v["median_abs_db"], v["max_abs_db"]) for q, v in qs.items()})


if __name__ == "__main__":
    main()
