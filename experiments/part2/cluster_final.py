"""Archetypes on the enlarged fit half (plan Part 2b, Task 10; D-24 (2), (4), P3; D-25).

    .venv/Scripts/python experiments/part2/cluster_final.py      (after fit_model.py)
One spectrum per vehicle (power mean of windows and of roll/pitch), divided by its chip's fitted low-pass
magnitude (model_fits.json), 48 log band powers 20-780 Hz from native bins, mean-removed. PCA (95 %) + k-means
(k 2-8, silhouette), bootstrap stability over vehicles (200), Q12 cross-tables. The D-24 P3 rule decides
(silhouette >= 0.5 and median ARI >= 0.8): clusters, or one generator. Writes clusters_final.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import archetypes as ar            # noqa: E402
import data_quality as dq          # noqa: E402
import noise_model as nm           # noqa: E402
import spectra_library as sl       # noqa: E402

TH = dq.FROZEN
SEED = 20261015


def feature_matrix():
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    vs = sl.vehicle_spectra(df, np.load(os.path.join(spec, "fit_psd_native.npz")))
    fits = json.load(open(os.path.join(HERE, "model_fits.json")))
    edges = sl.band_edges(TH)
    X, X_raw, att = [], [], []
    for v in vs:
        lp = fits["chips"][v["chip"]]["lowpass"]
        lp = None if lp is None else tuple(lp)
        H2 = nm.lowpass_gain2(v["f"], lp)
        X.append(sl.band_features(v["f"], (v["P"] / H2)[:, None], edges)[:, 0])
        X_raw.append(sl.band_features(v["f"], v["P"][:, None], edges)[:, 0])
        first = df[df.vehicle_key == v["vehicle_key"]].iloc[0]
        att.append(dict(vehicle=v["vehicle_key"], chip=v["chip"], rate=f"{round(first.fs_nominal, -2) / 1000:g} kHz",
                        firmware=str(first.version)[:4], has_lowpass=lp is not None))
    X, X_raw = np.array(X), np.array(X_raw)
    return X - X.mean(1, keepdims=True), X_raw - X_raw.mean(1, keepdims=True), pd.DataFrame(att), edges


def run(X, att):
    res = ar.fit_archetypes(X, k_range=range(2, 9), var=0.95, seed=SEED)
    st = ar.stability(X, att["vehicle"].to_numpy(), res["k"], n_boot=200, seed=SEED)
    sil = res["silhouette_by_k"][res["k"]]
    return dict(n=len(X), k=res["k"], silhouette_by_k={int(k): round(v, 4) for k, v in res["silhouette_by_k"].items()},
                silhouette=round(sil, 4), stability=st, stable=bool(ar.is_stable(sil, st["median_ari"])),
                pca_components=int(res["pca"].n_components_),
                pca_variance=round(float(res["pca"].explained_variance_ratio_.sum()), 4),
                crosstabs={a: ar.crosstab(res["labels"], att[a].to_numpy()).to_dict() for a in ("chip", "rate", "firmware")},
                labels=res["labels"].tolist(),
                centroids=np.round(res["pca"].inverse_transform(res["kmeans"].cluster_centers_), 4).tolist())


def main():
    X, X_raw, att, edges = feature_matrix()
    out = dict(rule=dict(silhouette_min=ar.SIL_MIN, ari_min=ar.ARI_MIN), band_edges=np.round(edges, 3).tolist(),
               vehicles=att["vehicle"].tolist(), chips=att["chip"].tolist(),
               after_chip_lowpass=run(X, att), without_correction=run(X_raw, att))
    a = out["after_chip_lowpass"]
    out["decision"] = "clusters" if a["stable"] else "one generator, chip as covariate (D-24 (4))"
    with open(os.path.join(HERE, "clusters_final.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    for name in ("after_chip_lowpass", "without_correction"):
        v = out[name]
        print(f"{name:20s} n={v['n']} k={v['k']} sil={v['silhouette']} ARI median={v['stability']['median_ari']:.2f} "
              f"p10={v['stability']['p10_ari']:.2f} stable={v['stable']} chip x cluster={v['crosstabs']['chip']}")
    print("DECISION (P3 rule):", out["decision"])


if __name__ == "__main__":
    main()
