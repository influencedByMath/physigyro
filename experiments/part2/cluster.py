"""Noise archetypes on the fit-half library: PCA + k-means for every candidate spectrum unit (plan Task 4).

    .venv/Scripts/python experiments/part2/cluster.py
Writes clusters.json next to this file: per variant (axes "mean"/"separate" x per "window"/"vehicle") the
chosen k, silhouette by k, PCA components and variance, bootstrap stability over vehicles (median ARI), and
the cross-tables of clusters against IMU chip, sample rate and firmware version (gate Q12). The author picks
the variant (D-24); nothing is chosen here.
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
import spectra_library as sl       # noqa: E402

TH = dq.FROZEN
SEED = 20261015


def library():
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    z = np.load(os.path.join(spec, "fit_psd_native.npz"))
    edges = sl.band_edges(TH)
    B = np.array([sl.band_features(z[f"{w}|f"].astype(float), z[f"{w}|P"].astype(float), edges)
                  for w in df["window_id"]])
    return df, B, edges


def attributes(df, ids, per):
    """Chip, rate class and firmware minor version for each feature row."""
    if per == "window":
        rows = df.iloc[[i for i, _ in ids]]
    else:
        first = df.groupby("vehicle_key").head(1).set_index("vehicle_key")
        rows = first.loc[[v for v, _ in ids]]
    rate = (rows["fs_nominal"].round(-2) / 1000).map(lambda x: f"{x:g} kHz").to_numpy()
    fw = rows["version"].astype(str).str.extract(r"^(\d+\.\d+)")[0].fillna("?").to_numpy()
    return dict(chip=rows["fifo_chip"].fillna("?").to_numpy(), rate=rate, firmware=fw,
                vehicle=rows["vehicle_key"].to_numpy())


def run_variant(df, B, axes, per):
    X, ids, groups = ar.features(B, df["vehicle_key"].to_numpy(), axes=axes, per=per)
    res = ar.fit_archetypes(X, k_range=range(2, 9), var=0.95, seed=SEED)
    st = ar.stability(X, groups, res["k"], n_boot=200, seed=SEED)
    att = attributes(df, ids, per)
    tabs = {a: ar.crosstab(res["labels"], att[a]).to_dict() for a in ("chip", "rate", "firmware")}
    # how much of each cluster is one vehicle / one chip (Q12 hardware-label check)
    purity = {}
    for c in range(res["k"]):
        m = res["labels"] == c
        purity[int(c)] = dict(n=int(m.sum()), vehicles=int(len(set(att["vehicle"][m]))),
                              top_chip_share=round(float(pd.Series(att["chip"][m]).value_counts(normalize=True).iloc[0]), 3))
    centroids = res["pca"].inverse_transform(res["kmeans"].cluster_centers_)
    return dict(n_rows=len(X), k=res["k"], silhouette_by_k={int(k): round(v, 4) for k, v in res["silhouette_by_k"].items()},
                pca_components=int(res["pca"].n_components_),
                pca_variance=round(float(res["pca"].explained_variance_ratio_.sum()), 4),
                stability=st, purity=purity, crosstabs=tabs, centroids=np.round(centroids, 4).tolist(),
                labels=res["labels"].tolist(), row_ids=[[str(a), int(b)] for a, b in ids])


def main():
    df, B, edges = library()
    out = dict(band_edges=np.round(edges, 3).tolist(), n_windows=len(df), n_vehicles=int(df["vehicle_key"].nunique()))
    for axes in ("mean", "separate"):
        for per in ("window", "vehicle"):
            v = run_variant(df, B, axes, per)
            out[f"{axes}|{per}"] = v
            print(f"{axes:8s} {per:7s} rows {v['n_rows']:3d} k {v['k']} sil {v['silhouette_by_k']} "
                  f"PCs {v['pca_components']} ARI {v['stability']['median_ari']:.2f} (p10 {v['stability']['p10_ari']:.2f}) "
                  f"purity {v['purity']}", flush=True)
    with open(os.path.join(HERE, "clusters.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
