"""Noise archetypes on the fit half: band features -> PCA -> k-means with silhouette (THESIS 7.5, gate Q12).

Features are log10 band powers in 48 log-spaced bands of the fixed band 20-780 Hz (spectra_library.band_features,
from native bins), mean-removed per spectrum (shape only; the level stays in the parametric model).
``axes``: "mean" averages roll and pitch POWER per band; "separate" makes roll and pitch two spectra.
``per``: "window" keeps every library window; "vehicle" averages a vehicle's windows (power mean) first.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score   # noqa: F401  (re-exported for callers)


def features(B, groups, axes="mean", per="window", remove_mean=True):
    """B: (N, bands, 3) log10 band powers; groups: (N,) vehicle keys. Returns X, row ids, row groups."""
    groups = np.asarray(groups)
    if axes == "mean":
        Y = np.log10((10 ** B[:, :, 0] + 10 ** B[:, :, 1]) / 2)[None]
    elif axes == "separate":
        Y = np.stack([B[:, :, 0], B[:, :, 1]])
    else:
        raise ValueError(axes)
    rows, ids, gs = [], [], []
    for a, Ya in enumerate(Y):
        if per == "window":
            rows.append(Ya)
            ids += [(i, a) for i in range(len(Ya))]
            gs += list(groups)
        elif per == "vehicle":
            for v in dict.fromkeys(groups):
                rows.append(np.log10((10 ** Ya[groups == v]).mean(0))[None])
                ids.append((v, a))
                gs.append(v)
        else:
            raise ValueError(per)
    X = np.vstack(rows)
    if remove_mean:
        X = X - X.mean(1, keepdims=True)
    return X, ids, np.array(gs)


def _fit(X, k, var, seed):
    pca = PCA(n_components=var, svd_solver="full").fit(X)
    Z = pca.transform(X)
    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(Z)
    return pca, Z, km


def fit_archetypes(X, k_range=range(2, 9), var=0.95, seed=20261015):
    """PCA keeping ``var`` of the variance, then k-means for each k; k = argmax silhouette (in PCA space)."""
    sil, fits = {}, {}
    for k in k_range:
        if k >= len(X):
            continue
        pca, Z, km = _fit(X, k, var, seed)
        sil[k] = float(silhouette_score(Z, km.labels_))
        fits[k] = (pca, Z, km)
    k = max(sil, key=sil.get)
    pca, Z, km = fits[k]
    return dict(k=k, pca=pca, Z=Z, kmeans=km, labels=km.labels_.copy(), silhouette_by_k=sil)


def bootstrap_indices(groups, rng):
    """Row indices of one bootstrap resample over VEHICLES: draw vehicles with replacement, keep all their rows."""
    groups = np.asarray(groups)
    uniq = np.array(list(dict.fromkeys(groups)))
    pick = rng.choice(uniq, size=len(uniq), replace=True)
    return np.concatenate([np.nonzero(groups == v)[0] for v in pick])


def stability(X, groups, k, n_boot=200, var=0.95, seed=20261015):
    """Bootstrap over VEHICLES (rows of one vehicle stay together): refit PCA + k-means on each resample,
    label all rows, compare with the full-data labels by the adjusted Rand index."""
    groups = np.asarray(groups)
    ref = _fit(X, k, var, seed)[2].labels_
    rng = np.random.default_rng(seed)
    aris = []
    for b in range(n_boot):
        idx = bootstrap_indices(groups, rng)
        if len(np.unique(X[idx], axis=0)) <= k:
            continue
        pca, _, km = _fit(X[idx], k, var, seed + b + 1)
        aris.append(adjusted_rand_score(ref, km.predict(pca.transform(X))))
    return dict(median_ari=float(np.median(aris)), p10_ari=float(np.percentile(aris, 10)), n_boot=len(aris))


def crosstab(labels, attribute):
    """Cluster x attribute counts (gate Q12: a cluster that is one chip type is a hardware label)."""
    return pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(attribute, name="attribute"))
