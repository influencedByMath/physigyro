"""Noise archetypes: band features, PCA + k-means with silhouette, bootstrap stability (THESIS 7.5)."""
import numpy as np
import pytest

import archetypes as ar


def families(n_per=12, seed=0):
    """Two spectrum families on 48 bands: a steep low-pass slope, and a flat floor with a strong 150 Hz peak."""
    rng = np.random.default_rng(seed)
    fc = np.geomspace(20, 780, 49)
    fc = np.sqrt(fc[:-1] * fc[1:])
    slope = -3.0 * np.log10(fc / 20)
    peak = 2.0 / (1 + ((fc - 150) / 15) ** 2)
    X = np.vstack([slope + 0.05 * rng.standard_normal((n_per, 48)), peak + 0.05 * rng.standard_normal((n_per, 48))])
    groups = np.repeat(np.arange(2 * n_per // 2), 2)              # two windows per "vehicle"
    return X, np.array([0] * n_per + [1] * n_per), groups


def test_features_mean_removed_and_units():
    B = np.zeros((4, 48, 3))
    B[:, :, 0], B[:, :, 1] = 1.0, 3.0                            # roll 10, pitch 1000 (log10 power)
    B[2:, :, :2] += 5.0
    groups = np.array(["a", "a", "b", "b"])
    X, ids, g = ar.features(B, groups, axes="mean", per="window")
    assert X.shape == (4, 48) and np.allclose(X.mean(1), 0)
    X2, ids2, g2 = ar.features(B, groups, axes="separate", per="window")
    assert X2.shape == (8, 48) and list(g2) == ["a", "a", "b", "b"] * 2
    X3, ids3, g3 = ar.features(B, groups, axes="mean", per="vehicle")
    assert X3.shape == (2, 48) and list(g3) == ["a", "b"]


def test_mean_axes_averages_power_not_log():
    B = np.zeros((1, 48, 3))
    B[0, :, 0], B[0, :, 1] = 0.0, 2.0
    B[0, 0, 1] = 4.0                                             # one band differs so the mean removal keeps it
    X, _, _ = ar.features(B, np.array(["a"]), axes="mean", per="window", remove_mean=False)
    assert X[0, 1] == pytest.approx(np.log10((1 + 100) / 2))


def test_two_families_are_found():
    X, truth, groups = families()
    res = ar.fit_archetypes(X, k_range=range(2, 6), var=0.95, seed=1)
    assert res["k"] == 2
    assert ar.adjusted_rand_score(truth, res["labels"]) == pytest.approx(1.0)
    assert res["pca"].explained_variance_ratio_.sum() >= 0.95
    assert set(res["silhouette_by_k"]) == {2, 3, 4, 5}


def test_reproducible_with_seed():
    X, _, _ = families()
    a = ar.fit_archetypes(X, k_range=range(2, 5), seed=7)
    b = ar.fit_archetypes(X, k_range=range(2, 5), seed=7)
    assert np.array_equal(a["labels"], b["labels"]) and a["silhouette_by_k"] == b["silhouette_by_k"]


def test_stability_high_for_clear_clusters():
    X, _, groups = families()
    s = ar.stability(X, groups, k=2, n_boot=30, seed=3)
    assert s["median_ari"] == pytest.approx(1.0) and s["n_boot"] == 30


def test_crosstab_counts():
    t = ar.crosstab(np.array([0, 0, 1, 1, 1]), np.array(["ICM", "BMI", "ICM", "ICM", "BMI"]))
    assert t.loc[0, "ICM"] == 1 and t.loc[1, "ICM"] == 2 and int(t.values.sum()) == 5


def test_vehicle_rows_are_power_means_of_their_windows():
    B = np.zeros((3, 48, 3))
    B[0, :, :2], B[1, :, :2], B[2, :, :2] = 0.0, 2.0, 5.0        # vehicle a: windows at 1 and 100; b: one window
    X, ids, g = ar.features(B, np.array(["a", "a", "b"]), axes="mean", per="vehicle", remove_mean=False)
    assert list(g) == ["a", "b"]
    assert np.allclose(X[0], np.log10((1 + 100) / 2)) and np.allclose(X[1], 5.0)


def test_pca_keeps_the_smallest_number_of_components_for_95_percent():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 48)) * np.linspace(3, 0.2, 48)    # variance spread over many directions
    res = ar.fit_archetypes(X, k_range=range(2, 4), var=0.95, seed=1)
    from sklearn.decomposition import PCA
    cum = np.cumsum(PCA().fit(X).explained_variance_ratio_)
    assert res["pca"].n_components_ == int(np.searchsorted(cum, 0.95) + 1) > 1


def test_bootstrap_keeps_vehicles_whole():
    groups = np.repeat(np.array(["a", "b", "c", "d", "e"]), [1, 2, 3, 2, 1])
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = ar.bootstrap_indices(groups, rng)
        for v in set(groups[idx]):
            n_rows = (groups == v).sum()
            assert (groups[idx] == v).sum() % n_rows == 0        # every pick of a vehicle brings all its rows


def test_stability_low_without_structure():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 48))
    s = ar.stability(X, np.arange(40), k=3, n_boot=40, seed=3)
    assert s["median_ari"] < 0.5


def test_stability_rule_d24_p3():
    assert ar.is_stable(0.5, 0.8) and ar.is_stable(0.7, 0.95)
    assert not ar.is_stable(0.49, 0.95) and not ar.is_stable(0.9, 0.79)
