"""Evidence for the Part 2 method decisions (plan Task 3), on the fit-half library only.

    .venv/Scripts/python experiments/part2/evidence.py
Writes evidence.json next to this file. Nothing is decided here; the author decides (D-24).

A. Unit of a spectrum: how different are roll and pitch in the same window? (48 band features, dB)
B. Dependence: are the (up to 3) windows of one vehicle near-copies? If the nearest neighbour of most windows
   is a window of the same vehicle, clustering windows partly clusters vehicles (silhouette inflated).
C. Floor shape (THESIS 7.5 assumes a flat floor): a robust floor (20th percentile of the log PSD in sliding
   1/6-decade windows, which ignores narrow peaks) is fitted by a flat line, a power law (straight line in
   log-log) and a 4-knot log-log spline; RMS error in dB over 20-780 Hz.
D. Usable count: vehicles with at least one window, Wilson 95 % CI, projection to the test half.
E. Gate Q12 (chip effects): high-frequency roll-off per IMU chip = mean log10 band power above 500 Hz minus
   100-300 Hz (roll-pitch power mean, per vehicle), in decades.
F. Does clustering on a band below the roll-off (20-300 Hz) remove the chip dependence? Same PCA + k-means
   (vehicle means), cross-tab against chip and rate.
G. Chip or mount? Roll-off per vehicle on the same board model (PX4_FMU_V6X: one isolated IMU board, several
   IMU chips), and for ICM-4268x chips on any board.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import data_quality as dq          # noqa: E402
import spectra_library as sl       # noqa: E402
from log_screen import wilson_ci   # noqa: E402

TH = dq.FROZEN
SPEC = os.path.join(ROOT, "data", "spectra")


def load():
    df = pd.read_csv(os.path.join(SPEC, "fit_windows.csv"))
    z = np.load(os.path.join(SPEC, "fit_psd_native.npz"))
    edges = sl.band_edges(TH)
    B = np.array([sl.band_features(z[f"{w}|f"].astype(float), z[f"{w}|P"].astype(float), edges)
                  for w in df["window_id"]])            # (N, 48, 3) log10
    return df, z, B


def q(x, p):
    return round(float(np.percentile(x, p)), 3)


def evidence_a(B):
    d = 10 * np.abs(B[:, :, 0] - B[:, :, 1])            # dB difference roll vs pitch per band
    level = 10 * np.abs(B[:, :, 0].mean(1) - B[:, :, 1].mean(1))
    shape = 10 * np.abs((B[:, :, 0] - B[:, :, 0].mean(1, keepdims=True)) - (B[:, :, 1] - B[:, :, 1].mean(1, keepdims=True)))
    return dict(per_window_median_db=dict(median=q(np.median(d, 1), 50), p90=q(np.median(d, 1), 90)),
                level_db=dict(median=q(level, 50), p90=q(level, 90)),
                shape_median_db=dict(median=q(np.median(shape, 1), 50), p90=q(np.median(shape, 1), 90)))


def evidence_b(df, B):
    X = B[:, :, :2].mean(2)                             # roll-pitch mean, mean-removed (THESIS 7.5 step 1)
    X = X - X.mean(1, keepdims=True)
    veh = df["vehicle_key"].to_numpy()
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    same = veh[:, None] == veh[None, :]
    multi = np.array([(veh == v).sum() > 1 for v in veh])
    nn_same = same[np.arange(len(X)), D.argmin(1)][multi]
    iu = np.triu_indices(len(X), 1)
    within, between = D[iu][same[iu]], D[iu][~same[iu]]
    return dict(n_windows=len(X), n_vehicles=int(len(set(veh))), windows_in_multi_window_vehicles=int(multi.sum()),
                nearest_neighbour_same_vehicle=round(float(nn_same.mean()), 3) if len(nn_same) else None,
                distance_within_median=q(within, 50) if len(within) else None, distance_between_median=q(between, 50),
                distance_within_p90=q(within, 90) if len(within) else None, distance_between_p10=q(between, 10))


def robust_floor(f, logP, f_lo, f_hi, width_dec=1 / 6, pct=20):
    m = (f >= f_lo) & (f <= f_hi)
    lf, y = np.log10(f[m]), logP[m]
    centres = np.linspace(lf[0], lf[-1], 60)
    fl = np.array([np.percentile(y[np.abs(lf - c) <= width_dec / 2], pct) for c in centres])
    return centres, fl


def floor_fits(centres, fl):
    out = {}
    out["flat"] = 10 * np.sqrt(np.mean((fl - fl.mean()) ** 2))
    p = np.polyfit(centres, fl, 1)
    out["power_law"] = 10 * np.sqrt(np.mean((fl - np.polyval(p, centres)) ** 2))
    knots = np.linspace(centres[0], centres[-1], 6)[1:-1]
    s = LSQUnivariateSpline(centres, fl, knots, k=3)
    out["spline_4knots"] = 10 * np.sqrt(np.mean((fl - s(centres)) ** 2))
    out["slope_dec_per_dec"] = p[0]
    out["range_dec"] = float(fl.max() - fl.min())
    return out


def evidence_c(df, z):
    res = []
    for w in df["window_id"]:
        f, P = z[f"{w}|f"].astype(float), z[f"{w}|P"].astype(float)
        logP = np.log10(P[:, :2].mean(1))
        res.append(floor_fits(*robust_floor(f, logP, TH.f_min, TH.f_max)))
    R = pd.DataFrame(res)
    return {c: dict(median=q(R[c], 50), p90=q(R[c], 90), min=q(R[c], 0), max=q(R[c], 100)) for c in R.columns}


def evidence_d(df):
    n_fit = int((pd.read_csv(os.path.join(ROOT, "data", "screen", "split.csv"))["split"] == "fit").sum())
    n_test = int((pd.read_csv(os.path.join(ROOT, "data", "screen", "split.csv"))["split"] == "test").sum())
    k = int(df["vehicle_key"].nunique())
    lo, hi = wilson_ci(k, n_fit)
    return dict(usable_fit=k, fit_vehicles=n_fit, share=round(k / n_fit, 3), ci95=[round(lo, 3), round(hi, 3)],
                test_vehicles=n_test, projected_test=round(k / n_fit * n_test, 1),
                projected_test_range=[round(lo * n_test, 1), round(hi * n_test, 1)],
                projected_total=round(k + k / n_fit * n_test, 1),
                projected_total_range=[round(k + lo * n_test, 1), round(k + hi * n_test, 1)],
                windows=len(df), windows_per_vehicle=df.groupby("vehicle_key").size().value_counts().sort_index().to_dict())


def evidence_e(df, B):
    fc = np.sqrt(sl.band_edges(TH)[:-1] * sl.band_edges(TH)[1:])
    rp = np.log10((10 ** B[:, :, 0] + 10 ** B[:, :, 1]) / 2)
    roll = rp[:, fc > 500].mean(1) - rp[:, (fc >= 100) & (fc <= 300)].mean(1)
    v = pd.DataFrame(dict(vehicle=df["vehicle_key"], chip=df["fifo_chip"].fillna("?"),
                          rate=(df["fs_nominal"] / 1000).round(1), roll=roll)).groupby("vehicle").agg(
        chip=("chip", "first"), rate=("rate", "first"), roll=("roll", "mean"))
    return {f"{c} @ {r:g} kHz": dict(n=int(len(g)), median_dec=q(g["roll"], 50), min_dec=q(g["roll"], 0),
                                     max_dec=q(g["roll"], 100))
            for (c, r), g in v.groupby(["chip", "rate"])}


def evidence_f(df, B, f_hi=300.0):
    import archetypes as ar
    fc = np.sqrt(sl.band_edges(TH)[:-1] * sl.band_edges(TH)[1:])
    keep = fc <= f_hi
    X, ids, groups = ar.features(B[:, keep, :], df["vehicle_key"].to_numpy(), axes="mean", per="vehicle")
    res = ar.fit_archetypes(X, k_range=range(2, 9), seed=20261015)
    first = df.groupby("vehicle_key").head(1).set_index("vehicle_key", drop=False).loc[[v for v, _ in ids]]
    st = ar.stability(X, groups, res["k"], n_boot=200, seed=20261015)
    return dict(band_hz=[TH.f_min, f_hi], n_bands=int(keep.sum()), k=res["k"],
                silhouette_by_k={int(k): round(v, 3) for k, v in res["silhouette_by_k"].items()},
                stability=st, chip=ar.crosstab(res["labels"], first["fifo_chip"].fillna("?").to_numpy()).to_dict(),
                rate=ar.crosstab(res["labels"], (first["fs_nominal"] / 1000).round(1).to_numpy()).to_dict())


def evidence_g(df, B):
    fc = np.sqrt(sl.band_edges(TH)[:-1] * sl.band_edges(TH)[1:])
    rp = np.log10((10 ** B[:, :, 0] + 10 ** B[:, :, 1]) / 2)
    roll = rp[:, fc > 500].mean(1) - rp[:, (fc >= 100) & (fc <= 300)].mean(1)
    v = pd.DataFrame(dict(vehicle=df["vehicle_key"], hw=df["sys_hw"], chip=df["fifo_chip"].fillna("?"),
                          roll=roll)).groupby("vehicle").agg(hw=("hw", "first"), chip=("chip", "first"), roll=("roll", "mean"))
    v6x = v[v["hw"] == "PX4_FMU_V6X"]
    icm = v[v["chip"].str.startswith("ICM4268")]
    return dict(fmu_v6x={c: sorted(round(float(x), 2) for x in g["roll"]) for c, g in v6x.groupby("chip")},
                icm4268x_any_board={f"{h} {c}": sorted(round(float(x), 2) for x in g["roll"])
                                    for (h, c), g in icm.groupby(["hw", "chip"])},
                other_chips_all=dict(n=int((~v["chip"].str.startswith("ICM4268")).sum()),
                                     median=q(v.loc[~v["chip"].str.startswith("ICM4268"), "roll"], 50),
                                     min=q(v.loc[~v["chip"].str.startswith("ICM4268"), "roll"], 0)),
                icm4268x_all=dict(n=int(len(icm)), median=q(icm["roll"], 50), max=q(icm["roll"], 100)))


def main():
    df, z, B = load()
    out = dict(A_roll_vs_pitch=evidence_a(B), B_dependence=evidence_b(df, B), C_floor=evidence_c(df, z),
               D_usable=evidence_d(df), E_chip_rolloff=evidence_e(df, B), F_cluster_below_300=evidence_f(df, B),
               G_chip_or_mount=evidence_g(df, B))
    print(json.dumps(out, indent=1, default=str))
    with open(os.path.join(HERE, "evidence.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
