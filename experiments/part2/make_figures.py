"""Figures of Phase 3 Part 2b (fit half): Fig. 2 real vs synthetic noise spectra (D-26 generator).

    .venv/Scripts/python experiments/part2/make_figures.py      (after fit_model.py)
Fig. 2: per-vehicle spectra (roll/pitch power mean, mean of windows) vs synthetic spectra from the generator,
median and 10-90 % band of log10 PSD on 20-780 Hz; one panel for ICM-4268x vehicles (on-chip roll-off) and one
for all other chips. Palette validated (dataviz validate_palette.js, light: all checks pass).
Writes figures/fig2_real_vs_synthetic.png and .pdf.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import data_quality as dq          # noqa: E402
import noise_generator as ng       # noqa: E402
import spectra_library as sl       # noqa: E402
from generator_check import jitter, SEED   # noqa: E402

REAL, SYN = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#1f2328", "#5b6470", "#e6e8eb"
TH = dq.FROZEN
BAND = (TH.f_min, TH.f_max)
F = np.arange(BAND[0], BAND[1] + 1e-9, 1.5)


def main():
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    native = np.load(os.path.join(spec, "fit_psd_native.npz"))
    vs = sl.vehicle_spectra(df, native)
    fits = json.load(open(os.path.join(HERE, "model_fits.json")))
    entries = []
    for v in vs:
        e = dict(fits["vehicles"][v["vehicle_key"]])
        e["lowpass"] = None if e["lowpass"] is None else tuple(e["lowpass"])
        entries.append(e)
    jit = jitter(df, native, sl.band_edges(TH))
    gen = ng.fit_generator(entries, BAND, jit["freq_sd"], jit["height_sd"])
    S, meta = ng.sample(gen, 2000, F, np.random.default_rng(SEED))
    real = {v["vehicle_key"]: np.interp(F, v["f"], v["P"]) for v, e in zip(vs, entries) if not e.get("dropped")}
    chip = {v["vehicle_key"]: v["chip"] for v in vs}
    icm = lambda c: c.startswith("ICM4268")                                 # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, (title, sel) in zip(axes, (("ICM-42688-P / 42686-P (on-chip roll-off)", True), ("All other IMU chips", False))):
        R = np.log10([p for k, p in real.items() if icm(chip[k]) == sel])
        Y = np.log10(S[[icm(m["chip"]) == sel for m in meta]])
        for X, col, lab in ((R, REAL, f"Real vehicles (n = {len(R)})"), (Y, SYN, f"Synthetic (n = {len(Y)})")):
            lo, md, hi = np.percentile(X, [10, 50, 90], axis=0)
            ax.fill_between(F, 10 ** lo, 10 ** hi, color=col, alpha=0.18, linewidth=0)
            ax.plot(F, 10 ** md, color=col, linewidth=2, label=lab)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*BAND)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("Frequency [Hz]", color=MUTED)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    axes[0].set_ylabel("Gyro PSD [(rad/s)$^2$/Hz]  (median, 10-90 % band)", color=MUTED)
    out = os.path.join(HERE, "figures")
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig2_real_vs_synthetic.{ext}"), dpi=200)
    print("written", out)


if __name__ == "__main__":
    main()
