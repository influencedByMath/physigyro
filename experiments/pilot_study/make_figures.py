"""Figures for the pilot-study report. Run after run_pilot.py:
    .venv/Scripts/python experiments/pilot_study/make_figures.py
"""
import json
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))
A = np.load(os.path.join(HERE, "arrays.npz"))
TH = R["thresholds_used"]

# categorical slots in fixed order (dataviz reference palette, light mode; validated for 3 slots)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#b9b8b2"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "lines.linewidth": 1.4})
LOGS = [r for r in R["logs"] if "stats" in r]
warnings.filterwarnings("ignore", "Mean of empty slice")      # bands above a slow log's Nyquist


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


def label_end(ax, x, y, text, color):
    ax.annotate(text, (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)


# 1. artificial dropouts: band error and Welch averages (two panels, one unit each)
if R.get("dropout_experiment"):
    d = R["dropout_experiment"]
    fr = np.array([float(k) for k in d]) * 100
    med = np.array([d[k]["median_abs_db"] for k in d])
    p95 = np.array([d[k]["p95_abs_db"] for k in d])
    few = np.array([100.0 * d[k]["few_averages"] / max(1, d[k]["n_trials"] - d[k]["failed"]) for k in d])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 2.9))
    a.plot(fr, med, "-o", color=C1, ms=5)
    a.plot(fr, p95, "-o", color=C2, ms=5)
    label_end(a, fr, med, "median", C1)
    label_end(a, fr, p95, "95th pct.", C2)
    a.axhline(1.0, color=INK2, lw=0.8, ls="--")
    a.text(fr.max() * 1.22, 1.05, "1 dB criterion", fontsize=7, color=INK2, va="bottom", ha="right")
    a.set_xlabel("artificial dropout fraction [%]")
    a.set_ylabel("|band-power error| [dB]")
    a.set_title(f"band error vs clean PSD ({len([1 for r in LOGS if r['stats']['dropout_fraction'] < 0.01 and r['windows']])} clean logs)",
                fontsize=9, loc="left")
    b.plot(fr, few, "-o", color=C1, ms=5)
    b.set_xlabel("artificial dropout fraction [%]")
    b.set_ylabel(f"windows with < {TH['min_averages']} Welch averages [%]")
    b.set_ylim(-3, 103)
    b.set_title("averages left (gate Q13)", fontsize=9, loc="left")
    for ax in (a, b):
        ax.set_xlim(0, fr.max() * 1.25)
    save(fig, "fig_p1_dropout.png")

# 2. usable band of every log: 0.8 x fs / 2 against the candidate f_max
fs = np.sort(np.array([r["stats"]["fs_measured"] for r in LOGS]))
fig, ax = plt.subplots(figsize=(7.5, 2.6))
usable = 0.8 * fs / 2
ax.plot(np.arange(len(fs)), usable, "o", color=C1, ms=6)
ax.axhline(TH["f_max"], color=INK2, lw=0.8, ls="--")
ax.text(0, TH["f_max"] * 1.04, f"candidate f_max = {TH['f_max']:.0f} Hz", fontsize=7, color=INK2)
ax.set_yscale("log")
ax.set_xlabel("pilot logs, sorted by measured FIFO rate")
ax.set_ylabel("0.8 x fs / 2 [Hz]")
ax.set_title("usable band of each log (gate Q5)", fontsize=9, loc="left")
save(fig, "fig_p2_rates.png")

# 3. steady-flight metrics over all airborne windows (ECDFs, candidate limits marked)
sp = np.sort([w["sp"] for r in LOGS for w in r["activity"] if w["sp"] is not None])
thr = np.sort([w["thr"] for r in LOGS for w in r["activity"] if w["thr"] is not None])
fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 2.9))
for ax, x, lim, lab in ((a, sp, TH["sp_std_max_dps"], "roll/pitch rate-setpoint std [deg/s]"),
                        (b, thr, TH["thr_std_max"], "std of mean motor command [-]")):
    if len(x):
        y = np.arange(1, len(x) + 1) / len(x) * 100
        ax.step(x, y, where="post", color=C1)
        ax.axvline(lim, color=INK2, lw=0.8, ls="--")
        kept = 100.0 * np.mean(x <= lim)
        ax.text(lim, 4, f"candidate {lim:g}: keeps {kept:.0f} % ", fontsize=7, color=INK2, ha="right")
        ax.set_xscale("log")
    ax.set_xlabel(lab)
    ax.set_ylabel("airborne windows below [%]")
a.set_title(f"setpoint activity ({len(sp)} windows of {TH['window_s']:.0f} s)", fontsize=9, loc="left")
b.set_title("throttle steadiness", fontsize=9, loc="left")
save(fig, "fig_p3_activity.png")

# 4. all pilot spectra (roll): absolute and mean-removed on the fixed band
keys = sorted({k.rsplit("|", 1)[0] for k in A.files})
vehicle = lambda k: k.split("|")[0]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.5, 3.2))
grid = np.geomspace(5, 1000, 400)
logs_abs, logs_rel = [], []
for k in keys:
    f, P = A[k + "|f"], A[k + "|P"][:, 0]
    m = (f >= 5) & (f <= min(1000, f[-1]))
    a.loglog(f[m], P[m], color=MUTED, lw=0.5, alpha=0.6)
    lp = np.interp(grid, f, np.log10(np.maximum(P, 1e-30)), right=np.nan)
    band = (grid >= TH["f_min"]) & (grid <= TH["f_max"])
    logs_abs.append(lp)
    logs_rel.append(lp - np.nanmean(lp[band]))
    b.semilogx(grid[band], (lp - np.nanmean(lp[band]))[band], color=MUTED, lw=0.5, alpha=0.6)
if keys:                                  # every log weighs the same: average its windows first
    per_log = {}
    for k, la, lr in zip(keys, logs_abs, logs_rel):
        per_log.setdefault(vehicle(k), []).append((la, lr))
    med_abs = np.nanmedian(np.array([np.nanmean([x[0] for x in v], axis=0) for v in per_log.values()]), axis=0)
    med_rel = np.nanmedian(np.array([np.nanmean([x[1] for x in v], axis=0) for v in per_log.values()]), axis=0)
    a.loglog(grid, 10 ** med_abs, color=C1, lw=1.6)
    band = (grid >= TH["f_min"]) & (grid <= TH["f_max"])
    b.semilogx(grid[band], med_rel[band], color=C1, lw=1.6)
    label_end(b, grid[band], med_rel[band], "median", C1)
for ax in (a, b):
    ax.axvspan(TH["f_min"], TH["f_max"], color=GRID, alpha=0.35, lw=0)
    ax.set_xlabel("frequency [Hz]")
a.set_ylabel("roll PSD [(rad/s)$^2$/Hz]")
b.set_ylabel("log10 PSD, band mean removed")
a.set_title(f"{len(keys)} steady windows of {len({vehicle(k) for k in keys})} logs, raw gyro (roll)", fontsize=9, loc="left")
b.set_title(f"clustering feature form ({TH['f_min']:.0f}-{TH['f_max']:.0f} Hz)", fontsize=9, loc="left")
save(fig, "fig_p4_spectra.png")
print("figures written to", FIG)
