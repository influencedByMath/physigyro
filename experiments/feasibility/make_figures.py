"""Figures for the feasibility report. Run after run_feasibility.py:
    .venv/Scripts/python experiments/feasibility/make_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
from filters import derivative_response, phase_delay, pt1_response   # noqa: E402
from loop_model import FilterSettings, gyro_stack   # noqa: E402
from sysid import plant_model                        # noqa: E402

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results.json")))
A = np.load(os.path.join(HERE, "arrays.npz"))

# categorical slots in fixed order (dataviz reference palette, light mode)
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "lines.linewidth": 1.4})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), dpi=150)
    plt.close(fig)


def settings(d):
    d = dict(d)
    d["notches"] = tuple(tuple(n) for n in d["notches"])
    return FilterSettings(**d)


# 1. raw gyro spectra with motor harmonics
fig, axes = plt.subplots(2, 1, figsize=(7.5, 5.2), sharex=True)
for ax, key in zip(axes, ["b27d09e3", "a2571040"]):
    d = R["data"][key]
    f, P = A[f"{key}_f"], A[f"{key}_P"]
    m = (f > 5) & (f < 1000)
    ax.semilogy(f[m], P[m, 0], color=C1, lw=1.0, label="roll")
    ax.semilogy(f[m], P[m, 1], color=C2, lw=1.0, label="pitch")
    if d["rpm_hz"]:
        fm = np.mean(d["rpm_hz"])
        for h in range(1, 10):
            if h * fm < 1000:
                ax.axvline(h * fm, color=INK2, lw=0.6, ls=":")
        ax.text(fm, P[m].max(), " 1st harmonic\n (ESC RPM)", fontsize=7, color=INK2, va="top")
    ax.axvspan(800, 1000, color=GRID, alpha=0.6, lw=0)
    ax.set_title(f"log {key} ({d['hw']}, FIFO {d['fs_measured']:.0f} Hz, window {d['window'][0]:.0f}-"
                 f"{d['window'][1]:.0f} s, {d['welch_averages']} Welch averages)", fontsize=8.5, loc="left")
    ax.set_ylabel("PSD [(rad/s)$^2$/Hz]")
    ax.legend(loc="upper right")
axes[-1].set_xlabel("frequency [Hz]   (dotted: multiples of the mean motor frequency; grey: above f_max = 800 Hz)")
save(fig, "fig1_raw_gyro_psd.png")

# 2. model vs logged (E10 filtered gyro, E11 torque), roll axis
f = A["E10_f"]
m = (f > 5) & (f < 400)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
for ax, k, title in [(axes[0], "rate", "filtered gyro (vehicle_angular_velocity)"),
                     (axes[1], "torque", "controller output (vehicle_torque_setpoint)")]:
    ax.semilogy(f[m], A[f"E10_meas_{k}"][m], color=INK, lw=1.6, label="logged by the flight controller")
    ax.semilogy(f[m], A[f"E10_pred_fixed_{k}"][m], color=C2, lw=1.1, label="model, fixed notch at mean RPM")
    ax.semilogy(f[m], A[f"E10_pred_track_{k}"][m], color=C1, lw=1.1, ls="--", label="model, tracking notch")
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xlabel("frequency [Hz]")
axes[0].set_ylabel("PSD (roll axis, 800 Hz data)")
axes[0].legend(loc="upper right", fontsize=7.5)
save(fig, "fig2_model_vs_logged.png")

# 3. plant identification
fig, axes = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True)
for name, col in (("roll", C1), ("pitch", C2)):
    p = R["plant"][name]
    f, G, sel = A[f"sysid_{name}_f"], A[f"sysid_{name}_G"], A[f"sysid_{name}_sel"].astype(bool)
    ff = np.geomspace(0.5, 30, 300)
    Gm = plant_model(ff, p["b"], p["tau_m"], p["Td"], 1 / R["data"]["b27d09e3"]["f_ctrl"])
    axes[0].loglog(f[sel], np.abs(G[sel]), "o", ms=4, color=col, label=f"{name}: measured (setpoint as instrument)")
    axes[0].loglog(ff, np.abs(Gm), color=col, lw=1.2,
                   label=f"{name}: fit b={p['b']:.0f}, tau_m={p['tau_m']*1e3:.0f} ms, Td={p['Td']*1e3:.1f} ms")
    ph = np.degrees(np.angle(G[sel]))
    axes[1].semilogx(f[sel], np.where(ph > 0, ph - 360, ph), "o", ms=4, color=col)
    axes[1].semilogx(ff, np.degrees(np.unwrap(np.angle(Gm))), color=col, lw=1.2)
axes[0].set_ylabel("|G| [rad/s per unit torque]")
axes[1].set_ylabel("phase [deg]")
axes[1].set_xlabel("frequency [Hz]")
axes[0].legend(fontsize=7)
save(fig, "fig3_plant_identification.png")

# 4. trade-off clouds and optima
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9))
for ax, tag, rowkey, title in [(axes[0], "real", "real_gain", "real gains + identified plant (ref = pilot setting)"),
                               (axes[1], "syn_B", None, "synthetic drone + real noise (ref = PX4 default)")]:
    pm, n = A[f"pareto_{tag}_PM"], A[f"pareto_{tag}_N"]
    ax.scatter(n, pm, s=3, color="#b7d3f6", lw=0, label="feasible settings (grid)")
    order = np.argsort(n)
    env = np.maximum.accumulate(pm[order])
    ax.plot(n[order], env, color=C1, lw=1.6, label="Pareto front")
    row = R["optimizer"]["real_gain"] if rowkey else next(r for r in R["optimizer"]["synthetic"]
                                                           if r.get("scenario") == "B")
    ref = row["reference"]
    ax.plot(ref["N_rel"], ref["PM"], "s", ms=7, color=INK, label="reference")
    for mode, col in (("A", C2), ("B", C3)):
        mm = row.get(f"mode_{mode}", {})
        if mm.get("feasible"):
            ax.plot(mm["metrics"]["N_rel"], mm["metrics"]["PM"], "o", ms=7, color=col, mec="#fcfcfb", mew=1.5,
                    label=f"mode {mode} optimum")
    if rowkey:
        d = row["px4_default"]
        ax.plot(d["N_rel"], d["PM"], "D", ms=6, color=C4, label="PX4 default (this drone)")
    ax.set_xscale("log")
    ax.set_xlabel("motor-command noise relative to reference")
    ax.set_title(title, fontsize=8.5, loc="left")
axes[0].set_ylabel("phase margin, nominal plant [deg]")
axes[0].legend(fontsize=7, loc="lower right")
save(fig, "fig4_tradeoff_and_optima.png")

# 5. gyro filter stack Bode: default, pilot, mode A optimum (real-gain case)
d = R["data"]["b27d09e3"]
motor = np.array(d["rpm_hz"])
FS = d["fs_nominal"]                 # filter rate of this log (gyro FIFO rate)
ff = np.geomspace(1, 1000, 3000)
rows = [("PX4 default 40/30 Hz", settings(d["default"]), INK2),
        ("pilot: 80/40 Hz + RPM notch", settings(d["actual"]), C2)]
ma = R["optimizer"]["real_gain"].get("mode_A", {})
if ma.get("feasible"):
    s = settings(d["actual"])
    lab = ma["option"]
    if "off" in lab:
        s = FilterSettings(ma["f_gyro"], ma["f_dterm"])
    else:
        bw, h = float(lab.split("bw ")[1].split(" Hz")[0]), int(lab.split("x")[-1])
        s = FilterSettings(ma["f_gyro"], ma["f_dterm"], dnf_esc=True, dnf_bw=bw, dnf_hmc=h, dnf_min=s.dnf_min)
    rows.append((f"mode A optimum: {ma['f_gyro']:.0f}/{ma['f_dterm']:.0f} Hz, {lab}", s, C1))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
fl = np.linspace(1, 50, 200)
for lab, s, col in rows:
    H = gyro_stack(ff, s, FS, motor)
    axes[0].semilogx(ff, 20 * np.log10(np.abs(H)), color=col, label=lab)
    Hg = gyro_stack(fl, s, FS, motor)
    Hd = Hg * derivative_response(fl, FS) * pt1_response(fl, s.f_dterm, FS) / (2j * np.pi * fl)
    axes[1].plot(fl, phase_delay(Hg, fl) * 1e3, color=col)
    axes[1].plot(fl, phase_delay(Hd, fl) * 1e3, color=col, ls="--")
axes[0].set_ylabel("|F_g| [dB]")
axes[0].set_ylim(-60, 5)
axes[0].set_xlabel("frequency [Hz]   (gyro path; RPM notches at the mean motor speeds)")
axes[0].legend(fontsize=7.5, loc="lower left")
axes[1].set_ylabel("equivalent delay [ms]")
axes[1].set_xlabel("frequency [Hz]   (solid: gyro path, P and I;  dashed: D-term path)")
axes[1].set_ylim(0, None)
save(fig, "fig5_filter_stack_bode.png")
print("figures written to", FIG)
