"""Fig. 1 (experiment E1): magnitude and delay of the PX4 v1.16 multicopter gyro filter stack (Phase 4 Task 2).

    .venv/Scripts/python experiments/phase4/fig1_bode.py

Exactly the filter chain of the loop model (Review Focus 3): src/loop_model.gyro_stack (ESC-RPM notch bank, then
the 2nd-order gyro low-pass IMU_GYRO_CUTOFF) and the D-term path F_g F_d D(z) / (j 2 pi f) of LoopCase.evaluate
(PT1 IMU_DGYRO_CUTOFF on the backward-difference derivative), all at the loop model's filter rate Rates().fs.
Settings: PX4_V116_MC_DEFAULT (D-09), with the ESC-RPM bank switched on at its default BW / HMC / MIN, for four
motors at the fit-half median hover frequency (priors_from_logs.json; only 6 vehicles with valid RPM).
Delay = phase lag / (2 pi f) (THESIS 4.4, the definition of tau_F). Marked: crossover range f_c of D-28 and the
noise band 5 f_c ... 780 Hz (D-23 f_max). Writes figures/fig1_filter_stack.{png,pdf}.
"""
import json
import os
import sys
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import filters as flt               # noqa: E402
import loop_model as lm             # noqa: E402

SETTINGS = replace(lm.PX4_V116_MC_DEFAULT, dnf_esc=True)
FS = lm.Rates().fs
_HOVER = json.load(open(os.path.join(HERE, "priors_from_logs.json")))["hover_motor"]["quantiles"]["0.5"]
MOTOR_HZ = np.full(4, float(_HOVER))
FC_RANGE = (4.0, 15.0)              # D-28
FC_LABEL = "crossover $f_c$ (D-28)"
NOISE_BAND_MAX = 780.0              # D-23

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated (dataviz validator, all pairs)
INK, MUTED, GRID, BAND = "#1f2328", "#5b6470", "#e6e8eb", "#f1f2f4"


def responses(f, fs, s, motor_hz):
    """Complex responses of each filter and of the two paths on frequencies f."""
    f = np.asarray(f, dtype=float)
    Fg = lm.gyro_stack(f, s, fs, motor_hz)
    Fd = flt.pt1_response(f, s.f_dterm, fs)
    return dict(gyro_lpf=flt.lowpass2_response(f, s.f_gyro, fs),
                esc_bank=lm.notch_stack(f, s, fs, motor_hz),
                dterm_pt1=Fd,
                gyro_path=Fg,
                dterm_path=Fg * Fd * flt.derivative_response(f, fs) / (2j * np.pi * f))


def factors(f, fs, s, motor_hz):
    """The same responses as lists of first/second-order factors (each notch, the low-pass, the PT1, D(z)/(j 2 pi f)).

    Every factor's phase stays inside (-180, 180] deg below fs/2, so the phase of a path is the SUM of the factors'
    principal phases. Unwrapping the product instead fails at a notch centre: the zero on the unit circle makes the
    phase jump by 180 deg, and np.unwrap can turn the lead above the notch into a 270 deg lag."""
    f = np.asarray(f, dtype=float)
    notches = [flt.notch_response(f, f0, bw, fs) for f0, bw in s.notches]
    if s.dnf_esc and motor_hz is not None:
        notches += [flt.notch_response(f, f0, s.dnf_bw, fs)
                    for f0 in flt.esc_notch_frequencies(motor_hz, s.dnf_hmc, s.dnf_bw, s.dnf_min)]
    lpf = flt.lowpass2_response(f, s.f_gyro, fs)
    pt1 = flt.pt1_response(f, s.f_dterm, fs)
    der = flt.derivative_response(f, fs) / (2j * np.pi * f)
    return dict(gyro_lpf=[lpf], esc_bank=notches or [np.ones(len(f), complex)], dterm_pt1=[pt1],
                gyro_path=notches + [lpf], dterm_path=notches + [lpf, pt1, der])


def delays_ms(f, fs, s, motor_hz):
    """Delay in ms of each filter and path: (sum of the factors' phase lags) / (2 pi f) (THESIS 4.4)."""
    f = np.asarray(f, dtype=float)
    return {k: 1e3 * sum(-np.angle(h) for h in facs) / (2 * np.pi * f) for k, facs in factors(f, fs, s, motor_hz).items()}


def make_figure(out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.geomspace(1.0, 1000.0, 4000)
    r = responses(f, FS, SETTINGS, MOTOR_HZ)
    series = [  # key, label, colour, line style, width
        ("gyro_lpf", f"gyro low-pass, 2nd order, {SETTINGS.f_gyro:.0f} Hz", BLUE, (0, (4, 2)), 1.2),
        ("esc_bank", f"ESC-RPM notch bank, 4 motors x {SETTINGS.dnf_hmc} harmonics, BW {SETTINGS.dnf_bw:.0f} Hz",
         AQUA, (0, (4, 2)), 1.2),
        ("dterm_pt1", f"D-term low-pass, PT1, {SETTINGS.f_dterm:.0f} Hz", ORANGE, (0, (4, 2)), 1.2),
        ("gyro_path", "gyro path $F_g$ (P and I terms)", BLUE, "-", 2.0),
        ("dterm_path", "D-term path $F_g F_d$ (with the derivative)", ORANGE, "-", 2.0),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw=dict(hspace=0.12))
    for ax in (ax1, ax2):
        ax.axvspan(*FC_RANGE, color=BAND, zorder=0)
        ax.axvspan(5 * FC_RANGE[0], NOISE_BAND_MAX, facecolor=BAND, alpha=0.55, zorder=0, hatch="////",
                   edgecolor="#d5d8dc", linewidth=0)
        ax.axvspan(5 * FC_RANGE[1], NOISE_BAND_MAX, color=BAND, zorder=0)
        ax.grid(True, which="major", color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_xscale("log")
    mag = {k: 20 * np.log10(np.maximum(np.abs(r[k]), 1e-12)) for k, *_ in series}
    dl = delays_ms(f, FS, SETTINGS, MOTOR_HZ)
    for k, lab, col, ls, lw in series:
        ax1.plot(f, mag[k], color=col, linestyle=ls, linewidth=lw, label=lab)
        ax2.plot(f, dl[k], color=col, linestyle=ls, linewidth=lw)
    ax1.set_ylim(-60, 6)
    ax1.set_ylabel("Magnitude [dB]", color=MUTED)
    ax1.set_title("(a) Magnitude", color=INK, fontsize=10, loc="left")
    ax2.set_ylim(-2, 16)
    ax2.set_ylabel("Delay = phase lag / $2\\pi f$ [ms]", color=MUTED)
    ax2.set_title("(b) Delay", color=INK, fontsize=10, loc="left")
    ax2.set_xlabel("Frequency [Hz]", color=MUTED)
    ax2.set_xlim(1.0, 1000.0)
    # direct labels on the two paths (delay panel, inside the crossover band) and band captions
    for k, dy in (("gyro_path", 0.5), ("dterm_path", 0.9)):
        y = np.interp(2.0, f, dl[k])
        ax2.text(1.15, y + dy, "gyro path" if k == "gyro_path" else "D-term path", color=INK, fontsize=8.5)
    for k, txt, x in (("gyro_lpf", "gyro LPF", 300.0), ("dterm_pt1", "D-term PT1", 300.0)):
        ax1.text(x, np.interp(x, f, mag[k]) + 1.5, txt, color=INK, fontsize=8)
    ax1.text(330.0, -5.0, "ESC notches", color=INK, fontsize=8, ha="left")
    ax1.text(np.sqrt(FC_RANGE[0] * FC_RANGE[1]), 2.0, FC_LABEL, color=MUTED, fontsize=8, ha="center")
    ax1.text(np.sqrt(5 * FC_RANGE[0] * NOISE_BAND_MAX), 2.0, "noise band $5 f_c$ ... 780 Hz", color=MUTED,
             fontsize=8, ha="center")
    from matplotlib.ticker import FuncFormatter
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    h, lab = ax1.get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK, loc="upper left", ncol=2,
               bbox_to_anchor=(0.07, 0.035))
    fig.text(0.07, -0.085, f"PX4 v1.16 multicopter defaults (ESC-RPM bank on at its defaults), filters at {FS:.0f} Hz; "
             f"motors at {MOTOR_HZ[0]:.1f} Hz (fit-half median hover). Hatched: noise band starts between 20 and 75 Hz.",
             color=MUTED, fontsize=7)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"fig1_filter_stack.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    probe = np.array([4.0, 10.0, 15.0])
    return {k: [round(float(x), 3) for x in v] for k, v in delays_ms(probe, FS, SETTINGS, MOTOR_HZ).items()}


if __name__ == "__main__":
    d = make_figure(os.path.join(HERE, "figures"))
    print("delay [ms] at f = 4, 10, 15 Hz:", json.dumps(d))
