"""Fig. 1 (experiment E1): magnitude and delay of the PX4 v1.16 multicopter gyro filter stack (Phase 4 Task 2; D-29).

    .venv/Scripts/python experiments/phase4/fig1_bode.py

Exactly the filter chain of the loop model (Review Focus 3): src/loop_model.gyro_stack (ESC-RPM notch bank, then
the 2nd-order gyro low-pass IMU_GYRO_CUTOFF) and the D-term path F_g F_d D(z) / (j 2 pi f) of LoopCase.evaluate
(PT1 IMU_DGYRO_CUTOFF on the backward-difference derivative), all at the loop model's filter rate Rates().fs.
Two panels (author's decision D-29): (a, main) PX4_V116_MC_DEFAULT (D-09), ESC-RPM notch bank off; (b) the same stack
with the bank on at its default BW / HMC / MIN, for four motors at the fit-half median hover frequency
(priors_from_logs.json; only 6 vehicles with valid RPM). Delay = phase lag / (2 pi f) (THESIS 4.4, the definition of
tau_F). Marked: crossover range f_c of D-28 and the noise band 5 f_c ... 780 Hz (D-23 f_max).
Writes figures/fig1_filter_stack.{png,pdf} and the caption figures/fig1_caption.md (numbers computed, not typed).
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
VARIANTS = {"off": lm.PX4_V116_MC_DEFAULT, "on": SETTINGS}    # PX4 default (ESC bank off) and with the bank on
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


def _style(ax):
    ax.axvspan(*FC_RANGE, color=BAND, zorder=0)
    ax.axvspan(5 * FC_RANGE[0], NOISE_BAND_MAX, facecolor=BAND, alpha=0.55, zorder=0, hatch="////",
               edgecolor="#d5d8dc", linewidth=0)
    ax.axvspan(5 * FC_RANGE[1], NOISE_BAND_MAX, color=BAND, zorder=0)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xscale("log")


def _draw(ax1, ax2, s, f):
    """Magnitude (ax1) and delay (ax2) of every filter and path of setting s; returns the legend handles."""
    r = responses(f, FS, s, MOTOR_HZ)
    series = [  # key, label, colour, line style, width
        ("gyro_lpf", f"gyro low-pass, 2nd order, {s.f_gyro:.0f} Hz", BLUE, (0, (4, 2)), 1.2),
        ("esc_bank", f"ESC-RPM notch bank, 4 motors x {s.dnf_hmc} harmonics, BW {s.dnf_bw:.0f} Hz",
         AQUA, (0, (4, 2)), 1.2),
        ("dterm_pt1", f"D-term low-pass, PT1, {s.f_dterm:.0f} Hz", ORANGE, (0, (4, 2)), 1.2),
        ("gyro_path", "gyro path $F_g$ (P and I terms)", BLUE, "-", 2.0),
        ("dterm_path", "D-term path $F_g F_d$ (with the derivative)", ORANGE, "-", 2.0),
    ]
    if not s.dnf_esc and not s.notches:
        series = [x for x in series if x[0] != "esc_bank"]
    for ax in (ax1, ax2):
        _style(ax)
    mag = {k: 20 * np.log10(np.maximum(np.abs(r[k]), 1e-12)) for k, *_ in series}
    dl = delays_ms(f, FS, s, MOTOR_HZ)
    for k, lab, col, ls, lw in series:
        ax1.plot(f, mag[k], color=col, linestyle=ls, linewidth=lw, label=lab)
        ax2.plot(f, dl[k], color=col, linestyle=ls, linewidth=lw)
    ax1.set_ylim(-60, 6)
    ax2.set_ylim(-2, 16)
    ax2.set_xlim(1.0, 1000.0)
    for k, dy in (("gyro_path", 0.5), ("dterm_path", 0.9)):
        y = np.interp(2.0, f, dl[k])
        ax2.text(1.15, y + dy, "gyro path" if k == "gyro_path" else "D-term path", color=INK, fontsize=8.5)
    for k, txt, x in (("gyro_lpf", "gyro LPF", 300.0), ("dterm_pt1", "D-term PT1", 300.0)):
        ax1.text(x, np.interp(x, f, mag[k]) + 1.5, txt, color=INK, fontsize=8)
    if s.dnf_esc:
        ax1.text(330.0, -5.0, "ESC notches", color=INK, fontsize=8, ha="left")
    ax1.text(np.sqrt(FC_RANGE[0] * FC_RANGE[1]), 2.0, FC_LABEL, color=MUTED, fontsize=8, ha="center")
    ax1.text(np.sqrt(5 * FC_RANGE[0] * NOISE_BAND_MAX), 2.0, "noise band $5 f_c$ ... 780 Hz", color=MUTED,
             fontsize=8, ha="center")
    from matplotlib.ticker import FuncFormatter
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    return ax1.get_legend_handles_labels()


def _probe(s):
    probe = np.array([4.0, 10.0, 15.0])
    return {k: [round(float(x), 3) for x in v] for k, v in delays_ms(probe, FS, s, MOTOR_HZ).items()}


def bank_delay_ms(f):
    """Delay [ms] of the ESC-RPM notch bank alone (panel b settings) at frequencies f."""
    return delays_ms(np.asarray(f, dtype=float), FS, SETTINGS, MOTOR_HZ)["esc_bank"]


def caption():
    """Caption of Fig. 1; every number is computed from the plotted settings."""
    d = bank_delay_ms(np.array(FC_RANGE))
    lag = 360.0 * np.array(FC_RANGE) * d * 1e-3
    f0 = flt.esc_notch_frequencies(MOTOR_HZ, SETTINGS.dnf_hmc, SETTINGS.dnf_bw, SETTINGS.dnf_min)
    centres = sorted(set(np.round(f0, 1)))
    s = VARIANTS["off"]
    return (f"Fig. 1. Magnitude (top) and delay (bottom; phase lag / 2πf) of the PX4 v1.16 multicopter gyro filter "
            f"stack, computed with the loop model's own filter functions at its filter rate ({FS:.0f} Hz). "
            f"(a) PX4 default (D-09): gyro low-pass {s.f_gyro:.0f} Hz (2nd order) and D-term low-pass {s.f_dterm:.0f} Hz "
            f"(PT1); the ESC-RPM notch bank is off. (b) The same stack with the ESC-RPM notch bank on at its defaults "
            f"(bandwidth {SETTINGS.dnf_bw:.0f} Hz, {SETTINGS.dnf_hmc} harmonics) for 4 motors at {MOTOR_HZ[0]:.1f} Hz, the "
            f"fit-half median hover frequency. Shaded: the crossover band f_c = {FC_RANGE[0]:g}-{FC_RANGE[1]:g} Hz (D-28) "
            f"and the noise band from 5 f_c to {NOISE_BAND_MAX:.0f} Hz (hatched from {5 * FC_RANGE[0]:g} to "
            f"{5 * FC_RANGE[1]:g} Hz, where its start depends on f_c). The spikes in the delay curves of (b) sit at the "
            f"notch centres ({', '.join(f'{x:.1f}' for x in centres[:-1])} and {centres[-1]:.1f} Hz): the phase of each "
            f"notch swings from lag to lead where its magnitude is almost zero. They lie far above the crossover band, so "
            f"they do not affect the stability margins at crossover. The bank does lower the margins a little: the tails "
            f"of its {len(f0)} notches add {d.mean():.1f} ms of delay throughout the crossover band ({lag[0]:.1f}° of "
            f"phase lag at {FC_RANGE[0]:g} Hz, {lag[1]:.1f}° at {FC_RANGE[1]:g} Hz). Whenever the bank is on, the margins "
            f"in this work include this lag, because the figure and the loop model use the same code.")


def _save(fig, out_dir, name):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_figure(out_dir):
    """Fig. 1 (D-29): (a, main) the PX4 default with the ESC bank off next to (b) the bank on (same axes and scales).
    Also writes the caption. Returns the panel order and the delay table of each panel at 4, 10, 15 Hz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    f = np.geomspace(1.0, 1000.0, 4000)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 6.4), sharex=True, sharey="row",
                             gridspec_kw=dict(hspace=0.12, wspace=0.06))
    handles = {}
    panels = (("off", "PX4 v1.16 default (D-09): ESC-RPM notch bank off"),
              ("on", "ESC-RPM notch bank on (defaults: BW 15 Hz, 3 harmonics)"))
    for col, (key, title) in enumerate(panels):
        h, lab = _draw(axes[0, col], axes[1, col], VARIANTS[key], f)
        handles.update(zip(lab, h))
        axes[0, col].set_title(f"({'ab'[col]}) {title}", color=INK, fontsize=10, loc="left")
        axes[1, col].set_xlabel("Frequency [Hz]", color=MUTED)
    axes[0, 0].set_ylabel("Magnitude [dB]", color=MUTED)
    axes[1, 0].set_ylabel("Delay = phase lag / $2\\pi f$ [ms]", color=MUTED)
    fig.legend(list(handles.values()), list(handles.keys()), frameon=False, fontsize=8, labelcolor=INK,
               loc="upper left", ncol=3, bbox_to_anchor=(0.06, 0.035))
    fig.text(0.06, -0.06, f"Filters at {FS:.0f} Hz; motors at {MOTOR_HZ[0]:.1f} Hz (fit-half median hover; panel b only). "
             "Hatched: noise band starts between 20 and 75 Hz (5 f_c). In (a) the gyro path equals the gyro low-pass "
             "(its dashed line lies under the solid one).", color=MUTED, fontsize=7)
    _save(fig, out_dir, "fig1_filter_stack")
    with open(os.path.join(out_dir, "fig1_caption.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(caption() + "\n")
    return dict(panels=[k for k, _ in panels], **{k: _probe(VARIANTS[k]) for k, _ in panels})


if __name__ == "__main__":
    d = make_figure(os.path.join(HERE, "figures"))
    print("delay [ms] at f = 4, 10, 15 Hz:", json.dumps(d))
    print(caption())
