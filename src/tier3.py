"""Tier 3 synthetic drones with the decided priors, gain design and limits (THESIS 7.1.1 Tier 3; D-28, D-33, D-36, D-37).

Per draw (fixed order, so that a seed gives the same drones):
1. design parameters from the design generator: crossover f_c, motor time constant tau_m, extra delay Td (net of the
   chip filter), D corner ratio r (log-uniform / uniform as in THESIS 7.1.1) and an empirical (FIFO rate, controller
   rate) pair (D-28);
2. a noise spectrum from ``draw_spectrum`` (its own random generator); it fixes the IMU chip (D-26: the chip of the
   source vehicle);
3. the chip delay of D-33: the lower bound is added to the nominal Td, the rest of the fit-error range to the delay
   step of the uncertainty set; the PID gains are designed on that nominal plant (crossover design, unfiltered PM in
   the D-36 window, integral corner 0.03);
4. Tier 3 step 3: the drone is kept if its PX4 v1.16 default filters (D-09) meet the nominal limits (D-30, D-31);
   a kept drone that fails the worst-case limits of D-36 over the uncertainty set is flagged, not dropped (D-37 (6)).
"""
import math

import numpy as np

import robust as rb
from loop_model import Gains, Plant, PX4_V116_MC_DEFAULT, Rates, design_gains

F_C = (3.0, 10.0)                                 # D-36 (5) [Hz], log-uniform
TAU_M = (0.015, 0.060)                            # D-36 (3) [s], log-uniform
TD = (0.0003, 0.008)                              # D-36 (3) [s], log-uniform, net of the chip filter
PM_WINDOW = (60.0, 95.0)                          # D-36 (4) unfiltered PM [deg]
I_RATIO = 0.03                                    # D-36 (4) integral corner / crossover
R_RANGE = (0.5, 3.0)                              # THESIS 7.1.1 Tier 3a: D corner / crossover, uniform
WIDTH = 0.20                                      # D-36 (1): delta_b = delta_tau
DELAY_SAMPLES = 1.0                               # D-36 (1): delay step in controller samples
MS_WC = 3.0                                       # D-36 (1)
NOMINAL = dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0)   # D-30, D-31, Section 7.8


def worst_case_limits(ms_wc=MS_WC):
    """Limits for every plant of the set implied by Ms_wc (D-32): PM >= 2 asin(1/(2 Ms)), GM >= 20 log10(Ms/(Ms-1))."""
    return dict(PM=math.degrees(2 * math.asin(1 / (2 * ms_wc))), Ms=ms_wc, GM=20 * math.log10(ms_wc / (ms_wc - 1)))


def limits():
    """Keyword arguments of robust.passes: nominal limits and the separate worst-case limits (D-32, D-36)."""
    return dict(NOMINAL, wc=worst_case_limits())


def rate_pairs(rate_rows):
    """(FIFO rate, controller rate) pairs of the logs whose FIFO IMU is the control IMU, logged at full rate (D-28)."""
    return [(float(r["fs_nominal"]), float(r["pub_rate"])) for r in rate_rows
            if r["fifo_is_control"] and r["full_rate"] and r["pub_rate"]]


def chip_delays(chip_delay_table, lowpass_terms, rule="rms"):
    """{chip: (lower, upper) delay [s]} for the chips with a fitted low-pass term; ``rule`` 'rms' is the fit-error rule
    of D-33 (priors_from_logs.json 'chip_delay'); ``lowpass_terms`` maps chip -> term or None (generator)."""
    return {c: tuple(x * 1e-3 for x in chip_delay_table[c][rule]["delay_ms"])
            for c, term in lowpass_terms.items() if term is not None}


def plant_and_step(tau_m, td, f_ctrl, chip, delays):
    """Nominal plant (b = 1; Td + the chip's lower delay bound) and the delay step of the set (1 controller sample
    + the rest of the chip-delay range) (D-33, D-36 (1))."""
    lo, hi = delays.get(chip, (0.0, 0.0))
    return Plant(1.0, tau_m, td + lo), DELAY_SAMPLES / f_ctrl + (hi - lo)


def default_check(d, delays, reference=PX4_V116_MC_DEFAULT):
    """robust_metrics of drone d at its default filters; (kept by Tier 3 step 3, flagged by the worst case, metrics)."""
    p, dd = plant_and_step(d["tau_m"], d["Td"], d["f_ctrl"], d["chip"], delays)
    rm = rb.robust_metrics(p, Gains(**d["gains"]), Rates(d["fs"], d["f_ctrl"]), reference, None, WIDTH, WIDTH, dd)
    n = rm["nominal"]
    kept = bool(n["stable"] and n["PM"] >= NOMINAL["PM_min"] and n["Ms"] <= NOMINAL["Ms_max"]
                and n["GM"] >= NOMINAL["GM_min"])
    return kept, not rb.passes(rm, **limits()), rm


def sample_drones(rate_rows, n, seed, draw_spectrum, delays, keep_all=False, max_tries=None):
    """Synthetic drones of Tier 3 (D-36). ``draw_spectrum(rng)`` returns dict(chip, motor_hz, S, meta).
    Returns dict(drones, draws, in_window, kept): n kept drones, or with ``keep_all`` the first n drones inside the
    PM window, each marked 'kept'."""
    pairs = rate_pairs(rate_rows)
    rng = np.random.default_rng(seed)                 # design draws (same order as f_options.synthetic_population)
    rng_s = np.random.default_rng([seed, 1])          # spectrum draws
    max_tries = 400 * n if max_tries is None else max_tries
    drones, draws, in_window, kept = [], 0, 0, 0
    while len(drones) < n:
        draws += 1
        if draws > max_tries:
            raise RuntimeError(f"only {len(drones)} of {n} drones in {max_tries} draws")
        fc = math.exp(rng.uniform(*np.log(F_C)))
        tau_m = math.exp(rng.uniform(*np.log(TAU_M)))
        td = math.exp(rng.uniform(*np.log(TD)))
        r = rng.uniform(*R_RANGE)
        fs, fctrl = pairs[int(rng.integers(len(pairs)))]
        spec = draw_spectrum(rng_s)
        p, dd = plant_and_step(tau_m, td, fctrl, spec["chip"], delays)
        g, m = design_gains(fc, r, p, Rates(fs, fctrl), i_ratio=I_RATIO)
        pm = float(m["PM"])
        if not PM_WINDOW[0] <= pm <= PM_WINDOW[1]:
            continue
        in_window += 1
        d = dict(f_c=fc, tau_m=tau_m, Td=td, Td_nominal=p.Td, delay_step=dd, r=r, fs=fs, f_ctrl=fctrl,
                 pm_unfiltered=pm, gains=dict(Kp=g.Kp, Ki=g.Ki, Kd=g.Kd), chip=spec["chip"],
                 chip_delay=list(delays.get(spec["chip"], (0.0, 0.0))), motor_hz=spec["motor_hz"], S=spec["S"],
                 meta=spec["meta"])
        ok, flagged, rm = default_check(d, delays)
        kept += ok
        if ok or keep_all:
            d.update(kept=ok, flagged=flagged, default=dict(nominal=rm["nominal"], worst_corners=rm["worst_corners"]))
            drones.append(d)
    return dict(drones=drones, draws=draws, in_window=in_window, kept=kept)
