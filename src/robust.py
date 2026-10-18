"""Robust stability of one setting over a plant uncertainty set (THESIS 7.1.1 Tier 3 step 4, 7.8 steps 2-3; D-12).

The uncertainty set is the nominal plant (b, tau_m, Td) plus the 8 corners b (1 +- delta_b), tau_m (1 +- delta_tau),
Td + {0, delta_d}. The open loop is the one of loop_model.LoopCase.evaluate (same functions), so the margins here
are the margins the optimizer and the safety check use. Used for the evidence pack of decisions A2-A4.
"""
import numpy as np

from loop_model import (Plant, _stability_grid, derivative_response, gyro_stack, integrator_response, margins,
                        n_integrators, plant_response, pt1_response, zoh_response)


def stability_grid(rates, n=1500):
    """The loop model's stability grid (LoopCase default)."""
    return _stability_grid(rates, n)


def open_loop(f, plant, gains, rates, s, motor_hz=None):
    """L(f) = F_g [Kp + Ki I(z) + Kd D(z) F_d] ZOH P, as in LoopCase.evaluate."""
    A = gains.Kp + gains.Ki * integrator_response(f, rates.f_ctrl)
    Dd = gains.Kd * derivative_response(f, rates.fs)
    Fg = gyro_stack(f, s, rates.fs, motor_hz)
    return Fg * (A + Dd * pt1_response(f, s.f_dterm, rates.fs)) * zoh_response(f, rates.f_ctrl) * plant_response(f, plant)


def corner_plants(p, delta_b, delta_tau, delta_d):
    """[nominal] + 8 corners: b (1 +- delta_b), tau_m (1 +- delta_tau), Td + {0, delta_d} [s]."""
    return [p] + [Plant(p.b * (1 + sb * delta_b), p.tau_m * (1 + st * delta_tau), p.Td + dd)
                  for sb in (-1, 1) for st in (-1, 1) for dd in (0.0, float(delta_d))]


def critical_frequencies(f, L):
    """(frequency of the peak sensitivity max |1/(1+L)|, frequency of the phase crossover that sets the smallest
    gain margin) of one open loop on grid f; the second is NaN without a phase crossover below |L| = 1."""
    f = np.asarray(f, dtype=float)
    f_ms = float(f[np.argmin(np.abs(1.0 + L))])
    ph = np.unwrap(np.angle(L))
    lm = np.log(np.abs(L))
    u = np.floor((ph + np.pi) / (2 * np.pi))
    best, f180 = np.inf, float("nan")
    for i in np.flatnonzero(u[:-1] != u[1:]):
        v = 2 * np.pi * max(u[i], u[i + 1]) - np.pi
        t = (v - ph[i]) / (ph[i + 1] - ph[i])
        lmc = lm[i] + t * (lm[i + 1] - lm[i])
        if lmc < 0 and -lmc < best:
            best = -lmc
            f180 = float(np.exp(np.log(f[i]) + t * (np.log(f[i + 1]) - np.log(f[i]))))
    return f_ms, f180


def robust_metrics(plant, gains, rates, s, motor_hz, delta_b, delta_tau, delta_d, n_stab=1500):
    """Margins of the nominal plant, the worst case over all 9 plants and over the 8 corners only."""
    f = stability_grid(rates, n_stab)
    plants = corner_plants(plant, delta_b, delta_tau, delta_d)
    L = np.stack([open_loop(f, p, gains, rates, s, motor_hz) for p in plants])
    m = margins(f, L, n_integrators(gains))

    def worst(sl):
        return dict(PM=float(np.min(m["PM"][sl])), GM=float(np.min(m["GM"][sl])), Ms=float(np.max(m["Ms"][sl])),
                    stable=bool(np.all(m["stable"][sl])))

    nominal = {k: (bool(m[k][0]) if k == "stable" else float(m[k][0])) for k in ("PM", "GM", "Ms", "fc", "stable")}
    nominal["f_Ms"], nominal["f_180"] = critical_frequencies(f, L[0])
    return dict(nominal=nominal, worst=worst(slice(None)), worst_corners=worst(slice(1, None)))


def corner_metrics(plant, gains, rates, s, motor_hz, delta_b, delta_tau, delta_d, n_stab=1500):
    """Margins and critical frequencies (f_Ms, f_180 of critical_frequencies) of every plant of the set, in the
    order of corner_plants: the nominal plant, then the corners labelled like 'b+ tau- Td+d'."""
    f = stability_grid(rates, n_stab)
    plants = corner_plants(plant, delta_b, delta_tau, delta_d)
    L = np.stack([open_loop(f, p, gains, rates, s, motor_hz) for p in plants])
    m = margins(f, L, n_integrators(gains))
    labels = ["nominal"] + [f"b{'-+'[sb > 0]} tau{'-+'[st > 0]} {'Td+d' if dd else 'Td'}"
                            for sb in (-1, 1) for st in (-1, 1) for dd in (0, 1)]
    rows = []
    for i, p in enumerate(plants):
        f_ms, f180 = critical_frequencies(f, L[i])
        rows.append(dict(label=labels[i], b=p.b, tau_m=p.tau_m, Td=p.Td, PM=float(m["PM"][i]), GM=float(m["GM"][i]),
                         Ms=float(m["Ms"][i]), fc=float(m["fc"][i]), stable=bool(m["stable"][i]), f_Ms=f_ms, f_180=f180))
    return rows


def passes(rm, PM_min, Ms_max, GM_min=6.0, wc=None):
    """True if the nominal plant meets (PM_min, Ms_max, GM_min) and every plant of the set meets the worst-case
    limits: the same ones (wc None) or separate, looser ones wc = dict(PM=, Ms=, GM=). Every plant must be
    Nyquist-stable."""
    n, w = rm["nominal"], rm["worst"]
    lim = wc or dict(PM=PM_min, Ms=Ms_max, GM=GM_min)
    ok_nom = n["stable"] and n["PM"] >= PM_min and n["Ms"] <= Ms_max and n["GM"] >= GM_min
    ok_wc = w["stable"] and w["PM"] >= lim["PM"] and w["Ms"] <= lim["Ms"] and w["GM"] >= lim["GM"]
    return bool(ok_nom and ok_wc)


def control_imu_raw_rate(selected_device_id, statuses):
    """Raw (FIFO) rate of the control IMU: the vehicle_imu_status entry whose gyro_device_id is the selected gyro
    (sensor_selection). statuses: [(gyro_device_id, gyro_raw_rate_hz), ...]. None if it cannot be matched."""
    if selected_device_id is None:
        return None
    for dev, rate in statuses:
        if int(dev) == int(selected_device_id):
            return float(rate)
    return None
