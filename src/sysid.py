"""Tier 2 plant identification from one log (THESIS.md Section 7.1.1).

Feasibility finding (2026-10-10): the time-domain ARX fit planned in 7.1.1
(equation error, least squares) gives negative held-out VAF on real closed-loop
data, because the torque command is correlated with the disturbances through
the feedback. This module uses the standard closed-loop alternative: the
external rate setpoint r is the instrument, and the plant frequency response is

    G(f) = S_r,omega(f) / S_r,u(f)        (u = torque setpoint, omega = body rate)

estimated with Welch cross-spectra where the coherence is high.

Choice of instrument (audit 2026-10-14, decision P2): the instrument must not
depend on the measured vehicle motion. In Acro the rate setpoint comes from the
sticks and is valid. In Stabilized/Altitude mode the rate setpoint comes from
the attitude controller, which feeds back the measured attitude, so the strictly
external signal is the attitude setpoint (``instrument="att_sp"``). On log
b27d09e3 (Stabilized) both gave the same FRF within 0.2 dB / 2 deg. In
Position/auto modes even the attitude setpoint is feedback (position loop):
check that the two instruments agree before accepting a fit.

The model
    G(s) = b / s * 1 / (tau_m s + 1) * exp(-s Td) * ZOH(s)
is fitted to log G (magnitude and phase), weighted by coherence.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import coherence, csd

from noise_data import GyroStream, _t


def _attitude_setpoint_angle(ulog, axis):
    a = ulog.get_dataset("vehicle_attitude_setpoint").data
    w, x, y, z = (a[f"q_d[{i}]"].astype(float) for i in range(4))
    if axis == 0:
        ang = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    elif axis == 1:
        ang = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    else:
        raise ValueError("att_sp instrument only for roll (0) and pitch (1)")
    return _t(ulog, a["timestamp"]), ang


def prepare(ulog, gs: GyroStream, t0, t1, axis=0, instrument="rate_sp"):
    """Signals on the torque-setpoint time base (controller rate): r, u, omega.

    instrument: "rate_sp" (vehicle_rates_setpoint) or "att_sp" (roll/pitch
    angle of vehicle_attitude_setpoint), both held between messages."""
    q = ulog.get_dataset("vehicle_torque_setpoint").data
    tq = _t(ulog, q["timestamp_sample"])
    m = (tq >= t0) & (tq < t1)
    tq, u = tq[m], q[f"xyz[{axis}]"][m].astype(float)
    Tc = float(np.median(np.diff(tq)))
    if instrument == "rate_sp":
        rs = ulog.get_dataset("vehicle_rates_setpoint").data
        tr, rv = _t(ulog, rs["timestamp"]), rs[["roll", "pitch", "yaw"][axis]].astype(float)
    elif instrument == "att_sp":
        tr, rv = _attitude_setpoint_angle(ulog, axis)
    else:
        raise ValueError(f"unknown instrument {instrument!r}")
    k = np.clip(np.searchsorted(tr, tq, side="right") - 1, 0, len(tr) - 1)   # zero-order hold
    r = rv[k]
    seg = next(((a, b) for a, b in gs.segments if gs.t[a] <= t0 and gs.t[b - 1] >= t1), None)
    if seg is None:
        raise ValueError(f"no gap-free gyro segment covers {t0}-{t1} s (logger dropout inside the window)")
    idx = np.arange(*seg)
    omega = np.interp(tq, gs.t[idx], gs.gyro[idx, axis])
    return dict(r=r, u=u, omega=omega, Tc=Tc, t=tq)


def frf(data, nperseg=1024):
    fs = 1.0 / data["Tc"]
    f, S_rw = csd(data["r"], data["omega"], fs=fs, nperseg=nperseg)
    _, S_ru = csd(data["r"], data["u"], fs=fs, nperseg=nperseg)
    _, coh_w = coherence(data["r"], data["omega"], fs=fs, nperseg=nperseg)
    _, coh_u = coherence(data["r"], data["u"], fs=fs, nperseg=nperseg)
    return f, S_rw / S_ru, np.minimum(coh_w, coh_u)


def plant_model(f, b, tau_m, Td, Tc):
    s = 2j * np.pi * f
    return b / s / (tau_m * s + 1) * np.exp(-s * Td) * (1 - np.exp(-s * Tc)) / (s * Tc)


def identify(data, f_band=(0.5, 20.0), min_coh=0.5, nperseg=1024):
    """Fit (b, tau_m, Td) to the instrument-variable FRF. Returns the fit and its quality."""
    f, G, coh = frf(data, nperseg)
    sel = (f >= f_band[0]) & (f <= f_band[1]) & (coh >= min_coh)
    if sel.sum() < 6:
        raise ValueError("too few frequencies with enough coherence (not enough excitation)")
    fsel, Gsel, w = f[sel], G[sel], np.sqrt(coh[sel])
    Tc = data["Tc"]

    def resid(p):
        e = np.log(plant_model(fsel, np.exp(p[0]), np.exp(p[1]), p[2], Tc)) - np.log(Gsel)
        return np.concatenate([e.real, np.angle(np.exp(1j * e.imag))]) * np.concatenate([w, w])

    sol = least_squares(resid, [np.log(300.0), np.log(0.04), 0.002],
                        bounds=([0.0, np.log(0.003), 0.0], [12.0, np.log(0.5), 0.03]))
    b, tau_m, Td = np.exp(sol.x[0]), np.exp(sol.x[1]), sol.x[2]
    e = np.log(plant_model(fsel, b, tau_m, Td, Tc)) - np.log(Gsel)
    return dict(b=b, tau_m=tau_m, Td=Td, n_freq=int(sel.sum()), f_used=(fsel.min(), fsel.max()),
                mag_err_db=float(np.sqrt(np.mean((20 / np.log(10) * e.real)**2))),
                phase_err_deg=float(np.sqrt(np.mean(np.degrees(np.angle(np.exp(1j * e.imag)))**2))),
                f=f, G=G, coh=coh, sel=sel)
