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


# ---------------------------------------------------------- logs with dropouts (Phase 4 evidence pack)

ACRO = {10}                      # vehicle_status NAVIGATION_STATE_ACRO (PX4 v1.16 VehicleStatus.msg)
ATTITUDE_FROM_STICKS = {0, 1, 15}  # MANUAL (multicopter Stabilized), ALTCTL, STAB


def instrument_for_mode(nav_state):
    """D-14: rate setpoint in Acro, attitude setpoint where roll/pitch attitude comes from the sticks
    (Stabilized, Altitude); every other mode (Position, auto, offboard) needs both instruments to agree."""
    nav_state = int(nav_state)
    return "rate_sp" if nav_state in ACRO else "att_sp" if nav_state in ATTITUDE_FROM_STICKS else "both"


def uniform_pieces(t, Tc, max_hold=3):
    """Split increasing sample times ``t`` into pieces on a uniform grid; returns [(t_grid, idx), ...].

    Up to ``max_hold`` missing samples in a row (skipped controller publications: the motor command holds its
    value) are filled by holding the previous sample; a longer gap (logger dropout) starts a new piece. Each sample
    gets an integer slot from its own rounded interval, so timestamp jitter and a slow clock offset cannot shift
    the grid against the samples; t_grid is the least-squares line through (slot, t). idx[k] is the last sample at
    or before slot k (zero-order hold)."""
    t = np.asarray(t, dtype=float)
    steps = np.rint(np.diff(t) / Tc).astype(int)
    breaks = np.where((steps > max_hold + 1) | (steps < 1))[0]
    out = []
    for i0, i1 in zip(np.r_[0, breaks + 1], np.r_[breaks + 1, len(t)]):
        if i1 - i0 < 2:
            continue
        slot = np.r_[0, np.cumsum(steps[i0:i1 - 1])]
        a, b = np.polyfit(slot, t[i0:i1], 1)
        k = np.arange(slot[-1] + 1)
        idx = i0 + np.searchsorted(slot, k, side="right") - 1
        out.append((b + a * k, idx))
    return out


def body_rate_on_grid(gs, axis, tg, f_cut=100.0, edge_s=0.05):
    """Raw FIFO body rate of ``axis`` at times ``tg``: zero-phase low-pass (4th-order Butterworth, ``f_cut``) inside
    each gap-free segment, then linear interpolation (THESIS 7.1.1 Tier 2 step 2). The low-pass removes vibration
    that would alias into the fit band at the controller rate; being zero-phase it adds no delay. NaN where tg is
    outside a segment or within ``edge_s`` of its ends (filter edge effects)."""
    from scipy.signal import butter, sosfiltfilt
    tg = np.asarray(tg, dtype=float)
    out = np.full(len(tg), np.nan)
    if len(tg) == 0:
        return out
    seg = np.asarray(gs.segments, dtype=int).reshape(-1, 2)
    t_first, t_last = gs.t[seg[:, 0]], gs.t[seg[:, 1] - 1]
    near = np.flatnonzero((t_last >= tg[0]) & (t_first <= tg[-1]))     # only segments that overlap tg
    for a, b in seg[near]:
        ts = gs.t[a:b]
        if len(ts) < 2 or ts[-1] - ts[0] < 2 * edge_s:
            continue
        m = (tg >= ts[0] + edge_s) & (tg <= ts[-1] - edge_s)
        if not m.any():
            continue
        fs = (len(ts) - 1) / (ts[-1] - ts[0])
        x = gs.gyro[a:b, axis].astype(float)
        if f_cut < 0.45 * fs and len(ts) > 30:
            x = sosfiltfilt(butter(4, f_cut, fs=fs, output="sos"), x)
        out[m] = np.interp(tg[m], ts, x)
    return out


def piece_spectra(pieces, fs, nperseg=1024):
    """Welch auto/cross spectra of (r, u, w) accumulated over gap-free pieces (dicts of equal-length arrays).

    Each piece uses scipy.signal.csd (Hann, 50 % overlap); pieces shorter than ``nperseg`` are skipped; the
    pieces are averaged with weights equal to their number of Welch segments."""
    acc, n_tot, f = {}, 0, None
    for p in pieces:
        n = len(p["r"])
        if n < nperseg:
            continue
        n_seg = 1 + (n - nperseg) // (nperseg // 2)
        for key, (x, y) in dict(Srw=("r", "w"), Sru=("r", "u"), Srr=("r", "r"), Sww=("w", "w"), Suu=("u", "u")).items():
            f, S = csd(p[x], p[y], fs=fs, nperseg=nperseg)
            acc[key] = acc.get(key, 0.0) + n_seg * S
        n_tot += n_seg
    if n_tot == 0:
        raise ValueError("no gap-free piece as long as one Welch segment")
    out = {k: v / n_tot for k, v in acc.items()}
    out.update(f=f, n_seg=n_tot)
    return out


def frf_from_spectra(S):
    """G = S_rw / S_ru and the smaller of the two coherences (as ``frf``)."""
    coh_w = np.abs(S["Srw"]) ** 2 / (S["Srr"].real * S["Sww"].real)
    coh_u = np.abs(S["Sru"]) ** 2 / (S["Srr"].real * S["Suu"].real)
    return S["f"], S["Srw"] / S["Sru"], np.minimum(coh_w, coh_u)


def instruments_agree(f, G1, coh1, G2, coh2, f_band=(0.5, 20.0), min_coh=0.5, max_db=1.0, max_deg=5.0, min_n=6):
    """D-14 check for Position/auto modes: the FRFs from the two instruments agree within ``max_db`` / ``max_deg``
    (RMS over the band where both coherences are high; candidate 1 dB / 5 deg). Returns (ok, rms_db, rms_deg, n)."""
    f = np.asarray(f)
    m = (f >= f_band[0]) & (f <= f_band[1]) & (coh1 >= min_coh) & (coh2 >= min_coh)
    n = int(m.sum())
    if n < min_n:
        return False, float("nan"), float("nan"), n
    ratio = G2[m] / G1[m]
    db = float(np.sqrt(np.mean((20 * np.log10(np.abs(ratio))) ** 2)))
    deg = float(np.sqrt(np.mean(np.degrees(np.angle(ratio)) ** 2)))
    return bool(db <= max_db and deg <= max_deg), db, deg, n


def plant_model(f, b, tau_m, Td, Tc):
    s = 2j * np.pi * f
    return b / s / (tau_m * s + 1) * np.exp(-s * Td) * (1 - np.exp(-s * Tc)) / (s * Tc)


def identify(data, f_band=(0.5, 20.0), min_coh=0.5, nperseg=1024):
    """Fit (b, tau_m, Td) to the instrument-variable FRF. Returns the fit and its quality."""
    f, G, coh = frf(data, nperseg)
    return fit_frf(f, G, coh, data["Tc"], f_band=f_band, min_coh=min_coh)


def fit_frf(f, G, coh, Tc, f_band=(0.5, 20.0), min_coh=0.5):
    """Fit (b, tau_m, Td) of plant_model to an FRF where the coherence is high (at least 6 frequencies)."""
    sel = (f >= f_band[0]) & (f <= f_band[1]) & (coh >= min_coh)
    if sel.sum() < 6:
        raise ValueError("too few frequencies with enough coherence (not enough excitation)")
    fsel, Gsel, w = f[sel], G[sel], np.sqrt(coh[sel])

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
