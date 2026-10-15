"""Parametric noise model fitted to each real spectrum (THESIS 7.5).

    S(f) = floor(f) + sum_i sum_h A_h / (1 + ((f - h f_m,i) / g_h)^2)  +  sum_r A_r / (1 + ((f - f_r) / g_r)^2)

Broadband floor, motor harmonics (when ESC RPM is valid, the motor frequencies f_m,i are FIXED to RPM/60 and
each harmonic h has one amplitude A_h and width g_h shared by the motors, Section 7.4.5 use 1) and free peaks
(frame resonances, or the motor peaks when there is no RPM). Fitted with scipy.optimize.least_squares on
log10 PSD over the fixed band only.

Floor shapes (THESIS assumes "flat"; the pilot found floors that change by 2-4 decades, so the choice is an
author decision, D-24): "flat" log10 S = c; "powerlaw" log10 S = a + b log10(f / 100 Hz); "spline" a cubic
B-spline in log10 f with 4 interior knots equally spaced in log f over the band.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from scipy.signal import find_peaks

FLOORS = ("flat", "powerlaw", "spline")
N_KNOTS = 4


def _spline_basis(lf, lo, hi):
    inner = np.linspace(lo, hi, N_KNOTS + 2)[1:-1]
    t = np.r_[[lo] * 4, inner, [hi] * 4]
    n = len(t) - 4
    return np.stack([BSpline(t, np.eye(n)[j], 3, extrapolate=True)(lf) for j in range(n)], 1)


def n_floor(floor):
    return {"flat": 1, "powerlaw": 2, "spline": N_KNOTS + 4}[floor]


def floor_log(f, coef, floor, band, basis=None):
    """log10 floor; ``basis`` = precomputed _spline_basis(log10 f, ...) for speed (same result)."""
    lf = np.log10(f)
    if floor == "flat":
        return np.full_like(lf, coef[0])
    if floor == "powerlaw":
        return coef[0] + coef[1] * (lf - 2.0)
    if floor == "spline":
        B = basis if basis is not None else _spline_basis(lf, np.log10(band[0]), np.log10(band[1]))
        return B @ coef
    raise ValueError(floor)


def lorentz(f, A, f0, g):
    return A / (1 + ((f - f0) / g) ** 2)


def model_psd(f, p, floor, band, motor_hz=None, n_harm=0, n_free=0, basis=None):
    """p = [floor coefs | per harmonic (log10 A_h, log10 g_h) | per free peak (log10 A, f0, log10 g)]."""
    k = n_floor(floor)
    S = 10 ** floor_log(f, p[:k], floor, band, basis)
    for h in range(n_harm):
        A, g = 10 ** p[k + 2 * h], 10 ** p[k + 2 * h + 1]
        for fm in motor_hz:
            S = S + lorentz(f, A, (h + 1) * fm, g)
    j = k + 2 * n_harm
    for r in range(n_free):
        S = S + lorentz(f, 10 ** p[j + 3 * r], p[j + 3 * r + 1], 10 ** p[j + 3 * r + 2])
    return S


def initial_floor(f, logP, floor, band, pct=20, width_dec=1 / 6):
    """Robust floor start: 20th percentile of log10 PSD in sliding 1/6-decade windows (ignores narrow peaks)."""
    lf = np.log10(f)
    base = np.array([np.percentile(logP[np.abs(lf - c) <= width_dec / 2], pct) for c in lf])
    if floor == "flat":
        return np.array([np.median(base)]), base
    if floor == "powerlaw":
        b, a = np.polyfit(lf - 2.0, base, 1)
        return np.array([a, b]), base
    B = _spline_basis(lf, np.log10(band[0]), np.log10(band[1]))
    return np.linalg.lstsq(B, base, rcond=None)[0], base


def fit_spectrum(f, P, band=(20.0, 780.0), floor="powerlaw", motor_hz=None, n_harm=3, prominence=0.5,
                 max_free=6, min_sep_hz=8.0):
    """Fit the model to one PSD (1-D, native bins). ``motor_hz``: valid mean motor frequencies [Hz] or None.
    Free peaks: the ``max_free`` most prominent peaks of log10 PSD above the robust floor (``prominence`` in
    decades), skipping peaks within ``min_sep_hz`` of a fixed motor harmonic.
    Returns dict(params, floor, rms_db, n_free, n_harm, peaks_hz, motor_hz)."""
    m = (f >= band[0]) & (f <= band[1])
    f, P = np.asarray(f, float)[m], np.asarray(P, float)[m]
    logP = np.log10(P)
    c0, base = initial_floor(f, logP, floor, band)
    use_m = motor_hz is not None and len(motor_hz) > 0
    n_h = 0
    if use_m:
        n_h = sum(1 for h in range(1, n_harm + 1) if h * max(motor_hz) < band[1])
    df = f[1] - f[0]
    pk, props = find_peaks(logP - base, prominence=prominence, distance=max(1, int(min_sep_hz / df)))
    order = np.argsort(props["prominences"])[::-1]
    free = []
    for i in pk[order]:
        if use_m and any(abs(f[i] - h * fm) < min_sep_hz for h in range(1, n_h + 1) for fm in motor_hz):
            continue
        free.append(i)
        if len(free) == max_free:
            break
    # floor coefficients bounded (log10 PSD of a gyro in (rad/s)^2/Hz lies far inside -20..5)
    p0, lo, hi = list(c0), [-20.0] * len(c0), [5.0] * len(c0)
    if floor == "powerlaw":
        lo[1], hi[1] = -10.0, 10.0
    for h in range(n_h):
        hf = [(h + 1) * fm for fm in motor_hz]
        near = np.abs(f[:, None] - np.array(hf)[None]).min(1) < 3 * df
        excess = max(np.max(10 ** logP[near]) - np.max(10 ** base[near]), 1e-12) if near.any() else 1e-12
        p0 += [np.log10(excess / len(motor_hz)), np.log10(3.0)]
        lo += [-20, np.log10(0.3)]
        hi += [5, np.log10(80.0)]
    for i in free:
        p0 += [np.log10(max(10 ** logP[i] - 10 ** base[i], 1e-12)), f[i], np.log10(3.0)]
        lo += [-20, f[i] - 10.0, np.log10(0.3)]
        hi += [5, f[i] + 10.0, np.log10(80.0)]
    p0 = np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9)

    basis = _spline_basis(np.log10(f), np.log10(band[0]), np.log10(band[1])) if floor == "spline" else None

    def res(p):
        return np.log10(model_psd(f, p, floor, band, motor_hz, n_h, len(free), basis)) - logP

    sol = least_squares(res, p0, bounds=(lo, hi), x_scale="jac", max_nfev=4000)
    r = sol.fun
    k = n_floor(floor) + 2 * n_h
    return dict(params=sol.x, floor=floor, rms_db=float(10 * np.sqrt(np.mean(r ** 2))), n_free=len(free),
                n_harm=n_h, motor_hz=None if not use_m else [float(x) for x in motor_hz],
                peaks_hz=[float(sol.x[k + 3 * j + 1]) for j in range(len(free))], success=bool(sol.success),
                n_params=len(sol.x), n_bins=len(f))
