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


def floor_basis(f, floor, band):
    """Design matrix of the log10 floor: log10 floor = floor_basis @ coefs (all three shapes are linear)."""
    lf = np.log10(f)
    if floor == "flat":
        return np.ones((len(f), 1))
    if floor == "powerlaw":
        return np.stack([np.ones_like(lf), lf - 2.0], 1)
    if floor == "spline":
        return _spline_basis(lf, np.log10(band[0]), np.log10(band[1]))
    raise ValueError(floor)


LN10 = np.log(10.0)


def lowpass_gain2(f, lowpass):
    """IMU-chip low-pass |H(f)|^2 = 1 / (1 + (f / f_c)^(2 n)) (D-24 (1); one per chip model, fitted jointly).
    ``lowpass`` = (f_c, n) or None (no term)."""
    if lowpass is None:
        return np.ones_like(np.asarray(f, float))
    fc, n = lowpass
    return 1.0 / (1.0 + (np.asarray(f, float) / fc) ** (2 * n))


def model_and_jac(f, p, Bf, motor_hz=None, n_harm=0, n_free=0):
    """Model PSD S(f) and its exact Jacobian dS/dp (same parameter layout as model_psd)."""
    k = Bf.shape[1]
    F = 10 ** (Bf @ p[:k])
    S = F.copy()
    J = np.empty((len(f), len(p)))
    J[:, :k] = (F * LN10)[:, None] * Bf
    for h in range(n_harm):
        A, g = 10 ** p[k + 2 * h], 10 ** p[k + 2 * h + 1]
        dA = dg = 0.0
        for fm in motor_hz:
            u = (f - (h + 1) * fm) / g
            d = 1 + u * u
            S = S + A / d
            dA = dA + LN10 * A / d
            dg = dg + LN10 * 2 * A * u * u / d ** 2
        J[:, k + 2 * h], J[:, k + 2 * h + 1] = dA, dg
    j = k + 2 * n_harm
    for r in range(n_free):
        a, f0, q = p[j + 3 * r: j + 3 * r + 3]
        A, g = 10 ** a, 10 ** q
        u = (f - f0) / g
        d = 1 + u * u
        S = S + A / d
        J[:, j + 3 * r] = LN10 * A / d
        J[:, j + 3 * r + 1] = 2 * A * u / (g * d ** 2)
        J[:, j + 3 * r + 2] = LN10 * 2 * A * u * u / d ** 2
    return S, J


def model_psd(f, p, floor, band, motor_hz=None, n_harm=0, n_free=0, basis=None, lowpass=None):
    """p = [floor coefs | per harmonic (log10 A_h, log10 g_h) | per free peak (log10 A, f0, log10 g)].
    ``lowpass`` (f_c, n): the chip low-pass multiplies the whole vibration model (the chip filters the signal)."""
    Bf = basis if basis is not None and floor == "spline" else floor_basis(f, floor, band)
    return model_and_jac(f, p, Bf, motor_hz, n_harm, n_free)[0] * lowpass_gain2(f, lowpass)


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
                 max_free=6, min_sep_hz=8.0, lowpass=None):
    """Fit the model to one PSD (1-D, native bins). ``motor_hz``: valid mean motor frequencies [Hz] or None.
    Free peaks: the ``max_free`` most prominent peaks of log10 PSD above the robust floor (``prominence`` in
    decades), skipping peaks within ``min_sep_hz`` of a fixed motor harmonic.
    Returns dict(params, floor, rms_db, n_free, n_harm, peaks_hz, motor_hz)."""
    m = (f >= band[0]) & (f <= band[1])
    f, P = np.asarray(f, float)[m], np.asarray(P, float)[m]
    logH = np.log10(lowpass_gain2(f, lowpass))      # known chip low-pass: fit the vibration model to P / |H|^2
    logP = np.log10(P) - logH
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

    Bf = floor_basis(f, floor, band)
    cache = {}

    def _eval(p):
        key = p.tobytes()
        if key not in cache:
            cache.clear()
            cache[key] = model_and_jac(f, p, Bf, motor_hz, n_h, len(free))
        return cache[key]

    def res(p):
        return np.log10(_eval(p)[0]) - logP

    def jac(p):
        S, J = _eval(p)
        return J / (S * LN10)[:, None]

    sol = least_squares(res, p0, jac=jac, bounds=(lo, hi), x_scale="jac", max_nfev=4000)   # residual unchanged by H
    r = sol.fun
    k = n_floor(floor) + 2 * n_h
    return dict(params=sol.x, floor=floor, rms_db=float(10 * np.sqrt(np.mean(r ** 2))), n_free=len(free),
                n_harm=n_h, motor_hz=None if not use_m else [float(x) for x in motor_hz],
                peaks_hz=[float(sol.x[k + 3 * j + 1]) for j in range(len(free))], success=bool(sol.success), nfev=int(sol.nfev), lowpass=lowpass,
                n_params=len(sol.x), n_bins=len(f))


def fit_chip_lowpass(spectra, grid, floor="spline", prominence=0.5, band=(20.0, 780.0)):
    """One low-pass per IMU chip model (D-24 (1), P1), shared by all its vehicles: every (f_c, n) of ``grid`` and
    "no term" are tried; the choice minimises the BIC of all the chip's spectra together (+2 parameters for a
    term). ``spectra``: list of (f, P, motor_hz). Only the presence and magnitude are trusted (D-25): the corner
    and order trade off against the spline floors and are not identifiable."""
    bic, best = {}, None
    for lp in [None] + list(grid):
        sse, n, k, fits = 0.0, 0, 0, []
        for f, P, motors in spectra:
            r = fit_spectrum(f, P, band, floor=floor, motor_hz=motors, prominence=prominence, lowpass=lp)
            sse += (r["rms_db"] / 10) ** 2 * r["n_bins"]
            n += r["n_bins"]
            k += r["n_params"]
            fits.append(r)
        k += 0 if lp is None else 2
        b = n * np.log(sse / n) + k * np.log(n)
        bic[str(lp)] = float(b)
        if best is None or b < best[0]:
            best = (b, lp, fits)
    return dict(lowpass=best[1], bic=bic, fits=best[2])
