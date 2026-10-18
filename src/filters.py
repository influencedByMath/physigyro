"""Frequency responses of the PX4 gyro filters (THESIS.md Section 4.4).

Each filter is a discrete-time IIR filter running at sample rate ``fs``. Its
response is evaluated at z = exp(j 2 pi f / fs). The coefficients copy the PX4
source (v1.16.0, checked 2026-10-10; same in v1.14 to main):

- ``LowPassFilter2p``  gyro lowpass, IMU_GYRO_CUTOFF: 2nd-order Butterworth,
  bilinear transform with pre-warping.
- ``AlphaFilter``      D-term lowpass, IMU_DGYRO_CUTOFF: **1st-order** (PT1).
- ``NotchFilter``      static notches, ESC-RPM notches and FFT notches; the
  width is given as a bandwidth in Hz.

All functions broadcast: pass ``f`` with shape (F,) and parameters with shape
(..., 1) to get a (..., F) array of responses for many settings at once.
"""
import numpy as np


def _zinv(f, fs):
    """z^-1 on the unit circle."""
    return np.exp(-2j * np.pi * np.asarray(f, dtype=float) / fs)


def biquad_response(b, a, f, fs):
    """H(z) = (b0 + b1 z^-1 + b2 z^-2) / (1 + a1 z^-1 + a2 z^-2); ``a`` = (a1, a2)."""
    zi = _zinv(f, fs)
    return (b[0] + b[1] * zi + b[2] * zi**2) / (1.0 + a[0] * zi + a[1] * zi**2)


# ---------------------------------------------------------------- coefficients

def lowpass2_coeffs(fc, fs):
    """PX4 LowPassFilter2p. Disabled (pass-through) when fc <= 0 or fc >= fs/2."""
    fc = np.asarray(fc, dtype=float)
    on = (fc > 0) & (fc < fs / 2)
    ohm = np.tan(np.pi * np.where(on, fc, 1.0) / fs)
    c = 1.0 + np.sqrt(2.0) * ohm + ohm**2
    b0 = np.where(on, ohm**2 / c, 1.0)
    a1 = np.where(on, 2.0 * (ohm**2 - 1.0) / c, 0.0)
    a2 = np.where(on, (1.0 - np.sqrt(2.0) * ohm + ohm**2) / c, 0.0)
    b1 = np.where(on, 2.0 * b0, 0.0)
    b2 = np.where(on, b0, 0.0)
    return (b0, b1, b2), (a1, a2)


def pt1_alpha(fc, fs):
    """PX4 AlphaFilter::setCutoffFreq: alpha = Ts / (tau + Ts), tau = 1 / (2 pi fc).

    fc <= 0 gives alpha = 1 (pass-through)."""
    fc = np.asarray(fc, dtype=float)
    on = fc > 0
    tau = 1.0 / (2.0 * np.pi * np.where(on, fc, 1.0))
    return np.where(on, (1.0 / fs) / (tau + 1.0 / fs), 1.0)


def notch_coeffs(f0, bw, fs):
    """PX4 NotchFilter::setParameters. Disabled when f0 <= 0, bw <= 0 or f0 >= fs/2.

    PX4 then raises f0 and bw to at least 0.001 fs (e.g. a 5 Hz bandwidth runs
    as 8 Hz at fs = 8 kHz). Same as scipy.signal.iirnotch(f0, Q=f0/bw, fs)."""
    f0 = np.asarray(f0, dtype=float)
    bw = np.asarray(bw, dtype=float)
    on = (f0 > 0) & (bw > 0) & (f0 < fs / 2)
    f0 = np.maximum(f0, 0.001 * fs)
    bw = np.maximum(bw, 0.001 * fs)
    alpha = np.tan(np.pi * np.where(on, bw, 1.0) / fs)
    beta = -np.cos(2.0 * np.pi * f0 / fs)
    b0 = np.where(on, 1.0 / (1.0 + alpha), 1.0)
    b1 = np.where(on, 2.0 * beta * b0, 0.0)
    a2 = np.where(on, (1.0 - alpha) * b0, 0.0)
    return (b0, b1, np.where(on, b0, 0.0)), (b1, a2)


# ------------------------------------------------------------------ responses

def lowpass2_response(f, fc, fs):
    b, a = lowpass2_coeffs(fc, fs)
    return biquad_response(b, a, f, fs)


def pt1_response(f, fc, fs):
    """H(z) = alpha / (1 - (1 - alpha) z^-1)."""
    alpha = pt1_alpha(fc, fs)
    return alpha / (1.0 - (1.0 - alpha) * _zinv(f, fs))


def notch_response(f, f0, bw, fs):
    b, a = notch_coeffs(f0, bw, fs)
    return biquad_response(b, a, f, fs)


def esc_notch_frequencies(motor_hz, n_harmonics, bw, f_min):
    """Centre frequencies of the PX4 ESC-RPM notch bank (IMU_GYRO_DNF_EN bit 0).

    Copies VehicleAngularVelocity::UpdateDynamicNotchEscRpm (v1.16.0): for motor
    i and harmonic h = 0..H-1 the notch sits at max(f_i (h+1), f_floor + h bw / 2)
    with f_floor = max(IMU_GYRO_DNF_MIN, bw) ("parked" notch at low RPM).
    Returns a flat array with one entry per (motor, harmonic)."""
    motor_hz = np.atleast_1d(np.asarray(motor_hz, dtype=float))
    h = np.arange(int(n_harmonics))
    f_floor = max(float(f_min), float(bw))
    return np.maximum(motor_hz[:, None] * (h + 1), f_floor + h * 0.5 * bw).ravel()


def notch_bank_response(f, centres, bw, fs):
    """Product of notches at ``centres`` (1-D array), all with bandwidth ``bw``."""
    out = np.ones(np.broadcast_shapes(np.shape(f), np.shape(bw)), dtype=complex)
    for f0 in np.atleast_1d(centres):
        out = out * notch_response(f, f0, bw, fs)
    return out


def derivative_response(f, fs):
    """Backward difference (x_n - x_(n-1)) * fs, as in FilterAngularAcceleration."""
    return (1.0 - _zinv(f, fs)) * fs


def phase_delay(H, f):
    """Equivalent delay tau(f) = phase lag / (2 pi f) in seconds (phase unwrapped along f).

    Valid below the first notch centre. A notch has a zero on the unit circle, so its phase jumps by 180 deg at
    f0 and np.unwrap may resolve the jump either way; above a notch, sum the phases of the single factors instead
    (experiments/phase4/fig1_bode.py, ``factors``)."""
    lag = -np.unwrap(np.angle(H), axis=-1)
    return lag / (2.0 * np.pi * np.asarray(f))
