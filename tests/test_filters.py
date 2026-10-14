"""E1: filter code against scipy.signal and the rules of thumb of THESIS.md Section 4.4."""
import numpy as np
import pytest
from scipy import signal

from filters import (derivative_response, esc_notch_frequencies, lowpass2_coeffs,
                     lowpass2_response, notch_coeffs, notch_response, phase_delay,
                     pt1_alpha, pt1_response)

F = np.geomspace(0.5, 3000, 400)


@pytest.mark.parametrize("fc,fs", [(40, 6400), (80, 6400), (30, 2000), (250, 8000)])
def test_lowpass2_matches_scipy_butter(fc, fs):
    b_ref, a_ref = signal.butter(2, fc, fs=fs)
    b, a = lowpass2_coeffs(fc, fs)
    np.testing.assert_allclose(b, b_ref, rtol=1e-10)
    np.testing.assert_allclose(a, a_ref[1:], rtol=1e-10, atol=1e-14)
    _, h_ref = signal.freqz(b_ref, a_ref, worN=F[F < fs / 2], fs=fs)
    np.testing.assert_allclose(lowpass2_response(F[F < fs / 2], fc, fs), h_ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("f0,bw,fs", [(100, 15, 6400), (220, 20, 2000), (400, 10, 8000)])
def test_notch_matches_scipy_iirnotch(f0, bw, fs):
    b_ref, a_ref = signal.iirnotch(f0, f0 / bw, fs=fs)
    b, a = notch_coeffs(f0, bw, fs)
    np.testing.assert_allclose(b, b_ref, rtol=1e-10)
    np.testing.assert_allclose(a, a_ref[1:], rtol=1e-10)
    assert abs(notch_response(f0, f0, bw, fs)) < 1e-9
    # -3 dB points are bw apart
    f = np.linspace(f0 - 2 * bw, f0 + 2 * bw, 20001)
    g = np.abs(notch_response(f, f0, bw, fs))
    above = f[g < 10**(-3 / 20)]
    assert abs((above[-1] - above[0]) - bw) < 0.02 * bw


@pytest.mark.parametrize("f0,bw,fs,f0_eff,bw_eff", [
    (100, 5, 8000, 100, 8.0),     # PX4 runs a 5 Hz notch as 8 Hz at 8 kHz (bw >= 0.001 fs)
    (100, 5, 6400, 100, 6.4),
    (100, 10, 8000, 100, 10),     # above the floor: unchanged
    (100, 5, 2000, 100, 5),       # floor 2 Hz: unchanged
    (3, 20, 8000, 8.0, 20),       # centre floor (NotchFilter::setParameters, v1.16.0)
])
def test_notch_lower_clamp_as_in_px4(f0, bw, fs, f0_eff, bw_eff):
    b_ref, a_ref = signal.iirnotch(f0_eff, f0_eff / bw_eff, fs=fs)
    b, a = notch_coeffs(f0, bw, fs)
    np.testing.assert_allclose(b, b_ref, rtol=1e-10)
    np.testing.assert_allclose(a, a_ref[1:], rtol=1e-10)


def test_pt1_matches_lfilter():
    fs, fc = 6400, 30
    alpha = pt1_alpha(fc, fs)
    _, h_ref = signal.freqz([alpha], [1, -(1 - alpha)], worN=F, fs=fs)
    np.testing.assert_allclose(pt1_response(F, fc, fs), h_ref, rtol=1e-10)
    # time domain: PX4 AlphaFilter update y += alpha (x - y)
    x = np.random.default_rng(0).standard_normal(500)
    y, state = np.empty_like(x), 0.0
    for i, xi in enumerate(x):
        state += alpha * (xi - state)
        y[i] = state
    np.testing.assert_allclose(y, signal.lfilter([alpha], [1, -(1 - alpha)], x), rtol=1e-12)


def test_disabled_filters_pass_through():
    assert np.allclose(lowpass2_response(F, 0, 6400), 1)
    assert np.allclose(lowpass2_response(F[F < 1000], 1000, 2000), 1)   # cutoff >= fs/2
    assert np.allclose(notch_response(F, 0, 20, 6400), 1)
    assert np.allclose(pt1_response(F, 0, 6400), 1)


@pytest.mark.parametrize("fc", [20, 40, 80])
def test_rule_of_thumb_delay_lowpass2(fc):
    tau = phase_delay(lowpass2_response(F, fc, 6400), F)
    rule = np.sqrt(2) / (2 * np.pi * fc)
    assert abs(np.interp(1.0, F, tau) / rule - 1) < 0.10


@pytest.mark.parametrize("fc", [20, 30, 40])
def test_rule_of_thumb_delay_pt1(fc):
    tau = phase_delay(pt1_response(F, fc, 6400), F)
    assert abs(np.interp(1.0, F, tau) * 2 * np.pi * fc - 1) < 0.10


def test_rule_of_thumb_delay_notch():
    tau = phase_delay(notch_response(F, 100, 15, 6400), F)
    assert abs(np.interp(1.0, F, tau) / (15 / (2 * np.pi * 100**2)) - 1) < 0.10


def test_derivative_is_jw_at_low_frequency():
    f = np.array([1.0, 5.0, 20.0])
    np.testing.assert_allclose(derivative_response(f, 6400), 2j * np.pi * f, rtol=0.02)


def test_esc_notch_frequency_rule():
    # PX4: max(f_motor (h+1), f_min + h bw / 2)
    c = esc_notch_frequencies([100.0, 30.0], n_harmonics=2, bw=10, f_min=50)
    np.testing.assert_allclose(c, [100, 200, 50, 60])
    # the floor is max(IMU_GYRO_DNF_MIN, bw): bw 30 > f_min 25 parks the notch at 30 Hz
    c = esc_notch_frequencies([20.0], n_harmonics=2, bw=30, f_min=25)
    np.testing.assert_allclose(c, [30, 45])
