"""Parametric noise model (THESIS 7.5): model, floors, fit recovery, fixed motor frequencies."""
import numpy as np
import pytest

import noise_model as nm

BAND = (20.0, 780.0)
F = np.arange(0, 1000, 1.5)


def synth(floor_coef=(-6.0, -1.5), peaks=((120.0, 3e-5, 4.0), (310.0, 1e-5, 6.0))):
    S = 10 ** (floor_coef[0] + floor_coef[1] * (np.log10(np.maximum(F, 1e-3)) - 2.0))
    for f0, A, g in peaks:
        S = S + nm.lorentz(F, A, f0, g)
    return S


def test_model_flat_floor_without_peaks_is_constant():
    S = nm.model_psd(F[10:], np.array([-5.0]), "flat", BAND)
    assert np.allclose(S, 1e-5)


def test_spline_floor_reproduces_a_power_law():
    f = F[(F >= 20) & (F <= 780)]
    target = -6 - 1.5 * (np.log10(f) - 2)
    B = nm._spline_basis(np.log10(f), np.log10(20), np.log10(780))
    coef = np.linalg.lstsq(B, target, rcond=None)[0]
    assert np.allclose(nm.floor_log(f, coef, "spline", BAND), target, atol=1e-9)


@pytest.mark.parametrize("floor", ["powerlaw", "spline"])
def test_fit_recovers_noiseless_peaks_and_floor(floor):
    fit = nm.fit_spectrum(F, synth(), BAND, floor=floor, prominence=0.3)
    assert fit["n_free"] == 2 and fit["rms_db"] < 0.1
    assert sorted(fit["peaks_hz"]) == pytest.approx([120.0, 310.0], abs=1.0)
    f = np.array([40.0, 500.0])
    got = nm.floor_log(f, fit["params"][:nm.n_floor(floor)], floor, BAND)
    assert 10 * np.abs(got - (-6 - 1.5 * (np.log10(f) - 2))) == pytest.approx([0, 0], abs=0.5)


def test_fit_with_welch_noise_is_close():
    rng = np.random.default_rng(4)
    noisy = synth() * rng.chisquare(50, len(F)) / 50          # 25 Welch averages: 50 degrees of freedom
    fit = nm.fit_spectrum(F, noisy, BAND, floor="powerlaw", prominence=0.5)
    assert sorted(fit["peaks_hz"])[:2] == pytest.approx([120.0, 310.0], abs=1.0)
    assert fit["rms_db"] < 1.2                                # estimation noise of 25 averages is ~0.9 dB


def test_motor_frequencies_stay_fixed():
    motors = [100.0, 104.0, 97.0, 110.0]
    S = synth(peaks=tuple((h * fm, 2e-5 / h, 3.0) for h in (1, 2) for fm in motors))
    fit = nm.fit_spectrum(F, S, BAND, floor="powerlaw", motor_hz=motors, n_harm=2, prominence=0.3)
    assert fit["motor_hz"] == motors and fit["n_harm"] == 2
    assert fit["n_free"] == 0 and fit["rms_db"] < 0.5


def test_harmonics_above_band_are_dropped():
    fit = nm.fit_spectrum(F, synth(), BAND, floor="flat", motor_hz=[300.0], n_harm=3, prominence=5.0)
    assert fit["n_harm"] == 2                                 # 900 Hz is outside 20-780 Hz
