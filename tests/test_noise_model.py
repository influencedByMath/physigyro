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


@pytest.mark.parametrize("floor", ["flat", "powerlaw", "spline"])
def test_analytic_jacobian_matches_finite_differences(floor):
    f = F[(F >= 20) & (F <= 780)]
    Bf = nm.floor_basis(f, floor, BAND)
    rng = np.random.default_rng(0)
    motors = [101.0, 97.0, 108.0, 104.0]
    p = np.r_[rng.normal(-6, 0.3, nm.n_floor(floor)),  # power law: slope about -6 (S spans 6 decades)
            [-5.0, 0.5, -5.5, 0.7], [-5.2, 233.0, 0.6, -5.8, 410.0, 0.9]]
    S, J = nm.model_and_jac(f, p, Bf, motors, 2, 2)
    assert np.allclose(S, nm.model_psd(f, p, floor, BAND, motors, 2, 2))
    eps = 1e-6
    JL = J / (S * np.log(10))[:, None]                        # d log10 S / dp: what the fit uses, scale-free
    for i in range(len(p)):
        dp = np.zeros_like(p)
        dp[i] = eps
        num = (np.log10(nm.model_and_jac(f, p + dp, Bf, motors, 2, 2)[0])
               - np.log10(nm.model_and_jac(f, p - dp, Bf, motors, 2, 2)[0])) / (2 * eps)
        assert np.allclose(JL[:, i], num, rtol=1e-5, atol=1e-7), i


def test_fit_converges_on_noisy_spectrum():
    rng = np.random.default_rng(4)
    noisy = synth() * rng.chisquare(50, len(F)) / 50
    fit = nm.fit_spectrum(F, noisy, BAND, floor="spline", prominence=0.5)
    assert fit["success"] and fit["nfev"] < 500


def test_lowpass_factor_multiplies_the_model():
    f = F[(F >= 20) & (F <= 780)]
    p = np.array([-6.0, -1.0])
    plain = nm.model_psd(f, p, "powerlaw", BAND)
    lp = nm.model_psd(f, p, "powerlaw", BAND, lowpass=(300.0, 3.0))
    assert np.allclose(lp, plain / (1 + (f / 300.0) ** 6))
    assert np.allclose(nm.lowpass_gain2(f, None), 1.0)


def test_fit_with_known_lowpass_recovers_the_unfiltered_model():
    S = synth() * nm.lowpass_gain2(F, (350.0, 4.0))
    fit = nm.fit_spectrum(F, S, BAND, floor="spline", prominence=0.3, lowpass=(350.0, 4.0))
    assert fit["rms_db"] < 0.1 and sorted(fit["peaks_hz"]) == pytest.approx([120.0, 310.0], abs=1.0)


def test_chip_lowpass_selected_by_bic_only_when_present():
    rng = np.random.default_rng(7)
    base = [synth(floor_coef=(-6.0 + 0.2 * i, -1.0 - 0.2 * i)) * rng.chisquare(50, len(F)) / 50 for i in range(3)]
    grid = [(250.0, 3.0), (400.0, 4.0)]
    flat = nm.fit_chip_lowpass([(F, P, None) for P in base], grid, prominence=0.5)
    assert flat["lowpass"] is None
    rolled = nm.fit_chip_lowpass([(F, P * nm.lowpass_gain2(F, (300.0, 3.0)), None) for P in base], grid, prominence=0.5)
    assert rolled["lowpass"] is not None and rolled["bic"][str(rolled["lowpass"])] < rolled["bic"]["None"]


def test_fixed_floor_keeps_the_robust_floor_and_fits_peaks_on_top():
    S = synth()
    f = F[(F >= 20) & (F <= 780)]
    c0, _ = nm.initial_floor(f, np.log10(S[(F >= 20) & (F <= 780)]), "spline", BAND)
    fit = nm.fit_spectrum(F, S, BAND, floor="spline", prominence=0.3, fix_floor=True)
    assert np.allclose(fit["params"][:nm.n_floor("spline")], c0, atol=1e-6)
    assert sorted(fit["peaks_hz"]) == pytest.approx([120.0, 310.0], abs=1.0) and fit["rms_db"] < 1.0
