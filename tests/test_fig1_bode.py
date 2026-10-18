"""Fig. 1 (experiment E1): Bode plot of the PX4 v1.16 filter stack (Phase 4 Task 2; Review Focus 3).

The figure must show exactly the filter chain of the loop model: the same functions, the same rate."""
import os
import sys
from dataclasses import replace

import numpy as np
import pytest
from scipy import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import fig1_bode as fb          # noqa: E402
import filters as flt           # noqa: E402
import loop_model as lm         # noqa: E402

F = np.geomspace(1.0, 1000.0, 400)


def test_settings_come_from_the_loop_model_default_table():
    s = fb.SETTINGS
    d = lm.PX4_V116_MC_DEFAULT
    assert (s.f_gyro, s.f_dterm, s.notches) == (d.f_gyro, d.f_dterm, d.notches) == (40.0, 30.0, ())
    assert (s.dnf_bw, s.dnf_hmc, s.dnf_min) == (d.dnf_bw, d.dnf_hmc, d.dnf_min) == (15.0, 3, 25.0)
    assert s.dnf_esc and not d.dnf_esc                    # the ESC bank is shown "when enabled"
    assert fb.FS == lm.Rates().fs                          # same filter rate as the loop model


def test_lowpass_and_pt1_equal_scipy_freqz_of_the_same_coefficients():
    r = fb.responses(F, fb.FS, fb.SETTINGS, fb.MOTOR_HZ)
    b, a = flt.lowpass2_coeffs(40.0, fb.FS)
    _, h = signal.freqz([float(x) for x in b], [1.0, float(a[0]), float(a[1])], worN=F, fs=fb.FS)
    assert np.max(np.abs(r["gyro_lpf"] - h)) < 1e-9
    al = float(flt.pt1_alpha(30.0, fb.FS))
    _, h = signal.freqz([al], [1.0, -(1.0 - al)], worN=F, fs=fb.FS)
    assert np.max(np.abs(r["dterm_pt1"] - h)) < 1e-9


def test_notch_bank_equals_scipy_freqz_product():
    r = fb.responses(F, fb.FS, fb.SETTINGS, fb.MOTOR_HZ)
    s = fb.SETTINGS
    h = np.ones(len(F), complex)
    for f0 in flt.esc_notch_frequencies(fb.MOTOR_HZ, s.dnf_hmc, s.dnf_bw, s.dnf_min):
        b, a = flt.notch_coeffs(f0, s.dnf_bw, fb.FS)
        _, hk = signal.freqz([float(x) for x in b], [1.0, float(a[0]), float(a[1])], worN=F, fs=fb.FS)
        h = h * hk
    assert np.max(np.abs(r["esc_bank"] - h)) < 1e-9


def test_totals_are_the_loop_model_paths():
    r = fb.responses(F, fb.FS, fb.SETTINGS, fb.MOTOR_HZ)
    Fg = lm.gyro_stack(F, fb.SETTINGS, fb.FS, fb.MOTOR_HZ)
    assert np.max(np.abs(r["gyro_path"] - Fg)) < 1e-12
    # D-term path as in LoopCase.evaluate (tau_D): F_g F_d D(z) / (j 2 pi f)
    Dp = Fg * flt.pt1_response(F, 30.0, fb.FS) * flt.derivative_response(F, fb.FS) / (2j * np.pi * F)
    assert np.max(np.abs(r["dterm_path"] - Dp)) < 1e-12


def test_factors_multiply_to_the_responses():
    r = fb.responses(F, fb.FS, fb.SETTINGS, fb.MOTOR_HZ)
    for key, facs in fb.factors(F, fb.FS, fb.SETTINGS, fb.MOTOR_HZ).items():
        assert np.max(np.abs(np.prod(facs, axis=0) - r[key])) < 1e-12


def test_delay_at_crossover_matches_loop_model_tau():
    f = np.geomspace(0.5, 3000.0, 20000)
    r = fb.responses(f, fb.FS, fb.SETTINGS, fb.MOTOR_HZ)
    d = fb.delays_ms(f, fb.FS, fb.SETTINGS, fb.MOTOR_HZ)
    for fc in (4.0, 10.0, 15.0):
        _, tau_F = lm.phase_lag_at(f, r["gyro_path"], fc)
        _, tau_D = lm.phase_lag_at(f, r["dterm_path"], fc)
        k = np.argmin(np.abs(f - fc))
        assert d["gyro_path"][k] == pytest.approx(1e3 * tau_F, rel=2e-3)
        assert d["dterm_path"][k] == pytest.approx(1e3 * tau_D, rel=2e-3)


def test_low_frequency_delay_of_gyro_lowpass_is_analytic():
    # THESIS 4.4: 2nd-order Butterworth delay sqrt(2) / (2 pi fc) well below the corner
    d = fb.delays_ms(np.array([0.5]), fb.FS, fb.SETTINGS, fb.MOTOR_HZ)
    assert d["gyro_lpf"][0] == pytest.approx(1e3 * np.sqrt(2) / (2 * np.pi * 40.0), rel=1e-3)


def test_notch_gives_phase_lead_just_above_its_centre():
    # a notch lags below f0 and LEADS above it; np.unwrap through the zero on the unit circle can turn
    # the lead into a 270 deg lag (seen in the first render), so the delay must not come from unwrapping
    s = replace(fb.SETTINGS, dnf_hmc=1)
    f0 = float(fb.MOTOR_HZ[0])
    f = np.array([0.95 * f0, 1.05 * f0])
    d = fb.delays_ms(f, fb.FS, s, fb.MOTOR_HZ)["esc_bank"]
    assert d[0] > 0 and d[1] < 0
    # 4 motors at the same frequency, one harmonic: 4 identical notches
    assert np.allclose(d, 1e3 * (-4 * np.angle(flt.notch_response(f, f0, s.dnf_bw, fb.FS))) / (2 * np.pi * f))


def test_markers_use_decision_d28_and_the_frozen_band():
    assert fb.FC_RANGE == (4.0, 15.0)              # D-28
    assert fb.NOISE_BAND_MAX == 780.0              # D-23 f_max
    assert fb.FC_LABEL.lower().find("candidate") < 0   # decided, not a candidate any more


# ------------------------------------------------------------- ESC bank off vs on (author's request)

def test_variants_are_the_px4_default_and_the_default_with_the_esc_bank():
    assert fb.VARIANTS["off"] == lm.PX4_V116_MC_DEFAULT
    assert fb.VARIANTS["on"] == replace(lm.PX4_V116_MC_DEFAULT, dnf_esc=True) == fb.SETTINGS


def test_off_variant_has_no_notch_and_lowpass_only_gyro_path():
    r = fb.responses(F, fb.FS, fb.VARIANTS["off"], fb.MOTOR_HZ)
    assert np.allclose(r["esc_bank"], 1.0)
    assert np.max(np.abs(r["gyro_path"] - r["gyro_lpf"])) < 1e-15
    d = fb.delays_ms(np.array([10.0]), fb.FS, fb.VARIANTS["off"], fb.MOTOR_HZ)
    assert d["gyro_path"][0] == pytest.approx(d["gyro_lpf"][0])


def test_comparison_figure_is_written(tmp_path):
    out = fb.make_comparison(str(tmp_path))
    for ext in ("png", "pdf"):
        assert (tmp_path / f"fig1_compare_esc_off_on.{ext}").stat().st_size > 10_000
    # the delay table it returns: the ESC bank adds its delay to both paths at the crossover
    assert out["on"]["gyro_path"][1] - out["off"]["gyro_path"][1] == pytest.approx(out["on"]["esc_bank"][1], rel=1e-9)
