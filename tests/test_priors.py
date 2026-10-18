"""Tier 1 plant priors (Phase 4 Task 1; THESIS 7.1.1, D-25): fit-half guard, chip-filter delay, log evidence."""
import numpy as np
import pandas as pd
import pytest
from scipy import signal

import priors as pr


# ------------------------------------------------------------- leakage guard (Review Focus 1)

@pytest.mark.parametrize("path", ["data/spectra/test_windows.csv", r"data\spectra\test_psd_native.npz",
                                  "test_logs.json", "/abs/data/spectra/test_native/x.npz"])
def test_require_fit_half_refuses_test_paths(path):
    with pytest.raises(ValueError, match="test half"):
        pr.require_fit_half(path)


@pytest.mark.parametrize("path", ["data/spectra/fit_windows.csv", "data/spectra/fit_logs.json",
                                  "experiments/part2/model_fits.json"])
def test_require_fit_half_accepts_fit_paths(path):
    assert pr.require_fit_half(path) == path


def test_require_fit_rows_refuses_any_test_row():
    df = pd.DataFrame(dict(split=["fit", "fit", "test"]))
    with pytest.raises(ValueError, match="test half"):
        pr.require_fit_rows(df)
    pr.require_fit_rows(df.iloc[:2])


# ------------------------------------------------------------- chip low-pass group delay (D-25)

@pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
def test_group_delay_low_frequency_equals_butterworth_formula(n):
    fc = 323.0
    tau0 = 1.0 / (2 * np.pi * fc * np.sin(np.pi / (2 * n)))       # sum of -Re(p_k)/|p_k|^2 over the poles
    assert pr.lowpass_group_delay(1e-3 * fc, fc, n) == pytest.approx(tau0, rel=1e-6)


def test_group_delay_second_order_is_thesis_rule_of_thumb():
    # THESIS 4.4: 2nd-order Butterworth delay sqrt(2) / (2 pi fc) well below the corner
    assert pr.lowpass_group_delay(0.01, 40.0, 2) == pytest.approx(np.sqrt(2) / (2 * np.pi * 40.0), rel=1e-6)


@pytest.mark.parametrize("fc,n", [(150.0, 2), (323.0, 3), (539.0, 6), (90.0, 1)])
def test_group_delay_matches_scipy_analog_butterworth(fc, n):
    b, a = signal.butter(n, 2 * np.pi * fc, analog=True)
    f = np.linspace(5.0, 400.0, 4000)
    _, h = signal.freqs(b, a, worN=2 * np.pi * f)
    num = -np.gradient(np.unwrap(np.angle(h)), 2 * np.pi * f)
    sel = slice(10, -10)
    assert np.allclose(pr.lowpass_group_delay(f, fc, n)[sel], num[sel], rtol=1e-4)


def test_group_delay_of_magnitude_matches_model_magnitude():
    # the delay belongs to the same |H|^2 = 1/(1 + (f/fc)^2n) that noise_model.lowpass_gain2 uses
    import noise_model as nm
    fc, n = 250.0, 3
    b, a = signal.butter(n, 2 * np.pi * fc, analog=True)
    f = np.array([20.0, 100.0, 300.0, 700.0])
    _, h = signal.freqs(b, a, worN=2 * np.pi * f)
    assert np.allclose(np.abs(h) ** 2, nm.lowpass_gain2(f, (fc, n)), rtol=1e-10)


def test_band_delay_range_over_band():
    # the group delay is not monotonic (it peaks below the corner), so compare with a dense evaluation
    lo, hi = pr.band_delay_range(150.0, 2, band=(20.0, 100.0))
    tau = pr.lowpass_group_delay(np.linspace(20.0, 100.0, 20001), 150.0, 2)
    assert lo == pytest.approx(tau.min(), rel=1e-4) and hi == pytest.approx(tau.max(), rel=1e-4)
    assert 0 < lo < hi


# ------------------------------------------------------------- indistinguishable chip terms

def _rows():
    mk = lambda term, bic, rms: dict(chip="C", term=term, bic=bic, rms_db_pooled=rms)
    return [mk(None, 0.0, 4.0), mk([300.0, 2.0], -100.0, 3.00), mk([250.0, 2.0], -95.0, 3.02),
            mk([400.0, 6.0], -60.0, 3.04), mk([150.0, 1.0], -10.0, 3.50), mk([1500.0, 6.0], -1.0, 3.9)]


def test_indistinguishable_terms_by_bic():
    terms = pr.indistinguishable_terms(_rows(), "C", rule="bic", tol=10.0)
    assert sorted(map(tuple, terms)) == [(250.0, 2.0), (300.0, 2.0)]


def test_indistinguishable_terms_by_fit_error():
    terms = pr.indistinguishable_terms(_rows(), "C", rule="rms", tol=0.015)       # within 1.5 % of 3.00 dB
    assert sorted(map(tuple, terms)) == [(250.0, 2.0), (300.0, 2.0), (400.0, 6.0)]


def test_indistinguishable_terms_never_include_no_term():
    rows = _rows() + [dict(chip="C", term=None, bic=-100.0, rms_db_pooled=3.0)]
    rows = [r for r in rows if not (r["term"] is None and r["bic"] == 0.0)]
    assert None not in pr.indistinguishable_terms(rows, "C", rule="rms", tol=0.5)


# ------------------------------------------------------------- log evidence

def test_control_rate_median_ignores_dropouts():
    t = np.arange(0, 20.0, 1 / 666.6667)
    keep = np.ones(len(t), bool)
    keep[2000:3500] = False                                        # a logger dropout of 2.25 s
    ts_us = (t[keep] * 1e6).astype(np.int64)
    assert pr.publication_rate(ts_us) == pytest.approx(666.6667, rel=1e-3)
    mean_rate = (len(ts_us) - 1) / ((ts_us[-1] - ts_us[0]) * 1e-6)
    assert mean_rate < 0.95 * 666.6667                             # why the mean rate is not used


def test_hover_motor_frequency_uses_valid_rpm_only():
    df = pd.DataFrame(dict(
        vehicle_key=["a", "a", "b", "c"], split="fit", rpm_valid=[True, True, False, True],
        rpm_hz=["[100, 110, 100, 110]", "[90, 100, 90, 100]", "[70, 70, 70, 70]", "[50, 50, 50, 50]"]))
    hm = pr.hover_motor_hz(df)
    assert hm == {"a": pytest.approx(100.0), "c": pytest.approx(50.0)}


# ------------------------------------------------------------- motor output protocol (T_d evidence)

def test_protocol_pwm_main_group():
    p = {"PWM_MAIN_FUNC1": 101, "PWM_MAIN_FUNC2": 102, "PWM_MAIN_TIM0": 400, "PWM_MAIN_TIM1": 400, "PWM_AUX_TIM0": -3}
    assert pr.motor_output_protocol(p) == dict(groups=["PWM_MAIN"], protocol="pwm")


def test_protocol_dshot_aux_group_ignores_other_groups():
    p = {"PWM_AUX_FUNC1": 101, "PWM_AUX_FUNC2": 104, "PWM_AUX_TIM0": -3, "PWM_AUX_TIM1": -5, "PWM_MAIN_TIM0": 400}
    assert pr.motor_output_protocol(p) == dict(groups=["PWM_AUX"], protocol="dshot")


def test_protocol_mixed_timers_in_motor_group_is_mixed():
    p = {"PWM_MAIN_FUNC1": 101, "PWM_MAIN_TIM0": -3, "PWM_MAIN_TIM1": 400}
    assert pr.motor_output_protocol(p)["protocol"] == "mixed"


def test_protocol_can_and_oneshot_and_two_groups():
    assert pr.motor_output_protocol({"UAVCAN_EC_FUNC1": 101})["protocol"] == "can"
    assert pr.motor_output_protocol({"PWM_MAIN_FUNC1": 101, "PWM_MAIN_TIM0": -1})["protocol"] == "oneshot"
    two = pr.motor_output_protocol({"PWM_MAIN_FUNC1": 101, "PWM_MAIN_TIM0": 400, "UAVCAN_EC_FUNC1": 102})
    assert two == dict(groups=["PWM_MAIN", "UAVCAN_EC"], protocol="mixed")


def test_protocol_legacy_parameters():
    assert pr.motor_output_protocol({"DSHOT_CONFIG": 600, "PWM_MAIN_RATE": 400})["protocol"] == "dshot"
    assert pr.motor_output_protocol({"DSHOT_CONFIG": 0, "PWM_MAIN_RATE": 0})["protocol"] == "oneshot"
    assert pr.motor_output_protocol({"DSHOT_CONFIG": 0, "PWM_MAIN_RATE": 400}) == dict(groups=["legacy"], protocol="pwm")


def test_protocol_unknown_without_motor_functions():
    assert pr.motor_output_protocol({"PWM_MAIN_FUNC1": 0, "PWM_MAIN_TIM0": 400})["protocol"] == "unknown"
    assert pr.motor_output_protocol({"HIL_ACT_FUNC1": 101})["protocol"] == "other"


def test_protocol_group_without_timer_parameters_is_unknown():
    assert pr.motor_output_protocol({"PWM_EXTR_FUNC1": 101})["protocol"] == "unknown"


def test_protocol_motor_function_range_is_101_to_112():
    for f in (100, 113, 201):
        assert pr.motor_output_protocol({"PWM_MAIN_FUNC1": f, "PWM_MAIN_TIM0": 400})["protocol"] == "unknown"
    for f in (101, 112):
        assert pr.motor_output_protocol({"PWM_MAIN_FUNC1": f, "PWM_MAIN_TIM0": 400})["protocol"] == "pwm"
