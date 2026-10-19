"""Real-gain cases for Task 3 (D-15, D-36 (6), D-37 (2)): identification and steady windows from the same log."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import rq1_cases as rc          # noqa: E402
from loop_model import Gains, Plant, Rates, design_gains   # noqa: E402

F = np.arange(0, 1000, 1.5)


def native_of(windows):
    out = {}
    for k, (w, scale) in enumerate(windows):
        out[w + "|f"] = F
        out[w + "|P"] = np.stack([scale * np.ones_like(F), 2 * scale * np.ones_like(F), np.ones_like(F)], 1)
    return out


def win(log, i, rpm=None, tracking=False):
    return dict(window_id=f"{log}|w{i}", log_id=log, vehicle_key="v-" + log, rpm_valid=str(rpm is not None),
                rpm_tracking=str(tracking), rpm_hz="null" if rpm is None else str(rpm), t0=str(10.0 * i),
                t1=str(10.0 * i + 10), fifo_chip="BMI088")


CASE = dict(log_id="L1", axis=1, cls="att_sp", plant=dict(b=400.0, tau_m=0.03, Td=0.002),
            gains=dict(Kp=0.1, Ki=0.2, Kd=0.002), f_ctrl=800.0, fs=8000.0,
            settings=dict(f_gyro=60.0, f_dterm=40.0, notches=[], dnf_esc=False, dnf_bw=15.0, dnf_hmc=3, dnf_min=25.0),
            motor_hz=None, flags=[], exclude=False, fifo_chip="BMI088", control_chip="BMI088")
LOGS = [dict(log_id="L1", vehicle_key="v-L1", nperseg=1333,
             fc=dict(default=dict(f_gyro=40.0, f_dterm=30.0, notches=[], dnf_esc=False, dnf_bw=15.0, dnf_hmc=3,
                                  dnf_min=25.0), defaults_known=True, esc_rate_hz=None)),
        dict(log_id="L2", vehicle_key="v-L2", nperseg=1333,
             fc=dict(default=CASE["settings"], defaults_known=False, esc_rate_hz=200.0))]


def test_case_spectrum_is_the_power_mean_of_the_case_axis_over_its_windows():
    rows = [win("L1", 0), win("L1", 1)]
    nat = native_of([("L1|w0", 1.0), ("L1|w1", 3.0)])
    f, P = rc.case_spectrum(rows, nat, axis=1)
    assert np.array_equal(f, F) and np.allclose(P, 2 * (1.0 + 3.0) / 2)       # pitch column, mean of 2 and 6
    nat["L1|w1|f"] = F + 0.5
    with pytest.raises(ValueError):
        rc.case_spectrum(rows, nat, axis=1)


def test_window_rpm_only_when_every_window_has_valid_rpm():
    assert rc.window_rpm([win("L1", 0, [100.0, 104.0]), win("L1", 1, [102.0, 106.0])]) == pytest.approx([101.0, 105.0])
    assert rc.window_rpm([win("L1", 0, [100.0, 104.0]), win("L1", 1)]) is None


def test_unfiltered_crossover_is_the_design_crossover():
    rates, plant = Rates(6400.0, 800.0), Plant(1.0, 0.03, 0.002)
    g, m = design_gains(6.0, 1.5, plant, rates, i_ratio=0.03)
    fc, pm = rc.unfiltered_crossover(plant, g, rates)
    assert fc == pytest.approx(6.0, rel=0.01) and pm == pytest.approx(float(m["PM"]), abs=0.2)


def test_build_cases_needs_windows_of_the_same_log_and_drops_excluded_cases():
    cases = [CASE, dict(CASE, log_id="L2", axis=0), dict(CASE, log_id="L3"), dict(CASE, log_id="L1", axis=0, exclude=True)]
    rows = [win("L1", 0), win("L1", 1), win("L2", 0, [100.0, 104.0, 98.0, 101.0], tracking=True)]
    nat = native_of([("L1|w0", 1.0), ("L1|w1", 3.0), ("L2|w0", 1.0)])
    logs = LOGS + [dict(LOGS[0], log_id="L3", vehicle_key="v-L3")]
    out, excluded = rc.build_cases(cases, rows, nat, logs, half="fit")
    assert [(c["log_id"], c["axis"]) for c in out] == [("L1", 1), ("L2", 0)]
    assert {(e["log_id"], e["axis"]): e["reason"] for e in excluded} == {
        ("L3", 1): "no steady window in the identified log", ("L1", 0): "loop parameters changed in flight"}
    a, b = out
    assert a["scenario"] == "A" and a["motor_hz"] is None and a["default"]["f_gyro"] == 40.0 and a["defaults_known"]
    assert a["half"] == "fit" and a["id"] == "fit|L1|pitch" and a["window_ids"] == ["L1|w0", "L1|w1"]
    assert b["scenario"] == "B" and b["motor_hz"] == pytest.approx([100.0, 104.0, 98.0, 101.0]) and b["rpm_tracking"]
    assert b["defaults_known"] is False
    assert a["fc_unfiltered"] > 0 and a["nperseg"] == 1333


def test_tracking_spectra_power_mean_over_windows_for_every_setting():
    rows = [win("L2", 0), win("L2", 1)]
    calls = []

    def fake(t0, t1, bw, hmc, fmin):
        calls.append((t0, t1, bw, hmc, fmin))
        return F, (1.0 if t0 == 0.0 else 3.0) * bw * np.ones_like(F)

    out = rc.tracking_spectra(rows, [(10.0, 2, 25.0), (5.0, 1, 50.0)], fake)
    assert set(out) == {rc.track_key(10.0, 2, 25.0), rc.track_key(5.0, 1, 50.0)}
    f, P = out[rc.track_key(10.0, 2, 25.0)]
    assert np.allclose(P, 20.0) and len(calls) == 4


def test_test_half_candidates_are_full_rate_logs_with_steady_windows():
    recs = [dict(log_id="A", fc=dict(has_highrate=True)), dict(log_id="B", fc=dict(has_highrate=False, has_sysid=False)),
            dict(log_id="C", fc=dict(has_sysid=True))]
    rows = [win("A", 0), win("B", 0)]
    assert [r["log_id"] for r in rc.test_candidates(recs, rows)] == ["A"]


def test_tracking_consistency_check_compares_bins_away_from_every_notch():
    """Far from the notch centres the tracking spectrum equals the library spectrum times the static bank response."""
    from filters import esc_notch_frequencies, notch_response
    f = np.arange(0, 1000, 1.5)
    P = 1e-6 * np.ones_like(f)
    motors = [100.0, 102.0, 104.0, 106.0]
    H2 = np.ones_like(f)
    for c in esc_notch_frequencies(np.array(motors), 2, 30.0, 25.0):
        H2 *= np.abs(notch_response(f, c, 30.0, 6400.0)) ** 2
    f32 = f.astype(np.float32).astype(float)                  # library grids are stored as float32
    assert rc.tracking_consistent(f32, P, f, P * H2, motors, 30.0, 2, 25.0, fs=6400.0)
    assert not rc.tracking_consistent(f32, P, f, 1.05 * P * H2, motors, 30.0, 2, 25.0, fs=6400.0)   # other samples
    assert not rc.tracking_consistent(f32, P, f * 1.001, P * H2, motors, 30.0, 2, 25.0, fs=6400.0)  # other grid
