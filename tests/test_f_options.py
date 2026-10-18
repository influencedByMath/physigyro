"""Helpers of experiments/phase4/f_options.py: options for findings F1-F4 and the A5 estimate (2026-10-18)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4")
sys.path.insert(0, HERE)
import f_options as fo          # noqa: E402
import robust_limits as rl      # noqa: E402
from loop_model import FilterSettings, Gains, LoopCase, Plant, Rates, PX4_V116_MC_DEFAULT   # noqa: E402

RATE_ROWS = json.load(open(os.path.join(HERE, "priors_from_logs.json")))["rate_rows"]


def test_population_reproduces_robust_limits_drones():
    a, ta = fo.synthetic_population(RATE_ROWS, 25, 7)
    b, tb, _ = rl.synthetic_drones(RATE_ROWS, 25, 7)
    assert ta == tb
    for x, y in zip(a, b):
        assert {k: x[k] for k in y} == y and x["source"] == "prior" and x["i"] == 0.1


def test_population_with_identified_plants_and_sampled_integral_corner():
    ident = [(0.012, 0.0004), (0.033, 0.0021)]
    a, _ = fo.synthetic_population(RATE_ROWS, 40, 3, identified=ident, identified_share=1.0, i_range=(0.01, 0.08),
                                   pm_window=(0.0, 180.0))
    assert {(d["tau_m"], d["Td"]) for d in a} <= set(ident) and all(d["source"] == "identified" for d in a)
    assert all(0.01 <= d["i"] <= 0.08 for d in a) and len({d["i"] for d in a}) > 30
    half, _ = fo.synthetic_population(RATE_ROWS, 400, 3, identified=ident, identified_share=0.5, pm_window=(0.0, 180.0))
    share = np.mean([d["source"] == "identified" for d in half])
    assert 0.42 < share < 0.58


def test_histogram_has_under_and_overflow_bins():
    h = fo.histogram([3.1, 4.0, 4.9, 9.99, 15.0, 15.9], (4.0, 5.0, 10.0, 15.0))
    assert h == {"< 4": 1, "4-5": 2, "5-10": 1, "10-15": 1, "> 15": 1}


def test_net_td_removes_the_chip_filter_once():
    low = {"ICM42688P": 0.0009, "BMI088": 0.0}
    assert fo.net_td(dict(plant=dict(Td=0.0020), control_chip="ICM42688P", fifo_chip="BMI088"), low) == pytest.approx(0.0011)
    assert fo.net_td(dict(plant=dict(Td=0.0005), control_chip="ICM42688P", fifo_chip="ICM42688P"), low) == 0.0
    # control chip unknown: Td still holds the FIFO chip's filter (case_from_log)
    assert fo.net_td(dict(plant=dict(Td=0.0020), control_chip=None, fifo_chip="ICM42688P"), low) == pytest.approx(0.0011)
    assert fo.net_td(dict(plant=dict(Td=0.0020), control_chip="IIM42652", fifo_chip="IIM42652"), low) == 0.0020


def test_prior_coverage_counts_box_and_lag():
    pairs = [(0.025, 0.001), (0.012, 0.0002), (0.070, 0.006), (0.015, 0.0003)]   # lag 61.1, 37.7, 98.8, 44.4 deg
    cov = fo.prior_coverage(pairs, (0.020, 0.060), (0.0005, 0.005))
    assert cov["n"] == 4 and cov["in_box"] == 1
    assert cov["faster_than_prior"] == 2 and cov["slower_than_prior"] == 1      # lag at 10 Hz outside the prior's range


def test_noise_split_matches_loopcase():
    p, g, r = Plant(431.1, 0.0292, 0.00261), Gains(0.21, 0.14, 0.00364), Rates(6400.0, 800.0)
    s = FilterSettings(f_gyro=80.0, f_dterm=40.0)
    f = np.linspace(0.0, 1500.0, 1001)
    psd = 1e-6 * (1 + (f / 100.0) ** 2) ** -1 + 1e-9
    case = LoopCase(p, g, r, f, psd, reference=s, f_hi=780.0, f_lo=12.0)
    n2_total, n2_below = fo.noise_split(p, g, r, s, None, f, psd, f_lo=12.0, f_split=20.0, f_hi=780.0)
    assert np.sqrt(n2_total) == pytest.approx(case.N_ref, rel=1e-9)
    assert 0 < n2_below < n2_total
    assert fo.noise_split(p, g, r, s, None, f, psd, f_lo=25.0, f_split=20.0, f_hi=780.0)[1] == 0.0


def test_f_lo_rules():
    rules = fo.f_lo_rules(3.2)
    assert rules == {"5 f_c": pytest.approx(16.0), "3 f_c": pytest.approx(9.6), "10 Hz if 5 f_c < 20 Hz": 10.0,
                     "max(5 f_c, 20 Hz)": 20.0}
    assert fo.f_lo_rules(8.0)["10 Hz if 5 f_c < 20 Hz"] == pytest.approx(40.0)


def test_scenario_b_counts_from_windows_and_generator_entries():
    windows = [dict(vehicle_key="a", rpm_valid="True"), dict(vehicle_key="a", rpm_valid="True"),
               dict(vehicle_key="b", rpm_valid="True"), dict(vehicle_key="b", rpm_valid="False"),
               dict(vehicle_key="c", rpm_valid="False")]
    fits = {"a": dict(motor_hz=[80.0] * 4, n_harm=3, dropped=False), "b": dict(motor_hz=None, n_harm=0, dropped=False),
            "c": dict(motor_hz=None, n_harm=0, dropped=False), "d": dict(motor_hz=[90.0] * 4, n_harm=2, dropped=True)}
    c = fo.scenario_b_counts(windows, fits)
    assert c == dict(vehicles=3, with_valid_rpm=2, generator_entries=3, generator_rpm_sources=1)
