"""Repeatability measures of experiments/phase4/evidence_tables.py (evidence pack for A4)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import evidence_tables as et   # noqa: E402
import sysid                   # noqa: E402


def test_half_difference_of_identical_fits_is_zero():
    h = dict(b=400.0, tau_m=0.03, Td=0.002)
    d = et.half_diff(h, h, 1 / 800)
    assert all(abs(d[k]) < 1e-12 for k in ("d_b", "d_tau", "dTd_ms", "dTd_samples", "gain_db_10", "phase_deg_10"))


def test_half_difference_values():
    h1, h2 = dict(b=440.0, tau_m=0.033, Td=0.002), dict(b=400.0, tau_m=0.030, Td=0.003)
    d = et.half_diff(h1, h2, 1 / 800)
    assert d["d_b"] == pytest.approx(40 / 420)                    # symmetric relative difference
    assert d["d_tau"] == pytest.approx(0.003 / 0.0315)
    assert d["dTd_ms"] == pytest.approx(-1.0) and d["dTd_samples"] == pytest.approx(-0.8)
    G1 = sysid.plant_model(np.array([10.0]), 440.0, 0.033, 0.002, 1 / 800)[0]
    G2 = sysid.plant_model(np.array([10.0]), 400.0, 0.030, 0.003, 1 / 800)[0]
    assert d["gain_db_10"] == pytest.approx(20 * np.log10(abs(G1 / G2)))
    assert d["phase_deg_10"] == pytest.approx(np.degrees(np.angle(G1 / G2)))


def test_abs_summary():
    s = et.abs_summary([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8, 0.9, -1.0])
    assert s["n"] == 10 and s["median"] == pytest.approx(0.55) and s["p90"] == pytest.approx(np.quantile(np.arange(1, 11) / 10, 0.9))
    assert et.abs_summary([])["n"] == 0


def test_plant_lag_is_motor_lag_plus_pure_delay():
    assert et.plant_lag_deg(0.0, 0.0) == pytest.approx(0.0)
    assert et.plant_lag_deg(1 / (2 * np.pi * 10), 0.0) == pytest.approx(45.0)        # corner at 10 Hz
    assert et.plant_lag_deg(0.0, 0.001) == pytest.approx(3.6)                         # 360 f Td
    assert et.plant_lag_deg(0.02, 0.0005, f=5.0) == pytest.approx(np.degrees(np.arctan(2 * np.pi * 5 * 0.02)) + 0.9)


def test_prior_lag_range_is_spanned_by_the_prior_corners():
    lo, hi = et.prior_lag_range((0.02, 0.06), (0.0005, 0.005))
    assert lo == pytest.approx(et.plant_lag_deg(0.02, 0.0005)) and hi == pytest.approx(et.plant_lag_deg(0.06, 0.005))


def test_corner_perturbation_at_10_hz():
    gain, phase = et.corner_perturbation(0.025, 0.001, 800.0, 0.0, 0.0)
    assert gain == pytest.approx(0.0, abs=1e-12) and phase == pytest.approx(0.0, abs=1e-12)
    gain, phase = et.corner_perturbation(0.025, 0.001, 800.0, 0.2, 0.0)
    G0 = sysid.plant_model(np.array([10.0]), 1.0, 0.025, 0.001, 1 / 800)[0]
    ratios = [sysid.plant_model(np.array([10.0]), sb, 0.025 * st, 0.001, 1 / 800)[0] / G0
              for sb in (0.8, 1.2) for st in (0.8, 1.2)]
    assert gain == pytest.approx(max(abs(20 * np.log10(abs(q))) for q in ratios), rel=1e-9)
    assert phase == pytest.approx(max(-np.degrees(np.angle(q)) for q in ratios), rel=1e-9)
    assert phase == pytest.approx(np.degrees(np.arctan(2 * np.pi * 10 * 0.03) - np.arctan(2 * np.pi * 10 * 0.025)), rel=1e-6)
    _, phase = et.corner_perturbation(0.025, 0.001, 800.0, 0.0, 1.0)
    assert phase == pytest.approx(360 * 10 / 800.0)                    # one controller sample at 10 Hz


def test_bootstrap_ci_of_the_90th_percentile():
    lo, hi = et.bootstrap_ci([1.0] * 15)
    assert lo == hi == 1.0
    v = np.linspace(0.1, 1.5, 15)
    lo, hi = et.bootstrap_ci(v, seed=1)
    assert lo <= np.quantile(np.abs(v), 0.9) <= hi and lo < hi
    assert et.bootstrap_ci(v, seed=1) == (lo, hi)                       # reproducible
    assert et.bootstrap_ci([], seed=1) == (None, None)


def test_log_funnel_counts_each_log_once_at_its_best_stage():
    ok = dict(b=400.0, tau_m=0.03, Td=0.002, n_freq=12, mag_err_db=1.0, phase_err_deg=3.0,
              at_bound=dict(tau_m=False, Td_zero=False, Td_max=False))
    bad = dict(ok, at_bound=dict(tau_m=True, Td_zero=False, Td_max=False))
    rows = [dict(log_id="A", axis=0, cls="att_sp", full=ok), dict(log_id="A", axis=1, cls="att_sp", full=bad),
            dict(log_id="B", axis=0, cls="att_sp", full=bad),
            dict(log_id="C", axis=0, cls="both", full=dict(error="too few frequencies")),
            dict(log_id="D", axis=0, error="no airborne gap-free piece as long as one Welch segment (0.8 s)"),
            dict(log_id="E", axis=0, error="pieces: ValueError('missing topic: vehicle_torque_setpoint')"),
            dict(log_id="E", axis=1, error="pieces: ValueError('missing topic: vehicle_torque_setpoint')")]
    f = et.log_funnel(rows)
    assert f == {"accepted identification": 1, "fit failed the gates": 1, "too little excitation": 1,
                 "no gap-free piece of one Welch segment": 1, "missing topic": 1}


def test_coverage_counts_pairs_inside_their_own_set():
    full = dict(b=400.0, tau_m=0.025, Td=0.001)
    same = dict(b=400.0, tau_m=0.025, Td=0.001)
    far = dict(b=800.0, tau_m=0.025, Td=0.001)                      # +6 dB: outside every candidate
    r = dict(f_ctrl=800.0, full=full)
    pairs = [("A", 0, r, same, same), ("B", 0, r, same, far), ("C", 0, r, same, same)]
    cov, n = et.coverage(pairs, lambda log, ax: None if log == "C" else 12.0, candidates=((0.15, 1.0),))
    assert n == 2 and cov[(0.15, 1.0)] == pytest.approx(0.5)
