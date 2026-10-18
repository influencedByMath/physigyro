"""Helpers of experiments/phase4/robust_limits.py (evidence pack for A2-A4)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import robust_limits as rl     # noqa: E402
from loop_model import Gains, Plant, Rates, PX4_V116_MC_DEFAULT, design_gains   # noqa: E402


def test_worst_case_limits_implied_by_ms():
    w = rl.wc_limits(2.0)                      # Astrom & Murray: Ms = 2 gives PM >= 29 deg, GM >= 6 dB
    assert w["PM"] == pytest.approx(28.955, abs=1e-3) and w["GM"] == pytest.approx(6.0206, abs=1e-4)
    w = rl.wc_limits(3.0)
    assert w["PM"] == pytest.approx(math.degrees(2 * math.asin(1 / 6))) and w["GM"] == pytest.approx(3.5218, abs=1e-4)


def test_limit_sets_cover_the_requested_grid():
    s = rl.limit_sets()
    assert len(s) == 3 * 3 * (1 + 2)
    assert s["PM30_Ms2_same"] == dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0, wc=None)
    assert s["PM25_Ms2.2_wcMs3"]["wc"]["Ms"] == 3.0


def test_chip_delays_read_both_rules_for_both_icm_chips():
    pri = dict(chip_delay={c: dict(bic=dict(delay_ms=[1.0, 1.1]), rms=dict(delay_ms=[0.9, 1.4])) for c in rl.CHIP_CHIPS})
    d = rl.chip_delays(pri)
    assert set(d) == {f"{c}|{r}" for c in rl.CHIP_CHIPS for r in ("bic", "rms")}
    assert d["ICM42688P|rms"] == pytest.approx((0.0009, 0.0014))


def _drone():
    plant, rates = Plant(1.0, 0.03, 0.002), Rates(8000.0, 800.0)
    g, _ = design_gains(8.0, 1.5, plant, rates)
    return plant, g, rates


def test_evaluate_covers_widths_and_delays_and_shifts_the_delay():
    plant, g, rates = _drone()
    e0 = rl.evaluate(plant, g, rates, PX4_V116_MC_DEFAULT, None)
    assert set(e0) == {f"w{w:g}_d{d:g}" for w in rl.WIDTHS for d in rl.DELAYS}
    e1 = rl.evaluate(plant, g, rates, PX4_V116_MC_DEFAULT, None, extra_nominal=0.001)
    assert e1["w0.2_d1"]["nominal"]["PM"] < e0["w0.2_d1"]["nominal"]["PM"] - 2.0     # 1 ms at about 8 Hz: ~3 deg
    e2 = rl.evaluate(plant, g, rates, PX4_V116_MC_DEFAULT, None, extra_corner=0.001)
    assert e2["w0.2_d1"]["nominal"]["PM"] == pytest.approx(e0["w0.2_d1"]["nominal"]["PM"])
    assert e2["w0.2_d1"]["worst"]["PM"] < e0["w0.2_d1"]["worst"]["PM"]
    # a wider set or a longer delay step can only make the worst case worse
    assert e0["w0.3_d1"]["worst"]["PM"] <= e0["w0.1_d1"]["worst"]["PM"]
    assert e0["w0.2_d2"]["worst"]["PM"] <= e0["w0.2_d0.5"]["worst"]["PM"]


def test_pass_rates_are_shares_of_cases():
    ok = dict(nominal=dict(PM=40, Ms=1.5, GM=10, stable=True), worst=dict(PM=35, Ms=1.7, GM=8, stable=True))
    bad = dict(nominal=dict(PM=40, Ms=1.5, GM=10, stable=True), worst=dict(PM=10, Ms=5.0, GM=1, stable=True))
    evals = [{"w0.1_d1": ok}, {"w0.1_d1": bad}, {"w0.1_d1": ok}, {"w0.1_d1": ok}]
    pr = rl.pass_rates(evals, {"X": dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0, wc=None)})
    assert pr == {"X|w0.1_d1": 0.75}


def test_synthetic_drones_follow_the_d28_priors_and_the_pm_window():
    rows = [dict(fs_nominal=8000.0, pub_rate=800.0, fifo_is_control=True, full_rate=True),
            dict(fs_nominal=2000.0, pub_rate=667.0, fifo_is_control=True, full_rate=True),
            dict(fs_nominal=6400.0, pub_rate=800.0, fifo_is_control=False, full_rate=True)]   # not a same-IMU pair
    d, tries, n_pairs = rl.synthetic_drones(rows, 40, seed=1)
    assert len(d) == 40 and tries >= 40 and n_pairs == 2
    for x in d:
        assert 4.0 <= x["f_c"] <= 15.0 and 0.02 <= x["tau_m"] <= 0.06 and 0.0005 <= x["Td"] <= 0.005
        assert 0.5 <= x["r"] <= 3.0 and 45.0 <= x["pm_unfiltered"] <= 75.0
        assert (x["fs"], x["f_ctrl"]) in {(8000.0, 800.0), (2000.0, 667.0)}
    d2, _, _ = rl.synthetic_drones(rows, 40, seed=1)
    assert d2 == d                                                     # reproducible


def test_eval_synth_is_evaluate_at_the_px4_defaults():
    plant, g, rates = _drone()
    d = dict(tau_m=plant.tau_m, Td=plant.Td, gains=dict(Kp=g.Kp, Ki=g.Ki, Kd=g.Kd), fs=rates.fs, f_ctrl=rates.f_ctrl)
    a = rl.eval_synth(d, 0.001, 0.0005)
    b = rl.evaluate(plant, g, rates, PX4_V116_MC_DEFAULT, None, extra_nominal=0.001, extra_corner=0.0005)
    assert a["w0.15_d2"]["worst"]["PM"] == pytest.approx(b["w0.15_d2"]["worst"]["PM"])


def _ev(nPM, nMs, nGM, wPM, wMs, wGM):
    m = dict(nominal=dict(PM=nPM, Ms=nMs, GM=nGM, stable=True), worst=dict(PM=wPM, Ms=wMs, GM=wGM, stable=True))
    return {"w0.1_d1": m, "w0.2_d1": m}


def test_nominal_pass_rates_ignore_the_uncertainty_set():
    evals = [_ev(40, 1.8, 8, 10, 9.0, 1), _ev(28, 1.8, 8, 27, 1.9, 7), _ev(40, 2.1, 8, 39, 2.15, 8)]
    r = rl.nominal_pass_rates(evals)
    assert r["PM30_Ms2"] == pytest.approx(1 / 3, abs=1e-4) and r["PM25_Ms2.2"] == pytest.approx(1.0)   # rounded to 4 decimals
    assert r["PM30_Ms2.2"] == pytest.approx(2 / 3, abs=1e-4)


def test_conditional_rates_count_robust_passes_among_nominal_passes():
    evals = [_ev(40, 1.8, 8, 35, 1.9, 7), _ev(40, 1.8, 8, 10, 9.0, 1), _ev(28, 1.8, 8, 27, 1.9, 7)]
    sets = {"PM30_Ms2_same": dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0, wc=None)}
    c = rl.conditional_rates(evals, sets)
    assert c["PM30_Ms2_same|w0.1_d1"] == pytest.approx(0.5)          # 1 of the 2 nominal passes is robust
    assert set(c) == {"PM30_Ms2_same|w0.1_d1", "PM30_Ms2_same|w0.2_d1"}


def test_fc_bins_split_synthetic_pass_rates_by_crossover():
    synth = [dict(f_c=5.0), dict(f_c=8.0), dict(f_c=12.0), dict(f_c=14.0)]
    evals = [_ev(40, 1.8, 8, 35, 1.9, 7), _ev(40, 1.8, 8, 35, 1.9, 7), _ev(40, 2.5, 8, 35, 1.9, 7), _ev(40, 1.8, 8, 35, 1.9, 7)]
    b = rl.fc_bins(synth, evals, {"PM30_Ms2_same": dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0, wc=None)}, ["w0.1_d1"],
                   edges=(4.0, 7.0, 10.0, 15.0))
    assert b["4-7 Hz"]["n"] == 1 and b["10-15 Hz"]["n"] == 2
    assert b["10-15 Hz"]["nominal|PM30_Ms2"] == pytest.approx(0.5)
    assert b["10-15 Hz"]["PM30_Ms2_same|w0.1_d1"] == pytest.approx(0.5)


def test_synthetic_drones_accept_other_prior_ranges():
    rows = [dict(fs_nominal=8000.0, pub_rate=800.0, fifo_is_control=True, full_rate=True)]
    d, _, _ = rl.synthetic_drones(rows, 30, seed=2, tau=(0.010, 0.015), td=(0.0002, 0.001), f_c=(8.0, 12.0))
    assert all(0.010 <= x["tau_m"] <= 0.015 and 0.0002 <= x["Td"] <= 0.001 and 8.0 <= x["f_c"] <= 12.0 for x in d)


def test_synthetic_drones_stop_when_the_pm_window_cannot_be_met():
    # a very fast plant at a low crossover always has an unfiltered PM above 75 deg: must not loop forever
    rows = [dict(fs_nominal=8000.0, pub_rate=800.0, fifo_is_control=True, full_rate=True)]
    with pytest.raises(RuntimeError, match="tries"):
        rl.synthetic_drones(rows, 5, seed=2, tau=(0.010, 0.012), td=(0.0001, 0.0002), f_c=(5.0, 6.0), max_tries=200)


def test_synthetic_drones_accept_a_pm_window_and_an_i_ratio():
    rows = [dict(fs_nominal=8000.0, pub_rate=800.0, fifo_is_control=True, full_rate=True)]
    d, _, _ = rl.synthetic_drones(rows, 20, seed=3, pm_window=(75.0, 100.0), i_ratio=0.03)
    assert all(75.0 <= x["pm_unfiltered"] <= 100.0 for x in d)
    assert all(x["gains"]["Ki"] / x["gains"]["Kp"] == pytest.approx(0.03 * 2 * np.pi * x["f_c"]) for x in d)


def test_nominal_pass_rates_require_the_gain_margin():
    r = rl.nominal_pass_rates([_ev(40, 1.8, 5.9, 35, 1.9, 5.9)])      # PM and Ms fine, GM below 6 dB
    assert r["PM30_Ms2"] == 0.0
