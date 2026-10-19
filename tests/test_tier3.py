"""Tier 3 synthetic drones with the decided priors, design and limits (THESIS 7.1.1 Tier 3; D-33, D-36, D-37)."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import robust as rb             # noqa: E402
import tier3 as t3              # noqa: E402
from loop_model import Plant, Rates, Gains, PX4_V116_MC_DEFAULT, design_gains   # noqa: E402

RATE_ROWS = [dict(fs_nominal=8000.0, pub_rate=800.0, fifo_is_control=True, full_rate=True),
             dict(fs_nominal=2000.0, pub_rate=666.7, fifo_is_control=True, full_rate=True),
             dict(fs_nominal=6400.0, pub_rate=800.0, fifo_is_control=False, full_rate=True),    # mixed IMUs: never used
             dict(fs_nominal=8000.0, pub_rate=None, fifo_is_control=True, full_rate=False)]
DELAYS = {"ICM42688P": (0.000906, 0.001372)}


def spectrum_of(chip):
    def draw(rng):
        rng.uniform()                                  # a generator draw consumes its own random numbers
        return dict(chip=chip, motor_hz=None, S=np.ones(3), meta=dict(source=0, scale=1.0))
    return draw


def test_constants_are_the_decided_values():
    assert t3.F_C == (3.0, 10.0) and t3.TAU_M == (0.015, 0.060) and t3.TD == (0.0003, 0.008)       # D-36 (3), (5)
    assert t3.PM_WINDOW == (60.0, 95.0) and t3.I_RATIO == 0.03 and t3.R_RANGE == (0.5, 3.0)      # D-36 (4)
    assert (t3.WIDTH, t3.DELAY_SAMPLES, t3.MS_WC) == (0.20, 1.0, 3.0)                            # D-36 (1)
    assert t3.NOMINAL == dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0)                               # D-30, D-31


def test_worst_case_limits_follow_from_ms_wc():
    wc = t3.worst_case_limits(3.0)
    assert wc["Ms"] == 3.0
    assert wc["PM"] == pytest.approx(math.degrees(2 * math.asin(1 / 6)))          # 19.19 deg
    assert wc["GM"] == pytest.approx(20 * math.log10(1.5))                         # 3.52 dB
    assert t3.limits() == dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0, wc=wc)


def test_rate_pairs_only_from_same_imu_full_rate_logs():
    assert t3.rate_pairs(RATE_ROWS) == [(8000.0, 800.0), (2000.0, 666.7)]


def test_chip_delays_only_for_chips_with_a_fitted_lowpass_term():
    table = {"ICM42688P": {"rms": {"delay_ms": [0.906, 1.372]}, "bic": {"delay_ms": [0.987, 1.041]}},
             "BMI088": {"rms": {"delay_ms": [0.1, 0.2]}, "bic": {"delay_ms": [0.1, 0.2]}}}
    out = t3.chip_delays(table, {"ICM42688P": [323.0, 3.0], "BMI088": None})
    assert list(out) == ["ICM42688P"]
    assert out["ICM42688P"] == pytest.approx((0.000906, 0.001372))               # fit-error rule (D-33), seconds


def test_uncertainty_set_puts_the_lower_bound_in_the_nominal_td_and_the_rest_in_the_step():
    p, dd = t3.plant_and_step(0.025, 0.002, 800.0, "ICM42688P", DELAYS)
    assert (p.b, p.tau_m) == (1.0, 0.025) and p.Td == pytest.approx(0.002 + 0.000906)
    assert dd == pytest.approx(1 / 800.0 + (0.001372 - 0.000906))
    p0, dd0 = t3.plant_and_step(0.025, 0.002, 800.0, "BMI088", DELAYS)
    assert p0.Td == 0.002 and dd0 == pytest.approx(1 / 800.0)


def test_drones_without_a_chip_term_reproduce_the_planning_population():
    """The design draws are those of f_options.synthetic_population with the D-36 parameters (planning run)."""
    import f_options as fo
    ref, _ = fo.synthetic_population(RATE_ROWS, 15, 7, tau=t3.TAU_M, td=t3.TD, pm_window=t3.PM_WINDOW,
                                     i_ratio=t3.I_RATIO, f_c=t3.F_C)
    out = t3.sample_drones(RATE_ROWS, 15, 7, spectrum_of("BMI088"), DELAYS, keep_all=True)
    got = out["drones"]
    assert len(got) == 15
    for a, b in zip(got, ref):
        for k in ("f_c", "tau_m", "Td", "r", "fs", "f_ctrl", "pm_unfiltered"):
            assert a[k] == pytest.approx(b[k], rel=1e-12), k
        assert a["gains"] == pytest.approx(b["gains"], rel=1e-12)


def test_gains_are_designed_on_the_plant_with_the_chip_delay():
    out = t3.sample_drones(RATE_ROWS, 5, 3, spectrum_of("ICM42688P"), DELAYS, keep_all=True)
    for d in out["drones"]:
        assert d["Td_nominal"] == pytest.approx(d["Td"] + 0.000906)
        g, m = design_gains(d["f_c"], d["r"], Plant(1.0, d["tau_m"], d["Td_nominal"]), Rates(d["fs"], d["f_ctrl"]),
                            i_ratio=t3.I_RATIO)
        assert d["gains"] == pytest.approx(dict(Kp=g.Kp, Ki=g.Ki, Kd=g.Kd), rel=1e-12)
        assert d["pm_unfiltered"] == pytest.approx(float(m["PM"]))
        assert t3.PM_WINDOW[0] <= d["pm_unfiltered"] <= t3.PM_WINDOW[1]


def test_kept_drones_meet_the_nominal_limits_at_the_defaults_and_flags_mark_worst_case_failures():
    out = t3.sample_drones(RATE_ROWS, 25, 11, spectrum_of("ICM42688P"), DELAYS)
    assert out["kept"] == 25 and out["draws"] >= out["in_window"] >= out["kept"]
    lim = t3.limits()
    for d in out["drones"]:
        p, dd = t3.plant_and_step(d["tau_m"], d["Td"], d["f_ctrl"], d["chip"], DELAYS)
        rm = rb.robust_metrics(p, Gains(**d["gains"]), Rates(d["fs"], d["f_ctrl"]), PX4_V116_MC_DEFAULT, None,
                               t3.WIDTH, t3.WIDTH, dd)
        n = rm["nominal"]
        assert n["stable"] and n["PM"] >= 30.0 and n["Ms"] <= 2.0 and n["GM"] >= 6.0       # Tier 3 step 3
        assert d["flagged"] == (not rb.passes(rm, **lim))
        assert d["default"]["nominal"]["PM"] == pytest.approx(rm["nominal"]["PM"])


def test_step3_drops_drones_that_fail_nominally_at_the_defaults():
    out = t3.sample_drones(RATE_ROWS, 40, 5, spectrum_of("BMI088"), DELAYS, keep_all=True)
    kept = [d for d in out["drones"] if d["kept"]]
    strict = t3.sample_drones(RATE_ROWS, len(kept), 5, spectrum_of("BMI088"), DELAYS)
    assert [d["f_c"] for d in strict["drones"]] == [d["f_c"] for d in kept]


def test_sampling_is_reproducible():
    a = t3.sample_drones(RATE_ROWS, 6, 9, spectrum_of("ICM42688P"), DELAYS)
    b = t3.sample_drones(RATE_ROWS, 6, 9, spectrum_of("ICM42688P"), DELAYS)
    assert [d["gains"] for d in a["drones"]] == [d["gains"] for d in b["drones"]]
