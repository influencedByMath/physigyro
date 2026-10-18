"""Robust stability evaluation for decisions A2-A4 (THESIS 7.1.1 Tier 3 step 4, 7.8 steps 2-3; D-12)."""
import json
import os

import numpy as np
import pytest

import robust as rb
from loop_model import FilterSettings, Gains, LoopCase, Plant, Rates, margins, PX4_V116_MC_DEFAULT

ROOT = os.path.join(os.path.dirname(__file__), "..")
P0 = Plant(b=431.1, tau_m=0.0292, Td=0.00261)
G0 = Gains(0.21, 0.14, 0.00364)
R0 = Rates(6400.0, 800.0)
S0 = FilterSettings(f_gyro=80.0, f_dterm=40.0, dnf_esc=True, dnf_bw=10.0, dnf_hmc=2, dnf_min=50.0)
M0 = np.array([97.9, 109.5, 102.7, 102.7])


def test_open_loop_gives_the_loopcase_margins():
    f = np.linspace(0.5, 1500, 3000)
    case = LoopCase(P0, G0, R0, f, np.full(len(f), 1e-6), motor_hz=M0, reference=PX4_V116_MC_DEFAULT)
    m_case = case.evaluate(S0, _noise=False)
    fg = rb.stability_grid(R0)
    m = margins(fg, rb.open_loop(fg, P0, G0, R0, S0, M0), 2)
    for k in ("PM", "GM", "Ms", "fc"):
        assert float(m[k]) == pytest.approx(m_case[k], rel=1e-9)


def test_corner_plants_are_nominal_plus_eight_corners():
    c = rb.corner_plants(P0, 0.2, 0.15, 0.00125)
    assert len(c) == 9 and c[0] == P0
    assert sorted({round(p.b / P0.b, 6) for p in c[1:]}) == [0.8, 1.2]
    assert sorted({round(p.tau_m / P0.tau_m, 6) for p in c[1:]}) == [0.85, 1.15]
    assert sorted({round(p.Td - P0.Td, 9) for p in c[1:]}) == [0.0, 0.00125]


def test_robust_metrics_worst_case_is_the_extreme_over_all_plants():
    rm = rb.robust_metrics(P0, G0, R0, S0, M0, 0.2, 0.2, 1 / 800)
    fg = rb.stability_grid(R0)
    each = [margins(fg, rb.open_loop(fg, p, G0, R0, S0, M0), 2) for p in rb.corner_plants(P0, 0.2, 0.2, 1 / 800)]
    assert rm["worst"]["PM"] == pytest.approx(min(float(m["PM"]) for m in each))
    assert rm["worst"]["Ms"] == pytest.approx(max(float(m["Ms"]) for m in each))
    assert rm["worst"]["GM"] == pytest.approx(min(float(m["GM"]) for m in each))
    assert rm["worst"]["stable"] == all(bool(m["stable"]) for m in each)
    assert rm["nominal"]["PM"] == pytest.approx(float(each[0]["PM"]))


def test_reproduces_the_feasibility_realism_check():
    """THESIS 7.1.1 / REPORT 5: pilot roll setting PM 39.5 deg, Ms 2.10; worst of 8 corners (+-20 %, +1 sample)
    PM 23.4 deg, Ms 3.87, GM 3.35 dB."""
    res = json.load(open(os.path.join(ROOT, "experiments", "feasibility", "results.json")))["plant"]["realism"]["roll"]
    rm = rb.robust_metrics(P0, G0, R0, S0, M0, 0.2, 0.2, 1 / 800)
    assert rm["nominal"]["PM"] == pytest.approx(res["pilot actual"]["PM"], abs=0.05)
    assert rm["nominal"]["Ms"] == pytest.approx(res["pilot actual"]["Ms"], abs=0.01)
    w = res["pilot actual, worst of 8 plant corners"]
    assert rm["worst_corners"]["PM"] == pytest.approx(w["PM"], abs=0.05)
    assert rm["worst_corners"]["Ms"] == pytest.approx(w["Ms"], abs=0.01)
    assert rm["worst_corners"]["GM"] == pytest.approx(w["GM"], abs=0.01)


def _rm(nPM, nMs, nGM, wPM, wMs, wGM, wst=True, nst=True):
    return dict(nominal=dict(PM=nPM, Ms=nMs, GM=nGM, stable=nst), worst=dict(PM=wPM, Ms=wMs, GM=wGM, stable=wst))


def test_passes_with_the_same_limits_for_nominal_and_worst_case():
    lim = dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0)
    assert rb.passes(_rm(40, 1.8, 8, 31, 1.95, 6.5), **lim)
    assert not rb.passes(_rm(40, 1.8, 8, 29, 1.95, 6.5), **lim)       # worst-case PM
    assert not rb.passes(_rm(40, 1.8, 8, 31, 2.05, 6.5), **lim)       # worst-case Ms
    assert not rb.passes(_rm(40, 1.8, 8, 31, 1.95, 5.9), **lim)       # worst-case GM
    assert not rb.passes(_rm(40, 1.8, 8, 31, 1.95, 6.5, wst=False), **lim)
    assert not rb.passes(_rm(29, 1.8, 8, 31, 1.95, 6.5), **lim)       # nominal


def test_passes_with_separate_worst_case_limits():
    lim = dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0, wc=dict(PM=20.0, Ms=3.0, GM=3.0))
    assert rb.passes(_rm(35, 1.9, 7, 21, 2.9, 3.5), **lim)
    assert not rb.passes(_rm(35, 1.9, 7, 19, 2.9, 3.5), **lim)
    assert not rb.passes(_rm(35, 1.9, 7, 21, 3.1, 3.5), **lim)
    assert not rb.passes(_rm(35, 1.9, 7, 21, 2.9, 2.9), **lim)
    assert not rb.passes(_rm(35, 2.1, 7, 21, 2.9, 3.5), **lim)        # the nominal limits still apply
    assert not rb.passes(_rm(35, 1.9, 7, 21, 2.9, 3.5, wst=False), **lim)


def test_control_imu_raw_rate_follows_sensor_selection():
    statuses = [(3407898, 6377.1), (3407890, 6446.3), (3407882, 6325.8)]
    assert rb.control_imu_raw_rate(3407882, statuses) == pytest.approx(6325.8)
    assert rb.control_imu_raw_rate(None, statuses) is None
    assert rb.control_imu_raw_rate(123, statuses) is None


def test_separate_limits_still_require_the_nominal_gain_margin():
    lim = dict(PM_min=30.0, Ms_max=2.0, GM_min=6.0, wc=dict(PM=20.0, Ms=3.0, GM=3.0))
    assert not rb.passes(_rm(35, 1.9, 5.9, 21, 2.9, 3.5), **lim)


def test_critical_frequencies_of_a_known_loop():
    # L = K exp(-s T) / s: phase -90 - 360 f T, so the phase crossover is at f = 1 / (4 T); |1 + L| is smallest there
    T, K = 0.01, 2 * np.pi * 5.0
    f = np.geomspace(0.1, 200.0, 20000)
    L = K * np.exp(-2j * np.pi * f * T) / (2j * np.pi * f)
    fm, f180 = rb.critical_frequencies(f, L)
    assert f180 == pytest.approx(25.0, rel=1e-3)
    assert fm == pytest.approx(f[np.argmin(np.abs(1 + L))])


def test_robust_metrics_report_where_the_margins_bind():
    rm = rb.robust_metrics(P0, G0, R0, S0, M0, 0.2, 0.2, 1 / 800)
    assert 5.0 < rm["nominal"]["f_Ms"] < 40.0 and 5.0 < rm["nominal"]["f_180"] < 60.0
    assert rm["nominal"]["f_Ms"] > rm["nominal"]["fc"]


def test_corner_metrics_list_every_plant_with_its_critical_frequencies():
    """Q1 of the author's A4 questions: which corner fails, and where its margins bind."""
    d = 1 / 800
    rows = rb.corner_metrics(P0, G0, R0, S0, M0, 0.2, 0.2, d)
    rm = rb.robust_metrics(P0, G0, R0, S0, M0, 0.2, 0.2, d)
    assert len(rows) == 9 and rows[0]["label"] == "nominal"
    assert [r["label"] for r in rows[1:]] == ["b- tau- Td", "b- tau- Td+d", "b- tau+ Td", "b- tau+ Td+d",
                                              "b+ tau- Td", "b+ tau- Td+d", "b+ tau+ Td", "b+ tau+ Td+d"]
    assert max(r["Ms"] for r in rows) == pytest.approx(rm["worst"]["Ms"], rel=1e-12)
    assert min(r["PM"] for r in rows) == pytest.approx(rm["worst"]["PM"], rel=1e-12)
    assert min(r["GM"] for r in rows) == pytest.approx(rm["worst"]["GM"], rel=1e-12)
    assert rows[0]["Ms"] == pytest.approx(rm["nominal"]["Ms"], rel=1e-12)
    assert rows[0]["f_Ms"] == rm["nominal"]["f_Ms"]
    fg = rb.stability_grid(R0)
    for r, p in zip(rows, rb.corner_plants(P0, 0.2, 0.2, d)):
        assert (r["b"], r["tau_m"], r["Td"]) == (p.b, p.tau_m, p.Td)
        f_ms, f180 = rb.critical_frequencies(fg, rb.open_loop(fg, p, G0, R0, S0, M0))
        assert r["f_Ms"] == f_ms
        assert r["f_180"] == f180 or (np.isnan(f180) and np.isnan(r["f_180"]))
