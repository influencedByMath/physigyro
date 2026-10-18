"""Helpers of experiments/phase4/a4_followup.py: the author's questions before finalizing A4 (2026-10-18)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import a4_followup as af        # noqa: E402
import plant_variation as pv    # noqa: E402
import robust as rb             # noqa: E402
import sysid                    # noqa: E402
from loop_model import FilterSettings, Gains, Plant, Rates   # noqa: E402

P0 = Plant(b=431.1, tau_m=0.0292, Td=0.00261)
G0 = Gains(0.21, 0.14, 0.00364)
R0 = Rates(6400.0, 800.0)
S0 = FilterSettings(f_gyro=80.0, f_dterm=40.0, dnf_esc=True, dnf_bw=10.0, dnf_hmc=2, dnf_min=50.0)
M0 = np.array([97.9, 109.5, 102.7, 102.7])


def _rm(PM, Ms, GM, wPM, wMs, wGM, stable=True):
    return dict(nominal=dict(PM=PM, Ms=Ms, GM=GM, stable=True), worst=dict(PM=wPM, Ms=wMs, GM=wGM, stable=stable))


def test_ms_wc_sweep_counts_with_the_implied_margins():
    rows = [_rm(45, 1.8, 9, 30, 2.4, 5.0),      # passes every wc >= 2.5
            _rm(45, 1.9, 9, 25, 3.2, 3.9),      # passes 3.5 and 4 only
            _rm(45, 1.9, 9, 25, 2.2, 3.0),      # GM 3.0 dB < 3.52 (Ms 3) but >= 2.92 (Ms 3.5)
            _rm(25, 2.5, 5, 10, 6.0, 1.0)]      # nominal fail
    out = af.ms_wc_sweep(rows, (2.5, 3.0, 3.5, 4.0))
    assert out[3.0]["n_nominal"] == 3 and out[3.0]["n_all"] == 4
    assert [out[m]["n_pass"] for m in (2.5, 3.0, 3.5, 4.0)] == [1, 1, 3, 3]
    assert out[3.0]["PM"] == pytest.approx(19.188, abs=1e-3) and out[3.0]["GM"] == pytest.approx(3.522, abs=1e-3)
    assert out[3.0]["share_of_nominal"] == pytest.approx(1 / 3)


def test_failing_cases_pass_nominally_but_fail_the_worst_case():
    rows = [_rm(45, 1.8, 9, 30, 2.4, 5.0), _rm(45, 1.9, 9, 25, 3.2, 3.9), _rm(25, 2.5, 5, 10, 6.0, 1.0),
            _rm(45, 1.9, 9, 25, 2.9, 3.9, stable=False)]
    assert af.failing_cases(rows, ms_wc=3.0) == [1, 3]


def test_measured_open_loop_equals_the_model_loop_for_an_exact_frf():
    f = np.geomspace(0.5, 60.0, 300)
    G = sysid.plant_model(f, P0.b, P0.tau_m, P0.Td, 1 / R0.f_ctrl)       # what the identification measures
    parts = af.loop_parts(f, G0, R0, S0, M0)
    L = rb.open_loop(f, P0, G0, R0, S0, M0)
    assert np.max(np.abs(af.measured_open_loop(G, f, 0.0, parts) / L - 1)) < 1e-12
    # a Td correction of the control chip acts as a pure delay on the measured response
    L2 = rb.open_loop(f, Plant(P0.b, P0.tau_m, P0.Td + 0.0004), G0, R0, S0, M0)
    assert np.max(np.abs(af.measured_open_loop(G, f, 0.0004, parts) / L2 - 1)) < 1e-12


def test_frf_random_error_follows_coherence_and_averages():
    assert af.frf_rel_error(0.5, 100) == pytest.approx(np.sqrt(0.5 / (0.5 * 100)))
    assert af.frf_rel_error(0.9, 100) < af.frf_rel_error(0.5, 100) < af.frf_rel_error(0.5, 25)


def test_refit_case_replaces_the_plant_and_applies_the_chip_correction():
    case = dict(plant=dict(b=400.0, tau_m=0.03, Td=0.0015), td_identified=0.0010, gains={}, exclude=False)
    new = af.refit_case(case, dict(b=420.0, tau_m=0.025, Td=0.0020), 0.0005)
    assert new["plant"] == dict(b=420.0, tau_m=0.025, Td=pytest.approx(0.0025))      # FIFO -> control chip correction
    assert new["td_identified"] == 0.0020 and case["plant"]["b"] == 400.0              # input not modified
    assert af.refit_case(case, dict(b=1, tau_m=1, Td=0.0005), -0.0008)["plant"]["Td"] == 0.0   # clamp of case_from_log
    assert af.refit_case(case, dict(b=1, tau_m=1, Td=0.0005), None)["plant"]["Td"] == 0.0005   # control chip unknown


def _sim_pieces():
    from test_sysid_pieces import _closed_loop
    r, u, w, Tc = _closed_loop()
    return [dict(t=np.arange(len(r)) * Tc, u=u, w=w, att_sp=r, rate_sp=r)], Tc


def test_variant_fits_reproduce_d14_and_double_the_segment():
    pieces, Tc = _sim_pieces()
    fits = af.variant_fits(pieces, Tc, 1024, "att_sp", af.ID_VARIANTS)
    assert list(fits) == list(af.ID_VARIANTS)
    assert fits["V0"] == pv.fit_pieces(pieces, Tc, 1024, "att_sp")          # V0 is the evidence-pack identification
    assert fits["V3"]["n_seg"] < fits["V0"]["n_seg"] / 1.9                  # 2.56 s segments: about half as many
    assert fits["V1"]["f_used"][1] <= 40.0


def test_variant_fits_record_a_failed_variant_instead_of_raising():
    pieces, Tc = _sim_pieces()
    short = [{k: v[:1500] for k, v in pieces[0].items()}]                   # 1.9 s: too short for 2.56 s segments
    fits = af.variant_fits(short, Tc, 1024, "att_sp", af.ID_VARIANTS)
    assert "error" in fits["V3"] and "error" in fits["V4"]


def test_feasible_filters_count_every_passing_grid_point():
    case = dict(plant=dict(b=P0.b, tau_m=P0.tau_m, Td=P0.Td), gains=dict(Kp=G0.Kp, Ki=G0.Ki, Kd=G0.Kd),
                fs=R0.fs, f_ctrl=R0.f_ctrl, motor_hz=M0.tolist(),
                settings=dict(f_gyro=S0.f_gyro, f_dterm=S0.f_dterm, notches=[], dnf_esc=True, dnf_bw=10.0, dnf_hmc=2,
                              dnf_min=50.0))
    out = af.feasible_filters(case, n_g=3, n_d=2, ms_wc=2.0)    # strict worst case: some nominal passes fail it
    for name, fields in af.NOTCH_OPTIONS.items():
        o = out[name]
        assert o["n_grid"] == 6
        lim = af.separate_limits(2.0)
        brute = nom = 0
        for fg in o["f_gyro"]:
            for fd in o["f_dterm"]:
                s = af.settings_with(case, f_gyro=fg, f_dterm=fd, **fields)
                rm = rb.robust_metrics(P0, G0, R0, s, M0, 0.2, 0.2, 1 / R0.f_ctrl)
                brute += rb.passes(rm, **lim)
                nom += af.nominal_ok(rm)
        assert o["n_pass"] == brute and o["n_nominal"] == nom
        assert brute < nom                                        # the two counts differ in this case
        assert o["best"]["f_gyro"] in o["f_gyro"] and o["best"]["f_dterm"] in o["f_dterm"]
        assert o["best"]["passes"] == (brute > 0)
