"""Helpers of experiments/phase4/plant_variation.py (repeated identification by halves; instrument hold)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import plant_variation as pv   # noqa: E402


def _p(t0, n):
    t = t0 + np.arange(n) * 0.01
    return dict(t=t, u=t * 2, w=t * 3, att_sp=t * 4, rate_sp=t * 5)


def test_halves_split_total_time_in_two_equal_parts_in_time_order():
    pieces = [_p(0.0, 30), _p(10.0, 50), _p(20.0, 20)]           # 100 samples
    a, b = pv.halves(pieces)
    assert sum(len(p["t"]) for p in a) == 50 and sum(len(p["t"]) for p in b) == 50
    assert max(p["t"].max() for p in a) < min(p["t"].min() for p in b)
    cut = [p for p in a if p["t"][0] == 10.0][0]                 # the middle piece is cut at sample 20
    assert len(cut["t"]) == 20 and np.allclose(cut["u"], cut["t"] * 2)


def test_halves_keep_all_samples_once():
    pieces = [_p(0.0, 7), _p(5.0, 8)]
    a, b = pv.halves(pieces)
    t_all = np.concatenate([p["t"] for p in a + b])
    assert np.array_equal(np.sort(t_all), np.sort(np.concatenate([p["t"] for p in pieces])))


def test_zoh_holds_previous_instrument_sample_and_marks_stale_gaps():
    t_src = np.array([0.0, 0.05, 0.10, 0.50])
    x = np.array([1.0, 2.0, 3.0, 4.0])
    tg = np.array([-0.01, 0.0, 0.07, 0.15, 0.3, 0.55])
    y = pv._zoh(t_src, x, tg)
    assert np.isnan(y[0])                                         # before the first sample
    assert y[1] == 1.0 and y[2] == 2.0 and y[3] == 3.0            # held
    assert np.isnan(y[4])                                         # 0.2 s after the last sample: a gap
    assert y[5] == 4.0


def test_halves_never_produce_empty_pieces():
    for sizes in ([40, 60], [30, 50, 20], [100], [1, 99], [50, 50]):
        pieces = [_p(10.0 * i, n) for i, n in enumerate(sizes)]
        a, b = pv.halves(pieces)
        assert all(len(p["t"]) > 0 for p in a + b), sizes
        assert abs(sum(len(p["t"]) for p in a) - sum(len(p["t"]) for p in b)) <= 1


def _fit(**kw):
    f = dict(b=400.0, tau_m=0.03, Td=0.002, n_freq=12, mag_err_db=1.5, phase_err_deg=5.0,
             at_bound=dict(tau_m=False, Td_zero=False, Td_max=False))
    f.update(kw)
    return f


def test_accept_fit_gates():
    assert pv.accept(_fit(), "att_sp") == (True, [])
    assert pv.accept(dict(error="too few frequencies"), "att_sp") == (False, ["no fit"])
    assert pv.accept(_fit(mag_err_db=3.5), "att_sp") == (False, ["mag_err"])
    assert pv.accept(_fit(phase_err_deg=16.0), "att_sp") == (False, ["phase_err"])
    assert pv.accept(_fit(at_bound=dict(tau_m=True, Td_zero=False, Td_max=False)), "att_sp") == (False, ["tau_m_bound"])
    assert pv.accept(_fit(at_bound=dict(tau_m=False, Td_zero=False, Td_max=True)), "att_sp") == (False, ["Td_max"])
    assert pv.accept(_fit(at_bound=dict(tau_m=False, Td_zero=True, Td_max=False)), "att_sp")[0]   # Td = 0 is allowed
    assert pv.accept(_fit(agree=dict(ok=False)), "both") == (False, ["instruments_disagree"])
    assert pv.accept(_fit(agree=dict(ok=True)), "both") == (True, [])
    assert pv.accept(_fit(mag_err_db=3.5), "att_sp", max_mag_db=4.0)[0]


class _NoTopics:
    data_list = []

    def get_dataset(self, name):
        raise IndexError("list index out of range")      # what pyulog does for a missing topic


def test_build_pieces_names_a_missing_topic():
    with pytest.raises(ValueError, match="missing topic: vehicle_torque_setpoint"):
        pv.build_pieces(_NoTopics(), None, 0)


def test_a_log_without_usable_pieces_still_gets_a_funnel_row(monkeypatch):
    # review Important 1: 46 of 102 logs left no row at all
    monkeypatch.setattr(pv.nd, "load", lambda path: object())
    monkeypatch.setattr(pv.nd, "gyro_stream", lambda u: object())
    monkeypatch.setattr(pv, "build_pieces", lambda u, gs, axis: ({"att_sp": [], "rate_sp": [], "both": []}, 1 / 800, 640))
    rec = dict(log_id="L", vehicle_key="V", fc=dict(hw="X", fifo_chip=None, control_chip=None, fifo_is_control_gyro=False),
               log_rejections=[])
    rows = pv.analyze(rec)
    assert len(rows) == 2 and all("no airborne gap-free piece" in r["error"] for r in rows)
    assert {r["axis"] for r in rows} == {0, 1}


def test_hold_bridges_the_periodic_controller_pauses_at_the_d14_resolution():
    # review Important 1: in many logs the torque setpoint (and the controller input vehicle_angular_velocity) pauses
    # for 5-6 periods about once per second, so gap-free pieces were about 1.0 s, shorter than one 1.28 s segment.
    # A shorter segment is NOT the fix: at 1.25 Hz resolution fewer bins pass the D-14 "6 coherent frequencies" rule.
    assert pv.SEG_S == 1.28 and pv.MAX_HOLD >= 8
    import sysid
    Tc = 1 / 800
    t = np.arange(0, 10.0, Tc)
    keep = np.ones(len(t), bool)
    for k in range(816, len(t), 816):                                   # a 6-period pause every 1.02 s
        keep[k:k + 5] = False
    pieces = sysid.uniform_pieces(t[keep], Tc, max_hold=pv.MAX_HOLD)
    assert len(pieces) == 1 and len(pieces[0][0]) == len(t)
