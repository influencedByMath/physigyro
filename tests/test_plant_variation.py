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
