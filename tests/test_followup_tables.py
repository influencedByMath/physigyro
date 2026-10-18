"""Helpers of experiments/phase4/followup_tables.py (tables for notes/followup_A4_F1_F4.md)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import followup_tables as ft     # noqa: E402


def _corner(label, Ms, f_Ms=15.0):
    return dict(label=label, Ms=Ms, PM=40.0, GM=6.0, f_Ms=f_Ms, f_180=20.0, fc=8.0, stable=True)


def test_worst_corner_and_worst_without_the_delay_step():
    corners = [_corner("nominal", 1.9), _corner("b+ tau- Td", 2.5), _corner("b+ tau- Td+d", 3.4, 17.0),
               _corner("b- tau+ Td+d", 1.7)]
    w = ft.worst_corner(corners)
    assert w["label"] == "b+ tau- Td+d" and w["f_Ms"] == 17.0
    assert ft.worst_without_delay_step(corners) == 2.5


def test_peak_comparison_uses_the_welch_point_nearest_the_peak():
    rows = [dict(f=15.0, coh=0.8, S_meas=1.8, S_model=1.9), dict(f=15.8, coh=0.7, S_meas=1.9, S_model=1.95),
            dict(f=16.6, coh=0.2, S_meas=2.6, S_model=1.9)]
    p = ft.peak_comparison(rows, f_peak=16.0, min_coh=0.5)
    assert p["f"] == 15.8 and p["coh"] == 0.7 and (p["S_meas"], p["S_model"]) == (1.9, 1.95)
    assert p["max_S_meas_coherent"] == 1.9          # the 0.2-coherence point is left out of the maximum
    assert ft.peak_comparison(rows, f_peak=30.0, min_coh=0.5)["f"] == 16.6   # nearest point even if incoherent


def _q2(group, **variants):
    return dict(log="x", axis="roll", group=group, variants=variants)


def test_variant_summary_counts_accepted_passing_lost_and_flipped():
    ok = dict(accepted=True, nominal_ok=True, passes=True, worst=dict(Ms=2.0), nominal=dict(Ms=1.5))
    fail = dict(accepted=True, nominal_ok=True, passes=False, worst=dict(Ms=3.2), nominal=dict(Ms=1.9))
    lost = dict(accepted=False, reasons=["mag_err"])
    rows = [_q2("passes", V0=ok, V1=ok), _q2("passes", V0=ok, V1=lost), _q2("fails worst case", V0=fail, V1=ok),
            _q2("fails nominal", V0=dict(fail, nominal_ok=False), V1=dict(fail, nominal_ok=False))]
    s = ft.variant_summary(rows, ["V0", "V1"])
    assert s["V0"] == dict(accepted=4, nominal_ok=3, passes=2, failing_now_pass=0, lost=0, pass_to_fail=0)
    assert s["V1"] == dict(accepted=3, nominal_ok=2, passes=2, failing_now_pass=1, lost=1, pass_to_fail=0)


def test_pct_formats_shares():
    assert ft.pct(0.4983) == "49.8 %" and ft.pct(None) == "-" and ft.pct(1.0, 0) == "100 %"
