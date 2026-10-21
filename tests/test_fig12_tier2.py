"""Fig. 12 (E12, D-42): histograms of the Tier 2 identifications, fit and test halves separately: tau_m, T_d net of the
chip filter, the unfiltered crossover, the PM of the flown setting."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import fig12_tier2 as f12       # noqa: E402
import tier3 as t3              # noqa: E402

P4 = os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4")


@pytest.fixture(scope="module")
def rows():
    return f12.load_rows()


def _task3(half):
    cases = {c["id"]: c for c in json.load(open(os.path.join(P4, f"{half}_rq1_cases.json")))["cases"]}
    res = {r["id"]: r for r in json.load(open(os.path.join(P4, f"{half}_rq1_realgain.json")))["results"]}
    return cases, res


def test_the_fit_half_holds_the_30_identifications_and_28_flown_loops(rows):
    fit = [r for r in rows if r["half"] == "fit"]
    assert len(fit) == 30 and len({r["vehicle_key"] for r in fit}) == 17
    flown = [r for r in fit if r["flown_valid"]]
    assert len(flown) == 28 and len({r["vehicle_key"] for r in flown}) == 16
    assert sorted(r["id"] for r in fit if not r["flown_valid"]) == ["fit|a3f3ae23|pitch", "fit|a3f3ae23|roll"]
    assert all(r["fc_unfiltered"] is None and r["pm_flown"] is None for r in fit if not r["flown_valid"])


def test_the_test_half_holds_the_28_task3_cases(rows):
    test = [r for r in rows if r["half"] == "test"]
    cases, _ = _task3("test")
    assert sorted(r["id"] for r in test) == sorted(cases) and len(test) == 28
    assert len({r["vehicle_key"] for r in test}) == 19 and all(r["flown_valid"] for r in test)


@pytest.mark.parametrize("half", ["fit", "test"])
def test_every_task3_case_gets_its_reported_crossover_and_flown_pm(rows, half):
    cases, res = _task3(half)
    by_id = {r["id"]: r for r in rows}
    assert set(cases) <= set(by_id)
    for cid, c in cases.items():
        r = by_id[cid]
        assert r["fc_unfiltered"] == c["fc_unfiltered"]
        assert r["pm_flown"] == pytest.approx(res[cid]["reference"]["PM"], abs=1e-9)
        assert r["tau_ms"] == pytest.approx(1e3 * c["plant"]["tau_m"], abs=1e-12)


def test_the_two_fit_cases_without_a_window_are_computed_the_same_way(rows):
    b = [r for r in rows if r["id"].startswith("fit|b79d2e3d")]
    assert len(b) == 2 and all(r["flown_valid"] for r in b)
    assert all(np.isfinite(r["fc_unfiltered"]) and np.isfinite(r["pm_flown"]) for r in b)


def test_td_net_subtracts_the_lower_chip_delay_of_the_control_chip_and_clamps_at_zero():
    low = {"ICM42688P": 0.000906}
    assert f12.td_net_ms(dict(plant=dict(Td=0.002), control_chip="ICM42688P", fifo_chip="BMI088"), low) == pytest.approx(1.094)
    assert f12.td_net_ms(dict(plant=dict(Td=0.002), control_chip=None, fifo_chip="ICM42688P"), low) == pytest.approx(1.094)
    assert f12.td_net_ms(dict(plant=dict(Td=0.0005), control_chip="ICM42688P", fifo_chip="ICM42688P"), low) == 0.0
    assert f12.td_net_ms(dict(plant=dict(Td=0.002), control_chip="BMI088", fifo_chip="BMI088"), low) == pytest.approx(2.0)


def test_rows_use_the_net_td(rows):
    low = f12.chip_lower()
    for r in rows:
        assert r["td_net_ms"] == pytest.approx(f12.td_net_ms(r["case"], low), abs=1e-12)
    assert any(r["td_net_ms"] < 1e3 * r["case"]["plant"]["Td"] for r in rows)        # some chips have a filter


def test_summary_counts_and_prior_coverage(rows):
    s = f12.summary(rows)
    assert s["fit"]["tau_ms"]["n"] == 30 and s["fit"]["fc_unfiltered"]["n"] == 28 and s["test"]["pm_flown"]["n"] == 28
    fit_tau = [r["tau_ms"] for r in rows if r["half"] == "fit"]
    assert s["fit"]["tau_ms"]["median"] == pytest.approx(float(np.median(fit_tau)))
    assert s["fit"]["tau_ms"]["inside_prior"] == sum(15.0 <= v <= 60.0 for v in fit_tau)
    test_fc = [r["fc_unfiltered"] for r in rows if r["half"] == "test"]
    assert s["test"]["fc_unfiltered"]["inside_prior"] == sum(3.0 <= v <= 10.0 for v in test_fc)
    pm = [r["pm_flown"] for r in rows if r["half"] == "test"]
    assert s["test"]["pm_flown"]["below_floor"] == sum(v < 30.0 for v in pm)


def test_prior_marks_are_the_decided_values():
    assert f12.PRIORS["tau_ms"] == pytest.approx(tuple(1e3 * x for x in t3.TAU_M))
    assert f12.PRIORS["td_net_ms"] == pytest.approx(tuple(1e3 * x for x in t3.TD))
    assert f12.PRIORS["fc_unfiltered"] == t3.F_C
    assert f12.PM_FLOOR == t3.NOMINAL["PM_min"]


def test_figure_has_one_row_per_half_and_one_column_per_quantity(rows, tmp_path):
    out = f12.make_figure(rows, str(tmp_path))
    assert [p["half"] for p in out["panels"]] == ["fit"] * 4 + ["test"] * 4
    assert [p["key"] for p in out["panels"]] == list(f12.KEYS) * 2
    for p in out["panels"]:
        vals = [r[p["key"]] for r in rows if r["half"] == p["half"] and r[p["key"]] is not None]
        assert sum(p["counts"]) == len(vals)
        assert np.array_equal(p["counts"], np.histogram(vals, bins=p["edges"])[0])
    edges = {p["key"]: [] for p in out["panels"]}
    for p in out["panels"]:
        edges[p["key"]].append(list(p["edges"]))
    assert all(e[0] == e[1] for e in edges.values())                 # the same bins in both halves
    assert (tmp_path / "fig12_tier2.png").exists() and (tmp_path / "fig12_tier2.pdf").exists()


def test_caption_states_the_computed_numbers(rows):
    s = f12.summary(rows)
    cap = f12.caption(s)
    assert "30 identifications from 17 vehicles" in cap and "28 cases from 19 vehicles" in cap
    assert f"{s['fit']['tau_ms']['inside_prior']} of 30" in cap
    assert f"{s['test']['fc_unfiltered']['inside_prior']} of 28" in cap
    assert "|" not in cap.replace("\n", "")                           # no table pipes in the caption text
