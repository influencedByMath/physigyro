"""Task 3 RQ1 runner (E3; D-36, D-37): per-case solve, generator draws, resumable parts, Fig. 4 drone choice."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import rq1_run as rr            # noqa: E402
import robust as rb             # noqa: E402
import tier3 as t3              # noqa: E402
from loop_model import FilterSettings, Plant, Rates, PX4_V116_MC_DEFAULT, design_gains   # noqa: E402

F = np.arange(20.0, 780.0 + 1e-9, 0.5)
PSD = 1e-6 * (F / 100.0) ** -1.0 + 1e-4 * np.exp(-0.5 * ((F - 210.0) / 3.0) ** 2)


def drone(fc=5.0, f_ctrl=800.0):
    rates, plant = Rates(6400.0, f_ctrl), Plant(1.0, 0.03, 0.002)
    g, _ = design_gains(fc, 1.2, plant, rates, i_ratio=0.03)
    return plant, g, rates


def test_solve_gives_full_grid_robust_optima_and_the_reference_flags():
    plant, g, rates = drone()
    out = rr.solve(plant, g, rates, F, PSD, None, PX4_V116_MC_DEFAULT, PX4_V116_MC_DEFAULT, "A", 1 / 800.0,
                   n_g=10, n_d=8)
    assert out["f_lo"] == pytest.approx(max(5 * out["reference"]["fc"], 20.0))
    assert out["peaks"] and abs(out["peaks"][0] - 210.0) < 2.0
    lim = t3.limits()
    assert out["reference_flags"]["nominal_ok"] == (out["reference"]["PM"] >= 30 and out["reference"]["Ms"] <= 2.0
                                                   and out["reference"]["GM"] >= 6.0 and out["reference"]["stable"])
    for mode in ("A", "B"):
        r = out[mode]
        assert r["feasible"]
        s = FilterSettings(**{k: (tuple(tuple(x) for x in v) if k == "notches" else v) for k, v in r["settings"].items()})
        rm = rb.robust_metrics(plant, g, rates, s, None, t3.WIDTH, t3.WIDTH, 1 / 800.0)
        assert rb.passes(rm, **lim)
    assert out["A"]["metrics"]["N_rel"] <= 1.0 + 1e-6
    assert out["B"]["metrics"]["PM"] >= out["reference"]["PM"] - 1e-6
    json.dumps(out)                                              # JSON-serialisable


def test_solve_never_returns_an_optimum_worse_than_a_feasible_reference():
    plant, g, rates = drone()
    out = rr.solve(plant, g, rates, F, PSD, None, PX4_V116_MC_DEFAULT, PX4_V116_MC_DEFAULT, "A", 1 / 800.0,
                   n_g=2, n_d=2)
    assert out["reference_flags"]["nominal_ok"] and out["reference_flags"]["worst_ok"]
    assert out["A"]["metrics"]["PM"] >= out["reference"]["PM"] - 1e-9
    assert out["B"]["metrics"]["N_rel"] <= 1.0 + 1e-9


def test_generator_draw_scenario_b_motors_follow_the_source_vehicle_and_the_scale():
    gen = dict(sources=[0, 1])
    entries = [dict(motor_hz=[80.0, 82.0], n_harm=2), dict(motor_hz=None, n_harm=0)]
    calls = iter([(np.ones((1, 3)), [dict(chip="ICM42688P", source=0, scale=1.01, peaks=[])]),
                  (np.ones((1, 3)), [dict(chip="BMI088", source=1, scale=0.99, peaks=[])])])
    draw = rr.generator_draw(gen, entries, F[:3], sample=lambda g, n, f, rng: next(calls))
    a, b = draw(np.random.default_rng(0)), draw(np.random.default_rng(0))
    assert a["chip"] == "ICM42688P" and a["motor_hz"] == pytest.approx([80.8, 82.82])
    assert b["chip"] == "BMI088" and b["motor_hz"] is None


def test_run_part_resumes_without_recomputing_finished_cases(tmp_path):
    calls = []

    def work(item):
        calls.append(item["id"])
        return dict(id=item["id"], value=item["x"] * 2)

    items = [dict(id=f"c{i}", x=i) for i in range(7)]
    out = tmp_path / "part.json"
    rr.run_part(items[:4], work, str(out), meta=dict(seed=1), chunk=3, n_jobs=1, finish=False)
    assert sorted(calls) == ["c0", "c1", "c2", "c3"] and not out.exists()
    calls.clear()
    res = rr.run_part(items, work, str(out), meta=dict(seed=1), chunk=3, n_jobs=1)
    assert sorted(calls) == ["c4", "c5", "c6"]
    assert [r["id"] for r in res["results"]] == [f"c{i}" for i in range(7)] and res["meta"]["seed"] == 1
    assert json.load(open(out))["results"][6]["value"] == 12
    assert not os.path.exists(str(out) + ".partial.jsonl")


def test_figure_drones_follow_fixed_rules():
    synth = [dict(id=f"s{i}", f_c=4.0 + 0.5 * i, A=dict(feasible=True, dpm=float(i))) for i in range(12)]
    real = [dict(id="fit|b27d09e3|roll", flag="worst case", worst_corner_ms=3.03),
            dict(id="fit|4e73d30d|pitch", flag="worst case", worst_corner_ms=3.60),
            dict(id="fit|aaaa|roll", flag="none", worst_corner_ms=2.0)]
    pick = rr.figure_drones(synth, real)
    assert pick["synthetic median"] == "s5" or pick["synthetic median"] == "s6"
    assert pick["synthetic 8-10 Hz"] in {"s8", "s9", "s10", "s11"}
    assert pick["Scenario B real"] == "fit|b27d09e3|roll" and pick["flagged real"] == "fit|4e73d30d|pitch"
