"""Optimizer with the D-37 limits: nominal limits on the nominal plant, D-36 worst-case limits on the 8 corners;
coarse search on the fast grid, refinement and every reported metric on the full grid (D-37 (1), (4))."""
from dataclasses import replace

import numpy as np
import pytest

import optimizer as opt
import robust as rb
import tier3 as t3
from loop_model import LoopCase, Plant, Rates, PX4_V116_MC_DEFAULT, design_gains

F = np.linspace(0, 3200, 4097)
PSD = 1e-6 + 1e-3 * np.exp(-0.5 * ((F - 200) / 5) ** 2)
OPTIONS = opt.scenario_a_options([200.0], bws=(10, 30))


def cases(f_ctrl=400.0, fc=8.0, tau=0.02, td=0.001):
    """A drone whose nominal-only mode-B optimum fails the worst case (worst-corner Ms 3.47 > 3)."""
    rates, plant = Rates(6400, f_ctrl), Plant(1.0, tau, td)
    g, _ = design_gains(fc, 1.0, plant, rates, i_ratio=0.03)
    full = LoopCase(plant, g, rates, F, PSD)
    fast = LoopCase(plant, g, rates, F, PSD, n_stab=opt.FAST["n_stab"], noise_df=opt.FAST["noise_df"])
    spec = opt.Robust(t3.WIDTH, t3.WIDTH, t3.DELAY_SAMPLES / f_ctrl, t3.worst_case_limits())
    return full, fast, spec


def passes_all(case, s, spec):
    rm = rb.robust_metrics(case.plant, case.gains, case.rates, s, case.motor_hz, spec.delta_b, spec.delta_tau,
                           spec.delta_d, n_stab=len(case.f_stab))
    return rb.passes(rm, **t3.limits()), rm


def test_limits_and_grids_are_the_decided_values():
    assert opt.LIMITS == dict(PM_min=30.0, GM_min=6.0, Ms_max=2.0)                       # D-30, D-31
    assert opt.FAST == dict(n_stab=500, noise_df=1.0, n_g=24, n_d=18)                    # D-37 (4)
    assert opt.FULL == dict(n_stab=1500, noise_df=0.5, n_g=40, n_d=30)


def test_nominal_only_optimum_of_the_fixture_fails_the_worst_case():
    full, fast, spec = cases()
    r = opt.optimize(full, PX4_V116_MC_DEFAULT, OPTIONS, "B", n_g=16, n_d=12)
    assert r["feasible"] and not passes_all(full, r["settings"], spec)[0]


@pytest.mark.parametrize("mode", ["A", "B"])
def test_robust_optimum_passes_every_plant_on_the_full_grid(mode):
    full, fast, spec = cases()
    r = opt.optimize(full, PX4_V116_MC_DEFAULT, OPTIONS, mode, robust=spec, coarse=fast, n_g=16, n_d=12)
    assert r["feasible"]
    ok, rm = passes_all(full, r["settings"], spec)
    assert ok
    m = full.evaluate(r["settings"])                       # reported metrics = full-grid metrics of the setting
    for k in ("PM", "GM", "Ms", "N_rel", "tau_F"):
        assert r["metrics"][k] == pytest.approx(m[k], rel=1e-12), k
    assert r["robust"]["worst_corners"]["Ms"] == pytest.approx(rm["worst_corners"]["Ms"], rel=1e-12)
    if mode == "A":
        assert m["N_rel"] <= 1.0 + 1e-6
    else:
        assert m["PM"] >= full.ref_metrics["PM"] - 1e-6


def test_worst_case_limits_cost_noise_in_the_fixture():
    full, fast, spec = cases()
    nom = opt.optimize(full, PX4_V116_MC_DEFAULT, OPTIONS, "B", coarse=fast, n_g=16, n_d=12)
    rob = opt.optimize(full, PX4_V116_MC_DEFAULT, OPTIONS, "B", robust=spec, coarse=fast, n_g=16, n_d=12)
    assert nom["feasible"] and rob["feasible"]
    assert rob["metrics"]["N_rel"] > nom["metrics"]["N_rel"]          # the corner limits bind
    assert not passes_all(full, nom["settings"], spec)[0]


def test_candidates_are_checked_on_the_full_grid():
    """A coarse-grid point that breaks a limit on the full grid is never returned: with a check that refuses every
    candidate the result is infeasible."""
    full, fast, spec = cases()
    r = opt.optimize(full, PX4_V116_MC_DEFAULT, OPTIONS, "A", robust=replace(spec, wc=dict(PM=89.0, Ms=1.01, GM=60.0)),
                     coarse=fast, n_g=16, n_d=12)
    assert not r["feasible"] and r["n_candidates"] > 0


def test_pareto_front_keeps_only_points_the_check_accepts():
    full, fast, spec = cases()
    grid = opt.grid_search(full, PX4_V116_MC_DEFAULT, OPTIONS, n_g=12, n_d=10)
    free = opt.pareto_front(grid, [0.5, 1.0, 2.0])
    assert any(pm is not None for _, pm, _ in free)
    assert all(pm is None for _, pm, _ in opt.pareto_front(grid, [0.5, 1.0, 2.0], accept=lambda o, fg, fd: False))
    chk = opt.robust_accept(full, PX4_V116_MC_DEFAULT, spec)
    rob = opt.pareto_front(grid, [0.5, 1.0, 2.0], accept=chk)
    for (nm, pm0, _), (_, pm1, lab) in zip(free, rob):
        assert pm1 is None or pm1 <= pm0 + 1e-9


@pytest.mark.parametrize("mode", ["A", "B"])
def test_brute_force_on_the_chosen_option_does_not_beat_the_refined_optimum(mode):
    """THESIS 7.3 / D-37 (4): a fine full-grid search of the chosen notch option (worst case checked) must not find a
    clearly better setting than the coarse search + refinement."""
    full, fast, spec = cases()
    r = opt.optimize(full, PX4_V116_MC_DEFAULT, OPTIONS, mode, robust=spec, coarse=fast, n_g=16, n_d=12)
    bf = opt.brute_force(full, PX4_V116_MC_DEFAULT, r["option"], mode, robust=spec, n_g=60, n_d=50)
    assert bf["feasible"]
    ok, _ = passes_all(full, bf["settings"], spec)
    assert ok
    if mode == "A":
        assert bf["metrics"]["PM"] <= r["metrics"]["PM"] + 0.3
    else:
        assert bf["metrics"]["N_rel"] >= r["metrics"]["N_rel"] * (1 - 0.01)
    assert set(bf) >= {"feasible", "settings", "metrics", "f_gyro", "f_dterm", "n_candidates"}


def test_refinement_uses_the_full_grid_version_of_each_option():
    """Options with an effective (tracking-notch) spectrum live on their case's noise grid: the coarse search uses the
    coarse versions, every full-grid check, the refinement and the reported metrics the full versions."""
    from loop_model import FilterSettings
    rates, plant = Rates(6400, 800), Plant(1.0, 0.03, 0.002)
    g, _ = design_gains(6.0, 1.5, plant, rates, i_ratio=0.03)
    psd = 1e-6 + 1e-3 * np.exp(-0.5 * ((F - 105) / 4) ** 2)
    motor = np.array([100.0, 104.0, 107.0, 110.0])
    base = FilterSettings(f_gyro=40.0, f_dterm=30.0, dnf_min=25.0)
    full = LoopCase(plant, g, rates, F, psd, motor_hz=motor)
    fast = LoopCase(plant, g, rates, F, psd, motor_hz=motor, n_stab=opt.FAST["n_stab"], noise_df=opt.FAST["noise_df"])
    track = lambda bw, h, fmin: (F, 0.2 * psd)                      # any effective spectrum will do
    o_fast = opt.scenario_b_options(fast, base, bws=(10, 20), hmcs=(1, 2), tracking=track)
    o_full = opt.scenario_b_options(full, base, bws=(10, 20), hmcs=(1, 2), tracking=track)
    spec = opt.Robust(0.2, 0.2, 1 / 800, t3.worst_case_limits())
    r = opt.optimize_checked(full, base, o_fast, "B", robust=spec, coarse=fast, n_g=12, n_d=10, options_full=o_full)
    assert r["feasible"]
    k = [o.label for o in o_full].index(r["option"].label)
    assert r["option"] is o_full[k]
    m = full.evaluate(r["settings"], S_eff=o_full[k].S_eff)
    assert r["metrics"]["N_rel"] == pytest.approx(m["N_rel"], rel=1e-12)


@pytest.mark.parametrize("mode", ["A", "B"])
def test_a_feasible_reference_bounds_the_optimum(mode):
    """The reference is a candidate and a refinement start: when it meets every limit, Mode A never ends below its PM
    and Mode B never above its noise, however coarse the grid (rq1 runs found up to -0.22 deg / +5 % without this)."""
    rates, plant = Rates(6400, 800), Plant(1.0, 0.03, 0.002)
    g, _ = design_gains(5.0, 1.2, plant, rates, i_ratio=0.03)
    full = LoopCase(plant, g, rates, F, PSD)
    fast = LoopCase(plant, g, rates, F, PSD, n_stab=opt.FAST["n_stab"], noise_df=opt.FAST["noise_df"], f_lo=full.f_lo)
    spec = opt.Robust(0.2, 0.2, 1 / 800, t3.worst_case_limits())
    assert passes_all(full, PX4_V116_MC_DEFAULT, spec)[0]
    options = opt.scenario_a_options([])                                    # 'no notch' only: the reference's option
    r = opt.optimize_checked(full, PX4_V116_MC_DEFAULT, options, mode, robust=spec, coarse=fast, n_g=2, n_d=2,
                             reference=PX4_V116_MC_DEFAULT)
    assert r["feasible"]
    if mode == "A":
        assert r["metrics"]["PM"] >= full.ref_metrics["PM"] - 1e-9
    else:
        assert r["metrics"]["N_rel"] <= 1.0 + 1e-9
    assert passes_all(full, r["settings"], spec)[0]


def test_refinement_itself_keeps_the_worst_case_limits():
    """refine() with a Robust spec must stay inside the worst-case limits on its own (the check after it only
    protects safety, not optimality): in the fixture the unconstrained mode-B refinement runs to worst-corner Ms 3.47."""
    full, fast, spec = cases()
    r = opt.optimize(full, PX4_V116_MC_DEFAULT, OPTIONS, "B", robust=spec, coarse=fast, n_g=16, n_d=12)
    fg0, fd0 = r["grid_point"]
    fg, fd, m, _ = opt.refine(full, PX4_V116_MC_DEFAULT, r["option"], fg0, fd0, "B", pm_ref=full.ref_metrics["PM"],
                              robust=spec)
    s = replace(PX4_V116_MC_DEFAULT, **r["option"].fields, f_gyro=fg, f_dterm=fd)
    assert passes_all(full, s, spec)[0]
