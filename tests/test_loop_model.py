"""E1: loop metrics against closed-loop poles, python-control and hand-derived cases."""
import control
import numpy as np
import pytest

from loop_model import (FilterSettings, Gains, LoopCase, Plant, Rates, design_gains,
                        margins, PX4_V116_MC_DEFAULT)

F = np.geomspace(1e-3, 1e4, 20000)
S = 2j * np.pi * F


def closed_loop_stable(num, den):
    """Exact answer: roots of den + num (continuous time)."""
    return np.all(np.roots(np.polyadd(den, num)).real < 0)


@pytest.mark.parametrize("K,expected", [(3.0, True), (5.9, True), (6.1, False), (10.0, False)])
def test_nyquist_type1_third_order(K, expected):
    # L = K / (s (s+1) (s+2)): stable iff K < 6 (Routh)
    L = K / (S * (S + 1) * (S + 2))
    m = margins(F, L, n_int=1)
    assert bool(m["stable"]) == expected == closed_loop_stable([K], [1, 3, 2, 0])


@pytest.mark.parametrize("a,expected", [(5.0, True), (15.0, True), (30.0, False), (60.0, False)])
def test_nyquist_type2_loop(a, expected):
    # L = K (s + a) / (s^2 (tau s + 1)), tau = 0.05: stable iff a < 1/tau = 20
    K, tau = 50.0, 0.05
    L = K * (S + a) / (S**2 * (tau * S + 1))
    m = margins(F, L, n_int=2)
    assert bool(m["stable"]) == expected == closed_loop_stable([K, K * a], [tau, 1, 0, 0])


def test_nyquist_counts_rhp_poles():
    K = 30.0  # L = K / (s (s+1) (s+2)) with K > 6 has 2 RHP closed-loop poles
    m = margins(F, K / (S * (S + 1) * (S + 2)), n_int=1)
    assert m["n_rhp"] == 2


def test_margins_match_python_control():
    sys = control.tf([2.0], [1, 3, 2, 0])
    gm, pm, wcg, wcp = control.margin(sys)
    m = margins(F, 2.0 / (S * (S + 1) * (S + 2)), n_int=1)
    assert abs(m["PM"] - pm) < 0.05
    assert abs(m["GM"] - 20 * np.log10(gm)) < 0.05
    assert abs(2 * np.pi * m["fc"] - wcp) < 1e-3 * wcp


def test_margins_with_delay_match_python_control_pade():
    # PI on an integrator with lag and delay, delay via a high-order Pade approximation
    Td, tau = 0.004, 0.05
    num_d, den_d = control.pade(Td, 10)
    sys = control.tf([20 * 0.3, 20 * 0.3 * 2], [tau, 1, 0, 0]) * control.tf(num_d, den_d)
    gm, pm, _, _ = control.margin(sys)
    L = 20 * 0.3 * (S + 2) / (S**2 * (tau * S + 1)) * np.exp(-S * Td)
    m = margins(F[F < 2000], L[F < 2000], n_int=2)
    assert abs(m["PM"] - pm) < 0.1
    assert abs(m["GM"] - 20 * np.log10(gm)) < 0.1


def test_design_gains_hits_crossover():
    rates = Rates(fs=6400, f_ctrl=800)
    g, m = design_gains(10.0, r=1.5, plant=Plant(1.0, 0.05, 0.002), rates=rates)
    assert abs(m["fc"] - 10.0) < 0.05
    assert m["stable"] and 30 < m["PM"] < 90


def test_loopcase_reference_and_trends():
    rates = Rates(fs=6400, f_ctrl=800)
    plant = Plant(1.0, 0.05, 0.002)
    g, _ = design_gains(10.0, 1.5, plant, rates)
    f = np.linspace(0, 3200, 2049)
    psd = 1e-6 + 1e-4 * np.exp(-0.5 * ((f - 105) / 4) ** 2)   # floor + one motor peak
    case = LoopCase(plant, g, rates, f, psd)
    ref = case.ref_metrics
    assert ref["N_rel"] == 1.0 and ref["stable"]
    lo = case.evaluate(FilterSettings(f_gyro=25, f_dterm=20))
    hi = case.evaluate(FilterSettings(f_gyro=100, f_dterm=60))
    # less filtering: less delay, more phase margin, more noise
    assert hi["tau_F"] < ref["tau_F"] < lo["tau_F"]
    assert hi["PM"] > ref["PM"] > lo["PM"]
    assert hi["N_rel"] > 1.0 > lo["N_rel"]
    # vectorized grid evaluation equals the single evaluation
    grid = case.evaluate_grid([25, 40, 100], [20, 30, 60], lambda f: np.ones_like(f, dtype=complex))
    assert abs(grid["PM"][1, 1] - ref["PM"]) < 1e-9
    assert abs(grid["N_rel"][2, 2] - hi["N_rel"]) < 1e-9
