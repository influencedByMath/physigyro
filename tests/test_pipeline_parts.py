"""Checks for the data, validation and optimizer helpers used in the feasibility test."""
import numpy as np
from scipy.signal import welch

import optimizer as opt
from filters import esc_notch_frequencies, notch_bank_response
from loop_model import FilterSettings, LoopCase, Plant, Rates, design_gains, PX4_V116_MC_DEFAULT
from noise_data import GyroStream, board_rotation_matrix, tracking_notch_psd
from validate_real import fold_psd


def test_board_rotation_table():
    np.testing.assert_allclose(board_rotation_matrix(0), np.eye(3), atol=1e-12)
    np.testing.assert_allclose(board_rotation_matrix(4), np.diag([-1, -1, 1]), atol=1e-12)   # yaw 180
    np.testing.assert_allclose(board_rotation_matrix(8), np.diag([1, -1, -1]), atol=1e-12)   # roll 180
    # yaw 90: body x = -board y, body y = board x
    np.testing.assert_allclose(board_rotation_matrix(2) @ [0, 1, 0], [-1, 0, 0], atol=1e-12)


def test_fold_psd_conserves_power():
    f = np.linspace(0, 3200, 4097)
    S = np.full_like(f, 2.0)                       # white noise at 6400 Hz
    f_out = np.linspace(0, 400, 513)
    S_out = fold_psd(f, S, 800.0, f_out)           # keep every 8th sample
    assert np.allclose(S_out[5:-5], 16.0, rtol=1e-6)
    assert abs(np.trapezoid(S_out, f_out) / np.trapezoid(S, f) - 1) < 0.01


class _FakeULog:
    """Just enough of pyulog.ULog for tracking_notch_psd."""

    def __init__(self, t_esc, rpm):
        self.start_timestamp = 0
        data = {"timestamp": (t_esc * 1e6).astype(np.int64), "esc_count": np.full(len(t_esc), rpm.shape[1])}
        for i in range(rpm.shape[1]):
            data[f"esc[{i}].esc_rpm"] = rpm[:, i]
        self._d = type("D", (), {"data": data})()

    def get_dataset(self, name):
        return self._d


def test_tracking_notch_with_constant_rpm_equals_static_notch():
    fs, T = 6400.0, 12.0
    rng = np.random.default_rng(1)
    t = np.arange(int(T * fs)) / fs
    x = rng.standard_normal((len(t), 3))
    gs = GyroStream(t, x, x.astype(np.int16), fs, fs, 0, [(0, len(t))], 0.0)
    t_esc = np.arange(0, T, 1 / 200)
    hz = np.array([100.0, 104.0, 97.0, 110.0])
    ulog = _FakeULog(t_esc, np.tile(hz * 60, (len(t_esc), 1)))
    f, P, _ = tracking_notch_psd(ulog, gs, 2.0, T, bw=10.0, n_harmonics=2, f_min=50.0, axes=(0,))
    _, P0 = welch(x[t >= 2.0, 0], fs=fs, nperseg=4096)
    H = notch_bank_response(f, esc_notch_frequencies(hz, 2, 10.0, 50.0), 10.0, fs)
    band = (f > 60) & (f < 400)
    # away from the notch centres the ratio must match |H|^2 (PSD estimates of the same data)
    ratio = P[band, 0] / (np.abs(H[band])**2 * P0[band])
    deep = np.abs(H[band]) < 0.3
    assert np.allclose(np.median(ratio[~deep]), 1.0, atol=0.02)


def test_optimizer_mode_a_improves_and_is_feasible():
    rates = Rates(6400, 800)
    plant = Plant(1.0, 0.05, 0.002)
    g, _ = design_gains(5.0, 1.5, plant, rates)
    f = np.linspace(0, 3200, 2049)
    psd = 1e-6 + 1e-3 * np.exp(-0.5 * ((f - 200) / 5) ** 2)
    case = LoopCase(plant, g, rates, f, psd, n_stab=500, noise_df=1.0)
    options = opt.scenario_a_options([200.0], bws=(10, 30))
    r = opt.optimize(case, PX4_V116_MC_DEFAULT, options, mode="A", n_g=16, n_d=12)
    assert r["feasible"]
    m = r["metrics"]
    assert m["PM"] > case.ref_metrics["PM"] and m["N_rel"] <= 1.0 + 1e-6 and m["Ms"] <= opt.LIMITS["Ms_max"]


# ------------------------------------------------------------ audit 2026-10-14

def test_grid_equals_single_evaluation_with_esc_bank_and_tracking_spectrum():
    """evaluate() and evaluate_grid() must agree when the noise path uses S_eff
    (ESC bank inside the spectrum) and the stability path the mean-RPM bank."""
    from dataclasses import replace
    rates = Rates(6400, 800)
    plant = Plant(1.0, 0.05, 0.002)
    g, _ = design_gains(6.0, 1.5, plant, rates)
    f = np.linspace(0, 3200, 4097)
    psd = 1e-6 + 1e-3 * np.exp(-0.5 * ((f - 105) / 4) ** 2)
    motor = np.array([100.0, 104.0, 107.0, 110.0])
    case = LoopCase(plant, g, rates, f, psd, motor_hz=motor)
    s = FilterSettings(f_gyro=60.0, f_dterm=45.0, notches=((210.0, 20.0),), dnf_esc=True, dnf_bw=10.0, dnf_hmc=2)
    S_eff = case.psd_on_noise_grid(f, 0.3 * psd)           # any spectrum will do for the consistency check
    single = case.evaluate(s, S_eff=S_eff)
    sn = replace(s, dnf_esc=False)
    from loop_model import notch_stack
    grid = case.evaluate_grid([60.0], [45.0], lambda ff: notch_stack(ff, s, 6400, motor), S_eff=S_eff,
                              noise_notch_fn=lambda ff: notch_stack(ff, sn, 6400, motor))
    for k in ("PM", "GM", "Ms", "N_rel"):
        assert abs(float(grid[k][0, 0]) - single[k]) < 1e-9 * max(1.0, abs(single[k])), k


def test_tracking_notch_uses_the_same_samples_as_window_psd():
    """A segment shorter than nperseg + 1 s of warm-up must not be dropped
    (it was before 2026-10-14, so S_eff and the raw PSD used different data)."""
    from noise_data import window_psd
    fs = 6400.0
    rng = np.random.default_rng(2)
    t = np.arange(int(6.0 * fs)) / fs
    x = rng.standard_normal((len(t), 3))
    cut = int(4.5 * fs)                                   # second segment: 1.5 s, > nperseg, < nperseg + warm-up
    gs = GyroStream(t, x, x.astype(np.int16), fs, fs, 0, [(0, cut), (cut, len(t))], 0.0)
    t_esc = np.arange(0, 6.0, 1 / 200)
    # notch at 3000 Hz, outside the compared band (PX4 runs any notch at least 0.001 fs = 6.4 Hz wide)
    ulog = _FakeULog(t_esc, np.full((len(t_esc), 4), 180000.0))
    _, P_raw, n_raw, _ = window_psd(gs, 1.0, 6.0)
    f, P_tr, n_tr = tracking_notch_psd(ulog, gs, 1.0, 6.0, bw=5.0, n_harmonics=1, f_min=25.0, axes=(0,))
    assert n_tr == n_raw
    away = (f > 20) & (f < 2800)
    np.testing.assert_allclose(P_tr[away, 0], P_raw[away, 0], rtol=5e-3)


def test_esc_rpm_uses_absolute_rpm_and_rejects_stopped_motors():
    from noise_data import esc_rpm
    t_esc = np.arange(0, 2.0, 1 / 100)
    rpm = np.tile([6000.0, -6200.0, 6100.0, -5900.0], (len(t_esc), 1))      # reversed motors report < 0
    r = esc_rpm(_FakeULog(t_esc, rpm), 0.0, 2.0)
    np.testing.assert_allclose(r["mean"], [6000, 6200, 6100, 5900])
    rpm[50, 2] = 0.0
    assert esc_rpm(_FakeULog(t_esc, rpm), 0.0, 2.0) is None


def test_sysid_instrument_variable_recovers_plant_in_closed_loop():
    """Closed-loop data with measurement noise and a disturbance: the FRF with
    the external setpoint as instrument recovers b and tau_m (decision P2)."""
    from scipy.signal import lfilter
    import sysid
    rng = np.random.default_rng(3)
    Tc, n = 1 / 800, 800 * 120
    b_true, tau_true, d_true = 400.0, 0.04, 2                     # delay of 2 control samples
    r = lfilter([0.05], [1, -0.95], rng.standard_normal(n)) * 3.0  # pilot-like setpoint, < 10 Hz
    dist = lfilter([0.02], [1, -0.98], rng.standard_normal(n)) * 40.0   # angular-acceleration disturbance
    noise = 0.02 * rng.standard_normal(n)
    a = np.exp(-Tc / tau_true)
    m = w = 0.0
    u = np.zeros(n)
    omega = np.zeros(n)
    integ = 0.0
    for k in range(n):
        y = w + noise[k]
        e = r[k] - y
        integ += 0.1 * e * Tc
        u[k] = 0.08 * e + integ
        uk = u[k - d_true] if k >= d_true else 0.0
        m = a * m + (1 - a) * uk                                   # motor lag (exact for constant input)
        w = w + Tc * (b_true * m + dist[k])
        omega[k] = w
    fit = sysid.identify(dict(r=r, u=u, omega=omega, Tc=Tc), f_band=(1.0, 20.0), min_coh=0.3)
    assert abs(fit["b"] / b_true - 1) < 0.10
    assert abs(fit["tau_m"] / tau_true - 1) < 0.20


def test_esc_rpm_is_none_when_no_esc_reports():
    """Found in the pilot (2026-10-15): esc_status logged with esc_count = 0 crashed esc_rpm."""
    from noise_data import esc_rpm
    t_esc = np.arange(0, 2.0, 1 / 100)
    assert esc_rpm(_FakeULog(t_esc, np.zeros((len(t_esc), 0))), 0.0, 2.0) is None
