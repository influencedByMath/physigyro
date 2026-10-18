"""Tier 2 identification on real logs with logger dropouts (D-14; Phase 4 evidence pack for A2-A4).

Real torque-setpoint streams have single skipped controller publications (the motor command holds its value)
and logger dropouts (long gaps). The FRF is therefore built from Welch spectra accumulated over gap-free pieces."""
import numpy as np
import pytest
from scipy.signal import csd, lfilter

import sysid


def _closed_loop(n=800 * 120, seed=3, b=400.0, tau=0.04, d=2, Tc=1 / 800):
    """Same simulator as test_pipeline_parts: PI rate loop, motor lag, delay, disturbance, gyro noise."""
    rng = np.random.default_rng(seed)
    r = lfilter([0.05], [1, -0.95], rng.standard_normal(n)) * 3.0
    dist = lfilter([0.02], [1, -0.98], rng.standard_normal(n)) * 40.0
    noise = 0.02 * rng.standard_normal(n)
    a = np.exp(-Tc / tau)
    m = w = integ = 0.0
    u, omega = np.zeros(n), np.zeros(n)
    for k in range(n):
        e = r[k] - (w + noise[k])
        integ += 0.1 * e * Tc
        u[k] = 0.08 * e + integ
        m = a * m + (1 - a) * (u[k - d] if k >= d else 0.0)
        w = w + Tc * (b * m + dist[k])
        omega[k] = w
    return r, u, omega, Tc


# ------------------------------------------------------------- pieces with held single gaps

def test_uniform_pieces_hold_single_skips_and_split_at_dropouts():
    Tc = 1 / 800
    t = np.arange(0, 3.0, Tc)
    keep = np.ones(len(t), bool)
    keep[100] = keep[300] = keep[301] = False          # 1 and 2 skipped publications: held
    keep[1000:1400] = False                            # 0.5 s logger dropout: split
    pieces = sysid.uniform_pieces(t[keep], Tc, max_hold=3)
    assert len(pieces) == 2
    tg, idx = pieces[0]
    assert tg[0] == pytest.approx(0.0) and np.allclose(np.diff(tg), Tc)
    assert len(tg) == 1000                              # 0 .. 999: the skipped samples are filled
    tk = t[keep]
    assert tk[idx[100]] == pytest.approx(t[99])         # held: the previous published value
    assert tk[idx[301]] == pytest.approx(t[299])
    assert pieces[1][0][0] == pytest.approx(t[1400])


def test_uniform_pieces_tolerate_timestamp_jitter():
    Tc = 1 / 800
    rng = np.random.default_rng(0)
    t = np.arange(0, 2.0, Tc) + rng.uniform(-0.04, 0.04, 1600) * Tc   # +-4 % jitter seen in real logs
    pieces = sysid.uniform_pieces(np.sort(t), Tc, max_hold=3)
    assert len(pieces) == 1 and len(pieces[0][0]) >= 1599


# ------------------------------------------------------------- spectra accumulated over pieces

def test_piece_spectra_of_one_piece_equal_scipy_csd():
    r, u, w, Tc = _closed_loop(n=800 * 20)
    S = sysid.piece_spectra([dict(r=r, u=u, w=w)], fs=1 / Tc, nperseg=1024)
    f, Srw = csd(r, w, fs=1 / Tc, nperseg=1024)
    _, Sru = csd(r, u, fs=1 / Tc, nperseg=1024)
    assert np.allclose(S["f"], f) and np.allclose(S["Srw"], Srw) and np.allclose(S["Sru"], Sru)
    assert S["n_seg"] == 1 + (len(r) - 1024) // 512


def test_piece_spectra_weight_pieces_by_segment_count_and_skip_short_pieces():
    r, u, w, Tc = _closed_loop(n=800 * 20)
    a, b = slice(0, 5000), slice(6000, 16000)
    S = sysid.piece_spectra([dict(r=r[a], u=u[a], w=w[a]), dict(r=r[b], u=u[b], w=w[b]),
                             dict(r=r[:500], u=u[:500], w=w[:500])], fs=1 / Tc, nperseg=1024)
    na, nb = 1 + (5000 - 1024) // 512, 1 + (10000 - 1024) // 512
    _, A = csd(r[a], w[a], fs=1 / Tc, nperseg=1024)
    _, B = csd(r[b], w[b], fs=1 / Tc, nperseg=1024)
    assert S["n_seg"] == na + nb                       # the 500-sample piece is shorter than nperseg: skipped
    assert np.allclose(S["Srw"], (na * A + nb * B) / (na + nb))


def test_piecewise_identification_recovers_the_plant_with_dropouts():
    r, u, w, Tc = _closed_loop()
    cuts = [slice(0, 20000), slice(21000, 45000), slice(47000, 96000)]      # two dropouts removed
    S = sysid.piece_spectra([dict(r=r[c], u=u[c], w=w[c]) for c in cuts], fs=1 / Tc, nperseg=1024)
    f, G, coh = sysid.frf_from_spectra(S)
    fit = sysid.fit_frf(f, G, coh, Tc, f_band=(1.0, 20.0), min_coh=0.3)
    assert abs(fit["b"] / 400 - 1) < 0.10 and abs(fit["tau_m"] / 0.04 - 1) < 0.20


def test_identify_still_uses_the_same_fit():
    r, u, w, Tc = _closed_loop(n=800 * 60)
    a = sysid.identify(dict(r=r, u=u, omega=w, Tc=Tc), f_band=(1.0, 20.0), min_coh=0.3)
    S = sysid.piece_spectra([dict(r=r, u=u, w=w)], fs=1 / Tc, nperseg=1024)
    f, G, coh = sysid.frf_from_spectra(S)
    b = sysid.fit_frf(f, G, coh, Tc, f_band=(1.0, 20.0), min_coh=0.3)
    assert (a["b"], a["tau_m"], a["Td"]) == pytest.approx((b["b"], b["tau_m"], b["Td"]), rel=1e-6)   # coherence rounding


# ------------------------------------------------------------- instrument rule (D-14)

@pytest.mark.parametrize("nav,expected", [(10, "rate_sp"), (0, "att_sp"), (1, "att_sp"), (15, "att_sp"),
                                          (2, "both"), (3, "both"), (4, "both"), (14, "both"), (17, "both")])
def test_instrument_follows_the_flight_mode(nav, expected):
    assert sysid.instrument_for_mode(nav) == expected


def test_instruments_agree_gate():
    f = np.linspace(0.5, 20, 40)
    G = 400 / (2j * np.pi * f)
    coh = np.full(40, 0.9)
    ok, db, deg, n = sysid.instruments_agree(f, G, coh, G * 10 ** (0.5 / 20) * np.exp(1j * np.radians(3)), coh)
    assert ok and db == pytest.approx(0.5) and deg == pytest.approx(3.0) and n == 40
    ok, db, deg, n = sysid.instruments_agree(f, G, coh, G * 10 ** (1.5 / 20), coh)
    assert not ok and db == pytest.approx(1.5)
    low = coh.copy()
    low[:35] = 0.1                                       # only 5 common coherent frequencies: not enough
    assert not sysid.instruments_agree(f, G, low, G, coh)[0]


def test_uniform_pieces_follow_a_slow_clock_without_drift():
    # 0.2 % clock offset over 20,000 samples = 40 periods: a grid at the nominal period would slide 40 samples
    Tc = 1 / 800
    t = np.arange(20000) * Tc * 1.002
    keep = np.ones(len(t), bool)
    keep[::97] = False                                  # regular single skips
    keep[0] = True
    tk = t[keep]
    (tg, idx), = sysid.uniform_pieces(tk, Tc, max_hold=3)
    assert len(tg) == 20000
    assert np.max(np.abs(tk[idx] - tg)[keep]) < 0.1 * Tc   # every present sample sits on its own slot


class _GS:
    """Minimal GyroStream stand-in: times, body rates and gap-free segments."""
    def __init__(self, t, gyro, segments):
        self.t, self.gyro, self.segments = t, gyro, segments


def test_body_rate_on_grid_is_zero_phase_and_removes_aliasing_vibration():
    fs = 8000.0
    t = np.arange(0, 6.0, 1 / fs)
    slow = 0.5 * np.sin(2 * np.pi * 5.0 * t)                    # motion: 5 Hz
    vib = 2.0 * np.sin(2 * np.pi * 795.0 * t)                    # vibration: would alias to 5 Hz at 800 Hz
    gs = _GS(t, np.stack([slow + vib, slow, slow], 1), [(0, len(t))])
    tg = np.arange(1.0, 5.0, 1 / 800)
    w = sysid.body_rate_on_grid(gs, 0, tg, f_cut=100.0)
    assert np.max(np.abs(w - 0.5 * np.sin(2 * np.pi * 5.0 * tg))) < 0.01   # no delay, no alias


def test_body_rate_on_grid_is_nan_outside_segments_and_near_their_edges():
    fs = 8000.0
    t = np.r_[np.arange(0, 2.0, 1 / fs), np.arange(3.0, 5.0, 1 / fs)]
    n1 = int(2.0 * fs)
    gs = _GS(t, np.zeros((len(t), 3)), [(0, n1), (n1, len(t))])
    tg = np.array([0.01, 1.0, 1.99, 2.5, 3.5])
    w = sysid.body_rate_on_grid(gs, 1, tg, f_cut=100.0, edge_s=0.05)
    assert np.isnan(w[[0, 2, 3]]).all() and np.isfinite(w[[1, 4]]).all()


def test_instruments_agree_gate_also_checks_phase():
    f = np.linspace(0.5, 20, 40)
    G = 400 / (2j * np.pi * f)
    coh = np.full(40, 0.9)
    ok, db, deg, _ = sysid.instruments_agree(f, G, coh, G * np.exp(1j * np.radians(10)), coh)
    assert not ok and db == pytest.approx(0.0, abs=1e-9) and deg == pytest.approx(10.0)


def _px4_timing(seed, T=150.0, b=400.0, tau=0.03, Td=0.0015, d_comp=0.0003, skip_frac=0.0, jitter=0.0):
    """PX4-like timing (adapted from the independent review, 2026-10-18): FIFO gyro at 8 kHz; the 800 Hz rate
    controller acts on the newest sample (its time = the torque setpoint's timestamp_sample) and its output is applied
    d_comp later as a zero-order hold; the plant adds Td and a motor lag. The chain must identify Td + d_comp."""
    fs, n_dec = 8000.0, 10
    dt, n = 1 / fs, int(T * fs)
    rng = np.random.default_rng(seed)
    nk = n // n_dec
    r_k = lfilter([0.05], [1, -0.95], rng.standard_normal(nk)) * 3.0
    dist = lfilter([0.002], [1, -0.998], rng.standard_normal(n)) * 40.0
    a = np.exp(-dt / tau)
    w = m = integ = u_now = 0.0
    lag = int(round((Td + d_comp) / dt))
    omega, u_fine = np.zeros(n), np.zeros(n)
    u_k, t_k = np.zeros(nk), np.zeros(nk)
    k = 0
    for i in range(n):
        if i % n_dec == 0 and k < nk:
            e = r_k[k] - (w + 0.01 * rng.standard_normal())
            integ += 0.15 * e / 800.0
            u_now = 0.06 * e + integ
            u_k[k], t_k[k] = u_now, i * dt
            k += 1
        if i + lag < n:
            u_fine[i + lag] = u_now
        m = a * m + (1 - a) * u_fine[i]
        w = w + dt * (b * m + dist[i])
        omega[i] = w
    t = np.arange(n) * dt
    gs = _GS(t, np.stack([omega, omega, omega], 1), [(0, n)])
    keep = np.ones(nk, bool)
    if skip_frac:
        keep[rng.random(nk) < skip_frac] = False
        keep[0] = True
    tq = t_k[keep] + jitter * rng.uniform(-1, 1, keep.sum()) / 800.0
    uq, rq = u_k[keep], r_k[keep]
    Tc = float(np.median(np.diff(tq)))
    pieces = []
    for tg, idx in sysid.uniform_pieces(tq, Tc, max_hold=3):
        wg = sysid.body_rate_on_grid(gs, 0, tg, f_cut=100.0)
        ok = np.isfinite(wg)
        pieces.append(dict(r=rq[idx][ok], u=uq[idx][ok], w=wg[ok]))
    S = sysid.piece_spectra(pieces, fs=1 / Tc, nperseg=1024)
    f, G, coh = sysid.frf_from_spectra(S)
    return sysid.fit_frf(f, G, coh, Tc, f_band=(0.5, 20.0), min_coh=0.5), Td + d_comp


@pytest.mark.parametrize("kw", [dict(seed=1), dict(seed=3, skip_frac=0.10, jitter=0.04)])
def test_end_to_end_timing_does_not_shift_the_identified_delay(kw):
    fit, true_td = _px4_timing(**kw)
    assert abs(fit["Td"] - true_td) < 0.0003            # half a controller sample would be 0.625 ms
    assert abs(fit["b"] / 400.0 - 1) < 0.10


def test_body_rate_resampling_has_no_measurable_time_shift():
    fs = 8000.0
    t = np.arange(0, 6.0, 1 / fs)
    slow = 0.5 * np.sin(2 * np.pi * 5.0 * t)
    gs = _GS(t, np.stack([slow + 2.0 * np.sin(2 * np.pi * 795.0 * t), slow, slow], 1), [(0, len(t))])
    tg = np.arange(1.0, 5.0, 1 / 800)
    w = sysid.body_rate_on_grid(gs, 0, tg, f_cut=100.0)
    assert np.max(np.abs(w - 0.5 * np.sin(2 * np.pi * 5.0 * tg))) < 1e-3     # a 64 us shift would already fail
