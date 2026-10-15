"""Fit/test spectrum library: window selection, axis gate, grids and band features (THESIS 7.4.4, 7.5; D-23)."""
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

import data_quality as dq
import noise_data as nd
import spectra_library as sl

TH = dq.FROZEN


def cand(t0, sp, n_avg=25, thr=0.01, clipped=0, W=10.0):
    return dict(t0=float(t0), t1=float(t0 + W), sp_std_dps=float(sp), thr_std=thr, n_averages=n_avg, clipped=clipped)


def test_select_windows_gates_before_overlap():
    # the quietest window is gappy (Q13 fails); the overlapping busier one passes and must be selectable
    a, b = cand(0, sp=2.0, n_avg=10), cand(5, sp=3.0, n_avg=25)
    assert [w["t0"] for w in sl.select_windows([a, b], TH)] == [5.0]


def test_select_windows_cap_and_order():
    cs = [cand(0, 5.0), cand(10, 1.0), cand(20, 4.0), cand(30, 2.0), cand(40, 3.0),
          cand(31, 0.5, thr=0.03),          # quietest, but unsteady throttle (Q10): never chosen
          cand(12, 1.5)]                    # overlaps the window at 10 s, which is quieter
    got = sl.select_windows(cs, TH)
    assert [w["t0"] for w in got] == [10.0, 30.0, 40.0]          # sp 1, 2, 3; sorted by time
    assert len(sl.select_windows(cs, replace(TH, windows_per_vehicle=5))) == 5


def test_select_windows_keeps_windows_without_throttle_data():
    assert len(sl.select_windows([cand(0, 1.0, thr=float("nan"))], TH)) == 1   # flagged Q10_no_throttle, kept


@pytest.mark.parametrize("rotation,check,rej,flags", [
    (0, dict(diagonal_ok=True), [], []),
    (6, dict(diagonal_ok=True), [], []),                      # unverified rotation, but the data confirm it
    (0, dict(diagonal_ok=False), ["Q7_axes"], []),
    (6, None, ["Q7_unverified"], []),
    (2, None, [], ["Q7_not_checked"]),
    (0, dict(diagonal_ok=None), [], ["Q7_not_checked"]),   # inconclusive check (too little motion)
    (6, dict(diagonal_ok=None), ["Q7_unverified"], []),
])
def test_axis_gate_rules(rotation, check, rej, flags):
    assert sl.axis_gate(rotation, check) == (rej, flags)


def test_common_grid_and_to_grid():
    grid = sl.common_grid(TH)
    assert grid[0] == 0.0 and grid[-1] == pytest.approx(780.0) and np.allclose(np.diff(grid), 1.5)
    fs, nper = 6387.6, 4258                                        # native resolution 1.50014 Hz
    f = np.arange(nper // 2 + 1) * fs / nper
    P = np.zeros((len(f), 3))
    P[1:] = (f[1:, None] ** -2.0) * np.array([1.0, 2.0, 3.0])
    P[0] = 7.0
    G = sl.to_grid(f, P, grid)
    m = grid >= f[1]
    assert np.allclose(np.log10(G[m]), np.log10(grid[m, None] ** -2.0 * [1.0, 2.0, 3.0]), atol=1e-6)
    assert np.allclose(G[0], 7.0)


def test_band_features_use_native_bins():
    f = np.arange(0, 1000, 1.5)
    P = np.ones((len(f), 2))
    P[f == 30.0, 0] = 101.0
    edges = np.array([20.0, 40.0, 780.0])
    B = sl.band_features(f, P, edges)
    inb = (f >= 20) & (f < 40)
    assert B.shape == (2, 2)
    assert B[0, 0] == pytest.approx(np.log10(P[inb, 0].mean()))
    assert B[0, 1] == pytest.approx(0.0) and B[1, 0] == pytest.approx(0.0)


def test_band_edges_log_spaced():
    e = sl.band_edges(TH)
    assert len(e) == 49 and e[0] == pytest.approx(20.0) and e[-1] == pytest.approx(780.0)
    assert np.allclose(np.diff(np.log(e)), np.log(780 / 20) / 48)


def stream(fs=2000.0, seconds=30.0, segments=None):
    n = int(fs * seconds)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    g = rng.standard_normal((n, 3))
    return nd.GyroStream(t, g, np.zeros((n, 3), dtype=np.int16), fs, fs, 1, segments or [(0, n)], 0.0)


def test_welch_count_matches_window_psd():
    gs = stream(segments=[(0, 9000), (9100, 15000), (15010, 60000)])
    nper = dq.nperseg_for(gs.fs_measured, TH.resolution_hz)
    for t0 in (0.0, 3.0, 4.0, 7.5, 20.0):
        n = nd.welch_count(gs, t0, t0 + 10.0, nper)
        assert n == nd.window_psd(gs, t0, t0 + 10.0, nperseg=nper)[2]
    assert nd.welch_count(stream(segments=[(i, i + 900) for i in range(0, 60000, 1000)]), 0, 10, nper) == 0


def test_process_half_survives_a_bad_log(tmp_path):
    bad = tmp_path / "bad.ulg"
    bad.write_bytes(b"not a log")
    rows = [dict(vehicle_key="uuid:a", split="fit", log_ids="bad")]
    seen = []

    def fake(path, key, th):
        seen.append(key)
        raise ValueError("corrupt")

    recs = sl.process_half(rows, "fit", str(tmp_path), TH, process=fake)
    assert seen == ["uuid:a"] and len(recs) == 1
    assert recs[0]["log_rejections"] == ["error"] and "corrupt" in recs[0]["error"]


def test_process_half_refuses_the_other_half(tmp_path):
    rows = [dict(vehicle_key="uuid:a", split="test", log_ids="x")]
    assert sl.process_half(rows, "fit", str(tmp_path), TH, process=lambda *a: 1 / 0) == []


def test_window_funnel_counts_first_reason():
    cs = [cand(0, 20.0), cand(10, 1.0, n_avg=5), cand(20, 1.0, clipped=3, n_avg=5), cand(30, 1.0)]
    assert sl.window_reasons(cs, TH) == Counter(Q10_unsteady=1, Q13_few_averages=1, Q9_clipping=1)


class _DS:
    def __init__(self, data):
        self.data = data


class FakeULog:
    """Minimal ULog stand-in: logged body rate only (no land detector: the whole log counts as airborne)."""
    start_timestamp = 0

    def __init__(self, t, xyz):
        self._v = {"timestamp": (t * 1e6).astype(np.int64), **{f"xyz[{i}]": xyz[:, i] for i in range(3)}}

    def get_dataset(self, name):
        if name != "vehicle_angular_velocity":
            raise KeyError(name)
        return _DS(self._v)


def motion(t, amp):
    rng = np.random.default_rng(5)
    return np.stack([amp * np.sin(2 * np.pi * f * t + p) for f, p in zip((0.7, 1.1, 0.4), rng.uniform(0, 6, 3))], 1)


@pytest.mark.parametrize("amp,perm,expected", [(0.5, [0, 1, 2], True), (0.5, [1, 0, 2], False),
                                               (0.001, [0, 1, 2], None)])
def test_axis_correlation_pass_fail_inconclusive(amp, perm, expected):
    fs, T = 1000.0, 60.0
    t = np.arange(int(fs * T)) / fs
    body = motion(t, amp)
    rng = np.random.default_rng(2)
    gyro = body[:, perm] + 0.02 * rng.standard_normal(body.shape)      # perm != identity: wrong rotation
    segs = [(0, 20000), (20500, 40000), (40100, len(t))]                # gappy: no single long segment
    gs = nd.GyroStream(t, gyro, np.zeros(gyro.shape, np.int16), fs, fs, 1, segs, 0.01)
    tl = np.arange(0, T, 1 / 200.0)
    u = FakeULog(tl, motion(tl, amp) + 0.002 * rng.standard_normal((len(tl), 3)))
    c = nd.axis_correlation(u, gs)
    assert c["diagonal_ok"] is expected


class FakeFlight:
    """land detector only (no rate setpoint topic)."""
    start_timestamp = 0

    def __init__(self, landed):
        t = np.arange(len(landed)) * 1e6
        self.last_timestamp = int(t[-1])
        self._ld = {"timestamp": t.astype(np.int64), "landed": np.array(landed, dtype=np.uint8)}

    def get_dataset(self, name):
        if name == "vehicle_land_detected":
            return _DS(self._ld)
        raise IndexError("list index out of range")             # what pyulog raises for a missing topic


def test_airborne_windows_empty_when_never_airborne():
    assert nd.airborne_windows(FakeFlight([1] * 100)) == []


def test_airborne_windows_without_rate_setpoint_is_explicit():
    with pytest.raises(nd.MissingTopic):
        nd.airborne_windows(FakeFlight([1] * 5 + [0] * 90 + [1] * 5))
