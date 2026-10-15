"""Data-quality gates with explicit thresholds (THESIS 7.4.3; nothing frozen before the pilot, D-10)."""
import os
from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

import data_quality as dq

REAL_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "raw_logs", "b27d09e3-4542-4496-a18b-2c3c5317024e.ulg")
needs_log = pytest.mark.skipif(not os.path.exists(REAL_LOG), reason="pilot log not downloaded")

GOOD_STATS = dict(dropout_fraction=0.001, fs_measured=6387.6, jitter=0.0, n_rate_changes=0, longest_segment_s=60.0)
GOOD_FC = dict(fifo_is_control_gyro=True, defaults_known=True, has_rpm_topic=True, esc_rate_hz=200.0, fft_notch=False,
               changed_in_flight=[])


@pytest.mark.parametrize("stats,fc,rejections,flags", [
    (dict(), dict(), [], []),
    (dict(dropout_fraction=0.08), dict(), ["Q4_dropout"], []),
    (dict(longest_segment_s=4.0), dict(), ["Q4_no_long_segment"], []),
    (dict(fs_measured=1000.0), dict(), ["Q5_rate"], []),                 # 0.8 x 500 < 800
    (dict(fs_measured=2000.0), dict(), [], []),                          # 0.8 x 1000 >= 800
    (dict(jitter=0.02), dict(), ["Q6_jitter"], []),
    (dict(n_rate_changes=2), dict(), [], ["Q6_rate_change"]),
    (dict(), dict(defaults_known=False), [], ["Q11_defaults_unknown"]),
    (dict(), dict(fifo_is_control_gyro=False), [], ["Q8_fifo_not_control"]),
    (dict(), dict(has_rpm_topic=False, esc_rate_hz=None), [], ["Q16_no_rpm"]),
    (dict(), dict(esc_rate_hz=4.0), [], ["D13_rpm_rate_low"]),
    (dict(), dict(fft_notch=True, changed_in_flight=["IMU_GYRO_CUTOFF"]), [], ["fft_notch", "changed_in_flight"]),
])
def test_log_gates_reason_codes(stats, fc, rejections, flags):
    rej, fl = dq.log_gates({**GOOD_STATS, **stats}, {**GOOD_FC, **fc}, dq.CANDIDATES)
    assert rej == rejections and fl == flags


def test_log_gates_use_given_thresholds():
    stats = {**GOOD_STATS, "dropout_fraction": 0.08}
    assert dq.log_gates(stats, GOOD_FC, replace(dq.CANDIDATES, d_max=0.10))[0] == []
    assert dq.log_gates(stats, GOOD_FC, replace(dq.CANDIDATES, d_max=0.05))[0] == ["Q4_dropout"]


@pytest.mark.parametrize("win,rejections", [
    (dict(sp_std_dps=5.0, thr_std=0.01, n_averages=25, clipped=0), []),
    (dict(sp_std_dps=40.0, thr_std=0.01, n_averages=25, clipped=0), ["Q10_unsteady"]),
    (dict(sp_std_dps=5.0, thr_std=0.20, n_averages=25, clipped=0), ["Q10_unsteady"]),
    (dict(sp_std_dps=5.0, thr_std=0.01, n_averages=25, clipped=3), ["Q9_clipping"]),
    (dict(sp_std_dps=5.0, thr_std=0.01, n_averages=6, clipped=0), ["Q13_few_averages"]),
])
def test_window_gates_reason_codes(win, rejections):
    assert dq.window_gates(win, dq.CANDIDATES)[0] == rejections


def test_nperseg_for():
    assert dq.nperseg_for(6400.0, 1.5) == 4267
    assert dq.nperseg_for(2000.0, 1.5) == 1333


def test_funnel_counts():
    f = dq.Funnel()
    f.add("logs screened with FIFO", 100, Counter())
    f.add("log gates", 100, Counter(Q4_dropout=20, Q5_rate=5))
    t = f.table()
    assert list(t["n_in"]) == [100, 100] and list(t["n_out"]) == [100, 75]
    assert t.iloc[1]["reasons"] == {"Q4_dropout": 20, "Q5_rate": 5}


@pytest.fixture(scope="module")
def real():
    import noise_data as nd
    u = nd.load(REAL_LOG)
    return u, nd.gyro_stream(u)


@needs_log
def test_log_stats_on_real_log(real):
    u, gs = real
    s = dq.log_stats(u, gs)
    assert s["fs_nominal"] == 6400.0 and abs(s["fs_measured"] - 6387.6) < 1.0     # REPORT Section 3
    assert s["dropout_fraction"] < 0.01 and s["n_rate_changes"] == 0
    assert s["airborne_s"] > 30.0 and s["longest_segment_s"] > 10.0
    assert s["clipped_samples"] == 0


@needs_log
def test_window_activity_matches_steady_windows(real):
    import noise_data as nd
    u, _ = real
    t0, t1, act = nd.steady_windows(u)[0]
    a = nd.window_activity(u, t0, t1)
    assert abs(a["sp_std_dps"] - act) < 1e-9
    assert 0.0 < a["thr_mean"] < 1.0 and np.isfinite(a["thr_std"])
    wins = nd.airborne_windows(u, W=8.0, t_edge=5.0, step=1.0)
    assert any(abs(w["t0"] - t0) < 1e-9 for w in wins)                          # same window grid, no limits applied
    assert max(w["sp_std_dps"] for w in wins) > 15.0                             # manoeuvres are kept for the pilot


def test_no_long_segment_uses_l_min_only():
    """L_min is its own threshold (the pilot may set it below W); it must not be raised to W silently."""
    stats = {**GOOD_STATS, "longest_segment_s": 4.0}
    assert dq.log_gates(stats, GOOD_FC, replace(dq.CANDIDATES, l_min_s=1.0))[0] == []
    assert dq.log_gates(stats, GOOD_FC, replace(dq.CANDIDATES, l_min_s=10.0))[0] == ["Q4_no_long_segment"]


def test_frozen_thresholds_are_d23():
    """THESIS D-23 (2026-10-15): any change of these values needs a new author decision."""
    th = dq.FROZEN
    assert (th.d_max, th.f_min, th.f_max, th.jitter_max, th.clip_max) == (0.20, 20.0, 780.0, 0.01, 0)
    assert (th.window_s, th.t_edge_s, th.sp_std_max_dps, th.thr_std_max) == (10.0, 5.0, 15.0, 0.02)
    assert (th.resolution_hz, th.min_averages, th.windows_per_vehicle, th.rpm_rate_min_hz) == (1.5, 20, 3, 50.0)
    assert th.l_min_s == pytest.approx(1 / 1.5)              # one Welch segment at every sample rate
    assert th.peak_prominence is None                        # not a data gate; set in Part 2 on the fit half
    for fs in (2000.0, 6400.0, 8000.0):
        assert dq.nperseg_for(fs, th.resolution_hz) / fs == pytest.approx(th.l_min_s, rel=1e-3)
