"""Data-quality gates with explicit thresholds and reason codes (THESIS.md Section 7.4.3).

Every threshold is a field of ``Thresholds`` and is passed in by the caller.
``CANDIDATES`` holds the candidate values written in THESIS.md; they are NOT
frozen. The pilot study (fit half only) proposes the final values, the author
approves them, and they are recorded as a decision in Section 2 (D-10) before
any test-half log is processed.

Gates return ``(rejections, flags)``: a rejection removes the log or window
(with its reason code, for the data funnel, Table 1); a flag keeps it and is
stored with it. Gates Q1-Q2 are the index filter (log_screen.is_candidate),
Q3 cannot occur while the screen takes one log per vehicle, Q7 (board
rotation) is applied in noise_data.gyro_stream, Q12 is the chip cross-table
after clustering, Q14 is the per-vehicle window cap, Q15 is provenance.
"""
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from noise_data import airborne_intervals, fifo_dataset


@dataclass(frozen=True)
class Thresholds:
    """All values are candidates, not frozen (decision D-10), until the pilot study is approved."""
    d_max: float               # Q4  largest dropout fraction of a log (0-1)
    l_min_s: float             # Q4  shortest useful gap-free segment [s] (at least the window length)
    f_min: float               # Q5  lower edge of the fixed band [Hz]
    f_max: float               # Q5  upper edge [Hz]; log dropped if 0.8 fs/2 < f_max
    jitter_max: float          # Q6  std/median of the FIFO sample interval field dt
    clip_max: int              # Q9  clipped samples allowed in a window
    window_s: float            # Q10 window length W [s]
    t_edge_s: float            # Q10 time skipped after take-off and before landing [s]
    sp_std_max_dps: float      # Q10 roll/pitch rate-setpoint std [deg/s]
    thr_std_max: float         # Q10 std of the mean motor command in the window (no THESIS candidate)
    resolution_hz: float       # Q13 Welch frequency resolution [Hz] (nperseg = fs / resolution)
    min_averages: int          # Q13 Welch averages per window
    windows_per_vehicle: int   # Q14 cap
    peak_prominence: float     # 7.4.4 step 6, in decades of the log10 PSD
    rpm_rate_min_hz: float     # D-13 logged esc_status rate needed for the tracking notch [Hz]


CANDIDATES = Thresholds(d_max=0.05, l_min_s=10.0, f_min=20.0, f_max=800.0, jitter_max=0.01, clip_max=0, window_s=10.0,
                        t_edge_s=5.0, sp_std_max_dps=15.0, thr_std_max=0.05, resolution_hz=1.5, min_averages=20,
                        windows_per_vehicle=3, peak_prominence=0.5, rpm_rate_min_hz=50.0)


def nperseg_for(fs, resolution_hz):
    """Welch segment length for one fixed frequency resolution at any sample rate (gate Q13)."""
    return int(round(fs / resolution_hz))


def log_stats(ulog, gs):
    """Log-level statistics for gates Q4-Q6 and Q9; no thresholds are applied here."""
    dt = fifo_dataset(ulog)["dt"].astype(float)
    try:
        air = float(sum(b - a for a, b in airborne_intervals(ulog)))
    except Exception:                      # no vehicle_land_detected
        air = np.nan
    return dict(dropout_fraction=float(gs.gap_fraction),
                logger_dropout_s=float(sum(d.duration for d in ulog.dropouts)) / 1000.0,
                fs_nominal=float(gs.fs), fs_measured=float(gs.fs_measured),
                jitter=float(np.std(dt) / np.median(dt)), n_rate_changes=int(np.count_nonzero(np.diff(dt))),
                n_segments=len(gs.segments),
                longest_segment_s=float(max(b - a for a, b in gs.segments) / gs.fs_measured),
                airborne_s=air, clipped_samples=int((np.abs(gs.raw_counts) >= 32000).any(axis=1).sum()))


def log_gates(stats, fc, th: Thresholds):
    """Log-level gates. ``fc`` is fc_data.fc_info(...) or None."""
    rej, flags = [], []
    if stats["dropout_fraction"] > th.d_max:
        rej.append("Q4_dropout")
    if stats["longest_segment_s"] < th.l_min_s:
        rej.append("Q4_no_long_segment")
    if 0.8 * stats["fs_measured"] / 2 < th.f_max:
        rej.append("Q5_rate")
    if stats["jitter"] > th.jitter_max:
        rej.append("Q6_jitter")
    if stats.get("n_rate_changes", 0) > 0:
        flags.append("Q6_rate_change")
    if fc is not None:
        if not fc["fifo_is_control_gyro"]:
            flags.append("Q8_fifo_not_control")
        if not fc["defaults_known"]:
            flags.append("Q11_defaults_unknown")
        if not fc["has_rpm_topic"]:
            flags.append("Q16_no_rpm")
        elif (fc["esc_rate_hz"] or 0.0) < th.rpm_rate_min_hz:
            flags.append("D13_rpm_rate_low")
        if fc["fft_notch"]:
            flags.append("fft_notch")
        if fc["changed_in_flight"]:
            flags.append("changed_in_flight")
    return rej, flags


def window_gates(win, th: Thresholds):
    """Window-level gates; ``win`` has sp_std_dps, thr_std, n_averages, clipped (and optionally rpm_valid)."""
    rej, flags = [], []
    thr = win.get("thr_std")
    thr_known = thr is not None and np.isfinite(thr)
    if win["sp_std_dps"] > th.sp_std_max_dps or (thr_known and thr > th.thr_std_max):
        rej.append("Q10_unsteady")
    if not thr_known:
        flags.append("Q10_no_throttle")
    if win["clipped"] > th.clip_max:
        rej.append("Q9_clipping")
    if win["n_averages"] < th.min_averages:
        rej.append("Q13_few_averages")
    if not win.get("rpm_valid", True):
        flags.append("Q16_no_rpm")
    return rej, flags


class Funnel:
    """Data funnel (THESIS Table 1): items in and out of each stage, with the removal reasons.

    ``reasons`` counts the FIRST rejection reason of every removed item, so the
    counts add up to the number removed."""

    def __init__(self):
        self.rows = []

    def add(self, stage, n_in, reasons: Counter):
        n_removed = int(sum(reasons.values()))
        self.rows.append(dict(stage=stage, n_in=int(n_in), n_removed=n_removed, n_out=int(n_in) - n_removed,
                              reasons=dict(reasons)))

    def table(self):
        return pd.DataFrame(self.rows, columns=["stage", "n_in", "n_removed", "n_out", "reasons"])
