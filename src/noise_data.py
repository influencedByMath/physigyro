"""Raw gyro noise and flight-controller data from a PX4 ULog (THESIS.md Sections 7.4.3-7.4.5).

Feasibility version: implements the gates needed for one log at a time
(Q4 dropouts/gaps, Q5 fixed band, Q7 board rotation, Q8 gyro instance,
Q9 clipping, Q10 steady flight, Q11 actual vs default parameters, Q13 Welch
averages, Q16 RPM validity). Thresholds marked TBD in THESIS.md are function
arguments here; the pilot study will fix them.
"""
from dataclasses import dataclass

import numpy as np
from pyulog import ULog
from scipy.signal import welch

from loop_model import FilterSettings, Gains, Rates

TOPICS = ["sensor_gyro_fifo", "sensor_selection", "vehicle_land_detected", "vehicle_status",
          "vehicle_rates_setpoint", "esc_status", "actuator_motors", "vehicle_angular_velocity",
          "vehicle_torque_setpoint", "battery_status", "vehicle_attitude_setpoint"]

# PX4 src/lib/conversion/rotation.h (v1.16.0): SENS_BOARD_ROT -> (roll, pitch, yaw) in degrees
ROT_LOOKUP = [(0, 0, 0), (0, 0, 45), (0, 0, 90), (0, 0, 135), (0, 0, 180), (0, 0, 225), (0, 0, 270),
              (0, 0, 315), (180, 0, 0), (180, 0, 45), (180, 0, 90), (180, 0, 135), (0, 180, 0),
              (180, 0, 225), (180, 0, 270), (180, 0, 315), (90, 0, 0), (90, 0, 45), (90, 0, 90),
              (90, 0, 135), (270, 0, 0), (270, 0, 45), (270, 0, 90), (270, 0, 135), (0, 90, 0),
              (0, 270, 0), (0, 180, 90), (0, 180, 270), (90, 90, 0), (180, 90, 0), (270, 90, 0),
              (90, 180, 0), (270, 180, 0), (90, 270, 0), (180, 270, 0), (270, 270, 0), (90, 180, 90),
              (90, 0, 270), (90, 68, 293), (0, 315, 0), (90, 315, 0)]


def board_rotation_matrix(rot):
    """Dcm(Euler(roll, pitch, yaw)), ZYX convention as in PX4 get_rot_matrix()."""
    r, p, y = np.radians(ROT_LOOKUP[int(rot)])
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


@dataclass
class GyroStream:
    t: np.ndarray            # sample times [s] from log start
    gyro: np.ndarray         # (n, 3) body-frame rate [rad/s]
    raw_counts: np.ndarray   # (n, 3) int16 counts (for the clipping gate)
    fs: float                # nominal FIFO rate 1e6 / dt [Hz]: the rate the firmware filters assume
    fs_measured: float       # rate from the sample timestamps (IMU clock error, gate Q6)
    device_id: int
    segments: list           # gap-free (i0, i1) index ranges
    gap_fraction: float
    is_external: bool = False
    rotation: int = 0        # rotation enum used (SENS_BOARD_ROT, or CAL_GYROn_ROT for external IMUs)


def load(path, topics=TOPICS):
    return ULog(str(path), topics)


def _t(ulog, ts):
    return (np.asarray(ts, dtype=np.int64) - int(ulog.start_timestamp)) * 1e-6


def fifo_dataset(ulog):
    """The sensor_gyro_fifo instance of the gyro the controller used (gate Q8), else the first one.

    Uses the last sensor_selection.gyro_device_id; check fifo_is_control_gyro
    in the caller when the selection changes during the log."""
    fifos = [d for d in ulog.data_list if d.name == "sensor_gyro_fifo"]
    if not fifos:
        raise ValueError("log has no sensor_gyro_fifo")
    sel_id = None
    try:
        sel_id = int(ulog.get_dataset("sensor_selection").data["gyro_device_id"][-1])
    except Exception:
        pass
    return next((x.data for x in fifos if sel_id is not None and int(x.data["device_id"][0]) == sel_id),
                fifos[0].data)


def rotation_enum(ulog, device_id, is_external):
    """PX4 calibration::Gyroscope: internal IMUs use SENS_BOARD_ROT, external ones
    their own CAL_GYROn_ROT (n = calibration slot whose CAL_GYROn_ID matches).
    The small trims SENS_BOARD_X/Y/Z_OFF are ignored (a fraction of a degree)."""
    p = ulog.initial_parameters
    if is_external:
        for i in range(4):
            if int(p.get(f"CAL_GYRO{i}_ID", 0)) % 2**32 == int(device_id) % 2**32:
                r = int(p.get(f"CAL_GYRO{i}_ROT", -1))
                if r >= 0:
                    return r
    return int(p.get("SENS_BOARD_ROT", 0))


def gyro_stream(ulog, gap_samples=3.0):
    """Rebuild the raw gyro stream from sensor_gyro_fifo (Section 7.4.4).

    The instance that matches sensor_selection.gyro_device_id is used (gate Q8).
    timestamp_sample jitters by up to about +-1 sample between messages, so
    gaps are found per message: a break is where timestamp_sample moves more
    than ``gap_samples`` samples away from the count of samples received, or
    at a logger dropout (gate Q4). Inside a gap-free segment the sample times
    are a straight-line fit (constant IMU clock)."""
    d = fifo_dataset(ulog)
    n = d["samples"].astype(int)
    dt_us = float(np.median(d["dt"]))
    mask = np.arange(32)[None, :] < n[:, None]
    counts = np.stack([np.stack([d[f"{ax}[{i}]"] for i in range(32)], 1)[mask] for ax in "xyz"], 1)
    scale = np.repeat(d["scale"].astype(float), n)
    gyro_board = counts.astype(float) * scale[:, None]
    t_end = _t(ulog, d["timestamp_sample"])
    is_ext = bool(d["is_external"][0]) if "is_external" in d else False
    rot = rotation_enum(ulog, int(d["device_id"][0]), is_ext)
    gyro = gyro_board @ board_rotation_matrix(rot).T          # board -> body (gate Q7)

    # gate Q4: message-level breaks (timestamp jumps and logger dropouts)
    msg_gap = np.diff(t_end) - n[1:] * dt_us * 1e-6
    brk = set(np.nonzero(np.abs(msg_gap) > gap_samples * dt_us * 1e-6)[0] + 1)
    for dr in ulog.dropouts:
        brk.add(int(np.searchsorted(t_end, _t(ulog, dr.timestamp))))
    medges = [0] + sorted(b for b in brk if 0 < b < len(n)) + [len(n)]
    first = np.concatenate([[0], np.cumsum(n)])               # sample index of each message start
    t = np.empty(first[-1])
    segments, periods, weights = [], [], []
    for ma, mb in zip(medges[:-1], medges[1:]):
        a, b = first[ma], first[mb]
        last_idx = first[ma + 1:mb + 1] - 1                     # index of each message's last sample
        if mb - ma >= 2:
            T, c = np.polyfit(last_idx - a, t_end[ma:mb], 1)
            periods.append(T)
            weights.append(b - a)
        else:
            T, c = dt_us * 1e-6, t_end[ma] - (n[ma] - 1) * dt_us * 1e-6
        t[a:b] = c + T * np.arange(b - a)
        segments.append((int(a), int(b)))
    span = t_end[-1] - t_end[0]
    gap = max(0.0, span - first[-1] * dt_us * 1e-6) / span
    fs_meas = 1.0 / np.average(periods, weights=weights) if periods else 1e6 / dt_us
    return GyroStream(t, gyro, counts, 1e6 / dt_us, float(fs_meas), int(d["device_id"][0]), segments, gap,
                      is_ext, rot)


def airborne_intervals(ulog):
    ld = ulog.get_dataset("vehicle_land_detected").data
    t, landed = _t(ulog, ld["timestamp"]), ld["landed"].astype(bool)
    out, start = [], None
    for ti, l in zip(t, landed):
        if ti < 0:      # first message can carry a zero timestamp
            continue
        if not l and start is None:
            start = ti
        if l and start is not None:
            out.append((start, ti))
            start = None
    if start is not None:
        out.append((start, _t(ulog, ulog.last_timestamp)))
    return out


def steady_windows(ulog, W=8.0, t_edge=5.0, max_sp_std_dps=15.0, step=1.0):
    """Gate Q10: in-air windows of length W, t_edge away from take-off/landing,
    with low rate-setpoint activity (roll and pitch setpoint std below the limit).
    Returns a list of (t0, t1, activity) sorted by activity (quietest first)."""
    rs = ulog.get_dataset("vehicle_rates_setpoint").data
    tr = _t(ulog, rs["timestamp"])
    out = []
    for a, b in airborne_intervals(ulog):
        for s in np.arange(a + t_edge, b - t_edge - W + 1e-9, step):
            m = (tr >= s) & (tr < s + W)
            if m.sum() < 10:
                continue
            act = np.degrees(max(rs["roll"][m].std(), rs["pitch"][m].std()))
            if act <= max_sp_std_dps:
                out.append((float(s), float(s + W), float(act)))
    return sorted(out, key=lambda w: w[2])


def window_psd(gs: GyroStream, t0, t1, nperseg=4096, clip_counts=32000):
    """Welch PSD per body axis inside [t0, t1), computed only inside gap-free
    segments (gate Q4) and averaged with weights = number of Welch averages.
    Returns f, psd (F, 3), n_averages, n_clipped (gate Q9)."""
    acc, n_avg = None, 0
    clipped = 0
    for a, b in gs.segments:
        m = np.arange(a, b)
        m = m[(gs.t[m] >= t0) & (gs.t[m] < t1)]
        if len(m) < nperseg:
            continue
        clipped += int((np.abs(gs.raw_counts[m]) >= clip_counts).any(axis=1).sum())
        # physical frequencies: use the measured IMU rate
        f, P = welch(gs.gyro[m], fs=gs.fs_measured, nperseg=nperseg, axis=0)   # Hann, 50 % overlap
        k = 1 + (len(m) - nperseg) // (nperseg // 2)
        acc = P * k if acc is None else acc + P * k
        n_avg += k
    if acc is None:
        raise ValueError("no gap-free segment longer than nperseg in the window")
    return f, acc / n_avg, n_avg, clipped


def esc_rpm(ulog, t0, t1, n_motors=None):
    """Gate Q16: mean and std of each motor's |RPM| in the window; None if invalid.

    PX4 uses abs(esc_rpm) (reversed motors may report negative RPM). Invalid:
    no reports, or any motor at 0 RPM somewhere in the window. ``rate_hz`` is
    the LOGGED report rate: the default logger profile writes esc_status at
    4 Hz while the firmware notches update much faster, so a tracking-notch
    spectrum (decision P1) needs a high logged rate."""
    try:
        e = ulog.get_dataset("esc_status").data
    except Exception:
        return None
    te = _t(ulog, e["timestamp"])
    m = (te >= t0) & (te < t1)
    n = int(n_motors or np.median(e["esc_count"]))
    rpm = np.abs(np.stack([e[f"esc[{i}].esc_rpm"][m] for i in range(n)], 1).astype(float))
    if rpm.size == 0 or (rpm == 0).any():
        return None
    return dict(mean=rpm.mean(0), std=rpm.std(0), rate_hz=m.sum() / (t1 - t0))


def tracking_notch_psd(ulog, gs: GyroStream, t0, t1, bw, n_harmonics, f_min, axes=(0, 1),
                       nperseg=4096, n_motors=None):
    """PSD of the raw gyro AFTER the PX4 ESC-RPM notch bank that follows the logged RPM.

    Why: the motor frequencies move during a window (several Hz), and the
    firmware notches move with them. A fixed notch at the mean RPM applied to
    the window PSD over-predicts the remaining noise near the harmonics by
    4-9 dB (feasibility test 2026-10-10). Filtering only this time-varying part
    in the time domain gives an exact "effective spectrum"; all time-invariant
    parts (LPF, D-term, PID, loop) can then stay in the frequency domain.

    Coefficients change at every ESC report; between reports the filter runs
    block-wise with scipy.signal.lfilter. The state is carried in direct form I
    (via lfiltic), as in PX4 NotchFilter; a jump larger than the bandwidth
    resets the state, as in NotchFilter::setParameters.

    The Welch PSD uses exactly the samples that window_psd uses (same segments,
    same [t0, t1)), so S_eff and the raw PSD are comparable; up to ``warm``
    seconds before t0 inside the same segment only settle the filters.
    Returns f, psd (F, len(axes)), n_averages."""
    from scipy.signal import lfilter, lfiltic
    from filters import esc_notch_frequencies, notch_coeffs
    e = ulog.get_dataset("esc_status").data
    te = _t(ulog, e["timestamp"])
    n = int(n_motors or np.median(e["esc_count"]))
    hz = np.abs(np.stack([e[f"esc[{i}].esc_rpm"] for i in range(n)], 1).astype(float)) / 60.0
    acc, n_avg, f = None, 0, None
    warm = 1.0                                      # seconds of settling before t0 (if inside the segment)
    for a, b in gs.segments:
        idx = np.arange(a, b)
        idx = idx[(gs.t[idx] >= t0 - warm) & (gs.t[idx] < t1)]
        if (gs.t[idx] >= t0).sum() < nperseg:      # same rule as window_psd
            continue
        x = gs.gyro[idx][:, list(axes)].copy()
        t = gs.t[idx]
        k = np.clip(np.searchsorted(te, t, side="right") - 1, 0, len(te) - 1)
        cuts = np.nonzero(np.diff(k))[0] + 1
        blocks = np.split(np.arange(len(t)), cuts)
        for ai in range(x.shape[1]):
            y = x[:, ai]
            for m in range(n):
                for h in range(int(n_harmonics)):
                    out = np.empty_like(y)
                    xp = yp = f_prev = None
                    for blk in blocks:
                        f0 = esc_notch_frequencies([hz[k[blk[0]], m]], n_harmonics, bw, f_min)[h]
                        (b0, b1, b2), (a1, a2) = notch_coeffs(f0, bw, gs.fs)
                        bb, aa = [float(b0), float(b1), float(b2)], [1.0, float(a1), float(a2)]
                        if xp is None or abs(f0 - f_prev) > bw:     # start, or PX4 forced reset
                            xp = [y[blk[0]], y[blk[0]]]
                            yp = [y[blk[0]], y[blk[0]]]          # notch DC gain is 1
                        f_prev = f0
                        zi = lfiltic(bb, aa, yp[::-1], xp[::-1])
                        out[blk], _ = lfilter(bb, aa, y[blk], zi=zi)
                        xi, yi = y[blk], out[blk]
                        xp = [xi[-2] if len(blk) > 1 else xp[1], xi[-1]]
                        yp = [yi[-2] if len(blk) > 1 else yp[1], yi[-1]]
                    y = out
            x[:, ai] = y
        keep = t >= t0
        f, P = welch(x[keep], fs=gs.fs_measured, nperseg=nperseg, axis=0)
        kk = 1 + (keep.sum() - nperseg) // (nperseg // 2)
        acc = P * kk if acc is None else acc + P * kk
        n_avg += kk
    if acc is None:
        raise ValueError("no gap-free segment long enough")
    return f, acc / n_avg, n_avg


FC_PARAM_PREFIXES = ("IMU_GYRO_", "IMU_DGYRO_", "MC_ROLLRATE_", "MC_PITCHRATE_", "MC_YAWRATE_")


def fc_settings(ulog, axis="ROLL"):
    """Gate Q11: actual and default filter settings, gains and loop rates from the log.

    A ULog stores a default only when it differs from the actual value, so a
    missing default means default == actual. Uses setup defaults (type 1).

    Values are the INITIAL parameters. ``changed_in_flight`` lists filter/gain
    parameters changed during the log (then the window needs the value valid at
    its time). ``fft_notch`` = IMU_GYRO_DNF_EN bit 1: the FFT dynamic notches
    are NOT modelled, so such logs must be flagged or excluded."""
    p = ulog.initial_parameters
    try:
        dflt = ulog.get_default_parameters(1)
    except Exception:
        dflt = {}
    both = {**p, **dflt}

    def settings(src):
        notches = tuple((src[f"IMU_GYRO_NF{i}_FRQ"], src.get(f"IMU_GYRO_NF{i}_BW", 20.0))
                        for i in (0, 1) if src.get(f"IMU_GYRO_NF{i}_FRQ", 0) > 0)
        return FilterSettings(f_gyro=src["IMU_GYRO_CUTOFF"], f_dterm=src["IMU_DGYRO_CUTOFF"],
                              notches=notches, dnf_esc=bool(int(src.get("IMU_GYRO_DNF_EN", 0)) & 1),
                              dnf_bw=src.get("IMU_GYRO_DNF_BW", 15.0), dnf_hmc=int(src.get("IMU_GYRO_DNF_HMC", 3)),
                              dnf_min=src.get("IMU_GYRO_DNF_MIN", 25.0))

    pre = f"MC_{axis}RATE_"
    gains = Gains.from_px4(p[pre + "P"], p[pre + "I"], p[pre + "D"], p.get(pre + "K", 1.0))
    changed = sorted({name for _, name, _ in ulog.changed_parameters if name.startswith(FC_PARAM_PREFIXES)})
    return dict(actual=settings(p), default=settings(both), gains=gains,
                fft_notch=bool(int(p.get("IMU_GYRO_DNF_EN", 0)) & 2), changed_in_flight=changed,
                f_ctrl=float(p.get("IMU_GYRO_RATEMAX", 400)),
                version=ulog.msg_info_dict.get("ver_sw_release"),
                hw=ulog.msg_info_dict.get("ver_hw"))


def rates_from(gs: GyroStream, fc: dict):
    return Rates(fs=gs.fs, f_ctrl=fc["f_ctrl"])
