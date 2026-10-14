"""Validation of the filter model and the noise metric on a real log (E10, E11).

Two checks:
1. Time-domain replica: run the PX4 v1.16 gyro filter chain (ESC-RPM notches
   that follow the logged RPM, static notches, gyro LPF2, derivative + PT1) on
   the raw FIFO gyro and compare sample by sample with the logged
   vehicle_angular_velocity (xyz and xyz_derivative). Agreement proves that the
   filter types, their order, the D-term path and the filter rate are right.
2. Spectral check (THESIS 7.4.5): predict the PSD of the filtered gyro and of
   the torque setpoint from the raw PSD with the frequency-domain model, fold
   it to the publication rate, compare in dB per band.
"""
import numpy as np
from scipy.signal import welch

from filters import (derivative_response, esc_notch_frequencies, lowpass2_coeffs, lowpass2_response,
                     notch_coeffs, pt1_alpha)
from loop_model import FilterSettings, Gains, gyro_stack, integrator_response, notch_stack, pt1_response
from noise_data import _t, board_rotation_matrix, fifo_dataset, rotation_enum


class _Biquad:
    """Direct form I, like PX4 NotchFilter / LowPassFilter2p."""

    def __init__(self, b, a):
        self.set(b, a)
        self.x1 = self.x2 = self.y1 = self.y2 = None

    def set(self, b, a):
        self.b0, self.b1, self.b2 = (float(v) for v in b)
        self.a1, self.a2 = (float(v) for v in a)

    def reset(self, x):
        # PX4 NotchFilter::reset / LowPassFilter2p::reset: steady state for constant input
        self.x1 = self.x2 = x
        dc = (self.b0 + self.b1 + self.b2) / (1 + self.a1 + self.a2)
        self.y1 = self.y2 = x * dc

    def run(self, xs):
        if self.x1 is None:
            self.reset(xs[0])
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        out = []
        for x in xs:
            y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, x, y1, y
            out.append(y)
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return out


def replicate_px4_filters(ulog, s: FilterSettings, t0, t1, warmup=2.0, axes=(0, 1),
                          settle=0.05, dterm_kind="pt1", d_after_lpf=True):
    """Run the PX4 filter chain on raw FIFO data in [t0 - warmup, t1).

    At every gap in the logged FIFO stream (logger dropout) the firmware still
    saw the missing samples, so all filter states are reset and the next
    ``settle`` seconds are not used. ``dterm_kind`` ('pt1' = PX4, 'lpf2' =
    wrong alternative) and ``d_after_lpf`` (True = PX4) allow testing
    alternative models. Returns dict(t_sample, rate, accel): one value per
    FIFO message = the last filtered sample, like CalibrateAndPublish.

    Uses the same FIFO instance as noise_data.gyro_stream and rotates the raw
    board axes to body ``axes`` before filtering. PX4 filters the sensor axes and
    rotates afterwards; with identical linear filters on all axes the order
    does not matter."""
    fifo = fifo_dataset(ulog)
    is_ext = bool(fifo["is_external"][0]) if "is_external" in fifo else False
    R = board_rotation_matrix(rotation_enum(ulog, int(fifo["device_id"][0]), is_ext))
    tm = _t(ulog, fifo["timestamp_sample"])
    sel = np.nonzero((tm >= t0 - warmup) & (tm < t1))[0]
    fs = 1e6 / float(np.median(fifo["dt"]))
    esc = ulog.get_dataset("esc_status").data if s.dnf_esc else None
    te = _t(ulog, esc["timestamp"]) if esc is not None else None
    n_esc = int(np.median(esc["esc_count"])) if esc is not None else 0

    lpf = [_Biquad(*lowpass2_coeffs(s.f_gyro, fs)) for _ in axes]
    stat = [[_Biquad(*notch_coeffs(f0, bw, fs)) for f0, bw in s.notches] for _ in axes]
    d_lpf2 = [_Biquad(*lowpass2_coeffs(s.f_dterm, fs)) for _ in axes]
    # notch[axis][esc][harmonic]
    rpm_notch = [[[None] * s.dnf_hmc for _ in range(n_esc)] for _ in axes]
    alpha = float(pt1_alpha(s.f_dterm, fs))
    d_state = [None] * len(axes)
    prev = [None] * len(axes)
    out_t, out_r, out_a, valid = [], [], [], []
    t_reset = -np.inf
    dt_s = 1.0 / fs
    for j, i in enumerate(sel):
        n = int(fifo["samples"][i])
        if j > 0 and abs(tm[i] - tm[sel[j - 1]] - n * dt_s) > 3 * dt_s:   # gap: reset everything
            for ai in range(len(axes)):
                for bq in [lpf[ai], d_lpf2[ai], *stat[ai], *[nf for row in rpm_notch[ai] for nf in row if nf]]:
                    bq.x1 = None
                prev[ai] = None
            t_reset = tm[i]
        if esc is not None:  # latest ESC report at this time (UpdateDynamicNotchEscRpm)
            k = np.searchsorted(te, tm[i], side="right") - 1
            if k >= 0:
                hz = np.abs(np.array([esc[f"esc[{e}].esc_rpm"][k] for e in range(n_esc)], float)) / 60.0
                for ai in range(len(axes)):
                    for e in range(n_esc):
                        fr = esc_notch_frequencies([hz[e]], s.dnf_hmc, s.dnf_bw, s.dnf_min)
                        for h in range(s.dnf_hmc):
                            nf = rpm_notch[ai][e][h]
                            if nf is None:
                                rpm_notch[ai][e][h] = nf = _Biquad(*notch_coeffs(fr[h], s.dnf_bw, fs))
                                nf.f0 = fr[h]
                            elif abs(nf.f0 - fr[h]) > 0.1:
                                # PX4: only b1 = a1 change; force reset if the jump exceeds the bandwidth
                                b, a = notch_coeffs(fr[h], s.dnf_bw, fs)
                                nf.set(b, a)
                                if abs(nf.f0 - fr[h]) > s.dnf_bw:
                                    nf.x1 = None
                                nf.f0 = fr[h]
        r_last, a_last = [], []
        board = np.array([[float(fifo[f"{c}[{j}]"][i]) for j in range(n)] for c in "xyz"]) * float(fifo["scale"][i])
        body = R @ board                                   # (3, n) body-frame samples (gate Q7)
        for ai, ax in enumerate(axes):
            xs = list(body[ax])
            for e in range(n_esc):
                for nf in rpm_notch[ai][e]:
                    if nf is not None:
                        xs = nf.run(xs)
            for nf in stat[ai]:
                xs = nf.run(xs)
            notched = xs
            xs = lpf[ai].run(xs)
            # FilterAngularAcceleration: backward difference of the LPF output, then AlphaFilter
            accs = []
            for x in (xs if d_after_lpf else notched):
                if prev[ai] is None:
                    prev[ai], d_state[ai] = x, 0.0
                accs.append((x - prev[ai]) * fs)
                prev[ai] = x
            if dterm_kind == "pt1":
                for acc in accs:
                    d_state[ai] += alpha * (acc - d_state[ai])
            else:
                d_state[ai] = d_lpf2[ai].run(accs)[-1]
            r_last.append(xs[-1])
            a_last.append(d_state[ai])
        out_t.append(tm[i])
        out_r.append(r_last)
        out_a.append(a_last)
        valid.append(tm[i] - t_reset >= settle)
    t = np.array(out_t)
    keep = (t >= t0) & np.array(valid)
    return dict(t_sample=t[keep], rate=np.array(out_r)[keep], accel=np.array(out_a)[keep])


def compare_replica(ulog, rep, axes=(0, 1)):
    """Match replica and logged vehicle_angular_velocity by timestamp_sample."""
    v = ulog.get_dataset("vehicle_angular_velocity").data
    tv = _t(ulog, v["timestamp_sample"])
    idx = np.searchsorted(tv, rep["t_sample"])
    idx = np.clip(idx, 0, len(tv) - 1)
    ok = np.abs(tv[idx] - rep["t_sample"]) < 1e-5
    res = {}
    for ai, ax in enumerate(axes):
        for key, field in (("rate", f"xyz[{ax}]"), ("accel", f"xyz_derivative[{ax}]")):
            logged = v[field][idx[ok]].astype(float)
            mine = rep[key][ok, ai]
            diff = logged - mine
            offset = np.median(diff)               # calibration offset and bias (constant)
            err = diff - offset
            res[f"{key}_{'xyz'[ax]}"] = dict(n=int(ok.sum()), offset=float(offset),
                                            rms_signal=float(np.std(logged)),
                                            rms_error=float(np.sqrt(np.mean(err**2))),
                                            rel_error=float(np.sqrt(np.mean(err**2)) / np.std(logged)))
    res["_series"] = dict(t=rep["t_sample"][ok], idx=idx[ok])
    return res


def fold_psd(f_fast, S_fast, f_slow_rate, f_out):
    """One-sided PSD after keeping every k-th sample (decimation to f_slow_rate).

    S_slow(f) = sum over k of S_fast(|f + k f_slow_rate|) inside the fast band."""
    fN = f_fast[-1]
    out = np.zeros_like(f_out, dtype=float)
    kmax = int(np.ceil(fN / f_slow_rate)) + 1
    for k in range(-kmax, kmax + 1):
        g = np.abs(f_out + k * f_slow_rate)
        inside = g <= fN
        out[inside] += np.interp(g[inside], f_fast, S_fast)[...]
    return out


def band_errors_db(f, S_pred, S_meas, bands):
    """Error of predicted vs measured band power in dB for each (lo, hi) band."""
    out = []
    for lo, hi in bands:
        m = (f >= lo) & (f < hi)
        out.append(10 * np.log10(np.trapezoid(S_pred[m], f[m]) / np.trapezoid(S_meas[m], f[m])))
    return np.array(out)


def measured_slow_psds(ulog, t0, t1, axis=0, nperseg=512):
    """PSDs of the logged filtered rate, its derivative (D-term input) and the
    torque setpoint in [t0, t1) at their publication rate."""
    v = ulog.get_dataset("vehicle_angular_velocity").data
    q = ulog.get_dataset("vehicle_torque_setpoint").data
    tv, tq = _t(ulog, v["timestamp"]), _t(ulog, q["timestamp"])
    mv, mq = (tv >= t0) & (tv < t1), (tq >= t0) & (tq < t1)
    rate_v = mv.sum() / (t1 - t0)
    f, S_v = welch(v[f"xyz[{axis}]"][mv], fs=rate_v, nperseg=nperseg)
    _, S_d = welch(v[f"xyz_derivative[{axis}]"][mv], fs=rate_v, nperseg=nperseg)
    _, S_q = welch(q[f"xyz[{axis}]"][mq], fs=mq.sum() / (t1 - t0), nperseg=nperseg)
    return dict(f=f, rate=S_v, accel=S_d, torque=S_q, pub_rate=rate_v)


def predicted_slow_psds(f_raw, S_raw, s: FilterSettings, gains: Gains, fs, f_pub, f_out, motor_hz,
                        dterm_kind="pt1", d_after_lpf=True):
    """Model prediction of the same three PSDs: |H|^2 S_raw at the filter rate,
    folded to the publication rate. Alternatives as in replicate_px4_filters."""
    fr = np.maximum(f_raw, 1e-3)
    Fn = notch_stack(fr, s, fs, motor_hz)
    Fg = gyro_stack(fr, s, fs, motor_hz)
    Fd = pt1_response(fr, s.f_dterm, fs) if dterm_kind == "pt1" else lowpass2_response(fr, s.f_dterm, fs)
    Hd = (Fg if d_after_lpf else Fn) * derivative_response(fr, fs) * Fd
    C = Fg * (gains.Kp + gains.Ki * integrator_response(fr, f_pub)) + gains.Kd * Hd
    return {k: fold_psd(f_raw, np.abs(H)**2 * S_raw, f_pub, f_out)
            for k, H in (("rate", Fg), ("accel", Hd), ("torque", C))}
