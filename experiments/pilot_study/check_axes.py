"""Gate Q7 check on the pilot logs: is the board-to-body rotation right for every rotation that occurs?

For each fit-half pilot log, the raw FIFO gyro rotated to body axes (noise_data.gyro_stream) is low-passed
to flight motion (5 Hz, zero phase) and correlated with the firmware's own logged body rate
(vehicle_angular_velocity) as a 3x3 matrix. A correct rotation gives a positive diagonal that is the
largest value in each row; a wrong one swaps or flips axes (stronger form of the check in
experiments/feasibility/pipeline_checks.py, whose 40 Hz smoothing let vibration lower the values).
    .venv/Scripts/python experiments/pilot_study/check_axes.py
Writes axes.json next to this file.
"""
import json
import os
import sys

import numpy as np
from scipy.signal import butter, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import noise_data as nd          # noqa: E402
from run_pilot import pilot_paths  # noqa: E402


def axis_correlation(u, gs, f_motion=5.0):
    """3x3 correlation of FIFO body axes (rows) with the logged body rate (columns), both low-passed to
    flight motion (zero-phase, f_motion Hz). Correct rotation: diagonal positive and largest in each row."""
    v = u.get_dataset("vehicle_angular_velocity").data
    tv = nd._t(u, v["timestamp_sample"] if "timestamp_sample" in v else v["timestamp"])
    a0, b0 = max(gs.segments, key=lambda s: s[1] - s[0])
    seg_t = gs.t[a0:b0]
    m = (tv > seg_t[0] + 1) & (tv < seg_t[-1] - 1)
    if m.sum() < 50:
        return None
    rate = (m.sum() - 1) / (tv[m][-1] - tv[m][0])
    fifo = sosfiltfilt(butter(2, f_motion, fs=gs.fs_measured, output="sos"), gs.gyro[a0:b0], axis=0)
    logged = sosfiltfilt(butter(2, min(f_motion, 0.4 * rate), fs=rate, output="sos"),
                         np.stack([v[f"xyz[{i}]"][m].astype(float) for i in range(3)], 1), axis=0)
    F = np.stack([np.interp(tv[m], seg_t, fifo[:, i]) for i in range(3)], 1)
    C = np.array([[np.corrcoef(F[:, i], logged[:, j])[0, 1] for j in range(3)] for i in range(3)])
    ok = all(np.argmax(np.abs(C[i])) == i and C[i, i] > 0 for i in range(3))
    return dict(matrix=np.round(C, 3).tolist(), diagonal_ok=bool(ok), n=int(m.sum()),
                seconds=round(float(seg_t[-1] - seg_t[0]), 1), logged_rate_hz=round(float(rate), 1))


def main():
    res = []
    for path, key in pilot_paths(type("A", (), {"logs": None})()):
        u = nd.load(path)
        try:
            gs = nd.gyro_stream(u)
            c = axis_correlation(u, gs)
            r = dict(log=os.path.basename(path)[:8], rotation=gs.rotation, external=gs.is_external, check=c)
        except Exception as e:
            r = dict(log=os.path.basename(path)[:8], error=repr(e)[:200])
        res.append(r)
        print(r, flush=True)
    with open(os.path.join(HERE, "axes.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
