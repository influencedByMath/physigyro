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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))
import noise_data as nd          # noqa: E402
from noise_data import axis_correlation  # noqa: E402  (moved to src for Part 2)
from run_pilot import pilot_paths  # noqa: E402


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
