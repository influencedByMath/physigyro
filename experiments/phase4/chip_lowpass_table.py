"""Full BIC and fit-error table of the per-chip low-pass terms (Phase 4 Task 1 step 4; D-25, D-26 (3)).

    .venv/Scripts/python experiments/phase4/chip_lowpass_table.py

experiments/part2/model_fits.json keeps only the 5 best terms per chip. The ICM chip-delay range (input to A4)
needs every term, so this script refits every (f_c, n) of the Part 2 grid, plus two lower corners (116, 90 Hz)
because the ICM-42686-P optimum sits on the 150 Hz grid edge. Same fit settings as experiments/part2/fit_model.py
(fixed robust spline floor, prominence 0.5, at most 15 free peaks, band 20-780 Hz, motor frequencies from RPM).
The BIC is computed exactly as noise_model.fit_chip_lowpass; the script checks that it reproduces the stored top-5
values. FIT HALF ONLY (no test-half leakage into the priors): it reads data/spectra/fit_* and nothing else.
Writes chip_lowpass_table.json next to this file.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "experiments", "part2"))
import fit_model as fm             # noqa: E402  (same constants as the Part 2 fits)
import noise_model as nm           # noqa: E402
import spectra_library as sl       # noqa: E402

CHIPS = ("ICM42688P", "ICM42686P", "ICM45686", "IIM42652")      # the chips that got a term (BIC gain >= 10)
EXTRA = [(float(fc), float(n)) for fc in (90, 116) for n in (1, 2, 3, 4, 6)]
GRID = list(fm.GRID) + EXTRA


def fit_term(chip, lp, spectra):
    sse, n, k, rms = 0.0, 0, 0, []
    for f, P, motors in spectra:
        r = nm.fit_spectrum(f, P, fm.BAND, floor=fm.FLOOR, motor_hz=motors, prominence=fm.PROMINENCE, lowpass=lp,
                            max_free=fm.MAX_FREE, fix_floor=fm.FIX_FLOOR)
        sse += (r["rms_db"] / 10) ** 2 * r["n_bins"]
        n += r["n_bins"]
        k += r["n_params"]
        rms.append(r["rms_db"])
    k += 0 if lp is None else 2
    return dict(chip=chip, term=None if lp is None else list(lp), bic=float(n * np.log(sse / n) + k * np.log(n)),
                sse=float(sse), n_bins=int(n), k=int(k), rms_db_pooled=float(10 * np.sqrt(sse / n)),
                rms_db_median=float(np.median(rms)))


def main():
    t0 = time.perf_counter()
    spec = os.path.join(ROOT, "data", "spectra")
    df = pd.read_csv(os.path.join(spec, "fit_windows.csv"))
    if set(df.split) != {"fit"}:
        raise RuntimeError("fit_windows.csv contains non-fit rows")
    native = np.load(os.path.join(spec, "fit_psd_native.npz"))
    vs = sl.vehicle_spectra(df, native)
    groups = {c: [(v["f"], v["P"], v["motor_hz"]) for v in vs if v["chip"] == c] for c in CHIPS}
    jobs = [(c, lp) for c in CHIPS for lp in [None] + GRID]
    rows = Parallel(n_jobs=6, verbose=5)(delayed(fit_term)(c, lp, groups[c]) for c, lp in jobs)
    stored = json.load(open(os.path.join(ROOT, "experiments", "part2", "model_fits.json")))["chips"]
    check = {}
    for c in CHIPS:
        mine = {str(None if r["term"] is None else tuple(r["term"])): r["bic"] for r in rows if r["chip"] == c}
        check[c] = max(abs(mine[key] - val) for key, val in stored[c]["bic_best5"])
    out = dict(grid=GRID, extra=EXTRA, chips=list(CHIPS), n_vehicles={c: len(g) for c, g in groups.items()},
               rows=rows, max_abs_diff_vs_stored_top5=check, seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(HERE, "chip_lowpass_table.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("max |BIC - stored top-5 BIC| per chip:", check, "seconds:", out["seconds"])


if __name__ == "__main__":
    main()
