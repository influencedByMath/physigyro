"""Build the spectrum library of one split half with the frozen gates (D-23; THESIS 7.4.2 E-G, Table 1).

    .venv/Scripts/python experiments/part2/build_library.py --half fit

Only vehicles of the requested half in data/screen/split.csv are processed (no-ID vehicles are not in it:
excluded by D-23). Resumable: each finished log is saved at once (data/spectra/<half>_logs.json and
data/spectra/<half>_native/<log_id>.npz), so a restart skips it. At the end it writes
    <half>_windows.csv     one row per selected window: metadata, FC data, provenance (CC-BY)
    <half>_psd_grid.npz    grid (G,), psd (N, 3, G) on the common 0-780 Hz grid, window_id (N,)
    <half>_psd_native.npz  native f and P of every window (keys "<window_id>|f", "<window_id>|P")
    <half>_funnel.csv      data funnel (THESIS Table 1): logs, then windows
"""
import argparse
import csv
import gzip
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import data_quality as dq          # noqa: E402
import spectra_library as sl       # noqa: E402

TH = dq.FROZEN
OUT = os.path.join(ROOT, "data", "spectra")
RAW = os.path.join(ROOT, "data", "raw_logs")


def _atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def screen_rows():
    with gzip.open(os.path.join(ROOT, "data", "screen", "screen_results.jsonl.gz"), "rt") as fh:
        return {r["log_id"]: r for r in (json.loads(x) for x in fh if x.strip())}


def windows_table(recs, screen, half):
    rows = []
    for r in recs:
        if not r.get("windows"):
            continue
        s, fc, sc = r["stats"], r["fc"], screen.get(r["log_id"], {})
        for i, w in enumerate(r["windows"]):
            rows.append(dict(
                window_id=f"{r['log_id']}|w{i}", log_id=r["log_id"], vehicle_key=r["vehicle_key"], split=half,
                log_date=sc.get("log_date"), url=sc.get("download_url"), licence=sl.LICENCE,
                sys_hw=fc.get("hw"), version=fc.get("version"), mav_type=sc.get("mav_type"),
                fifo_chip=(fc.get("fifo_chip") or {}).get("name"), control_chip=(fc.get("control_chip") or {}).get("name"),
                fifo_is_control_gyro=fc.get("fifo_is_control_gyro"), is_external=r.get("is_external"),
                fs_nominal=s["fs_nominal"], fs_measured=s["fs_measured"], nperseg=r["nperseg"],
                rotation=r["rotation"], dropout_fraction=s["dropout_fraction"],
                f_ctrl_param=fc.get("f_ctrl"), f_ctrl_logged=(r.get("axis_check") or {}).get("logged_rate_hz"),
                t0=w["t0"], t1=w["t1"], sp_std_dps=w["sp_std_dps"], thr_std=w["thr_std"], thr_mean=w["thr_mean"],
                n_averages=w["n_averages"], rpm_valid=w["rpm_valid"], rpm_tracking=w["rpm_tracking"],
                rpm_rate_hz=w["rpm_rate_hz"], rpm_hz=json.dumps(w["rpm_hz"]), rpm_std_hz=json.dumps(w["rpm_std_hz"]),
                defaults_known=fc.get("defaults_known"), theta_actual=json.dumps(fc.get("actual")),
                theta_default=json.dumps(fc.get("default")), gains_roll=json.dumps(fc.get("gains_roll")),
                gains_pitch=json.dumps(fc.get("gains_pitch")), has_highrate=fc.get("has_highrate"),
                has_sysid=fc.get("has_sysid"), fft_notch=fc.get("fft_notch"),
                log_flags=";".join(r["log_flags"]), window_flags=";".join(w["gates"][1])))
    return pd.DataFrame(rows)


def funnel(recs, n_vehicles):
    F = dq.Funnel()
    F.add("fit-half vehicles (screened log each)", n_vehicles, Counter())
    parsed = [r for r in recs if r["log_rejections"] != ["error"]]
    errs = Counter("not_downloaded" if "No such file" in r.get("error", "") or "FileNotFound" in r.get("error", "")
                   else "error" for r in recs if r["log_rejections"] == ["error"])
    F.add("downloaded and parsed", n_vehicles, errs)
    passed = [r for r in parsed if not r["log_rejections"]]
    F.add("log gates Q4-Q7 (and Q10 topic)", len(parsed), Counter(r["log_rejections"][0] for r in parsed if r["log_rejections"]))
    F.add("at least one good window", len(passed), Counter("no_good_window" for r in passed if not r["windows"]))
    cand = sum(r["n_candidates"] for r in passed)
    reasons = Counter()
    for r in passed:
        reasons.update(r["window_reasons"])
    F.add("windows: gates Q10, Q9, Q13", cand, reasons)
    good = cand - sum(reasons.values())
    sel = sum(len(r["windows"]) for r in passed)
    F.add("windows: overlap and cap Q14 (3 per vehicle)", good, Counter(not_selected=good - sel))
    return F.table()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", choices=["fit"], default="fit")   # test half: only after Part 2 (author)
    a = ap.parse_args()
    half = a.half
    os.makedirs(os.path.join(OUT, f"{half}_native"), exist_ok=True)
    with open(os.path.join(ROOT, "data", "screen", "split.csv"), newline="") as fh:
        rows = list(csv.DictReader(fh))
    logs_json = os.path.join(OUT, f"{half}_logs.json")
    done = {r["log_id"]: r for r in json.load(open(logs_json))} if os.path.exists(logs_json) else {}
    done = {k: v for k, v in done.items() if v["log_rejections"] != ["error"]}     # retry errors on resume
    recs_so_far = list(done.values())

    def sink(rec, arrays):
        np.savez_compressed(os.path.join(OUT, f"{half}_native", rec["log_id"] + ".npz"), **arrays)
        recs_so_far.append(rec)
        _atomic_json(recs_so_far, logs_json)
        print(f"{rec['log_id'][:8]} rej={rec['log_rejections']} windows={len(rec['windows'])} "
              f"{rec.get('error', '')[:120]}", flush=True)

    recs = sl.process_half(rows, half, RAW, TH, done=done, sink=sink)
    _atomic_json(recs, logs_json)
    n_veh = sum(r["split"] == half for r in rows)
    assert len(recs) == n_veh, (len(recs), n_veh)

    df = windows_table(recs, screen_rows(), half)
    df.to_csv(os.path.join(OUT, f"{half}_windows.csv"), index=False)
    grid = sl.common_grid(TH)
    native, psd = {}, []
    for wid in df["window_id"]:
        log_id, wi = wid.split("|")
        z = np.load(os.path.join(OUT, f"{half}_native", log_id + ".npz"))
        f, P = z[f"{log_id}|{wi}|f"], z[f"{log_id}|{wi}|P"]
        native[f"{wid}|f"], native[f"{wid}|P"] = f, P
        psd.append(sl.to_grid(f.astype(float), P.astype(float), grid).T)
    np.savez_compressed(os.path.join(OUT, f"{half}_psd_grid.npz"), grid=grid,
                        psd=np.array(psd, dtype=np.float32), window_id=df["window_id"].to_numpy(str))
    np.savez_compressed(os.path.join(OUT, f"{half}_psd_native.npz"), **native)
    ft = funnel(recs, n_veh)
    ft.to_csv(os.path.join(OUT, f"{half}_funnel.csv"), index=False)
    # consistency checks (a failure here is a landmine: stop and report)
    assert int(ft["n_out"].iloc[-1]) == len(df), "funnel does not end at the number of library windows"
    assert int(ft["n_out"].iloc[-3]) == df["vehicle_key"].nunique(), "funnel log count != usable vehicles"
    assert df.groupby("vehicle_key").size().max() <= TH.windows_per_vehicle
    assert all(not w["gates"][0] for r in recs for w in r["windows"]), "a selected window fails a gate"
    assert np.isfinite(np.array(psd)).all() and (np.array(psd) > 0).all(), "non-positive PSD on the grid"
    print(ft.to_string())
    print(f"usable vehicles: {df['vehicle_key'].nunique()} of {n_veh}; windows {len(df)}")


if __name__ == "__main__":
    main()
