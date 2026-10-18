"""THESIS Table 1: the data funnel from the public log index to the usable vehicles (E2).

    .venv/Scripts/python experiments/part2/table1.py
Every number is recomputed from the committed data files (index snapshot, screen results, split, the per-half
library records and funnels). Writes table1.csv and table1.md next to this file.
"""
import gzip
import json
import os
import sys
from collections import Counter

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import log_screen as ls            # noqa: E402
import split as sp                 # noqa: E402


def main():
    idx = ls.load_index(ls._latest_index_path())
    cands = [e for e in idx if ls.is_candidate(e)]
    groups = {ls._group(e) for e in cands}
    order = ls.sampling_order(cands, ls.SCREEN_SALT, ls.AUDIT_RANDOM_FIFO_LOGS)
    with gzip.open(os.path.join(ROOT, "data", "screen", "screen_results.jsonl.gz"), "rt") as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    screened = {r["log_id"] for r in rows}
    assert screened == {e["log_id"] for e in order[:len(rows)]}, "screen is not an exact prefix of the D-21 order"
    status = Counter(r["status"] for r in rows)
    fifo_rows = [r for r in rows if r.get("fifo") and r.get("status") == "ok"]
    keys = Counter(sp.vehicle_key(r) for r in fifo_rows)
    with_id = sorted(k for k in keys if k.startswith("uuid:"))
    no_id = sorted(k for k in keys if not k.startswith("uuid:"))
    split = pd.read_csv(os.path.join(ROOT, "data", "screen", "split.csv"))
    assert sorted(split.vehicle_key) == with_id, "split.csv differs from the screened FIFO vehicles with a board ID"

    out = [dict(stage="Public logs in the Flight Review index (snapshot 2026-10-14)", n=len(idx), unit="logs"),
           dict(stage="Index filter (Q1, Q2): multirotor, real hardware, >= 60 s, since 2021", n=len(cands), unit="logs"),
           dict(stage="  of which distinct vehicles (vehicle_uuid groups)", n=len(groups), unit="vehicles"),
           dict(stage="Header screen (newest log per vehicle, D-21 order, first 4,327 vehicles)", n=len(rows),
                unit="vehicles"),
           dict(stage=f"  read successfully ({status.get('error', 0)} unreadable: HTTP 403)", n=status.get("ok", 0),
                unit="vehicles"),
           dict(stage="Raw gyro FIFO logged (sensor_gyro_fifo)", n=len(fifo_rows), unit="vehicles (index groups)"),
           dict(stage="  distinct vehicle keys (header board ID, or gate-Q14 fallback key without an ID)", n=len(keys),
                unit="vehicle keys"),
           dict(stage="  FIFO vehicle keys without a board ID: excluded (D-23)", n=len(no_id), unit="vehicle keys"),
           dict(stage="FIFO vehicles with a board ID, split 50/50 by vehicle (D-20)", n=len(with_id), unit="vehicles")]
    for half in ("fit", "test"):
        ft = pd.read_csv(os.path.join(ROOT, "data", "spectra", f"{half}_funnel.csv"))
        w = pd.read_csv(os.path.join(ROOT, "data", "spectra", f"{half}_windows.csv"))
        n_half = int((split.split == half).sum())
        assert int(ft.n_in.iloc[0]) == n_half
        st = {r.stage: r for r in ft.itertuples()}
        logs = st["log gates Q4-Q7 (and Q10 topic)"]
        good = st["at least one good window (newest log)"]
        fbk = next((r for k, r in st.items() if k.startswith("fallback")), None)
        usable = int(w.vehicle_key.nunique())
        assert usable == int(good.n_out) + (int(fbk.n_out) if fbk is not None else 0)
        out += [dict(stage=f"{half.upper()} half: vehicles (one screened log each), downloaded and parsed", n=n_half,
                     unit="vehicles"),
                dict(stage=f"  {half}: pass the log gates (removed: {logs.reasons})", n=int(logs.n_out), unit="vehicles"),
                dict(stage=f"  {half}: at least one steady window passing Q9/Q10/Q13 (removed: {good.reasons})",
                     n=int(good.n_out), unit="vehicles"),
                dict(stage=f"  {half}: next-newest-log fallback (D-24): {int(fbk.n_in)} triggered, recovered",
                     n=int(fbk.n_out), unit="vehicles"),
                dict(stage=f"  {half}: USABLE vehicles", n=usable, unit="vehicles"),
                dict(stage=f"  {half}: windows in the library (<= 3 per vehicle, Q14)", n=len(w), unit="windows")]
    tot_v = sum(pd.read_csv(os.path.join(ROOT, "data", "spectra", f"{h}_windows.csv")).vehicle_key.nunique()
                for h in ("fit", "test"))
    tot_w = sum(len(pd.read_csv(os.path.join(ROOT, "data", "spectra", f"{h}_windows.csv"))) for h in ("fit", "test"))
    out += [dict(stage="TOTAL usable vehicles (fit + test)", n=int(tot_v), unit="vehicles"),
            dict(stage="TOTAL library windows", n=int(tot_w), unit="windows")]
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(HERE, "table1.csv"), index=False)
    with open(os.path.join(HERE, "table1.md"), "w", encoding="utf-8") as fh:
        fh.write("| Stage | Count | Unit |\n|---|---|---|\n")
        for r in out:
            fh.write(f"| {r['stage']} | {r['n']:,} | {r['unit']} |\n")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
