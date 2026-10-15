"""Vehicle keys and the frozen 50/50 vehicle split (THESIS.md 7.4.2 step C; decisions D-17, D-19, D-20).

The split happens BEFORE full logs are downloaded or any result is looked at.
A vehicle is fit if SHA-256(SALT + "|" + vehicle key) is even, else test.
The vehicles already inspected while the method was designed are always fit
(D-17, D-19). The salt never changes; ``write_split`` refuses to change the
assignment of a vehicle that is already in the split file.

Vehicle key (gate Q14): the flight-controller board ID (``sys_uuid``, equal to
the index's ``vehicle_uuid``); without a valid ID, a fallback key from the
board name, the airframe and the filter and roll/pitch PID parameters.

Run from the repository root (after the screen): .venv/Scripts/python src/split.py
"""
import hashlib
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from log_screen import ROOT, SCREEN_JSONL, clean_uuid   # noqa: E402

SALT = "px4-noise-split-2026-10-14"
FORCED_FIT_UUIDS = ("000600000000383832353233510d002e0040",     # b27d09e3 (D-17)
                    "000600000000383236313533511100370018",     # a2571040 (D-17)
                    "0006000000003332333533335117004a0045",     # b4a9719c (D-17)
                    "000600000000373635363233511500260030")     # 220b5990 (D-19)
FORCED_FIT_KEYS = frozenset("uuid:" + u for u in FORCED_FIT_UUIDS)
FALLBACK_PREFIXES = ("IMU_GYRO_", "IMU_DGYRO_", "MC_ROLLRATE_", "MC_PITCHRATE_")
COLUMNS = ["vehicle_key", "split", "forced", "log_ids", "sys_hw", "n_fifo_logs"]
SPLIT_CSV = os.path.join(ROOT, "data", "screen", "split.csv")


def vehicle_key(row):
    info, params = row.get("info") or {}, row.get("params") or {}
    u = clean_uuid(info.get("sys_uuid")) or clean_uuid(row.get("vehicle_uuid"))
    if u:
        return "uuid:" + u
    hw = info.get("ver_hw") or row.get("sys_hw") or ""
    fil = sorted((k, v) for k, v in params.items() if k.startswith(FALLBACK_PREFIXES))
    text = json.dumps([hw, params.get("SYS_AUTOSTART"), fil])
    return "fb:" + hashlib.sha256(text.encode()).hexdigest()[:16]


def assign_split(key):
    if key in FORCED_FIT_KEYS:
        return "fit"
    return "fit" if int(hashlib.sha256(f"{SALT}|{key}".encode()).hexdigest(), 16) % 2 == 0 else "test"


def _fifo_by_vehicle(rows):
    by = {}
    for r in rows:
        if r.get("fifo") and r.get("status") == "ok":
            by.setdefault(vehicle_key(r), []).append(r)
    return by


def make_split(rows):
    """One line per vehicle with a board ID and at least one FIFO log (screen status ok).

    Vehicles without a board ID (fallback key "fb:") are NOT assigned: the fallback key can give one
    drone two keys when its parameters changed between flights, so their rule waits for the author
    (see no_id_vehicles)."""
    by = {k: v for k, v in _fifo_by_vehicle(rows).items() if k.startswith("uuid:")}
    out = [dict(vehicle_key=k, split=assign_split(k), forced=k in FORCED_FIT_KEYS,
                log_ids=";".join(sorted(r["log_id"] for r in rs)),
                sys_hw=(rs[0].get("info") or {}).get("ver_hw") or rs[0].get("sys_hw"), n_fifo_logs=len(rs))
           for k, rs in sorted(by.items())]
    return pd.DataFrame(out, columns=COLUMNS)


def no_id_vehicles(rows):
    """FIFO vehicles without a board ID (Linux boards such as VOXL2 / Navio2, redacted IDs): listed, not split."""
    out = [dict(vehicle_key=k, log_ids=";".join(sorted(r["log_id"] for r in rs)),
                sys_hw=(rs[0].get("info") or {}).get("ver_hw") or rs[0].get("sys_hw"),
                sys_autostart=(rs[0].get("params") or {}).get("SYS_AUTOSTART"), n_fifo_logs=len(rs))
           for k, rs in sorted(_fifo_by_vehicle(rows).items()) if not k.startswith("uuid:")]
    return pd.DataFrame(out, columns=["vehicle_key", "log_ids", "sys_hw", "sys_autostart", "n_fifo_logs"])


def write_split(df, path=SPLIT_CSV):
    """Write the split; vehicles already in the file keep their assignment (else ValueError)."""
    if os.path.exists(path):
        old = pd.read_csv(path)
        both = old.merge(df, on="vehicle_key", suffixes=("_old", ""))
        changed = both[both["split_old"] != both["split"]]
        if len(changed):
            raise ValueError(f"split would change for {len(changed)} vehicles, e.g. {changed['vehicle_key'].iloc[0]}")
        df = pd.concat([old[~old["vehicle_key"].isin(df["vehicle_key"])], df], ignore_index=True)
    df.sort_values("vehicle_key")[COLUMNS].to_csv(path, index=False)


def pilot_set(split_df, n_random=25):
    """Pilot-study vehicles (fit half only): the forced vehicles, then ``n_random`` other fit vehicles
    in the order of SHA-256(SALT + "|pilot|" + key)."""
    fit = split_df[split_df["split"] == "fit"]
    forced = [k for k in ("uuid:" + u for u in FORCED_FIT_UUIDS) if k in set(fit["vehicle_key"])]
    rest = sorted((k for k in fit["vehicle_key"] if k not in FORCED_FIT_KEYS),
                  key=lambda k: hashlib.sha256(f"{SALT}|pilot|{k}".encode()).hexdigest())[:n_random]
    return fit.set_index("vehicle_key").loc[forced + rest].reset_index()[COLUMNS]


def main():
    with open(SCREEN_JSONL) as fh:
        rows = [json.loads(x) for x in fh if x.strip()]
    df = make_split(rows)
    write_split(df)
    pend = no_id_vehicles(rows)
    pend.to_csv(os.path.join(os.path.dirname(SPLIT_CSV), "split_pending_no_id.csv"), index=False)
    print(f"{len(df)} FIFO vehicles: fit {int((df['split'] == 'fit').sum())}, test {int((df['split'] == 'test').sum())}, "
          f"forced {int(df['forced'].sum())}; {len(pend)} vehicles without board ID pending (no split yet)")


if __name__ == "__main__":
    main()
