"""Find public PX4 logs with raw gyro FIFO (THESIS.md Section 7.4.2, steps A-B; decision D-21).

A. ``fetch_index`` saves a snapshot of the Flight Review index (gzipped JSON)
   and ``is_candidate`` applies the index filter.
B. ``sampling_order`` puts vehicles in a random but reproducible order (salted
   SHA-256) and picks the newest qualifying log of each vehicle;
   ``screen_log`` reads only the start of a log with an HTTP range request and
   parses it with ``ulog_header.HeaderParser``; ``run_screen`` does this for
   many logs, politely (at most 4 parallel requests, 1 s pause after each),
   appending one JSON line per log so an interrupted run can resume.

Run from the repository root:
    .venv/Scripts/python src/log_screen.py fetch-index
    .venv/Scripts/python src/log_screen.py screen --target 200
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ulog_header import HeaderParser   # noqa: E402

INDEX_URL = "https://review.px4.io/dbinfo"
USER_AGENT = "px4-noise-thesis/0.1 (academic research; PX4 Flight Review public logs, CC-BY)"
SCREEN_SALT = "px4-noise-screen-2026-10-14"        # sampling order only; the split has its own salt (D-20)
MULTIROTORS = {"Quadrotor", "Hexarotor", "Octorotor"}
INDEX_FIELDS = ("log_id", "log_date", "sys_hw", "ver_sw_release", "vehicle_uuid", "mav_type", "airframe_name",
                "airframe_type", "sys_autostart_id", "duration_s", "flight_modes", "flight_mode_durations", "source",
                "download_url")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SCREEN_JSONL = os.path.join(ROOT, "data", "screen", "screen_results.jsonl")
SNAPSHOT_META = os.path.join(ROOT, "data", "screen", "index_snapshot.json")

# FIFO logs found by the random sample of the 2026-10-06 audit (reused, decision D-21)
AUDIT_RANDOM_FIFO_LOGS = [
    "b27d09e3-4542-4496-a18b-2c3c5317024e", "220b5990-ac65-44d7-afb4-951cf3f23702",
    "2d9b63a2-78b8-4d7f-bde1-6877c514b2f2", "a2571040-29ef-4b1a-b7cf-7776f1a78a1f",
    "bc8a8834-499d-482e-b30c-1cab3b4ba7fd", "4e73d30d-eda1-4b60-8112-51fa6061f16b",
    "48506b01-5111-4f18-9eb7-92623e0574d8", "b4a9719c-e22a-462d-8223-542cb928cc24",
    "9b36955b-5cf1-4b96-bd0d-445ac9231ea0", "7e8f0064-d008-47bf-8752-bebdfb620572",
    "78039e95-cf04-4635-ae75-e4db9c0ef92d", "6ca23b60-8e83-4ffd-a4cd-a83975cc0ebe",
    "dbd58de2-cffa-4327-82ca-90a047595e85", "57aa4fad-35a5-49dd-a2f7-0df8e5641c23",
    "27ee9f0f-4959-4fdf-9494-c59d522646b9", "b0af518f-1627-46f5-8340-5b85d73a2267",
    "2705973a-b217-4f2d-b8df-ae2320a21375", "215afd8c-c39d-4790-86d5-1416894a3474",
    "a17352c6-2805-4c2b-a1de-9d051179fc49", "276f3593-7c01-422e-8d4e-eaeb17b2c072",
    "fcdef3a8-df10-40ed-a6f0-85883f155983", "e89f6a31-e041-49c1-901d-7792a65748d3",
    "bab9d88e-84dc-4847-9d81-15e8a325ec4f"]


# ----------------------------------------------------------------- step A: index

def fetch_index(dest_dir=os.path.join(ROOT, "data", "index"), meta_path=SNAPSHOT_META):
    """Download the Flight Review index once; store it gzipped with its provenance."""
    req = urllib.request.Request(INDEX_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=600) as r:
        raw, url = r.read(), r.geturl()
    if raw[:2] != b"\x1f\x8b":
        raw = gzip.compress(raw)
    date = time.strftime("%Y-%m-%d")
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, f"dbinfo_{date}.json.gz")
    with open(path, "wb") as fh:
        fh.write(raw)
    meta = dict(date=date, url=url, sha256=hashlib.sha256(raw).hexdigest(), n_entries=len(load_index(path)),
                file=os.path.relpath(path, ROOT).replace(os.sep, "/"))
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def load_index(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return json.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)


def is_candidate(entry):
    """Step A: real multirotor hardware, at least 60 s, from 2021 on."""
    hw = (entry.get("sys_hw") or "").upper()
    return (entry.get("mav_type") in MULTIROTORS and "SITL" not in hw and "SIM" not in hw
            and entry.get("source") != "CI" and (entry.get("duration_s") or 0) >= 60
            and (entry.get("log_date") or "") >= "2021-01-01")


def clean_uuid(raw):
    """Vehicle ID from vehicle_uuid / sys_uuid (= flight-controller board ID), or None when it is
    empty, a placeholder ("REDACTED") or a board prefix followed only by zeros."""
    u = (raw or "").replace("\x00", "").strip()
    if not u or "REDACTED" in u.upper() or set(u[4:]) <= {"0"}:
        return None
    return u


# ----------------------------------------------------------------- step B: sampling and screen

def _group(entry):
    u = clean_uuid(entry.get("vehicle_uuid"))
    return "uuid:" + u if u else "log:" + entry["log_id"]


def sampling_order(candidates, salt=SCREEN_SALT, first_log_ids=()):
    """One log per vehicle (the newest; ties: longer, then larger log_id), vehicles in salted-hash
    order. Logs in ``first_log_ids`` come first and replace their vehicle's pick (decision D-21)."""
    by_id = {e["log_id"]: e for e in candidates}
    first = [by_id[i] for i in first_log_ids if i in by_id]
    taken = {_group(e) for e in first}
    best = {}
    for e in candidates:
        g = _group(e)
        if g in taken:
            continue
        key = (e.get("log_date") or "", e.get("duration_s") or 0, e["log_id"])
        if g not in best or key > best[g][0]:
            best[g] = (key, e)
    rest = sorted(best, key=lambda g: hashlib.sha256(f"{salt}|{g}".encode()).hexdigest())
    return first + [best[g][1] for g in rest]


def screen_log(entry, max_bytes=3_000_000, opener=urllib.request.urlopen, chunk=65536, timeout=60):
    """Read the start of one log with a range request; returns index provenance + parser result.

    Network errors give status "error" (unknown), never "no FIFO"."""
    row = {k: entry.get(k) for k in INDEX_FIELDS}
    t = time.perf_counter()
    p, read, err = HeaderParser(), 0, None
    try:
        req = urllib.request.Request(entry["download_url"],
                                     headers={"Range": f"bytes=0-{max_bytes - 1}", "User-Agent": USER_AGENT})
        with opener(req, timeout=timeout) as r:
            total = ((getattr(r, "headers", None) or {}).get("Content-Range") or "").split("/")[-1]
            row["size_bytes"] = int(total) if total.isdigit() else None
            while read < max_bytes and not p.done():
                c = r.read(chunk)
                if not c:
                    break
                read += len(c)
                p.feed(c)
    except Exception as e:                       # timeouts, HTTP errors, dropped connections
        err = repr(e)[:200]
    res = p.result()
    if err is not None:
        res["status"] = "error"
    elif res["status"] == "not_ulog":
        err = "not a ULog file"
    row.update(res, bytes_read=read, seconds=round(time.perf_counter() - t, 2), err=err)
    return row


def _vehicle(row):
    u = clean_uuid(row.get("vehicle_uuid")) or clean_uuid((row.get("info") or {}).get("sys_uuid"))
    return "uuid:" + u if u else "log:" + row["log_id"]


def _paced(entry, pause_s):
    row = screen_log(entry)
    time.sleep(pause_s)
    return row


def run_screen(order, out_jsonl=SCREEN_JSONL, target_fifo_vehicles=200, max_logs=4000, workers=4, pause_s=1.0,
               log=print):
    """Screen logs in ``order`` until ``target_fifo_vehicles`` FIFO vehicles are found (resumable).

    Rows are appended to ``out_jsonl`` in the main thread as requests complete."""
    rows = []
    if os.path.exists(out_jsonl):
        with open(out_jsonl) as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
    done_ids = {r["log_id"] for r in rows}
    fifo = {_vehicle(r) for r in rows if r.get("fifo") and r.get("status") != "error"}
    todo = iter([e for e in order if e["log_id"] not in done_ids][:max_logs])
    n_new, pending = 0, set()
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "a") as fh, ThreadPoolExecutor(workers) as ex:
        def submit_next():
            if len(fifo) >= target_fifo_vehicles:
                return
            e = next(todo, None)
            if e is not None:
                pending.add(ex.submit(_paced, e, pause_s))
        for _ in range(workers):
            submit_next()
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                pending.discard(fut)
                row = fut.result()
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                n_new += 1
                if row.get("fifo") and row.get("status") != "error":
                    fifo.add(_vehicle(row))
                if n_new % 25 == 0:
                    log(f"screened {n_new} new logs, FIFO vehicles {len(fifo)}")
                submit_next()
    return dict(screened_now=n_new, total_rows=len(rows) + n_new, fifo_vehicles=len(fifo))


# ----------------------------------------------------------------- summary (for the author, before the split)

SUMMARY_PARAMS = ("SDLOG_PROFILE", "IMU_GYRO_RATEMAX", "IMU_GYRO_CUTOFF", "IMU_DGYRO_CUTOFF", "IMU_GYRO_DNF_EN",
                  "IMU_GYRO_NF0_FRQ", "IMU_GYRO_FFT_EN", "MC_ROLLRATE_P", "MC_ROLLRATE_I", "MC_ROLLRATE_D",
                  "MC_ROLLRATE_K", "DSHOT_BIDIR_EN", "SENS_BOARD_ROT", "SYS_AUTOSTART")
SUMMARY_TOPICS = ("esc_status", "vehicle_angular_velocity", "vehicle_torque_setpoint", "sensor_gyro_fft")


def wilson_ci(k, n, z=1.96):
    """95 % Wilson score interval of a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return c - h, c + h


def summarize(rows):
    """Counts the author sees before the split: statuses, FIFO hit rate (known rows only), profiles."""
    from collections import Counter
    status = Counter(r.get("status") for r in rows)
    known = [r for r in rows if r.get("status") == "ok"]
    fifo = [r for r in known if r.get("fifo")]
    k, n = len({_vehicle(r) for r in fifo}), len({_vehicle(r) for r in known})
    lo, hi = wilson_ci(k, n)
    par = lambda name: Counter(str((r.get("params") or {}).get(name)) for r in fifo)
    return dict(n_rows=len(rows), status=dict(status), fifo_logs=len(fifo), fifo_vehicles=k,
                hit_rate=k / n if n else 0.0, hit_rate_ci95=(lo, hi),
                fifo_defaults_known_share=(sum(1 for r in fifo if (r.get("n_default_records") or 0) > 0) / len(fifo)
                                           if fifo else 0.0),
                fifo_profiles=dict(par("SDLOG_PROFILE")), fifo_ratemax=dict(par("IMU_GYRO_RATEMAX")),
                fifo_with_topic={t: sum(1 for r in fifo if t in (r.get("topics") or {})) for t in SUMMARY_TOPICS},
                fifo_no_data_in_window=sum(1 for r in fifo if not r.get("fifo_data_msgs")))


def write_summary(rows, csv_path, gz_path):
    """One CSV line per screened log (small, in git) and the complete rows gzipped (record)."""
    import csv
    cols = (["log_id", "log_date", "sys_hw", "ver_sw_release", "vehicle_uuid", "airframe_name", "duration_s", "status",
             "err", "fifo", "fifo_data_msgs", "n_default_records", "bytes_read", "size_bytes", "download_url"]
            + list(SUMMARY_PARAMS) + [f"profile_bit{b}" for b in (3, 4, 8)] + [f"topic_{t}" for t in SUMMARY_TOPICS])
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            p = r.get("params") or {}
            prof = int(p.get("SDLOG_PROFILE") or 0)
            line = {c: r.get(c) for c in cols[:15]}
            line.update({c: p.get(c) for c in SUMMARY_PARAMS})
            line.update({f"profile_bit{b}": bool(prof & (1 << b)) for b in (3, 4, 8)})
            line.update({f"topic_{t}": t in (r.get("topics") or {}) for t in SUMMARY_TOPICS})
            w.writerow(line)
    with gzip.open(gz_path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _latest_index_path():
    with open(SNAPSHOT_META) as fh:
        return os.path.join(ROOT, json.load(fh)["file"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch-index")
    s = sub.add_parser("screen")
    s.add_argument("--target", type=int, default=200)
    s.add_argument("--max-logs", type=int, default=4000)
    s.add_argument("--workers", type=int, default=4)
    sub.add_parser("summary")
    a = ap.parse_args(argv)
    if a.cmd == "fetch-index":
        print(json.dumps(fetch_index(), indent=1))
    elif a.cmd == "screen":
        cands = [e for e in load_index(_latest_index_path()) if is_candidate(e)]
        order = sampling_order(cands, SCREEN_SALT, AUDIT_RANDOM_FIFO_LOGS)
        print(f"{len(cands)} candidate logs, {len(order)} vehicles in the sampling order")
        print(json.dumps(run_screen(order, SCREEN_JSONL, a.target, a.max_logs, min(a.workers, 4))))
    elif a.cmd == "summary":
        with open(SCREEN_JSONL) as fh:
            rows = [json.loads(x) for x in fh if x.strip()]
        out = os.path.dirname(SCREEN_JSONL)
        write_summary(rows, os.path.join(out, "screen_summary.csv"), os.path.join(out, "screen_results.jsonl.gz"))
        print(json.dumps(summarize(rows), indent=1))


if __name__ == "__main__":
    main()
