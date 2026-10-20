"""Task 4: the training set of the ML tuner (THESIS 7.6; D-38, D-40; plan notes/task4_plan.md).

    .venv/Scripts/python -u experiments/phase4/dataset_gen.py run             (the whole run; resumable)
    .venv/Scripts/python experiments/phase4/dataset_gen.py verify             (checks of the merged set)
    .venv/Scripts/python experiments/phase4/dataset_gen.py check-spectra      (regenerate the spectra file, SHA-256)

10,000 cases, all generator cases (D-38): Tier 3 synthetic drones of the D-36 sampler (``tier3.sample_drones``,
seed 20261018) with spectra of the Phase 3 generator fitted on the fit half only, labelled with exactly the D-37
optimizer (``rq1_run._synthetic_work``: Modes A and B, nominal and worst-case limits, fast coarse grid with the
full-grid check and refinement, reference-seeded). The sampler is sequential, so its first 1,000 drones are the RQ1
drones; their labels are reused from ``rq1_synthetic.json`` (code 1e9c561) after a re-solve of 20 random RQ1 cases
with the current code matches to 1e-9 (plan Section 2). No test-half path is opened (D-38 (6)).

Order of the run (plan Section 4): clean working tree; sample all drones; count check; spectra file; reuse check;
label the 9,000 new cases in chunks (the first chunk is the smoke chunk; every chunk is checked for format and
limits, and the source files must not change while the workers import them); merge; verify; meta.

Outputs (T4-2 (a)): ``data/labels/train_v1.jsonl.gz`` (one record per case, the RQ1 record format plus the PID gains
and the origin), ``train_v1_meta.json`` (in git) and ``train_v1_spectra.npz`` (float32 input spectra in record
order; git-ignored, SHA-256 in the meta, regenerated bit-exactly by ``check-spectra``). The partial file
``train_v1.partial.jsonl`` starts with a header (seed, n, git hash, config hash): a restart under another code version
or config refuses to resume.
"""
import gzip
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import optimizer as opt                     # noqa: E402
import priors as pr                         # noqa: E402
import rq1_run as rr                        # noqa: E402
import tier3 as t3                          # noqa: E402

SEED = rr.SEED                              # 20261018, the RQ1 seed (plan Section 6)
N = 10000                                   # D-38 (1)
N_RQ1 = rr.N_SYNTH                          # the first 1,000 (D-38 (3))
RQ1_COMMIT = "1e9c561"                      # code of rq1_synthetic.json (Task 3)
CHUNK = 48                                  # plan Section 4
N_JOBS = 6                                  # 8 logical cores, 2 stay free
N_REUSE = 20                                # plan Section 2
REUSE_SEED = SEED + 6                       # picks of the reuse check (the brute-force check used SEED + 5)
REUSE_TOL = 1e-9                            # plan Section 2
COUNT_Z = 5.0                               # count check: a share more than 5 sd from the RQ1 share stops the run

LABELS = os.path.join(ROOT, "data", "labels")
RECORDS = os.path.join(LABELS, "train_v1.jsonl.gz")
META = os.path.join(LABELS, "train_v1_meta.json")
SPECTRA = os.path.join(LABELS, "train_v1_spectra.npz")
PARTIAL = os.path.join(LABELS, "train_v1.partial.jsonl")
RUNLOG = os.path.join(LABELS, "train_v1.runlog.jsonl")

INPUTS = {                                  # every input of the set (all fit half, plan Section 6)
    "model_fits.json": pr.fit_path(ROOT, "experiments", "part2", "model_fits.json"),
    "generator_check.json": pr.fit_path(ROOT, "experiments", "part2", "generator_check.json"),
    "priors_from_logs.json": pr.fit_path(HERE, "priors_from_logs.json"),
    "fit_windows.csv": pr.fit_path(ROOT, "data", "spectra", "fit_windows.csv"),
    "fit_psd_native.npz": pr.fit_path(ROOT, "data", "spectra", "fit_psd_native.npz"),
    "rq1_synthetic.json": pr.fit_path(HERE, "rq1_synthetic.json"),
}

# the label record of rq1_run._synthetic_work (the RQ1 format)
RECORD_KEYS = ("id", "part", "half", "vehicle", "f_c", "tau_m", "Td", "Td_nominal", "delay_step", "r", "fs", "f_ctrl",
               "pm_unfiltered", "chip", "chip_delay", "motor_hz", "flagged", "meta", "scenario", "f_lo", "peaks",
               "n_options", "reference", "reference_robust", "reference_flags", "A", "B", "seconds")
ADDED_KEYS = ("gains", "origin")            # added at the merge: PID gains of the drone; "rq1" or "task4"
DRONE_KEYS = ("f_c", "tau_m", "Td", "Td_nominal", "delay_step", "r", "fs", "f_ctrl", "pm_unfiltered", "chip",
              "chip_delay", "motor_hz", "flagged", "meta")
# what Task 5 needs (THESIS 7.7): drone and loop features, scenario, spectrum source, default metrics, labels A and B
TASK5_FIELDS = ("id", "scenario", "meta.source", "meta.scale", "f_c", "pm_unfiltered", "tau_m", "Td", "Td_nominal",
                "delay_step", "r", "fs", "f_ctrl", "gains.Kp", "gains.Ki", "gains.Kd", "chip", "chip_delay",
                "motor_hz", "flagged", "f_lo", "peaks", "reference.PM", "reference.N_rel", "reference.tau_F",
                "reference_robust.worst_corners", "reference_flags", "A.feasible", "A.settings", "A.metrics",
                "A.robust.worst_corners", "A.option", "B.feasible", "B.settings", "B.metrics",
                "B.robust.worst_corners", "B.option", "seconds", "origin")


class GuardError(RuntimeError):
    """A run guard refused (plan Section 4): stop and report."""


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_hashes():
    return {name: sha256(path) for name, path in INPUTS.items()}


def load_rq1():
    with open(INPUTS["rq1_synthetic.json"]) as fh:
        return json.load(fh)


def sample(n, seed=SEED):
    """(sampler result, chip delays): the first n kept Tier 3 drones with generator spectra, ids syn0000..."""
    return rr.synthetic_drones(seed, n)


# ------------------------------------------------------------------ git guards

def _git(root, *args):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True)
    if r.returncode != 0:
        raise GuardError(f"git {' '.join(args)} failed in {root}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def git_hash(root=ROOT):
    return _git(root, "rev-parse", "HEAD").decode().strip()


def require_clean_tree(root=ROOT):
    """Refuse a dirty working tree (modified, staged or untracked files), so the stored git hash is the code that
    runs (plan Section 4, guard 2)."""
    out = _git(root, "status", "--porcelain", "--untracked-files=all").decode(errors="replace")
    if out.strip():
        raise GuardError("working tree is dirty; commit the code before the run:\n" + out.rstrip())


def source_digest(root=ROOT):
    """SHA-256 over every tracked Python file (path and content). The workers import these files from disk, so a
    change while the run goes (e.g. a mutation check) stops it (plan Section 4, guard 3)."""
    paths = sorted(p for p in _git(root, "ls-files", "-z", "--", "*.py").decode().split("\0") if p)
    h = hashlib.sha256()
    for p in paths:
        h.update(p.encode() + b"\0" + bytes.fromhex(sha256(os.path.join(root, p))))
    return h.hexdigest()


# ------------------------------------------------------------------ resumable chunks

def make_header(seed, n, git, config_hash):
    return dict(kind="header", seed=seed, n=n, git=git, config_hash=config_hash)


def config(delays, inputs):
    """Everything that decides the labels; its hash goes into the partial-file header."""
    return json.loads(json.dumps(dict(
        seed=SEED, n=N, n_rq1=N_RQ1, rq1_commit=RQ1_COMMIT, limits=t3.limits(), width=t3.WIDTH,
        delay_samples=t3.DELAY_SAMPLES, fast=opt.FAST, full=opt.FULL, f_hi=rr.F_HI,
        f_grid=dict(start=float(rr.F_SYN[0]), stop=float(rr.F_SYN[-1]), step=0.5, n=len(rr.F_SYN)),
        chip_delays=delays, inputs=inputs, spectra_dtype="float32",
        reuse=dict(n=N_REUSE, seed=REUSE_SEED, tol=REUSE_TOL))))


def config_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def read_partial(path, header):
    """Finished records of the partial file {id: record}. A new file gets the header line first. A file whose header
    differs (code version, config, seed, n) or has none refuses to resume. A torn last line (a crash during a write)
    is dropped, so those cases are labelled again; any other broken line refuses."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(header) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        return {}
    data = open(path, "rb").read()
    lines = data.decode("utf-8").split("\n")
    try:
        found = json.loads(lines[0])
    except ValueError:
        found = None
    if found != header:
        raise GuardError(f"refusing to resume {path}: its header {found} differs from the run's header {header}")
    if lines[-1] != "":                                   # no newline at the end: a torn last line
        cut = data.rfind(b"\n") + 1
        with open(path, "r+b") as fh:
            fh.truncate(cut)
        print(f"  dropped a torn last line of {path} ({len(data) - cut} bytes); its case is labelled again",
              flush=True)
        lines = lines[:-1] + [""]
    done = {}
    for k, line in enumerate(lines[1:-1], 2):
        try:
            r = json.loads(line)
        except ValueError:
            raise GuardError(f"refusing to resume {path}: line {k} is broken")
        if "id" not in r or r["id"] in done:
            raise GuardError(f"refusing to resume {path}: line {k} has no id or a repeated id")
        done[r["id"]] = r
    return done


def _print(msg):
    print(msg, flush=True)


def label_chunks(items, work, partial, header, chunk=CHUNK, n_jobs=N_JOBS, check=None, guard=None,
                 source_digest=None, log=_print):
    """``work`` on every item (dict with 'id') that the partial file does not hold yet, in chunks run with joblib
    (as rq1_run.run_part). Each finished chunk is appended (and synced) before its checks: ``check(record)`` returns
    a list of problems (format, limits) and ``guard()`` must still return ``source_digest``; either failure stops the
    run with GuardError. Returns {id: record} of every finished item."""
    from joblib import Parallel, delayed
    done = read_partial(partial, header)
    todo = [it for it in items if it["id"] not in done]
    t0 = time.perf_counter()
    for k in range(0, len(todo), chunk):
        block = todo[k:k + chunk]
        res = (Parallel(n_jobs=n_jobs)(delayed(work)(it) for it in block) if n_jobs != 1
               else [work(it) for it in block])
        if [r["id"] for r in res] != [it["id"] for it in block]:
            raise GuardError("a chunk returned records in another order or with other ids")
        with open(partial, "a", encoding="utf-8") as fh:
            for r in res:
                fh.write(json.dumps(r) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        for r in res:
            done[r["id"]] = r
        n_new = k + len(block)
        el = time.perf_counter() - t0
        eta = el / n_new * (len(todo) - n_new)
        log(f"  {sum(it['id'] in done for it in items)}/{len(items)} labelled ({el / n_new:.2f} s per case, "
            f"ETA {eta / 3600:.2f} h) {now()}")
        problems = [f"{r['id']}: {p}" for r in res for p in (check(r) if check else [])]
        if problems:
            raise GuardError("format/limits check failed: " + "; ".join(problems[:20]))
        if guard is not None and guard() != source_digest:
            raise GuardError("source files changed while the run goes (the workers import them from disk)")
    return done


# ------------------------------------------------------------------ checks

def _f(x):
    if x is None:
        return float("nan")
    return float(x)                                       # also "inf" / "-inf" as written by rq1_run._num


def check_record(r):
    """Problems of one label record: the RQ1 format and every feasible optimum inside the D-37 limits (nominal
    limits, worst-case limits on the 8 corners, the mode constraint) and inside the parameter ranges."""
    probs = []
    if set(r) != set(RECORD_KEYS):
        probs.append(f"keys: missing {sorted(set(RECORD_KEYS) - set(r))}, extra {sorted(set(r) - set(RECORD_KEYS))}")
    scen = "B" if r.get("motor_hz") else "A"
    if r.get("scenario") != scen:
        probs.append(f"scenario {r.get('scenario')} but motor_hz {r.get('motor_hz')}")
    fl = r.get("reference_flags") or {}
    if fl.get("nominal_ok") is not True or fl.get("worst_ok") is not (not r.get("flagged")):
        probs.append(f"reference flags {fl} do not match a kept drone with flagged={r.get('flagged')}")
    lim, wc = t3.limits(), t3.worst_case_limits()
    pm_ref = _f((r.get("reference") or {}).get("PM"))
    rg = opt.ranges(r.get("fs") or 0.0)
    for mode in ("A", "B"):
        m = r.get(mode)
        if not isinstance(m, dict) or "feasible" not in m:
            probs.append(f"{mode}: no result")
            continue
        if not m["feasible"]:
            continue
        miss = [k for k in ("settings", "option", "metrics", "robust", "start") if k not in m]
        if miss:
            probs.append(f"{mode}: missing {miss}")
            continue
        met, w, s = m["metrics"], m["robust"]["worst_corners"], m["settings"]
        if met.get("stable") is not True:
            probs.append(f"{mode}: nominal plant not stable")
        if not _f(met.get("PM")) >= lim["PM_min"]:
            probs.append(f"{mode}: nominal PM {met.get('PM')} < {lim['PM_min']}")
        if not _f(met.get("GM")) >= lim["GM_min"]:
            probs.append(f"{mode}: nominal GM {met.get('GM')} < {lim['GM_min']}")
        if not _f(met.get("Ms")) <= lim["Ms_max"]:
            probs.append(f"{mode}: nominal Ms {met.get('Ms')} > {lim['Ms_max']}")
        if w.get("stable") is not True:
            probs.append(f"{mode}: a worst-corner plant is not stable")
        if not _f(w.get("PM")) >= wc["PM"]:
            probs.append(f"{mode}: worst-corner PM {w.get('PM')} < {wc['PM']:.3f}")
        if not _f(w.get("GM")) >= wc["GM"]:
            probs.append(f"{mode}: worst-corner GM {w.get('GM')} < {wc['GM']:.3f}")
        if not _f(w.get("Ms")) <= wc["Ms"]:
            probs.append(f"{mode}: worst-corner Ms {w.get('Ms')} > {wc['Ms']}")
        if mode == "A" and not _f(met.get("N_rel")) <= 1.0:
            probs.append(f"A: N_rel {met.get('N_rel')} > 1")
        if mode == "B" and not _f(met.get("PM")) >= pm_ref:
            probs.append(f"B: PM {met.get('PM')} below the reference PM {pm_ref}")
        for key in ("f_gyro", "f_dterm"):              # refinement bounds hold to 1e-6 in log (optimizer.refine)
            lo, hi = rg[key]
            if not lo * (1 - 1e-5) <= _f(s.get(key)) <= hi * (1 + 1e-5):
                probs.append(f"{mode}: {key} {s.get(key)} outside [{lo}, {hi}]")
    return probs


def count_check(counts, ref, z=COUNT_Z):
    """Shares of the run against the RQ1 shares (in window per draw, kept per in window, Scenario B and flagged per
    kept drone): a share more than z combined binomial sd away is a landmine (plan: 'a count far from the plan')."""
    rows = [("in window", "in_window", "draws"), ("kept", "kept", "in_window"), ("Scenario B", "scenario_b", "kept"),
            ("flagged", "flagged", "kept")]
    probs = []
    for name, num, den in rows:
        p, p0 = counts[num] / counts[den], ref[num] / ref[den]
        sd = math.sqrt(p0 * (1 - p0) * (1 / ref[den] + 1 / counts[den]))
        if abs(p - p0) > z * sd:
            probs.append(f"{name} share {p:.4f} vs RQ1 {p0:.4f} ({abs(p - p0) / sd:.1f} sd)")
    return probs


def compare(a, b, tol=REUSE_TOL, skip=("seconds",), path=""):
    """Differences [(path, a, b)] between two JSON values: numbers beyond ``tol`` (absolute), anything else unequal."""
    def sub(k):
        return f"{path}.{k}" if path else str(k)
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k in skip:
                continue
            if k not in a or k not in b:
                out.append((sub(k), a.get(k), b.get(k)))
            else:
                out += compare(a[k], b[k], tol, skip, sub(k))
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [(path, a, b)]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in compare(x, y, tol, skip, sub(i))]
    if isinstance(a, bool) or isinstance(b, bool):        # bools are ints in Python: compare them exactly
        return [] if (type(a) is type(b) and a == b) else [(path, a, b)]
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isfinite(a) and math.isfinite(b):
            return [] if abs(a - b) <= tol else [(path, a, b)]
        return [] if (a == b or (math.isnan(a) and math.isnan(b))) else [(path, a, b)]
    return [] if a == b else [(path, a, b)]


def _max_abs_diff(diffs):
    m = 0.0
    for _, a, b in diffs:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
                and math.isfinite(a) and math.isfinite(b):
            m = max(m, abs(a - b))
    return m


def reuse_check(items_by_id, rq1_results, work, n=N_REUSE, seed=REUSE_SEED, n_jobs=N_JOBS):
    """Re-solve n random RQ1 cases with the current code; every setting and metric must match to REUSE_TOL
    (everything but the solver seconds). Returns the picked ids, the largest numeric difference and the mismatches."""
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rq1_results), n, replace=False).tolist())
    picked = [rq1_results[i]["id"] for i in idx]
    block = [items_by_id[i] for i in picked]
    if n_jobs != 1:
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=n_jobs)(delayed(work)(it) for it in block)
    else:
        res = [work(it) for it in block]
    mismatches, max_diff = [], 0.0
    for i, new in zip(idx, res):
        old = rq1_results[i]
        new = json.loads(json.dumps(new))                 # the stored (JSON) form, as the RQ1 file
        max_diff = max(max_diff, _max_abs_diff(compare(old, new, tol=0.0)))
        d = compare(old, new)
        if d:
            mismatches.append(dict(id=old["id"], n=len(d), first=[f"{p}: {x!r} vs {y!r}" for p, x, y in d[:5]]))
    return dict(picked=picked, n=n, seed=seed, tol=REUSE_TOL, ok=not mismatches, n_mismatch=len(mismatches),
                max_abs_diff=max_diff, mismatches=mismatches)


# ------------------------------------------------------------------ merge, files, verification

def _drone_problems(d, src):
    probs = [f"drone field {k} differs ({src.get(k)!r} vs {d[k]!r})" for k in DRONE_KEYS if src.get(k) != d[k]]
    if src.get("reference_robust") != rr.robust_of_dict(d["default"]):
        probs.append("the sampler's default metrics differ from reference_robust")
    return probs


def merge(drones, rq1_results, new_by_id):
    """One record per drone, in drone order: the RQ1 label record for the first len(rq1_results) drones, the new
    label for the rest, each with the drone's PID gains and its origin. Every record must belong to its drone
    (drone fields, default metrics); ids must be unique and complete."""
    ids = [d["id"] for d in drones]
    if len(set(ids)) != len(ids):
        raise ValueError("a drone id appears twice")
    n_rq1 = len(rq1_results)
    if [r["id"] for r in rq1_results] != ids[:n_rq1]:
        raise ValueError("the RQ1 records are not the first drones in their order")
    for k in new_by_id:
        if k in ids[:n_rq1]:
            raise ValueError(f"{k} is labelled twice (RQ1 and new)")
        if k not in ids:
            raise ValueError(f"{k} is not a drone of the set")
    missing = [i for i in ids[n_rq1:] if i not in new_by_id]
    if missing:
        raise ValueError(f"{len(missing)} labels missing, e.g. {missing[:5]}")
    out = []
    for i, d in enumerate(drones):
        src = rq1_results[i] if i < n_rq1 else new_by_id[d["id"]]
        if src["id"] != d["id"]:
            raise ValueError(f"record {src['id']} is not in the order of drone {d['id']}")
        probs = _drone_problems(d, src)
        if probs:
            raise ValueError(f"{d['id']}: " + "; ".join(probs))
        rec = dict(src)
        rec["gains"] = {k: float(v) for k, v in d["gains"].items()}
        rec["origin"] = "rq1" if i < n_rq1 else "task4"
        out.append(rec)
    return out


def write_records(path, records):
    """jsonl.gz, one record per line; no time stamp in the gzip header, so the same records give the same bytes."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            for r in records:
                gz.write((json.dumps(r) + "\n").encode("utf-8"))
    os.replace(tmp, path)


def read_records(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def spectra_arrays(drones):
    """Input spectra in record order: f (Hz, the 0.5 Hz grid 20-780 Hz), S (float32, one row per drone), ids."""
    for d in drones:
        if not np.array_equal(d["psd_f"], rr.F_SYN):
            raise ValueError(f"{d['id']}: spectrum not on the generator grid")
    return dict(f=np.array(rr.F_SYN, dtype=np.float64), S=np.stack([d["S"] for d in drones]).astype(np.float32),
                ids=np.array([d["id"] for d in drones]))


def save_spectra(path, arr):
    """Uncompressed npz (NumPy writes a fixed zip time stamp, so the same arrays give the same bytes)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, f=arr["f"], S=arr["S"], ids=arr["ids"])
    os.replace(tmp, path)


def verify(records, rq1_results, spectra_path, spectra_sha, n=N):
    """Problems of the merged set: count and unique ordered ids, RQ1 records unchanged, every optimum inside the
    limits, the spectra file (SHA-256, ids in record order)."""
    probs = []
    ids = [r["id"] for r in records]
    if len(records) != n:
        probs.append(f"{len(records)} records, expected {n}")
    if len(set(ids)) != len(ids):
        probs.append("ids not unique")
    elif ids != [f"syn{i:04d}" for i in range(len(ids))]:
        probs.append("ids not in order syn0000...")
    by_id = {r["id"]: r for r in records}
    for q in rq1_results:
        r = by_id.get(q["id"])
        if r is None or {k: v for k, v in r.items() if k not in ADDED_KEYS} != q or r.get("origin") != "rq1":
            probs.append(f"RQ1 record {q['id']} changed or missing")
    for r in records:
        if r.get("origin") not in ("rq1", "task4") or set(r.get("gains") or {}) != {"Kp", "Ki", "Kd"}:
            probs.append(f"{r['id']}: origin or gains missing")
        p = check_record({k: v for k, v in r.items() if k not in ADDED_KEYS})
        if p:
            probs.append(f"{r['id']}: limits/format: " + "; ".join(p))
    if sha256(spectra_path) != spectra_sha:
        probs.append("spectra file SHA-256 differs from the meta")
    z = np.load(spectra_path)
    if list(z["ids"]) != ids or z["S"].shape != (len(ids), len(rr.F_SYN)):
        probs.append("spectra ids or shape do not match the records")
    return probs


# ------------------------------------------------------------------ the run

def runlog(event, **kw):
    os.makedirs(LABELS, exist_ok=True)
    with open(RUNLOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(event=event, time=now(), **kw)) + "\n")


def counts_of(res):
    ds = res["drones"]
    return dict(draws=res["draws"], in_window=res["in_window"], kept=res["kept"],
                scenario_b=int(sum(bool(d["motor_hz"]) for d in ds)), flagged=int(sum(d["flagged"] for d in ds)))


def check_rq1_meta(meta, cfg):
    """The RQ1 file must come from the RQ1 commit and the same seed, limits, set, grids and chip delays."""
    want = dict(part="synthetic", half="fit", seed=cfg["seed"], limits=cfg["limits"], width=cfg["width"],
                delay_samples=cfg["delay_samples"], fast=cfg["fast"], full=cfg["full"], f_hi=cfg["f_hi"],
                chip_delays=cfg["chip_delays"])
    bad = [k for k, v in want.items() if json.loads(json.dumps(meta.get(k))) != v]
    if not str(meta.get("git", "")).startswith(RQ1_COMMIT):
        bad.append(f"git {meta.get('git')}")
    if bad:
        raise GuardError(f"rq1_synthetic.json does not match the Task 4 config: {bad}")


def summary(records):
    out = dict(n=len(records))
    for mode in ("A", "B"):
        out[f"feasible_{mode}"] = int(sum(r[mode]["feasible"] for r in records))
    out["scenario_b"] = int(sum(r["scenario"] == "B" for r in records))
    out["flagged"] = int(sum(bool(r["flagged"]) for r in records))
    sec = [r["seconds"] for r in records]
    out["solver_seconds"] = dict(total=round(float(np.sum(sec)), 1), median=float(np.median(sec)),
                                 p90=float(np.percentile(sec, 90))) if sec else None
    return out


def run():
    t_start = now()
    require_clean_tree()
    git = git_hash()
    digest = source_digest()
    print(f"Task 4 run: git {git}, start {t_start}", flush=True)
    inputs = input_hashes()
    rq1 = load_rq1()
    t = time.perf_counter()
    res, delays = sample(N)
    drones = res["drones"]
    cfg = config(delays, inputs)
    chash = config_hash(cfg)
    runlog("start", start=t_start, git=git, config_hash=chash, pid=os.getpid())
    print(f"sampled {len(drones)} drones in {time.perf_counter() - t:.0f} s", flush=True)
    check_rq1_meta(rq1["meta"], cfg)
    counts = counts_of(res)
    ref = dict(draws=rq1["meta"]["draws"], in_window=rq1["meta"]["in_window"], kept=rq1["meta"]["kept"],
               scenario_b=int(sum(r["scenario"] == "B" for r in rq1["results"])), flagged=rq1["meta"]["flagged"])
    print(f"counts {counts} (RQ1 {ref})", flush=True)
    probs = count_check(counts, ref)
    if probs:
        raise GuardError("counts far from the plan: " + "; ".join(probs))
    for d, q in zip(drones[:N_RQ1], rq1["results"]):
        p = _drone_problems(d, q) if d["id"] == q["id"] else ["id"]
        if p:
            raise GuardError(f"re-sampled drone {d['id']} is not the RQ1 drone: {p}")
    print(f"the first {N_RQ1} drones are the RQ1 drones (all fields)", flush=True)

    arr = spectra_arrays(drones)
    tmp = SPECTRA + ".new.npz"
    save_spectra(tmp, arr)
    sp_sha = sha256(tmp)
    if os.path.exists(SPECTRA) and sha256(SPECTRA) != sp_sha:
        raise GuardError("the existing spectra file differs from the regenerated one")
    os.replace(tmp, SPECTRA)
    runlog("spectra", sha256=sp_sha)
    print(f"spectra file written: {arr['S'].shape} float32, SHA-256 {sp_sha}", flush=True)

    t = time.perf_counter()
    reuse = reuse_check({d["id"]: d for d in drones[:N_RQ1]}, rq1["results"], rr._synthetic_work, n=N_REUSE,
                        seed=REUSE_SEED, n_jobs=N_JOBS)
    runlog("reuse", **{k: v for k, v in reuse.items() if k != "mismatches"}, mismatches=reuse["mismatches"])
    print(f"reuse check: {reuse['n'] - reuse['n_mismatch']}/{reuse['n']} match, largest difference "
          f"{reuse['max_abs_diff']:.3g} ({time.perf_counter() - t:.0f} s); picked {reuse['picked']}", flush=True)
    if not reuse["ok"]:
        raise GuardError(f"reuse check mismatch (nothing reused): {reuse['mismatches']}")

    header = make_header(SEED, N, git, chash)
    new_items = drones[N_RQ1:]
    kw = dict(chunk=CHUNK, n_jobs=N_JOBS, check=check_record, guard=source_digest, source_digest=digest)
    done = label_chunks(new_items[:CHUNK], rr._synthetic_work, PARTIAL, header, **kw)
    smoke = [done[d["id"]] for d in new_items[:CHUNK]]
    print(f"smoke chunk OK (format and limits checked): {summary(smoke)}", flush=True)
    runlog("smoke", **summary(smoke))
    done = label_chunks(new_items, rr._synthetic_work, PARTIAL, header, **kw)

    records = merge(drones, rq1["results"], {d["id"]: done[d["id"]] for d in new_items})
    write_records(RECORDS, records)
    back = read_records(RECORDS)
    probs = verify(back, rq1["results"], SPECTRA, sp_sha, n=N)
    if probs:
        raise GuardError(f"verification of the merged set failed ({len(probs)}): {probs[:10]}")
    starts = [e for e in (json.loads(x) for x in open(RUNLOG, encoding="utf-8"))
              if e["event"] == "start" and e["git"] == git and e["config_hash"] == chash]
    t_end = now()
    meta = dict(
        task="Task 4 training set (THESIS 7.6; D-38, D-40; notes/task4_plan.md)", code="experiments/phase4/dataset_gen.py",
        seed=SEED, n=N, git=git, config_hash=chash,
        rq1=dict(file="experiments/phase4/rq1_synthetic.json", commit=rq1["meta"]["git"], n=N_RQ1,
                 reuse_check={k: v for k, v in reuse.items()}),
        limits=cfg["limits"], width=cfg["width"], delay_samples=cfg["delay_samples"],
        delay_step="1 controller sample + the rest of the chip-delay range (D-33, D-36 (1))",
        fast=cfg["fast"], full=cfg["full"], f_hi=cfg["f_hi"], f_grid=cfg["f_grid"], chip_delays=cfg["chip_delays"],
        spectra=dict(file="data/labels/train_v1_spectra.npz", in_git=False, arrays=dict(f="float64 (1521,)",
                     S=f"float32 {list(arr['S'].shape)}", ids=f"{arr['ids'].dtype} ({len(arr['ids'])},)"),
                     note="labels were computed on the float64 spectra; regenerate with 'dataset_gen.py check-spectra'"),
        records=dict(file="data/labels/train_v1.jsonl.gz", format="one JSON record per line (rq1_run._synthetic_work "
                     "record + gains + origin)", added_keys=list(ADDED_KEYS)),
        sha256=dict(inputs=inputs, outputs={"train_v1.jsonl.gz": sha256(RECORDS), "train_v1_spectra.npz": sp_sha}),
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=__import__("scipy").__version__,
                      joblib=__import__("joblib").__version__, platform=platform.platform()),
        n_jobs=N_JOBS, chunk=CHUNK, start=starts[0]["start"] if starts else t_start, end=t_end,
        invocations=[e["start"] for e in starts],
        counts=dict(counts, scenario_b_share=counts["scenario_b"] / counts["kept"],
                    kept_share=counts["kept"] / counts["in_window"], in_window_share=counts["in_window"] / counts["draws"],
                    flagged_share=counts["flagged"] / counts["kept"], rq1_reference=ref),
        labels=dict(all=summary(back), new=summary(back[N_RQ1:]), rq1=summary(back[:N_RQ1])))
    with open(META, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
        fh.write("\n")
    runlog("end", end=t_end)
    print(f"done: {len(back)} records, meta {META}, end {t_end}", flush=True)


def verify_cli():
    meta = json.load(open(META))
    recs = read_records(RECORDS)
    rq1 = load_rq1()
    probs = verify(recs, rq1["results"], SPECTRA, meta["sha256"]["outputs"]["train_v1_spectra.npz"], n=meta["n"])
    if sha256(RECORDS) != meta["sha256"]["outputs"]["train_v1.jsonl.gz"]:
        probs.append("records SHA-256 differs from the meta")
    c = meta["counts"]
    probs += count_check(c, c["rq1_reference"])
    print(f"verify: {len(recs)} records, {summary(recs)}")
    print("verify: OK" if not probs else f"verify: {len(probs)} problems: {probs[:20]}")
    return probs


def check_spectra_cli():
    meta = json.load(open(META))
    res, _ = sample(meta["n"], meta["seed"])
    tmp = os.path.join(LABELS, "train_v1_spectra.check.npz")
    save_spectra(tmp, spectra_arrays(res["drones"]))
    sha = sha256(tmp)
    os.remove(tmp)
    want = meta["sha256"]["outputs"]["train_v1_spectra.npz"]
    print(f"regenerated spectra SHA-256 {sha}; meta {want}; {'MATCH' if sha == want else 'MISMATCH'}")
    return sha == want


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "run":
        run()
    elif cmd == "verify":
        sys.exit(1 if verify_cli() else 0)
    elif cmd == "check-spectra":
        sys.exit(0 if check_spectra_cli() else 1)
    else:
        sys.exit(__doc__)
