"""Mutation check of the Task 4 tests: break dataset_gen.py one way at a time, run tests/test_dataset_gen.py, restore
byte-identical (SHA-256 checked).

    .venv/Scripts/python experiments/phase4/mutation_check_task4.py

Never run it while a dataset_gen.py run is running: its worker processes import the source files from disk (the
run's source guard would stop it).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mutation_check_priors as mcp  # noqa: E402

mcp.TIMEOUT_S = 600     # the whole test file takes about 140 s (it samples 1,000 drones); keep timeouts meaningful

SRC = os.path.join(HERE, "dataset_gen.py")
MUTANTS = [
    # inputs and leakage
    ("a test-half input", 'pr.fit_path(ROOT, "data", "spectra", "fit_windows.csv")',
     'os.path.join(ROOT, "data", "spectra", "test_windows.csv")'),
    # guards
    ("header ignored", "    if found != header:", "    if False:"),
    ("torn line kept", '    if lines[-1] != "":', "    if False:"),
    ("no resume", '    todo = [it for it in items if it["id"] not in done]', "    todo = list(items)"),
    ("chunk check ignored", "        problems = [f\"{r['id']}: {p}\" for r in res for p in (check(r) if check else [])]",
     "        problems = []"),
    ("source guard off", "        if guard is not None and guard() != source_digest:", "        if False:"),
    ("source digest without content", 'h.update(p.encode() + b"\\0" + bytes.fromhex(sha256(os.path.join(root, p))))',
     "h.update(p.encode())"),
    ("dirty tree allowed", "    if out.strip():", "    if False:"),
    ("untracked files allowed", '"--untracked-files=all"', '"--untracked-files=no"'),
    # limits check
    ("nominal stability unchecked", '        if met.get("stable") is not True:', "        if False:"),
    ("worst-corner Ms unchecked", '        if not _f(w.get("Ms")) <= wc["Ms"]:', "        if False:"),
    ("worst-corner PM unchecked", '        if not _f(w.get("PM")) >= wc["PM"]:', "        if False:"),
    ("Mode A noise unchecked", '        if mode == "A" and not _f(met.get("N_rel")) <= 1.0:', "        if False:"),
    ("Mode B reference unchecked", '        if mode == "B" and not _f(met.get("PM")) >= pm_ref:', "        if False:"),
    ("ranges unchecked", "            if not lo * (1 - 1e-5) <= _f(s.get(key)) <= hi * (1 + 1e-5):", "            if False:"),
    ("keys unchecked", "    if set(r) != set(RECORD_KEYS):", "    if False:"),
    # counts, reuse
    ("count check off", "        if abs(p - p0) > z * sd:", "        if False:"),
    ("compare tolerance ignored", "            return [] if abs(a - b) <= tol else [(path, a, b)]", "            return []"),
    ("seconds compared", 'def compare(a, b, tol=REUSE_TOL, skip=("seconds",), path=""):',
     'def compare(a, b, tol=REUSE_TOL, skip=(), path=""):'),
    ("reuse mismatch ignored", "ok=not mismatches,", "ok=True,"),
    ("reuse picks unseeded", "    rng = np.random.default_rng(seed)", "    rng = np.random.default_rng()"),
    # merge and files
    ("drone fields unchecked",
     '    probs = [f"drone field {k} differs ({src.get(k)!r} vs {d[k]!r})" for k in DRONE_KEYS if src.get(k) != d[k]]',
     "    probs = []"),
    ("default metrics unchecked", '    if src.get("reference_robust") != rr.robust_of_dict(d["default"]):', "    if False:"),
    ("missing labels allowed", "    if missing:", "    if False:"),
    ("origin always task4", '        rec["origin"] = "rq1" if i < n_rq1 else "task4"', '        rec["origin"] = "task4"'),
    ("gzip time stamp", "mtime=0)", "mtime=None)"),
    ("spectra float64", 'S=np.stack([d["S"] for d in drones]).astype(np.float32),', 'S=np.stack([d["S"] for d in drones]),'),
    ("verify RQ1 off", '        if r is None or {k: v for k, v in r.items() if k not in ADDED_KEYS} != q or r.get("origin") != "rq1":',
     "        if r is None:"),
    ("verify SHA-256 off", "    if sha256(spectra_path) != spectra_sha:", "    if False:"),
    ("verify limits off", '        p = check_record({k: v for k, v in r.items() if k not in ADDED_KEYS})', "        p = []"),
]

if __name__ == "__main__":
    mcp.main(SRC, MUTANTS, "tests/test_dataset_gen.py")
