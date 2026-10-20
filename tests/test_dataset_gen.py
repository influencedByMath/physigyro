"""Task 4 training set (THESIS 7.6; D-38, D-40; plan notes/task4_plan.md Section 7): reproducible sampling, the
leakage guard, resumable chunks with the header and source guards, the dirty-tree refusal, the record schema, the
merge with the RQ1 records, the limits check, the reuse check and the bit-exact spectra file."""
import builtins
import copy
import gzip
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "phase4"))
import dataset_gen as dg        # noqa: E402
import priors as pr             # noqa: E402
import rq1_run as rr            # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RQ1 = os.path.join(ROOT, "experiments", "phase4", "rq1_synthetic.json")
DRONE_KEYS = ("f_c", "tau_m", "Td", "Td_nominal", "delay_step", "r", "fs", "f_ctrl", "pm_unfiltered", "chip",
              "chip_delay", "motor_hz", "flagged", "meta")


@pytest.fixture(scope="module")
def rq1():
    return json.load(open(RQ1))


@pytest.fixture(scope="module")
def sampled():
    """The first 1,000 drones of the Task 4 sampler (seed 20261018)."""
    res, _ = dg.sample(1000)
    return res


# ------------------------------------------------------------------ 2. leakage guard

def test_building_the_set_opens_no_test_half_path(monkeypatch):
    opened = []
    real_open, real_load = builtins.open, np.load

    def rec_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def rec_load(file, *a, **k):
        opened.append(str(file))
        return real_load(file, *a, **k)
    monkeypatch.setattr(builtins, "open", rec_open)
    monkeypatch.setattr(np, "load", rec_load)
    res, _ = dg.sample(3)
    dg.load_rq1()
    dg.input_hashes()
    assert len(res["drones"]) == 3 and opened
    bad = [p for p in opened if pr._TEST_PART.search(p)]
    assert not bad, bad
    assert any(p.endswith("fit_windows.csv") for p in opened) and any(p.endswith("fit_psd_native.npz") for p in opened)


def test_every_input_is_a_fit_half_file():
    assert set(dg.INPUTS) == {"model_fits.json", "generator_check.json", "priors_from_logs.json", "fit_windows.csv",
                              "fit_psd_native.npz", "rq1_synthetic.json"}
    for name, path in dg.INPUTS.items():
        assert os.path.exists(path), name
        pr.require_fit_half(path)


# ------------------------------------------------------------------ 3. resumable chunks and the header guard

def _work(item):
    return dict(id=item["id"], value=item["x"] * 2)


def _items(n=7):
    return [dict(id=f"c{i}", x=i) for i in range(n)]


HEADER = dict(kind="header", seed=1, n=7, git="abc", config_hash="h1")


def test_stopping_after_a_chunk_and_restarting_gives_the_same_file(tmp_path):
    full = tmp_path / "full.partial.jsonl"
    dg.label_chunks(_items(), _work, str(full), HEADER, chunk=3, n_jobs=1)

    part = tmp_path / "stop.partial.jsonl"

    def failing(item):
        if item["id"] == "c4":
            raise RuntimeError("simulated crash")
        return _work(item)
    with pytest.raises(RuntimeError):
        dg.label_chunks(_items(), failing, str(part), HEADER, chunk=3, n_jobs=1)
    lines = open(part, encoding="utf-8").read().splitlines()
    assert json.loads(lines[0]) == HEADER and [json.loads(x)["id"] for x in lines[1:]] == ["c0", "c1", "c2"]
    calls = []

    def counting(item):
        calls.append(item["id"])
        return _work(item)
    done = dg.label_chunks(_items(), counting, str(part), HEADER, chunk=3, n_jobs=1)
    assert calls == ["c3", "c4", "c5", "c6"]
    assert sorted(done) == [f"c{i}" for i in range(7)] and done["c6"]["value"] == 12
    assert open(part, "rb").read() == open(full, "rb").read()


@pytest.mark.parametrize("field,value", [("git", "other"), ("config_hash", "h2"), ("seed", 2), ("n", 8)])
def test_a_restart_under_another_version_or_config_refuses_to_resume(tmp_path, field, value):
    part = tmp_path / "p.partial.jsonl"
    dg.label_chunks(_items(3), _work, str(part), HEADER, chunk=3, n_jobs=1)
    before = open(part, "rb").read()
    with pytest.raises(dg.GuardError, match="refusing to resume"):
        dg.label_chunks(_items(), _work, str(part), dict(HEADER, **{field: value}), chunk=3, n_jobs=1)
    assert open(part, "rb").read() == before


def test_a_partial_file_without_a_header_refuses(tmp_path):
    part = tmp_path / "p.partial.jsonl"
    part.write_text(json.dumps(dict(id="c0", value=0)) + "\n", encoding="utf-8")
    with pytest.raises(dg.GuardError, match="refusing to resume"):
        dg.read_partial(str(part), HEADER)


def test_a_torn_last_line_is_dropped_and_recomputed(tmp_path):
    part = tmp_path / "p.partial.jsonl"
    dg.label_chunks(_items(3), _work, str(part), HEADER, chunk=3, n_jobs=1)
    with open(part, "a", encoding="utf-8") as fh:
        fh.write('{"id": "c3", "val')                       # crash in the middle of a write
    done = dg.read_partial(str(part), HEADER)
    assert sorted(done) == ["c0", "c1", "c2"]
    done = dg.label_chunks(_items(5), _work, str(part), HEADER, chunk=3, n_jobs=1)
    assert sorted(done) == [f"c{i}" for i in range(5)]
    assert [json.loads(x)["id"] for x in open(part, encoding="utf-8").read().splitlines()[1:]] == [
        "c0", "c1", "c2", "c3", "c4"]


def test_a_broken_line_inside_the_file_refuses(tmp_path):
    part = tmp_path / "p.partial.jsonl"
    part.write_text(json.dumps(HEADER) + "\n" + "{broken\n" + json.dumps(dict(id="c1", value=2)) + "\n",
                    encoding="utf-8")
    with pytest.raises(dg.GuardError):
        dg.read_partial(str(part), HEADER)


def test_a_source_change_during_the_run_stops_after_the_chunk(tmp_path):
    part = tmp_path / "p.partial.jsonl"
    digests = iter(["d0", "d1"])
    with pytest.raises(dg.GuardError, match="source"):
        dg.label_chunks(_items(), _work, str(part), HEADER, chunk=3, n_jobs=1, guard=lambda: next(digests),
                        source_digest="d0")
    ids = [json.loads(x)["id"] for x in open(part, encoding="utf-8").read().splitlines()[1:]]
    assert ids == ["c0", "c1", "c2", "c3", "c4", "c5"]       # the first chunk passed, the second chunk is kept too


def test_a_record_that_fails_the_check_stops_the_run(tmp_path):
    part = tmp_path / "p.partial.jsonl"

    def check(r):
        return ["bad value"] if r["id"] == "c4" else []
    with pytest.raises(dg.GuardError, match="c4"):
        dg.label_chunks(_items(), _work, str(part), HEADER, chunk=3, n_jobs=1, check=check)


def test_source_digest_changes_with_a_tracked_python_file(tmp_path):
    repo = _git_repo(tmp_path)
    d0 = dg.source_digest(str(repo))
    (repo / "a.py").write_text("x = 2\n")
    assert dg.source_digest(str(repo)) != d0


# ------------------------------------------------------------------ 4. dirty working tree

def _git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                   capture_output=True)


def _git_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.py").write_text("x = 1\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_a_clean_tree_passes(tmp_path):
    dg.require_clean_tree(str(_git_repo(tmp_path)))


def test_a_modified_tracked_file_refuses_to_start(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "a.py").write_text("x = 2\n")
    with pytest.raises(dg.GuardError, match="dirty"):
        dg.require_clean_tree(str(repo))


def test_an_untracked_file_refuses_to_start(tmp_path):
    repo = _git_repo(tmp_path)
    (repo / "b.py").write_text("y = 1\n")
    with pytest.raises(dg.GuardError, match="dirty"):
        dg.require_clean_tree(str(repo))


# ------------------------------------------------------------------ 5. record schema and limits

def test_the_rq1_records_have_the_label_record_keys(rq1):
    assert all(set(r) == set(dg.RECORD_KEYS) for r in rq1["results"])


def test_every_rq1_record_passes_the_limits_check(rq1):
    for r in rq1["results"]:
        assert dg.check_record(r) == [], r["id"]


@pytest.mark.parametrize("edit,what", [
    (lambda r: r["A"]["metrics"].__setitem__("Ms", 2.0001), "Ms"),
    (lambda r: r["A"]["metrics"].__setitem__("PM", 29.99), "PM"),
    (lambda r: r["B"]["metrics"].__setitem__("GM", 5.99), "GM"),
    (lambda r: r["A"]["metrics"].__setitem__("stable", False), "stable"),
    (lambda r: r["A"]["robust"]["worst_corners"].__setitem__("Ms", 3.0001), "worst"),
    (lambda r: r["B"]["robust"]["worst_corners"].__setitem__("PM", 19.0), "worst"),
    (lambda r: r["A"]["metrics"].__setitem__("N_rel", 1.0001), "N_rel"),
    (lambda r: r["B"]["metrics"].__setitem__("PM", r["reference"]["PM"] - 1e-6), "reference"),
    (lambda r: r["A"]["settings"].__setitem__("f_gyro", 14.0), "f_gyro"),
    (lambda r: r.pop("peaks"), "keys"),
    (lambda r: r.__setitem__("scenario", "B"), "scenario"),
])
def test_the_limits_check_catches_each_violation(rq1, edit, what):
    r = copy.deepcopy(rq1["results"][0])
    edit(r)
    probs = dg.check_record(r)
    assert probs and any(what in p for p in probs), probs


def test_an_infeasible_mode_is_not_a_limits_violation(rq1):
    r = copy.deepcopy(rq1["results"][0])
    r["B"] = dict(feasible=False, n_candidates=0, n_feasible_coarse=0)
    assert dg.check_record(r) == []


# ------------------------------------------------------------------ 6. merge

def _label(d, value=0.0):
    """A label record of drone d in the RQ1 format (drone fields copied, label values fake)."""
    r = {k: d.get(k) for k in DRONE_KEYS}
    r.update(id=d["id"], part="synthetic", half="fit", vehicle=None, scenario="B" if d["motor_hz"] else "A",
             reference_robust=rr.robust_of_dict(d["default"]), value=value)
    return r


def test_records_file_round_trips_and_is_deterministic(tmp_path, rq1, monkeypatch):
    recs = rq1["results"][:5]
    a, b = tmp_path / "a.jsonl.gz", tmp_path / "b.jsonl.gz"
    dg.write_records(str(a), recs)
    monkeypatch.setattr(gzip.time, "time", lambda: 2e9)      # a later clock must not change the bytes
    dg.write_records(str(b), recs)
    assert open(a, "rb").read() == open(b, "rb").read()
    assert dg.read_records(str(a)) == recs
    assert gzip.open(a, "rt", encoding="utf-8").read().count("\n") == 5


# ------------------------------------------------------------------ reuse check (plan Section 2)

def test_compare_finds_differences_above_the_tolerance_and_ignores_seconds():
    a = dict(id="x", seconds=1.0, A=dict(metrics=dict(PM=50.0, GM="inf"), option="o", settings=dict(notches=[[1.0, 5]])))
    b = copy.deepcopy(a)
    b["seconds"] = 9.0
    assert dg.compare(a, b) == []
    b["A"]["metrics"]["PM"] = 50.0 + 5e-10
    assert dg.compare(a, b) == []
    b["A"]["metrics"]["PM"] = 50.0 + 2e-9
    assert [d[0] for d in dg.compare(a, b)] == ["A.metrics.PM"]
    c = copy.deepcopy(a)
    c["A"]["option"] = "p"
    c["A"]["settings"]["notches"][0][1] = 10
    c["A"]["metrics"]["GM"] = 7.0
    assert sorted(d[0] for d in dg.compare(a, c)) == ["A.metrics.GM", "A.option", "A.settings.notches.0.1"]
    d = copy.deepcopy(a)
    d["A"]["extra"] = 1
    assert [x[0] for x in dg.compare(a, d)] == ["A.extra"]


def test_reuse_check_passes_identical_resolves_and_reports_a_mismatch():
    rq1_res = [dict(id=f"syn{i:04d}", v=float(i), seconds=1.0) for i in range(30)]
    items = {r["id"]: dict(id=r["id"], v=r["v"]) for r in rq1_res}

    def same(it):
        return dict(id=it["id"], v=it["v"], seconds=5.0)
    out = dg.reuse_check(items, rq1_res, same, n=20, n_jobs=1)
    assert out["ok"] and out["n_mismatch"] == 0 and len(out["picked"]) == 20 == len(set(out["picked"]))
    assert out["picked"] == dg.reuse_check(items, rq1_res, same, n=20, n_jobs=1)["picked"]

    def off(it):
        return dict(id=it["id"], v=it["v"] + (1e-6 if it["id"] == out["picked"][3] else 0.0), seconds=5.0)
    bad = dg.reuse_check(items, rq1_res, off, n=20, n_jobs=1)
    assert not bad["ok"] and bad["n_mismatch"] == 1 and bad["mismatches"][0]["id"] == out["picked"][3]


# ------------------------------------------------------------------ counts

def test_count_check_accepts_the_rq1_shares_and_flags_a_far_share():
    ref = dict(draws=2223, in_window=1152, kept=1000, scenario_b=101, flagged=44)
    ok = dict(draws=22230, in_window=11520, kept=10000, scenario_b=1010, flagged=440)
    assert dg.count_check(ok, ref) == []
    assert dg.count_check(dict(ok, scenario_b=1050, flagged=470), ref) == []      # inside the sampling spread
    assert any("Scenario B" in p for p in dg.count_check(dict(ok, scenario_b=2000), ref))
    assert any("kept" in p for p in dg.count_check(dict(ok, in_window=12500), ref))
    assert any("flagged" in p for p in dg.count_check(dict(ok, flagged=1000), ref))
    assert any("in window" in p for p in dg.count_check(dict(ok, draws=30000), ref))


# ------------------------------------------------------------------ 7. spectra file (T4-2 (a))


def test_spectra_file_regenerates_bit_exactly_from_the_seed(tmp_path):
    a, b = tmp_path / "a.npz", tmp_path / "b.npz"
    dg.save_spectra(str(a), dg.spectra_arrays(dg.sample(5)[0]["drones"]))
    dg.save_spectra(str(b), dg.spectra_arrays(dg.sample(5)[0]["drones"]))
    assert dg.sha256(str(a)) == dg.sha256(str(b)) == hashlib.sha256(open(a, "rb").read()).hexdigest()
    z = np.load(str(a))
    assert z["S"].dtype == np.float32 and list(z["ids"]) == [f"syn{i:04d}" for i in range(5)]


# ------------------------------------------------------------------ tests that need the 1,000 sampled drones
# (last, so that a failing fast test stops a mutation run early)


# ------------------------------------------------------------------ 1. reproducible sampling (Review Focus 4)

def test_same_seed_gives_the_same_first_100_drones_and_spectra(sampled):
    res, _ = dg.sample(100)
    assert [d["id"] for d in res["drones"]] == [f"syn{i:04d}" for i in range(100)]
    for a, b in zip(res["drones"], sampled["drones"][:100]):
        for k in DRONE_KEYS + ("gains", "default"):
            assert a[k] == b[k], (a["id"], k)
        assert np.array_equal(a["S"], b["S"])
    a, b = dg.spectra_arrays(res["drones"]), dg.spectra_arrays(sampled["drones"][:100])
    assert all(np.array_equal(a[k], b[k]) for k in ("f", "S", "ids"))


def test_the_first_1000_drones_are_the_rq1_drones(sampled, rq1):
    assert len(sampled["drones"]) == 1000 and len(rq1["results"]) == 1000
    for d, r in zip(sampled["drones"], rq1["results"]):
        assert d["id"] == r["id"]
        for k in DRONE_KEYS:
            assert d[k] == r[k], (d["id"], k)
        assert rr.robust_of_dict(d["default"]) == r["reference_robust"]       # NaN / inf stored as in RQ1
    assert (sampled["draws"], sampled["in_window"], sampled["kept"]) == (rq1["meta"]["draws"], rq1["meta"]["in_window"],
                                                                         rq1["meta"]["kept"])


def test_the_record_holds_every_field_task5_needs(rq1, sampled):
    recs = dg.merge(sampled["drones"][:2], rq1["results"][:2], {})
    r = recs[0]
    for path in dg.TASK5_FIELDS:
        v = r
        for k in path.split("."):
            assert k in v, path
            v = v[k]
    assert set(r["gains"]) == {"Kp", "Ki", "Kd"} and all(isinstance(x, float) for x in r["gains"].values())
    for mode in ("A", "B"):
        assert {"f_gyro", "f_dterm", "notches", "dnf_esc", "dnf_bw", "dnf_hmc"} <= set(r[mode]["settings"])
    json.dumps(recs)


def test_merge_gives_unique_ordered_ids_and_keeps_the_rq1_records_unchanged(sampled):
    drones = sampled["drones"][:5]
    rq1_part = [_label(d, 1.0) for d in drones[:3]]
    snapshot = json.dumps(rq1_part, sort_keys=True)
    new = {d["id"]: _label(d, 2.0) for d in drones[3:]}
    recs = dg.merge(drones, rq1_part, new)
    assert [r["id"] for r in recs] == [d["id"] for d in drones]
    assert json.dumps(rq1_part, sort_keys=True) == snapshot
    for r, q in zip(recs[:3], rq1_part):
        assert {k: v for k, v in r.items() if k not in dg.ADDED_KEYS} == q and r["origin"] == "rq1"
    assert [r["origin"] for r in recs[3:]] == ["task4", "task4"] and recs[4]["value"] == 2.0
    assert recs[0]["gains"] == {k: float(v) for k, v in drones[0]["gains"].items()}


def test_merge_refuses_a_missing_duplicate_or_mismatched_case(sampled):
    drones = sampled["drones"][:4]
    rq1_part = [_label(d) for d in drones[:2]]
    new = {d["id"]: _label(d) for d in drones[2:]}
    with pytest.raises(ValueError, match="missing"):
        dg.merge(drones, rq1_part, {drones[2]["id"]: new[drones[2]["id"]]})
    with pytest.raises(ValueError, match="twice"):
        dg.merge(drones, rq1_part, dict(new, **{drones[1]["id"]: _label(drones[1])}))
    bad = dict(new)
    bad[drones[3]["id"]] = dict(new[drones[3]["id"]], f_c=1.0)
    with pytest.raises(ValueError, match="f_c"):
        dg.merge(drones, rq1_part, bad)
    bad = dict(new)
    bad[drones[3]["id"]] = copy.deepcopy(new[drones[3]["id"]])
    bad[drones[3]["id"]]["reference_robust"]["nominal"]["PM"] += 1e-3
    with pytest.raises(ValueError, match="default"):
        dg.merge(drones, rq1_part, bad)
    with pytest.raises(ValueError, match="order"):
        dg.merge(drones, rq1_part[::-1], new)


def test_spectra_arrays_are_float32_in_record_order(sampled):
    arr = dg.spectra_arrays(sampled["drones"][:4])
    assert arr["S"].dtype == np.float32 and arr["S"].shape == (4, 1521)
    assert np.array_equal(arr["f"], np.arange(20.0, 780.0 + 1e-9, 0.5))
    assert list(arr["ids"]) == ["syn0000", "syn0001", "syn0002", "syn0003"]
    assert np.array_equal(arr["S"][2], sampled["drones"][2]["S"].astype(np.float32))


def test_the_stored_spectra_file_matches_the_seed_and_the_meta(sampled):
    if not (os.path.exists(dg.SPECTRA) and os.path.exists(dg.META)):
        pytest.skip("the Task 4 spectra file and meta are written by the run")
    meta = json.load(open(dg.META))
    assert dg.sha256(dg.SPECTRA) == meta["sha256"]["outputs"]["train_v1_spectra.npz"]
    z = np.load(dg.SPECTRA)
    ref = dg.spectra_arrays(sampled["drones"][:200])
    assert z["S"].shape == (dg.N, 1521) and np.array_equal(z["f"], ref["f"])
    assert np.array_equal(z["S"][:200], ref["S"]) and list(z["ids"][:200]) == list(ref["ids"])


# ------------------------------------------------------------------ verification of the merged set

def test_verify_accepts_a_good_set_and_names_each_problem(tmp_path, rq1, sampled):
    drones = sampled["drones"][:3]
    rq1_part = rq1["results"][:3]
    recs = dg.merge(drones, rq1_part, {})
    sp = tmp_path / "s.npz"
    dg.save_spectra(str(sp), dg.spectra_arrays(drones))
    sha = dg.sha256(str(sp))
    assert dg.verify(recs, rq1_part, str(sp), sha, n=3) == []
    assert any("ids" in p for p in dg.verify(recs[:2] + [recs[1]], rq1_part, str(sp), sha, n=3))
    bad = copy.deepcopy(recs)
    bad[1]["A"]["metrics"]["PM"] += 1.0
    assert any("RQ1" in p for p in dg.verify(bad, rq1_part, str(sp), sha, n=3))
    assert any("SHA-256" in p for p in dg.verify(recs, rq1_part, str(sp), "0" * 64, n=3))
    bad = copy.deepcopy(recs)
    bad[2]["A"]["metrics"]["Ms"] = 2.5
    assert any("limits" in p for p in dg.verify(bad, rq1_part, str(sp), sha, n=3))
