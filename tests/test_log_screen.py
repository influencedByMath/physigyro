"""Index filter, sampling order and the polite header screen (THESIS 7.4.2 A-B, decision D-21)."""
import io
import json
import os
import urllib.error

import pytest

import log_screen as ls

REAL_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "raw_logs", "b27d09e3-4542-4496-a18b-2c3c5317024e.ulg")


def _entry(**kw):
    e = dict(log_id="x", log_date="2024-05-01", mav_type="Quadrotor", sys_hw="PX4_FMU_V6X", source="webui",
             duration_s=120, vehicle_uuid="000600000000383832353233510d002e0040",
             download_url="https://cdn.logs.px4.io/x.ulg")
    e.update(kw)
    return e


@pytest.mark.parametrize("kw,expected", [
    (dict(), True),
    (dict(sys_hw="PX4_SITL"), False),
    (dict(sys_hw="px4_sitl_default"), False),
    (dict(sys_hw="CUSTOM_SIM_BOARD"), False),
    (dict(source="CI"), False),
    (dict(duration_s=59), False),
    (dict(duration_s=60), True),
    (dict(log_date="2020-12-31"), False),
    (dict(log_date="2021-01-01"), True),
    (dict(mav_type="VTOL Tailsitter"), False),
    (dict(mav_type="Hexarotor"), True),
    (dict(mav_type="Octorotor"), True),
    (dict(duration_s=None), False),
])
def test_is_candidate_table(kw, expected):
    assert ls.is_candidate(_entry(**kw)) is expected


@pytest.mark.parametrize("raw,expected", [
    ("000600000000383832353233510d002e0040", "000600000000383832353233510d002e0040"),
    ("X1-CPX1002605\n", "X1-CPX1002605"),
    ("0023001D3332510233313832", "0023001D3332510233313832"),
    ("", None),
    (None, None),
    ("REDACTED\x00\x00\x00", None),
    ("000a00000000000000000000000000000000", None),
])
def test_clean_uuid_rules(raw, expected):
    assert ls.clean_uuid(raw) == expected


def test_sampling_order_newest_per_vehicle_and_stable():
    A, B, C = "0006000000000000000000000000000000aa", "0006000000000000000000000000000000bb", "0006000000000000000000000000000000cc"
    logs = [_entry(log_id="a-old", vehicle_uuid=A, log_date="2024-01-01"),
            _entry(log_id="a-short", vehicle_uuid=A, log_date="2026-02-01", duration_s=100),
            _entry(log_id="a-long", vehicle_uuid=A, log_date="2026-02-01", duration_s=300),
            _entry(log_id="b", vehicle_uuid=B),
            _entry(log_id="n1", vehicle_uuid=""), _entry(log_id="n2", vehicle_uuid="")]
    order = [e["log_id"] for e in ls.sampling_order(logs, "salt", [])]
    assert sorted(order) == ["a-long", "b", "n1", "n2"]          # one log per vehicle; no-ID logs stand alone
    more = [e["log_id"] for e in ls.sampling_order(logs + [_entry(log_id="c", vehicle_uuid=C)], "salt", [])]
    assert [x for x in more if x != "c"] == order                 # adding a vehicle keeps the relative order
    first = [e["log_id"] for e in ls.sampling_order(logs, "salt", ["a-old"])]
    assert first[0] == "a-old" and "a-long" not in first           # first logs replace their vehicle's pick


class _Resp(io.BytesIO):
    headers = {}


@pytest.mark.skipif(not os.path.exists(REAL_LOG), reason="pilot log not downloaded")
def test_screen_log_reads_header_and_stops_early():
    row = ls.screen_log(_entry(log_id="b27d09e3"), opener=lambda req, timeout: open(REAL_LOG, "rb"))
    assert row["err"] is None and row["status"] == "ok"
    assert row["fifo"] is True
    assert row["bytes_read"] < 3_000_000
    assert row["log_id"] == "b27d09e3" and row["vehicle_uuid"] == "000600000000383832353233510d002e0040"


def test_screen_log_reports_error_not_negative():
    def http_404(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)
    row = ls.screen_log(_entry(), opener=http_404)
    assert row["status"] == "error" and "404" in row["err"]
    row = ls.screen_log(_entry(), opener=lambda req, timeout: _Resp(b"<html><body>Not here</body></html>"))
    assert row["status"] == "not_ulog" and row["err"]


def test_screen_log_sends_range_and_user_agent():
    seen = {}

    def opener(req, timeout):
        seen.update(range=req.get_header("Range"), ua=req.get_header("User-agent"))
        return _Resp(b"")
    ls.screen_log(_entry(), opener=opener, max_bytes=3_000_000)
    assert seen["range"] == "bytes=0-2999999"
    assert seen["ua"].startswith("px4-noise-thesis/")


def _fake_rows(monkeypatch, fifo=False):
    calls = []

    def fake(entry, **kw):
        calls.append(entry["log_id"])
        return dict(log_id=entry["log_id"], vehicle_uuid=entry["vehicle_uuid"], status="ok", fifo=fifo, info={},
                    err=None)
    monkeypatch.setattr(ls, "screen_log", fake)
    return calls


def test_run_screen_resumes_without_duplicates(tmp_path, monkeypatch):
    order = [_entry(log_id=f"L{i}", vehicle_uuid=f"0006{i:032d}1") for i in range(5)]
    out = tmp_path / "screen.jsonl"
    out.write_text("".join(json.dumps(dict(log_id=f"L{i}", vehicle_uuid=order[i]["vehicle_uuid"], status="ok",
                                            fifo=False, info={}, err=None)) + "\n" for i in range(2)))
    calls = _fake_rows(monkeypatch)
    res = ls.run_screen(order, str(out), target_fifo_vehicles=100, workers=2, pause_s=0.0)
    assert sorted(calls) == ["L2", "L3", "L4"]
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert sorted(r["log_id"] for r in lines) == [f"L{i}" for i in range(5)]
    assert res["total_rows"] == 5 and res["screened_now"] == 3


def test_run_screen_stops_at_target(tmp_path, monkeypatch):
    order = [_entry(log_id=f"L{i}", vehicle_uuid=f"0006{i:032d}1") for i in range(50)]
    calls = _fake_rows(monkeypatch, fifo=True)
    res = ls.run_screen(order, str(tmp_path / "s.jsonl"), target_fifo_vehicles=3, workers=2, pause_s=0.0)
    assert res["fifo_vehicles"] >= 3
    assert len(calls) <= 3 + 2                                     # at most the in-flight requests on top


# ------------------------------------------------------------ Task 4: screen summary

def test_wilson_ci_matches_audit_numbers():
    lo, hi = ls.wilson_ci(23, 300)                                 # audit 2026-10-06: 7.7 % (5.2-11.2 %)
    assert abs(lo - 0.0516) < 1e-3 and abs(hi - 0.1124) < 1e-3
    assert ls.wilson_ci(0, 0) == (0.0, 0.0)


def _row(log_id, uuid, status="ok", fifo=False, profile=1, n_q=10, **params):
    return dict(log_id=log_id, vehicle_uuid=uuid, status=status, fifo=fifo, err=None if status != "error" else "x",
                info={"sys_uuid": uuid}, params=dict(SDLOG_PROFILE=profile, **params), n_default_records=n_q,
                topics={"sensor_gyro_fifo": [0]} if fifo else {}, fifo_data_msgs=5 if fifo else 0,
                log_date="2026-01-01", sys_hw="HW", download_url=f"https://cdn.logs.px4.io/{log_id}.ulg")


def test_summarize_counts_and_rates():
    rows = [_row("a", "0006000000000000000000000000000000a1", fifo=True, profile=273, IMU_GYRO_RATEMAX=800),
            _row("b", "0006000000000000000000000000000000b1", fifo=True, profile=1, n_q=0),
            _row("c", "0006000000000000000000000000000000c1"),
            _row("d", "0006000000000000000000000000000000d1", status="incomplete"),
            _row("e", "0006000000000000000000000000000000e1", status="error")]
    s = ls.summarize(rows)
    assert s["n_rows"] == 5 and s["status"] == {"ok": 3, "incomplete": 1, "error": 1}
    assert s["fifo_logs"] == 2 and s["fifo_vehicles"] == 2
    assert s["hit_rate"] == 2 / 3                                  # unknown (incomplete, error) not in the denominator
    assert s["fifo_defaults_known_share"] == 0.5
    assert s["fifo_profiles"] == {"273": 1, "1": 1}


def test_write_summary_files(tmp_path):
    import csv
    rows = [_row("a", "0006000000000000000000000000000000a1", fifo=True, profile=273, IMU_GYRO_RATEMAX=800)]
    ls.write_summary(rows, str(tmp_path / "s.csv"), str(tmp_path / "s.jsonl.gz"))
    with open(tmp_path / "s.csv", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert got[0]["log_id"] == "a" and got[0]["fifo"] == "True" and got[0]["SDLOG_PROFILE"] == "273"
    assert got[0]["profile_bit8"] == "True" and got[0]["profile_bit4"] == "True"
    import gzip
    assert [json.loads(x) for x in gzip.open(tmp_path / "s.jsonl.gz", "rt")] == rows


# ------------------------------------------------------------ Task 8: full download

class _Server:
    """Serves ``data`` with HTTP range semantics; the first response can drop after ``cut`` bytes."""

    def __init__(self, data, cut_first=None):
        self.data, self.cut, self.ranges = data, cut_first, []

    def __call__(self, req, timeout):
        rng = req.get_header("Range")
        self.ranges.append(rng)
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = self.data[start:]
        if self.cut is not None and len(self.ranges) == 1:
            body = body[:self.cut]
        r = _Resp(body)
        r.headers = {"Content-Range": f"bytes {start}-{len(self.data) - 1}/{len(self.data)}"}
        return r


ULOG_BYTES = b"ULog\x01\x12\x35\x01" + bytes(8) + bytes(range(256)) * 40


def test_download_resumes_and_checks_size(tmp_path):
    srv = _Server(ULOG_BYTES, cut_first=3000)
    path = ls.download_log(_entry(log_id="abc"), str(tmp_path), opener=srv, pause_s=0.0, chunk=1024)
    assert path == str(tmp_path / "abc.ulg")
    assert open(path, "rb").read() == ULOG_BYTES
    assert srv.ranges == ["bytes=0-", "bytes=3000-"]
    assert not os.path.exists(path + ".part")
    again = _Server(ULOG_BYTES)
    assert ls.download_log(_entry(log_id="abc"), str(tmp_path), opener=again, pause_s=0.0) == path
    assert again.ranges == []                                       # already complete: no request


def test_download_rejects_non_ulog(tmp_path):
    with pytest.raises(ValueError):
        ls.download_log(_entry(log_id="bad"), str(tmp_path), opener=_Server(b"<html>" + bytes(5000)), pause_s=0.0)
    assert os.listdir(tmp_path) == []


def test_download_many_reports_each_log(tmp_path):
    srv = _Server(ULOG_BYTES)
    bad = _Server(b"<html>" + bytes(100))
    opener = lambda req, timeout: (bad if "bad" in req.full_url else srv)(req, timeout)
    entries = [_entry(log_id="good1", download_url="https://cdn.logs.px4.io/good1.ulg"),
               _entry(log_id="bad1", download_url="https://cdn.logs.px4.io/bad1.ulg"),
               _entry(log_id="good2", download_url="https://cdn.logs.px4.io/good2.ulg")]
    res = ls.download_many(entries, str(tmp_path), workers=2, opener=opener, pause_s=0.0)
    assert res["good1"]["ok"] and res["good2"]["ok"] and not res["bad1"]["ok"]
    assert "not a ULog" in res["bad1"]["error"]
    assert sorted(os.listdir(tmp_path)) == ["good1.ulg", "good2.ulg"]


def test_run_screen_retries_errors_and_survives_cut_last_line(tmp_path, monkeypatch):
    order = [_entry(log_id=f"L{i}", vehicle_uuid=f"0006{i:032d}1") for i in range(4)]
    out = tmp_path / "screen.jsonl"
    ok = dict(log_id="L0", vehicle_uuid=order[0]["vehicle_uuid"], status="ok", fifo=False, info={}, err=None)
    err = dict(log_id="L1", vehicle_uuid=order[1]["vehicle_uuid"], status="error", fifo=False, info={}, err="timeout")
    out.write_text(json.dumps(ok) + "\n" + json.dumps(err) + "\n" + '{"log_id": "L2", "sta')   # power cut mid-line
    calls = _fake_rows(monkeypatch)
    ls.run_screen(order, str(out), target_fifo_vehicles=100, workers=2, pause_s=0.0)
    assert sorted(calls) == ["L1", "L2", "L3"]
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert sorted(r["log_id"] for r in rows) == ["L0", "L1", "L2", "L3"]
    assert all(r["status"] == "ok" for r in rows)


def test_run_screen_repairs_missing_final_newline(tmp_path, monkeypatch):
    order = [_entry(log_id=f"L{i}", vehicle_uuid=f"0006{i:032d}1") for i in range(2)]
    out = tmp_path / "screen.jsonl"
    out.write_text(json.dumps(dict(log_id="L0", vehicle_uuid=order[0]["vehicle_uuid"], status="ok", fifo=False,
                                   info={}, err=None)))                       # complete row, newline lost
    _fake_rows(monkeypatch)
    ls.run_screen(order, str(out), target_fifo_vehicles=100, workers=1, pause_s=0.0)
    assert [json.loads(x)["log_id"] for x in out.read_text().splitlines()] == ["L0", "L1"]


class _NoRangeServer(_Server):
    """A server that ignores Range and always sends the whole file (HTTP 200, Content-Length only)."""

    def __call__(self, req, timeout):
        self.ranges.append(req.get_header("Range"))
        body = self.data[:self.cut] if (self.cut is not None and len(self.ranges) == 1) else self.data
        r = _Resp(body)
        r.headers = {"Content-Length": str(len(self.data))}
        return r


def test_download_restarts_when_server_ignores_range(tmp_path):
    srv = _NoRangeServer(ULOG_BYTES, cut_first=3000)
    path = ls.download_log(_entry(log_id="nr"), str(tmp_path), opener=srv, pause_s=0.0, chunk=1024)
    assert open(path, "rb").read() == ULOG_BYTES                     # no duplicated prefix


def test_download_retries_when_first_request_fails(tmp_path):
    srv, calls = _Server(ULOG_BYTES), []

    def flaky(req, timeout):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset")
        return srv(req, timeout)
    path = ls.download_log(_entry(log_id="fl"), str(tmp_path), opener=flaky, pause_s=0.0)
    assert open(path, "rb").read() == ULOG_BYTES and len(calls) == 2


def test_half_entries_only_returns_the_requested_half():
    split_rows = [dict(vehicle_key="uuid:a", split="fit", log_ids="L1"),
                  dict(vehicle_key="uuid:b", split="test", log_ids="L2"),
                  dict(vehicle_key="uuid:c", split="fit", log_ids="L3")]
    by_id = {i: dict(log_id=i) for i in ("L1", "L2", "L3")}
    assert [e["log_id"] for e in ls.half_entries(split_rows, by_id, "fit")] == ["L1", "L3"]
    assert [e["log_id"] for e in ls.half_entries(split_rows, by_id, "test")] == ["L2"]
    with pytest.raises(ValueError):
        ls.half_entries(split_rows, by_id, "both")


def test_run_screen_order_limit_is_a_resume_safe_prefix(tmp_path, monkeypatch):
    """Extended screen (D-24): screen exactly the first N vehicles of the order; a restart never overshoots."""
    order = [_entry(log_id=f"L{i}", vehicle_uuid=f"0006{i:032d}1") for i in range(20)]
    out = tmp_path / "s.jsonl"
    out.write_text("".join(json.dumps(dict(log_id=f"L{i}", vehicle_uuid=order[i]["vehicle_uuid"], status="ok",
                                            fifo=True, info={}, err=None)) + "\n" for i in range(5)))
    calls = _fake_rows(monkeypatch, fifo=True)
    res = ls.run_screen(order, str(out), target_fifo_vehicles=None, order_limit=12, workers=2, pause_s=0.0)
    assert sorted(calls, key=lambda x: int(x[1:])) == [f"L{i}" for i in range(5, 12)]   # FIFO count never stops it
    again = ls.run_screen(order, str(out), target_fifo_vehicles=None, order_limit=12, workers=2, pause_s=0.0)
    assert again["screened_now"] == 0 and res["total_rows"] == 12
    assert sorted(json.loads(x)["log_id"] for x in out.read_text().splitlines()) == sorted(f"L{i}" for i in range(12))


def test_download_finishes_a_complete_part_file_on_http_416(tmp_path):
    """Review finding LOW-3: interrupted after the last byte but before the rename -> the resume asks for
    bytes=<size>-, the server answers 416 'bytes */<size>'; the complete .part must be accepted, not retried forever."""
    part = tmp_path / "done.ulg.part"
    part.write_bytes(ULOG_BYTES)

    def http_416(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable",
                                     {"Content-Range": f"bytes */{len(ULOG_BYTES)}"}, None)
    path = ls.download_log(_entry(log_id="done"), str(tmp_path), opener=http_416, pause_s=0.0, max_tries=2)
    assert open(path, "rb").read() == ULOG_BYTES and not part.exists()


def test_download_416_with_wrong_size_is_not_accepted(tmp_path):
    (tmp_path / "bad.ulg.part").write_bytes(ULOG_BYTES[:100])

    def http_416(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable",
                                     {"Content-Range": f"bytes */{len(ULOG_BYTES)}"}, None)
    with pytest.raises(IOError):
        ls.download_log(_entry(log_id="bad"), str(tmp_path), opener=http_416, pause_s=0.0, max_tries=2)
