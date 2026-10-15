"""Next-newest-log fallback (D-24 (5), details P2): trigger, Q3 duplicates, search order and caps."""
import fallback as fb


def rec(rej=(), n_candidates=10, q10=10, error=False):
    r = dict(log_rejections=["error"] if error else list(rej), n_candidates=n_candidates,
             window_reasons={"Q10_unsteady": q10} if q10 else {}, windows=[])
    return r


def test_fallback_trigger_only_zero_steady():
    assert fb.needs_fallback(rec(n_candidates=10, q10=10))            # flew, every window unsteady
    assert fb.needs_fallback(rec(n_candidates=0, q10=0))              # never airborne
    assert not fb.needs_fallback(rec(n_candidates=10, q10=9))         # one steady window (even if it failed Q13)
    assert not fb.needs_fallback(rec(rej=["Q4_dropout"], n_candidates=0, q10=0))   # log gate: never triggers
    assert not fb.needs_fallback(rec(rej=["Q10_no_rate_setpoint"], n_candidates=0, q10=0))
    assert not fb.needs_fallback(rec(error=True, n_candidates=0, q10=0))


def row(log_id, date="2026-05-01", dur=300, size=1000, fifo=True, status="ok", utc=10800):
    # utc: time_ref_utc is the TIME-ZONE OFFSET (PX4 logger.cpp: SDLOG_UTC_OFFSET * 60), never a start time
    return dict(log_id=log_id, log_date=date, duration_s=dur, size_bytes=size, info={"time_ref_utc": utc},
                fifo=fifo, status=status)


def test_fallback_skips_q3_duplicates():
    """D-25: a duplicate upload has the same log date, the same duration and the same file size."""
    assert fb.is_q3_duplicate(row("N"), row("C"))                                   # byte-identical re-upload
    assert not fb.is_q3_duplicate(row("N"), row("D", dur=301))
    assert not fb.is_q3_duplicate(row("N"), row("E", date="2026-05-02"))
    assert not fb.is_q3_duplicate(row("N"), row("F", size=1001))                    # same day and length, other file
    assert not fb.is_q3_duplicate(row("N", utc=10800), row("G", date="2026-04-01", utc=10800))   # same TZ offset only
    assert not fb.is_q3_duplicate(row("N"), row("H", size=None))                    # size unknown: not confirmed


def entry(log_id, date, dur=300, group="g"):
    return dict(log_id=log_id, log_date=date, duration_s=dur, vehicle_uuid=group)


def test_fallback_newest_first_and_capped():
    screened = entry("S", "2026-06-01")
    cands = [entry("S", "2026-06-01"), entry("old1", "2026-01-01"), entry("newer", "2026-07-01"),
             entry("old2", "2026-05-01"), entry("old3", "2026-03-01"), entry("x", "2026-05-15", group="h")]
    order = fb.older_logs(screened, cands, group=lambda e: e["vehicle_uuid"])
    assert [e["log_id"] for e in order] == ["old2", "old3", "old1"]     # older than S only, newest first, same vehicle

    reads = []

    def fake_screen(e):
        reads.append(e["log_id"])
        return {"old2": row("old2", fifo=False), "old3": row("old3", date="2026-03-01"),
                "old1": row("old1", date="2026-01-01")}[e["log_id"]]

    got, tried = fb.find_fallback(row("S", date="2026-06-01"), order, screen=fake_screen, max_reads=10)
    assert got["log_id"] == "old3" and reads == ["old2", "old3"]     # first older FIFO log, stops there
    reads.clear()
    got, tried = fb.find_fallback(row("S", date="2026-06-01"), order, screen=fake_screen, max_reads=1)
    assert got is None and reads == ["old2"]                           # cap of header reads


def test_fallback_retries_network_errors_without_counting_them():
    calls = []

    def flaky(e):
        calls.append(e["log_id"])
        return row(e["log_id"], status="error") if calls.count(e["log_id"]) < 3 else row(e["log_id"], date="2025-01-01")

    got, tried = fb.find_fallback(row("S"), [entry("a", "2025-01-01")], screen=flaky, max_reads=1, retries=3)
    assert got["log_id"] == "a" and calls == ["a", "a", "a"]


def test_fallback_duplicate_is_skipped_in_search():
    def scr(e):
        return {"dup": row("dup"), "ok": row("ok", date="2026-02-01")}[e["log_id"]]

    got, tried = fb.find_fallback(row("S"), [entry("dup", "2026-05-01"), entry("ok", "2026-02-01")], screen=scr)
    assert got["log_id"] == "ok" and [t["reason"] for t in tried] == ["Q3_duplicate", "taken"]


def test_fallback_rejects_a_log_of_another_board():
    def scr(e):
        r = row(e["log_id"], date="2026-01-01")
        r["info"]["sys_uuid"] = {"other": "B", "same": "A"}[e["log_id"]]
        return r

    got, tried = fb.find_fallback(row("S"), [entry("other", "2026-03-01"), entry("same", "2026-01-01")], screen=scr,
                                  same_vehicle=lambda r: r["info"]["sys_uuid"] == "A")
    assert got["log_id"] == "same" and tried[0]["reason"] == "other_vehicle"


def test_fallback_stops_on_persistent_network_error():
    calls = []

    def down(e):
        calls.append(e["log_id"])
        return row(e["log_id"], status="error")

    got, tried = fb.find_fallback(row("S"), [entry("a", "2026-03-01"), entry("b", "2026-01-01")], screen=down, retries=3)
    assert got is None and calls == ["a", "a", "a"] and tried == [dict(log_id="a", reason="network_error")]
    assert fb.search_incomplete(tried)


def test_library_records_replace_only_triggered_vehicles():
    newest = [dict(rec(n_candidates=0, q10=0), vehicle_key="v1", log_id="n1"),          # triggered
              dict(rec(n_candidates=5, q10=2), vehicle_key="v2", log_id="n2"),          # has steady windows
              dict(rec(rej=["Q4_dropout"]), vehicle_key="v3", log_id="n3")]             # log gate
    fbr = {k: dict(rec(n_candidates=4, q10=1), vehicle_key=k, log_id="f" + k[-1]) for k in ("v1", "v2", "v3")}
    out = fb.library_records(newest, fbr)
    assert [r["log_id"] for r in out] == ["f1", "n2", "n3"]
    assert out[0]["fallback_of"] == "n1" and "fallback_of" not in out[1]
