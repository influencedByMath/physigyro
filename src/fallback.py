"""Next-newest-log fallback (decision D-24 (5), details P2).

When a vehicle's screened log passed every log gate but has ZERO steady windows (no airborne window passes
Q10, including a vehicle that never flew), the vehicle's older qualifying logs are header-screened newest
first, and the first one that has FIFO and is not a duplicate upload of the screened flight (gate Q3) replaces
it. At most one fallback log and ``max_reads`` header reads per vehicle; network errors are retried and do not
count as reads (a persistent error stops the search, which a rerun retries). Logs rejected by a log gate (dropouts, rate, axes, missing topic) never trigger it.
The rule is the same for both halves; it can only be evaluated on a processed (downloaded) log.
"""
from log_screen import screen_log


def needs_fallback(rec):
    """True if the processed log passed the log gates and no candidate window passed Q10."""
    if rec.get("log_rejections"):
        return False
    steady = rec.get("n_candidates", 0) - rec.get("window_reasons", {}).get("Q10_unsteady", 0)
    return steady == 0


def _utc(r):
    return int((r.get("info") or {}).get("time_ref_utc") or 0)


def is_q3_duplicate(a, b):
    """Gate Q3: same flight uploaded twice. Same GPS start time when both logs have one; otherwise the same
    log date and the same duration (whole seconds, as in the index)."""
    if _utc(a) and _utc(b):
        return _utc(a) == _utc(b)
    return a.get("log_date") == b.get("log_date") and a.get("duration_s") == b.get("duration_s")


def _key(e):                                   # the same ordering key as log_screen.sampling_order
    return (e.get("log_date") or "", e.get("duration_s") or 0, e["log_id"])


def older_logs(screened, candidates, group):
    """Qualifying logs of the screened log's vehicle that are older than it (by the D-21 key), newest first."""
    g, k = group(screened), _key(screened)
    return sorted((e for e in candidates if group(e) == g and _key(e) < k), key=_key, reverse=True)


def find_fallback(screened_row, older, screen=screen_log, max_reads=10, retries=3, same_vehicle=lambda r: True):
    """Header-screen ``older`` (newest first) until a FIFO log that is not a Q3 duplicate is found.
    ``same_vehicle(row)`` must hold for the header (board ID = the vehicle key), else the log is skipped.
    Returns (row or None, tried) where tried lists every header read with its outcome."""
    tried, reads = [], 0
    for e in older:
        if reads >= max_reads:
            break
        for _ in range(retries):
            r = screen(e)
            if r.get("status") != "error":
                break
        if r.get("status") == "error":
            tried.append(dict(log_id=e["log_id"], reason="network_error"))
            return None, tried                    # stop: skipping it could take a later log; retried on a rerun
        reads += 1
        if not r.get("fifo") or r.get("status") != "ok":
            tried.append(dict(log_id=e["log_id"], reason="no_fifo" if r.get("status") == "ok" else r.get("status")))
        elif not same_vehicle(r):
            tried.append(dict(log_id=e["log_id"], reason="other_vehicle"))
        elif is_q3_duplicate(screened_row, r):
            tried.append(dict(log_id=e["log_id"], reason="Q3_duplicate"))
        else:
            tried.append(dict(log_id=e["log_id"], reason="taken"))
            return r, tried
    return None, tried


def search_incomplete(tried):
    """The search stopped on a network error: run it again later (never take a later log instead)."""
    return bool(tried) and tried[-1]["reason"] == "network_error"


def library_records(newest, fallback_by_vehicle):
    """Records used for the library: a vehicle's fallback record replaces its newest-log record only when the
    newest log triggered the fallback (needs_fallback); the newest log id is kept in ``fallback_of``."""
    out = []
    for r in newest:
        fr = fallback_by_vehicle.get(r["vehicle_key"])
        out.append(dict(fr, fallback_of=r["log_id"]) if fr is not None and needs_fallback(r) else r)
    return out
