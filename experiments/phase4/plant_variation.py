"""Real plant variation from FIT-HALF logs (evidence pack for A2-A4; THESIS 7.1.1 Tier 2, D-14).

    .venv/Scripts/python experiments/phase4/plant_variation.py

For every fit-half log whose logger profile writes the controller signals at full rate (high-rate or sysid,
priors.full_rate_logged) and that has vehicle_torque_setpoint at >= 200 Hz:
- pieces on the controller grid (sysid.uniform_pieces: up to 8 skipped publications held, longer gaps split);
- airborne time with 5 s trimmed at take-off and landing (D-23 Q10 T_edge);
- the instrument by flight mode (D-14, sysid.instrument_for_mode): attitude setpoint in Stabilized/Altitude,
  rate setpoint in Acro, both (must agree within 1 dB / 5 deg, the D-14 candidate) in Position/auto/offboard;
- body rate = raw FIFO gyro, zero-phase low-passed at 100 Hz, on the controller grid (sysid.body_rate_on_grid);
- Welch spectra over all pieces (1.28 s segments), FRF, fit of (b, tau_m, Td) at 0.5-20 Hz where coherence >= 0.5
  (D-14), at least 6 frequencies;
- repeated identification: the same fit on the first and the second half of the usable time (split by time).
Roll and pitch separately. Writes plant_variation.json next to this file. Nothing reads the test half.
"""
import json
import os
import sys
import time

import numpy as np
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import noise_data as nd             # noqa: E402
import priors as pr                 # noqa: E402
import sysid                        # noqa: E402

T_EDGE = 5.0              # D-23 Q10
F_BAND = (0.5, 20.0)      # D-14 / feasibility
MIN_COH = 0.5             # D-14
SEG_S = 1.28              # Welch segment length (1024 samples at 800 Hz, as in the feasibility test): the D-14 rule of
                          # 6 coherent frequencies was set at this 0.78 Hz resolution (a 0.8 s segment lost 9 fits)
MAX_HOLD = 8              # skipped publications held (the motor command holds its value). 8, not 3: many logs pause
                          # the torque setpoint AND the controller input vehicle_angular_velocity for 5-6 periods about
                          # once per second (independent review, Important 1; fit-half probe: 79 of 83 and 227 of 232
                          # pauses in 852b8368 / d1af0dea, mostly without a logger-dropout record)
F_CUT = 100.0             # zero-phase low-pass of the raw gyro before resampling
MIN_TQ_RATE = 200.0
STALE_S = 0.1             # an instrument sample older than this is a data gap
AGREE_DB, AGREE_DEG = 1.0, 5.0   # D-14 candidate


def _zoh(t_src, x_src, tg):
    k = np.searchsorted(t_src, tg, side="right") - 1
    ok = (k >= 0) & (tg - t_src[np.clip(k, 0, None)] <= STALE_S)
    return np.where(ok, x_src[np.clip(k, 0, None)], np.nan)


def build_pieces(u, gs, axis):
    """Usable pieces of one log and axis, grouped by instrument class; also returns the controller period."""
    present = {d.name for d in u.data_list}
    for name in ("vehicle_torque_setpoint", "vehicle_status", "vehicle_rates_setpoint", "vehicle_land_detected"):
        if name not in present:
            raise ValueError(f"missing topic: {name}")
    q = u.get_dataset("vehicle_torque_setpoint").data
    tq = nd._t(u, q["timestamp_sample"] if "timestamp_sample" in q else q["timestamp"])
    Tc = float(np.median(np.diff(tq)))
    if 1.0 / Tc < MIN_TQ_RATE:
        raise ValueError(f"torque setpoint logged at {1 / Tc:.0f} Hz")
    ux = q[f"xyz[{axis}]"].astype(float)
    air = [(a + T_EDGE, b - T_EDGE) for a, b in nd.airborne_intervals(u) if b - a > 2 * T_EDGE]
    st = u.get_dataset("vehicle_status").data
    ts_st, nav = nd._t(u, st["timestamp"]), st["nav_state"].astype(int)
    rs = u.get_dataset("vehicle_rates_setpoint").data
    r_rate_t, r_rate = nd._t(u, rs["timestamp"]), rs[["roll", "pitch", "yaw"][axis]].astype(float)
    try:
        r_att_t, r_att = sysid._attitude_setpoint_angle(u, axis)
    except Exception:
        r_att_t, r_att = None, None
    nseg = int(round(SEG_S / Tc))
    nseg += nseg % 2
    out = {"att_sp": [], "rate_sp": [], "both": []}
    for tg, idx in sysid.uniform_pieces(tq, Tc, max_hold=MAX_HOLD):
        inair = np.zeros(len(tg), bool)
        for a, b in air:
            inair |= (tg >= a) & (tg <= b)
        if not inair.any():
            continue
        k = np.clip(np.searchsorted(ts_st, tg, side="right") - 1, 0, len(ts_st) - 1)
        cls = np.array([sysid.instrument_for_mode(n) for n in nav[k]])
        w = sysid.body_rate_on_grid(gs, axis, tg, f_cut=F_CUT)
        rr = _zoh(r_rate_t, r_rate, tg)
        ra = _zoh(r_att_t, r_att, tg) if r_att is not None else np.full(len(tg), np.nan)
        base = inair & np.isfinite(w)
        for c in out:
            need = {"att_sp": np.isfinite(ra), "rate_sp": np.isfinite(rr), "both": np.isfinite(ra) & np.isfinite(rr)}[c]
            m = base & (cls == c) & need
            edges = np.flatnonzero(np.diff(np.r_[0, m.astype(int), 0]))
            for s0, s1 in zip(edges[::2], edges[1::2]):
                if s1 - s0 >= nseg:
                    out[c].append(dict(t=tg[s0:s1], u=ux[idx[s0:s1]], w=w[s0:s1], att_sp=ra[s0:s1], rate_sp=rr[s0:s1]))
    return out, Tc, nseg


def halves(pieces):
    """Split pieces (time order) at the middle of their total length."""
    n = sum(len(p["t"]) for p in pieces)
    first, second, acc = [], [], 0
    for p in pieces:
        L = len(p["t"])
        if acc + L <= n // 2:
            first.append(p)
        elif acc >= n // 2:
            second.append(p)
        else:
            cut = n // 2 - acc
            first.append({k: v[:cut] for k, v in p.items()})
            second.append({k: v[cut:] for k, v in p.items()})
        acc += L
    return first, second


def fit_pieces(pieces, Tc, nseg, cls, f_band=F_BAND, min_coh=MIN_COH):
    """D-14 fit of one class; returns a dict (error string when it fails). nseg is the Welch segment length;
    f_band and min_coh (defaults: D-14) apply to the fit and to the instrument-agreement check alike."""
    instr = ["att_sp", "rate_sp"] if cls == "both" else [cls]
    frfs = {}
    for name in instr:
        S = sysid.piece_spectra([dict(r=p[name], u=p["u"], w=p["w"]) for p in pieces], fs=1 / Tc, nperseg=nseg)
        frfs[name] = sysid.frf_from_spectra(S) + (S["n_seg"],)
    res = dict(seconds=round(sum(len(p["t"]) for p in pieces) * Tc, 1), n_seg=frfs[instr[0]][3])
    if cls == "both":
        f, G1, c1, _ = frfs["att_sp"]
        _, G2, c2, _ = frfs["rate_sp"]
        ok, db, deg, n = sysid.instruments_agree(f, G1, c1, G2, c2, f_band, min_coh, AGREE_DB, AGREE_DEG)
        res["agree"] = dict(ok=ok, rms_db=None if np.isnan(db) else round(db, 2),
                            rms_deg=None if np.isnan(deg) else round(deg, 2), n=n)
    f, G, coh, _ = frfs[instr[0]]
    try:
        fit = sysid.fit_frf(f, G, coh, Tc, f_band=f_band, min_coh=min_coh)
    except ValueError as e:
        res["error"] = str(e)
        return res
    res.update(b=float(fit["b"]), tau_m=float(fit["tau_m"]), Td=float(fit["Td"]), n_freq=fit["n_freq"],
               f_used=[round(float(x), 2) for x in fit["f_used"]], mag_err_db=round(fit["mag_err_db"], 3),
               phase_err_deg=round(fit["phase_err_deg"], 3),
               at_bound=dict(tau_m=bool(fit["tau_m"] <= 0.00301 or fit["tau_m"] >= 0.499), Td_zero=bool(fit["Td"] <= 1e-5),
                             Td_max=bool(fit["Td"] >= 0.0299)))
    return res


def accept(fit, cls, max_mag_db=3.0, max_phase_deg=15.0):
    """Provisional acceptance of one identification (evidence pack; the author decides the final gates):
    a fit exists (>= 6 coherent frequencies, D-14), both instruments agree in Position/auto modes (D-14),
    fit error <= max_mag_db / max_phase_deg (feasibility fits: 1.5-1.9 dB, 3.8-6 deg), tau_m not on a fit bound,
    Td not on its 30 ms bound (Td = 0 is allowed: the band ends at 20 Hz, so small delays are not resolved)."""
    if "b" not in fit:
        return False, ["no fit"]
    bad = []
    if cls == "both" and not (fit.get("agree") or {}).get("ok", False):
        bad.append("instruments_disagree")
    if fit["mag_err_db"] > max_mag_db:
        bad.append("mag_err")
    if fit["phase_err_deg"] > max_phase_deg:
        bad.append("phase_err")
    if fit["at_bound"]["tau_m"]:
        bad.append("tau_m_bound")
    if fit["at_bound"]["Td_max"]:
        bad.append("Td_max")
    return not bad, bad


def analyze(rec):
    t_start = time.perf_counter()
    out = _analyze(rec)
    for r in out:
        r["log_seconds"] = round(time.perf_counter() - t_start, 1)
    return out


def _analyze(rec):
    out = []
    path = pr.fit_path(ROOT, "data", "raw_logs", rec["log_id"] + ".ulg")
    base = dict(log_id=rec["log_id"], vehicle_key=rec["vehicle_key"], hw=rec["fc"].get("hw"),
                fifo_chip=(rec["fc"].get("fifo_chip") or {}).get("name"),
                control_chip=(rec["fc"].get("control_chip") or {}).get("name"),
                fifo_is_control=rec["fc"].get("fifo_is_control_gyro"), usable_noise=not rec["log_rejections"])
    try:
        u = nd.load(path)
        gs = nd.gyro_stream(u)
    except Exception as e:
        return [dict(base, error=f"load: {e!r}"[:200])]
    for axis in (0, 1):
        try:
            groups, Tc, nseg = build_pieces(u, gs, axis)
        except Exception as e:
            out.append(dict(base, axis=axis, error=f"pieces: {e!r}"[:200]))
            continue
        if not any(groups.values()):
            out.append(dict(base, axis=axis, error=f"no airborne gap-free piece as long as one Welch segment ({SEG_S} s)"))
            continue
        for cls, pieces in groups.items():
            if not pieces:
                continue
            r = dict(base, axis=axis, cls=cls, f_ctrl=round(1 / Tc, 2))
            try:
                r["full"] = fit_pieces(pieces, Tc, nseg, cls)
                h1, h2 = halves(pieces)
                r["half1"] = fit_pieces(h1, Tc, nseg, cls) if h1 else dict(error="empty")
                r["half2"] = fit_pieces(h2, Tc, nseg, cls) if h2 else dict(error="empty")
            except ValueError as e:
                r["error"] = str(e)[:200]
            out.append(r)
    return out


def main():
    t0 = time.perf_counter()
    logs = json.load(open(pr.fit_path(ROOT, "data", "spectra", "fit_logs.json")))
    todo = [r for r in logs if pr.full_rate_logged(r["fc"])]
    rows = Parallel(n_jobs=4, verbose=5)(delayed(analyze)(r) for r in todo)
    rows = [x for group in rows for x in group]
    out = dict(settings=dict(T_EDGE=T_EDGE, F_BAND=F_BAND, MIN_COH=MIN_COH, SEG_S=SEG_S, MAX_HOLD=MAX_HOLD, F_CUT=F_CUT,
                             MIN_TQ_RATE=MIN_TQ_RATE, STALE_S=STALE_S, AGREE_DB=AGREE_DB, AGREE_DEG=AGREE_DEG),
               n_logs=len(todo), rows=rows, seconds=round(time.perf_counter() - t0, 1))
    with open(os.path.join(HERE, "plant_variation.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    n_fit = sum(1 for r in rows if "full" in r and "b" in r["full"])
    print(f"{len(todo)} logs, {len(rows)} rows, {n_fit} full fits, {out['seconds']} s")


if __name__ == "__main__":
    main()
