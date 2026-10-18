"""Tier 1 plant priors: evidence from our own FIT-HALF logs (Phase 4 Task 1 steps 2 and 4; THESIS 7.1.1).

    .venv/Scripts/python experiments/phase4/priors_from_logs.py

Reads only fit-half inputs (every path through priors.require_fit_half, every table through require_fit_rows):
- data/spectra/fit_logs.json         the screened log of all 111 fit-half vehicles (rates, gains, firmware);
- data/spectra/fit_windows.csv       the 142 library windows of the 52 usable vehicles (hover RPM);
- data/raw_logs/<fit log id>.ulg     publication interval of vehicle_angular_velocity (true controller rate);
- experiments/phase4/chip_lowpass_table.json   (from chip_lowpass_table.py, fit half) for the ICM chip delay.
Writes priors_from_logs.json next to this file and prints the evidence tables.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import priors as pr                 # noqa: E402

SPEC = os.path.join(ROOT, "data", "spectra")
PX4_V116_RATE_GAINS = dict(P=0.15, I=0.2, D=0.003, K=1.0)   # mc_rate_control_params.c v1.16.0, lines 51-199
CHIP_BAND = (20.0, 100.0)            # plan Task 1 step 4: group delay at 20-100 Hz (near the crossover band)
RMS_TOL = 0.015                      # D-25 profile test: settings within 1.5 % of the best fit error
BIC_TOL = 10.0                       # D-26 (3) / plan step 4


def q(x, qs=(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1)):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return {str(k): round(float(np.quantile(x, k)), 4) for k in qs} if len(x) else None


def raw_log_info(log_id):
    """(median publication rate of vehicle_angular_velocity or None, motor output protocol) of a fit-half raw log."""
    from pyulog import ULog
    path = pr.fit_path(ROOT, "data", "raw_logs", log_id + ".ulg")
    u = ULog(path, ["vehicle_angular_velocity"])
    proto = pr.motor_output_protocol(u.initial_parameters)
    try:
        d = u.get_dataset("vehicle_angular_velocity").data
    except Exception:
        return None, proto
    ts = d["timestamp_sample"] if "timestamp_sample" in d else d["timestamp"]
    return (pr.publication_rate(ts) if len(ts) > 100 else None), proto


def rates_and_gains(logs):
    rows = []
    for r in logs:
        fc, st = r["fc"], r["stats"]
        rate, proto = raw_log_info(r["log_id"])
        g = fc["gains_roll"]
        rows.append(dict(log_id=r["log_id"], vehicle_key=r["vehicle_key"], chip=(fc.get("fifo_chip") or {}).get("name"),
                         version=fc.get("version"), hw=fc.get("hw"), ratemax=fc.get("f_ctrl"),
                         fs_nominal=st.get("fs_nominal"), fs_measured=st.get("fs_measured"),
                         dropout=st.get("dropout_fraction"), fifo_is_control=fc.get("fifo_is_control_gyro"),
                         pub_rate=rate, protocol=proto["protocol"], motor_groups=proto["groups"], gains_roll=g, gains_pitch=fc["gains_pitch"],
                         f_gyro=fc["actual"]["f_gyro"], f_dterm=fc["actual"]["f_dterm"],
                         dnf_esc=fc["actual"]["dnf_esc"], usable=not r["log_rejections"] and bool(r["windows"])))
    return rows


def summarize_rates(rows):
    full = [r for r in rows if r["pub_rate"] and r["ratemax"] and r["pub_rate"] >= 0.5 * r["ratemax"]]
    off = [r for r in full if abs(r["pub_rate"] / r["ratemax"] - 1) > 0.05]
    by_setting = {}
    for r in full:
        key = f"{int(r['ratemax'])}"
        by_setting.setdefault(key, []).append(round(r["pub_rate"], 1))
    return dict(
        n_logs=len(rows),
        ratemax_counts={str(k): v for k, v in sorted(Counter(r["ratemax"] for r in rows).items(), key=lambda kv: str(kv[0]))},
        n_full_rate_logged=len(full),
        measured_rate_by_ratemax={k: dict(n=len(v), quantiles=q(v)) for k, v in sorted(by_setting.items())},
        n_rate_off_by_more_than_5pct=len(off),
        rate_off=[dict(log=r["log_id"][:8], ratemax=r["ratemax"], measured=round(r["pub_rate"], 1),
                       fifo_chip=r["chip"], fifo_is_control=r["fifo_is_control"], hw=r["hw"]) for r in off],
        measured_rate_all=q([r["pub_rate"] for r in full]),
        fifo_nominal_counts={str(round(k)): v for k, v in sorted(Counter(r["fs_nominal"] for r in rows).items())},
        fifo_measured_low_dropout=q([r["fs_measured"] for r in rows if r["dropout"] is not None and r["dropout"] < 0.5]),
    )


def summarize_gains(rows):
    out = {}
    for axis in ("roll", "pitch"):
        g = [r[f"gains_{axis}"] for r in rows]
        out[axis] = {k: q([x[k] for x in g]) for k in ("Kp", "Ki", "Kd")}
    d = PX4_V116_RATE_GAINS
    dflt = sum(abs(r["gains_roll"]["Kp"] - d["P"] * d["K"]) < 1e-6 and abs(r["gains_roll"]["Ki"] - d["I"] * d["K"]) < 1e-6
               and abs(r["gains_roll"]["Kd"] - d["D"] * d["K"]) < 1e-7 for r in rows)
    out["px4_v116_default"] = dict(Kp=d["P"] * d["K"], Ki=d["I"] * d["K"], Kd=d["D"] * d["K"],
                                   n_roll_equal_to_default=int(dflt), n=len(rows))
    out["actual_filters"] = dict(f_gyro=q([r["f_gyro"] for r in rows]), f_dterm=q([r["f_dterm"] for r in rows]),
                                 n_dnf_esc=int(sum(bool(r["dnf_esc"]) for r in rows)))
    return out


def summarize_motor(windows):
    hm = pr.hover_motor_hz(windows)
    mav = windows.groupby("vehicle_key").mav_type.first()
    per = [dict(vehicle_key=v, hover_hz=round(f, 2), mav_type=mav[v]) for v, f in hm.items()]
    return dict(n_vehicles=len(per), n_usable=int(windows.vehicle_key.nunique()), quantiles=q([p["hover_hz"] for p in per]),
                per_vehicle=sorted(per, key=lambda p: p["hover_hz"]))


def summarize_chips(table):
    out = {}
    rows = table["rows"]
    grid_fc = sorted({t[0] for t in table["grid"]})
    for chip in table["chips"]:
        terms = [r for r in rows if r["chip"] == chip and r["term"] is not None]
        best_bic = min(terms, key=lambda r: r["bic"])
        best_rms = min(terms, key=lambda r: r["rms_db_pooled"])
        none = next(r for r in rows if r["chip"] == chip and r["term"] is None)
        res = dict(n_vehicles=table["n_vehicles"][chip], best_term_bic=best_bic["term"],
                   best_term_rms=best_rms["term"], best_rms_db=round(best_rms["rms_db_pooled"], 4),
                   no_term_rms_db=round(none["rms_db_pooled"], 4),
                   best_on_grid_edge=best_bic["term"][0] in (grid_fc[0], grid_fc[-1]),
                   best_on_original_grid_edge=best_bic["term"][0] == 150.0)
        for rule, tol in (("bic", BIC_TOL), ("rms", RMS_TOL)):
            sel = pr.indistinguishable_terms(rows, chip, rule=rule, tol=tol)
            rng = [pr.band_delay_range(fc, n, CHIP_BAND) for fc, n in sel]
            res[rule] = dict(tol=tol, n_terms=len(sel), terms=sorted(sel),
                             delay_ms=[round(1e3 * min(a for a, _ in rng), 3), round(1e3 * max(b for _, b in rng), 3)])
        a, b = pr.band_delay_range(*best_bic["term"], CHIP_BAND)
        res["best_term_delay_ms"] = [round(1e3 * a, 3), round(1e3 * b, 3)]
        out[chip] = res
    return out


def main():
    logs = json.load(open(pr.fit_path(SPEC, "fit_logs.json")))
    windows = pr.load_fit_table(pr.fit_path(SPEC, "fit_windows.csv"))
    table = json.load(open(pr.fit_path(HERE, "chip_lowpass_table.json")))
    rows = rates_and_gains(logs)
    protocols = dict(counts=dict(Counter(r["protocol"] for r in rows)),
                     by_group=dict(Counter(f"{'+'.join(r['motor_groups']) or '-'}:{r['protocol']}" for r in rows)),
                     note="coarse: group-level timer types; MAIN outputs on boards with an IO chip (+5.4 ms, PX4 docs) "
                          "are not resolved")
    out = dict(protocols=protocols, rates=summarize_rates(rows), gains=summarize_gains(rows), gains_usable=summarize_gains([r for r in rows if r["usable"]]),
               hover_motor=summarize_motor(windows), chip_delay=summarize_chips(table),
               settings=dict(chip_band_hz=CHIP_BAND, rms_tol=RMS_TOL, bic_tol=BIC_TOL, px4_v116_rate_gains=PX4_V116_RATE_GAINS))
    with open(os.path.join(HERE, "priors_from_logs.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "hover_motor"}, indent=1)[:12000])
    print("hover motor:", out["hover_motor"]["n_vehicles"], "of", out["hover_motor"]["n_usable"], out["hover_motor"]["quantiles"])


if __name__ == "__main__":
    main()
