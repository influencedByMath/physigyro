"""How many pilot logs and windows survive a threshold set? (projection for the go/no-go, THESIS 7.4.2 H)

Applies the log gates (Q4 dropout, Q5 band, optional L_min) and the window gates (Q10 activity and
throttle, Q13 Welch averages, Q9 clipping) to the per-log results of run_pilot.py, for the THESIS
candidates and for the variants discussed in REPORT.md. Nothing is frozen here.
    .venv/Scripts/python experiments/pilot_study/usable.py
Writes usable.json next to this file.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))

VARIANTS = {
    "THESIS candidates (f_max 800, measured fs, D_max 5 %, L_min 10 s)":
        dict(f_max=800, fs="measured", d_max=0.05, l_min=10.0, sp=15.0, thr=0.05, avg=20),
    "f_max 800 on the nominal rate":
        dict(f_max=800, fs="nominal", d_max=0.05, l_min=10.0, sp=15.0, thr=0.05, avg=20),
    "f_max 780 on the measured rate":
        dict(f_max=780, fs="measured", d_max=0.05, l_min=10.0, sp=15.0, thr=0.05, avg=20),
    "f_max 780, L_min = one Welch segment (window gate Q13 decides)":
        dict(f_max=780, fs="measured", d_max=0.05, l_min=0.0, sp=15.0, thr=0.05, avg=20),
    "f_max 780, L_min = one Welch segment, D_max 20 %":
        dict(f_max=780, fs="measured", d_max=0.20, l_min=0.0, sp=15.0, thr=0.05, avg=20),
    "PROPOSED: f_max 780, L_min one Welch segment, D_max 20 %, throttle 0.02":
        dict(f_max=780, fs="measured", d_max=0.20, l_min=0.0, sp=15.0, thr=0.02, avg=20),
}


def apply(v):
    logs, n_win, reasons = [], 0, {}
    for r in R["logs"]:
        if "stats" not in r:
            reasons["crash"] = reasons.get("crash", 0) + 1
            continue
        s = r["stats"]
        fs = s["fs_measured"] if v["fs"] == "measured" else s["fs_nominal"]
        why = None
        if s["dropout_fraction"] > v["d_max"]:
            why = "Q4_dropout"
        elif 0.8 * fs / 2 < v["f_max"]:
            why = "Q5_rate"
        elif s["longest_segment_s"] < v["l_min"]:
            why = "Q4_no_long_segment"
        if why is None:
            wins = [w for w in r["windows"] if w["sp_std_dps"] <= v["sp"]
                    and (w["thr_std"] is None or w["thr_std"] != w["thr_std"] or w["thr_std"] <= v["thr"])
                    and w["n_averages"] >= v["avg"] and w["clipped"] == 0]
            if not wins:
                why = "no_window"
        if why:
            reasons[why] = reasons.get(why, 0) + 1
        else:
            logs.append(r["vehicle_key"])
            n_win += min(3, len(wins))                       # Q14 cap of 3 windows per vehicle
    return dict(usable_logs=len(logs), of=len(R["logs"]), windows_capped=n_win, removed=reasons)


if __name__ == "__main__":
    out = {name: dict(apply(v), thresholds=v) for name, v in VARIANTS.items()}
    for name, o in out.items():
        print(f"{o['usable_logs']:2d}/{o['of']} logs, {o['windows_capped']:3d} windows | {name} | removed {o['removed']}")
    json.dump(out, open(os.path.join(HERE, "usable.json"), "w"), indent=1)
