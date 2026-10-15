"""Independent re-check of every number in REPORT.md Section 2 before the thresholds are frozen (D-23).

Reads only results.json (written by run_pilot.py) and recomputes each claim from the per-log and
per-window records, without using summarize() or usable.py. Prints CLAIM / VALUE / OK lines.
    .venv/Scripts/python experiments/pilot_study/verify_numbers.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
LOGS = [r for r in R["logs"] if "stats" in r]
BAD = []


def check(claim, value, ok):
    print(f"{'OK  ' if ok else 'FAIL'} | {claim} | {value}")
    if not ok:
        BAD.append(claim)


fs = np.array([r["stats"]["fs_measured"] for r in LOGS])
nominal = np.array([r["stats"]["fs_nominal"] for r in LOGS])
check("14 of 29 pilot logs are 2 kHz IMUs (nominal rate)", int((nominal == 2000).sum()),
      (nominal == 2000).sum() == 14 and len(LOGS) == 29)
reliable = np.array([r["stats"]["dropout_fraction"] <= 0.20 for r in LOGS])   # rate re-checked: fs_check
two = fs[(nominal == 2000) & reliable]
check("8 of them pass D_max 20 % (measured rate reliable there)", len(two), len(two) == 8)
check("their clocks 1,994.0-2,000.5 Hz", (round(two.min(), 1), round(two.max(), 1)),
      round(two.min(), 1) == 1994.0 and round(two.max(), 1) == 2000.5)
check("-0.30 % to +0.03 %", (round(100 * (two.min() / 2000 - 1), 2), round(100 * (two.max() / 2000 - 1), 2)),
      round(100 * (two.min() / 2000 - 1), 2) == -0.30 and round(100 * (two.max() / 2000 - 1), 2) == 0.03)
check("f_max 800 would drop 7 of these 8 logs", int(np.sum(0.8 * two / 2 < 800)), np.sum(0.8 * two / 2 < 800) == 7)
for fmax, want in ((800, 58.6), (790, 89.7), (780, 89.7)):
    share = 100 * np.mean(0.8 * fs / 2 >= fmax)
    check(f"{want} % of logs pass at f_max {fmax}", round(share, 1), round(share, 1) == want)
check("780 Hz leaves margin for clocks 2.5 % slow", 0.8 * 2000 * 0.975 / 2, 0.8 * 2000 * 0.975 / 2 >= 780)

h3 = max(3 * max(w["rpm_hz"]) for r in LOGS for w in r["windows"] if w.get("rpm_valid") and w.get("rpm_hz"))
check("motor 3rd harmonics reach at most 428 Hz", round(h3, 1), round(h3) == 428)


def good_windows(r, sp=15.0, thr=0.05, avg=20):
    return [w for w in r["windows"] if w["sp_std_dps"] <= sp
            and (w["thr_std"] is None or w["thr_std"] != w["thr_std"] or w["thr_std"] <= thr)
            and w["n_averages"] >= avg and w["clipped"] == 0]


cases = sorted((r["stats"]["dropout_fraction"], len(good_windows(r)), r["stats"]["longest_segment_s"])
               for r in LOGS if 0.04 < r["stats"]["dropout_fraction"] < 0.05)
check("two logs 4.1 % / 4.4 % dropout, 7 and 6 good windows, longest piece 8.4 / 6.5 s",
      [(round(100 * d, 1), n, round(s, 1)) for d, n, s in cases],
      [(round(100 * d, 1), n, round(s, 1)) for d, n, s in cases] == [(4.1, 7, 8.4), (4.4, 6, 6.5)])

g = R["gap_lengths_s"]
check("median gap 0.5 ms", round(1000 * g["median"], 2), round(1000 * g["median"], 1) == 0.5)
ngaps = max(r["gaps"]["n"] for r in LOGS if r.get("gaps"))
check("up to ~60,000 gaps per log", ngaps, 55000 <= ngaps <= 65000)
e1 = R["dropout_experiment"]["0.01"]
check("1 % spread dropout leaves about half of windows below 20 averages",
      f"{e1['few_averages']}/{e1['n_trials']}", 0.45 <= e1["few_averages"] / e1["n_trials"] <= 0.55)
mid = [(round(100 * r["stats"]["dropout_fraction"], 2), len(good_windows(r)))
       for r in LOGS if 0.05 < r["stats"]["dropout_fraction"] <= 0.2]
check("logs up to 13.8 % dropout had 5-8 good windows", mid, all(5 <= n <= 8 for _, n in mid)
      and max(d for d, _ in mid) == 13.75)
hi = [len(good_windows(r)) for r in LOGS if r["stats"]["dropout_fraction"] >= 0.2]
lo_hi = min(r["stats"]["dropout_fraction"] for r in LOGS if r["stats"]["dropout_fraction"] >= 0.2)
check("from 23 % on no log had any good window", (round(100 * lo_hi, 1), max(hi)), max(hi) == 0
      and round(100 * lo_hi) == 23)

wl = R["window_length"]
check("10 s vs 20 s: 0.49 dB median, 1.7 dB p90", (wl["10"]["median_abs_db"], wl["10"]["p90_abs_db"]),
      round(wl["10"]["median_abs_db"], 2) == 0.49 and round(wl["10"]["p90_abs_db"], 1) == 1.7)
check("5 s vs 20 s: 0.74 / 2.8 dB", (wl["5"]["median_abs_db"], wl["5"]["p90_abs_db"]),
      round(wl["5"]["median_abs_db"], 2) == 0.74 and round(wl["5"]["p90_abs_db"], 1) == 2.8)

avgs = [w["n_averages"] for r in LOGS if r["stats"]["dropout_fraction"] < 0.01 for w in r["windows"]]
check("10 s windows of logs < 1 % dropout give 24-29 averages (at least 20)", (min(avgs), max(avgs)),
      min(avgs) >= 20 and max(avgs) == 29)
for f in (2000, 6400, 8000):
    n = round(f / 1.5)
    check(f"nperseg at {f} Hz / 1.5 Hz = {n}; 10 s holds {2 * (10 * f // n) - 1} 50 %-overlap segments",
          2 * (10 * f // n) - 1, 2 * (10 * f // n) - 1 >= 20)

act = [a for r in LOGS for a in r["activity"]]
sp = np.array([a["sp"] for a in act])
check("3,706 airborne windows", len(act), len(act) == 3706)
check("median 6.7 deg/s, p75 15.9 deg/s", (round(np.median(sp), 1), round(np.percentile(sp, 75), 1)),
      round(np.median(sp), 1) == 6.7 and round(np.percentile(sp, 75), 1) == 15.9)
check("15 deg/s keeps 74 %", round(100 * np.mean(sp <= 15), 1), round(100 * np.mean(sp <= 15)) == 74)
thr = np.array([np.nan if a["thr"] is None else a["thr"] for a in act])
calm = ~np.isnan(thr) & (sp <= 15)
for lim, want, want_all in ((0.02, 9.5, 6.9), (0.05, 2.0, 1.4)):
    rem = 100 * np.mean(thr[calm] > lim)
    rem_all = 100 * np.sum(thr[calm] > lim) / np.sum(sp <= 15)
    check(f"throttle {lim} removes {want} % of calm windows with throttle data ({want_all} % of all calm windows)",
          (round(rem, 1), round(rem_all, 1)), round(rem, 1) == want and round(rem_all, 1) == want_all)
check("947 windows without actuator_motors", int(np.isnan(thr).sum()), int(np.isnan(thr).sum()) == 947)

check("no FIFO dt rate change in any log", sum(r["stats"]["n_rate_changes"] for r in LOGS),
      all(r["stats"]["n_rate_changes"] == 0 and r["stats"]["jitter"] <= 0.01 for r in LOGS))
check("no clipped sample", sum(r["stats"]["clipped_samples"] for r in LOGS),
      all(r["stats"]["clipped_samples"] == 0 for r in LOGS))
rates = sorted(round(r["fc"]["esc_rate_hz"], 1) for r in LOGS if r["fc"].get("has_rpm_topic"))
check("logged RPM rates 4, 4.2, 10 / 200, 200, 399 Hz; 50 Hz separates them", rates,
      max(x for x in rates if x < 50) <= 10 and min(x for x in rates if x >= 50) >= 200)
per = [len(good_windows(r)) for r in LOGS if good_windows(r)]
check("good logs give 1-8 non-overlapping windows", (min(per), max(per)), min(per) == 1 and max(per) == 8)


def usable(d_max, l_min, f_max, thr_lim):
    n, w = 0, 0
    for r in LOGS:
        s = r["stats"]
        if s["dropout_fraction"] > d_max or 0.8 * s["fs_measured"] / 2 < f_max or s["longest_segment_s"] < l_min:
            continue
        gw = good_windows(r, thr=thr_lim)
        if gw:
            n, w = n + 1, w + min(3, len(gw))
    return n, w


for args, want in (((0.05, 10, 800, 0.05), (6, 16)), ((0.05, 10, 780, 0.05), (11, 28)),
                   ((0.05, 0, 780, 0.05), (13, 34)), ((0.20, 0, 780, 0.05), (15, 40)),
                   ((0.20, 0, 780, 0.02), (14, 38))):
    got = usable(*args)
    check(f"usable logs/windows with (D_max, L_min, f_max, thr) = {args}: {want}", got, got == want)

print("\nALL CLAIMS VERIFIED" if not BAD else f"\n{len(BAD)} CLAIM(S) FAILED")
