"""Fig. 12 (E12; THESIS 7.1.1 Tier 2 step 8; D-42): the Tier 2 plant identifications of real drones.

    .venv/Scripts/python experiments/phase4/fig12_tier2.py [out_dir]

Histograms of the existing identifications only (no new identification), the fit half and the test half shown
separately (one row each, the same bins per column):
- tau_m;
- T_d net of the IMU chip filter (the identified T_d minus the lower chip-delay bound of the control chip, or of the
  FIFO chip when the control chip is unknown; f_options.net_td), comparable to the D-36 (3) prior;
- the unfiltered crossover f_c (rq1_cases.unfiltered_crossover), comparable to the 3-10 Hz training prior (D-37 (5));
- the PM of the flown setting (pilot's gains and filters; the Tier 2 realism check), with the nominal 30° floor.
Sets: fit half = the 30 accepted identifications of the evidence pack (a4_followup.json, 17 vehicles); crossover and
PM only for the 28 whose loop parameters did not change in flight (as the pass rates and Task 3). Test half = the 28
Task 3 real-gain cases (test_rq1_cases.json, 19 vehicles; evaluation only, D-37 (2)). Every value of a Task 3 case
equals the value reported in Task 3 (tested); the flown PM uses the window RPM of the Task 3 case, as Task 3 did.
Writes figures/fig12_tier2.{png,pdf} and figures/fig12_caption.md.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import f_options as fo                      # noqa: E402
import priors as pr                         # noqa: E402
import robust as rb                         # noqa: E402
import robust_limits as rl                  # noqa: E402
import rq1_cases as rc                      # noqa: E402
import rq1_run as rr                        # noqa: E402
import tier3 as t3                          # noqa: E402
from loop_model import Gains, Plant, Rates  # noqa: E402
from rq1_report import BLUE, GRID, INK, MUTED, _save, _style   # noqa: E402

KEYS = ("tau_ms", "td_net_ms", "fc_unfiltered", "pm_flown")
LABELS = dict(tau_ms="motor time constant τ_m [ms]", td_net_ms="extra delay T_d, net of the chip filter [ms]",
              fc_unfiltered="crossover f_c, unfiltered [Hz]", pm_flown="phase margin of the flown setting [deg]")
PRIORS = dict(tau_ms=tuple(1e3 * x for x in t3.TAU_M), td_net_ms=tuple(1e3 * x for x in t3.TD),
              fc_unfiltered=t3.F_C)                                    # D-36 (3), (5)
PM_FLOOR = t3.NOMINAL["PM_min"]                                        # D-30
BIN_WIDTH = dict(tau_ms=2.5, td_net_ms=0.5, fc_unfiltered=0.5, pm_flown=5.0)
PRIOR_SHADE = "#d9dde2"


def chip_lower():
    """{chip: lower chip-delay bound [s]} of Task 1 (fit-error rule), as used for the D-36 (3) comparison."""
    return rl.chip_lower_bounds(json.load(open(pr.fit_path(HERE, "priors_from_logs.json"))))


def td_net_ms(case, low):
    return 1e3 * fo.net_td(case, low)


def flown_pm(plant, gains, rates, settings, motor_hz):
    """Nominal PM of the flown setting (the open loop of robust.robust_metrics, the one Task 3 reports)."""
    mh = None if motor_hz is None else np.asarray(motor_hz, float)
    return float(rb.robust_metrics(plant, gains, rates, rr.settings_of(settings), mh, 0.0, 0.0, 0.0)["nominal"]["PM"])


def _row(cid, half, vehicle_key, case, settings, motor_hz, valid, low):
    plant, gains, rates = Plant(**case["plant"]), Gains(**case["gains"]), Rates(float(case["fs"]), float(case["f_ctrl"]))
    r = dict(id=cid, half=half, vehicle_key=vehicle_key, flown_valid=bool(valid), tau_ms=1e3 * case["plant"]["tau_m"],
             td_net_ms=td_net_ms(case, low), fc_unfiltered=None, pm_flown=None,
             case=dict(plant=case["plant"], control_chip=case.get("control_chip"), fifo_chip=case.get("fifo_chip")))
    if valid:
        r["fc_unfiltered"] = rc.unfiltered_crossover(plant, gains, rates)[0]
        r["pm_flown"] = flown_pm(plant, gains, rates, settings, motor_hz)
    return r


def _fit_vehicle_keys():
    keys = {}
    for name in ("fit_logs.json", "fit_fallback_logs.json"):
        for rec in json.load(open(pr.fit_path(ROOT, "data", "spectra", name))):
            if "log_id" in rec and "vehicle_key" in rec:
                keys.setdefault(rec["log_id"], rec["vehicle_key"])
    return keys


def load_rows():
    low = chip_lower()
    rows = []
    t3_fit = {c["id"]: c for c in json.load(open(pr.fit_path(HERE, "fit_rq1_cases.json")))["cases"]}
    vkeys = _fit_vehicle_keys()
    for c in json.load(open(pr.fit_path(HERE, "a4_followup.json")))["cases"]:
        cid = f"fit|{c['log_id'][:8]}|{'roll' if c['axis'] == 0 else 'pitch'}"
        mh = t3_fit[cid]["motor_hz"] if cid in t3_fit else c["motor_hz"]      # window RPM where Task 3 has it
        rows.append(_row(cid, "fit", vkeys[c["log_id"]], c, c["settings"], mh, not c["exclude"], low))
    test = json.load(open(os.path.join(HERE, "test_rq1_cases.json")))            # evaluation only (D-37 (2))
    for c in test["cases"]:
        rows.append(_row(c["id"], "test", c["vehicle_key"], c, c["actual"], c["motor_hz"], True, low))
    return rows


def summary(rows):
    out = {}
    for half in ("fit", "test"):
        out[half] = {}
        for k in KEYS:
            sel = [r for r in rows if r["half"] == half and r[k] is not None]
            v = np.array([r[k] for r in sel])
            d = dict(n=len(v), n_vehicles=len({r["vehicle_key"] for r in sel}), median=float(np.median(v)),
                     min=float(v.min()), max=float(v.max()))
            if k in PRIORS:
                lo, hi = PRIORS[k]
                d["inside_prior"] = int(sum(lo <= x <= hi for x in v))
            else:
                d["below_floor"] = int(sum(x < PM_FLOOR for x in v))
            out[half][k] = d
    return out


def _edges(values, width, extra=()):
    lo = math.floor(min(list(values) + list(extra)) / width) * width
    hi = math.ceil(max(list(values) + list(extra)) / width) * width
    if hi <= lo:
        hi = lo + width
    return np.round(np.arange(lo, hi + width / 2, width), 10)


def make_figure(rows, out_dir):
    """2 rows (fit, test) x 4 columns (KEYS); the same bins per column; the prior shaded, the PM floor dotted, the
    median dashed. Returns the plotted counts and edges."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = summary(rows)
    fig, axes = plt.subplots(2, len(KEYS), figsize=(12.4, 5.6), squeeze=False)
    panels = []
    for j, k in enumerate(KEYS):
        allv = [r[k] for r in rows if r[k] is not None]
        edges = _edges(allv, BIN_WIDTH[k], PRIORS.get(k, (PM_FLOOR,)))
        for i, half in enumerate(("fit", "test")):
            ax = axes[i, j]
            _style(ax)
            v = [r[k] for r in rows if r["half"] == half and r[k] is not None]
            if k in PRIORS:
                ax.axvspan(*PRIORS[k], color=PRIOR_SHADE, alpha=0.6, linewidth=0, zorder=0,
                           label="training prior (D-36)" if (i == 0 and j == 0) else None)
            else:
                ax.axvline(PM_FLOOR, color=INK, linewidth=1.2, linestyle=(0, (1, 1.5)),
                           label=f"nominal PM floor {PM_FLOOR:g}° (D-30)" if i == 0 else None)
            counts, _, _ = ax.hist(v, bins=edges, color=BLUE, edgecolor="white", linewidth=0.6, zorder=2)
            ax.axvline(np.median(v), color=INK, linewidth=1.0, linestyle=(0, (3, 2)), zorder=3,
                       label="median" if (i == 0 and j == 0) else None)
            ax.set_xlim(edges[0], edges[-1])
            ax.yaxis.get_major_locator().set_params(integer=True)
            panels.append(dict(half=half, key=k, edges=edges.tolist(), counts=[int(c) for c in counts]))
            if i == 0:
                ax.set_title(LABELS[k], color=INK, fontsize=9)
            if i == 1:
                ax.set_xlabel(LABELS[k].split(" [")[-1].rstrip("]") if "[" in LABELS[k] else "", color=MUTED, fontsize=8)
            if j == 0:
                d = s[half]["tau_ms"]
                ax.set_ylabel(f"{half} half\n{d['n']} identifications, {d['n_vehicles']} vehicles\ncount",
                              color=MUTED, fontsize=8.5)
    fig.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out_dir, "fig12_tier2")
    order = {(h, k): n for n, (h, k) in enumerate((h, k) for h in ("fit", "test") for k in KEYS)}
    return dict(panels=sorted(panels, key=lambda p: order[(p["half"], p["key"])]))   # row by row


def _q(d, nd=1):
    return f"{d['median']:.{nd}f} ({d['min']:.{nd}f}-{d['max']:.{nd}f})"


def caption(s):
    f, t = s["fit"], s["test"]
    return (
        "**Fig. 12.** Tier 2 plant identifications of real drones (E12, D-42). Top row: fit half, "
        f"{f['tau_ms']['n']} identifications from {f['tau_ms']['n_vehicles']} vehicles (crossover and phase margin: "
        f"{f['pm_flown']['n']} identifications from {f['pm_flown']['n_vehicles']} vehicles; a3f3ae23 is left out there "
        "because its gains changed in flight). Bottom row: test half, "
        f"{t['tau_ms']['n']} cases from {t['tau_ms']['n_vehicles']} vehicles (the Task 3 real-gain cases). "
        "Shaded: the training prior of the synthetic drones (D-36): τ_m 15-60 ms, T_d 0.3-8 ms net of the IMU chip "
        "filter, unfiltered crossover 3-10 Hz. Dashed: median. Dotted: the nominal PM floor of 30° (D-30). "
        f"Inside the prior, fit / test: τ_m {f['tau_ms']['inside_prior']} of {f['tau_ms']['n']} / "
        f"{t['tau_ms']['inside_prior']} of {t['tau_ms']['n']}; T_d {f['td_net_ms']['inside_prior']} of "
        f"{f['td_net_ms']['n']} / {t['td_net_ms']['inside_prior']} of {t['td_net_ms']['n']}; f_c "
        f"{f['fc_unfiltered']['inside_prior']} of {f['fc_unfiltered']['n']} / {t['fc_unfiltered']['inside_prior']} of "
        f"{t['fc_unfiltered']['n']}. Flown PM below 30°: {f['pm_flown']['below_floor']} of {f['pm_flown']['n']} / "
        f"{t['pm_flown']['below_floor']} of {t['pm_flown']['n']}. Median (range), fit / test: τ_m {_q(f['tau_ms'])} / "
        f"{_q(t['tau_ms'])} ms; T_d {_q(f['td_net_ms'], 2)} / {_q(t['td_net_ms'], 2)} ms; f_c {_q(f['fc_unfiltered'], 2)} / "
        f"{_q(t['fc_unfiltered'], 2)} Hz; PM {_q(f['pm_flown'])} / {_q(t['pm_flown'])} deg.\n")


def main(out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "figures")
    rows = load_rows()
    make_figure(rows, out_dir)
    cap = caption(summary(rows))
    with open(os.path.join(out_dir, "fig12_caption.md"), "w", encoding="utf-8") as fh:
        fh.write(cap)
    sys.stdout.reconfigure(encoding="utf-8")
    print(cap)


if __name__ == "__main__":
    main(*(sys.argv[1:2] or [None]))
