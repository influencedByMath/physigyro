"""Tables for notes/followup_A4_F1_F4.md (the author's questions of 2026-10-18), generated from a4_followup.json,
f_options.json and robust_limits.json, so that no number in the notes is copied by hand.

    .venv/Scripts/python experiments/phase4/followup_tables.py

Writes followup_tables.md next to this file.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import priors as pr     # noqa: E402

CELLS = ("w0.2_d0.5", "w0.2_d1", "w0.15_d1", "w0.1_d1")
VARIANT_ORDER = ("V0", "V1", "V2", "V3", "V4")


def pct(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f} %"


def worst_corner(corners):
    """The plant with the largest Ms (corners: robust.corner_metrics rows)."""
    return max(corners, key=lambda c: c["Ms"])


def worst_without_delay_step(corners):
    """Largest Ms over the corners without the delay step (labels ending in ' Td')."""
    return max(c["Ms"] for c in corners if c["label"].endswith(" Td"))


def peak_comparison(rows, f_peak, min_coh=0.5, span=0.3):
    """Measured vs model sensitivity at the Welch frequency nearest f_peak, and the largest measured sensitivity over
    the points with coherence >= min_coh within +-span of f_peak (None if there is none)."""
    near = min(rows, key=lambda r: abs(r["f"] - f_peak))
    coh = [r["S_meas"] for r in rows if r["coh"] >= min_coh and abs(r["f"] - f_peak) <= span * f_peak]
    return dict(f=near["f"], coh=near["coh"], S_meas=near["S_meas"], S_model=near["S_model"],
                max_S_meas_coherent=max(coh) if coh else None)


def variant_summary(q2, variants=VARIANT_ORDER):
    """Per identification variant: accepted, nominal passes, passes; failing cases (group 'fails worst case') that
    pass now; identifications lost and passes turned into failures relative to V0."""
    out = {}
    for k in variants:
        acc = [r for r in q2 if r["variants"][k].get("accepted")]
        out[k] = dict(accepted=len(acc), nominal_ok=sum(r["variants"][k]["nominal_ok"] for r in acc),
                      passes=sum(r["variants"][k]["passes"] for r in acc),
                      failing_now_pass=sum(1 for r in acc if r["group"] == "fails worst case" and r["variants"][k]["passes"]),
                      lost=sum(1 for r in q2 if r["variants"]["V0"].get("accepted") and not r["variants"][k].get("accepted")),
                      pass_to_fail=sum(1 for r in acc if r["variants"]["V0"].get("passes") and not r["variants"][k]["passes"]))
    return out


# ------------------------------------------------------------------ sections

def section_q1(a, rl):
    by = {(c["log"], c["axis"]): c for c in rl["real"]["cases"]}
    L = ["## Q1. The real flown settings that pass nominally but fail the worst case (±20 % / +1 sample, M_s,wc 3)", "",
         "| Case | f_ctrl [Hz] | Nominal PM / M_s / GM / f_c | Worst plant | Worst PM / M_s / GM | M_s peak nominal / worst [Hz] "
         "| Coherent band ≥ 0.5 (≥ 0.3 to 40 Hz) [Hz] | Peak inside the band? | Measured / model sensitivity at the peak "
         "(coherence) |", "|---|---|---|---|---|---|---|---|---|"]
    for q in a["q1"]:
        n, w = q["corners"][0], worst_corner(q["corners"])
        c05, c03 = q["coherent_05"], q["coherent_03_to_40"]
        top05, top03 = (max(c05) if c05 else None), (max(c03) if c03 else None)
        inside = top05 is not None and top05 >= w["f_Ms"] and top05 >= n["f_Ms"]
        p = peak_comparison(q["measured_vs_model"], n["f_Ms"])
        wm = by[(q["log"], q["axis"])]["worst"]["w0.2_d1"]
        L.append(f"| {q['log']} {q['axis']} | {q['f_ctrl']:.0f} | {n['PM']:.1f}° / {n['Ms']:.2f} / {n['GM']:.1f} dB / "
                 f"{n['fc']:.1f} Hz | {w['label']} | {wm['PM']:.1f}° / {wm['Ms']:.2f} / {wm['GM']:.2f} dB | "
                 f"{n['f_Ms']:.1f} / {w['f_Ms']:.1f} | {min(c05):.1f}-{top05:.1f} ({top03:.1f}) | "
                 f"{'yes' if inside else 'no: extrapolated'} | {p['S_meas']:.2f} / {p['S_model']:.2f} at {p['f']:.1f} Hz "
                 f"({p['coh']:.2f}) |")
    L += ["", "Worst-case M_s of the same five under other uncertainty sets (robust_limits.json; ✓ = passes M_s,wc 3 with "
          "its PM and GM), and the largest M_s of the corners WITHOUT the delay step:", "",
          "| Case | ±20 % +0.5 | ±20 % +1 | ±15 % +1 | ±10 % +1 | ±20 %, no delay step |", "|---|---|---|---|---|---|"]
    wc3 = rl["settings"]["wc_limits"]["3.0"]
    for q in a["q1"]:
        c = by[(q["log"], q["axis"])]
        cells = []
        for k in CELLS:
            w = c["worst"][k]
            ok = w["stable"] and w["PM"] >= wc3["PM"] and w["Ms"] <= 3.0 and w["GM"] >= wc3["GM"]
            cells.append(f"{w['Ms']:.2f} {'✓' if ok else '✗'}")
        L.append(f"| {q['log']} {q['axis']} | " + " | ".join(cells) + f" | {worst_without_delay_step(q['corners']):.2f} |")
    return "\n".join(L)


def section_q2(a):
    V = a["settings"]["ID_VARIANTS"]
    s = variant_summary(a["q2"])
    L = ["## Q2. Wider band, relaxed coherence, longer segments (all 28 included cases, provisional gates)", "",
         "| Variant | Settings | Accepted (of 28) | Nominal passes | Pass (separate, M_s,wc 3) | Of the 5 failing: pass now "
         "| Identifications lost vs V0 | Passes turned into failures |", "|---|---|---|---|---|---|---|---|"]
    for k in VARIANT_ORDER:
        x = s[k]
        L.append(f"| {k} | {V[k]['label']} | {x['accepted']} | {x['nominal_ok']} | {x['passes']} | {x['failing_now_pass']} "
                 f"| {x['lost']} | {x['pass_to_fail']} |")
    L += ["", "The five failing cases under each variant (worst-case M_s; ✓ pass, ✗ fail; band = highest fitted frequency):", "",
          "| Case | " + " | ".join(VARIANT_ORDER) + " |", "|---|" + "---|" * len(VARIANT_ORDER)]
    for r in a["q2"]:
        if r["group"] != "fails worst case":
            continue
        cells = []
        for k in VARIANT_ORDER:
            v = r["variants"][k]
            if not v.get("accepted"):
                cells.append("rejected: " + ", ".join(v.get("reasons", [])))
            else:
                cells.append(f"{v['worst']['Ms']:.2f} {'✓' if v['passes'] else '✗'} (band {v['fit']['f_used'][1]:.1f}, "
                             f"τ_m {1e3 * v['fit']['tau_m']:.1f} ms, T_d {1e3 * v['fit']['Td']:.2f} ms)")
        L.append(f"| {r['log']} {r['axis']} (peak {r['f_Ms_nominal']:.1f} Hz) | " + " | ".join(cells) + " |")
    lost = sorted({(r["log"], r["axis"]) for r in a["q2"] for k in VARIANT_ORDER
                   if r["variants"]["V0"].get("accepted") and not r["variants"][k].get("accepted")})
    L += ["", "Identifications lost in at least one variant (fit gates: magnitude or phase error, τ_m on a bound): "
          + ", ".join(f"{x[0]} {x[1]}" for x in lost) + "."]
    return "\n".join(L)


def section_q3(a):
    L = ["## Q3. Pass rates for each worst-case limit M_s,wc (nominal PM ≥ 30°, M_s ≤ 2.0, GM ≥ 6 dB always)", "",
         "| M_s,wc | Implied PM / GM | Real ±20 % +1 (of 23 nominal passes) | Real ±15 % +1 | Synthetic ±20 % +1 (of 498) "
         "| Synthetic ±15 % +1 |", "|---|---|---|---|---|---|"]
    q = a["q3"]
    for m in q["real"]:
        r, r15, sy, sy15 = q["real"][m], q["real_w015"][m], q["synthetic"][m], q["synthetic_w015"][m]
        L.append(f"| {float(m):g} | {r['PM']:.1f}° / {r['GM']:.2f} dB | {r['n_pass']} ({pct(r['share_of_nominal'], 0)}) | "
                 f"{r15['n_pass']} ({pct(r15['share_of_nominal'], 0)}) | {sy['n_pass']} ({pct(sy['share_of_nominal'], 1)}) | "
                 f"{sy15['n_pass']} ({pct(sy15['share_of_nominal'], 1)}) |")
    return "\n".join(L)


def section_q4(a):
    L = ["## Q4. Can a filter setting make the failing flown settings pass? (grid 24 × 18 over f_gyro 15-300 Hz × "
         "f_dterm 10-200 Hz, ≤ 0.45 × filter rate; ±20 % / +1 sample, M_s,wc 3; stability only, noise not checked)", "",
         "| Case | Why it fails | Flown f_gyro / f_dterm [Hz] | Flown nominal M_s, worst M_s | Passing settings, notches as flown "
         "| Passing settings, notches off | Best setting: nominal PM / M_s, worst M_s |", "|---|---|---|---|---|---|---|"]
    for q in a["q4"]:
        fl, o1, o2 = q["flown"], q["options"]["notches as flown"], q["options"]["all notches off"]
        b = o1["best"]
        L.append(f"| {q['log']} {q['axis']} | {q['group']} | {fl['f_gyro']:g} / {fl['f_dterm']:g} | "
                 f"{fl['nominal']['Ms']:.2f}, {fl['worst']['Ms']:.2f} | {o1['n_pass']} of {o1['n_grid']} | "
                 f"{o2['n_pass']} of {o2['n_grid']} | {b['f_gyro']:.0f} / {b['f_dterm']:.0f} Hz: {b['nominal']['PM']:.1f}° / "
                 f"{b['nominal']['Ms']:.2f}, {b['worst']['Ms']:.2f} |")
    return "\n".join(L)


def _hist_row(h):
    return " | ".join(str(v) for v in h.values())


def section_f1(f):
    P = f["settings"]["PRIORS"]
    L = ["## F1. Plant priors (synthetic drones at the PX4 defaults; Tier 3a design of THESIS: PM window 45-75°, "
         "integral corner 0.1)", "",
         f"Coverage of the {f['coverage']['P0']['n']} identified real plants (T_d net of the chip filter, `f_options.net_td`):", "",
         "| Option | Prior | Plants inside the box | Plants faster than the fastest prior plant (lag at 10 Hz) | Slower than the slowest "
         "| Nominal pass at defaults | Robust pass (M_s,wc 3) | Kept drones with f_c 8-10 Hz / ≥ 10 Hz |",
         "|---|---|---|---|---|---|---|---|"]
    for k in P:
        c, p = f["coverage"][k], f["populations"][f"{k}+D0"]
        hn = p["hist_nominal"]
        L.append(f"| {k} | {P[k]['label']} | {c['in_box']} | {c['faster_than_prior']} | {c['slower_than_prior']} | "
                 f"{pct(p['nominal_share'])} | {pct(p['robust_share'])} | {hn['8-10']} / {hn['10-12.5'] + hn['12.5-15']} |")
    L += ["", "Identified plants (T_d of the loop and net of the chip filter):", "", "| Plant | Chip | τ_m [ms] | T_d [ms] | T_d net [ms] |",
          "|---|---|---|---|---|"]
    for x in f["identified"]:
        L.append(f"| {x['log']} {'roll' if x['axis'] == 0 else 'pitch'} | {x['chip']} | {1e3 * x['tau_m']:.1f} | "
                 f"{1e3 * x['Td']:.2f} | {1e3 * x['Td_net']:.2f} |")
    return "\n".join(L)


def section_f2_f3(f):
    D = f["settings"]["DESIGNS"]
    P = f["populations"]
    bins = list(P["P0+D0"]["hist_generated"])
    L = ["## F2 and F3. Tier 3a gain design and the crossover distribution at the PX4 defaults (priors D-28 unless named)", "",
         "| Population | Design | Tries per kept drone | Nominal pass at defaults | Robust pass (M_s,wc 3) | Unfiltered PM "
         "p10 / median / p90 [°] |", "|---|---|---|---|---|---|"]
    names = [("P0+D0", D["D0"]["label"])] + [(f"P0+{k}", v["label"]) for k, v in D.items() if k != "D0"]
    names += [(f"{a}+{b}", f"{f['settings']['PRIORS'][a]['label']}; {D[b]['label']}") for a, b in f["settings"]["COMBOS"].values()]
    for k, lab in names:
        p = P[k]
        L.append(f"| {k} | {lab} | {p['tries'] / p['n']:.2f} | {pct(p['nominal_share'])} | {pct(p['robust_share'])} | "
                 f"{' / '.join(f'{x:g}' for x in p['pm_unfiltered_q'])} |")
    L += ["", f"Real fit-half drones (28 included): unfiltered PM p10 / median / p90 = "
              f"{' / '.join(f'{x:g}' for x in f['real']['pm_unfiltered_q'])}°; integral corner min / p10 / median / p90 / max = "
              f"{' / '.join(f'{x:g}' for x in f['real']['i_q'])}.", "",
          "Crossover f_c [Hz] of the drones: generated (after the PM window), kept by Tier 3 step 3 (nominal pass at the PX4 "
          "defaults), and passing the robust check; real drones for comparison:", "",
          "| Population | Stage | " + " | ".join(bins) + " |", "|---|---|" + "---|" * len(bins)]
    for k, _ in names:
        for stage, key in (("generated", "hist_generated"), ("kept (nominal)", "hist_nominal"), ("robust", "hist_robust")):
            L.append(f"| {k} | {stage} | {_hist_row(P[k][key])} |")
    L.append(f"| real (28) | unfiltered f_c | {_hist_row(f['real']['hist_unfiltered'])} |")
    L.append(f"| real (28) | flown f_c (own filters) | {_hist_row(f['real']['hist_flown'])} |")
    return "\n".join(L)


def section_f4(f):
    rows = [r for r in f["f4"] if r["nominal_ok"]]
    rules = list(rows[0]["rules"])
    L = ["## F4. Noise band lower edge f_lo for the 23 nominally passing real settings", "",
         "Share of the noise power N² at the pilot's setting (library windows of the fit half) that comes from below 20 Hz, "
         "the lower edge of the frozen noise-metric band (D-23 Q5):", "",
         "| Case | f_c [Hz] | Robust pass | " + " | ".join(f"{k}: f_lo / share" for k in rules) + " |",
         "|---|---|---|" + "---|" * len(rules)]
    for r in sorted(rows, key=lambda r: r["fc"]):
        if r["rules"]["3 f_c"]["f_lo"] >= 20.0:
            continue
        cells = [f"{v['f_lo']:.1f} / {pct(v['share_below_20'])}" if v["share_below_20"] is not None else f"{v['f_lo']:.1f} / no spectrum"
                 for v in r["rules"].values()]
        L.append(f"| {r['log']} {r['axis']} | {r['fc']:.2f} | {'yes' if r['passes'] else 'no'} | " + " | ".join(cells) + " |")
    L += [""] + [f"- {k}: f_lo below 20 Hz for {sum(1 for r in rows if r['rules'][k]['f_lo'] < 20.0)} of {len(rows)} settings."
                 for k in rules]
    nospec = [f"{r['log']} {r['axis']}" for r in f["f4"] if r["n_windows"] == 0]
    L += [f"- Cases without a library spectrum (no steady window): {', '.join(nospec) if nospec else 'none'}."]
    return "\n".join(L)


def section_a5(f):
    c = f["a5"]["counts"]
    return "\n".join(["## A5. Scenario B (ESC RPM) counts, fit half", "",
                      f"- Usable fit vehicles: {c['vehicles']}; with at least one valid-RPM window: {c['with_valid_rpm']} "
                      f"({', '.join(f['a5']['rpm_logs'])}).",
                      f"- Generator entries: {c['generator_entries']}; entries with motor harmonics (possible sources of "
                      f"synthetic Scenario B drones): {c['generator_rpm_sources']}.",
                      f"- RPM vehicles that also have an accepted Tier 2 identification (real-gain Scenario B): "
                      f"{', '.join(f['a5']['rpm_and_identified'])}."])


def main():
    a = json.load(open(pr.fit_path(HERE, "a4_followup.json")))
    f = json.load(open(pr.fit_path(HERE, "f_options.json")))
    rl = json.load(open(pr.fit_path(HERE, "robust_limits.json")))
    parts = ["# Follow-up tables for the author's questions of 2026-10-18 (generated by followup_tables.py; fit half only)",
             section_q1(a, rl), section_q2(a), section_q3(a), section_q4(a), section_f1(f), section_f2_f3(f), section_f4(f),
             section_a5(f)]
    with open(os.path.join(HERE, "followup_tables.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n\n".join(parts) + "\n")
    print("followup_tables.md written")


if __name__ == "__main__":
    main()
