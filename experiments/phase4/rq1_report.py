"""Task 3 S4: Fig. 3, Fig. 4, Table 2 and Table S1 of RQ1 (E3; D-08, D-37 (5), (7)).

    .venv/Scripts/python experiments/phase4/rq1_report.py

Reads the RQ1 results of rq1_run.py (rq1_synthetic.json, {fit,test}_rq1_realgain.json, {fit,test}_rq1_realnoise.json,
rq1_brute.json, rq1_pareto.json) and writes figures/fig3_pm_gain.{png,pdf}, figures/fig4_pareto.{png,pdf} and
rq1_tables.md. Every comparison reports noise, delay and stability together (D-08):
- PM gain at equal noise = PM of the Mode A optimum - PM of the reference (synthetic: PX4 default; real: flown);
- delay change of the gyro path (tau_F) and the D-term path (tau_D) at crossover, same optimum;
- Mode B: the least relative noise N_rel at PM >= PM of the reference;
all optima meet the nominal limits and the D-36 worst-case limits over the uncertainty set (D-37 (1)).
"Reference unsafe" is always given with both numbers, one sentence each (D-37 (7)); "above 10 Hz" by the unfiltered
crossover with the flown count second (D-37 (5)); crossovers outside the 3-10 Hz training range are counted too.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

BANDS = (3.0, 4.0, 5.0, 6.5, 8.0, 10.0)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # validated palette of Fig. 1 (dataviz validator)
INK, MUTED, GRID = "#1f2328", "#5b6470", "#e6e8eb"


def _q(values):
    v = np.asarray([x for x in values if x is not None], float)
    if not len(v):
        return dict(n=0, median=None, p10=None, p90=None)
    return dict(n=int(len(v)), median=float(np.median(v)), p10=float(np.percentile(v, 10)),
                p90=float(np.percentile(v, 90)))


def fc_of(r):
    return r.get("fc_unfiltered", r.get("f_c"))


def gains(r):
    """(dPM, dtau_F, dtau_D) of the Mode A optimum against the reference, or None if Mode A is infeasible."""
    if not r["A"]["feasible"]:
        return None
    a, ref = r["A"]["metrics"], r["reference"]
    return a["PM"] - ref["PM"], a["tau_F"] - ref["tau_F"], a["tau_D"] - ref["tau_D"]


def population_row(rows):
    g = [gains(r) for r in rows]
    fl = [r["reference_flags"] for r in rows]
    nom_fail = sum(not f["nominal_ok"] for f in fl)
    worst_only = sum(f["nominal_ok"] and not f["worst_ok"] for f in fl)
    fcs = [fc_of(r) for r in rows]
    return dict(n=len(rows), A_feasible=sum(x is not None for x in g),
                dPM=_q([x[0] for x in g if x]), dtau_F=_q([x[1] for x in g if x]), dtau_D=_q([x[2] for x in g if x]),
                N_B=_q([r["B"]["metrics"]["N_rel"] for r in rows if r["B"]["feasible"]]),
                B_feasible=sum(r["B"]["feasible"] for r in rows),
                wc_ms_A=_q([r["A"]["robust"]["worst_corners"]["Ms"] for r in rows if r["A"]["feasible"]]),
                ref_nominal_fail=nom_fail, ref_worst_only_fail=worst_only, n_nominal_pass=len(rows) - nom_fail,
                above_10_unfiltered=sum(f > 10.0 for f in fcs),
                above_10_flown=sum((r["reference"].get("fc") or 0) > 10.0 for r in rows),
                outside_3_10=sum(f < 3.0 or f > 10.0 for f in fcs))


def _pct(a, b):
    return f"{100.0 * a / b:.0f} %" if b else "n/a"


def unsafe_sentences(st, what="reference settings"):
    """The two numbers of D-37 (7), one sentence each."""
    s1 = (f"{st['ref_worst_only_fail']} of {st['n_nominal_pass']} nominally passing {what} "
          f"({_pct(st['ref_worst_only_fail'], st['n_nominal_pass'])}) fail the worst case.")
    unsafe = st["ref_nominal_fail"] + st["ref_worst_only_fail"]
    s2 = f"{unsafe} of {st['n']} {what} ({_pct(unsafe, st['n'])}) are \"reference unsafe\" (nominal or worst case)."
    return s1, s2


def band_of(f):
    if f < BANDS[0]:
        return "< 3"
    if f >= BANDS[-1]:
        return "> 10"
    k = int(np.searchsorted(BANDS, f, side="right")) - 1
    return f"{BANDS[k]:g}-{BANDS[k + 1]:g}"


def band_table(rows):
    names = ["< 3"] + [f"{a:g}-{b:g}" for a, b in zip(BANDS[:-1], BANDS[1:])] + ["> 10"]
    out = []
    for nm_ in names:
        sel = [r for r in rows if band_of(fc_of(r)) == nm_]
        st = population_row(sel) if sel else None
        out.append(dict(band=nm_, n=len(sel), dPM=None if st is None else st["dPM"], N_B=None if st is None else st["N_B"],
                        unsafe=0 if st is None else st["ref_nominal_fail"] + st["ref_worst_only_fail"]))
    return out


def _f(q, scale=1.0, nd=1):
    if q is None or q["median"] is None:
        return "-"
    return f"{scale * q['median']:.{nd}f} [{scale * q['p10']:.{nd}f}, {scale * q['p90']:.{nd}f}]"


def table2_md(pops):
    """Table 2: one row per population (and scenario)."""
    head = ("| Population | n | Ref. unsafe: nominal / worst case only (of nominal passes) | Mode A feasible | "
            "ΔPM [deg] median [p10, p90] | Δτ_F [ms] | Δτ_D [ms] | Mode B N_rel | Worst-corner M_s of the A optimum, "
            "max | f_c > 10 Hz unfiltered (flown) | Outside 3-10 Hz |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for name, rows in pops.items():
        groups = [(name, rows)]
        if len({r["scenario"] for r in rows}) > 1:
            groups += [(f"{name}, Scenario {s}", [r for r in rows if r["scenario"] == s]) for s in ("A", "B")]
        for label, rs in groups:
            if not rs:
                continue
            st = population_row(rs)
            wc = st["wc_ms_A"]
            lines.append(f"| {label} | {st['n']} | {st['ref_nominal_fail']} / {st['ref_worst_only_fail']} "
                         f"(of {st['n_nominal_pass']}) | {st['A_feasible']} | {_f(st['dPM'])} | {_f(st['dtau_F'], 1e3, 2)} | "
                         f"{_f(st['dtau_D'], 1e3, 2)} | {_f(st['N_B'], 1.0, 3)} | "
                         f"{'-' if wc['n'] == 0 else format(max(r['A']['robust']['worst_corners']['Ms'] for r in rs if r['A']['feasible']), '.2f')} | "
                         f"{st['above_10_unfiltered']} ({st['above_10_flown']}) | {st['outside_3_10']} |")
    return head + "\n".join(lines) + "\n"


def band_md(name, rows):
    out = [f"**{name}, by unfiltered crossover**\n", "| f_c band [Hz] | n | ΔPM [deg] | Mode B N_rel | Ref. unsafe |",
           "|---|---|---|---|---|"]
    for b in band_table(rows):
        out.append(f"| {b['band']} | {b['n']} | {_f(b['dPM'])} | {_f(b['N_B'], 1.0, 3)} | {b['unsafe']} |")
    return "\n".join(out) + "\n"


def table_s1_md(rows):
    head = ("| Case | Scen. | f_c unf. [Hz] | Default PM / M_s / N_rel | Flown PM / GM / M_s / worst-corner M_s | Flag | "
            "Mode A PM / N_rel / Δτ_F / Δτ_D [ms] | Mode B N_rel / PM | Notes |\n|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        ref, fl = r["reference"], r["reference_flags"]
        flag = "nominal" if not fl["nominal_ok"] else ("worst case" if not fl["worst_ok"] else "-")
        d = r.get("default")
        ds = "-" if d is None else f"{d['PM']:.1f} / {d['Ms']:.2f} / {d['N_rel']:.2f}"
        a, b = r["A"], r["B"]
        g = gains(r)
        as_ = "infeasible" if g is None else f"{a['metrics']['PM']:.1f} / {a['metrics']['N_rel']:.2f} / {1e3 * g[1]:+.2f} / {1e3 * g[2]:+.2f}"
        bs = "infeasible" if not b["feasible"] else f"{b['metrics']['N_rel']:.2f} / {b['metrics']['PM']:.1f}"
        gm = ref["GM"] if isinstance(ref["GM"], (int, float)) else float("inf")
        notes = "; ".join(sorted({f.split(":")[0].split("(")[0].strip() for f in (r.get("flags") or [])}))
        lines.append(f"| {r['id']} | {r['scenario']} | {fc_of(r):.2f} | {ds} | {ref['PM']:.1f} / {gm:.1f} / {ref['Ms']:.2f} / "
                     f"{r['reference_robust']['worst_corners']['Ms']:.2f} | {flag} | {as_} | {bs} | {notes} |")
    return head + "\n".join(lines) + "\n"


def _style(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID)


def _save(fig, out_dir, name):
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3(pops, out_dir):
    """Small multiples: rows ΔPM, Δτ_F, Δτ_D of the Mode A optimum; one column per population. Flagged references
    (orange ticks) and crossovers outside 3-10 Hz (aqua ticks) are marked below each histogram."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows_spec = (("ΔPM at equal noise [deg]", 0, 1.0), ("Δτ_F gyro path [ms]", 1, 1e3), ("Δτ_D D-term path [ms]", 2, 1e3))
    fig, axes = plt.subplots(3, len(pops), figsize=(2.9 * len(pops) + 0.6, 7.2), squeeze=False)
    for j, (name, rows) in enumerate(pops.items()):
        g = [(gains(r), r) for r in rows]
        g = [(x, r) for x, r in g if x is not None]
        for i, (lab, k, sc) in enumerate(rows_spec):
            ax = axes[i, j]
            _style(ax)
            v = np.array([sc * x[k] for x, _ in g])
            if len(v):
                ax.hist(v, bins=min(30, max(6, len(v) // 8)), color=BLUE, edgecolor="white", linewidth=0.6)
                ax.axvline(np.median(v), color=INK, linewidth=1.0, linestyle=(0, (3, 2)))
                y0 = -0.06 * ax.get_ylim()[1]
                flag = [sc * x[k] for x, r in g if not (r["reference_flags"]["nominal_ok"] and r["reference_flags"]["worst_ok"])]
                out = [sc * x[k] for x, r in g if not 3.0 <= fc_of(r) <= 10.0]
                ax.plot(flag, [y0] * len(flag), "|", color=ORANGE, markersize=9, mew=1.6,
                        label="reference unsafe" if (i == 0 and j == 0) else None)
                ax.plot(out, [2 * y0] * len(out), "|", color=AQUA, markersize=9, mew=1.6,
                        label="crossover outside 3-10 Hz" if (i == 0 and j == 0) else None)
                ax.set_ylim(3 * y0, None)
            if i == 0:
                ax.set_title(f"{name}\n(n = {len(rows)}, A feasible {len(g)})", color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, color=MUTED, fontsize=8.5)
    fig.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, out_dir, "fig3_pm_gain")


def fig4(pareto, out_dir):
    """Mode C fronts (best PM for each noise limit, worst-case limits met) of the 4 Fig. 4 drones, with the reference
    (and default) setting and the Mode A / B optima."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), squeeze=False)
    for ax, r in zip(axes.ravel(), pareto):
        _style(ax)
        fr = [(p["n_max"], p["PM"]) for p in r["front"] if p["PM"] is not None]
        if fr:
            x, y = zip(*fr)
            ax.step(x, y, where="post", color=BLUE, linewidth=2.0, label="best PM at the noise limit (Mode C)")
        ref = r["reference"]
        ax.plot([1.0], [ref["PM"]], "o", color=INK, markersize=7, label="reference")
        if r.get("default"):
            ax.plot([r["default"]["N_rel"]], [r["default"]["PM"]], "s", color=MUTED, markersize=7, label="log default")
        for mode, mk in (("A", "^"), ("B", "v")):
            if r[mode]["feasible"]:
                m = r[mode]["metrics"]
                ax.plot([m["N_rel"]], [m["PM"]], mk, color=ORANGE, markersize=8, label=f"Mode {mode} optimum")
        ax.set_xscale("log")
        ax.set_title(f"{r['role']}: {r['id']}", color=INK, fontsize=9, loc="left")
        ax.set_xlabel("relative noise N_rel", color=MUTED, fontsize=8.5)
        ax.set_ylabel("PM [deg]", color=MUTED, fontsize=8.5)
    h, lab = [], []
    for ax in axes.ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, frameon=False, fontsize=8, labelcolor=INK, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, out_dir, "fig4_pareto")


def _load(name):
    p = os.path.join(HERE, name)
    return json.load(open(p))["results"] if os.path.exists(p) else None


def brute_md(rows):
    out = ["| Case | Mode | refined | brute force (120 x 100) | brute - refined |", "|---|---|---|---|---|"]
    worst = {"A": 0.0, "B": 0.0}
    for r in rows:
        for mode, key in (("A", "PM"), ("B", "N_rel")):
            x = r[mode]
            if not (x["feasible"] and x.get("brute", {}).get("feasible")):
                continue
            a, b = x["metrics"][key], x["brute"]["metrics"][key]
            d = b - a if mode == "A" else (b - a) / a
            worst[mode] = max(worst[mode], d if mode == "A" else -d)
            out.append(f"| {r['id']} | {mode} | {a:.3f} | {b:.3f} | {d:+.3f}{'' if mode == 'A' else ' (rel.)'} |")
    return "\n".join(out) + "\n", worst


def main():
    pops = {}
    for label, name in (("synthetic (generator)", "rq1_synthetic.json"), ("real noise (fit)", "fit_rq1_realnoise.json"),
                        ("real noise (test)", "test_rq1_realnoise.json"), ("real gain (fit)", "fit_rq1_realgain.json"),
                        ("real gain (test)", "test_rq1_realgain.json")):
        rows = _load(name)
        if rows is not None:
            pops[label] = rows
    figs = os.path.join(HERE, "figures")
    fig3(pops, figs)
    pareto = _load("rq1_pareto.json")
    if pareto:
        fig4(pareto, figs)
    md = ["# Task 3 (RQ1) tables\n", "Generated by `experiments/phase4/rq1_report.py`; every number below is computed.\n",
          "## Table 2\n", table2_md(pops)]
    md.append("\n## Reference unsafe (D-37 (7): both numbers, one sentence each)\n")
    for name, rows in pops.items():
        what = "flown settings" if name.startswith("real gain") else "reference settings (PX4 defaults)"
        s1, s2 = unsafe_sentences(population_row(rows), what)
        md.append(f"- **{name}:** {s1} {s2}\n")
    md.append("\n## Results per crossover band\n")
    for name, rows in pops.items():
        md.append(band_md(name, rows))
    for name in ("real gain (fit)", "real gain (test)"):
        if name in pops:
            md.append(f"\n## Table S1: {name}\n")
            md.append(table_s1_md(pops[name]))
    br = _load("rq1_brute.json")
    if br:
        t, worst = brute_md(br)
        md.append("\n## Brute-force check (THESIS 7.3, D-37 (4))\n")
        md.append(f"Largest brute-force improvement over the refined optimum: Mode A {worst['A']:+.3f} deg PM; "
                  f"Mode B {100 * worst['B']:.2f} % less noise.\n\n" + t)
    with open(os.path.join(HERE, "rq1_tables.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join(x if x.endswith("\n") else x + "\n" for x in md))
    print("written: rq1_tables.md, figures/fig3_pm_gain, figures/fig4_pareto" if pareto else "written (no Pareto yet)")


if __name__ == "__main__":
    main()
