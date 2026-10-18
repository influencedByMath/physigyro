"""Tables of the evidence pack for decisions A2-A4, made from the committed JSON outputs (fit half only).

    .venv/Scripts/python experiments/phase4/evidence_tables.py

Inputs: plant_variation.json, robust_limits.json, priors_from_logs.json (this folder), experiments/part2/model_fits.json,
data/spectra/fit_logs.json. Writes evidence_tables.md next to this file; notes/evidence_pack_A2_A4.md quotes it.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import plant_variation as pv   # noqa: E402
import priors as pr            # noqa: E402
import sysid                   # noqa: E402

AXES = {0: "roll", 1: "pitch"}
ICM = ("ICM42688P", "ICM42686P")


def half_diff(h1, h2, Tc, f=10.0):
    """Repeatability of two identifications of the same vehicle and axis: symmetric relative differences of b and
    tau_m, the Td difference (ms and controller samples), and the plant's gain (dB) and phase (deg) difference at f."""
    G1 = sysid.plant_model(np.array([f]), h1["b"], h1["tau_m"], h1["Td"], Tc)[0]
    G2 = sysid.plant_model(np.array([f]), h2["b"], h2["tau_m"], h2["Td"], Tc)[0]
    return dict(d_b=(h1["b"] - h2["b"]) / (0.5 * (h1["b"] + h2["b"])),
                d_tau=(h1["tau_m"] - h2["tau_m"]) / (0.5 * (h1["tau_m"] + h2["tau_m"])),
                dTd_ms=1e3 * (h1["Td"] - h2["Td"]), dTd_samples=(h1["Td"] - h2["Td"]) / Tc,
                gain_db_10=float(20 * np.log10(abs(G1 / G2))), phase_deg_10=float(np.degrees(np.angle(G1 / G2))))


def plant_lag_deg(tau_m, Td, f=10.0):
    """Phase lag [deg] of the motor lag and the pure delay at f: atan(2 pi f tau_m) + 360 f Td (the part of the
    plant that the uncertainty set perturbs; the integrator and the ZOH are the same for every drone)."""
    return float(np.degrees(np.arctan(2 * np.pi * f * tau_m)) + 360.0 * f * Td)


def prior_lag_range(tau_range, td_range, f=10.0):
    """(min, max) of plant_lag_deg over a box prior of tau_m and Td (monotone in both)."""
    return plant_lag_deg(tau_range[0], td_range[0], f), plant_lag_deg(tau_range[1], td_range[1], f)


def corner_perturbation(tau_m, Td, f_ctrl, width, delay_samples, f=10.0):
    """Size of the uncertainty set at f for one plant: the largest |gain change| [dB] and the largest extra phase lag
    [deg] of the 8 corners b (1 +- width), tau_m (1 +- width), Td + {0, delay_samples / f_ctrl} vs the nominal plant."""
    Tc = 1.0 / f_ctrl
    G0 = sysid.plant_model(np.array([f]), 1.0, tau_m, Td, Tc)[0]
    gain, lag = 0.0, 0.0
    for sb in (1 - width, 1 + width):
        for st in (1 - width, 1 + width):
            for dd in (0.0, delay_samples * Tc):
                q = sysid.plant_model(np.array([f]), sb, tau_m * st, Td + dd, Tc)[0] / G0
                gain = max(gain, abs(20 * np.log10(abs(q))))
                lag = max(lag, -np.degrees(np.angle(q)))
    return float(gain), float(lag)


def bootstrap_ci(values, q=0.9, n_boot=4000, seed=20261018, level=0.95):
    """Percentile-bootstrap confidence interval of the q-quantile of |values| (small samples: n = 15 pairs)."""
    x = np.abs(np.asarray(values, dtype=float))
    if x.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    stats = np.quantile(x[rng.integers(0, x.size, (n_boot, x.size))], q, axis=1)
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def abs_summary(values):
    """n, median and 90th percentile of |values|."""
    a = np.abs(np.asarray(values, dtype=float))
    if a.size == 0:
        return dict(n=0, median=None, p90=None)
    return dict(n=int(a.size), median=float(np.median(a)), p90=float(np.quantile(a, 0.9)))


# ------------------------------------------------------------------------------------------- report sections

def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def accepted_rows(pvj):
    """Per (log, axis): the accepted identification with the most data; plus every rejection reason."""
    best, reasons = {}, Counter()
    for r in pvj["rows"]:
        if "full" not in r:
            reasons["no fit: " + str(r.get("error", "no usable piece"))[:90]] += 1
            continue
        ok, why = pv.accept(r["full"], r["cls"])
        if not ok:
            for w in why:
                reasons[f"{r['cls']}: {w}"] += 1
            continue
        key = (r["log_id"], r["axis"])
        if key not in best or r["full"]["seconds"] > best[key]["full"]["seconds"]:
            best[key] = r
    return best, reasons


FUNNEL_ORDER = ("accepted identification", "fit failed the gates", "too little excitation",
                "no gap-free piece of one Welch segment", "torque setpoint below 200 Hz", "missing topic", "other error")


def _row_stage(r):
    if "full" in r:
        if "b" not in r["full"]:
            return "too little excitation"
        return "accepted identification" if pv.accept(r["full"], r["cls"])[0] else "fit failed the gates"
    e = str(r.get("error", ""))
    if "missing topic" in e:
        return "missing topic"
    if "logged at" in e:
        return "torque setpoint below 200 Hz"
    if "gap-free piece" in e:
        return "no gap-free piece of one Welch segment"
    return "other error"


def log_funnel(rows):
    """Every log counted once, at the best stage any of its rows reached (FUNNEL_ORDER, best first)."""
    best = {}
    for r in rows:
        stage = _row_stage(r)
        if r["log_id"] not in best or FUNNEL_ORDER.index(stage) < FUNNEL_ORDER.index(best[r["log_id"]]):
            best[r["log_id"]] = stage
    return dict(Counter(best.values()))


def section_identification(pvj):
    best, reasons = accepted_rows(pvj)
    L = ["## A. Tier 2 identifications on the fit half (plant_variation.json)", "",
         f"- Logs with the controller signals at full rate (high-rate or sysid profile): **{pvj['n_logs']}**.",
         f"- Accepted (vehicle, axis) identifications: **{len(best)}** from **{len({k[0] for k in best})}** vehicles.",
         "- Every log at the best stage it reached (each log counted once): " + "; ".join(
             f"{k} {v}" for k, v in sorted(log_funnel(pvj["rows"]).items(), key=lambda kv: FUNNEL_ORDER.index(kv[0]))) + ".",
         "- Rows that did not give an accepted fit (one row per log, axis and instrument class):", ""]
    L += [f"  - {k}: {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
    sens = []
    for db, deg in ((2.0, 10.0), (3.0, 15.0), (4.0, 20.0)):
        n = sum(1 for r in pvj["rows"] if "full" in r and pv.accept(r["full"], r["cls"], max_mag_db=db, max_phase_deg=deg)[0])
        sens.append(f"{db:g} dB / {deg:g}°: {n} rows")
    L += ["", "- Gate sensitivity (rows accepted; 3 dB / 15° is used): " + "; ".join(sens) + "."]
    fitted = [r for r in pvj["rows"] if "full" in r and "b" in r["full"]]
    rejected = [r for r in fitted if not pv.accept(r["full"], r["cls"])[0]]
    tau_b = sum(1 for r in rejected if r["full"]["at_bound"]["tau_m"])
    L.append(f"- Fits rejected by the gates: {len(rejected)} of {len(fitted)}; {tau_b} of them have τ_m on its fit bound "
             f"(fit errors {min(r['full']['phase_err_deg'] for r in rejected):.0f}-{max(r['full']['phase_err_deg'] for r in rejected):.0f}°)."
             if rejected else "- Fits rejected by the gates: none.")
    ag = [r["full"]["agree"] for r in best.values() if r["cls"] == "both"]
    if ag:
        L.append(f"- Instrument agreement of the accepted Position/auto rows: {min(a['rms_db'] for a in ag):.2f}-"
                 f"{max(a['rms_db'] for a in ag):.2f} dB, {min(a['rms_deg'] for a in ag):.1f}-{max(a['rms_deg'] for a in ag):.1f}° (RMS).")
    n_fc = sum(1 for r in best.values() if not r["fifo_is_control"])
    L.append(f"- Accepted rows whose logged FIFO IMU is not (or not known to be) the control IMU: {n_fc} of {len(best)}.")
    L += ["", "| Log | Axis | Instrument class | Usable s | b | τ_m [ms] | T_d [ms] | Freq. | Band [Hz] | Fit error | FIFO chip | Control chip |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (log, ax), r in sorted(best.items(), key=lambda kv: -kv[1]["full"]["seconds"]):
        f = r["full"]
        L.append(f"| {log[:8]} | {AXES[ax]} | {r['cls']} | {f['seconds']} | {f['b']:.0f} | {1e3 * f['tau_m']:.1f} | "
                 f"{1e3 * f['Td']:.2f} | {f['n_freq']} | {f['f_used'][0]}-{f['f_used'][1]} | "
                 f"{f['mag_err_db']:.1f} dB / {f['phase_err_deg']:.1f}° | {r['fifo_chip']} | {r['control_chip']} |")
    return "\n".join(L), best


def repeat_pairs(best):
    diffs, rp = [], []
    for (log, ax), r in best.items():
        h1, h2 = r.get("half1", {}), r.get("half2", {})
        if pv.accept(h1, r["cls"])[0] and pv.accept(h2, r["cls"])[0]:
            diffs.append(dict(half_diff(h1, h2, 1 / r["f_ctrl"]), log=log, axis=ax))
    for log in sorted({k[0] for k in best}):
        if (log, 0) in best and (log, 1) in best:
            a, b = best[(log, 0)]["full"], best[(log, 1)]["full"]
            rp.append(dict(log=log, b_ratio=a["b"] / b["b"], d_tau=(a["tau_m"] - b["tau_m"]) / (0.5 * (a["tau_m"] + b["tau_m"])),
                           dTd_ms=1e3 * (a["Td"] - b["Td"])))
    return diffs, rp


def _row(lab, vals, scale, unit):
    s = abs_summary(vals)
    med = None if s["median"] is None else s["median"] * scale
    p90 = None if s["p90"] is None else s["p90"] * scale
    return f"| {lab} | {s['n']} | {_f(med)}{unit} | {_f(p90)}{unit} |"


def section_repeatability(best):
    diffs, rp = repeat_pairs(best)
    L = ["## B. Repeatability: first vs second half of the same flight (same vehicle, same axis)", "",
         f"Pairs with both halves accepted: **{len(diffs)}**. Absolute differences:", "",
         "| Quantity | n | Median | 90th percentile |", "|---|---|---|---|"]
    for k, lab, scale, unit in (("d_b", "b (symmetric relative difference)", 100, " %"),
                                ("d_tau", "τ_m (symmetric relative difference)", 100, " %"),
                                ("dTd_ms", "T_d difference", 1, " ms"), ("dTd_samples", "T_d difference", 1, " samples"),
                                ("gain_db_10", "plant gain at 10 Hz", 1, " dB"), ("phase_deg_10", "plant phase at 10 Hz", 1, "°")):
        L.append(_row(lab, [d[k] for d in diffs], scale, unit))
    L += ["", "Per pair:", "", "| Log | Axis | Δb | Δτ_m | ΔT_d [ms] | gain at 10 Hz | phase at 10 Hz |", "|---|---|---|---|---|---|---|"]
    for d in diffs:
        L.append(f"| {d['log'][:8]} | {AXES[d['axis']]} | {100 * d['d_b']:+.1f} % | {100 * d['d_tau']:+.1f} % | "
                 f"{d['dTd_ms']:+.2f} | {d['gain_db_10']:+.2f} dB | {d['phase_deg_10']:+.1f}° |")
    tau = float(np.median([r["full"]["tau_m"] for r in best.values()]))
    td = float(np.median([r["full"]["Td"] for r in best.values()]))
    L += ["", f"Size of each candidate uncertainty set at 10 Hz for the median identified plant (τ_m {1e3 * tau:.1f} ms, "
          f"T_d {1e3 * td:.2f} ms): largest gain change / largest extra phase lag over the 8 corners. Compare with the "
          "90th percentiles above.", "",
          "| Width | +0.5 sample, 800 Hz | +1 sample, 800 Hz | +2 samples, 800 Hz | +1 sample, 400 Hz |", "|---|---|---|---|---|"]
    for w in (0.10, 0.15, 0.20, 0.30):
        cells = []
        for d, fctrl in ((0.5, 800.0), (1.0, 800.0), (2.0, 800.0), (1.0, 400.0)):
            g, p = corner_perturbation(tau, td, fctrl, w, d)
            cells.append(f"{g:.2f} dB / {p:.1f}°")
        L.append(f"| ±{100 * w:.0f} % | " + " | ".join(cells) + " |")
    L += ["", f"Roll vs pitch of the same vehicle (same motors and ESCs; b differs by the inertia): **{len(rp)}** vehicles.", "",
          "| Quantity | n | Median | 90th percentile |", "|---|---|---|---|"]
    L.append(_row("τ_m roll vs pitch", [d["d_tau"] for d in rp], 100, " %"))
    L.append(_row("T_d roll vs pitch", [d["dTd_ms"] for d in rp], 1, " ms"))
    if rp:
        L.append(f"| b roll / pitch (ratio) | {len(rp)} | {np.median([d['b_ratio'] for d in rp]):.2f} | "
                 f"range {min(d['b_ratio'] for d in rp):.2f}-{max(d['b_ratio'] for d in rp):.2f} |")
    return "\n".join(L)


CANDIDATES = ((0.10, 0.5), (0.10, 1.0), (0.15, 1.0), (0.20, 1.0), (0.30, 1.0), (0.20, 2.0))


def accepted_pairs(best):
    return [(log, ax, r, r["half1"], r["half2"]) for (log, ax), r in sorted(best.items())
            if pv.accept(r.get("half1", {}), r["cls"])[0] and pv.accept(r.get("half2", {}), r["cls"])[0]]


def coverage(pairs, freq_of, candidates=CANDIDATES):
    """Share of half-flight pairs whose gain AND phase differences at the drone's own frequency freq_of(log, axis)
    lie inside the candidate set built around that drone's identified plant; pairs without a frequency are skipped.
    Returns ({candidate: share}, n)."""
    inside = {c: [] for c in candidates}
    for log, ax, r, h1, h2 in pairs:
        fq = freq_of(log, ax)
        if fq is None or not np.isfinite(fq):
            continue
        d = half_diff(h1, h2, 1 / r["f_ctrl"], f=fq)
        for w, ds in candidates:
            g, p = corner_perturbation(r["full"]["tau_m"], r["full"]["Td"], r["f_ctrl"], w, ds, f=fq)
            inside[(w, ds)].append(abs(d["gain_db_10"]) <= g and abs(d["phase_deg_10"]) <= p)
    n = len(next(iter(inside.values())))
    return {c: (float(np.mean(v)) if v else None) for c, v in inside.items()}, n


def section_frequency(best, rl):
    pairs = accepted_pairs(best)
    cases = {(c["log"], c["axis"]): c for c in rl["real"]["cases"]}
    tau = float(np.median([r["full"]["tau_m"] for r in best.values()]))
    td = float(np.median([r["full"]["Td"] for r in best.values()]))
    L = ["## B2. Repeatability where the margins bind (independent review, Important 2)", "",
         f"Half-flight differences at several frequencies ({len(pairs)} pairs): 90th percentile with a 95 % bootstrap "
         f"interval, next to the size of two candidate sets for the median plant (τ_m {1e3 * tau:.1f} ms, "
         f"T_d {1e3 * td:.2f} ms, 800 Hz).", "",
         "| Frequency | Phase, p90 [95 % CI] | Gain, p90 [95 % CI] | ±15 % +1 sample | ±20 % +1 sample |", "|---|---|---|---|---|"]
    for fq in (5.0, 10.0, 15.0, 20.0):
        d = [half_diff(h1, h2, 1 / r["f_ctrl"], f=fq) for _, _, r, h1, h2 in pairs]
        ph = [x["phase_deg_10"] for x in d]
        ga = [x["gain_db_10"] for x in d]
        pl, phh = bootstrap_ci(ph)
        gl, gh = bootstrap_ci(ga)
        s15 = corner_perturbation(tau, td, 800.0, 0.15, 1.0, f=fq)
        s20 = corner_perturbation(tau, td, 800.0, 0.20, 1.0, f=fq)
        L.append(f"| {fq:g} Hz | {abs_summary(ph)['p90']:.1f}° [{pl:.1f}-{phh:.1f}] | {abs_summary(ga)['p90']:.2f} dB "
                 f"[{gl:.2f}-{gh:.2f}] | {s15[0]:.2f} dB / {s15[1]:.1f}° | {s20[0]:.2f} dB / {s20[1]:.1f}° |")

    def freq(key):
        return lambda log, ax: (cases.get((log[:8], AXES[ax])) or {}).get("nominal", {}).get(key)
    cov_ms, n_ms = coverage(pairs, freq("f_Ms"))
    cov_pc, n_pc = coverage(pairs, freq("f_180"))
    L += ["", "Share of pairs whose gain AND phase differences lie inside each candidate set, evaluated at the drone's "
          "OWN frequency of the M_s peak and of its critical phase crossover (set built around its own plant):", "",
          f"| Candidate set | at the M_s peak (n {n_ms}) | at the phase crossover (n {n_pc}) |", "|---|---|---|"]
    for w, ds in CANDIDATES:
        a, b = cov_ms[(w, ds)], cov_pc[(w, ds)]
        L.append(f"| ±{100 * w:.0f} % +{ds:g} sample | {'-' if a is None else f'{100 * a:.0f} %'} | "
                 f"{'-' if b is None else f'{100 * b:.0f} %'} |")
    inc = [c for c in rl["real"]["cases"] if not c.get("exclude")]
    fms = [c["nominal"].get("f_Ms") for c in inc]
    above = sum(1 for c in inc if c.get("fit_band") and c["nominal"].get("f_Ms", 0) > c["fit_band"][1])
    above180 = sum(1 for c in inc if c.get("fit_band") and np.isfinite(c["nominal"].get("f_180", np.nan))
                   and c["nominal"]["f_180"] > c["fit_band"][1])
    if fms:
        L += ["", f"M_s peak of the real flown settings: {' / '.join(f'{v:.1f}' for v in np.quantile(fms, [0, 0.5, 1]))} Hz "
              f"(min / median / max). The M_s peak lies above the highest coherent fit frequency in **{above} of {len(inc)}** "
              f"cases and the critical phase crossover in {above180}: there the margins rest on the plant model "
              "extrapolated beyond the identified band."]
    return "\n".join(L)


def section_across(best):
    tau = [1e3 * r["full"]["tau_m"] for r in best.values()]
    td = [1e3 * r["full"]["Td"] for r in best.values()]

    def q(x):
        return " / ".join(f"{v:.1f}" for v in np.quantile(x, [0, 0.1, 0.5, 0.9, 1])) if x else "-"
    it = 100 * np.mean([(20 <= t <= 60) for t in tau]) if tau else float("nan")
    idl = 100 * np.mean([(0.5 <= t <= 5) for t in td]) if td else float("nan")
    lag = [plant_lag_deg(r["full"]["tau_m"], r["full"]["Td"]) for r in best.values()]
    lo, hi = prior_lag_range((0.020, 0.060), (0.0005, 0.005))
    lo10, _ = prior_lag_range((0.010, 0.060), (0.0001, 0.005))
    il = 100 * np.mean([(lo <= x <= hi) for x in lag]) if lag else float("nan")
    below = sum(1 for x in lag if x < lo)
    return "\n".join(["## C. Across vehicles (accepted identifications) and the D-28 priors", "",
                      "| Quantity | n | min / p10 / median / p90 / max | Share inside the D-28 prior |", "|---|---|---|---|",
                      f"| τ_m [ms] | {len(tau)} | {q(tau)} | {it:.0f} % (20-60 ms) |",
                      f"| T_d [ms] | {len(td)} | {q(td)} | {idl:.0f} % (0.5-5 ms) |",
                      f"| Motor + delay lag at 10 Hz [°] | {len(lag)} | {q(lag)} | {il:.0f} % ({lo:.1f}-{hi:.1f}°) |", "",
                      f"τ_m and T_d trade off below the 20 Hz band edge, so the lag at 10 Hz (near the crossover) is the robust "
                      f"comparison: {below} of {len(lag)} identified plants lag less than the fastest plant the D-28 priors "
                      f"can produce ({lo:.1f}°). A prior of τ_m 10-60 ms and T_d 0.1-5 ms would reach down to {lo10:.1f}°."])


def section_chips(best, pri, logs, fits):
    cd = pri["chip_delay"]
    L = ["## D. ICM chip delay under the two rules (Task 1; D-25)", "",
         "| Chip | Fit vehicles | BIC rule (within 10): terms, delay [ms] | Fit-error rule (within 1.5 %): terms, delay [ms] |",
         "|---|---|---|---|"]
    for chip, v in cd.items():
        L.append(f"| {chip} | {v['n_vehicles']} | {v['bic']['n_terms']}, {v['bic']['delay_ms'][0]}-{v['bic']['delay_ms'][1]} | "
                 f"{v['rms']['n_terms']}, {v['rms']['delay_ms'][0]}-{v['rms']['delay_ms'][1]} |")
    L += ["", "Delay of the same fit-error-rule terms at 5 Hz (near the crossover), against the 20-100 Hz band of the plan:", ""]
    for chip, v in cd.items():
        d5 = [1e3 * float(pr.lowpass_group_delay(5.0, fc, n)) for fc, n in v["rms"]["terms"]]
        L.append(f"- {chip}: {min(d5):.2f}-{max(d5):.2f} ms at 5 Hz (20-100 Hz band: {v['rms']['delay_ms'][0]}-{v['rms']['delay_ms'][1]} ms).")
    L.append("- All of these are delays of minimum-phase filters with the fitted magnitude, i.e. lower bounds of the true "
             "delay; the largest of them is not an upper bound.")
    fifo = Counter((r["fc"].get("fifo_chip") or {}).get("name") for r in logs)
    ctrl = Counter((r["fc"].get("control_chip") or {}).get("name") for r in logs)
    gen = Counter(v["chip"] for v in fits["vehicles"].values() if not v.get("dropped"))
    share = 100 * sum(gen.get(c, 0) for c in ICM) / max(1, sum(gen.values()))
    L += ["", "Who carries an ICM-4268x chip (fit half; the test half is not counted, leakage rule):", "",
          f"- FIFO (noise) IMU of the {len(logs)} screened fit logs: ICM-42688-P {fifo.get('ICM42688P', 0)}, "
          f"ICM-42686-P {fifo.get('ICM42686P', 0)}.",
          f"- Control IMU of the {len(logs)} fit logs (known in {sum(v for k, v in ctrl.items() if k)}): "
          f"ICM-42688-P {ctrl.get('ICM42688P', 0)}, ICM-42686-P {ctrl.get('ICM42686P', 0)}.",
          f"- Generator sources ({sum(gen.values())} spectra, drawn uniformly by the synthetic drones): ICM-42688-P "
          f"{gen.get('ICM42688P', 0)}, ICM-42686-P {gen.get('ICM42686P', 0)}, i.e. {share:.0f} % of synthetic drones.",
          f"- Accepted Tier 2 identifications with an ICM-4268x FIFO IMU: {sum(1 for r in best.values() if r['fifo_chip'] in ICM)} "
          f"of {len(best)}; with an ICM-4268x control IMU: {sum(1 for r in best.values() if r['control_chip'] in ICM)}."]
    a = [1e3 * r["full"]["Td"] for r in best.values() if r["fifo_chip"] in ICM]
    b = [1e3 * r["full"]["Td"] for r in best.values() if r["fifo_chip"] not in ICM]
    if a and b:
        L.append(f"- Identified T_d, median: ICM-4268x FIFO IMU {np.median(a):.2f} ms (n {len(a)}), other chips "
                 f"{np.median(b):.2f} ms (n {len(b)}). An identified T_d already contains the FIFO chip's own filter.")
    return "\n".join(L)


def _rate_table(rates, kind, delay, names):
    widths = ("0.1", "0.15", "0.2", "0.3")
    L = [f"| Limits ({kind}; delay step {delay} sample) | ±10 % | ±15 % | ±20 % | ±30 % |", "|---|---|---|---|---|"]
    for name in names:
        cells = [f"{100 * rates[f'{name}|w{w}_d{delay}']:.0f} %" for w in widths]
        L.append(f"| {name.replace('_', ' ')} | " + " | ".join(cells) + " |")
    return L


def section_pass(rl):
    pms, mss, wcs = rl["settings"]["PM_FLOORS"], rl["settings"]["MS_MAX"], rl["settings"]["WC_MS"]
    same = [f"PM{p:g}_Ms{m:g}_same" for p in pms for m in mss]
    sep = [f"PM{p:g}_Ms{m:g}_wcMs{w:g}" for p in pms for m in mss for w in wcs]
    L = ["## E. Pass rates (robust_limits.json)", "",
         "A case passes if the nominal plant meets PM ≥ floor, M_s ≤ M_s,max and GM ≥ 6 dB, and every plant of the set "
         "(nominal + 8 corners) is Nyquist-stable and meets the worst-case limits: the same ones (\"same\"), or separate "
         "ones (\"wcMs\"): M_s ≤ M_s,wc with the PM and GM that M_s,wc guarantees "
         + ", ".join(f"M_s,wc {w} → PM ≥ {v['PM']:.1f}°, GM ≥ {v['GM']:.1f} dB" for w, v in rl["settings"]["wc_limits"].items())
         + ".", ""]
    for label, block in (("Real-gain fit-half drones, pilot's actual filters", rl["real"]),
                         ("Synthetic Tier 3a drones at the PX4 default filters", rl["synthetic"])):
        if not block or not block.get("pass_rates"):
            L += [f"### {label}: no cases", ""]
            continue
        n = block.get("n_included", block.get("n_cases", block.get("n")))
        extra = (f"; {block['n_cases'] - block['n_included']} more rows flagged and left out (loop parameters changed in flight)"
                 if block.get("n_included") is not None and block["n_cases"] != block["n_included"] else "")
        L += [f"### {label} (n = {n}{extra})", ""]
        nom = block.get("nominal_pass_rates") or {}
        if nom:
            L += ["Nominal limits only (no uncertainty set): " + "; ".join(
                f"{k.replace('_', ' ')} {100 * v:.0f} %" for k, v in nom.items()) + ".", ""]
        for d in ("0.5", "1", "2"):
            L += _rate_table(block["pass_rates"], "same limits", d, same) + [""]
        for d in ("0.5", "1", "2"):
            L += _rate_table(block["pass_rates"], "separate worst-case limits", d, sep) + [""]
        cond = block.get("conditional_rates") or {}
        if cond:
            names = ["PM30_Ms2_same", "PM30_Ms2_wcMs2.5", "PM30_Ms2_wcMs3", "PM30_Ms2.2_same", "PM30_Ms2.2_wcMs3"]
            L += ["Among the cases that pass the NOMINAL limits, the share that also passes the robust check:", "",
                  "| Limits | ±10 % +0.5 | ±10 % +1 | ±15 % +1 | ±20 % +1 | ±30 % +1 | ±20 % +2 |", "|---|---|---|---|---|---|---|"]
            for name in names:
                cells = []
                for key in ("w0.1_d0.5", "w0.1_d1", "w0.15_d1", "w0.2_d1", "w0.3_d1", "w0.2_d2"):
                    v = cond.get(f"{name}|{key}")
                    cells.append("-" if v is None else f"{100 * v:.0f} %")
                L.append(f"| {name.replace('_', ' ')} | " + " | ".join(cells) + " |")
            L.append("")
        bins = block.get("fc_bins")
        if bins:
            cols = [k for k in next(iter(bins.values())) if k != "n"]
            L += ["By crossover frequency (synthetic drones):", "", "| f_c bin | n | " + " | ".join(c.replace("_", " ") for c in cols) + " |",
                  "|---|---|" + "---|" * len(cols)]
            for b, row in bins.items():
                L.append(f"| {b} | {row['n']} | " + " | ".join("-" if row[c] is None else f"{100 * row[c]:.0f} %" for c in cols) + " |")
            L.append("")
    return "\n".join(L)


def section_cases(rl):
    L = ["## F. Single drones: nominal and worst-case margins of the flown setting", "",
         "Worst case = smallest PM / largest M_s over the 9 plants; \"+1\" = delay step of 1 controller sample.", "",
         "| Drone | Axis | b | τ_m [ms] | T_d [ms] | f_ctrl | Nominal PM / M_s / GM | ±10 % +1 | ±15 % +1 | ±20 % +1 | ±30 % +1 | ±20 % +0.5 | ±20 % +2 | Flags |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in rl["feasibility"]["cases"] + rl["real"]["cases"]:
        n = c["nominal"]
        cells = []
        for key in ("w0.1_d1", "w0.15_d1", "w0.2_d1", "w0.3_d1", "w0.2_d0.5", "w0.2_d2"):
            w = c["worst"][key]
            cells.append(f"{w['PM']:.1f}° / {w['Ms']:.2f}" + ("" if w["stable"] else " UNSTABLE"))
        L.append(f"| {c['log']} | {c['axis']} | {c['b']:.0f} | {c['tau_ms']} | {c['Td_ms']} | {c['f_ctrl']:.0f} | "
                 f"{n['PM']:.1f}° / {n['Ms']:.2f} / {n['GM']:.1f} dB | " + " | ".join(cells) + f" | {'; '.join(c['flags']) or '-'} |")
    return "\n".join(L)


def section_chip_pass(rl):
    keys = ["PM30_Ms2_same", "PM30_Ms2_wcMs3", "PM25_Ms2.2_same", "PM25_Ms2.2_wcMs3"]
    base = rl["synthetic"]["pass_rates"]
    chip = rl["synthetic"]["chip_delay"]
    head = " | ".join(f"{k.replace('|', ' ')} {m}" for k in chip for m in ("shift", "in step"))
    L = ["## G. Synthetic drones with an ICM-4268x chip delay (±20 %, delay step 1 sample)", "",
         "\"shift\": nominal T_d + lower delay bound, delay step + (upper - lower); \"in step\": delay step + upper bound.", "",
         f"| Limits | No chip delay | {head} |", "|---|---|" + "---|" * (2 * len(chip))]
    for k in keys:
        cells = [f"{100 * base[f'{k}|w0.2_d1']:.0f} %"]
        for v in chip.values():
            cells += [f"{100 * v['nominal_shift'][f'{k}|w0.2_d1']:.0f} %", f"{100 * v['in_delay_step'][f'{k}|w0.2_d1']:.0f} %"]
        L.append(f"| {k.replace('_', ' ')} | " + " | ".join(cells) + " |")
    return "\n".join(L)


def section_design(ps):
    real = ps["real"]
    fl = np.array([c["fc_flown"] for c in real["cases"]])
    L = ["## H. Is the synthetic Tier 3a design like the real drones? (prior_sensitivity.json)", "",
         "Real fit-half drones (identified plant + the log's gains), in the terms of THESIS 7.1.1 Tier 3a:", "",
         f"- Crossover with the flown filters: min / p10 / median / p90 / max = "
         f"{' / '.join(f'{v:.1f}' for v in np.quantile(fl, [0, 0.1, 0.5, 0.9, 1]))} Hz "
         f"({int(np.sum(fl < 4.0))} of {len(fl)} below 4 Hz).",
         f"- PM without filters: {' / '.join(f'{v}' for v in real['pm_unfiltered_q'].values())}° "
         f"(min / p10 / median / p90 / max); inside the Tier 3a window 45-75°: {100 * real['share_in_45_75']:.0f} %.",
         f"- D-corner ratio r: median {real['r_q']['0.5']} (p10 {real['r_q']['0.1']}, p90 {real['r_q']['0.9']}); inside "
         f"[0.5, 3]: {100 * real['r_share_in_0p5_3']:.0f} %. I-corner ratio: median {real['i_median']} (Tier 3a uses 0.1).", "",
         "Synthetic drones at the PX4 default filters, by design (1,000 each, same seed):", "",
         "| Design | Tries | Nominal PM30/Ms2 | Nominal PM25/Ms2.2 | 4-6.5 Hz nominal | 6.5-10 Hz nominal | 10-15 Hz nominal | 10-15 Hz nominal PM25/Ms2.2 | 6.5-10 Hz robust (wcMs3, ±15 % +1) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, d in ps["designs"].items():
        b = d["fc_bins"]
        L.append(f"| {name} | {d['tries']} | {100 * d['nominal_pass_rates']['PM30_Ms2']:.0f} % | "
                 f"{100 * d['nominal_pass_rates']['PM25_Ms2.2']:.0f} % | "
                 + " | ".join(f"{100 * b[k]['nominal|PM30_Ms2']:.0f} % (n {b[k]['n']})" for k in ("4-6.5 Hz", "6.5-10 Hz", "10-15 Hz"))
                 + f" | {100 * b['10-15 Hz']['nominal|PM25_Ms2.2']:.1f} % | {100 * b['6.5-10 Hz']['PM30_Ms2_wcMs3|w0.15_d1']:.0f} % |")
    return "\n".join(L)


def main():
    pvj = json.load(open(pr.fit_path(HERE, "plant_variation.json")))
    rl = json.load(open(pr.fit_path(HERE, "robust_limits.json")))
    pri = json.load(open(pr.fit_path(HERE, "priors_from_logs.json")))
    logs = json.load(open(pr.fit_path(ROOT, "data", "spectra", "fit_logs.json")))
    fits = json.load(open(pr.fit_path(ROOT, "experiments", "part2", "model_fits.json")))
    ps = json.load(open(pr.fit_path(HERE, "prior_sensitivity.json")))
    a, best = section_identification(pvj)
    parts = ["# Evidence tables for A2-A4 (generated by experiments/phase4/evidence_tables.py; fit half only)", "",
             a, "", section_repeatability(best), "", section_frequency(best, rl), "", section_across(best), "",
             section_chips(best, pri, logs, fits), "",
             section_pass(rl), "", section_cases(rl), "", section_chip_pass(rl), "", section_design(ps), ""]
    with open(os.path.join(HERE, "evidence_tables.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print("written evidence_tables.md")


if __name__ == "__main__":
    main()
