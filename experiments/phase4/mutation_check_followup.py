"""Mutation check of the code added for the author's questions of 2026-10-18 (A4 follow-up, F1-F4 options, Fig. 1 D-29).

    .venv/Scripts/python experiments/phase4/mutation_check_followup.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from mutation_check_priors import main  # noqa: E402

ROBUST = [
    ("corner labels swap b and tau", "labels = [\"nominal\"] + [f\"b{'-+'[sb > 0]} tau{'-+'[st > 0]}",
     "labels = [\"nominal\"] + [f\"b{'-+'[st > 0]} tau{'-+'[sb > 0]}"),
    ("critical frequencies of the nominal plant only", "f_ms, f180 = critical_frequencies(f, L[i])",
     "f_ms, f180 = critical_frequencies(f, L[0])"),
    ("Ms of the nominal plant in every row", 'Ms=float(m["Ms"][i])', 'Ms=float(m["Ms"][0])'),
]
PV = [
    ("fit ignores the band", "fit = sysid.fit_frf(f, G, coh, Tc, f_band=f_band, min_coh=min_coh)",
     "fit = sysid.fit_frf(f, G, coh, Tc, f_band=F_BAND, min_coh=min_coh)"),
    ("fit ignores the coherence gate", "fit = sysid.fit_frf(f, G, coh, Tc, f_band=f_band, min_coh=min_coh)",
     "fit = sysid.fit_frf(f, G, coh, Tc, f_band=f_band, min_coh=MIN_COH)"),
    ("agreement check ignores the band", "sysid.instruments_agree(f, G1, c1, G2, c2, f_band, min_coh,",
     "sysid.instruments_agree(f, G1, c1, G2, c2, F_BAND, MIN_COH,"),
]
AF = [
    ("sweep uses the nominal limits for every plant", "        lim = separate_limits(m)",
     "        lim = dict(PM_min=PM_MIN, Ms_max=MS_MAX, GM_min=GM_MIN, wc=None)"),
    ("failing cases ignore the nominal check", "if nominal_ok(r) and not rb.passes(r, **lim)]", "if not rb.passes(r, **lim)]"),
    ("chip correction as a lead", "np.exp(-2j * np.pi * np.asarray(f) * td_shift)", "np.exp(2j * np.pi * np.asarray(f) * td_shift)"),
    ("FRF error without coherence", "return np.sqrt((1.0 - coh) / (coh * n_seg))", "return np.sqrt(1.0 / n_seg)"),
    ("refit drops the chip correction", 'td = fit["Td"] if corr is None else max(fit["Td"] + corr, 0.0)', 'td = fit["Td"]'),
    ("segment length not doubled", 'nseg * v["seg_mult"], cls,', "nseg, cls,"),
    ("feasibility counts nominal passes", "                n_pass += ok", "                n_pass += nom"),
    ("loop parts without the D-term filter",
     "return gyro_stack(f, s, rates.fs, motor_hz) * (A + Dd * pt1_response(f, s.f_dterm, rates.fs))",
     "return gyro_stack(f, s, rates.fs, motor_hz) * (A + Dd)"),
]
FO = [
    ("identified plants never used", "if identified_share > 0 and rng.uniform() < identified_share:", "if False:"),
    ("integral corner range ignored", "ir = i_ratio if i_range is None else math.exp(rng.uniform(*np.log(i_range)))",
     "ir = i_ratio"),
    ("last histogram bin open", "k = min(int(np.searchsorted(edges, v, side=\"right\")) - 1, len(edges) - 2)",
     "k = int(np.searchsorted(edges, v, side=\"right\")) - 1"),
    ("net Td uses the FIFO chip first", 'chip = case.get("control_chip") or case.get("fifo_chip")',
     'chip = case.get("fifo_chip") or case.get("control_chip")'),
    ("box check uses the tau range for Td", "tau[0] <= t <= tau[1] and td[0] <= d <= td[1]",
     "tau[0] <= t <= tau[1] and tau[0] <= d <= tau[1]"),
    ("noise integrand without the ZOH", "PZ = zoh_response(fn, rates.f_ctrl) * plant_response(fn, plant)",
     "PZ = plant_response(fn, plant)"),
    ("10 Hz rule for every drone", "10.0 if 5.0 * fc < F_SPLIT else 5.0 * fc", "10.0"),
    ("dropped generator entries counted", 'used = {k: v for k, v in fits.items() if not v.get("dropped")}',
     "used = dict(fits)"),
]
FT = [
    ("no-delay maximum includes the delay corners", 'if c["label"].endswith(" Td"))', 'if "Td" in c["label"])'),
    ("peak maximum ignores coherence", 'coh = [r["S_meas"] for r in rows if r["coh"] >= min_coh and',
     'coh = [r["S_meas"] for r in rows if True and'),
    ("failing-now-pass counts every group", 'if r["group"] == "fails worst case" and r["variants"][k]["passes"])',
     'if r["variants"][k]["passes"])'),
]
FIG = [
    ("caption delay at 10 Hz only", "    d = bank_delay_ms(np.array(FC_RANGE))", "    d = bank_delay_ms(np.array([10.0, 10.0]))"),
    ("panels reversed", "return dict(panels=[k for k, _ in panels],", "return dict(panels=[k for k, _ in panels][::-1],"),
    ("bank delay of the default settings", 'return delays_ms(np.asarray(f, dtype=float), FS, SETTINGS, MOTOR_HZ)["esc_bank"]',
     'return delays_ms(np.asarray(f, dtype=float), FS, VARIANTS["off"], MOTOR_HZ)["esc_bank"]'),
]

if __name__ == "__main__":
    main(os.path.join(ROOT, "src", "robust.py"), ROBUST, "tests/test_robust.py")
    main(os.path.join(HERE, "plant_variation.py"), PV, "tests/test_plant_variation.py")
    main(os.path.join(HERE, "a4_followup.py"), AF, "tests/test_a4_followup.py")
    main(os.path.join(HERE, "f_options.py"), FO, "tests/test_f_options.py")
    main(os.path.join(HERE, "followup_tables.py"), FT, "tests/test_followup_tables.py")
    main(os.path.join(HERE, "fig1_bode.py"), FIG, "tests/test_fig1_bode.py")
