"""Mutation check of the Fig. 12 tests (E12, D-42): break fig12_tier2.py one way at a time, run
tests/test_fig12_tier2.py, restore byte-identical (SHA-256 checked).

    .venv/Scripts/python experiments/phase4/mutation_check_fig12.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mutation_check_priors as mcp  # noqa: E402

SRC = os.path.join(HERE, "fig12_tier2.py")
MUTANTS = [
    ("raw Td instead of net", "td_net_ms=td_net_ms(case, low),", 'td_net_ms=1e3 * case["plant"]["Td"],'),
    ("in-flight change not excluded", 'mh, not c["exclude"], low))', "mh, True, low))"),
    ("flight-mean RPM for Task 3 cases", 'mh = t3_fit[cid]["motor_hz"] if cid in t3_fit else c["motor_hz"]', 'mh = c["motor_hz"]'),
    ("filtered crossover", 'r["fc_unfiltered"] = rc.unfiltered_crossover(plant, gains, rates)[0]',
     'r["fc_unfiltered"] = float(rb.robust_metrics(plant, gains, rates, rr.settings_of(settings), None, 0.0, 0.0, 0.0)["nominal"]["fc"])'),
    ("old f_c prior", "fc_unfiltered=t3.F_C)", "fc_unfiltered=(4.0, 15.0))"),
    ("bins from the fit half only", "allv = [r[k] for r in rows if r[k] is not None]",
     'allv = [r[k] for r in rows if r[k] is not None and r["half"] == "fit"]'),
    ("FIFO chip before the control chip", 'case=dict(plant=case["plant"], control_chip=case.get("control_chip"), fifo_chip=case.get("fifo_chip")))',
     'case=dict(plant=case["plant"], control_chip=case.get("fifo_chip"), fifo_chip=case.get("fifo_chip")))'),
]

if __name__ == "__main__":
    mcp.main(SRC, MUTANTS, "tests/test_fig12_tier2.py")
