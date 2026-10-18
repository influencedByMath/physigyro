"""Mutation check of src/robust.py and the experiments/phase4/robust_limits.py helpers.

    .venv/Scripts/python experiments/phase4/mutation_check_robust.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from mutation_check_priors import main  # noqa: E402

ROBUST = [
    ("D term without its filter", "Fg * (A + Dd * pt1_response(f, s.f_dterm, rates.fs))", "Fg * (A + Dd)"),
    ("no zero-order hold", "* zoh_response(f, rates.f_ctrl) * plant_response(f, plant)", "* plant_response(f, plant)"),
    ("delay corner symmetric", "for dd in (0.0, float(delta_d))]", "for dd in (-float(delta_d), float(delta_d))]"),
    ("tau corner uses delta_b", "p.tau_m * (1 + st * delta_tau)", "p.tau_m * (1 + st * delta_b)"),
    ("worst PM is the max", 'PM=float(np.min(m["PM"][sl]))', 'PM=float(np.max(m["PM"][sl]))'),
    ("worst Ms is the min", 'Ms=float(np.max(m["Ms"][sl]))', 'Ms=float(np.min(m["Ms"][sl]))'),
    # equivalent mutant: the nominal plant lies inside the corner box and is never the worst one, so including it
    # cannot change a result; kept so the report shows it
    ("corners include nominal", "worst_corners=worst(slice(1, None))", "worst_corners=worst(slice(None))"),
    ("GM not checked nominally", 'and n["Ms"] <= Ms_max and n["GM"] >= GM_min', 'and n["Ms"] <= Ms_max'),
    ("worst-case stability ignored", 'ok_wc = w["stable"] and w["PM"]', 'ok_wc = True and w["PM"]'),
    ("separate limits ignored", "lim = wc or dict(PM=PM_min, Ms=Ms_max, GM=GM_min)", "lim = dict(PM=PM_min, Ms=Ms_max, GM=GM_min)"),
    ("imu rate of first status", "if int(dev) == int(selected_device_id):", "if True:"),
]
RL = [
    ("wrong PM bound", "PM=math.degrees(2 * math.asin(1 / (2 * ms)))", "PM=math.degrees(math.asin(1 / (2 * ms)))"),
    ("delay step in seconds", "w, w, d / rates.f_ctrl + extra_corner", "w, w, d + extra_corner"),
    ("nominal shift ignored", "p = Plant(plant.b, plant.tau_m, plant.Td + extra_nominal)", "p = plant"),
    ("pairs include other IMUs", 'if r["fifo_is_control"] and r["full_rate"] and r["pub_rate"]]', 'if r["pub_rate"]]'),
    ("no PM window", "pm_window=(45.0, 75.0), i_ratio=0.1)", "pm_window=(0.0, 180.0), i_ratio=0.1)"),
    ("i_ratio not passed on", "design_gains(fc, r, plant, rates, i_ratio=i_ratio)", "design_gains(fc, r, plant, rates)"),
    ("no max-tries guard", "        if tries > max_tries:", "        if False:"),
    ("uniform instead of log-uniform tau", "tau_m = math.exp(rng.uniform(*np.log(tau)))", "tau_m = rng.uniform(*tau) * 3"),
    ("nominal pass ignores GM", 'n["Ms"] <= ms and n["GM"] >= GM_MIN)', 'n["Ms"] <= ms)'),
    ("conditional over all cases", 'ok = [e for e in evals if _nominal_ok(e, lim["PM_min"], lim["Ms_max"])]', "ok = list(evals)"),
]

if __name__ == "__main__":
    main(os.path.join(ROOT, "src", "robust.py"), ROBUST, "tests/test_robust.py")
    main(os.path.join(HERE, "robust_limits.py"), RL, "tests/test_robust_limits.py")
