"""Mutation check of the computing helpers of experiments/phase4/evidence_tables.py.

    .venv/Scripts/python experiments/phase4/mutation_check_evidence.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mutation_check_priors import main  # noqa: E402

MUTANTS = [
    ("b difference not symmetric", 'd_b=(h1["b"] - h2["b"]) / (0.5 * (h1["b"] + h2["b"]))', 'd_b=(h1["b"] - h2["b"]) / h2["b"]'),
    ("Td samples use seconds", 'dTd_samples=(h1["Td"] - h2["Td"]) / Tc', 'dTd_samples=(h1["Td"] - h2["Td"])'),
    ("phase at 1 Hz", "def half_diff(h1, h2, Tc, f=10.0):", "def half_diff(h1, h2, Tc, f=1.0):"),
    ("summary without abs", "a = np.abs(np.asarray(values, dtype=float))", "a = np.asarray(values, dtype=float)"),
    ("p90 is p50", 'p90=float(np.quantile(a, 0.9))', 'p90=float(np.quantile(a, 0.5))'),
    ("lag delay term in rad", "+ 360.0 * f * Td)", "+ 2 * np.pi * f * Td)"),
    ("prior range from the wrong corners", "return plant_lag_deg(tau_range[0], td_range[0], f), plant_lag_deg(tau_range[1], td_range[1], f)",
     "return plant_lag_deg(tau_range[0], td_range[1], f), plant_lag_deg(tau_range[1], td_range[0], f)"),
    ("corner lag sign", "lag = max(lag, -np.degrees(np.angle(q)))", "lag = max(lag, np.degrees(np.angle(q)))"),
    ("corner delay in seconds not samples", "for dd in (0.0, delay_samples * Tc):", "for dd in (0.0, delay_samples):"),
    ("bootstrap of the median", "stats = np.quantile(x[rng.integers(0, x.size, (n_boot, x.size))], q, axis=1)",
     "stats = np.quantile(x[rng.integers(0, x.size, (n_boot, x.size))], 0.5, axis=1)"),
    ("funnel keeps the worst stage", "FUNNEL_ORDER.index(stage) < FUNNEL_ORDER.index(best[r[\"log_id\"]])",
     "FUNNEL_ORDER.index(stage) > FUNNEL_ORDER.index(best[r[\"log_id\"]])"),
    ("coverage gain OR phase", 'inside[(w, ds)].append(abs(d["gain_db_10"]) <= g and abs(d["phase_deg_10"]) <= p)',
     'inside[(w, ds)].append(abs(d["gain_db_10"]) <= g or abs(d["phase_deg_10"]) <= p)'),
]

if __name__ == "__main__":
    main(os.path.join(HERE, "evidence_tables.py"), MUTANTS, "tests/test_evidence_tables.py")
