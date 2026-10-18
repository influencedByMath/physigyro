"""Mutation check of the Phase 4 additions to src/sysid.py and of experiments/phase4/plant_variation.py helpers.

    .venv/Scripts/python experiments/phase4/mutation_check_sysid.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from mutation_check_priors import main  # noqa: E402

SYSID = [
    ("acro gets attitude", 'return "rate_sp" if nav_state in ACRO else', 'return "att_sp" if nav_state in ACRO else'),
    ("position counted as stick attitude", "ATTITUDE_FROM_STICKS = {0, 1, 15}", "ATTITUDE_FROM_STICKS = {0, 1, 2, 15}"),
    ("no hold: split at every skip", "breaks = np.where((steps > max_hold + 1) | (steps < 1))[0]",
     "breaks = np.where((steps > 1) | (steps < 1))[0]"),
    ("grid at nominal period (drift)", "out.append((b + a * k, idx))", "out.append((b + Tc * k, idx))"),
    ("hold picks next sample", 'idx = i0 + np.searchsorted(slot, k, side="right") - 1',
     'idx = i0 + np.minimum(np.searchsorted(slot, k, side="left"), len(slot) - 1)'),
    ("unweighted piece average", "acc[key] = acc.get(key, 0.0) + n_seg * S", "acc[key] = acc.get(key, 0.0) + S"),
    ("short pieces used", "        if n < nperseg:", "        if n < 2:"),
    ("coherence of omega only", "return S[\"f\"], S[\"Srw\"] / S[\"Sru\"], np.minimum(coh_w, coh_u)",
     "return S[\"f\"], S[\"Srw\"] / S[\"Sru\"], coh_w"),
    ("agreement ignores phase", "return bool(db <= max_db and deg <= max_deg), db, deg, n", "return bool(db <= max_db), db, deg, n"),
    ("no low-pass before resampling", "x = sosfiltfilt(butter(4, f_cut, fs=fs, output=\"sos\"), x)", "x = x"),
    ("causal low-pass (delay)", "x = sosfiltfilt(butter(4, f_cut, fs=fs, output=\"sos\"), x)",
     "x = __import__('scipy.signal', fromlist=['sosfilt']).sosfilt(butter(4, f_cut, fs=fs, output=\"sos\"), x)"),
    ("edges kept", "m = (tg >= ts[0] + edge_s) & (tg <= ts[-1] - edge_s)", "m = (tg >= ts[0]) & (tg <= ts[-1])"),
]
PV = [
    ("halves uneven", "        if acc + L <= n // 2:", "        if acc + L <= n // 3:"),
    ("zoh not stale-checked", "ok = (k >= 0) & (tg - t_src[np.clip(k, 0, None)] <= STALE_S)", "ok = k >= 0"),
    ("zoh takes next sample", 'k = np.searchsorted(t_src, tg, side="right") - 1', 'k = np.searchsorted(t_src, tg, side="left")'),
]

if __name__ == "__main__":
    main(os.path.join(ROOT, "src", "sysid.py"), SYSID, "tests/test_sysid_pieces.py")
    main(os.path.join(HERE, "plant_variation.py"), PV, "tests/test_plant_variation.py")
