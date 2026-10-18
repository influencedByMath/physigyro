"""Mutation check of the Fig. 1 changes for D-36 (crossover band 3-10 Hz, noise band from max(5 f_c, 20 Hz), UTF-8 output).

    .venv/Scripts/python experiments/phase4/mutation_check_d36.py

No mutant of the output-folder argument: it would make the test run write into the real figures/ folder.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mutation_check_priors import main  # noqa: E402

FIG = [
    ("noise band ignores the 20 Hz floor", "    return max(5.0 * fc, NOISE_BAND_MIN)", "    return 5.0 * fc"),
    ("hatched band starts at the upper crossover", "noise_hatched=(noise_band_start(FC_RANGE[0])",
     "noise_hatched=(noise_band_start(FC_RANGE[1])"),
    ("crossover band of D-28", "FC_RANGE = (3.0, 10.0)", "FC_RANGE = (4.0, 15.0)"),
    ("caption cites D-28", "{FC_RANGE[1]:g} Hz (D-36)", "{FC_RANGE[1]:g} Hz (D-28)"),
    ("script prints with the console encoding", 'sys.stdout.reconfigure(encoding="utf-8")', "sys.stdout.reconfigure()"),
]

if __name__ == "__main__":
    main(os.path.join(HERE, "fig1_bode.py"), FIG, "tests/test_fig1_bode.py")
