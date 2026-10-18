"""Mutation check of experiments/phase4/fig1_bode.py: each mutant must make tests/test_fig1_bode.py fail; the
file is restored byte-identical (SHA-256 checked).

    .venv/Scripts/python experiments/phase4/mutation_check_fig1.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mutation_check_priors import main  # noqa: E402

SRC = os.path.join(HERE, "fig1_bode.py")
MUTANTS = [
    ("settings not from the default table", "SETTINGS = replace(lm.PX4_V116_MC_DEFAULT, dnf_esc=True)",
     "SETTINGS = replace(lm.PX4_V116_MC_DEFAULT, dnf_esc=True, f_gyro=50.0)"),
    ("other filter rate", "FS = lm.Rates().fs", "FS = 8000.0"),
    ("D-term path without the derivative", "dterm_path=Fg * Fd * flt.derivative_response(f, fs) / (2j * np.pi * f))",
     "dterm_path=Fg * Fd)"),
    ("gyro path without notches", "Fg = lm.gyro_stack(f, s, fs, motor_hz)", "Fg = flt.lowpass2_response(f, s.f_gyro, fs)"),
    ("factor list misses the low-pass", "gyro_path=notches + [lpf], dterm_path", "gyro_path=notches, dterm_path"),
    ("factor list misses the derivative", "dterm_path=notches + [lpf, pt1, der])", "dterm_path=notches + [lpf, pt1])"),
    ("delay from the unwrapped product", "return {k: 1e3 * sum(-np.angle(h) for h in facs) / (2 * np.pi * f)",
     "return {k: 1e3 * -np.unwrap(np.angle(np.prod(facs, axis=0))) / (2 * np.pi * f)"),
    ("candidate f_c range", "FC_RANGE = (4.0, 15.0)", "FC_RANGE = (3.0, 15.0)"),
    ("band edge", "NOISE_BAND_MAX = 780.0", "NOISE_BAND_MAX = 800.0"),
    ("off variant has the bank", 'VARIANTS = {"off": lm.PX4_V116_MC_DEFAULT, "on": SETTINGS}',
     'VARIANTS = {"off": SETTINGS, "on": SETTINGS}'),
]

if __name__ == "__main__":
    main(SRC, MUTANTS, "tests/test_fig1_bode.py")
