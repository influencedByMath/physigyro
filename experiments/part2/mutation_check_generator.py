"""Mutation check of tests/test_noise_generator.py: break src/noise_generator.py one way at a time, run the tests,
report which tests fail; the file is restored byte-identical. Run from the repo root."""
import re
import subprocess
import sys

PATH = "src/noise_generator.py"
MUTANTS = {
    "G1 low-pass applied to every chip": (
        'S[i] = spec * nm.lowpass_gain2(f, gen["lowpass"][chip])',
        'S[i] = spec * nm.lowpass_gain2(f, next(v for v in gen["lowpass"].values() if v))'),
    "G2 no peaks sampled": ('for _ in range(int(rng.choice(gen["n_peaks"]))):', 'for _ in range(0):'),
    "G3 peak height ignores the floor": (
        'A = 10 ** (lr + nm.floor_log(np.array([f0]), coef, FLOOR, band)[0])', 'A = 10 ** lr'),
    "G4 width not clipped": ('g = float(np.clip(10 ** lg, *G_RANGE))', 'g = float(10 ** lg) * 1000'),
    "G5 motor harmonics dropped": ('    for h in range(e["n_harm"]):\n        logA, logg', '    for h in range(0):\n        logA, logg'),
    "G6 chips equally likely": ('probs = np.array([gen["chip_share"][c] for c in names])',
                                'probs = np.ones(len(names)) / len(names)'),
    "G7 peak frequency not clipped to band": ('f0 = 10 ** np.clip(lf0, lf_lo, lf_hi)', 'f0 = 10 ** (lf0 + 1.0)'),
    "G8 floor mean wrong": ('floor_mean=floors.mean(0)', 'floor_mean=floors.mean(0) + 1.0'),
    "G9 seed ignored": ('chip = names[rng.choice(len(names), p=probs)]',
                        'chip = names[np.random.default_rng().choice(len(names), p=probs)]'),
}
orig = open(PATH, "rb").read()
src = orig.decode()
try:
    for name, (a, b) in MUTANTS.items():
        assert src.count(a) == 1, f"{name}: pattern not found exactly once"
        open(PATH, "wb").write(src.replace(a, b).encode())
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                            "tests/test_noise_generator.py"], capture_output=True, text=True)
        failed = sorted(set(re.findall(r"FAILED tests/test_noise_generator.py::(\w+)", r.stdout)))
        print(f"{name:42s} -> {'CAUGHT by ' + ', '.join(failed) if failed else 'NOT CAUGHT'}", flush=True)
finally:
    open(PATH, "wb").write(orig)
print("restored byte-identical:", open(PATH, "rb").read() == orig)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_noise_generator.py"],
                   capture_output=True, text=True)
print("intact code:", r.stdout.strip().splitlines()[-1])
