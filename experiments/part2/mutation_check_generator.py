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
    "G2 no peaks sampled": ('for lf0, lr, lg in gen["peak_sets"][j]:', 'for lf0, lr, lg in []:'),
    "G3 peak height ignores the floor": (
        'A = 10 ** (lr + nm.floor_log(np.array([f0]), coef, FLOOR, band)[0])', 'A = 10 ** lr'),
    "G4 width not clipped": ('g = float(np.clip(10 ** lg, *G_RANGE))', 'g = float(10 ** lg) * 1000'),
    "G5 motor harmonics dropped": ('    for h in range(e["n_harm"]):\n        logA, logg',
                                   '    for h in range(0):\n        logA, logg'),
    "G6 chip not from the source vehicle": ('chip = gen["chips"][j]', 'chip = gen["chips"][0]'),
    "G7 independent frequency scale per peak": ('            f0 = 10 ** lf0 * scale',
                                                '            f0 = 10 ** lf0 * 10 ** rng.normal(0.0, gen["freq_sd"] + 1e-12)'),
    "G8 floor mean wrong": ('floor_mean = {c: floors[cls == c].mean(0) for c in sorted(set(cls))}',
                            'floor_mean = {c: floors[cls == c].mean(0) + 1.0 for c in sorted(set(cls))}'),
    "G13 chip class ignored when sampling": ('rng.multivariate_normal(gen["floor_mean"][chip_class(chip)], gen["floor_cov"])',
                                             'rng.multivariate_normal(gen["floor_mean"]["other"] if "other" in gen["floor_mean"] else next(iter(gen["floor_mean"].values())), gen["floor_cov"])'),
    "G14 covariance around the pooled mean": ('resid = floors - np.array([floor_mean[c] for c in cls])',
                                              'resid = floors - floors.mean(0)'),
    "G9 seed ignored": ('j = int(rng.integers(len(gen["sources"])))',
                        'j = int(np.random.default_rng().integers(len(gen["sources"])))'),
    "G10 height jitter doubled": ('lr = lr + (rng.normal(0.0, gen["height_sd"])', 'lr = lr + (rng.normal(0.0, 2 * gen["height_sd"])'),
    "G11 dropped fits used": ('use = [i for i, e in enumerate(entries) if not e.get("dropped")]',
                              'use = list(range(len(entries)))'),
    "G12 peaks outside the band kept": ('            if not band[0] <= f0 <= band[1]:\n                continue',
                                        '            if False:\n                continue'),
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
