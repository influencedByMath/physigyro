"""Mutation check of src/priors.py (author's rule): each mutant must make tests/test_priors.py fail;
the file is restored byte-identical afterwards (SHA-256 checked).

    .venv/Scripts/python experiments/phase4/mutation_check_priors.py
"""
import hashlib
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(ROOT, "src", "priors.py")
MUTANTS = [
    ("guard off", 'if _TEST_PART.search(str(path)):', 'if False:'),
    ("guard only on basename start", r'r"(^|[\\/])test_[^\\/]*"', r'r"^test_"'),
    ("row guard off", 'bad = set(df["split"]) - {"fit"}', 'bad = set()'),
    ("delay sign of Re p", "(-p.real / ((w[..., None]", "(p.real / ((w[..., None]"),
    ("delay pole angle", "(2 * k - 1) * np.pi / (2 * int(n))", "(2 * k) * np.pi / (2 * int(n))"),
    ("band endpoints only", "f = np.geomspace(band[0], band[1], n_points)", "f = np.array(band, float)"),
    ("bic tolerance ignored", 'keep = [r for r in terms if r["bic"] - best <= tol]', 'keep = [r for r in terms if r["bic"] == best]'),
    ("no-term allowed", 'terms = [r for r in rows if r["chip"] == chip and r["term"] is not None]',
     'terms = [r for r in rows if r["chip"] == chip]'),
    ("rate from mean", "return float(1e6 / np.median(dt))", "return float(1e6 / np.mean(dt))"),
    ("rpm validity ignored", 'df[df["rpm_valid"].astype(bool)]', 'df[df["rpm_hz"].notna()]'),
    ("oneshot read as dshot", 'return "dshot" if v <= -2 else', 'return "dshot" if v <= -1 else'),
    ("motor range off by one", "if _MOTOR_FUNC.match(k) and 101 <= v <= 112})", "if _MOTOR_FUNC.match(k) and 100 <= v <= 112})"),
    ("timers of all groups", "if _PWM_TIM.match(k) and _PWM_TIM.match(k).group(1) == g}", "if _PWM_TIM.match(k)}"),
    ("legacy dshot ignored", 'if params.get("DSHOT_CONFIG", 0) > 0:', 'if False:'),
    ("can as pwm", 'types.add("can")', 'types.add("pwm")'),
    ("full rate from high-rate only", 'return bool(fc.get("has_highrate") or fc.get("has_sysid"))',
     'return bool(fc.get("has_highrate"))'),
    ("mixed groups collapse", 'protocol=types.pop() if len(types) == 1 else "mixed")', 'protocol=types.pop())'),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(SRC=SRC, MUTANTS=MUTANTS, TEST="tests/test_priors.py"):
    orig = open(SRC, "rb").read()
    h0 = sha(SRC)
    caught = 0
    try:
        for name, old, new in MUTANTS:
            text = orig.decode()
            if text.count(old) != 1:
                raise RuntimeError(f"mutant {name!r}: pattern found {text.count(old)} times")
            open(SRC, "wb").write(text.replace(old, new).encode())
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", TEST], cwd=ROOT,
                               capture_output=True, text=True)
            ok = r.returncode != 0
            caught += ok
            print(f"{'CAUGHT ' if ok else 'MISSED '} {name}")
    finally:
        open(SRC, "wb").write(orig)
    print(f"{caught}/{len(MUTANTS)} caught; restored byte-identical: {sha(SRC) == h0}")


if __name__ == "__main__":
    main()
