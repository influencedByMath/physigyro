"""Mutation check of tests/test_archetypes.py (author request 2026-10-15): break src/archetypes.py one way at a time, run the tests,
record which tests fail, ALWAYS restore the original file. Run from the repo root."""
import re
import subprocess
import sys

PATH = "src/archetypes.py"
MUTANTS = {
    "M1 no mean removal": ("    if remove_mean:\n", "    if False:\n"),
    "M2 roll/pitch mean of logs, not of power": (
        "Y = np.log10((10 ** B[:, :, 0] + 10 ** B[:, :, 1]) / 2)[None]", "Y = ((B[:, :, 0] + B[:, :, 1]) / 2)[None]"),
    "M3 vehicle mean = first window only": (
        "rows.append(np.log10((10 ** Ya[groups == v]).mean(0))[None])", "rows.append(Ya[groups == v][:1])"),
    "M4 k = worst silhouette": ("k = max(sil, key=sil.get)", "k = min(sil, key=sil.get)"),
    "M5 PCA keeps 1 component": ('PCA(n_components=var, svd_solver="full")', 'PCA(n_components=1, svd_solver="full")'),
    "M6 seed ignored": ("KMeans(n_clusters=k, n_init=50, random_state=seed)", "KMeans(n_clusters=k, n_init=1, random_state=None)"),
    "M7 bootstrap over rows, not vehicles": (
        "return np.concatenate([np.nonzero(groups == v)[0] for v in pick])",
        "return rng.choice(len(groups), size=len(groups), replace=True)"),
    "M8 stability always perfect": (
        'return dict(median_ari=float(np.median(aris))', 'return dict(median_ari=1.0 + 0 * float(np.median(aris))'),
    "M10 stability always zero": (
        'return dict(median_ari=float(np.median(aris))', 'return dict(median_ari=0 * float(np.median(aris))'),
    "M9 crosstab transposed": (
        'pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(attribute, name="attribute"))',
        'pd.crosstab(pd.Series(attribute, name="attribute"), pd.Series(labels, name="cluster"))'),
}

orig_bytes = open(PATH, "rb").read()
orig = orig_bytes.decode("utf-8")
try:
    for name, (a, b) in MUTANTS.items():
        assert orig.count(a) == 1, f"{name}: pattern not found exactly once"
        open(PATH, "wb").write(orig.replace(a, b).encode("utf-8"))
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_archetypes.py"],
                           capture_output=True, text=True)
        failed = sorted(set(re.findall(r"FAILED tests/test_archetypes.py::(\w+)", r.stdout)))
        print(f"{name:45s} -> {'CAUGHT by ' + ', '.join(failed) if failed else 'NOT CAUGHT'}", flush=True)
finally:
    open(PATH, "wb").write(orig_bytes)
print("restored byte-identical:", open(PATH, "rb").read() == orig_bytes)
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_archetypes.py"],
                   capture_output=True, text=True)
print("intact code:", r.stdout.strip().splitlines()[-1])
