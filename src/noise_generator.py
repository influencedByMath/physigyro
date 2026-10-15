"""Synthetic gyro-noise generator fitted to the parametric model fits of the fit half (THESIS 7.5; D-24 (4)).

One generator for all spectra, with the IMU chip as a covariate (decision D-24 (4), applied when the clusters are
not stable; D-24 P3). Each fitted vehicle model (fit_model.py) is reduced to:
- floor: the spline coefficients (log10 PSD); multivariate Gaussian;
- peaks: every free peak and every motor harmonic (one peak per motor) as (log10 f0, log10 A / floor(f0),
  log10 width); the number of peaks follows the observed counts, the peak triples a 3-D Gaussian;
- chip: categorical with the observed shares; the chip's fitted low-pass (magnitude only, D-25) is applied.
Correlations between floor, peak count and peak triples are not modelled (stated limitation).
"""
import numpy as np

import noise_model as nm

FLOOR = "spline"
G_RANGE = (0.3, 80.0)          # Hz, the fit bounds of a peak width


def entry_peaks(e, band):
    """(log10 f0, log10 peak/floor ratio, log10 width) of every peak of one fitted model entry."""
    p = np.asarray(e["params"], float)
    k = nm.n_floor(FLOOR)
    coef = p[:k]

    def ratio(f0, logA):
        return logA - nm.floor_log(np.array([f0]), coef, FLOOR, band)[0]

    out = []
    for h in range(e["n_harm"]):
        logA, logg = p[k + 2 * h], p[k + 2 * h + 1]
        for fm in e["motor_hz"]:
            f0 = (h + 1) * fm
            out.append((np.log10(f0), ratio(f0, logA), logg))
    j = k + 2 * e["n_harm"]
    for r in range(e["n_free"]):
        logA, f0, logg = p[j + 3 * r: j + 3 * r + 3]
        out.append((np.log10(f0), ratio(f0, logA), logg))
    return out


def fit_generator(entries, band):
    """entries: fitted vehicle models (dicts with chip, lowpass, motor_hz, n_harm, n_free, params)."""
    k = nm.n_floor(FLOOR)
    floors = np.array([np.asarray(e["params"][:k], float) for e in entries])
    peaks = [entry_peaks(e, band) for e in entries]
    triples = np.array([t for pk in peaks for t in pk])
    chips = [e["chip"] for e in entries]
    names = sorted(set(chips))
    lowpass = {}
    for e in entries:
        lowpass.setdefault(e["chip"], e["lowpass"])
        if lowpass[e["chip"]] != e["lowpass"]:
            raise ValueError(f"chip {e['chip']}: entries with different low-pass terms")
    return dict(band=list(band), floor_mean=floors.mean(0), floor_cov=np.cov(floors.T) + 1e-9 * np.eye(k),
                n_peaks=[len(pk) for pk in peaks], peak_mean=triples.mean(0), peak_cov=np.cov(triples.T),
                chip_share={c: chips.count(c) / len(chips) for c in names}, lowpass=lowpass,
                n_entries=len(entries))


def sample(gen, n, f, rng):
    """n synthetic spectra on frequencies f (inside the band). Returns S (n, len f) and the sampled settings."""
    band = tuple(gen["band"])
    names = list(gen["chip_share"])
    probs = np.array([gen["chip_share"][c] for c in names])
    lf_lo, lf_hi = np.log10(band[0]), np.log10(band[1])
    S, meta = np.empty((n, len(f))), []
    for i in range(n):
        chip = names[rng.choice(len(names), p=probs)]
        coef = rng.multivariate_normal(gen["floor_mean"], gen["floor_cov"])
        spec = 10 ** nm.floor_log(f, coef, FLOOR, band)
        pk = []
        for _ in range(int(rng.choice(gen["n_peaks"]))):
            lf0, lr, lg = rng.multivariate_normal(gen["peak_mean"], gen["peak_cov"])
            f0 = 10 ** np.clip(lf0, lf_lo, lf_hi)
            g = float(np.clip(10 ** lg, *G_RANGE))
            A = 10 ** (lr + nm.floor_log(np.array([f0]), coef, FLOOR, band)[0])
            spec = spec + nm.lorentz(f, A, f0, g)
            pk.append((float(f0), float(lr), g))
        S[i] = spec * nm.lowpass_gain2(f, gen["lowpass"][chip])
        meta.append(dict(chip=chip, floor=coef.tolist(), peaks=pk))
    return S, meta
