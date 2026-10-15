"""Synthetic gyro-noise generator fitted to the parametric model fits of the fit half (THESIS 7.5; D-24 (4), D-26).

One generator for all spectra, with the IMU chip as a covariate (D-24 (4): the clusters were not stable, P3).
Design D-26 (2), chosen because independent Gaussian peak triples gave spectra 32-59 dB too high:
- floor: multivariate Gaussian on the spline coefficients of the fixed robust floors (log10 PSD);
- peaks: the WHOLE peak set of one real vehicle (free peaks and every motor harmonic, as (log10 f0,
  log10 A / floor(f0), log10 width)), with one common frequency scale per spectrum (log10 sd ``freq_sd``)
  and independent height jitter (log10 sd ``height_sd``); peaks moved outside the band are dropped;
- chip: that of the source vehicle, with its fitted low-pass magnitude (D-25).
Vehicles are drawn uniformly, so chip shares follow the fit half. Fits flagged ``dropped`` (error above 5 dB,
D-26 (4)) are not used.
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


def fit_generator(entries, band, freq_sd, height_sd):
    """entries: fitted vehicle models (dicts with chip, lowpass, motor_hz, n_harm, n_free, params[, dropped]).
    freq_sd, height_sd: jitter in log10 units (measured within vehicles, see generator_check.py)."""
    k = nm.n_floor(FLOOR)
    use = [i for i, e in enumerate(entries) if not e.get("dropped")]
    floors = np.array([np.asarray(entries[i]["params"][:k], float) for i in use])
    chips = [entries[i]["chip"] for i in use]
    lowpass = {}
    for i in use:
        e = entries[i]
        lowpass.setdefault(e["chip"], e["lowpass"])
        if lowpass[e["chip"]] != e["lowpass"]:
            raise ValueError(f"chip {e['chip']}: entries with different low-pass terms")
    return dict(band=list(band), floor_mean=floors.mean(0), floor_cov=np.cov(floors.T) + 1e-9 * np.eye(k),
                sources=use, peak_sets=[entry_peaks(entries[i], band) for i in use], chips=chips,
                chip_share={c: chips.count(c) / len(chips) for c in sorted(set(chips))}, lowpass=lowpass,
                freq_sd=float(freq_sd), height_sd=float(height_sd), n_entries=len(use))


def sample(gen, n, f, rng):
    """n synthetic spectra on frequencies f (inside the band). Returns S (n, len f) and the sampled settings
    (meta: chip, floor coefficients, peaks as (f0 Hz, log10 ratio, width Hz), source = index in the entries)."""
    band = tuple(gen["band"])
    S, meta = np.empty((n, len(f))), []
    for i in range(n):
        j = int(rng.integers(len(gen["sources"])))
        chip = gen["chips"][j]
        coef = rng.multivariate_normal(gen["floor_mean"], gen["floor_cov"])
        spec = 10 ** nm.floor_log(f, coef, FLOOR, band)
        scale = 10 ** rng.normal(0.0, gen["freq_sd"]) if gen["freq_sd"] > 0 else 1.0
        pk = []
        for lf0, lr, lg in gen["peak_sets"][j]:
            f0 = 10 ** lf0 * scale
            lr = lr + (rng.normal(0.0, gen["height_sd"]) if gen["height_sd"] > 0 else 0.0)
            if not band[0] <= f0 <= band[1]:
                continue
            g = float(np.clip(10 ** lg, *G_RANGE))
            A = 10 ** (lr + nm.floor_log(np.array([f0]), coef, FLOOR, band)[0])
            spec = spec + nm.lorentz(f, A, f0, g)
            pk.append((float(f0), float(lr), g))
        S[i] = spec * nm.lowpass_gain2(f, gen["lowpass"][chip])
        meta.append(dict(chip=chip, floor=coef.tolist(), peaks=pk, source=gen["sources"][j], scale=float(scale)))
    return S, meta
