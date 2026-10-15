"""Synthetic noise generator (THESIS 7.5; D-24 (4): one generator, chip as covariate)."""
import numpy as np
import pytest

import noise_generator as ng
import noise_model as nm

BAND = (20.0, 780.0)
F = np.arange(0, 1000, 1.5)
FB = F[(F >= 20) & (F <= 780)]


def fake_fit(rng, chip, lowpass):
    """A fitted-model entry as written by fit_model.py: spline floor, 2 free peaks, no motors."""
    coef = list(rng.normal(-6.0, 0.2, nm.n_floor("spline")))
    peaks = []
    for f0 in (rng.uniform(100, 140), rng.uniform(280, 330)):
        peaks += [float(rng.normal(-5.0, 0.2)), float(f0), float(np.log10(rng.uniform(2, 6)))]
    return dict(chip=chip, lowpass=lowpass, motor_hz=None, n_harm=0, n_free=2, params=coef + peaks)


def entries(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return [fake_fit(rng, "ICM42688P" if i % 4 == 0 else "BMI088", (300.0, 3.0) if i % 4 == 0 else None)
            for i in range(n)]


def test_peaks_of_an_entry_include_motor_harmonics():
    e = dict(chip="X", lowpass=None, motor_hz=[100.0, 110.0], n_harm=2, n_free=1,
             params=[-6.0] * 8 + [-5.0, 0.5, -5.5, 0.7] + [-5.2, 233.0, 0.6])
    pk = ng.entry_peaks(e, BAND)
    assert len(pk) == 2 * 2 + 1                                # 2 harmonics x 2 motors + 1 free peak
    assert sorted(round(10 ** p[0]) for p in pk) == [100, 110, 200, 220, 233]


def test_generator_samples_valid_spectra():
    gen = ng.fit_generator(entries(), BAND)
    S, meta = ng.sample(gen, 200, FB, np.random.default_rng(1))
    assert S.shape == (200, len(FB)) and np.isfinite(S).all() and (S > 0).all()
    for m in meta:
        assert all(BAND[0] <= f0 <= BAND[1] for f0, _, _ in m["peaks"])
        assert all(0.3 <= g <= 80.0 for _, _, g in m["peaks"])


def test_generator_reproduces_the_population():
    gen = ng.fit_generator(entries(400, seed=2), BAND)
    assert np.allclose(gen["floor_mean"], -6.0, atol=0.05)
    assert gen["chip_share"]["ICM42688P"] == pytest.approx(0.25)
    S, meta = ng.sample(gen, 4000, FB, np.random.default_rng(3))
    share = np.mean([m["chip"] == "ICM42688P" for m in meta])
    assert share == pytest.approx(0.25, abs=0.03)
    n_peaks = np.mean([len(m["peaks"]) for m in meta])
    assert n_peaks == pytest.approx(2.0, abs=0.05)


def test_chip_lowpass_applied_only_to_its_chip():
    gen = ng.fit_generator(entries(), BAND)
    S, meta = ng.sample(gen, 400, FB, np.random.default_rng(4))
    hi = FB > 600
    icm = np.array([m["chip"] == "ICM42688P" for m in meta])
    ratio = np.median(np.log10(S[icm][:, hi].mean(1))) - np.median(np.log10(S[~icm][:, hi].mean(1)))
    assert ratio < -1.0                                       # 300 Hz order 3: about -2 decades above 600 Hz


def test_sampling_is_reproducible():
    gen = ng.fit_generator(entries(), BAND)
    a, _ = ng.sample(gen, 5, FB, np.random.default_rng(9))
    b, _ = ng.sample(gen, 5, FB, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_peak_heights_are_relative_to_the_floor():
    gen = ng.fit_generator(entries(), BAND)
    S, meta = ng.sample(gen, 300, FB, np.random.default_rng(5))
    got, want = [], []
    for s, m in zip(S, meta):
        if m["chip"] != "BMI088" or not m["peaks"]:
            continue
        fl = 10 ** nm.floor_log(FB, np.array(m["floor"]), "spline", BAND)
        f0, lr, g = max(m["peaks"], key=lambda p: p[1])                 # the highest peak dominates its bin
        i = np.argmin(np.abs(FB - f0))
        got.append(np.log10(s[i] / fl[i]))
        want.append(np.log10(1 + 10 ** lr))
    assert abs(np.median(got) - np.median(want)) < 0.3 and len(got) > 50
