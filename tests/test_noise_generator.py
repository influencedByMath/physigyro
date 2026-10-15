"""Synthetic noise generator, decision D-26 (2): Gaussian floor + whole real peak sets with jitter; chip from the
source vehicle (one generator, chip as covariate, D-24 (4))."""
import numpy as np
import pytest

import noise_generator as ng
import noise_model as nm

BAND = (20.0, 780.0)
F = np.arange(0, 1000, 1.5)
FB = F[(F >= 20) & (F <= 780)]


def fake_fit(rng, chip, lowpass, peaks=(120.0, 300.0)):
    """A fitted-model entry as written by fit_model.py: spline floor, free peaks, no motors."""
    coef = list(rng.normal(-6.0, 0.2, nm.n_floor("spline")))
    pk = []
    for f0 in peaks:
        pk += [float(rng.normal(-5.0, 0.2)), float(f0), float(np.log10(rng.uniform(2, 6)))]
    return dict(chip=chip, lowpass=lowpass, motor_hz=None, n_harm=0, n_free=len(peaks), params=coef + pk)


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
    gen = ng.fit_generator(entries(), BAND, freq_sd=0.05, height_sd=0.3)
    S, meta = ng.sample(gen, 200, FB, np.random.default_rng(1))
    assert S.shape == (200, len(FB)) and np.isfinite(S).all() and (S > 0).all()
    for m in meta:
        assert all(BAND[0] <= f0 <= BAND[1] for f0, _, _ in m["peaks"])
        assert all(0.3 <= g <= 80.0 for _, _, g in m["peaks"])


def test_peak_sets_are_resampled_whole_with_one_frequency_scale():
    es = entries()
    gen = ng.fit_generator(es, BAND, freq_sd=0.02, height_sd=0.0)
    _, meta = ng.sample(gen, 100, FB, np.random.default_rng(2))
    for m in meta:
        src = ng.entry_peaks(es[m["source"]], BAND)
        ratio = np.array([f0 for f0, _, _ in m["peaks"]]) / 10 ** np.array([p[0] for p in src])
        assert np.allclose(ratio, ratio[0])                    # one common scale per spectrum
        assert np.allclose([lr for _, lr, _ in m["peaks"]], [p[1] for p in src])   # heights untouched (sd 0)
        assert m["chip"] == es[m["source"]]["chip"]            # chip from the source vehicle


def test_without_jitter_the_source_peaks_are_reproduced():
    es = entries()
    gen = ng.fit_generator(es, BAND, freq_sd=0.0, height_sd=0.0)
    _, meta = ng.sample(gen, 20, FB, np.random.default_rng(3))
    for m in meta:
        src = ng.entry_peaks(es[m["source"]], BAND)
        assert np.allclose([np.log10(f0) for f0, _, _ in m["peaks"]], [p[0] for p in src])


def test_generator_reproduces_the_population():
    gen = ng.fit_generator(entries(400, seed=2), BAND, freq_sd=0.01, height_sd=0.1)
    assert np.allclose(gen["floor_mean"], -6.0, atol=0.05)
    assert gen["chip_share"]["ICM42688P"] == pytest.approx(0.25)
    S, meta = ng.sample(gen, 4000, FB, np.random.default_rng(3))
    assert np.mean([m["chip"] == "ICM42688P" for m in meta]) == pytest.approx(0.25, abs=0.03)
    assert np.mean([len(m["peaks"]) for m in meta]) == pytest.approx(2.0, abs=0.01)


def test_chip_lowpass_applied_only_to_its_chip():
    gen = ng.fit_generator(entries(), BAND, freq_sd=0.01, height_sd=0.1)
    S, meta = ng.sample(gen, 400, FB, np.random.default_rng(4))
    hi = FB > 600
    icm = np.array([m["chip"] == "ICM42688P" for m in meta])
    ratio = np.median(np.log10(S[icm][:, hi].mean(1))) - np.median(np.log10(S[~icm][:, hi].mean(1)))
    assert ratio < -1.0                                       # 300 Hz order 3: about -2 decades above 600 Hz


def test_sampling_is_reproducible():
    gen = ng.fit_generator(entries(), BAND, freq_sd=0.05, height_sd=0.3)
    a, _ = ng.sample(gen, 5, FB, np.random.default_rng(9))
    b, _ = ng.sample(gen, 5, FB, np.random.default_rng(9))
    assert np.array_equal(a, b)


def test_peak_heights_are_relative_to_the_floor():
    gen = ng.fit_generator(entries(), BAND, freq_sd=0.01, height_sd=0.1)
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


def test_height_jitter_has_the_requested_spread():
    es = entries()
    gen = ng.fit_generator(es, BAND, freq_sd=0.0, height_sd=0.2)
    _, meta = ng.sample(gen, 2000, FB, np.random.default_rng(6))
    d = [lr - p[1] for m in meta for (_, lr, _), p in zip(m["peaks"], ng.entry_peaks(es[m["source"]], BAND))]
    assert np.std(d) == pytest.approx(0.2, rel=0.05) and abs(np.mean(d)) < 0.02


def test_dropped_fits_are_not_used():
    es = entries()
    es[0]["dropped"] = True
    gen = ng.fit_generator(es, BAND, freq_sd=0.0, height_sd=0.0)
    assert gen["n_entries"] == len(es) - 1
    _, meta = ng.sample(gen, 300, FB, np.random.default_rng(7))
    assert 0 not in {m["source"] for m in meta}


def test_peaks_scaled_outside_the_band_are_dropped():
    rng = np.random.default_rng(10)
    es = [fake_fit(rng, "BMI088", None, peaks=(150.0, 770.0)) for _ in range(10)]
    gen = ng.fit_generator(es, BAND, freq_sd=0.05, height_sd=0.0)          # +-12 % scale: 770 Hz often > 780
    _, meta = ng.sample(gen, 300, FB, np.random.default_rng(11))
    assert all(BAND[0] <= f0 <= BAND[1] for m in meta for f0, _, _ in m["peaks"])
    n = [len(m["peaks"]) for m in meta]
    assert 1 in n and 2 in n                                                 # dropped when scaled out, kept otherwise
