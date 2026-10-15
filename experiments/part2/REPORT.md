# Phase 3, Part 2: fit-half spectrum library, first clustering, and a stop for author decisions

Day 18, 2026-10-15. Plan: `notes/phase3_plan.md`, Part 2. Thresholds: decision **D-23**. Fit half only; no test-half log has been downloaded.

Sources of the numbers:
- `data/spectra/fit_funnel.csv` and `fit_windows.csv` (`build_library.py`);
- `evidence.json` (`evidence.py`);
- `clusters.json` (`cluster.py`).

Every number here is produced by code in the repository.

**Status: STOPPED at the Task 3 checkpoint.** A new landmine was found: the clusters follow the IMU chip (Section 3). Five method decisions are needed before clustering and the parametric fit can be finished (Section 5).

---

## 1. Fit-half data funnel (THESIS Table 1, fit half)

| Stage | In | Removed | Out | Reasons |
|---|---|---|---|---|
| Fit-half vehicles (one screened log each) | 84 | 0 | 84 | |
| Downloaded and parsed | 84 | 0 | 84 | all 84 files checked (size, ULog header) |
| Log gates Q4-Q7 (and Q10 topic) | 84 | 36 | 48 | Q4 dropout > 20 %: 35; Q4 no gap-free piece ≥ 0.67 s: 1 |
| At least one good window | 48 | 9 | **39** | 3 vehicles never flew; 6 flew only manoeuvres (no window ≤ 15 °/s) |
| Windows: gates Q10, Q9, Q13 | 6,616 | 2,466 | 4,150 | Q10 unsteady: 2,092; Q13 < 20 averages: 374; Q9 clipping: 0 |
| Windows: overlap + cap Q14 (3 per vehicle) | 4,150 | 4,044 | **106** | 31 vehicles give 3 windows, 5 give 2, 3 give 1 |

**Usable: 39 of 84 fit-half vehicles (46 %, 95 % CI 36-57 %).** The pilot projected 48 % (CI 31-66 %), so the full fit half confirms it.

**Projection to the go/no-go (THESIS 7.4.2 H):**
- test half (74 vehicles): about 34 usable (27-42);
- **total: about 73 vehicles (66-81).**

That is above the minimum of 30 but below the target of 100. The test half alone may fall below 30 (lower end of its range: 27).

The runner checks itself after every run:
- the funnel ends at the number of library windows;
- no vehicle has more than 3 windows;
- every selected window passes every gate;
- every PSD on the grid is positive.

All checks pass.

**Library** (`data/spectra/`, 3.4 MB in git): `fit_windows.csv` holds one row per window. Its columns are:
- the log id and URL, with the CC-BY 4.0 attribution (Q15);
- the vehicle key, date, board, IMU chip, sample rates and board rotation;
- the window times, activity, throttle and number of Welch averages;
- RPM (valid / tracking);
- the actual and default filters, PID gains and flags.

The PSDs are in `fit_psd_native.npz` (native 1.5 Hz bins up to Nyquist) and `fit_psd_grid.npz` (common grid 0-780 Hz in 1.5 Hz steps).

---

## 2. Landmines found and fixed in Part 2

Each fix has a test that failed before it.

| # | Problem | Fix |
|---|---|---|
| 1 | **Three numbers in the pilot REPORT were wrong.** Throttle 0.02 removes 9.5 % of calm windows (not 5 %), and 0.05 removes 2.0 % (not 1 %). The 2 kHz clock range "−0.4 to +0.8 %" came from two logs with 50 % and 63 % dropout | Corrected before the freeze; `verify_numbers.py` re-derives every claim. No threshold changed: f_max 800 Hz would drop 7 of the 8 good 2 kHz logs, whose clocks run 0.0-0.3 % slow. The measured rate agrees with a long-segment fit within 0.006 % on every log up to 20 % dropout. THESIS 7.4.4 corrected too |
| 2 | **The axis check (Q7) rejected 2 clean logs.** It used only the longest gap-free segment: one vehicle never flew, and the other had no pitch/yaw motion there | Now the check uses all airborne gap-free time. It is *inconclusive* when roll or pitch moved less than 0.03 rad/s, and then the D-23 rule decides (only rotations verified on data pass). All 17 pilot verifications still pass (diagonal 0.85-1.00) |
| 3 | **A log without `vehicle_rates_setpoint` crashed** (it also never flew) | Never airborne → no windows. Airborne but no rate setpoint → reason `Q10_no_rate_setpoint` (0 logs so far) |
| 4 | **Selecting windows before gating** (as in the pilot) let a gappy quiet window block an overlapping good one | Every candidate is gated first, then the 3 quietest non-overlapping windows are selected |
| 5 | **The parametric fit was far too slow:** it rebuilt the spline basis at every model evaluation, about 6 min per spectrum. The floor parameters also had no bounds | Basis computed once and floor bounded; now about 16 s, but the fit still stops at the evaluation limit. Needs an analytic Jacobian. That is Task 5 work, after decision 3 below |

---

## 3. NEW LANDMINE: the main cluster split is the IMU chip, not vibration physics (gate Q12)

**Evidence** (`clusters.json`, `evidence.json` E-G):

- **All four clustering variants give k = 2 with weak structure.** The variants are roll/pitch averaged or separate, and windows or vehicle means. Silhouette is 0.33-0.35. Bootstrap stability over vehicles: median ARI 0.68-0.73, but the worst 10 % of bootstraps fall to 0.10-0.34.
- **The small cluster is the ICM-4268x chips.** Vehicle means: 10 vehicles, of which 9 are ICM-42688-P / 42686-P and 9 sample at 8-8.4 kHz. No 2 kHz vehicle is in it. Its centroid falls by about 2 decades above 300 Hz.
- **High-frequency roll-off per vehicle** (band power above 500 Hz minus 100-300 Hz, in decades):
  - ICM-4268x (11 vehicles): median **−3.1** (from −3.9 to −1.1);
  - all other chips (28 vehicles): median **−0.15** (lowest −1.9).
- **The chip, not the mount, causes the roll-off.** On the same board model (Pixhawk FMU-V6X, one isolated IMU board):

  | Chip | Roll-off (decades) |
  |---|---|
  | BMI088 | +0.7, +0.9 |
  | ICM-45686 | −0.9 to +0.7 |
  | IIM-42652 | +0.2 |
  | ICM-42688-P | **−2.1, −3.3, −3.9** |

  The ICM-42688-P on CUAV X7 Pro, NORA and AMOVLAB boards drops just as much.
- **PX4 driver source** (v1.14 and v1.16, read via the GitHub API): the ICM-42688-P on-chip anti-alias filter is set to 585 Hz, and the user filter to its widest setting. The v1.13 driver does not set the anti-alias filter. **A 585 Hz filter alone cannot explain a 30-40 dB drop between 300 and 600 Hz** (ICM-42686-P: −4 decades already at 300 Hz). The exact on-chip filter path is therefore a new VERIFY item.
- **Clustering on 20-300 Hz only** (below the roll-off) weakens the chip dependence: the small cluster becomes 7 of 11 ICM-4268x. But the structure nearly vanishes: silhouette 0.23, and the worst 10 % of bootstraps have ARI 0.01.

**What it means:**

1. **Clustering.** With the full 20-780 Hz band, the "archetypes" mainly say which IMU chip the drone has. THESIS 7.5 step 5 requires revisiting the features in that case.
2. **Noise realism.** On ICM-4268x drones, PX4 itself receives this already-filtered signal, so the spectrum is the correct input noise for those drones. But the on-chip filter adds **phase delay that the loop model does not include**. This matters for the real-drone evaluation, and for any synthetic drone given an ICM-type spectrum.
3. **Gate Q8 gets more serious.** For several vehicles the logged FIFO chip is not the chip the controller used (CUAV X7 Pro and NORA: FIFO from ICM-42688-P, control on ICM-20689). The feasibility check (filtered gyro predicted within 1 dB) was on a vehicle whose two IMUs were the *same* chip type. Where the chips differ, the FIFO spectrum may not be the noise the controller saw. Such logs are flagged (`Q8_fifo_not_control`), not dropped.

---

## 4. Other evidence for the decisions

| | Finding | Number |
|---|---|---|
| A | Roll and pitch differ in the same window | Median 2.9 dB per band (p90 6.0 dB): level 2.3 dB, shape 2.5 dB |
| B | **The windows of one vehicle are near-copies** | For 100 % of windows, the nearest neighbour is a window of the same vehicle. Median feature distance 1.7 within a vehicle vs 9.5 between vehicles (10th percentile between: 6.6). Clustering windows therefore over-weights vehicles with 3 windows and inflates the silhouette |
| C | **A flat floor (THESIS 7.5) does not fit real spectra** | Robust floor fitted by a flat line: median 6.8 dB RMS error (p90 12.6); power law 5.1 dB (p90 9.2); 4-knot log-log spline 2.3 dB (p90 3.5). Floors span 2.4 decades (median; up to 6.6) |
| D | Usable count | Section 1 |

---

## 5. Decisions for the author (to be recorded as D-24)

1. **IMU chip roll-off (Section 3).** Options:
   - (a) **Model it explicitly (recommended).** Add a per-spectrum low-pass term to the parametric model (corner and order fitted, which also gives its delay). Cluster on the spectrum after removing the term (or on 20-300 Hz). Report the chip as a hardware covariate, never as an archetype.
   - (b) Cluster only on 20-300 Hz, and keep the full band for the noise metric.
   - (c) Keep it as is and report two hardware classes.
   - (d) Exclude ICM-4268x vehicles. This loses 11 of 39, so it is not recommended.
2. **Unit of clustering (Section 4 B).** Recommended: **one spectrum per vehicle** (power mean of its windows, roll and pitch power-averaged), because windows of a vehicle are near-copies. The alternative is all windows (inflated, vehicle-weighted).
3. **Noise-floor shape (Section 4 C).** Recommended: the **4-knot log-log spline** (2.3 dB, against 5.1 for a power law and 6.8 for the flat floor). The alternative is a power law (2 parameters, simpler generator, but 5.1 dB error).
4. **Archetypes or a continuum?** All variants show only weak structure (silhouette ≤ 0.35, unstable at the low end). If decision 1 (a) leaves no stable clusters, I recommend a single generator (Gaussian on all fitted log-parameters, with the chip class as a covariate) instead of per-cluster generators, and to report this honestly as a finding. The alternative is to keep k-means clusters whatever their strength.
5. **More data (go/no-go).** See Section 6.

---

## 6. Do we need more data? Honest recommendation

**Yes.** I recommend screening more vehicles now, before the test half is processed:

- **The projection misses the target.** About 73 vehicles in total (66-81) against the target of 100. The test half alone, about 34 (27-42), may fall below the minimum of 30.
- **The clusters are unstable at the low end** (bootstrap p10 ARI 0.1-0.3 with 39 vehicles). More fit-half vehicles directly improve the archetypes and the generator. Only 11 ICM-4268x vehicles describe that hardware class.
- **The cost is small and needs no new decision about the gates.** Each screened vehicle gives 0.0497 FIFO vehicles with a board ID, and 46 % of those are usable: 2.31 % per screened vehicle.

| Usable rate assumed | Screened vehicles needed for 100 in total | FIFO vehicles to download | Download |
|---|---|---|---|
| 46 % (measured) | about 1,150 | about 57 | about 4.6 GB |
| 36 % (low end of CI) | about 2,400 | about 119 | about 9.6 GB |
| 57 % (high end of CI) | about 350 | about 17 | about 1.4 GB |

- **The index has room.** 26,163 vehicle groups exist and 3,177 are screened.
- **The split stays stable.** New vehicles get a stable split automatically (D-20). About half land in the test half, and those are not downloaded until you say so.
- **Time:** the header screen is about 30-60 min of unattended network time. Downloading the new fit-half vehicles takes about 1 h.

**A cheaper complement (needs your decision, as it changes D-21):** 9 vehicles passed the log gates but their screened flight had no steady window or never flew. A rule "if the newest FIFO log gives no usable window, take the next-newest FIFO log of the same vehicle" (applied identically to both halves) could recover some vehicles without new vehicles. Dropout failures (35) are likely to repeat on the same board and SD card, so this mainly helps the 9.

---

## 7. What has NOT been done yet (on purpose)

- **Final clustering, Fig. 2 and the parametric model fits.** They wait for decisions 1-4. The code exists and is tested (`archetypes.py`, `noise_model.py`), but its results are not final.
- **Peak prominence and the fit-error drop limit.** Set in Task 5 on the fit half.
- **Test half.** Not downloaded (author instruction).

---

# Part 2b (after D-24 and D-25): enlarged fit half, model fits, clustering, generator. STOPPED (new landmine)

Status 2026-10-15. Everything is fit half only; the test half is not downloaded.

## 8. More data (D-24 (5))

- **Extended screen:** exactly the first 4,327 vehicles of the D-21 order. 272 FIFO vehicles; FIFO share 6.3 % (CI 5.6-7.1 %), unchanged. 1 log is unreadable (HTTP 403).
- **Split:** 216 vehicles, **111 fit / 105 test**. 58 are new; none of the 158 old assignments changed. 42 no-ID vehicles are excluded.
- **Fallback (D-24 P2, Q3 rule D-25):** 14 vehicles had zero steady windows. For 9 of them an older FIFO log was found; 2 of those were recovered; 5 vehicles had no older FIFO log. Q3 rejected 2 uploads, both verified byte-identical.
- **Funnel:** 111 vehicles → 64 pass the log gates (44 dropout, 3 no gap-free piece) → **52 usable vehicles (47 %, CI 38-56 %), 142 windows**.
- **Projection: about 101 vehicles in total (92-111).** The test half projects to about 49 (40-59).

## 9. Model fits (D-24 (1), (3); D-25)

`model_fits.json` (one spectrum per vehicle, spline floor, chip low-pass by BIC, at most 15 free peaks):

| Chip (vehicles) | Low-pass chosen | Gain at 500 / 780 Hz | BIC gain over "none" |
|---|---|---|---|
| BMI055 (12) | none | - | - |
| BMI088 (11) | 899 Hz, n 4 | −0.0 / −1.2 dB | **1 (a tie)** |
| ICM-42686-P (3) | 150 Hz, n 3 | −31 / −43 dB | 791 |
| ICM-42688-P (13) | 323 Hz, n 2 | −8 / −15 dB | 2,120 |
| ICM-45686 (9) | 696 Hz, n 6 | −0.1 / −6.9 dB | 158 |
| IIM-42652 (2) | 539 Hz, n 3 | −2 / −10 dB | 6 |
| IIM-42653 (2) | 696 Hz, n 6 | −0.1 / −6.9 dB | 30 |

- **Fit error:** median 2.85 dB (p90 3.79, max 4.80). Welch estimation noise is only 0.36 dB, so **the misfit is the model**. It is largest at 95-130 Hz: motor peaks are wider or split, because 4 motors run at slightly different speeds and the speed varies inside a window. 2 of 52 fits did not converge.
- **Peak cap (my ruling):** with a cap of 6, 42 of 52 fits hit it (`peak_cap_check.py`). With 15, only 2 do, the median count is 8, and the error barely changes (2.98 → 2.85 dB).
- **Prominence sensitivity:**

  | Prominence | Fit error | Peaks (median) |
  |---|---|---|
  | 0.3 | 2.66 dB | 13 |
  | 0.5 (used) | 2.85 dB | 8 |
  | 1.0 | 3.07 dB | 5 |

- **The ICM-42686-P magnitude is not stable either** (3 vehicles): its gain at 500 Hz was −11 dB with cap 6 and −31 dB with cap 15. D-25 already treats the corner and order as not identifiable; with this few vehicles even the magnitude is uncertain.

## 10. Final clustering (D-24 (2), P3)

`clusters_final.json`: one spectrum per vehicle (52), chip low-pass divided out.
- **Result:** k = 2, silhouette **0.19**, bootstrap median ARI **0.65** (p10 0.10).
- **The P3 rule therefore decides one generator with the chip as a covariate.**
- The correction weakens the chip dependence but does not remove it: one cluster holds 11 of the 13 ICM-42688-P.
- Without the correction: silhouette 0.33, ARI 0.84, and the clusters are essentially "ICM-4268x vs the rest". That hardware split is not stable by the rule either, and would not be a noise type anyway.

## 11. NEW LANDMINE: the generator, as specified in THESIS 7.5, does not reproduce real spectra

**Finding 1: free floor vs broad peaks is not identified.**
- With a free spline floor, 41 of 52 fitted floors hit the −20 coefficient bound, i.e. the floor is about 0 in places, and broad Lorentzian tails carry the spectrum there.
- The curve fit is good (median log10 level real −4.71 vs model −4.71), but "floor" and "peaks" lose their meaning: peak/floor ratios have a median of 2.8 decades and reach 10^18.

**Fixed floor (new tested option `fix_floor`):**
- The floor is the robust 20th-percentile envelope (as in evidence C), and only the peaks are fitted on top.
- Result: sane parameters (ratios median 1.45, max 5 decades; no bound hit), fit error 3.39 dB.
- The fixed-floor generator's floors match the real ones (median log10 at 100 Hz: −5.59 real vs −5.56 synthetic).

**Finding 2: independent Gaussian peaks are far too strong.** Sampling every peak independently from one pooled Gaussian on (log f0, log peak/floor, log width), as THESIS 7.5 suggests, gives spectra:

| Floor | Median band level off by | Real vs synthetic AUC |
|---|---|---|
| free | **59 dB** | 0.92 |
| fixed | **32 dB** | 0.90 |

`generator_check.json` is therefore an INVALID result, kept as evidence.

**Cause, verified:**
- The model rebuilt through the generator's representation reproduces each fit exactly (0.000 dB), so the representation is correct.
- Sampling breaks two things:
  - the structure inside a spectrum: motor harmonics belong at h × the same motor frequencies, and the peak heights belong together;
  - the linear power, through the Gaussian tails: peak/floor median 28×, but mean 926×.
- With **real vehicles' whole peak sets** placed on sampled floors, the median band difference is **2.5 dB**.

**Not done until decided:** the generator design, the synthetic-vs-real check and Fig. 2.

## 12. Decisions needed

1. **Floor fitting:** fixed robust floor (recommended; identifiable parameters, 3.39 dB) or free floor (2.85 dB, meaningless floor/peak split).
2. **Generator design:**
   - (A, recommended) Gaussian floor; **peak sets resampled as whole sets** from real vehicles, with jitter (one common frequency scale factor per spectrum, heights jittered by the within-vehicle spread); chip taken from the source vehicle.
   - (B) Keep independent peaks but model them better (mixture, truncation).
   - (C) Gaussian on PCA scores of band powers: matches the statistics by construction, but blurs peaks into 1.6-25 Hz bands, which is not good enough for notch tuning.

   (A) partly copies real peak sets. A leave-one-vehicle-out check must show that it generalises, and it will be reported.
3. **Chip low-pass selection margin:** BMI088 got a term by a BIC gain of 1. Recommended: require a BIC gain of at least 10 (strong evidence), else no term.
4. **Peak prominence** (7.4.4, D-23 left it open): 0.5 recommended (THESIS candidate; see the sensitivity table).
5. **Fit-error drop limit** (7.5): proposed **5 dB**. It drops no spectrum now (max 4.80 dB free floor; to be re-checked on the chosen floor); it only guards against broken fits.
