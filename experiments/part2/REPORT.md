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
