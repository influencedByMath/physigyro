# Evidence pack for decisions A2, A3 and A4 (Phase 4, before Task 3)

Made 2026-10-18 (Day 21), branch `phase4`; **revised the same day after an independent review.** Written for: the author, to decide A2 (PM floor), A3 ($M_{s,max}$) and A4 (plant uncertainty set, worst-case limits, ICM chip delay). **Status 2026-10-18 (author):** A2 = D-30 (PM floor 30°), A3 = D-31 ($M_{s,max}$ 2.0), separate worst-case limits = D-32 (value $M_{s,wc}$ 3.0 in D-36), chip-delay rule = D-33. The rest of A4 and the findings F1-F4 were decided as **D-36** after the follow-up (`notes/followup_A4_F1_F4.md`).

**Rules followed**
- Fit half only. Every input goes through `priors.require_fit_half` / `require_fit_rows`.
- Raw logs are protected differently: their file names are UUIDs, so the path guard cannot see them. Only log IDs listed in `fit_logs.json` are opened.
- Every number comes from code in this repository: `experiments/phase4/plant_variation.py`, `robust_limits.py`, `prior_sensitivity.py` and `evidence_tables.py`. The full generated tables are in `experiments/phase4/evidence_tables.md`; this note quotes them.
- New code is test-first where possible; several helper tests came after their code (the plan ledger records which). Every new module has a mutation check:

| Check | Caught |
|---|---|
| `src/sysid.py` | 13/13 |
| `plant_variation.py` helpers | 5/5 |
| `src/robust.py` | 12/13 (one documented equivalent mutant) |
| `robust_limits.py` helpers | 16/16 |
| `evidence_tables.py` helpers | 12/12 |

**What the independent review changed** (no critical findings; six important ones, all fixed):
1. **Funnel:** 46 of 102 logs had left no record. Every log is now in the funnel.
2. **Skipped publications:** many logs pause the controller for 5-6 periods about once per second. Holding up to 8 skipped samples, instead of 3, recovered **3 more vehicles**.
3. **The A4 comparison is now made where the margins bind** (each drone's own $M_s$-peak and phase-crossover frequencies, with confidence intervals), not only at 10 Hz. **This changed the A4 width recommendation from ±15 % to ±20 %.**
4. **Finding F3 was overstated;** it is restated below.
5. **The control IMU's filter rate is now really used.** The first version never loaded `vehicle_imu_status`.
6. **Real-gain fixes:**
   - the identified T_d is corrected so that it contains the control chip's filter once;
   - rows whose gains changed in flight are excluded from the rates.

---

## 1. What was done

### 1.1 Real plant variation (Tier 2 identification on the fit half)

The feasibility test identified one vehicle (b27d09e3) from hand-picked windows. Real logs have two kinds of gaps:
- **Logger dropouts:** long gaps.
- **Skipped controller publications:**
  - In some logs about 10 % of the torque-setpoint intervals are two periods long.
  - In many logs the torque setpoint pauses for 5-6 periods about every 1.02 s. The controller's own input (`vehicle_angular_velocity`) is missing in the same pauses: 79 of 83 in log 852b8368 and 227 of 232 in d1af0dea, mostly without a logger-dropout record.
  - So the controller did not run, and the motors held the last command.

New code in `src/sysid.py` (23 tests, mutation check 13/13):
- **`uniform_pieces`:** puts the torque setpoint on a uniform controller grid. Up to 8 skipped publications are held; a longer gap starts a new piece. Each sample gets its own slot, so jitter and clock offset cannot shift the grid.
  - An end-to-end PX4-timing simulation recovers T_d within 0.3 ms, including with skipped samples and ±4 % jitter (test, adapted from the review).
- **`body_rate_on_grid`:** low-passes the raw FIFO gyro with a zero-phase filter before resampling. Vibration near the controller rate cannot fold into the fit band, and the filter adds no delay.
- **`piece_spectra`:** Welch spectra averaged over every gap-free piece (1.28 s segments, as in the feasibility test).
  - A shorter 0.8 s segment was tried and **rejected**. At its 1.25 Hz resolution, fewer bins pass the D-14 rule of 6 coherent frequencies, so 9 fits were lost. The segment length is part of the D-14 gate.
- **`instrument_for_mode`, `instruments_agree`:** the D-14 rule.
  - Attitude setpoint in Stabilized/Altitude, rate setpoint in Acro.
  - In Position/auto/offboard modes both instruments must agree within 1 dB / 5°.
- **`fit_frf`:** fit of $(b, \tau_m, T_d)$ at 0.5-20 Hz where the coherence is at least 0.5, with at least 6 frequencies (D-14).

The runner fits every fit-half log whose logger profile writes the controller signals at full rate:
- roll and pitch separately, on the airborne time with 5 s trimmed at take-off and landing;
- **repeated identification:** the same fit on the first and on the second half of the usable time.

**Provisional acceptance gates** (`plant_variation.accept`; the author decides the final ones):
- a fit exists;
- in Position/auto modes, the two instruments agree;
- fit error at most 3 dB and 15°;
- τ_m and T_d not on a fit bound.

### 1.2 Who passes which limits (`robust_limits.py`; `src/robust.py`)

**The uncertainty set** is THESIS 7.1.1 Tier 3 step 4:
- the nominal plant plus 8 corners: $b(1\pm\delta)$, $\tau_m(1\pm\delta)$, $T_d + \{0, \Delta_d\}$;
- widths δ = 10, 15, 20, 30 %;
- delay steps $\Delta_d$ = 0.5, 1, 2 controller samples.

**The limits:**
- PM floor 25 / 30 / 35°, $M_{s,max}$ 1.8 / 2.0 / 2.2, and GM ≥ 6 dB always (THESIS 7.3).
- The worst-case limits are either the **same** as the nominal ones, or **separate**: $M_s \le M_{s,wc}$ (2.5 or 3.0), together with the PM and GM that this $M_s$ guarantees.
  - The distance from the Nyquist curve to −1 is $1/M_s$ (Åström and Murray, *Feedback Systems*, Sec. 12.2).
  - Therefore $PM \ge 2\arcsin(1/(2M_s))$ and $GM \ge M_s/(M_s-1)$.
  - $M_{s,wc}$ 2.5 gives PM ≥ 23.1° and GM ≥ 4.4 dB; 3.0 gives PM ≥ 19.2° and GM ≥ 3.5 dB.

**Check of the code:** it reproduces the feasibility realism check exactly (test): nominal PM 39.5°, $M_s$ 2.10; worst of the 8 corners at ±20 % / +1 sample: PM 23.4°, $M_s$ 3.87.

**The robust check also reports where the margins bind:** the frequency of the $M_s$ peak, and of the phase crossover that sets the smallest GM.

**Populations:**
1. **Real-gain fit-half drones:** every accepted identification, with:
   - the log's own gains;
   - the pilot's actual filters ($\theta_{actual}$, D-15);
   - the measured controller rate (D-28);
   - the control IMU's raw rate as filter rate: matched in 23 of 30 rows; otherwise the FIFO rate, flagged;
   - the mean motor frequencies if the ESC-RPM notch is on.

   Two corrections apply:
   - **The identified T_d is corrected** so that it contains the control chip's on-chip filter once: T_d − delay(FIFO chip) + delay(control chip), with the lower delay bounds of Task 1. The uncorrected result is kept as a sensitivity.
   - **Rows whose loop gains or gyro filters changed during the flight are left out of the rates:** a3f3ae23, 2 rows.

   The b27d09e3 plants of the feasibility test are added as reference rows.
2. **Synthetic Tier 3a drones** (1,000, seed 20261018): the D-28 priors, gains by the crossover design, kept if the unfiltered PM is 45-75°, evaluated at the PX4 v1.16 defaults (their reference and safety-check fallback, D-09).
   - Chip-delay variants are evaluated for ICM-4268x drones.

---

## 2. Real plant variation (tables A, B, B2, C)

**Coverage.** 102 fit-half logs write the controller signals at full rate. Each log at the best stage it reached:

| Stage | Logs |
|---|---|
| **Accepted identification** | **17** |
| Fit failed the gates | 5 |
| Too little excitation (fewer than 6 coherent frequencies) | 20 |
| No airborne gap-free piece of one Welch segment | 38 |
| Torque setpoint below 200 Hz | 2 |
| Missing topic (older firmware) | 20 |

- **30 accepted identifications from 17 vehicles:** 13 rows in Stabilized/Altitude, 17 in Position/auto modes, where the two instruments agreed within 0.06-0.39 dB and 0.7-3.7°.
- **The gates:** the fit-error thresholds barely matter (29 / 30 / 31 rows at 2 dB / 10°, 3 dB / 15°, 4 dB / 20°). The τ_m-bound gate does the work: 6 of the 8 rejected fits have τ_m on its bound, with fit errors of 13-50°.

**Cross-check with the feasibility test** (unchanged by the revision). The automatic method on b27d09e3:

| Axis | b | τ_m | T_d |
|---|---|---|---|
| Roll | 464 (feasibility 431) | 32.0 ms (29.2) | 1.32 ms (2.61) |
| Pitch | 443 (feasibility 474) | 32.6 ms (34.2) | 1.10 ms (0.0) |

**Repeated identification** (first vs second half of the same flight, 18 pairs). Absolute differences:

| Quantity | Median | 90th percentile | Largest |
|---|---|---|---|
| b | 7.1 % | 28.0 % | 51.6 % (ab1aecd2 pitch) |
| τ_m | 6.3 % | 25.4 % | 34.9 % (c901f9c1 pitch) |
| T_d | 0.4 ms (0.2 samples) | 2.6 ms (1.3 samples) | 3.0 ms |
| Plant gain at 10 Hz | 0.3 dB | 1.0 dB | 5.2 dB (ab1aecd2) |
| Plant phase at 10 Hz | 1.7° | 6.1° | 8.5° (d32cc24c) |

- The parameters move a lot, but the plant response moves less: τ_m and T_d trade off below the 20 Hz band edge.
- **Roll vs pitch** of the same vehicle (13 vehicles): τ_m differs by a median 8.8 % (90th percentile 31.5 %), T_d by 0.5 ms (1.9 ms).

**Where the margins bind (table B2).** The $M_s$ peak of the real flown settings lies at 5.6-21.0 Hz (median 13.2 Hz), above the 10 Hz used before. The half-flight differences grow with frequency:

| Frequency | Phase, p90 [95 % CI] | Gain, p90 [95 % CI] | Set ±15 % +1 sample | Set ±20 % +1 sample |
|---|---|---|---|---|
| 5 Hz | 3.8° [2.4-5.0] | 1.22 dB [0.76-4.87] | 1.88 dB / 6.1° | 2.56 dB / 7.3° |
| 10 Hz | 6.1° [2.2-8.5] | 1.01 dB [0.67-5.15] | 2.28 dB / 8.1° | 3.08 dB / 9.2° |
| 15 Hz | 9.5° [6.3-12.8] | 1.23 dB [0.58-5.28] | 2.44 dB / 9.6° | 3.29 dB / 10.4° |
| 20 Hz | **13.7° [9.8-18.9]** | 1.33 dB [0.60-5.34] | 2.51 dB / 11.3° | 3.38 dB / 12.0° |

(Set sizes for the median plant: τ_m 23.3 ms, T_d 1.91 ms, 800 Hz. With n = 18, a 90th percentile is very uncertain; see the intervals.)

**The direct test, drone by drone.** Does each drone's half-flight change at its OWN $M_s$-peak frequency (or critical phase crossover) lie inside a set built around its OWN plant?

| Set | At the $M_s$ peak | At the phase crossover |
|---|---|---|
| ±10 % +0.5 sample | 83 % | 78 % |
| ±10 % +1 sample | 89 % | 89 % |
| **±15 % +1 sample** | **89 % (16 of 18)** | **89 %** |
| **±20 % +1 sample** | **94 % (17 of 18)** | **94 %** |
| ±30 % +1 sample | 94 % | 94 % |
| ±20 % +2 samples | 94 % | 94 % |

- Outside ±15 %: ab1aecd2 pitch (+5.35 dB at 21 Hz) and d32cc24c roll (−10.6° at 14 Hz, against 10.2°).
- Outside ±20 %: only ab1aecd2 pitch.

**The margins often rest on extrapolation:** the $M_s$ peak lies above the highest coherent fit frequency in **12 of 28** cases, and the critical phase crossover in **22 of 28**.

**Limits of this evidence:**
1. **Both halves come from one flight.**
   - Changes between flights (battery, payload, temperature, propeller wear) are not covered, which makes the spread too small.
   - Each half has only half of the data, so the spread also contains estimation noise, which makes it too large.
   - So it is neither a strict lower nor a strict upper bound.
2. The error against the true plant is unknown.
3. **T_d is weakly fixed** (fit band at most 0.5-20 Hz); 5 accepted fits sit on the T_d = 0 bound. A real T_d is at least the computation plus chip delay, so those drones' margins at 13-18 Hz may be optimistic.
4. In 18 of 30 identifications the logged FIFO IMU is not (or not known to be) the control IMU. The correction of Section 1.2 handles the known pairs; 7 rows have an unknown control IMU.
5. In Position/auto modes both instruments are fed back. Their agreement cannot detect a bias they share (D-14).
6. **Not implemented:** THESIS 7.1.1 Tier 2 step 3 asks to exclude motor saturation.
7. **The corner set:** with notches, the worst plant can lie inside the box rather than on a corner.
8. 2 vehicles fly PX4 v1.17 (same filter structure as the v1.16 model). 17 vehicles in total.

## 3. ICM chip delay: the two rules and three open points (table D, G)

| Chip (fit vehicles) | BIC rule (within 10) | Fit-error rule (within 1.5 %) | Fit-error terms at 5 Hz |
|---|---|---|---|
| ICM-42688-P (13) | 1 term, 0.99-1.04 ms | 3 terms, 0.91-1.37 ms | 0.90-1.16 ms |
| ICM-42686-P (3) | 2 terms, 2.00-3.02 ms | 2 terms, 2.00-3.02 ms; not bounded below in corner | 1.94-2.51 ms |
| ICM-45686 (9) | 1 term, 0.88-0.89 ms | 10 terms, 0.25-0.89 ms | 0.25-0.88 ms |
| IIM-42652 (2) | 1 term, 1.14-1.16 ms | 1 term, 1.14-1.16 ms | 1.14 ms |

**Reading the table:**
- **Every value is a lower bound:** it is the delay of the minimum-phase filter with the fitted magnitude. The largest value in a range is the largest lower bound, not an upper bound on the true delay.
- The 20-100 Hz band of the plan overstates the delay that matters near the crossover (compare the 5 Hz column).
- **The BIC rule is falsely narrow:** its gains are inflated because neighbouring Welch bins are correlated. The fit-error rule is the one D-25 used to call the corner "not identifiable".

**Who is affected** (fit half; the test half is not counted):
- The ICM-4268x is the noise IMU in 28 of 111 fit logs, and the control IMU in 10 of the 71 logs where the control IMU is known.
- 32 % of synthetic drones (16 of 50 generator sources).
- Among the identifications: 4 rows from only 2 vehicles have an ICM-4268x FIFO IMU, and 1 row has an ICM-4268x control IMU.
- Their identified T_d (median 2.80 ms against 1.44 ms) is consistent with about 1 ms of extra delay, but it rests on 2 vehicles.

**What changes the outcome.** Synthetic pass rate at PM 30°, $M_s$ 2.0, ±20 %, +1 sample (separate limit M_s,wc 3.0):

| Variant | Pass rate |
|---|---|
| No chip delay | 48 % |
| ICM-42688-P, as a nominal shift (either rule) | 41 % |
| ICM-42688-P, inside the delay step | 45 % (BIC) / 44 % (fit-error) |
| ICM-42686-P | 32 % (shift) / 35 % (in step) |

**The rule moves the result by at most 2 points in table G; the treatment by 3-4 points; the chip itself by 7-16 points.**

**Three open points (the review):**
1. **FIFO IMU ≠ control IMU.**
   - The identified T_d contains the FIFO chip's filter, not the control chip's. It is now corrected (Section 1.2).
   - 5 rows were changed: 225bab55 and c901f9c1 by −0.91 ms (1 clamped at 0), df9a5d9e by +0.91 ms.
   - The effect on the rates is small: nominal 82 % with the correction against 79 % without; robust ±20 % with M_s,wc 3, 78 % against 77 % of the nominal passes.
2. **Synthetic priors taken from Tier 2** (THESIS Tier 3 step 1, or a revision of D-28 after F1):
   - Identified T_d values already contain the FIFO chip's filter and the computation delay.
   - If the synthetic T_d prior is taken from them, the chip increment must not be added again (or the chip delay must be removed first).
3. **Other chips with a fitted term.**
   - ICM-45686 (0.25-0.89 ms; 9 sources) and IIM-42652 (about 1.14 ms; 2 sources) also have low-pass terms; that is 22 % of the generator sources.
   - D-25 named only the ICM-4268x. Consistency suggests the same treatment for every chip whose term passed the BIC ≥ 10 rule (D-26 (3)).

## 4. Who passes which limits (tables E, F)

### 4.1 Real flown settings (28 identifications, the pilot's own filters; 2 rows left out because gains changed in flight)

**The PM floor never matters for these drones.** The rows for 25, 30 and 35° are identical in every table, because $M_s$ binds first.

**Nominal limits only:** **82 %** pass at M_s,max 2.0 (23 of 28); 86 % at 2.2; 61 % at 1.8.

**The 5 nominal failures** are "reference outside limits" under D-15, whatever A4 is. Their nominal PM / $M_s$:
- ab1aecd2 pitch: 34.4° / 2.59 (FFT notch on, not modelled);
- c901f9c1 roll and pitch: 20.8° / 4.73 and 50.9° / 2.13. The roll value is suspicious (see F5);
- d32cc24c roll and pitch: 39.9° / 3.20 and 55.1° / 2.27 (FFT notch on; ESC notch on without valid RPM).

**Do the candidate limits reject most real settings?** Overall pass rate and, in brackets, the share of the 23 nominal passes:

| Limit variant | ±10 % +0.5 | ±10 % +1 | ±15 % +1 | ±20 % +1 | ±30 % +1 | ±20 % +2 |
|---|---|---|---|---|---|---|
| Same limits for every plant (THESIS candidate) | 50 % (61 %) | 43 % (52 %) | 36 % (43 %) | 32 % (39 %) | 18 % (22 %) | 21 % (26 %) |
| Separate worst-case limit, M_s,wc 3.0 | 82 % (100 %) | 82 % (100 %) | 79 % (96 %) | 64 % (78 %) | 46 % (57 %) | 54 % (65 %) |
| Separate worst-case limit, M_s,wc 2.5 | 82 % (100 %) | 71 % (87 %) | 61 % (74 %) | 50 % (61 %) | 43 % (52 %) | 43 % (52 %) |

- **With the same limits for every plant, the candidate set rejects most real settings.** At ±20 % / +1 sample, only 32 % pass, i.e. 39 % of the nominal passes (9 of 23).
- **Narrowing the set alone barely helps:** ±15 % / +1 sample keeps 43 %; ±10 % / +0.5 sample keeps 61 %.
- **Separating the worst-case limit helps much more.** With M_s,wc 3.0:
  - ±20 % / +1 sample keeps 78 % of the nominal passes (18 of 23);
  - ±15 % / +1 sample keeps 96 % (22 of 23).

**b27d09e3, the pilot's roll setting:**
- Feasibility plant: nominal 39.5° / 2.10; worst case at ±20 % / +1 sample 23.4° / 3.87.
- Automatic plant: nominal 42.6° / 1.88; worst case 28.6° / 3.03.

### 4.2 Synthetic Tier 3a drones at the PX4 default filters (1,000)

**Nominal limits only:** 50 % pass (PM 30°, M_s 2.0); 59 % at PM 25°, M_s 2.2.

| Limit variant | ±15 % +1 | ±20 % +1 |
|---|---|---|
| Same limits for every plant | 24 % (49 % of the nominal passes) | 20 % (41 %) |
| Separate worst-case limit, M_s,wc 3.0 | 50 % (99 %) | 48 % (97 %) |

**By crossover** (nominal, PM 30°, M_s 2.0): 90 % at 4-6.5 Hz, 23 % at 6.5-10 Hz, 0 % at 10-15 Hz. See F3.

## 5. New findings (landmines; they need the author before Tasks 3-4)

**F1. The D-28 plant priors are slower than many real plants** (table C).
- The identified τ_m spans 11.7-64.9 ms (median 23.3 ms); 60 % lie inside 20-60 ms.
- T_d spans 0-8.1 ms; 67 % lie inside 0.5-5 ms.
- In the robust comparison, the motor-plus-delay lag at 10 Hz, **9 of 30 real plants lag less than the fastest plant D-28 can produce** (53.3°).
- A prior of τ_m 10-60 ms with T_d 0.1-5 ms would reach down to 32.5°.

**F2. The Tier 3a gain design is unlike the real drones** (table H).
- The real drones' PM without filters is 48.6-106.6° (median 72.8°); only 53 % lie inside the 45-75° window.
- Their integral corner is 0.030 of the crossover; the design uses 0.1.
- The D corner matches: median 1.16 of the crossover, 83 % inside 0.5-3.
- A design matched to the real drones (window 60-95°, the real 10-90 % range; integral corner 0.03) raises the share of synthetic drones that are safe at their defaults from **50 % to 76 %**.

**F3 (restated after the review). With the PX4 defaults as reference, synthetic drones at f_c ≥ 10 Hz are rare, not impossible.**
- At 10-15 Hz, **0 of 117-196** synthetic drones pass the nominal limits in designs a-c (THESIS, matched window, matched window and integral corner).
- With the faster priors of F1 (design d), **6.5 %** pass at PM 25° / M_s 2.2 (14 of 214). None pass at PM 30° / M_s 2.0, and 0.5 % pass robustly (±15 %, M_s,wc 3).
- Above 10 Hz it is $M_s$ and GM that fail, not PM.
- Two real drones fly near 10 Hz on the defaults: 94bd7879 pitch (9.3 Hz, PM 59.5°, $M_s$ 1.80) and 4e73d30d (9.5 Hz, 53.3° / 1.93).
- That real drones above 10 Hz raise their cutoffs rests on 2 vehicles: b27d09e3 (80/40 Hz) and ab1aecd2 (50/30 Hz, which fails anyway).
- Options for Tier 3 step 3 together with D-09:
  1. Keep 4-15 Hz and state that the kept drones end near 10 Hz.
  2. Sample f_c in 4-10 Hz.
  3. Give synthetic drones a pilot-like reference whose cutoffs grow with f_c. This changes D-09.
  4. Keep unsafe-default drones, flagged "reference unsafe".

**F4. Real crossovers below 4 Hz exist.**
- The flown settings cross over at 3.1-15.9 Hz (median 6.5 Hz); 4 of 30 lie below 4 Hz.
- For them, f_lo = 5 f_c falls below the validated 20 Hz band edge. Task 3 needs a rule, e.g. f_lo = max(5 f_c, 20 Hz).

**F5. Single identifications can be wrong.**
- c901f9c1 roll: its b is 1.82 times the pitch value, and it gives a nominal PM of 20.8° for a drone that flies.
- 3 accepted vehicles use the FFT notch, which is not modelled.
- 5 fits sit on the T_d = 0 bound.
- Model-predicted margins of real drones are therefore uncertain beyond the uncertainty set. This is the motivation for E10/E11.

## 6. Recommendations (the author decides; data-backed)

| Decision | Recommendation | Main evidence |
|---|---|---|
| **A2** PM floor | **30°** | It never changes a real-drone result: $M_s$ binds first. Åström and Murray: "a phase margin of 30 to 60 is typically required for robustness to modeling errors and non-oscillatory response" (2006 draft, Sec. 11.2). $M_s$ ≤ 2 already implies PM ≥ 29°. |
| **A3** M_s,max (nominal) | **2.0** | The textbook pairing with GM ≥ 6 dB. 23 of 28 real flown settings meet it nominally. 2.2 would add one (c901f9c1 pitch, 2.13) and the hand-identified b27d09e3 roll (2.10). |
| **A4** uncertainty set | **±20 % on b and τ_m, delay step 1 controller sample** (the THESIS candidate, kept) | See below. |
| **A4** worst-case limits | **Separate: every plant of the set Nyquist-stable with $M_s$ ≤ 3.0** (so PM ≥ 19.2°, GM ≥ 3.5 dB); the nominal plant keeps PM ≥ 30°, $M_s$ ≤ 2.0, GM ≥ 6 dB | Keeps 78 % of the nominally passing real flown settings and 97 % of the synthetic defaults. With the same limits for every plant, only 39 % and 41 % survive, so the safety check would mostly return "reference unsafe". |
| **A4** ICM chip delay | **Fit-error rule, near-crossover values.** Synthetic drones whose chip has a fitted term (ICM-4268x, and for consistency also ICM-45686 and IIM-42652): nominal T_d + the lower bound, delay step + the rest of the range. Real-gain identified plants: the FIFO→control correction only. If D-28 is revised from Tier 2 T_d values: no chip increment on top. | The BIC rule is falsely narrow; the lower bound is a physical minimum for the measured magnitude; the rule changes pass rates by at most 2 points. This refines D-25 ("inside the delay uncertainty"). |

**Why ±20 % rather than ±15 %:**
- ±20 % covers the measured half-flight changes at each drone's own $M_s$-peak and phase-crossover frequencies in 17 of 18 pairs; ±15 % covers 16, and ±10 % / +0.5 sample covers 15 at the $M_s$ peak.
- The margins rest on the plant extrapolated beyond the identified band in most cases (12 / 22 of 28).
- Changes between flights are not measured.

**The price:** the separate limit then keeps 78 % instead of 96 % of the nominally passing real settings. **±15 % is the alternative** if fewer "reference unsafe" flags matter more than the one extra pair.

**Also needed before Task 4** (findings F1-F4):
1. Revisit D-28 τ_m / T_d (recommend τ_m 10-60 ms, T_d 0.1-5 ms).
2. Match the Tier 3a design to the real drones (recommend PM window 60-95°, integral corner 0.03).
3. Choose how to handle synthetic drones with f_c ≥ 10 Hz (F3, options 1-4).
4. A rule for f_lo when a real crossover is below 4 Hz (recommend f_lo = max(5 f_c, 20 Hz)).
