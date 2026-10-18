# Evidence pack for decisions A2, A3 and A4 (Phase 4, before Task 3)

Made 2026-10-18 (Day 21), branch `phase4`. Written for: the author, to decide A2 (PM floor), A3 ($M_{s,max}$) and A4 (plant uncertainty set, including the ICM chip delay). **Nothing here is decided yet**; Task 3 waits for these decisions.

**Rules followed**
- Fit half only. Every input goes through `priors.require_fit_half` / `require_fit_rows`; the test half is not read or counted.
- Every number comes from code in this repository: `experiments/phase4/plant_variation.py`, `robust_limits.py` and `evidence_tables.py`. The full generated tables are in `experiments/phase4/evidence_tables.md`; this note quotes them.
- New code is test-first where possible, and every new module has a mutation check: `mutation_check_sysid.py`, `mutation_check_robust.py`.

---

## 1. What was done

### 1.1 Real plant variation (Tier 2 identification on the fit half)

The feasibility test identified one vehicle (b27d09e3) from hand-picked windows. Two problems stopped a batch run on more logs:
- real logs have **logger dropouts** (long gaps);
- they have **single skipped controller publications**: in some logs about 10 % of the torque-setpoint intervals are exactly two periods.

New code in `src/sysid.py` (20 tests, mutation check 12/12):
- `uniform_pieces`: puts the torque setpoint on a uniform controller grid. A skipped publication is held, because the motor command keeps its value; a dropout starts a new piece. Each sample gets its own slot, so timestamp jitter and a slow clock offset cannot shift the grid against the samples.
- `body_rate_on_grid`: low-passes the raw FIFO gyro with a zero-phase filter (no delay) before resampling. This removes vibration near the controller rate, which would otherwise fold down into the fit band (THESIS 7.1.1 Tier 2 step 2).
- `piece_spectra`: Welch cross-spectra averaged over all gap-free pieces. With one piece it equals `scipy.signal.csd`.
- `instrument_for_mode`, `instruments_agree`: the D-14 rule.
  - Attitude setpoint in Stabilized/Altitude, rate setpoint in Acro.
  - Position/auto/offboard modes need both instruments to agree within 1 dB / 5° (the D-14 candidate).
- `fit_frf`: the existing fit of $(b, \tau_m, T_d)$ at 0.5-20 Hz where the coherence is at least 0.5, with at least 6 frequencies (D-14).

The batch runner (`plant_variation.py`) fits every fit-half log whose logger profile writes the controller signals at full rate:
- roll and pitch separately, on the airborne time with 5 s trimmed at take-off and landing (D-23 Q10);
- **repeated identification:** the same fit on the first and the second half of the usable time.

**Provisional acceptance gates** (`plant_variation.accept`; the author decides final ones):
- a fit exists;
- in Position/auto modes, the two instruments agree;
- fit error at most 3 dB and 15° (the feasibility fits had 1.5-1.9 dB and 3.8-6°);
- τ_m and T_d are not on a fit bound.

### 1.2 Who passes which limits (`robust_limits.py`; `src/robust.py`, 8 tests, mutation check 10/11, one equivalent mutant)

**The uncertainty set** is THESIS 7.1.1 Tier 3 step 4:
- the nominal plant plus 8 corners: $b(1\pm\delta)$, $\tau_m(1\pm\delta)$, $T_d + \{0, \Delta_d\}$;
- widths δ = 10, 15, 20, 30 %;
- delay steps $\Delta_d$ = 0.5, 1, 2 controller samples.

**The limits:**
- PM floor 25 / 30 / 35°, $M_{s,max}$ 1.8 / 2.0 / 2.2, and GM ≥ 6 dB always (THESIS 7.3).
- The worst-case limits are either the **same** as the nominal ones, or **separate**: $M_s \le M_{s,wc}$ (2.5 or 3.0), together with the PM and GM that this $M_s$ guarantees.
  - Why $M_s$ guarantees margins: the distance from the Nyquist curve to −1 is $1/M_s$ (Åström and Murray, *Feedback Systems*, Sec. 12.2).
  - Hence at the gain crossover $|1 + e^{j\varphi}| = 2\sin(PM/2) \ge 1/M_s$, and at the phase crossover $1 - |L| \ge 1/M_s$.
  - So $PM \ge 2\arcsin(1/(2M_s))$ and $GM \ge M_s/(M_s-1)$: $M_{s,wc}$ 2.5 gives PM ≥ 23.1° and GM ≥ 4.4 dB; 3.0 gives PM ≥ 19.2° and GM ≥ 3.5 dB.

**Check of the code:** it reproduces the feasibility realism check exactly (test): pilot roll setting nominal PM 39.5°, $M_s$ 2.10; worst of the 8 corners at ±20 % / +1 sample: PM 23.4°, $M_s$ 3.87, GM 3.35 dB.

**Populations:**
1. **Real-gain fit-half drones:** every accepted identification, with the log's own gains, the pilot's actual filters ($\theta_{actual}$, D-15), the measured controller rate (D-28), the control IMU's raw rate as filter rate, and the mean motor frequencies if the ESC-RPM notch is on.
   - The b27d09e3 plants of the feasibility test are added as reference rows.
2. **Synthetic Tier 3a drones** (1,000, seed 20261018): the D-28 priors, gains by the crossover design, kept if the unfiltered PM is 45-75°, evaluated at the PX4 v1.16 defaults (their reference and safety-check fallback, D-09).
   - **Chip-delay variants** for ICM-4268x drones, with the delay of either rule:
     - as a **nominal shift** (lower bound) plus the rest in the delay step;
     - or entirely **inside the delay step**.

---

## 2. Real plant variation (tables A, B, C)

**Coverage.** 102 fit-half logs write the controller signals at full rate. They give **25 accepted identifications from 14 vehicles**:
- 11 in Stabilized/Altitude (attitude-setpoint instrument);
- 14 in Position/auto modes, where the two instruments agreed (RMS difference 0.06-0.39 dB, 0.7-3.7°).

**Why there are not more:**
- 18 logs lack `vehicle_torque_setpoint` (older firmware, 36 rows) and 2 lack `vehicle_status` (4 rows).
- 2 logs write the torque setpoint at only 50 Hz (4 rows), and 2 rows have no gap-free piece of one Welch segment.
- 49 class fits had too little excitation (fewer than 6 coherent frequencies).
- 6 fits failed the quality gates: τ_m on a bound, or a large fit error.

The gates are not decisive: 24 / 25 / 25 rows pass at 2 dB / 10°, 3 dB / 15° and 4 dB / 20°.

**Cross-check with the feasibility test.** The automatic method on b27d09e3 gives:

| Axis | b | τ_m | T_d |
|---|---|---|---|
| Roll | 464 (feasibility 431) | 32.0 ms (29.2) | 1.32 ms (2.61) |
| Pitch | 443 (feasibility 474) | 32.6 ms (34.2) | 1.10 ms (0.0) |

b and τ_m agree within 8 % and 3 ms. T_d differs, as expected: it is the weakly identified parameter.

**Repeated identification** (first vs second half of the same flight, 15 pairs). Absolute differences:

| Quantity | Median | 90th percentile | Largest |
|---|---|---|---|
| b | 7.8 % | 28.3 % | 51.6 % (ab1aecd2 pitch) |
| τ_m | 6.3 % | 25.7 % | 34.9 % |
| T_d | 0.5 ms (0.3 samples) | 2.4 ms (1.2 samples) | 3.0 ms |
| **Plant gain at 10 Hz** | **0.3 dB** | **1.0 dB** | 5.2 dB (ab1aecd2) |
| **Plant phase at 10 Hz** | **1.8°** | **5.1°** | 8.5° (d32cc24c) |

- **The parameters move a lot, but the plant response near the crossover does not.** τ_m and T_d trade off below the 20 Hz band edge, so their errors partly cancel in the phase. The margins depend on the response near the crossover, not on the parameters one by one.
- **Roll vs pitch** of the same vehicle (11 vehicles; same motors and ESCs): τ_m differs by a median 10.6 % (90th percentile 32.3 %), T_d by 1.3 ms (2.0 ms).

**Size of the candidate sets at 10 Hz** for the median identified plant (τ_m 21.2 ms, T_d 1.53 ms), as the largest gain change / extra phase lag over the 8 corners:

| Set | 800 Hz controller | 400 Hz controller |
|---|---|---|
| ±10 %, +0.5 sample | 1.46 dB / 4.8° | - |
| ±15 %, +1 sample | 2.23 dB / 8.3° | 2.23 dB / 12.8° |
| ±20 %, +1 sample | 3.02 dB / 9.4° | 3.02 dB / 13.9° |

- The smallest set (±10 %, +0.5 sample) is about the size of the measured 90th percentile of within-flight changes (1.0 dB / 5.1°).
- ±15 % / +1 sample is about twice as wide.
- A delay step counted in controller samples is larger for slower loops (400 Hz: 2.5 ms per sample).

**Limits of this evidence:**
1. Both halves come from **one flight**. Changes between flights (battery, payload, temperature, propeller wear) are not covered, so this is a lower bound of real variation.
2. The error against the true plant is unknown; only the repeatability is measured.
3. T_d is weakly fixed (fit band at most 0.5-20 Hz).
4. In 15 of 25 identifications the logged FIFO IMU is not (or not known to be) the control IMU. The identified T_d then contains the FIFO chip's on-chip filter, not the control chip's.
5. In Position/auto modes both instruments are fed back. Their agreement is a consistency check, not a proof (D-14).
6. 14 vehicles.

## 3. ICM chip delay: the two rules (tables D and G)

| Chip (fit vehicles) | BIC rule (within 10) | Fit-error rule (within 1.5 %) |
|---|---|---|
| ICM-42688-P (13) | 1 term, 0.99-1.04 ms | 3 terms, **0.91-1.37 ms** |
| ICM-42686-P (3) | 2 terms, 2.00-3.02 ms | 2 terms, 2.00-3.02 ms, **not bounded below in corner, so the delay can be larger** |

**The trade-off:**
- **BIC rule:** BIC treats neighbouring Welch bins as independent, so its gains are inflated (Phase 3 review, LOW 5). It therefore keeps only the single best term: a falsely precise range.
- **Fit-error rule:** this is the rule D-25 used to call the corner "not identifiable". It gives the honest, wider range.
- **Both rules give lower bounds:** the delay is computed for the minimum-phase filter with the fitted magnitude.

**Who is affected** (fit half only; the test half is not counted):
- The ICM-4268x is the noise IMU in 28 of 111 fit logs, and the control IMU in 10 of the 71 logs where the control IMU is known.
- Synthetic drones carry the chip of their generator source: 16 of 50 sources, so **32 % of synthetic drones**.
- Among the 25 identifications, 4 have an ICM-4268x FIFO IMU and 1 has an ICM-4268x control IMU.

**What changes the outcome.** Synthetic pass rate at PM 30°, M_s 2.0, M_s,wc 3.0, ±20 %, +1 sample:

| Variant | Pass rate |
|---|---|
| No chip delay | 48 % |
| ICM-42688-P, as a nominal shift (either rule) | 41 % |
| ICM-42688-P, inside the delay step | 45 % (BIC) / 44 % (fit-error) |
| ICM-42686-P | 32 % (shift) / 35 % (in step) |

**The rule moves the result by at most 1 point; the treatment (shift or in the step) by 3-4 points; the chip itself by 7-16 points.**

**Identified T_d hints at the chip delay**, but only weakly: the median is 2.80 ms with an ICM-4268x FIFO IMU (n 4) against 1.35 ms with other chips (n 21). That is consistent with about 1 ms of extra delay, but n = 4 and T_d is weakly fixed, so it is suggestive only.

**Never count it twice:**
- An identified plant already contains the FIFO chip's filter, so real-gain drones get no increment.
- Synthetic drones get the increment of their source chip.

## 4. Who passes which limits (tables E and F)

### 4.1 Real flown settings (25 identifications, the pilot's own filters)

**The PM floor never matters for these drones.** The rows for 25, 30 and 35° are identical in every table, because M_s binds first.

**Nominal limits only** (no uncertainty set): **68 %** pass at M_s,max 2.0 (56 % at 1.8, 72 % at 2.2).

**The 8 nominal failures** are "reference outside limits" under D-15, whatever A4 is. Their nominal PM / M_s:
- a3f3ae23 roll and pitch: 26° / 2.60 and 29° / 2.69 (only 10.5 s of data, 250 Hz loop);
- ab1aecd2 pitch: 34° / 2.59 (FFT notch on, not modelled);
- c901f9c1 roll: 17.6° / 5.64. Suspicious: its roll b is 1.82 times its pitch b. Pitch: 48.9° / 2.23;
- d32cc24c roll and pitch: 39.9° / 3.20 and 55.1° / 2.27 (FFT notch on, not modelled; ESC notch on without valid RPM);
- 225bab55 pitch: 40.4° / 2.03.

**Your question: do the candidate limits reject most real settings?** Yes. Overall pass rate and, in brackets, the share of the nominal passes that survive:

| Limit variant | ±10 % +0.5 | ±10 % +1 | ±15 % +1 | ±20 % +1 | ±30 % +1 | ±20 % +2 |
|---|---|---|---|---|---|---|
| Same limits for every plant (THESIS candidate) | 48 % (71 %) | 40 % (59 %) | 36 % (53 %) | 36 % (53 %) | 16 % (24 %) | 20 % (29 %) |
| Separate worst-case limit, M_s,wc 3.0 | 68 % (100 %) | 68 % (100 %) | 64 % (94 %) | 56 % (82 %) | 44 % (65 %) | 48 % (71 %) |
| Separate worst-case limit, M_s,wc 2.5 | 68 % (100 %) | 60 % (88 %) | 56 % (82 %) | 44 % (65 %) | 40 % (59 %) | 36 % (53 %) |

(Overall rates: the pass-rate tables of `evidence_tables.md` E; shares in brackets: its conditional table.)

- With the same limits for every plant (the THESIS candidate) at ±20 % / +1 sample, only **36 %** pass, i.e. **53 % of the settings that pass nominally**.
- **Narrowing the set alone barely helps:** ±15 % / +1 sample gives 36 % (53 %); ±10 % / +0.5 sample gives 48 % (71 %).
- **Separating the worst-case limit helps much more.** With M_s,wc 3.0 (PM ≥ 19.2°, GM ≥ 3.5 dB, every corner stable), ±20 % / +1 sample keeps **82 %** of the nominal passes and ±15 % / +1 sample keeps **94 %**.

**b27d09e3, the pilot's roll setting:**
- With the feasibility plant it reproduces exactly: nominal PM 39.5°, M_s 2.10; worst case at ±20 % / +1 sample PM 23.4°, M_s 3.87.
- With the automatic plant: nominal 42.6° / 1.88; worst case 28.6° / 3.03.

### 4.2 Synthetic Tier 3a drones at the PX4 default filters (1,000)

**Nominal limits only:** 50 % pass (PM 30°, M_s 2.0); 59 % at PM 25°, M_s 2.2.

| Limit variant | ±15 % +1 | ±20 % +1 |
|---|---|---|
| Same limits for every plant | 24 % (49 % of the nominal passes) | 20 % (41 %) |
| Separate worst-case limit, M_s,wc 3.0 | 50 % (99 %) | 48 % (97 %) |

**By crossover** (nominal, PM 30°, M_s 2.0): **90 % at 4-6.5 Hz, 23 % at 6.5-10 Hz, 0 % at 10-15 Hz.** See finding F3.

## 5. New findings (landmines; they need the author before Tasks 3-4)

**F1. The D-28 plant priors are slower than many real plants** (table C).
- The identified τ_m spans 11.7-64.9 ms (median 21.2); 13 of 25 are inside 20-60 ms (52 %).
- In the robust comparison, the motor-plus-delay lag at 10 Hz, **9 of 25 real plants lag less than the fastest plant D-28 can produce** (53.3°).
- A prior of τ_m 10-60 ms with T_d 0.1-5 ms would cover them.

**F2. The Tier 3a gain design is unlike the real drones** (table H; THESIS 7.1.1 Tier 3a text, not a D-decision).
- The real drones' PM without filters is 48.6-106.6° (median 78°); **only 48 % lie inside the 45-75° window**.
- Their integral corner is 0.034 of the crossover (median); the design uses 0.1.
- The D corner matches (median r 1.18; 92 % inside 0.5-3).
- A design matched to the real drones (window 55-100°, i 0.03) raises the share of synthetic drones that are safe at their defaults from **50 % to 71 %**.

**F3. With the PX4 defaults as reference, no synthetic drone at f_c ≥ 10 Hz survives.**
- This holds under any candidate limits and in all four designs tested, including F1's faster priors and F2's matched design (0 % each).
- Real drones above 10 Hz fly with raised cutoffs: b27d09e3 uses 80/40 Hz. With the 40/30 Hz defaults its roll PM would be 27°, as the feasibility test found.
- So Tier 3 step 3 (reject drones that are unsafe at their own defaults) together with D-09 (synthetic defaults = PX4 v1.16) **in effect limits the synthetic set to f_c up to about 10 Hz**.
- Options:
  1. Keep 4-15 Hz and state that the kept drones end near 10 Hz.
  2. Sample f_c in 4-10 Hz.
  3. Give synthetic drones a pilot-like reference whose cutoffs grow with f_c. This changes D-09.
  4. Keep unsafe-default drones, flagged "reference unsafe", as D-15 does for real drones.

**F4. Real crossovers below 4 Hz exist.**
- The flown settings cross over at 3.3-15.9 Hz (median 6.4); 3 of 25 lie below 4 Hz (3.3-3.6 Hz).
- For them, f_lo = 5 f_c is below the validated 20 Hz band edge. Task 3 needs a rule for real-gain drones, e.g. f_lo = max(5 f_c, 20 Hz).

**F5. Single identifications can be wrong.**
- c901f9c1 roll: its b is 1.82 times the pitch value, and it gives a nominal PM of 17.6° for a drone that flies.
- a3f3ae23: only 10.5 s of data, halves differ by about 28 %.
- Two logs use the FFT notch, which is not modelled.
- Model-predicted margins of real drones are therefore uncertain beyond the uncertainty set. This is the motivation for E10/E11.

## 6. Recommendations (the author decides; data-backed)

| Decision | Recommendation | Main evidence |
|---|---|---|
| **A2** PM floor | **30°** | It never changes a real-drone result: M_s binds first in every table. Åström and Murray: "a phase margin of 30 to 60 is typically required for robustness to modeling errors and non-oscillatory response" (2006 draft, Sec. 11.2). M_s ≤ 2 already implies PM ≥ 29°. |
| **A3** M_s,max (nominal) | **2.0** | The textbook pairing with GM ≥ 6 dB (M_s 2 implies GM ≥ 6 dB, PM ≥ 29°). 17 of 25 real flown settings meet it nominally. 2.2 would add one (225bab55 pitch, 2.03) and the hand-identified b27d09e3 roll (2.10); it is the alternative if those should pass. |
| **A4** uncertainty set | **±15 % on b and τ_m, delay step 1 controller sample** | About twice the 90th percentile of the measured within-flight changes at 10 Hz (2.23 dB / 8.3° vs 1.0 dB / 5.1°). That leaves room for flight-to-flight changes, which the halves cannot see. ±20 % is wider than the data asks for; ±10 % / +0.5 sample has no room. |
| **A4** worst-case limits | **Separate: every plant of the set Nyquist-stable with M_s ≤ 3.0** (hence PM ≥ 19.2°, GM ≥ 3.5 dB); the nominal plant keeps PM ≥ 30°, M_s ≤ 2.0, GM ≥ 6 dB | Keeps 94 % of the nominally passing real flown settings and 99 % of the synthetic defaults. With the same limits for every plant only 53 % and 49 % survive, so the safety check would mostly return "reference unsafe". |
| **A4** ICM chip delay | **Fit-error rule.** Synthetic ICM-4268x drones: nominal T_d + lower bound, delay step + (upper − lower): ICM-42688-P +0.91 ms, step +0.47 ms; ICM-42686-P +2.00 ms, step +1.02 ms. Identified real plants: no increment. | The BIC rule is falsely narrow. The lower bound is a physical minimum (minimum phase) for the measured magnitude. The choice of rule changes pass rates by at most 1 point. This refines D-25 ("inside the delay uncertainty"). |

Also needed before Task 4 (findings F1-F4):
1. Revisit D-28 τ_m / T_d (recommend τ_m 10-60 ms, T_d 0.1-5 ms).
2. Match the Tier 3a design to the real drones (recommend PM window 55-100°, integral corner 0.03).
3. Choose how to handle synthetic drones with f_c ≥ 10 Hz (F3, options 1-4).
4. A rule for f_lo when a real crossover is below 4 Hz (recommend f_lo = max(5 f_c, 20 Hz)).
