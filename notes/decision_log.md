# Decision log for the paper (Methods, Data and Limitations sections)

Written for: the author, as the source when writing the paper. **The full wording of every decision is in THESIS.md Section 2** (authoritative). This file adds what the paper needs:
- which **experiment or evidence** caused each decision;
- what **changed or was superseded** later, and why;
- the small **implementation rulings** made during the work;
- every **correction** of an earlier claim.

Keep it updated in every phase. Last update: 2026-10-18 (Phase 4, D-28).

Status values: **active** (in force), **extended** (still in force, with additions by a later decision), **partly superseded** (one part replaced; named), **done** (a one-off action).

---

## A. Decisions D-01 to D-28

| ID | Date | Decision (short) | Evidence or reason | Status | Paper section |
|---|---|---|---|---|---|
| D-01 | 09-28 | No own flights: simulation and public data only | Author's choice | active | Methods (scope) |
| D-02 | 09-28 | Public datasets allowed; no own data collection or labelling | Author's choice | active | Data |
| D-03 | 09-30 | AI/ML must be visible; simple CPU methods (scikit-learn) | Professor's requirement; 40-day limit | active | Methods |
| D-04 | 09-30 | Core target: the PX4 filter stack (Betaflight optional) | Public high-rate noise exists only for PX4 | active | Introduction, Methods |
| D-05 | 09-30 | Excluded: RL, NN filter replacement, deep learning on raw gyro, LLM methods | Too slow or risky to verify in the time | active | Methods (scope) |
| D-06 | 09-30 | Deliverables: arXiv preprint, journal submission, public code | MSc applications | active | - |
| D-07 | 09-30 | Rejected topics (fault localisation, multi-model inference, multi-task backbone) | Novelty and feasibility checks | done | - |
| D-08 | 10-07 | One design goal: minimum noise, minimum delay, maximum stability; always reported together | Author's direction | active | Methods 7.0, all results |
| D-09 | 10-07 | "Default settings" are read from each log; synthetic drones use the PX4 v1.16 defaults | Audit 10-06: defaults differ by version and board | active | Methods (baselines) |
| D-10 | 10-07 | Thresholds are set in a pilot on fit-half logs only, then frozen | Leakage avoidance | done, by D-23 | Data (protocol) |
| D-11 | 10-07 | Three plant tiers: literature, system identification, synthetic | Realism | active (Tier 1 = Phase 4) | Methods 7.1.1 |
| D-12 | 10-07 | Safety check: Nyquist, PM/GM/Ms, uncertainty set, bisection repair | The old loop gave no guarantee | active (TBDs in Phase 4) | Methods 7.8 |
| D-13 | 10-14 | Tracking-notch spectrum for real logs with RPM; logged RPM rate gate (50 Hz, frozen in D-23) | Feasibility: mean-RPM notch up to 9 dB off; tracking within 0.9-1.8 dB | active | Methods (noise metric) |
| D-14 | 10-14 | Tier 2 system identification: closed-loop FRF with an external instrument, chosen by flight mode | Feasibility: least squares gave negative VAF; instruments agree within 0.2 dB | active (Tier 2 optional) | Methods 7.1.1 |
| D-15 | 10-14 | Real-gain drones: the reference is the pilot's actual setting | With the pilot's gains, the default filters were unsafe (PM 27°) | active | Methods (baselines) |
| D-16 | 10-14 | Static-notch bandwidth search over the full PX4 range, 5-100 Hz | The optimum sat on the old 50 Hz edge | active | Methods (search space) |
| D-17 | 10-14 | The 3 inspected vehicles are forced into the fit half | They shaped D-13 to D-16 (leakage) | active | Data (split) |
| D-18 | 10-14 | Timeline +1 week: Day 47 = 2026-11-13 | Schedule | active | - |
| D-19 | 10-14 | The audit vehicle 220b5990 is also forced into the fit half | Fully inspected in the audit (leakage) | active | Data (split) |
| D-20 | 10-14 | Split by vehicle: salted SHA-256 parity, fixed salt | Reproducible, order-independent, stable when more vehicles are screened | active (the extended screen relied on it) | Data (split) |
| D-21 | 10-14 | Header-screen sampling: random vehicle order, the newest qualifying log per vehicle | Logs per vehicle very uneven (median 2, max 1,276) | active; **extended by D-24 (5)** (fallback log) | Data (sampling) |
| D-22 | 10-14 | Raw `.ulg` logs kept on D: until Phase 9 | Slow downloads; re-processing after fixes | active (it made every rerun possible) | - |
| D-23 | 10-15 | **All data-quality thresholds frozen; no-ID vehicles excluded** | Pilot on 29 fit logs; every number re-derived by `verify_numbers.py` (3 report claims corrected first, see D below) | active | Data (gates); Table 1 |
| D-24 | 10-15 | Part 2 methods: (1) chip roll-off modelled as a low-pass, the chip is a hardware label; (2) one spectrum per drone; (3) spline floor; (4) one generator if clusters are unstable; (5) extended screen and next-newest-log fallback | Clusters followed the IMU chip; windows of a vehicle are near-copies (nearest neighbour 100 %); flat floor 6.8 dB misfit; projected total 73 < 100 | active, but **(P2) superseded by D-25 (1)**; the delay part of (1) is **qualified by D-25 (2)**; (3) **refined by D-26 (1)**; (4) **refined by D-26 (2), D-27** | Methods 7.5, Data |
| D-25 | 10-15 | (1) Q3 duplicate = same date + duration + **file size**; (2) chip low-pass for **magnitude only**; the delay goes into the T_d uncertainty | (1) `time_ref_utc` is the time-zone offset (PX4 `logger.cpp`) and wrongly rejected 5 flights; (2) a known synthetic filter was not recovered: corner and order are not identifiable | active | Data (Q3); Methods; Limitations |
| D-26 | 10-15 | Fixed robust floor; whole-peak-set generator; BIC gain ≥ 10 for a chip term; prominence 0.5; drop > 5 dB | A free floor is not identified (41 of 52 floors at the bound); independent Gaussian peaks gave spectra 32-59 dB too high; BMI088 term won by a BIC gain of only 1 | active | Methods 7.5 |
| D-27 | 10-16 | Generator floor mean per chip class, shared covariance | Pooled floor: +6.8 / −3.8 dB bias at 20-100 Hz; class mean: +4.4 / +1.2 dB | active | Methods 7.5; Limitations |
| D-28 | 10-18 | Tier 1 priors (A1): f_c 4-15 Hz, τ_m 20-60 ms, T_d 0.5-5 ms (IO chip only as sensitivity); control and FIFO rate as empirical pairs with the **measured** controller rate | `notes/tier1_priors.md`: Xu 2019, Bangura 2017, PX4 docs and source, Tier 2 b27d09e3; f_c ≥ 4 Hz from the 20 Hz band edge; controller rate ≠ `IMU_GYRO_RATEMAX` in 39 of 99 fit logs | active | Methods 7.1.1; Data |

---

## B. How the method changed because of experiment results (Phase 3 story for the paper)

1. **Flat noise floor (THESIS 7.5) → spline floor (D-24) → fixed robust spline floor (D-26).**
   - Evidence: real floors span 2.4 decades (median); a flat floor misfits by 6.8 dB, a power law by 5.1 dB, a spline by 2.3 dB.
   - A free spline is not identified against broad Lorentzian tails.
2. **Noise "archetypes" by k-means (THESIS 7.5) → no stable types → one generator with the chip as a covariate (D-24 P3 rule, D-26, D-27).**
   - The only stable split is the IMU chip (ARI 0.84 without correction).
   - With the chip roll-off removed: silhouette 0.26, ARI 0.57.
3. **Gaussian on fitted log-parameters (THESIS 7.5) → whole real peak sets on a Gaussian floor (D-26).**
   - Independent peak triples broke the structure inside a spectrum (harmonics, matched heights), and the heavy tails exploded in linear power.
4. **Chip low-pass "fitted, gives the delay" (D-24) → magnitude only, delay as T_d uncertainty (D-25).**
   - The profile test showed corner and order are not identifiable from spectra.
5. **Clustering unit: windows → one spectrum per drone (D-24 (2)).** Windows of one drone are near-copies.
6. **More data: extended screen (+1,150 vehicles) and fallback (D-24 (5)).**
   - The projection of 73 rose to an actual 95.
   - The fallback recovered 2 vehicles (fit), 0 (test).
7. **Q3 rule corrected (D-25).** A field misunderstanding (`time_ref_utc`), caught by inspecting suspicious output (5 "duplicates" of one flight).
8. **Thresholds (D-23) vs the THESIS candidates:**

   | Threshold | Candidate | Frozen | Evidence |
   |---|---|---|---|
   | f_max | 800 Hz | **780 Hz** | 800 drops 7 of 8 good 2 kHz logs (clocks 0.0-0.3 % slow) |
   | L_min | ≥ W | **one Welch segment** | Welch never crosses a gap |
   | D_max | 5 % | **20 %** | real gaps are tiny and frequent |
   | throttle limit | none | **0.02** | added by the pilot |

---

## C. Implementation rulings (made during the work, reported to the author)

| Ruling | Why | Where |
|---|---|---|
| Windows are gated first, then the quietest non-overlapping ones selected (≤ 3) | Selecting first let a gappy window block a good one | `spectra_library.select_windows` |
| The Q7 axis check uses all airborne gap-free time, and is "inconclusive" below 0.03 rad/s roll/pitch motion | The longest-segment version gave 2 false rejections | `noise_data.axis_correlation` |
| Logs missing a Q10 topic get a reason code (`Q10_no_rate_setpoint`, `Q10_no_land_detector`) | Crashes would hide them | `spectra_library.process_log` |
| Free-peak cap of 15 | A cap of 6 bound in 42 of 52 fits and truncated the generator's peak counts | `fit_model.py` |
| Generator jitter measured within vehicles: frequency 0.65 % (4 drones), height 1.1 dB (49 drones) | Data-driven, not guessed | `generator_check.jitter` |
| The extended screen is "the exact first N of the D-21 order" (resume-safe) | `--max-logs` would overshoot on restart | `log_screen.run_screen(order_limit)` |
| The Q7 rule was not changed after seeing the test half (3 vehicles lost) | That would be tuning on test data | REPORT 15 |
| (Phase 4) The empirical (FIFO rate, controller rate) pairs of D-28 come only from logs whose logged FIFO IMU is the control IMU (38 of 111 fit logs) and whose controller rate is logged at full rate | Otherwise the pair mixes two IMUs (e.g. 8 kHz filters with a 667 Hz controller from a 2 kHz IMU), which PX4 never runs | `notes/tier1_priors.md` 1.4; Task 4 |
| (Phase 4) Hover f_m of synthetic drones = source vehicle's RPM/60 times the spectrum's frequency scale, not an independent draw | The generator resamples whole real peak sets (D-26); an independent f_m would put the ESC notch away from the noise peaks | `notes/tier1_priors.md` 1.6; Task 4 |
| (Phase 4) Chip-filter delay = group delay of the minimum-phase (analog Butterworth) filter with the fitted magnitude, at 20-100 Hz | The magnitude is all that is identified (D-25); a causal filter with that magnitude cannot lag less, so it is a lower bound | `src/priors.py` |

## D. Corrections of earlier claims (state honestly; useful for Limitations)

| Claim | Correction | Found by |
|---|---|---|
| Pilot: throttle 0.02 removes 5 % of calm windows (0.05: 1 %) | 9.5 % (2.0 %) | `verify_numbers.py` before the freeze |
| 2 kHz IMU clocks −0.4 to +0.8 %; THESIS "2,015 instead of 2,000 Hz" | Those came from logs with 50-63 % dropout. Reliable logs: −0.30 to +0.03 % | `verify_numbers.py`, long-segment fit check |
| "The fitted chip low-pass also gives its delay" | Not identifiable; magnitude only (D-25) | profile test |
| Q3 via `time_ref_utc` | It is the time-zone offset (D-25) | output inspection, PX4 source |
| Generator: leave-one-vehicle-out "76 %, 7.7 dB" | 56 %, 8.36 vs 8.24 dB at equal sample sizes | independent code review |
| Generator: "not copies" | Peak sets are near-copies; only the floor is new | independent code review |
| Test count "197" in an early report draft | 196 at that time (200 after the review fixes) | test run |

## E. Open for Phase 4 (to be decided; see `notes/phase4_plan.md`)

- ~~A1: Tier 1 prior ranges.~~ Decided as D-28 (2026-10-18).
- A4 input (found in Task 1): the plan's "BIC within 10" rule gives a falsely narrow ICM chip-delay range (BIC gains inflated, LOW 5); the 1.5 % fit-error rule gives ICM-42688-P 0.91-1.37 ms. ICM-42686-P is not bounded below (optimum moved from the 150 Hz grid edge to 116 Hz; 90 Hz within BIC 10): delay ≥ 2.0 ms, no upper limit from the data.
- A2: PM floor.
- A3: M_s,max.
- A4: uncertainty set, including the ICM chip-delay range (D-25).
- A5: training-set size.
- VERIFY: the ICM-42688-P on-chip filter path (datasheet vs PX4 driver; 585 Hz AAF cannot explain the measured roll-off).

## F. Limitations to state in the paper (Phase 3)

1. **+4.4 dB generator bias at 20-100 Hz** for non-ICM chips (ICM +1.2 dB).
2. **About 3 dB model misfit at the motor band** (95-130 Hz).
3. **The frequency jitter rests on only 4 drones**; peak sets are near-copies of real ones.
4. **Small chip groups:** IIM-42652 and IIM-42653 have 2 drones each, ICM-42686-P 3. The BIC treats Welch bins as independent, which inflates the gains.
5. **95 usable vehicles**, 5 short of the target of 100. Dropout loss dominates.
6. **The chip delay is not identified from data** (D-25).
