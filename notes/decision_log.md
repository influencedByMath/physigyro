# Decision log for the paper (Methods, Data and Limitations sections)

Written for: the author, as the source when writing the paper. **The full wording of every decision is in THESIS.md Section 2** (authoritative). This file adds what the paper needs:
- which **experiment or evidence** caused each decision;
- what **changed or was superseded** later, and why;
- the small **implementation rulings** made during the work;
- every **correction** of an earlier claim.

Keep it updated in every phase. Last update: 2026-10-18 (Phase 4, D-36).

Status values: **active** (in force), **extended** (still in force, with additions by a later decision), **partly superseded** (one part replaced; named), **done** (a one-off action).

---

## A. Decisions D-01 to D-36

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
| D-28 | 10-18 | Tier 1 priors (A1): f_c 4-15 Hz, τ_m 20-60 ms, T_d 0.5-5 ms (IO chip only as sensitivity); control and FIFO rate as empirical pairs with the **measured** controller rate | `notes/tier1_priors.md`: Xu 2019, Bangura 2017, PX4 docs and source, Tier 2 b27d09e3; f_c ≥ 4 Hz from the 20 Hz band edge; controller rate ≠ `IMU_GYRO_RATEMAX` in 42 of 102 fit logs | active; **f_c, τ_m and T_d ranges superseded by D-36** | Methods 7.1.1; Data |
| D-29 | 10-18 | Fig. 1 with both ESC-bank variants: (a, main) PX4 default, bank off; (b) bank on (capability); caption explains the notch-centre spikes and the 1.8 ms the bank adds in the crossover band | Author's choice after the side-by-side figure; bank delay computed and checked analytically (1.81 ms) | active; crossover band redrawn for D-36 (3-10 Hz) | Results (E1), Fig. 1 |
| D-30 | 10-18 | A2: PM floor 30° (nominal plant) | Evidence pack: never binds for real drones (Ms binds first) | active | Methods 7.8 |
| D-31 | 10-18 | A3: Ms_max 2.0 (6 dB) nominal, GM ≥ 6 dB | 23 of 28 real flown settings; Skogestad and Postlethwaite 2005 Sec. 2.4.3 | active | Methods 7.8 |
| D-32 | 10-18 | Separate worst-case limits (concept; value open): every plant of the set stable with Ms ≤ Ms_wc and its implied PM/GM | Same limits keep 39 % of nominally passing real settings, separate (Ms_wc 3) 78 % | active; **value decided in D-36** (Ms_wc 3.0) | Methods 7.8 |
| D-33 | 10-18 | ICM chip delay by the 1.5 % fit-error rule: lower bound as nominal Td shift, rest in the delay step (synthetic); real identified plants FIFO → control correction only | BIC rule falsely narrow; rule changes pass rates ≤ 2 points | active; **refines D-25 (2)**; scope (rule and treatment) confirmed in D-36 | Methods 7.1.1, 7.8 |
| D-34 | 10-18 | Author accepted all implementation rulings made without asking (Section C, ledger); provisional identification gates frozen with A4 | Author | active; gates frozen in D-36 | - |
| D-35 | 10-18 | The 31 fit logs with logger dropouts are not identified (cut list) | Author; new path with its own risks | active | Limitations |
| D-36 | 10-18 | **A4 and F1-F4 final:** (1) ±20 % on b and τ_m, delay step 1 controller sample, Ms_wc 3.0 (nominal limits of D-30/D-31 unchanged); chip delay by D-33, rule and treatment; D-14 identification settings kept, gates frozen; (2) flagged real drones kept in the test set and reported; (3) τ_m 15-60 ms, T_d 0.3-8 ms (net of the chip filter), identified plants only as a sensitivity check; (4) PM window 60-95°, integral corner 0.03; (5) f_c 3-10 Hz, real drones above 10 Hz reported separately, f_lo = max(5 f_c, 20 Hz); (6) Task 3 real-gain cases need an identification and a steady window | `notes/followup_A4_F1_F4.md`: the 5 worst-case failures are nominally marginal (Ms 1.81-1.94) and fail by the delay step, not extrapolation; band extension loses 1-8 other identifications; Ms_wc 3 midway between "about 2" and "about 4" (convention; Skogestad and Postlethwaite 2005): keeps 78 % of real and 97 % of synthetic nominal passes; the prior covers 19 of 28 identified plants (D-28: 11); the design keeps 76 % instead of 50 % at the defaults; no synthetic drone with f_c ≥ 10 Hz safe at the defaults in 13 populations; D-23 Q5 band; fit half: 26 of 28 identified cases qualify for Task 3 | active | Methods 7.1.1, 7.2, 7.8; Results (E3); Limitations |

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

**Phase 4 (how the evidence changed the THESIS candidates; all decided by the author):**

9. **Plant priors: literature only (D-28) → widened to cover the identified real plants (D-36 (3)).** D-28 was slower than 9 of 30 identified fit-half plants; τ_m 15-60 ms and T_d 0.3-8 ms cover 19 of 28 (D-28: 11).
10. **Tier 3a gain design: PM window 45-75°, integral corner 0.1 (THESIS) → 60-95°, 0.03 (D-36 (4)),** the p10-p90 window and the median corner of the identified real drones; drones kept at the PX4 defaults 50 % → 76 %.
11. **Crossover prior 4-15 Hz (D-28) → 3-10 Hz (D-36 (5)).** No synthetic drone with f_c ≥ 10 Hz was safe at the PX4 defaults in any design (F3); the slowest real drones cross over at 3.0-3.6 Hz (F4).
12. **Noise band f_lo = 5 f_c (THESIS 7.2) → max(5 f_c, 20 Hz) (D-36 (5)),** so the noise metric never leaves the frozen band of D-23 Q5.
13. **Robust limits: the same limits for every plant (THESIS 7.8) → separate worst-case limits (D-32), Ms_wc 3.0 (D-36 (1)).** With the same limits, ±20 % / +1 sample kept only 39 % of the nominally passing real flown settings; with Ms_wc 3, 78 %.
14. **Tier 2 acceptance: held-out VAF (THESIS) → frequency-response gates (D-14, D-34), frozen in D-36,** because least squares on closed-loop data gave negative VAF.

---

## C. Implementation rulings (made during the work, reported to the author)

All rulings in this table up to the evidence pack were accepted by the author on 2026-10-18 (D-34). The rows marked (follow-up) were made afterwards and are reported in `notes/followup_A4_F1_F4.md` Section 10.

| Ruling | Why | Where |
|---|---|---|
| Windows are gated first, then the quietest non-overlapping ones selected (≤ 3) | Selecting first let a gappy window block a good one | `spectra_library.select_windows` |
| The Q7 axis check uses all airborne gap-free time, and is "inconclusive" below 0.03 rad/s roll/pitch motion | The longest-segment version gave 2 false rejections | `noise_data.axis_correlation` |
| Logs missing a Q10 topic get a reason code (`Q10_no_rate_setpoint`, `Q10_no_land_detector`) | Crashes would hide them | `spectra_library.process_log` |
| Free-peak cap of 15 | A cap of 6 bound in 42 of 52 fits and truncated the generator's peak counts | `fit_model.py` |
| Generator jitter measured within vehicles: frequency 0.65 % (4 drones), height 1.1 dB (49 drones) | Data-driven, not guessed | `generator_check.jitter` |
| The extended screen is "the exact first N of the D-21 order" (resume-safe) | `--max-logs` would overshoot on restart | `log_screen.run_screen(order_limit)` |
| The Q7 rule was not changed after seeing the test half (3 vehicles lost) | That would be tuning on test data | REPORT 15 |
| (Phase 4) The empirical (FIFO rate, controller rate) pairs of D-28 come only from logs whose logged FIFO IMU is the control IMU (38 of 111 fit logs) and whose controller rate is logged at full rate | Otherwise the pair mixes two IMUs (e.g. 8 kHz filters with a 667 Hz controller from a 2 kHz IMU), which PX4 never runs. **Author agreed 2026-10-18.** 37 such pairs exist (`priors_from_logs.json` `rate_rows`) | `notes/tier1_priors.md` 1.4; Task 4 |
| (Phase 4) Hover f_m of synthetic drones = source vehicle's RPM/60 times the spectrum's frequency scale, not an independent draw | The generator resamples whole real peak sets (D-26); an independent f_m would put the ESC notch away from the noise peaks | `notes/tier1_priors.md` 1.6; Task 4 |
| (Phase 4) Chip-filter delay = group delay of the minimum-phase (analog Butterworth) filter with the fitted magnitude, at 20-100 Hz | The magnitude is all that is identified (D-25); a causal filter with that magnitude cannot lag less, so it is a lower bound | `src/priors.py` |
| (Phase 4, evidence pack) Tier 2 identification without hand-picked windows: Welch spectra over every gap-free airborne piece of one instrument class; up to 3 skipped controller publications held (the motor command holds its value), longer gaps split | Real logs have logger dropouts and single skipped publications (about 10 % of intervals in some logs); a single gap-free window is rarely long enough | `src/sysid.py`, `experiments/phase4/plant_variation.py` |
| (evidence pack) Raw FIFO gyro zero-phase low-passed at 100 Hz before resampling to the controller grid | Vibration near the controller rate would fold into the 0.5-20 Hz fit band; zero phase adds no delay (no T_d bias) | `sysid.body_rate_on_grid` |
| (review fix) Skipped controller publications held up to 8 samples (was 3); Welch segment stays 1.28 s | Fit-half probe: the torque setpoint AND vehicle_angular_velocity pause 5-6 periods about every 1.02 s (controller paused, command held); a 0.8 s segment was tried and rejected because it weakens the D-14 6-frequency rule (25 -> 19 fits) | `plant_variation.MAX_HOLD` |
| (review fix) Identified Td corrected to hold the control chip's filter once: Td - d(FIFO chip) + d(control chip), lower bounds of Task 1; rows with loop parameters changed in flight excluded from pass rates | The identified Td contains the FIFO chip's filter (raw FIFO gyro as output); initial gains are wrong after an in-flight change | `robust_limits.case_from_log` |
| (evidence pack) Provisional acceptance gates for identifications: fit error <= 3 dB / 15 deg, tau_m and Td not on a fit bound, D-14 instrument agreement in Position/auto modes | Evidence settings, not decisions; gate sensitivity 24 / 25 / 25 rows at 2 / 3 / 4 dB | `plant_variation.accept`; evidence pack 1.1 |
| (evidence pack) Repeatability = first vs second half of the usable time of one flight, also compared as the plant gain and phase at 10 Hz | tau_m and Td trade off below the 20 Hz band edge; the margins depend on the response near the crossover | `evidence_tables.half_diff` |
| (evidence pack) Separate worst-case limits are expressed by Ms_wc with the PM and GM it guarantees (PM >= 2 asin(1/(2 Ms)), GM >= Ms/(Ms-1)) | Distance from the Nyquist curve to -1 is 1/Ms (Astrom and Murray Sec. 12.2); one number keeps PM, GM and Ms consistent | `experiments/phase4/robust_limits.py` |
| (follow-up) The identification variants of the author's question 2 (band to 40 Hz, coherence 0.3, 2.56 s segments, combinations) were run on all 28 included cases, not only the 5 failing ones | Selecting the 5 would select on the outcome; 40 Hz covers every critical frequency (at most 25.5 Hz) | `experiments/phase4/a4_followup.py` |
| (follow-up) Filter feasibility of flagged references: 24 x 18 grid over the optimizer ranges, notches as flown or all off, stability limits only | Answers whether a safe label exists; noise belongs to Task 3 | `a4_followup.feasible_filters` |
| (follow-up) Identified Td compared with Td priors net of the chip filter (lower bound of the control chip, else of the FIFO chip) | D-33 adds the chip delay to synthetic drones separately; no double counting | `f_options.net_td` |
| (follow-up) F-option populations evaluated without the ICM chip delay (as in robust_limits); the option 'identified plants' at a 50 % share with net Td | Comparability between the options | `experiments/phase4/f_options.py` |
| (follow-up) The author's F4 options read as: 3 f_c for every drone; 10 Hz only where 5 f_c < 20 Hz | A fixed 10 Hz for every drone would put f_lo inside the control band of every drone above 2 Hz | `f_options.f_lo_rules` |
| (follow-up) A5 estimate from the fit half only; the test-half RPM count was not opened | The test half is only for evaluation | `notes/followup_A4_F1_F4.md` Section 8 |
| (D-36 round) A4 final and F1-F4 final recorded as one decision, D-36, with parts (1)-(6), as the author's "record D-29 to D-36" asked; composite rows have precedent (D-23, D-24) | Parts can be cited as "D-36 (3)" | THESIS Section 2 |
| (D-36 round) Fig. 1 redrawn for the decided crossover band (3-10 Hz) and noise band (max(5 f_c, 20 Hz), hatched 20-50 Hz); its script now prints with UTF-8 (it crashed on a cp1252 console after writing the files) | Fig. 1 marks the decided range; a stale band would contradict D-36 | `experiments/phase4/fig1_bode.py`, 2 new tests; `mutation_check_d36.py` 5/5 |
| (D-36 round) E11 validates over the same band as the noise metric: above max(5 f_c, 20 Hz) | E11 validates the metric; below 20 Hz the gyro also holds commanded motion | THESIS 6, 7.4.5, 8 |
| (D-36 round) Stale TBDs updated where a decision exists: THESIS 7.2 (Ms_max, f_max), 7.3 (PM floor), Section 6 (robust stability), 7.1.1 Tier 2 step 6 (frozen gates), Tier 3 step 1 (no sampling from Tier 2 distributions) | They contradicted D-23, D-30, D-31, D-34, D-36 | THESIS |
| (D-36 round) The open question whether the optimizer's limits S include the worst-case limits is left to the author with the Task 3 plan | It decides what "optimal" means in RQ1 and costs run time | `notes/task3_plan.md` |

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
| (Phase 4) Controller rate off `IMU_GYRO_RATEMAX` in "39 of 99" fit logs (shown to the author at the A1 STOP) | **42 of 102**. "Logged at full rate" was first judged by measured ≥ 0.5 × setting, which dropped 3 full-rate logs (settings 2000 → 999 Hz, 1000 → 499 Hz, 0 → 795 Hz); now by the logger profile (PX4 `logged_topics.cpp`). D-28 is unchanged | self-check before the Task 2 report |
| (Phase 4) Evidence pack v1 (committed 250aef6): "±15 % is about twice the measured repeatability"; "F3: 0 % in every design"; real-gain margins "with the control IMU's filter rate"; "never count twice" without the FIFO/control caveat | Compared only at 10 Hz; at each drone's own M_s peak ±15 % covers 16 and ±20 % 17 of 18 pairs, so the recommendation is ±20 %. F3 is rare-not-impossible (6.5 % with faster priors). vehicle_imu_status was never loaded (all cases used the FIFO rate; now matched in 23 of 30). The caveat and a Td correction were added | independent review, 2026-10-18 |
| (Phase 4, `notes/tier1_priors.md` 1.6) Scenario B synthetic drones possible with the 6 fit vehicles that have valid RPM | Only 5 can be generator sources: 51603946 has valid RPM in 1 of its 3 windows, and the vehicle spectrum keeps motor frequencies only if every window has them (`spectra_library.vehicle_spectra`); 5 of the 50 generator entries carry motor harmonics | A5 count in the follow-up, 2026-10-18 |
| (follow-up, reviewer's point 4) "At 2 kHz the Fig. 1 curves change by at most 0.16 ms (D-term path)" | That holds for panel (a); in panel (b) the ESC bank adds about 0.02 ms, so the D-term path changes by up to 0.19 ms (crossover band). 6,400 Hz stays | recomputed while writing the THESIS limitation, 2026-10-18 |

## E. Open for Phase 4 (to be decided; see `notes/phase4_plan.md`)

- ~~A1: Tier 1 prior ranges.~~ Decided as D-28 (2026-10-18).
- A4 input (found in Task 1): the plan's "BIC within 10" rule gives a falsely narrow ICM chip-delay range (BIC gains inflated, LOW 5); the 1.5 % fit-error rule gives ICM-42688-P 0.91-1.37 ms. ICM-42686-P is not bounded below (optimum moved from the 150 Hz grid edge to 116 Hz; 90 Hz within BIC 10): delay ≥ 2.0 ms, no upper limit from the data.
- **Evidence pack for A2-A4 (2026-10-18; revised after the independent review): `notes/evidence_pack_A2_A4.md`** (tables `experiments/phase4/evidence_tables.md`). Recommendations: PM floor 30 deg; Ms_max 2.0; ±20 % on b and tau_m with a delay step of 1 sample (changed from ±15 % after the frequency-resolved comparison); separate worst-case limit Ms <= 3.0; chip delay by the fit-error rule (near-crossover lower bound as nominal shift, rest in the delay step) for every chip with a fitted term; identified real plants get only the FIFO -> control chip correction.
- New findings for the author (evidence pack Section 5): F1 the D-28 plant priors are slower than 9 of 30 identified plants; F2 the Tier 3a gain design (PM window 45-75 deg, integral corner 0.1) is unlike the real drones (median unfiltered PM 72.8 deg, corner 0.030); F3 (restated) synthetic drones with f_c >= 10 Hz are rare, not impossible, at the PX4 defaults (0 of 117-196 in designs a-c, 6.5 % with faster priors); F4 4 of 30 real crossovers below 4 Hz make f_lo < 20 Hz; F5 single identifications can be wrong.
- ~~Fig. 1 variant.~~ Decided as D-29 (both panels: default off as the main panel, bank on as the second).
- ~~A2: PM floor.~~ Decided as D-30 (30°).
- ~~A3: M_s,max.~~ Decided as D-31 (2.0).
- ~~A4.~~ Decided as D-32, D-33 and D-36 (±20 %, 1 controller sample, Ms_wc 3.0; D-14 identification settings kept; flagged drones kept and reported).
- A5: training-set size and Scenario B share (preliminary estimate in `notes/followup_A4_F1_F4.md` Section 8: 5 generator RPM sources, natural share 10 %; about 5 test vehicles with RPM). **Author 2026-10-18: deferred until F1-F4 are recorded (now D-36); the author decides the final size next.** Input: the optimizer run time (`notes/task3_plan.md`, risk R4).
- ~~F1-F4.~~ Decided as D-36 (3)-(5) (the recommendations, with f_c 3-10 Hz).
- **Task 3 plan: `notes/task3_plan.md`, waits for the author's approval** (questions T3-1 to T3-6: optimizer limits, which half, sample size, optimizer grid, the 10 Hz definition, synthetic drones that fail only the worst case at the defaults).
- Phase 5: fallback for flagged real drones (the repair line of THESIS 7.8 ends at an unsafe reference; a known-stable setting exists for all 10 flagged fit-half settings).
- VERIFY: the ICM-42688-P on-chip filter path (datasheet vs PX4 driver; 585 Hz AAF cannot explain the measured roll-off).

## F. Limitations to state in the paper

1. **+4.4 dB generator bias at 20-100 Hz** for non-ICM chips (ICM +1.2 dB).
2. **About 3 dB model misfit at the motor band** (95-130 Hz).
3. **The frequency jitter rests on only 4 drones**; peak sets are near-copies of real ones.
4. **Small chip groups:** IIM-42652 and IIM-42653 have 2 drones each, ICM-42686-P 3. The BIC treats Welch bins as independent, which inflates the gains.
5. **95 usable vehicles**, 5 short of the target of 100. Dropout loss dominates.
6. **The chip delay is not identified from data** (D-25).

**Phase 4** (full wording in THESIS Section 11, item 10; the reviewer's 10 declined points stand as limitations, D-34, author 2026-10-18):

7. **Shared feedback bias:** in Position/auto modes two agreeing instruments cannot reveal a bias they share (D-14).
8. **Interior worst case, no μ-analysis:** only the nominal plant and the 8 corners are checked; with notches the worst plant can lie inside the box.
9. **Ms_wc 3.0 is a convention;** the flagged share depends on it (22 % / 4 % / 0 % of the real nominal passes at 3.0 / 3.5 / 4.0).
10. **Fig. 1 at 6,400 Hz** although 2 kHz is the most common FIFO rate (up to 0.16 ms in panel a, 0.19 ms in panel b at crossover).
11. **Mutation checks rewrite source files in place** (restored byte-identically).
12. **The real set mixes in one FPV airframe (d32cc24c) and v1.17 logs** evaluated with the v1.16 filter model.
13. **The "42 of 102" rate statistic includes one log with `IMU_GYRO_RATEMAX` = 0.**
14. **Tier 2 is judged by frequency-response gates only** (no held-out time-domain check).
15. **`body_rate_on_grid` re-filters every piece** (run time only).
16. **Held skipped publications** (up to 8) assume that the motor command holds.
17. **Scenario B rests on 5 generator sources** (about 10 % of training) and about 5 test vehicles.
18. **Synthetic crossovers cover 3-10 Hz only;** real drones above 10 Hz are reported separately (2 of 26 fit-half real-gain cases by the unfiltered crossover).
