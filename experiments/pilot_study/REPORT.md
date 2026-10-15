# Pilot study: proposed data-quality thresholds (Phase 3, Part 1)

Day 18, 2026-10-15. Plan: `notes/phase3_plan.md`, Task 9. Spec: THESIS.md 7.4.3 (gates Q1-Q16, pilot steps 1-5), decision D-10.

**Frozen 2026-10-15 as decision D-23** (THESIS Section 2): the author approved all values of Section 2 and decided to exclude the 33 vehicles without a board ID (Section 4.1, option a).

Every number of Section 2 was re-derived independently by `verify_numbers.py` before the freeze (2026-10-15: all claims verified after three corrections, marked below). Numbers come from `results.json` (`run_pilot.py`), `usable.json` (`usable.py`), `axes.json` (`check_axes.py`) and `data/screen/` (`log_screen.py`, `split.py`). Figures: `figures/` (`make_figures.py`).

---

## 1. Data so far

| Step | Result |
|---|---|
| Index snapshot | 2026-10-14, 477,079 public logs; 182,209 pass the index filter (step A); 26,163 vehicle groups |
| Header screen (step B, D-21) | 3,177 logs read (0.4 MB each), 1 network error (retried after a restart), **200 FIFO vehicles**. FIFO share per vehicle **6.3 %** (95 % CI 5.5-7.2 %); the audit's 7.7 % was per log |
| FIFO logs | 2024: 34, 2025: 70, 2026: 96. Firmware v1.11 to v1.17. 180 quadrotors, 12 hexa, 8 octo. 15.3 GB in total (median 52 MB) |
| Vehicle split (step C, D-17, D-19, D-20) | 158 vehicles with a board ID: **84 fit, 74 test** (the 4 inspected vehicles are fit). Every assignment re-derived independently from the D-20 rule |
| Not split (open decision, Section 4.1) | 33 vehicles **without a board ID**: 21 ModalAI VOXL2, 11 Emlid Navio2, 1 with a redacted ID |
| Pilot set | 29 fit-half logs: the 4 inspected vehicles + 25 fit vehicles by hash; 2.3 GB, every file checked (size and ULog header) |

The pilot ran on fit-half vehicles only; `run_pilot.py` stops if a log belongs to a test-half vehicle (checked).

---

## 2. Proposed thresholds

| Gate | Threshold | THESIS candidate | **Proposed** | Evidence |
|---|---|---|---|---|
| Q5 | $f_{max}$ (upper edge of the fixed band) | 800 Hz | **780 Hz**, on the measured IMU rate | 14 of 29 pilot logs are 2 kHz IMUs. On the 8 of them that pass $D_{max}$ the measured clock is 1,994.0-2,000.5 Hz (−0.30 % to +0.03 %); the measured rate agrees with a single long-segment fit within 0.006 % on every log up to 20 % dropout (`verify_numbers.py`, re-check 2026-10-15). The rule $0.8 f_s/2 \ge 800$ needs $f_s \ge$ 2,000 Hz exactly, so 800 Hz drops every 2 kHz IMU whose clock is even slightly slow: 7 of these 8 good logs. Over all 29 logs, 58.6 % pass at 800 Hz and 89.7 % at 780 or 790 Hz (Fig. p2). 780 Hz leaves margin for clocks up to 2.5 % slow (other rates: 6.4 kHz −0.19 to +0.62 %, 8 kHz −1.0 to +1.5 %, one 8,370 Hz nominal IMU −3.0 %, still far above the limit). Noise above 800 Hz changed the noise metric by less than 0.002 (feasibility check L3). Motor 3rd harmonics in RPM logs reach at most 428 Hz |
| Q5 | $f_{min}$ | 20 Hz | **20 Hz** | Below about 20 Hz the raw spectrum is flight motion (Fig. p4, left). Store PSDs from 0 Hz anyway: the noise metric starts at $5 f_c$, which can be 15 Hz |
| Q4 | $L_{min}$ (shortest usable gap-free piece) | ≥ window length (10 s) | **one Welch segment** ($n_{perseg}/f_s$ ≈ 0.67 s); the window gate Q13 decides | A window may contain gaps: Welch averages only gap-free pieces, so the estimate stays unbiased and only loses averages. With "≥ 10 s" two logs with 4.1 % and 4.4 % dropout and 7 and 6 good windows were rejected (longest gap-free piece 8.4 s and 6.5 s) |
| Q4 | $D_{max}$ (dropout fraction of a log) | 5 % | **20 %** (coarse log filter only) | Real gaps are mostly tiny and frequent (median 0.5 ms, up to 60,000 gaps per log). So the fraction alone does not predict usability: in the artificial-dropout test even 1 % spread like the real gaps leaves half of the windows below 20 averages (Fig. p1), while 1 % as a few gaps does no harm. In the pilot, logs up to 13.8 % dropout still had 5-8 good windows; from 23 % on none had any. The real gate is Q13 per window; 20 % only skips hopeless logs |
| Q10 | Window length $W$ | 10 s | **10 s** | 10 s windows differ from the 20 s window around them by 0.49 dB median (90th percentile 1.7 dB); 5 s windows by 0.74 / 2.8 dB, and give fewer than 20 averages at 1.5 Hz resolution |
| Q13 | Resolution / minimum averages | 1.5 Hz / 20 | **1.5 Hz / 20** | $n_{perseg} = f_s / 1.5$; 10 s windows of logs with less than 1 % dropout give 24-29 averages at every rate |
| Q10 | Rate-setpoint activity | 15 °/s | **15 °/s** | Median airborne window 6.7 °/s, 75th percentile 15.9 °/s; 15 °/s keeps 74 % of 3,706 airborne windows (Fig. p3) |
| Q10 | Throttle steadiness (std of mean motor command) | none | **0.02** | Removes 9.5 % of the calm windows (activity ≤ 15 °/s) that have throttle data, 6.9 % of all calm windows (climbs, descents: the motor frequency moves inside the window); 0.05 removes 2.0 % (1.4 %). **Corrected 2026-10-15** (the first version said 5 % and 1 %). Cost: 1 pilot log loses all its windows (14 instead of 15 usable logs). 947 windows of old logs have no `actuator_motors`: flagged `Q10_no_throttle`, not rejected |
| Q10 | $T_{edge}$ | 5 s | **5 s** | Not tested separately; no evidence against it |
| Q6 | Jitter limit | 1 % | **1 %** | No log has any change of the FIFO sample interval (`dt`) |
| Q9 | Clipped samples per window | 0 | **0** | No clipped sample in any pilot log |
| Q14 | Windows per vehicle | 3 | **3** | Good logs give 1-8 non-overlapping good windows |
| D-13 | Logged RPM rate for the tracking notch | 50 Hz | **50 Hz** | Logged `esc_status` rates are 4, 4.2, 10 Hz or 200, 200, 399 Hz: 50 Hz separates the two groups cleanly |
| 7.4.4 | Peak prominence | 0.5 decades | not decided now | It selects notch positions (Scenario A), not data; set in Part 2 on the fit half |

**Effect on the data** (`usable.json`, pilot logs with at least one window that passes every gate):

| Threshold set | Usable logs | Windows (≤ 3 per vehicle) |
|---|---|---|
| THESIS candidates as written | 6 of 29 | 16 |
| + $f_{max}$ = 780 Hz | 11 of 29 | 28 |
| + $L_{min}$ = one Welch segment | 13 of 29 | 34 |
| + $D_{max}$ = 20 % | 15 of 29 | 40 |
| + throttle 0.02 (**proposed set**) | **14 of 29** | **38** |

The biggest loss is dropouts (13 of 29 logs), not the thresholds. Dropouts depend on the board and its SD card, not on the logging profile: profile 2047 gives 0.2 % on one board and 41 % on another.

---

## 3. Other results of the pilot

| Check | Result | Meaning |
|---|---|---|
| Board rotation (gate Q7), `check_axes.py` | 17 of 17 checkable logs correct (3x3 correlation with the logged body rate, diagonal 0.87-1.00 and largest in every row), rotations 0, 2, 4 | **Rotation 2 (yaw 90°) is now verified on data.** Rotations 6, 8, 12, 14 occur only in logs with heavy dropouts: still unverified |
| FIFO gyro = controller gyro (Q8) | 35 % | 65 % of logs are flagged; the feasibility test showed the neighbouring IMU still predicts the filtered gyro within 1 dB |
| Defaults known (Q11) | 97 % | 1 log (v1.11) has no default record: flagged |
| ESC RPM present (Q16) | 21 % (6 of 29) | Scenario B on real data will be small (risk R13) |
| RPM unit check | strongest peak at 0.84-1.17 x RPM/60 (and x 2 RPM/60) in 18 windows | No factor-2 or pole-pair error; the spread comes from 4 motors at different speeds and slow RPM logging |
| FFT dynamic notch on | 10 % | Flagged; raw spectra still usable, actual settings not modelled |
| Sample rates | 2 kHz (14 logs), 6.4 kHz, 8 kHz | Rates 1,251 / 1,259 / 1,823 Hz (and 2,015 Hz on log b4a9719c) appear only in logs with more than 50 % dropout: the measured rate is **not reliable** when most samples are missing. Those logs fail Q4 first |
| High-frequency roll-off | PSD 600-780 Hz vs 100-300 Hz: +6 to −52 dB; strongest on ICM-42688P boards at 8 kHz | Probably on-chip filters of the IMU. Consequences in Section 5 |

---

## 4. Decisions for the author

1. **Vehicles without a board ID (33 Linux boards).** Their gate-Q14 fallback key (board + airframe + filter/PID values) can give one drone two keys when a parameter changed between flights, so the same drone could land in both halves. For VOXL2 with airframe 4001 the 13 fallback keys would split 7 test / 6 fit, and nobody can tell whether they are 13 drones. Options: (a) **exclude them** (simplest, no leakage; recommended unless the go/no-go needs them); (b) one key per board type + airframe (no leakage, but distinct drones are merged and capped at 3 windows together); (c) keep the fine fallback key (accept the leakage risk, state it).
2. **The thresholds of Section 2** (approve, or change any value).
3. **Go/no-go projection.** With the proposed set, 14 of 29 pilot logs are usable (48 %, 95 % CI 31-66 %): roughly **76 of the 158** split vehicles (range 50-104). That is above the minimum of 30 but probably below the target of 100. To reach 100, risk R1 (a) applies: about 49 more FIFO vehicles, i.e. screening about 780 more vehicles (about 20 min of header reads at 6.3 % FIFO share) and about 3.8 GB of downloads. New vehicles get a stable split automatically (D-20).

---

## 5. Landmines found, and their status

**Fixed (with tests, on branch `phase3`):**

| # | Problem | Fix |
|---|---|---|
| 1 | Header-screen restart after the laptop shut down: logs with a network error would never be retried; a half-written last line would crash the restart | Error rows are retried; cut or unterminated lines are repaired (2 tests) |
| 2 | `noise_data.esc_rpm` crashed on logs that publish `esc_status` with no ESC report (`esc_count` 0); `fc_info` called such logs "with RPM" | Returns "no RPM" (gate Q16); `has_rpm_topic` needs a real report (2 tests) |
| 3 | Gate `Q4_no_long_segment` silently raised $L_{min}$ to the window length | Uses $L_{min}$ only (1 test) |
| 4 | Q5 rejected clean 2 kHz logs because of a 0.2 % clock error | Proposal: $f_{max}$ 780 Hz (Section 2) |
| 5 | Vehicles without board ID could leak across the split | Held out of `split.csv` until decision 4.1 |

**Open, for Part 2 (no action needed before your decision):**

- `IMU_GYRO_RATEMAX` is 0 (2 logs, v1.11) or 250 Hz (1 log) in the screen: the loop model takes the controller rate from this parameter. Rule needed: measure the rate from the logged `vehicle_angular_velocity`, or drop such logs.
- Accepted logs with an unverified board rotation (6, 8, 12, 14, ...) must pass `check_axes.py`, or be dropped.
- Duplicate uploads (gate Q3): some no-ID groups contain the same flight twice (same date and duration). Part 2 must keep one.
- On-chip IMU filters: the raw FIFO gyro is already low-passed inside some IMUs. Clusters may then follow the chip (gate Q12 cross-table), and the filter delay is in the real loop but not in the model (absorbed in $T_d$ for identified plants; to be stated as a limitation; the PX4 driver settings are a VERIFY item).
- The parametric noise model of THESIS 7.5 has a flat noise floor; the pilot spectra fall or rise by 2-4 decades across 20-800 Hz (Fig. p4). A floor shape must be chosen in Part 2, with evidence.
