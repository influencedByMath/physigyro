# Tier 1 plant priors: evidence table (Phase 4 Task 1)

Made 2026-10-18 (Day 21), branch `phase4`. Written for: the author, to decide A1 (and inputs to A4). **A1 decided 2026-10-18 as D-28** (the four proposals of Section 3 for f_c, τ_m, T_d and the rates; THESIS 7.1.1 updated). The FPV row is not decided. **Revised the same day by D-36** after the Tier 2 identifications (`notes/followup_A4_F1_F4.md` Section 7): f_c 3-10 Hz, τ_m 15-60 ms, T_d 0.3-8 ms (net of the chip filter); the rate pairs and hover f_m are unchanged.

**Rules followed**
- No number without a source that was read. What could not be found stays **VERIFY**, with the reason.
- Our own log evidence uses the **fit half only**. Every input goes through `src/priors.py` (`require_fit_half`, `require_fit_rows`); a test proves that `test_*` inputs are refused.
- Code: `src/priors.py` (34 tests, mutation check 17/17), `experiments/phase4/priors_from_logs.py` → `priors_from_logs.json`, `experiments/phase4/chip_lowpass_table.py` → `chip_lowpass_table.json`.

---

## 1. Evidence table

"Read" = the exact place in the source that was read in this session (2026-10-18).

### 1.1 Motor time constant τ_m

| Source | Exact place | Value as printed | Conversion | Result |
|---|---|---|---|---|
| Bangura & Mahony 2017, IEEE T-RO 33(2) [bangura2017thrust] (ANU open-access PDF) | p. 13, Sec. "C. Transient response", Fig. 10 | "transient response of both control architectures are in the order of 50ms rise time", limited by "current saturation"; 10-inch propeller, thrust step 2.5 → 3.5 N | The paper does not define "rise time". First-order: 10-90 % rise = 2.2 τ → τ ≈ 23 ms; if read as the 63 % time → τ ≈ 50 ms | **23-50 ms** (large rotor, closed-loop speed control) |
| Feasibility Tier 2 (our own, log b27d09e3, PX4 FMU-V6X, fit half) | `experiments/feasibility/REPORT.md` Sec. 5 | roll 29.2 ms, pitch 34.2 ms | none | **29-34 ms**, one vehicle |
| Blaha et al. 2024, arXiv 2406.11723 | Sec. II-C (3-inch FPV quad, Betaflight, ICM-42688-P, 2 kHz loop); Table I | bench "nominal" τ = 20.00 ms; identified in flight 25.91 / 26.21 / 26.64 / 26.61 ms (4 motors) | none | **20-27 ms** (FPV) |
| Blaha et al. 2024 | Sec. II-D (their simulation set-up) | random crafts with "time constant τ ∈ [15, 35] ms" | their modelling choice for small quads, not a measurement | 15-35 ms (FPV, design range) |
| Eschmann et al. 2024, arXiv 2404.07837 | Sec. IV-A | Crazyflie (27 g, brushed): T_m = 0.072 s; manufacturer step response ≈ 0.073 s; another study 0.065 s | none | 65-73 ms. **Brushed micro drone: outside both of our classes**; shown only to explain why "small" does not always mean "fast" |

Not found: a measured τ_m above about 50 ms for a brushless multirotor. The THESIS candidate upper bound of 100 ms (large drone) has **no source**.

### 1.2 Extra delay T_d (ESC protocol, output path; never the ZOH, which the loop model already has)

| Source | Exact place | Value as printed | Conversion | Result |
|---|---|---|---|---|
| PX4 docs v1.16, "MC Filter Tuning & Control Latency" [px4docs_filtertuning] | section on control latency | "The IO chip (MAIN pins) adds about 5.4 ms latency compared to using the AUX pins" | none | **+5.4 ms** if the motors are on an IO-chip MAIN output |
| PX4 docs v1.16, "OneShot ESCs and Servos" | page text | PWM pulse "1000us (zero) and 2000us (full power)"; "For PWM this rate is close to 500 Hz"; OneShot125 pulse "125us (zero power) to 250us (full power)" | PWM: the ESC knows the value only at the end of the pulse (1-2 ms) and a new value waits for the next PWM period (0-2.5 ms at the PX4 default 400 Hz) → about **1-4.5 ms** (our derivation from the timing, not a measurement). OneShot125: 0.125-0.25 ms | PWM 1-4.5 ms; OneShot125 0.13-0.25 ms |
| PX4 docs v1.16, "DShot ESCs" + PX4 v1.16.0 source `platforms/nuttx/src/px4/stm/stm32_common/dshot/dshot.c` line 497 | docs: "the number is the speed in kilobits per second"; source: `frame_us = (16 * 1000000) / _dshot_frequency` | 16-bit frame | DShot150/300/600/1200 frame = 107 / 53 / 27 / 13 µs | **< 0.11 ms** for the frame itself |
| ESC firmware processing time (BLHeli, AM32, ...) | - | - | - | **VERIFY** (no source found) |
| Feasibility Tier 2 (b27d09e3) | REPORT Sec. 5 | T_d = 2.6 ms (roll), 0.0 ms (pitch); "poorly fixed" because the band ends at 9-16 Hz | none | 0-2.6 ms, one vehicle, weak |
| Xu et al. 2019 ICUAS, arXiv 1903.06393 [xu2019tailsitter] | Sec. III-A1, eq. (17) | identified P_delay = e^(-0.021 s) (Pixhawk 4 tail-sitter, 250 Hz loop) | It lumps all unmodelled dynamics into one delay (their plant has only one pole besides the integrator) | 21 ms lumped; **not comparable** to our T_d |
| **Fit-half census** (our own, 111 vehicles, `motor_output_protocol`) | `priors_from_logs.json` → `protocols` | PWM only 55; DShot only 10; OneShot 1; UAVCAN 3; mixed timer types in the motor group 39; unknown 3 | coarse (output-to-timer maps are board specific) | **PWM is the most common protocol on large PX4 drones**; whether MAIN is on an IO chip is not resolved |

### 1.3 Crossover frequency f_c

| Source | Exact place | Value | Result |
|---|---|---|---|
| Xu et al. 2019 (Pixhawk 4 tail-sitter) | Sec. III-A3 (PID compensator), Fig. 14 | "The bandwidth of pitch angular velocity is 6.8 Hz ... The phase margin is 44°" (open-loop response); without the notch "less than 4 Hz" | **6.8 Hz** (4 Hz without the notch) |
| Feasibility (b27d09e3, real gains, identified plant) | REPORT Sec. 5 | 11.0 Hz roll, 10.3 Hz pitch | **10-11 Hz**, one vehicle |
| FPV crossover | - | - | **VERIFY** (no source found) |
| **Hard constraint from our data** | `src/noise_generator.py` (spectra only inside the 20-780 Hz band, D-23 Q5) | f_lo = 5 f_c must be ≥ 20 Hz | **f_c ≥ 4 Hz**. The THESIS candidate 3-15 Hz breaks it |

### 1.4 Control-loop rate (fit half, 111 vehicles; `priors_from_logs.json` → `rates`)

| What | Value |
|---|---|
| `IMU_GYRO_RATEMAX` setting | 800 Hz: 74; 400: 14; 2000: 11; 1000: 9; 250: 2; 0: 1 |
| Measured rate = median interval of `vehicle_angular_velocity` (102 logs logged at full rate: high-rate or sysid logger profile, PX4 `logged_topics.cpp` lines 135, 269, 355) | 250 Hz to 1,070 Hz; median 799 Hz |
| Setting 800 Hz → measured | 664-819 Hz; **25 % quantile 667 Hz** |
| Setting 2000 Hz → measured | 999-1,070 Hz (never 2 kHz); setting 1000 Hz → 499-1,067 Hz |
| **Measured more than 5 % away from the setting** | **42 of 102 logs** (first count 39 of 99 used a ratio rule that wrongly dropped 3 full-rate logs; corrected) |

**New finding (landmine for the loop model, THESIS L2).** The real controller rate is often not `IMU_GYRO_RATEMAX`.
- **Mechanism, checked in the PX4 v1.16.0 source:** `BMI088_Gyroscope::ConfigureSampleRate` rounds the FIFO interval to whole samples (`roundf((1e6 / rate) / FIFO_SAMPLE_DT)`), so a 2 kHz IMU asked for 800 Hz reads 3 samples per batch = **666.7 Hz**. `VehicleAngularVelocity::Run()` calls `CalibrateAndPublish` once per FIFO message (with the last filtered sample), and that publishes only if the interval check at line 925 passes; so it publishes at most once per FIFO batch. The rate controller runs on each publication.
- The rate depends on the **control IMU**, which is often not the IMU whose FIFO is logged (gate Q8). Example: on an FMU-V6X the ICM-42688-P FIFO is logged at 8 kHz, but the controller runs at 667 Hz from the 2 kHz BMI088.
- **Consequences:** (1) synthetic drones should sample the control rate from the measured fit-half distribution, not from the setting; (2) real-gain cases (Task 3) should use the measured rate of each log. The feasibility vehicle b27d09e3 runs at 800.0 Hz, so its results are unchanged.

### 1.5 Gyro FIFO rate (the rate of all gyro filters; fit half, 111 vehicles)

Nominal: 2,000 Hz: 54; 6,400: 14; 8,000: 15; 8,192: 3; about 8,371: 25. Measured (dropout below 50 %): median 6,353 Hz, range 1,000-8,192 Hz. FPV (3.2-8 kHz): **VERIFY** (Blaha 2024 runs a 2 kHz loop with an ICM-42688-P; no FIFO-rate source).

### 1.6 Hover motor frequency f_m (fit half; `priors_from_logs.json` → `hover_motor`)

- Valid ESC RPM in steady windows: only **6 of the 52 usable vehicles**: 46.3, 51.7, 68.1, 101.5, 103.0, 126.4 Hz (median 85 Hz).
- Literature: Xu et al. 2019, rotor disturbance "between 75 Hz and 90 Hz" (Sec. III-A1).
- **Design consequence for Task 4:** the generator draws whole real peak sets (D-26). So f_m must **not** be sampled independently; it must be the source vehicle's RPM/60 times the same frequency scale (`noise_generator.sample` returns `source` and `scale`). Otherwise an ESC notch would sit away from the noise peaks. Scenario B is therefore possible only with the RPM source vehicles. **Correction (2026-10-18 follow-up): only 5 of the 6 RPM vehicles are generator sources**: 51603946 has valid RPM in 1 of its 3 windows, and the vehicle spectrum keeps motor frequencies only if every window has them (`spectra_library.vehicle_spectra`).

### 1.7 Real PID gains (fit half, 111 vehicles; inputs to Tier 3b)

| | Kp | Ki | Kd |
|---|---|---|---|
| Roll, median (min-max) | 0.12 (0.026-0.50) | 0.165 (0-0.69) | 0.0024 (0-0.0099) |
| PX4 v1.16 default (`mc_rate_control_params.c` v1.16.0, lines 51-199) | 0.15 | 0.2 | 0.003 |

28 of 111 roll axes use exactly the default gains. Actual filters: gyro cutoff median 40 Hz (15-120 Hz), D-term median 30 Hz (0-80 Hz); 14 of 111 use the ESC-RPM notch.

---

## 2. ICM chip delay (D-25; input to A4, not to A1)

`chip_lowpass_table.json`: every term of the Part 2 grid, plus 116 and 90 Hz (the old optimum of ICM-42686-P sat on the 150 Hz edge). It reproduces the stored top-5 BIC values exactly (largest difference 0.0). The delay is the group delay at 20-100 Hz of the minimum-phase filter with the fitted magnitude (an analog Butterworth). A causal filter with the same magnitude cannot have less phase lag, so this is a **lower bound**.

| Chip (vehicles) | Best term | Terms within BIC 10 (plan rule) | Delay, BIC rule | Terms within 1.5 % of the best fit error (D-25 profile style) | Delay, 1.5 % rule |
|---|---|---|---|---|---|
| ICM-42688-P (13) | 323 Hz, n 3 | 1 | 0.99-1.04 ms | 3 (194/2, 250/2, 323/3) | **0.91-1.37 ms** |
| ICM-42686-P (3) | **116 Hz, n 2** (new grid point) | 2 (90/2, 116/2) | 2.00-3.02 ms | 2 | **2.00-3.02 ms, open at the low end** |
| ICM-45686 (9) | 696 Hz, n 6 | 1 | 0.88-0.89 ms | 10 | 0.25-0.89 ms |
| IIM-42652 (2) | 539 Hz, n 6 | 1 | 1.14-1.16 ms | 1 | 1.14-1.16 ms |

**Two problems with the plan's rule (step 4):**
1. **The BIC rule gives a falsely narrow range.** BIC treats neighbouring Welch bins as independent (REPORT 16, LOW 5), so its gains are inflated, and only one term survives for ICM-42688-P. The fit-error rule (the one D-25 used for "not identifiable") gives the honest, wider range.
2. **ICM-42686-P is not bounded below.** On the original grid its optimum was 150 Hz, the lowest grid value. With 116 and 90 Hz added, the optimum moves to 116 Hz and 90 Hz is within BIC 10. Its corner trades off against the floor (only 3 vehicles). Its delay is at least 2 ms, with no upper limit from the data.
   - **Effect on Phase 3:** the generator uses the stored magnitude (150 Hz, n 2). The two terms differ by about 1.1 dB at 100 Hz. This is inside D-25 ("corner and order not identifiable") and limitation 4 (small chip group). No Phase 3 number changes unless the author decides to refit.

**PX4 driver (v1.16.0 `ICM42688P.hpp`, register config, read today):** UI filter order bits cleared (1st order) and bandwidth bits cleared (ODR/2, i.e. 4 kHz at 8 kHz ODR); anti-alias filter on at 585 Hz; on-chip notch left enabled (`GYRO_NF_DIS` cleared). A 2nd-order 585 Hz filter alone would delay about 0.4 ms, less than the 0.9-1.4 ms lower bound from the measured roll-off. So the measured roll-off is still not explained (**VERIFY stays open**; the TDK datasheet could not be downloaded here).

---

## 3. Proposals for A1 (the author decides)

| Parameter | THESIS candidate | Evidence (above) | Proposal |
|---|---|---|---|
| Control loop rate | 800 Hz; 400, 1000, 2000 | Measured rates 250-1,070 Hz; 42 of 102 off the setting | Sample from the **measured fit-half rates** (empirical); real-gain cases use the log's measured rate |
| FIFO rate | 2-8 kHz | 2,000 / 6,400 / 8,000 / 8,192 / 8,371 Hz | Sample (FIFO rate, control rate) **pairs** from the fit-half logs, so the pair stays physical. Ruling after A1: only logs whose logged FIFO IMU is the control IMU (38 of 111) and whose controller rate is logged at full rate; other pairs mix two IMUs |
| f_c (large) | 3-15 Hz | 6.8 Hz (Xu), 10-11 Hz (ours); constraint f_c ≥ 4 Hz | **4-15 Hz** log-uniform |
| τ_m (large) | 30-100 ms | 23-50 ms (Bangura), 29-34 ms (ours); nothing above 50 ms | **20-60 ms** log-uniform (100 ms has no source) |
| T_d (large) | 0.5-5 ms | PWM 1-4.5 ms (derived), DShot < 0.11 ms + ESC (VERIFY), IO chip +5.4 ms, PWM most common | **0.5-5 ms**; the IO-chip case (up to about 10 ms) only as a sensitivity check |
| f_m (large) | 50-150 Hz | 46-126 Hz (6 vehicles), 75-90 Hz (Xu) | Not sampled: taken from the generator's source vehicle (Sec. 1.6) |
| FPV extension | 10-40 Hz, 10-40 ms, 0.1-1 ms, 2-8 kHz | τ_m 20-27 ms measured, 15-35 ms design range (Blaha); nothing for f_c, rates | τ_m 15-35 ms; the rest stays **VERIFY** (extension is on the cut list) |
