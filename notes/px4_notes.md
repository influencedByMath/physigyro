# PX4 (and Betaflight) source notes and VERIFY answers

Made 2026-10-14 (Day 17, Phase 1 task "read the PX4 filter tuning guide, skim `VehicleAngularVelocity.cpp`, answer the VERIFY items"). Earlier source checks (2026-10-06 and 2026-10-10) are already written into THESIS.md Sections 4.1, 4.2 and 4.4.

---

## 1. PX4 "MC Filter Tuning & Control Latency" guide (docs.px4.io/main, read 2026-10-14)

What the official guide says, in short:

- **Pipeline:** on-chip DLPF off (or highest cutoff) → notch filters (2 static: `IMU_GYRO_NF0/1_FRQ/_BW`; dynamic: `IMU_GYRO_DNF_EN` bit 0 = ESC RPM, bit 1 = onboard FFT, with `IMU_GYRO_DNF_MIN/_BW/_HMC`) → gyro lowpass `IMU_GYRO_CUTOFF` → separate D-term lowpass `IMU_DGYRO_CUTOFF` → optional motor slew-rate limit. "Sampling and filtering always run at the full raw sensor rate." This matches our model (THESIS 4.1-4.2).
- **Latency table in the guide:** gyro cutoff 30 Hz ≈ 8 ms, 60 Hz ≈ 3.8 ms, 120 Hz ≈ 1.9 ms. Our rule of thumb √2/(2π f_c) gives 7.5 / 3.75 / 1.9 ms → agrees within 7 %. (Sanity check for `filters.py`; no action.)
- **Other latency sources named:** soft mounts, on-chip filters, IO-chip (MAIN) outputs add ≈ 5.4 ms on Pixhawk vs AUX outputs, PWM vs DShot. → Supports the range of the extra delay $T_d$ (Tier 1). Note: a drone with motors on MAIN pins can have $T_d$ several ms larger.
- **Recommended procedure:** hover + roll/pitch manoeuvres in Stabilized mode for < 30 s; set `IMU_DGYRO_CUTOFF` = 30, then raise `IMU_GYRO_CUTOFF` in 10 Hz steps, one flight and one log per step, and compare the "Actuator Controls FFT" plot in Flight Review; pick the highest value before noise rises. Then the same for the D-term cutoff. Static notch only for a low-frequency spike.
- **Typical values given:** FPV racers gyro ≈ 120 Hz, D-term 50-80 Hz; larger vehicles gyro ≈ 80 Hz. "Do not keep the cutoffs too far apart" (example of a bad pair: D = 15 with gyro = 80).
- **Logging:** the guide tells users to enable the **high-rate** logging profile (bit 4), which gives full-rate controller outputs, not raw gyro.

**Why this matters for the thesis (Introduction / motivation):** the official method is **manual trial and error**: about 5-10 test flights per filter, judged by eye on an FFT plot, with no stability check. This is the clearest statement of the problem we solve. Quote it in the Introduction (cite [px4docs_filtertuning]).

**Inconsistency found:** the guide says "the default `IMU_GYRO_CUTOFF` (30 Hz) is very low". The source code (v1.15.4-v1.17.0 `imu_gyro_parameters.c`) has a default of **40 Hz**, and real logs confirm 40 Hz as the most common value (audit 2026-10-06). The docs are out of date. Always use the source / the log (decision D-09, gate Q11); do not cite the guide for the default value.

---

## 2. VERIFY items answered today

### 2.1 Betaflight phase-compensation filter (THESIS Section 4.3) → ANSWERED: not in the filter path

Checked in Betaflight `master` (last change of `common/filter.c`: 2026-09-27):

- `phaseCompInit/Update/Apply/Reset` are defined in `src/main/common/filter.c`.
- The only user is `pidRuntime.chirpFilter` in `flight/pid.c` / `flight/pid_init.c`, inside `#ifdef USE_CHIRP`. It shapes the **chirp excitation signal** (a lead-lag on the test signal) used by the optional "chirp" system-identification flight mode.
- It is **not** applied to the gyro or the D-term, and it only runs in `CHIRP_MODE`.

**Consequence:** do not include it in the Betaflight filter model (extension E9).

### 2.2 `sensor_gyro_fifo.timestamp_sample` = time of the last sample? (THESIS 7.4.4) → ANSWERED: yes (checked for the ICM-42688-P driver)

PX4 v1.16.0 `src/drivers/imu/invensense/icm42688p/ICM42688P.cpp`, `RunImpl()` and `FIFORead()`:

- With data-ready interrupts, `timestamp_sample` is the time of the data-ready interrupt that fires when the FIFO watermark (`_fifo_gyro_samples` samples) is reached, i.e. when the **newest** sample arrived. Without the interrupt it is `now` (the time of the FIFO read).
- If the FIFO holds one sample more than expected, the driver subtracts one sample period (`timestamp_sample -= FIFO_SAMPLE_DT`) and reads one sample less, so the timestamp again belongs to the **last sample read**.
- `PX4Gyroscope::updateFIFO()` copies `timestamp_sample` unchanged into `sensor_gyro_fifo`.

**Consequence:** our assumption (timestamp = last sample of the message) is right. The observed ±1-sample jitter is explained by the polled path (`now` instead of the interrupt time) and the "one extra sample" correction. Our handling (per-message gap detection + one fitted time line per segment, REPORT Section 3) is correct. Other IMU drivers (BMI088, ICM-20602, ...) follow the same `PX4Gyroscope::updateFIFO` interface; check one more driver only if a pilot-study log shows a timing problem.

### 2.3 Still open (not Phase 1; tracked in THESIS Section 17)

| Item | Phase |
|---|---|
| `SENS_BOARD_ROT` rotations other than 0 and 4; board trims `SENS_BOARD_*_OFF` | Phase 3 pilot study |
| `esc_status.esc_rpm` = mechanical RPM for every ESC type | Phase 3 (per-vehicle unit check, gate Q16) |
| Tier 1 plant priors from literature ($f_c$, $\tau_m$, $T_d$) | Phase 2 leftover; sources listed in `notes/literature.md` (Bangura 2017, Bergamasco 2014, NeuroBEM, this guide for $T_d$) |
| m2cgen support for the chosen boosting model | Phase 6 |
| APCs of IEEE Access, Sensors, Aerospace; IEEE ESL charges | Phase 7 |
