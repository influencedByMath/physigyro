# Phase 3, Part 1 Implementation Plan: from the log index to the pilot-study proposal

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find about 200 public PX4 logs with raw gyro FIFO, split their vehicles 50/50 into fit and test halves (fixed hash), download a fit-half pilot set, and produce a pilot-study report that **proposes** every TBD data-quality threshold for the author's approval.

**Architecture:** A pure, stdlib-only streaming ULog header parser (`ulog_header.py`) is driven by a polite HTTP range-request screener (`log_screen.py`) over a filtered snapshot of the Flight Review index. `split.py` turns screened FIFO logs into vehicle keys and assigns fit/test by a salted SHA-256 hash. `fc_data.py` and `data_quality.py` compute per-log flight-controller data, gate statistics and reason codes, with every threshold passed in as an argument (`Thresholds`). The pilot study (`experiments/pilot_study/`) only measures and proposes; nothing is frozen without the author.

**Tech Stack:** Python 3.13 (`.venv`), numpy, scipy, pandas, pyulog 1.2.4, matplotlib, urllib (stdlib). No new dependencies.

**Spec:** `THESIS.md` Sections 7.4.1-7.4.5 (pipeline, gates Q1-Q16, pilot study, FC data), 10 (Phase 3 checklist), decisions D-09, D-10, D-13, D-17 and the new D-19 to D-21 (Task 1).

## Global Constraints

- Run Python as `.venv/Scripts/python`; tests with `.venv/Scripts/python -m pytest -q tests` (44 pass before this plan; keep all green).
- No new packages (no pyarrow): tables are CSV or JSONL, arrays NPZ.
- Network politeness (THESIS 7.4.2 B): at most 4 parallel requests, a pause of at least 1 s after every request, User-Agent `px4-noise-thesis/0.1 (academic research; PX4 Flight Review public logs, CC-BY)`, header reads with `Range: bytes=0-2999999`.
- Index snapshot: `https://review.px4.io/dbinfo` (redirects to `https://cdn.logs.px4.io/dbinfo.json`, gzipped JSON list).
- Step A filter (THESIS 7.4.2 A): `mav_type` in {Quadrotor, Hexarotor, Octorotor}; `sys_hw` upper-case contains neither `SITL` nor `SIM`; `source` != `CI`; `duration_s` >= 60; `log_date` >= `2021-01-01`.
- Split before downloading full logs; the pilot uses fit-half vehicles only; no test-half log is downloaded before the thresholds are frozen as a Section 2 decision (D-10).
- Forced into the fit half (D-17, D-19), by `sys_uuid`: `000600000000383832353233510d002e0040` (b27d09e3), `000600000000383236313533511100370018` (a2571040), `0006000000003332333533335117004a0045` (b4a9719c), `000600000000373635363233511500260030` (220b5990).
- Split salt (fixed now, never changed): `px4-noise-split-2026-10-14`.
- Every number that may reach the paper is produced by code in this repository.
- Code style: match `src/` (short plain-English docstrings with THESIS section references, numpy-vectorized, no classes where a function will do).
- Git: commit only if the author agrees (harness rule); if yes, on branch `phase3`.

## Review Focus

1. The server returns an HTML error page, a 404, or the stream drops mid-header: the screen row must carry `err` and `status="error"`, never `fifo=False`. (Task 3 test `test_screen_log_reports_error_not_negative`.)
2. The header runs past `max_bytes` before the first subscription (`A`) message: `status="incomplete"` (unknown), not "no FIFO". (Task 2 test `test_incomplete_header_is_unknown`.)
3. `sensor_gyro_fifo` is subscribed but no FIFO data message arrives inside the read window: keep `fifo=True` and report `fifo_data_msgs=0` so the pilot can count such logs. (Task 2 test `test_fifo_subscribed_without_data`.)
4. Invalid vehicle IDs (empty, `REDACTED...`, board prefix followed only by zeros, trailing newline or NULs): the fallback key of gate Q14 is used, and it is deterministic. (Task 5 test `test_vehicle_key_rules`.)
5. A log without any default-parameter (`Q`) record (older firmware): `defaults_known=False` and `default=None`, never "default = actual". (Task 6 test `test_defaults_unknown_without_q_records`.)

---

### Task 1: Record the new decisions

**Files:**
- Modify: `THESIS.md` Section 2 (rows D-19, D-20, D-21), Section 10 Phase 2 (P2-1 note), Section 17 (session log row, VERIFY list), Section 7.4.3 gate Q11 (defaults-unknown rule)
- Modify: `.gitignore` (add `data/index/`)

- [ ] **Step 1:** Add D-19 (vehicle of log 220b5990 forced into the fit half; reason: fully processed in the 2026-10-06 audit, its 49 % dropout is evidence for gate Q4), D-20 (split = parity of SHA-256 of `salt|vehicle_key`, salt `px4-noise-split-2026-10-14`; reason: reproducible, order-independent, stable when more vehicles are screened), D-21 (screen = random vehicle order by salted hash, newest qualifying log per vehicle; the 23 random-sample FIFO logs of the audit are reused; the 21 keyword-selected logs only as R1 fallback; reason: logs per vehicle are very uneven, median 2, max 1,276).
- [ ] **Step 2:** Gate Q11 text: "If the log has no default-parameter record at all, the defaults are unknown (flag `defaults_known = False`); otherwise a missing default means default = actual" (PX4 v1.16.0 `logger.cpp::write_parameter_defaults` writes a default only when it differs from the current value).
- [ ] **Step 3:** Phase 2 note: P2-1 fixed 2026-10-14 (`notch_coeffs` clamps f0 and bandwidth to >= 0.001 fs as PX4 `NotchFilter::setParameters`; no feasibility result changed: smallest chosen bandwidth 7.5 Hz at 6.4 kHz); independent check: exact single-rate discrete closed-loop poles agree 200/200, PM within 0.006°.
- [ ] **Step 4:** Run the test suite; expected `44 passed`.

### Task 2: Streaming ULog header parser

**Files:**
- Create: `src/ulog_header.py`
- Test: `tests/test_ulog_header.py`

**Interfaces:**
- Produces: `class HeaderParser(param_prefixes=PARAM_PREFIXES, stop_after_data=3000)` with `feed(chunk: bytes) -> None`, `done() -> bool`, `result() -> dict`. `result()` keys: `status` (`"ok"`, `"incomplete"`, `"not_ulog"`), `info` (dict: `sys_uuid`, `ver_hw`, `ver_sw`, `ver_sw_release`, `sys_name`, ... decoded by the key's type), `params` (dict name -> value, only names starting with a prefix in `PARAM_PREFIXES`), `defaults_system`, `defaults_setup` (same filter), `n_default_records` (all `Q` records, unfiltered), `topics` (dict name -> sorted list of multi_ids), `data_counts` (dict name -> number of `D` messages seen), `n_data`, `fifo` (bool: `sensor_gyro_fifo` subscribed), `fifo_data_msgs`, `n_dropouts`, `bytes_parsed`.
- `PARAM_PREFIXES = ("IMU_", "SDLOG_", "MC_ROLLRATE_", "MC_PITCHRATE_", "MC_YAWRATE_", "SENS_BOARD_", "CAL_GYRO", "DSHOT_", "SYS_AUTOSTART", "MPC_THR_HOVER", "CA_ROTOR_COUNT", "UAVCAN_", "PWM_MAIN_", "MOT_")`.

- [ ] **Step 1: Write the failing tests.** A test helper `_ulog_bytes(messages)` builds a file from literal pieces: 16-byte header (`b"ULog\x01\x12\x35"` + version byte + uint64 timestamp), then `uint16 size, uint8 type, payload` per message. Tests:
  - `test_parses_info_params_defaults_and_topics`: `I` `char[36] sys_uuid` = `"0006…"`, `I` `uint32_t ver_sw_release` = `0x01100000`, `P` `float IMU_GYRO_CUTOFF` = 80.0, `P` `int32_t SDLOG_PROFILE` = 273, `P` `float NAV_ACC_RAD` = 2.0 (filtered out), `Q` types=3 `float IMU_GYRO_CUTOFF` = 40.0, `Q` types=1 `int32_t SDLOG_PROFILE` = 1, `A` multi 0 id 5 `sensor_gyro_fifo`, `A` multi 1 id 6 `sensor_gyro_fifo`, then `D` messages for id 5 → `params == {"IMU_GYRO_CUTOFF": 80.0, "SDLOG_PROFILE": 273}`, `defaults_setup == {"IMU_GYRO_CUTOFF": 40.0}`, `defaults_system == {"IMU_GYRO_CUTOFF": 40.0, "SDLOG_PROFILE": 1}`, `n_default_records == 2`, `topics == {"sensor_gyro_fifo": [0, 1]}`, `fifo is True`, `info["ver_sw_release"] == 0x01100000`.
  - `test_feeding_in_tiny_chunks_gives_same_result`: the same bytes fed in 1-byte and 7-byte chunks give an identical `result()`.
  - `test_not_ulog`: `b"<html>..."` gives `status == "not_ulog"` and `done() is True`.
  - `test_incomplete_header_is_unknown`: definitions only (no `A`), stream ends: `status == "incomplete"`, `fifo is False` but status shows it is unknown.
  - `test_fifo_subscribed_without_data`: `A sensor_gyro_fifo` plus 3,000 `D` for another topic: `fifo is True`, `fifo_data_msgs == 0`, `done() is True`.
  - `test_matches_pyulog_on_real_log` (skip if `data/raw_logs/b27d09e3-4542-4496-a18b-2c3c5317024e.ulg` is missing): feed the first 3,000,000 bytes; compare `info["sys_uuid"]`, all filtered `params`, `defaults_setup`, `defaults_system` with `pyulog.ULog(path, [])` (`msg_info_dict`, `initial_parameters`, `get_default_parameters(1)`, `get_default_parameters(0)`, filtered by the same prefixes); `"sensor_gyro_fifo" in topics`.
- [ ] **Step 2:** Run `.venv/Scripts/python -m pytest -q tests/test_ulog_header.py`; expected: all fail (module missing).
- [ ] **Step 3:** Implement `HeaderParser`. Message layouts (ULog spec): `I` = `uint8 key_len, key "type name", value`; `P` = `uint8 key_len, key, value (int32 or float)`; `Q` = `uint8 default_types (bit0 system, bit1 current setup), uint8 key_len, key, value`; `A` = `uint8 multi_id, uint16 msg_id, name`; `D` = `uint16 msg_id, data`; `O` = `uint16 duration`; skip `B F M L C S R T`. `status="ok"` once at least one `A` was seen; `done()` when not ULog, or when `n_data >= stop_after_data` after the first `A`.
- [ ] **Step 4:** Run the test file, then the whole suite; expected all pass.

### Task 3: Index snapshot, Step A filter, sampling order, polite screen runner

**Files:**
- Create: `src/log_screen.py`
- Test: `tests/test_log_screen.py`

**Interfaces:**
- Consumes: `ulog_header.HeaderParser`.
- Produces:
  - `fetch_index(dest_dir: str) -> dict` (downloads the snapshot to `data/index/dbinfo_<date>.json.gz`, returns provenance `{date, url, sha256, n_entries}`; also written to `data/screen/index_snapshot.json`).
  - `load_index(path: str) -> list[dict]`; `is_candidate(entry: dict) -> bool` (Step A).
  - `clean_uuid(raw: str) -> str | None` (strip whitespace and NULs; `None` if empty, contains `REDACTED`, or everything after the first 4 characters is `0`).
  - `sampling_order(candidates: list[dict], salt: str, first_log_ids: list[str]) -> list[dict]`: one log per index vehicle (`clean_uuid`; logs without a valid uuid are each their own group), the newest `log_date` (ties: longer `duration_s`, then `log_id`), groups ordered by `sha256(salt + "|" + group)`; logs in `first_log_ids` come first.
  - `screen_log(entry: dict, max_bytes=3_000_000, opener=urllib.request.urlopen) -> dict` (one row: index provenance + `HeaderParser.result()` + `bytes_read`, `seconds`, `err`).
  - `run_screen(order, out_jsonl, target_fifo_vehicles=200, max_logs=4000, workers=4, pause_s=1.0) -> dict` (resumable: skips `log_id`s already in `out_jsonl`; appends one JSON line per log; stops at the target; returns counts).
  - CLI: `python src/log_screen.py screen --target 200`.
- Constant: `AUDIT_RANDOM_FIFO_LOGS` = the 23 `log_id`s of the 2026-10-06 random-sample FIFO logs (from the audit's `probe_results2.json`, group `rand`).

- [ ] **Step 1: Write the failing tests:**
  - `test_is_candidate_table`: literal entries: Quadrotor/120 s/2024 → True; `PX4_SITL` → False; `source="CI"` → False; 59 s → False; `2020-12-31` → False; `mav_type="VTOL Tailsitter"` → False; Hexarotor and Octorotor → True.
  - `test_clean_uuid_rules`: `"000600000000383832353233510d002e0040"` kept; `"X1-CPX1002605\n"` → `"X1-CPX1002605"`; `""`, `"REDACTED\x00\x00"`, `"000a00000000000000000000000000000000"` → `None`.
  - `test_sampling_order_newest_per_vehicle_and_stable`: 3 logs of vehicle A (dates 2024-01-01, 2026-02-01, 2026-02-01 with 100 s and 300 s) and 1 of B: A appears once with the 2026-02-01/300 s log; adding a vehicle C does not change the relative order of A and B; `first_log_ids` are placed first.
  - `test_screen_log_reads_header_and_stops_early` (skip if the b27d09e3 file is missing): `opener` returns a file object over the local log; row has `fifo is True`, `bytes_read < 3_000_000`, `err is None`.
  - `test_screen_log_reports_error_not_negative`: `opener` raises `urllib.error.HTTPError(404)`; another returns `b"<html>"`: rows have `status == "error"` / `"not_ulog"` and `err` set.
  - `test_run_screen_resumes_without_duplicates`: fake `screen_log` via monkeypatch on the module, an existing JSONL with 2 rows; the second run screens only the remaining logs; total lines = number of logs.
- [ ] **Step 2:** Run the test file; expected all fail.
- [ ] **Step 3:** Implement. Pacing: each worker sleeps `pause_s` after every request; a lock guards the JSONL append and the counters.
- [ ] **Step 4:** Run the tests and the suite; expected all pass.

### Task 4: Run the screen

- [ ] **Step 1:** `.venv/Scripts/python src/log_screen.py fetch-index` → snapshot plus `data/screen/index_snapshot.json`.
- [ ] **Step 2:** In the background: `.venv/Scripts/python src/log_screen.py screen --target 200` → `data/screen/screen_results.jsonl`. Expected run time 1-2 h.
- [ ] **Step 3:** Summary printed by `python src/log_screen.py summary`: logs screened, status counts, FIFO logs, FIFO vehicles, hit rate with Wilson 95 % CI, profiles, sample-rate-relevant parameters, `defaults_known` share. The author gets these numbers before the split.

### Task 5: Vehicle keys and the 50/50 split

**Files:**
- Create: `src/split.py`
- Test: `tests/test_split.py`
- Output: `data/screen/split.csv` (columns `vehicle_key, split, forced, log_ids, sys_hw, n_fifo_logs`)

**Interfaces:**
- Consumes: screen rows (Task 3), `log_screen.clean_uuid`.
- Produces: `SALT = "px4-noise-split-2026-10-14"`; `FORCED_FIT_UUIDS` (the 4 IDs of the Global Constraints); `vehicle_key(row: dict) -> str` (`"uuid:" + clean_uuid(info.sys_uuid or index vehicle_uuid)`, else `"fb:" + sha256(sys_hw | SYS_AUTOSTART | sorted IMU_GYRO_*, IMU_DGYRO_*, MC_ROLLRATE_*, MC_PITCHRATE_* values)[:16]`, gate Q14); `assign_split(key: str) -> str` (`"fit"` if the SHA-256 integer of `SALT + "|" + key` is even, else `"test"`; forced keys → `"fit"`); `make_split(rows: list[dict]) -> pandas.DataFrame`; `write_split(df, path)` refuses to overwrite a file whose existing rows would change.

- [ ] **Step 1: Write the failing tests:** `test_vehicle_key_rules` (valid uuid → `uuid:` key; invalid uuid → `fb:` key, same parameters → same key, a different `MC_ROLLRATE_D` → different key); `test_assign_split_is_deterministic_and_balanced` (10,000 synthetic keys: share of `fit` within 0.48-0.52; calling twice gives the same answer); `test_forced_vehicles_are_fit` (each of the 4 forced uuids → `fit`); `test_write_split_refuses_changes` (writing a different assignment over an existing file raises `ValueError`; adding new vehicles is allowed).
- [ ] **Step 2:** Run; expected fail. **Step 3:** Implement. **Step 4:** Run tests and suite; pass.
- [ ] **Step 5:** Create `data/screen/split.csv` from the screen results. Report the fit/test vehicle counts.

### Task 6: Flight-controller data per log (`fc_data.py`)

**Files:**
- Create: `src/fc_data.py`
- Test: `tests/test_fc_data.py`

**Interfaces:**
- Consumes: `noise_data.fc_settings`, `noise_data.esc_rpm`, `noise_data.fifo_dataset`, `noise_data._t`.
- Produces: `defaults_known(ulog) -> bool` (any `Q` record of either type); `imu_chip(device_id: int) -> dict` (`devtype = (device_id >> 16) & 0xFF`, `address = (device_id >> 8) & 0xFF`, `bus = (device_id >> 3) & 0x1F`, `bus_type = device_id & 0x7`, `name` from a table copied from PX4 v1.16.0 `src/drivers/drv_sensor.h`, `"unknown"` otherwise); `logged_rate_hz(ulog, topic: str) -> float | None` (median message rate); `fc_info(ulog) -> dict` with `actual` (FilterSettings), `default` (FilterSettings or `None` if defaults unknown), `defaults_known`, `gains_roll`, `gains_pitch` (Gains), `f_ctrl`, `version`, `hw`, `sys_uuid`, `profile` (`SDLOG_PROFILE`), `fifo_chip`, `control_chip`, `fifo_is_control_gyro`, `fft_notch`, `changed_in_flight`, `esc_rate_hz`, `has_rpm_topic`, `has_highrate` (`vehicle_angular_velocity` and `vehicle_torque_setpoint` logged at >= 0.8 x `IMU_GYRO_RATEMAX`), `has_sysid` (profile bit 3 or 4).

- [ ] **Step 1: Write the failing tests:** `test_imu_chip_bit_layout` (`device_id = (0x26 << 16) | (0 << 8) | (1 << 3) | 2` → devtype 0x26, bus 1, bus_type 2, name = the drv_sensor.h name of 0x26); `test_defaults_unknown_without_q_records` (a `pyulog`-like stub whose `get_default_parameters` returns `{}` for both types → `defaults_known` False and `fc_info(...)["default"] is None`); `test_fc_info_on_real_log` (skip if missing; b27d09e3: `defaults_known` True, `default.f_gyro == 40.0`, `default.f_dterm == 30.0`, `actual.f_gyro == 80.0`, `has_highrate` True, `esc_rate_hz` within 180-220, `fifo_is_control_gyro` False).
- [ ] **Step 2-4:** Run (fail), implement, run (pass), suite green.

### Task 7: Gate statistics and gates with explicit thresholds (`data_quality.py`)

**Files:**
- Create: `src/data_quality.py`
- Modify: `src/noise_data.py` (`steady_windows` also returns the throttle steadiness, the std of the mean `actuator_motors` command in the window; existing callers unchanged)
- Test: `tests/test_data_quality.py`

**Interfaces:**
- Produces: `@dataclass(frozen=True) class Thresholds` with fields `d_max, l_min_s, f_min, f_max, jitter_max, clip_max, window_s, t_edge_s, sp_std_max_dps, thr_std_max, resolution_hz, min_averages, windows_per_vehicle, peak_prominence, rpm_rate_min_hz`; `CANDIDATES = Thresholds(0.05, 10.0, 20.0, 800.0, 0.01, 0, 10.0, 5.0, 15.0, 0.05, 1.5, 20, 3, 0.5, 50.0)` (THESIS candidates; `thr_std_max = 0.05` has no THESIS candidate). Every field's docstring says "candidate, not frozen (D-10)".
- `nperseg_for(fs: float, resolution_hz: float) -> int` (= `round(fs / resolution_hz)`).
- `log_stats(ulog, gs) -> dict` (no thresholds: `dropout_fraction`, `fs_nominal`, `fs_measured`, `jitter`, `n_rate_changes`, `n_segments`, `longest_segment_s`, `airborne_s`, `clipped_samples`).
- `log_gates(stats, fc, th: Thresholds) -> tuple[list[str], list[str]]` and `window_gates(win, th) -> tuple[list[str], list[str]]` return `(rejections, flags)`. Rejection codes: `Q4_dropout`, `Q5_rate`, `Q6_jitter`, `Q9_clipping`, `Q10_unsteady`, `Q13_few_averages`. Flag codes (the item stays, the flag is stored): `Q8_fifo_not_control`, `Q11_defaults_unknown`, `Q16_no_rpm`, `fft_notch`, `changed_in_flight`. An empty rejection list means pass. Gate Q3 (duplicate uploads) cannot occur in Part 1 because the screen takes one log per vehicle; Part 2 adds it if a vehicle contributes several logs.
- `Funnel` helper: `add(stage: str, n_in: int, reasons: Counter)`; `table() -> pandas.DataFrame` (THESIS Table 1 layout).

- [ ] **Step 1: Write the failing tests:** table-driven `test_log_gates_reason_codes` (literal stats dicts: dropout 0.08 with `d_max` 0.05 → rejections `["Q4_dropout"]`; fs 1000 with `f_max` 800 → `["Q5_rate"]` because 0.8 x 500 < 800; fs 2000 → no rejection; jitter 0.02 → `["Q6_jitter"]`; `defaults_known` False → flag `Q11_defaults_unknown`, no rejection); `test_log_gates_use_given_thresholds` (the dropout-0.08 stats pass with `d_max = 0.10`); `test_nperseg_for` (6400 Hz, 1.5 Hz → 4267; 2000 Hz → 1333); `test_funnel_counts` (two stages, counts and reasons add up).
- [ ] **Step 2-4:** Run (fail), implement, run (pass), suite green.

### Task 8: Pilot downloads (fit half only)

**Files:**
- Modify: `src/log_screen.py` (add `download_log(entry, dest_dir) -> str`: resumes with a Range request from the existing size, checks the final size against `Content-Range`/`Content-Length` and the ULog magic, polite pause)
- Test: `tests/test_log_screen.py::test_download_resumes_and_checks_magic` (opener stub serving a byte string in two parts)
- Output: `data/screen/pilot_set.csv`, files in `data/raw_logs/`

- [ ] **Step 1-4:** TDD for `download_log` as above.
- [ ] **Step 5:** Pilot set = the 4 forced vehicles (logs already local except 220b5990) plus the first 25 fit-half vehicles in `sha256(SALT + "|pilot|" + key)` order. Download their screened FIFO logs. Expected about 1 GB.

### Task 9: Pilot study and threshold proposal (STOP for approval)

**Files:**
- Create: `experiments/pilot_study/run_pilot.py`, `experiments/pilot_study/REPORT.md`, `experiments/pilot_study/results.json`, `experiments/pilot_study/figures/`

The runner implements THESIS 7.4.3 pilot steps 1-5 on the pilot set only:
1. **D_max / L_min:** on every pilot log with dropout fraction < 1 %, insert artificial gaps drawn from the real gap lengths of the pilot logs at fractions 0, 1, 2, 5, 10, 20, 30 %; band powers in the 8 bands of `pipeline_checks.py` (20-800 Hz); proposal = largest fraction with median |error| < 1 dB and at least one segment >= L_min.
2. **f_max / f_min:** histogram of measured FIFO rates; proposal = largest f_max with >= 90 % of logs meeting 0.8 fs/2 >= f_max; on RPM logs check that 3 x the highest motor frequency < f_max; f_min from the plots (vibration-dominated) and <= 5 x the smallest modelled crossover.
3. **W and resolution:** 5, 10, 20 s windows inside the same steady flight; W = shortest with median band difference < 1 dB to the 20 s estimate; averages at that W for each sample rate.
4. **Steady flight:** distributions of rate-setpoint std and throttle std over all airborne windows; proposed limits and the share of airborne time kept.
5. **Jitter, clipping, RPM logged rate, windows per vehicle, defaults_known share, FIFO-not-control-gyro share:** counts and proposals.

- [ ] **Step 1:** Write and run the runner (it reads only `pilot_set.csv` vehicles; it asserts that every vehicle is `fit` in `split.csv`).
- [ ] **Step 2:** Write `REPORT.md`: a table "threshold | THESIS candidate | proposed | evidence (figure / number)", plus open decisions found (e.g., second log per vehicle when the first fails; parametric floor model).
- [ ] **Step 3:** STOP. Ask the author to approve or change the values. Only after approval: record them as one new Section 2 decision (D-10 freeze) and start Part 2.

---

## Part 2 (separate plan after the freeze; listed so nothing is lost)

Download all screened FIFO logs of both halves; apply the frozen gates; build `data/spectra/` (per-window, per-axis PSD on one common grid plus metadata, CC-BY provenance); data funnel Table 1; PCA + k-means on the fit half (Fig. 2) with the chip / rate / firmware cross-tab; parametric noise model (floor model to be decided with evidence) with motor frequencies fixed from RPM where valid; generator and the synthetic-vs-real comparison; go/no-go (>= 100 vehicles, minimum 30).
