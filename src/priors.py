"""Tier 1 plant priors (THESIS 7.1.1; Phase 4 Task 1): fit-half guard, chip-filter delay and log evidence.

Rules (author, Phase 3/4): every prior that uses our own logs reads the FIT half only. The test half is for
evaluation, never for tuning (this includes the priors), so every input path and table goes through
``require_fit_half`` / ``require_fit_rows``.

Chip low-pass delay (D-25): the per-chip term |H|^2 = 1 / (1 + (f/f_c)^2n) is identified in magnitude only. Its
delay is computed for the minimum-phase filter with that magnitude, i.e. the analog Butterworth of order n. A
causal filter with the same magnitude cannot have less phase lag, so this is the smallest delay consistent with
the fitted magnitude.
"""
import json
import os
import re

import numpy as np

_TEST_PART = re.compile(r"(^|[\\/])test_[^\\/]*")


def require_fit_half(path):
    """Return ``path``; raise ValueError if any part of it names a test-half file or folder (``test_*``)."""
    if _TEST_PART.search(str(path)):
        raise ValueError(f"refusing a test half input (priors use the fit half only): {path}")
    return path


def require_fit_rows(df):
    """Raise ValueError if a table has any row whose ``split`` is not 'fit'."""
    bad = set(df["split"]) - {"fit"}
    if bad:
        raise ValueError(f"refusing rows of the test half (split values {sorted(bad)})")
    return df


# ------------------------------------------------------------------ chip low-pass delay (D-25)

def lowpass_group_delay(f, fc, n):
    """Group delay [s] at ``f`` of the n-th order analog Butterworth with corner ``fc``.

    Poles p_k = w_c exp(j (pi/2 + (2k-1) pi / (2n))), k = 1..n; tau(w) = sum_k -Re p_k / ((w - Im p_k)^2 + Re p_k^2).
    At w -> 0 this is 1 / (w_c sin(pi / 2n)) (sqrt(2) / w_c for n = 2, THESIS 4.4)."""
    w = 2 * np.pi * np.asarray(f, dtype=float)
    wc = 2 * np.pi * float(fc)
    k = np.arange(1, int(n) + 1)
    p = wc * np.exp(1j * (np.pi / 2 + (2 * k - 1) * np.pi / (2 * int(n))))
    tau = (-p.real / ((w[..., None] - p.imag) ** 2 + p.real ** 2)).sum(-1)
    return tau


def band_delay_range(fc, n, band=(20.0, 100.0), n_points=161):
    """(min, max) group delay [s] of the chip term over ``band`` (log-spaced points)."""
    f = np.geomspace(band[0], band[1], n_points)
    tau = lowpass_group_delay(f, fc, n)
    return float(tau.min()), float(tau.max())


def indistinguishable_terms(rows, chip, rule="bic", tol=10.0):
    """Low-pass terms of ``chip`` that fit about as well as the best term (never "no term").

    rows: dicts with chip, term ([f_c, n] or None), bic, rms_db_pooled (chip_lowpass_table.json).
    rule "bic": BIC within ``tol`` of the best term's BIC (plan Task 1 step 4).
    rule "rms": pooled fit error within a fraction ``tol`` of the best term's (D-25 profile test: 1.5 %)."""
    terms = [r for r in rows if r["chip"] == chip and r["term"] is not None]
    if rule == "bic":
        best = min(r["bic"] for r in terms)
        keep = [r for r in terms if r["bic"] - best <= tol]
    elif rule == "rms":
        best = min(r["rms_db_pooled"] for r in terms)
        keep = [r for r in terms if r["rms_db_pooled"] <= best * (1 + tol)]
    else:
        raise ValueError(f"unknown rule {rule!r}")
    return [list(r["term"]) for r in keep]


# ------------------------------------------------------------------ log evidence

def publication_rate(ts_us):
    """Publication rate [Hz] from the median interval of increasing timestamps [us].

    The median ignores logger dropouts, which lower a count / duration rate (THESIS 7.4.3, Q4)."""
    dt = np.diff(np.asarray(ts_us, dtype=np.int64))
    dt = dt[dt > 0]
    return float(1e6 / np.median(dt))


def full_rate_logged(fc):
    """True if vehicle_angular_velocity is logged at full rate, i.e. its intervals measure the controller rate.

    PX4 v1.16.0 logged_topics.cpp: the default profile logs it every 20 ms (line 135); the high-rate profile
    (line 269) and the system-identification profile (line 355) log every sample."""
    return bool(fc.get("has_highrate") or fc.get("has_sysid"))


def hover_motor_hz(df):
    """Mean motor rotation frequency [Hz] per vehicle over its windows with valid ESC RPM (RPM/60, all motors).
    Vehicles without any valid window are left out."""
    require_fit_rows(df)
    out = {}
    for v, g in df[df["rpm_valid"].astype(bool)].groupby("vehicle_key", sort=False):
        out[v] = float(np.mean([np.mean(json.loads(x)) for x in g["rpm_hz"]]))
    return out


_MOTOR_FUNC = re.compile(r"^(.+)_FUNC\d+$")
_PWM_TIM = re.compile(r"^(PWM_[A-Z]+)_TIM\d+$")


def _timer_type(v):
    """PX4 v1.16 pwm_out module.yaml: -5/-4/-3 (-2) DShot150/300/600 (1200), -1 OneShot, > 0 PWM rate in Hz."""
    return "dshot" if v <= -2 else "oneshot" if v == -1 else "pwm" if v > 0 else "unknown"


def motor_output_protocol(params):
    """Coarse ESC protocol of the motors from the logged parameters (evidence for the T_d prior).

    v1.13+ (control allocation): motors are outputs whose *_FUNCn is 101-112 (Motor 1-12). Group PWM_MAIN /
    PWM_AUX / PWM_EXTR: the protocol is the common type of ALL that group's timers (PWM_<group>_TIMk); the
    output-to-timer map is board specific and not resolved, so timers of different types give "mixed".
    UAVCAN_EC = "can"; any other group "other". Older firmware: DSHOT_CONFIG > 0 -> dshot, PWM_MAIN_RATE 0 ->
    oneshot, else pwm (group "legacy"). Returns dict(groups, protocol); "unknown" if no motor output is found."""
    groups = sorted({_MOTOR_FUNC.match(k).group(1) for k, v in params.items()
                     if _MOTOR_FUNC.match(k) and 101 <= v <= 112})
    if not groups:
        if "DSHOT_CONFIG" in params or "PWM_MAIN_RATE" in params:
            if params.get("DSHOT_CONFIG", 0) > 0:
                proto = "dshot"
            else:
                proto = "oneshot" if params.get("PWM_MAIN_RATE", 400) == 0 else "pwm"
            return dict(groups=["legacy"], protocol=proto)
        return dict(groups=[], protocol="unknown")
    types = set()
    for g in groups:
        if g == "UAVCAN_EC":
            types.add("can")
        elif g.startswith("PWM_"):
            t = {_timer_type(v) for k, v in params.items() if _PWM_TIM.match(k) and _PWM_TIM.match(k).group(1) == g}
            types.add("unknown" if not t else t.pop() if len(t) == 1 else "mixed")
        else:
            types.add("other")
    return dict(groups=groups, protocol=types.pop() if len(types) == 1 else "mixed")


def load_fit_table(path):
    """Read a fit-half CSV through both guards."""
    import pandas as pd
    return require_fit_rows(pd.read_csv(require_fit_half(path)))


def fit_path(*parts):
    return require_fit_half(os.path.join(*parts))
