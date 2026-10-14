"""Flight-controller data of one log (THESIS.md Section 7.4.5; gates Q8, Q11, Q12, Q16).

``fc_info`` collects what the pipeline needs besides the raw gyro: the actual
and default filter settings (gate Q11, with the "defaults unknown" rule), the
roll and pitch PID gains, the loop rate, the IMU chip of the FIFO data and of
the controller's gyro (gates Q8, Q12), and which extra topics are logged and
how fast (ESC RPM for Scenario B and decision D-13; full-rate filtered gyro and
torque for E10/E11; profile bits for Tier 2).
"""
import numpy as np

from noise_data import _t, fc_settings, fifo_dataset

# PX4 v1.16.0 src/drivers/drv_sensor.h: devtype of every gyro-capable sensor (IMU and GYR entries)
GYRO_DEVTYPES = {
    0x11: "LSM303D", 0x14: "IMU_SIM", 0x21: "MPU6000", 0x22: "L3GD20", 0x24: "MPU9250", 0x25: "ICM20649",
    0x26: "ICM42688P", 0x27: "ICM40609D", 0x28: "ICM20948", 0x29: "ICM42605", 0x2A: "ICM42670P", 0x2B: "IIM42652",
    0x2C: "IAM20680HP", 0x2D: "ICM42686P", 0x2E: "IIM42653", 0x34: "ICM45686", 0x35: "MPU6050", 0x36: "MPU6500",
    0x37: "BMI270", 0x38: "ICM20602", 0x3A: "ICM20608G", 0x3C: "ICM20689", 0x42: "BMI055", 0x44: "LSM9DS1",
    0x54: "FXAS2100C", 0x57: "ADIS16448", 0x58: "ADIS16470", 0x59: "ADIS16477", 0x5A: "ADIS16507", 0x5B: "SCH16T",
    0x63: "ADIS16497", 0x66: "BMI088", 0x6D: "BMI085", 0x86: "GYR_UAVCAN", 0x87: "IMU_UAVCAN"}


def imu_chip(device_id):
    """Decode a PX4 DeviceId (bus_type:3, bus:5, address:8, devtype:8 bits from the low end)."""
    d = int(device_id) % 2**32
    devtype = (d >> 16) & 0xFF
    return dict(devtype=devtype, address=(d >> 8) & 0xFF, bus=(d >> 3) & 0x1F, bus_type=d & 0x7,
                name=GYRO_DEVTYPES.get(devtype, "unknown"))


def defaults_known(ulog):
    """False when the log has no default-parameter record at all (older firmware): then a missing
    default does NOT mean "default = actual" (PX4 v1.16.0 logger.cpp, write_parameter_defaults)."""
    return bool(ulog.get_default_parameters(0)) or bool(ulog.get_default_parameters(1))


def logged_rate_hz(ulog, topic):
    """Median logging rate of a topic (first instance), or None if it is not logged."""
    try:
        t = _t(ulog, ulog.get_dataset(topic).data["timestamp"])
    except Exception:
        return None
    dt = np.diff(t[t > 0])
    dt = dt[dt > 0]
    return float(1.0 / np.median(dt)) if len(dt) else None


def version_string(v):
    """ver_sw_release (0xMMmmpptt) -> '1.16.0', with '-dev'/'-alpha'/'-beta'/'-rc' for non-release types."""
    if v is None:
        return None
    v = int(v)
    tag = {255: "", 192: "-rc", 128: "-beta", 64: "-alpha", 0: "-dev"}.get(v & 255, f"-type{v & 255}")
    return f"{(v >> 24) & 255}.{(v >> 16) & 255}.{(v >> 8) & 255}{tag}"


def fc_info(ulog):
    p = ulog.initial_parameters
    roll, pitch = fc_settings(ulog, "ROLL"), fc_settings(ulog, "PITCH")
    known = defaults_known(ulog)
    fifo_id = int(fifo_dataset(ulog)["device_id"][0])
    try:
        sel = [int(x) for x in ulog.get_dataset("sensor_selection").data["gyro_device_id"]]
    except Exception:
        sel = []
    profile = int(p.get("SDLOG_PROFILE", 1))
    ratemax = float(p.get("IMU_GYRO_RATEMAX", 400))
    rates = {t: logged_rate_hz(ulog, t) for t in ("esc_status", "vehicle_angular_velocity", "vehicle_torque_setpoint")}
    fast = lambda t: rates[t] is not None and rates[t] >= 0.8 * ratemax
    return dict(actual=roll["actual"], default=roll["default"] if known else None, defaults_known=known,
                gains_roll=roll["gains"], gains_pitch=pitch["gains"], f_ctrl=roll["f_ctrl"],
                version=version_string(roll["version"]), hw=roll["hw"],
                sys_uuid=ulog.msg_info_dict.get("sys_uuid"), profile=profile,
                fifo_chip=imu_chip(fifo_id), control_chip=imu_chip(sel[-1]) if sel else None,
                fifo_is_control_gyro=bool(sel) and all(s == fifo_id for s in sel),
                fft_notch=roll["fft_notch"], changed_in_flight=roll["changed_in_flight"],
                esc_rate_hz=rates["esc_status"], has_rpm_topic=rates["esc_status"] is not None,
                has_highrate=fast("vehicle_angular_velocity") and fast("vehicle_torque_setpoint"),
                has_sysid=bool(profile & (1 << 3) or profile & (1 << 4)))
