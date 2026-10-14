"""Flight-controller data per log (THESIS 7.4.3 gates Q8, Q11, Q12, Q16; 7.4.5)."""
import os

import pytest

import fc_data as fd

REAL_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "raw_logs", "b27d09e3-4542-4496-a18b-2c3c5317024e.ulg")
needs_log = pytest.mark.skipif(not os.path.exists(REAL_LOG), reason="pilot log not downloaded")


def test_imu_chip_bit_layout():
    c = fd.imu_chip((0x26 << 16) | (0x00 << 8) | (1 << 3) | 2)       # PX4 DeviceId: bus_type:3, bus:5, address, devtype
    assert c == dict(devtype=0x26, address=0, bus=1, bus_type=2, name="ICM42688P")
    c = fd.imu_chip((0x66 << 16) | (0x69 << 8) | (3 << 3) | 1)
    assert (c["devtype"], c["address"], c["bus"], c["bus_type"], c["name"]) == (0x66, 0x69, 3, 1, "BMI088")
    assert fd.imu_chip(0xF0 << 16)["name"] == "unknown"


class _DefaultsOnly:
    def __init__(self, d0, d1):
        self._d = {0: d0, 1: d1}

    def get_default_parameters(self, t):
        return self._d[t]


def test_defaults_known_stub():
    assert fd.defaults_known(_DefaultsOnly({}, {})) is False
    assert fd.defaults_known(_DefaultsOnly({"IMU_GYRO_CUTOFF": 40.0}, {})) is True
    assert fd.defaults_known(_DefaultsOnly({}, {"SDLOG_PROFILE": 1})) is True


@pytest.fixture(scope="module")
def real_ulog():
    import noise_data as nd
    return nd.load(REAL_LOG)


@needs_log
def test_fc_info_on_real_log(real_ulog):
    i = fd.fc_info(real_ulog)
    assert i["defaults_known"] is True
    assert i["default"].f_gyro == 40.0 and i["default"].f_dterm == 30.0
    assert i["actual"].f_gyro == 80.0
    assert i["has_highrate"] is True
    assert 180.0 <= i["esc_rate_hz"] <= 220.0 and i["has_rpm_topic"] is True
    assert i["fifo_is_control_gyro"] is False
    assert i["sys_uuid"] == "000600000000383832353233510d002e0040"
    assert i["version"] == "1.16.0" and i["profile"] == 273 and i["has_sysid"] is True
    assert i["fifo_chip"]["name"] != "unknown" and i["control_chip"]["name"] != "unknown"


@needs_log
def test_defaults_unknown_without_q_records(real_ulog, monkeypatch):
    monkeypatch.setattr(real_ulog, "get_default_parameters", lambda t: {})
    i = fd.fc_info(real_ulog)
    assert i["defaults_known"] is False
    assert i["default"] is None
    assert i["actual"].f_gyro == 80.0
