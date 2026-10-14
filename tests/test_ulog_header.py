"""Streaming ULog header parser (THESIS 7.4.2 B): literal ULog fixtures and one real log against pyulog."""
import os
import struct

import pytest

from ulog_header import PARAM_PREFIXES, HeaderParser

REAL_LOG = os.path.join(os.path.dirname(__file__), "..", "data", "raw_logs", "b27d09e3-4542-4496-a18b-2c3c5317024e.ulg")


def _msg(kind, payload):
    return struct.pack("<HB", len(payload), ord(kind)) + payload


def _kv(type_name, value):
    key = type_name.encode()
    return bytes([len(key)]) + key + value


def _ulog_bytes(messages):
    head = b"ULog\x01\x12\x35" + bytes([1]) + struct.pack("<Q", 123456)
    return head + b"".join(_msg(k, p) for k, p in messages)


def _info(type_name, value):
    return ("I", _kv(type_name, value))


def _param(type_name, value):
    return ("P", _kv(type_name, value))


def _default(types, type_name, value):
    return ("Q", bytes([types]) + _kv(type_name, value))


def _add(multi_id, msg_id, name):
    return ("A", struct.pack("<BH", multi_id, msg_id) + name.encode())


def _data(msg_id, n_bytes=8):
    return ("D", struct.pack("<H", msg_id) + bytes(n_bytes))


UUID = b"000600000000383832353233510d002e0040"
DEFINITIONS = [
    _info("char[36] sys_uuid", UUID),
    _info("uint32_t ver_sw_release", struct.pack("<I", 0x01100000)),
    _param("float IMU_GYRO_CUTOFF", struct.pack("<f", 80.0)),
    _param("int32_t SDLOG_PROFILE", struct.pack("<i", 273)),
    _param("float NAV_ACC_RAD", struct.pack("<f", 2.0)),
    _default(3, "float IMU_GYRO_CUTOFF", struct.pack("<f", 40.0)),
    _default(1, "int32_t SDLOG_PROFILE", struct.pack("<i", 1)),
]
FULL = _ulog_bytes(DEFINITIONS + [_add(0, 5, "sensor_gyro_fifo"), _add(1, 6, "sensor_gyro_fifo"),
                                  _data(5), _data(5)])


def _parse(data, chunk=None):
    p = HeaderParser()
    if chunk is None:
        p.feed(data)
    else:
        for i in range(0, len(data), chunk):
            p.feed(data[i:i + chunk])
    return p


def test_parses_info_params_defaults_and_topics():
    r = _parse(FULL).result()
    assert r["status"] == "ok"
    assert r["info"]["sys_uuid"] == UUID.decode()
    assert r["info"]["ver_sw_release"] == 0x01100000
    assert r["params"] == {"IMU_GYRO_CUTOFF": 80.0, "SDLOG_PROFILE": 273}
    assert r["defaults_setup"] == {"IMU_GYRO_CUTOFF": 40.0}
    assert r["defaults_system"] == {"IMU_GYRO_CUTOFF": 40.0, "SDLOG_PROFILE": 1}
    assert r["n_default_records"] == 2
    assert r["topics"] == {"sensor_gyro_fifo": [0, 1]}
    assert r["fifo"] is True
    assert r["fifo_data_msgs"] == 2
    assert r["bytes_parsed"] == len(FULL)


def test_feeding_in_tiny_chunks_gives_same_result():
    whole = _parse(FULL).result()
    assert _parse(FULL, chunk=1).result() == whole
    assert _parse(FULL, chunk=7).result() == whole


def test_not_ulog():
    p = _parse(b"<html><body>404 Not Found</body></html>")
    assert p.done() is True
    assert p.result()["status"] == "not_ulog"


def test_incomplete_header_is_unknown():
    p = _parse(_ulog_bytes(DEFINITIONS))
    r = p.result()
    assert p.done() is False
    assert r["status"] == "incomplete"
    assert r["fifo"] is False


def test_fifo_subscribed_without_data():
    msgs = DEFINITIONS + [_add(0, 1, "sensor_gyro_fifo"), _add(0, 2, "vehicle_status")] + [_data(2)] * 3000
    p = _parse(_ulog_bytes(msgs))
    r = p.result()
    assert p.done() is True
    assert r["fifo"] is True
    assert r["fifo_data_msgs"] == 0
    assert r["data_counts"] == {"vehicle_status": 3000}


@pytest.mark.skipif(not os.path.exists(REAL_LOG), reason="pilot log not downloaded")
def test_matches_pyulog_on_real_log():
    from pyulog import ULog
    with open(REAL_LOG, "rb") as fh:
        head = fh.read(3_000_000)
    p = HeaderParser()
    for i in range(0, len(head), 65536):
        if p.done():
            break
        p.feed(head[i:i + 65536])
    r = p.result()
    u = ULog(REAL_LOG, [])
    keep = lambda d: {k: v for k, v in d.items() if k.startswith(PARAM_PREFIXES)}
    assert r["status"] == "ok" and p.done()
    assert r["info"]["sys_uuid"] == u.msg_info_dict["sys_uuid"]
    assert r["info"]["ver_sw_release"] == u.msg_info_dict["ver_sw_release"]
    assert r["params"] == keep(u.initial_parameters)
    assert r["defaults_setup"] == keep(u.get_default_parameters(1))
    assert r["defaults_system"] == keep(u.get_default_parameters(0))
    assert "sensor_gyro_fifo" in r["topics"] and r["fifo"] is True
