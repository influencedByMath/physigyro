"""Streaming parser for the start of a PX4 ULog file (THESIS.md Section 7.4.2, step B).

The header screen reads only the first megabytes of each public log with an
HTTP range request. This parser takes the bytes chunk by chunk and keeps what
the screen needs: info messages (sys_uuid, ver_hw, ver_sw_release, ...), the
initial parameters and their defaults (only names with a prefix in
PARAM_PREFIXES), and the logged topics ('A' records). It stops as soon as
data logging is clearly under way. Standard library only; message layouts
follow the ULog file format and match pyulog 1.2.

Status: "ok" once a topic subscription was seen, "not_ulog" for anything that
is not a ULog file (e.g. an HTML error page), "incomplete" when the stream
ended before the first subscription (the FIFO question is then unanswered).
"""
import struct

ULOG_MAGIC = b"ULog\x01\x12\x35"
PARAM_PREFIXES = ("IMU_", "SDLOG_", "MC_ROLLRATE_", "MC_PITCHRATE_", "MC_YAWRATE_", "SENS_BOARD_", "CAL_GYRO",
                  "DSHOT_", "SYS_AUTOSTART", "MPC_THR_HOVER", "CA_ROTOR_COUNT", "UAVCAN_", "PWM_MAIN_", "MOT_")
DEFINITION_TYPES = set("BFIMPQ")      # messages of the definitions section
UNPACK = {"int8_t": "b", "uint8_t": "B", "int16_t": "h", "uint16_t": "H", "int32_t": "i", "uint32_t": "I",
          "int64_t": "q", "uint64_t": "Q", "float": "f", "double": "d", "bool": "?"}


def _value(type_name, raw):
    """Decode an info or parameter value from its ULog type name, as pyulog does."""
    if type_name.startswith("char["):
        return raw.decode("utf-8", errors="replace")
    fmt = UNPACK.get(type_name)
    if fmt is not None and struct.calcsize("<" + fmt) == len(raw):
        return struct.unpack("<" + fmt, raw)[0]
    return raw.hex()


def _key_value(p):
    """'uint8 key_len, key "type name", value' -> (type, name, decoded value)."""
    n = p[0]
    type_name, name = p[1:1 + n].decode("utf-8", errors="replace").split(" ", 1)
    return type_name, name, _value(type_name, p[1 + n:])


class HeaderParser:
    def __init__(self, param_prefixes=PARAM_PREFIXES, stop_after_data=3000):
        self.prefixes = tuple(param_prefixes)
        self.stop_after_data = stop_after_data
        self._buf = b""
        self._is_ulog = None
        self._in_definitions = True
        self.info, self.params, self.defaults = {}, {}, ({}, {})   # defaults: (system, current setup)
        self.n_default_records = 0
        self._names = {}                      # msg_id -> topic name
        self.topics = {}                      # topic -> set of multi_ids
        self.data_counts = {}
        self.n_data = self.n_dropouts = 0
        self.bytes_parsed = 0

    def feed(self, chunk):
        if self.done():
            return
        self._buf += chunk
        if self._is_ulog is None:
            if len(self._buf) < 16:
                if not ULOG_MAGIC.startswith(self._buf[:7]):
                    self._is_ulog = False
                return
            self._is_ulog = self._buf[:7] == ULOG_MAGIC
            if not self._is_ulog:
                return
            self._buf = self._buf[16:]
            self.bytes_parsed = 16
        pos, b = 0, self._buf
        while pos + 3 <= len(b) and not self.done():
            size, kind = struct.unpack_from("<HB", b, pos)
            if pos + 3 + size > len(b):
                break
            self._handle(chr(kind), b[pos + 3:pos + 3 + size])
            pos += 3 + size
        self.bytes_parsed += pos
        self._buf = b[pos:]

    def _handle(self, kind, p):
        if kind not in DEFINITION_TYPES:
            self._in_definitions = False
        if kind == "I":
            _, name, value = _key_value(p)
            self.info[name] = value
        elif kind == "P" and self._in_definitions:          # later 'P' records are in-flight changes
            _, name, value = _key_value(p)
            if name.startswith(self.prefixes):
                self.params[name] = value
        elif kind == "Q":
            self.n_default_records += 1
            _, name, value = _key_value(p[1:])
            if name.startswith(self.prefixes):
                for bit in (0, 1):                          # bit 0 system default, bit 1 current setup
                    if p[0] & (1 << bit):
                        self.defaults[bit][name] = value
        elif kind == "A":
            multi_id, msg_id = struct.unpack_from("<BH", p, 0)
            name = p[3:].decode("utf-8", errors="replace")
            self._names[msg_id] = name
            self.topics.setdefault(name, set()).add(multi_id)
        elif kind == "D":
            self.n_data += 1
            name = self._names.get(struct.unpack_from("<H", p, 0)[0], "?")
            self.data_counts[name] = self.data_counts.get(name, 0) + 1
        elif kind == "O":
            self.n_dropouts += 1

    def done(self):
        return self._is_ulog is False or (bool(self.topics) and self.n_data >= self.stop_after_data)

    def result(self):
        if self._is_ulog is False:
            status = "not_ulog"
        else:
            status = "ok" if self.topics else "incomplete"
        return dict(status=status, info=dict(self.info), params=dict(self.params),
                    defaults_system=dict(self.defaults[0]), defaults_setup=dict(self.defaults[1]),
                    n_default_records=self.n_default_records,
                    topics={k: sorted(v) for k, v in sorted(self.topics.items())},
                    data_counts=dict(self.data_counts), n_data=self.n_data,
                    fifo="sensor_gyro_fifo" in self.topics,
                    fifo_data_msgs=self.data_counts.get("sensor_gyro_fifo", 0),
                    n_dropouts=self.n_dropouts, bytes_parsed=self.bytes_parsed)
