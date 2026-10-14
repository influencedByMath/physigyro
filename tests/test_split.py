"""Vehicle keys (gate Q14) and the frozen 50/50 split (THESIS 7.4.2 C, decisions D-17, D-19, D-20)."""
import hashlib

import pandas as pd
import pytest

import split as sp

GOOD = "000600000000383832353233510d002e0040"


def _row(log_id="L1", sys_uuid=GOOD, vehicle_uuid=GOOD, fifo=True, status="ok", **params):
    p = dict(SYS_AUTOSTART=4001, IMU_GYRO_CUTOFF=40.0, IMU_DGYRO_CUTOFF=30.0, MC_ROLLRATE_P=0.15,
             MC_ROLLRATE_D=0.003, MC_PITCHRATE_P=0.15, SDLOG_PROFILE=273)
    p.update(params)
    return dict(log_id=log_id, vehicle_uuid=vehicle_uuid, sys_hw="PX4_FMU_V6X", status=status, fifo=fifo,
                info={"sys_uuid": sys_uuid, "ver_hw": "PX4_FMU_V6X"}, params=p)


def test_vehicle_key_rules():
    assert sp.vehicle_key(_row()) == "uuid:" + GOOD
    assert sp.vehicle_key(_row(sys_uuid=None)) == "uuid:" + GOOD                 # index ID when the header has none
    bad = dict(sys_uuid="REDACTED\x00\x00", vehicle_uuid="")
    k1 = sp.vehicle_key(_row(**bad))
    assert k1.startswith("fb:") and len(k1) == 3 + 16
    assert sp.vehicle_key(_row(**bad)) == k1                                     # deterministic
    assert sp.vehicle_key(_row(**bad, SDLOG_PROFILE=1)) == k1                    # not part of the key
    assert sp.vehicle_key(_row(**bad, MC_ROLLRATE_D=0.004)) != k1                # gains are part of the key
    assert sp.vehicle_key(_row(sys_uuid="000a00000000000000000000000000000000", vehicle_uuid="")).startswith("fb:")


def test_assign_split_is_deterministic_and_balanced():
    keys = [f"uuid:vehicle-{i}" for i in range(10_000)]
    first = [sp.assign_split(k) for k in keys]
    assert first == [sp.assign_split(k) for k in keys]
    share = first.count("fit") / len(keys)
    assert 0.48 <= share <= 0.52


def _odd_key():
    """A key whose salted hash is odd (would be 'test'), found without the code under test."""
    for i in range(1000):
        k = f"uuid:probe-{i}"
        if int(hashlib.sha256(f"px4-noise-split-2026-10-14|{k}".encode()).hexdigest(), 16) % 2 == 1:
            return k


def test_forced_vehicles_are_fit(monkeypatch):
    for u in ("000600000000383832353233510d002e0040", "000600000000383236313533511100370018",
              "0006000000003332333533335117004a0045", "000600000000373635363233511500260030"):
        assert sp.assign_split("uuid:" + u) == "fit"
    odd = _odd_key()
    assert sp.assign_split(odd) == "test"
    monkeypatch.setattr(sp, "FORCED_FIT_KEYS", sp.FORCED_FIT_KEYS | {odd})
    assert sp.assign_split(odd) == "fit"


def test_make_split_groups_fifo_rows_by_vehicle():
    rows = [_row("L1"), _row("L2"), _row("L3", sys_uuid="0006000000000000000000000000000000c1",
                                         vehicle_uuid="0006000000000000000000000000000000c1"),
            _row("L4", sys_uuid="0006000000000000000000000000000000d1", fifo=False),
            _row("L5", sys_uuid="0006000000000000000000000000000000e1", status="error")]
    df = sp.make_split(rows)
    assert list(df.columns) == ["vehicle_key", "split", "forced", "log_ids", "sys_hw", "n_fifo_logs"]
    assert len(df) == 2
    v = df.set_index("vehicle_key").loc["uuid:" + GOOD]
    assert v["log_ids"] == "L1;L2" and v["n_fifo_logs"] == 2 and bool(v["forced"]) is True and v["split"] == "fit"


def test_write_split_refuses_changes(tmp_path):
    path = str(tmp_path / "split.csv")
    df = pd.DataFrame(dict(vehicle_key=["uuid:a", "uuid:b"], split=["fit", "test"], forced=[False, False],
                           log_ids=["1", "2"], sys_hw=["X", "Y"], n_fifo_logs=[1, 1]))
    sp.write_split(df, path)
    flipped = df.assign(split=["test", "test"])
    with pytest.raises(ValueError):
        sp.write_split(flipped, path)
    grown = pd.concat([df, pd.DataFrame(dict(vehicle_key=["uuid:c"], split=["fit"], forced=[False], log_ids=["3"],
                                             sys_hw=["Z"], n_fifo_logs=[1]))], ignore_index=True)
    sp.write_split(grown, path)
    assert sorted(pd.read_csv(path)["vehicle_key"]) == ["uuid:a", "uuid:b", "uuid:c"]
