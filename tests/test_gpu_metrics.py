import gzip
import json
import sqlite3

from apertus_bench import gpu_metrics


def test_bus_key_matches_nvidia_smi_and_nsys_spellings():
    assert gpu_metrics.bus_key("00000009:01:00.0") == gpu_metrics.bus_key("0009:01:00.0")
    assert gpu_metrics.bus_key("01:00.0") == (0, 1, 0)


def test_arm_roles_split_pd_gpus_in_server_order():
    status = {"gpus": [2, 3], "servers": [{"role": "prefill"}, {"role": "decode"}]}
    assert gpu_metrics.arm_roles(status) == {"prefill": [2], "decode": [3]}
    assert gpu_metrics.arm_roles({"gpus": [1], "servers": []}) == {"serve": [1]}


def _report(path, start_ns):
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE TARGET_INFO_SESSION_START_TIME (utcEpochNs INTEGER)")
    db.execute("INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?)", (start_ns,))
    db.execute(
        "CREATE TABLE TARGET_INFO_GPU_METRICS "
        "(typeId INTEGER, sourceId INTEGER, typeName TEXT, metricId INTEGER, metricName TEXT)"
    )
    db.execute("CREATE TABLE TARGET_INFO_GPU (id INTEGER, busLocation TEXT)")
    db.execute("CREATE TABLE GPU_METRICS (typeId INTEGER, timestamp INTEGER, value INTEGER, "
               "metricId INTEGER)")
    for nsys_id, bus in ((0, "0009:01:00.0"), (1, "0019:01:00.0")):
        db.execute("INSERT INTO TARGET_INFO_GPU VALUES (?, ?)", (nsys_id, bus))
        db.execute("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, 0, 'GH100', 7, 'Tensor Active')",
                   (nsys_id,))
    # nsys GPU 1 is physical GPU 0 here; two samples in second 100, one in 101.
    for timestamp, value in ((0, 40), (500_000_000, 60), (1_200_000_000, 10)):
        db.execute("INSERT INTO GPU_METRICS VALUES (1, ?, ?, 7)", (timestamp, value))
    db.commit()
    db.close()


def test_cells_average_steady_seconds_of_the_decode_gpu(tmp_path):
    sweep = tmp_path / "sweep"
    metrics = sweep / "gpu-metrics"
    metrics.mkdir(parents=True)
    (metrics / "gpus.csv").write_text(
        "index, pci.bus_id, uuid, name\n"
        "0, 00000019:01:00.0, a, GH200\n1, 00000009:01:00.0, b, GH200\n"
    )
    _report(metrics / "chunk-001.sqlite", 100 * 10**9)
    (metrics / "chunk-001.nsys-rep").write_text("")
    arm = sweep / "pd"
    cell = arm / "opb" / "opb" / "c8"
    cell.mkdir(parents=True)
    (arm / "status.json").write_text(json.dumps(
        {"arm": "pd", "gpus": [1, 0], "servers": [{"role": "prefill"}, {"role": "decode"}]}
    ))
    (cell / "metadata.json").write_text(json.dumps(
        {"started_at": "1970-01-01T00:01:39+00:00", "ended_at": "1970-01-01T00:01:43+00:00"}
    ))
    (cell / "metrics_timeseries.jsonl").write_text(
        "\n".join(json.dumps({"t": t, "running": r}) for t, r in ((0, 2), (1, 8), (2, 8), (3, 1)))
    )
    gpu_metrics.export(sweep, nsys="/nonexistent")
    with gzip.open(metrics / "per-second.csv.gz", "rt") as handle:
        assert "100,0,Tensor Active,50.0,2" in handle.read()
    [record] = json.loads((metrics / "cells.json").read_text())
    decode = record["gpus"]["decode:0"]
    assert decode["steady"]["metrics"] == {"Tensor Active": 30.0}
    assert decode["window"]["seconds"] == 2
    assert record["gpus"]["prefill:1"]["window"]["seconds"] == 0
