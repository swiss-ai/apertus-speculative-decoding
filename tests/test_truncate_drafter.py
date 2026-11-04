import importlib.util
import json
import struct
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "methods/dspark/train/truncate_drafter.py"
spec = importlib.util.spec_from_file_location("truncate_drafter", SCRIPT)
assert spec and spec.loader
truncate_drafter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(truncate_drafter)


def _write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for name, data in tensors.items():
        offsets = [offset, offset + len(data)]
        header[name] = {"dtype": "U8", "shape": [len(data)], "data_offsets": offsets}
        offset += len(data)
    blob = json.dumps(header).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(blob)) + blob + b"".join(tensors.values()))


def _read_safetensors(path: Path) -> dict[str, bytes]:
    header, start = truncate_drafter.read_header(path)
    data = path.read_bytes()
    return {
        name: data[start + meta["data_offsets"][0] : start + meta["data_offsets"][1]]
        for name, meta in header.items()
        if name != "__metadata__"
    }


def _source(tmp_path: Path, layers: int = 5) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    tensors = {"fc.weight": b"fc" * 3, "markov_head.markov_w1.weight": b"m1"}
    for i in range(layers):
        tensors[f"layers.{i}.mlp.up_proj.weight"] = bytes([i]) * (i + 2)
    tensors["norm.weight"] = b"nn"
    _write_safetensors(src / "model.safetensors", tensors)
    config = {
        "speculators_model_type": "dspark",
        "transformer_layer_config": {
            "num_hidden_layers": layers,
            "layer_types": ["sliding_attention"] * layers,
        },
    }
    (src / "config.json").write_text(json.dumps(config))
    (src / "train_command.txt").write_text("argv")
    return src


def test_keeps_the_first_layers_and_every_head(tmp_path: Path) -> None:
    src = _source(tmp_path)
    out = tmp_path / "out"
    assert truncate_drafter.main([str(src), str(out), "--layers", "2"]) == 0
    tensors = _read_safetensors(out / "model.safetensors")
    original = _read_safetensors(src / "model.safetensors")
    assert sorted(tensors) == sorted(
        ["fc.weight", "markov_head.markov_w1.weight", "norm.weight",
         "layers.0.mlp.up_proj.weight", "layers.1.mlp.up_proj.weight"]
    )
    assert all(tensors[name] == original[name] for name in tensors)
    layer_config = json.loads((out / "config.json").read_text())["transformer_layer_config"]
    assert layer_config["num_hidden_layers"] == 2
    assert layer_config["layer_types"] == ["sliding_attention"] * 2
    report = json.loads((out / "truncation.json").read_text())
    assert report["num_hidden_layers"] == 2 and report["source_num_hidden_layers"] == 5
    assert len(report["sha256"]) == 64 and report["sha256"] != report["source_sha256"]
    assert (out / "source-train_command.txt").read_text() == "argv"


def test_rejects_more_layers_than_the_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        truncate_drafter.main([str(_source(tmp_path)), str(tmp_path / "out"), "--layers", "6"])


def test_refuses_to_overwrite(tmp_path: Path) -> None:
    src = _source(tmp_path)
    out = tmp_path / "out"
    truncate_drafter.main([str(src), str(out), "--layers", "1"])
    with pytest.raises(SystemExit):
        truncate_drafter.main([str(src), str(out), "--layers", "1"])
