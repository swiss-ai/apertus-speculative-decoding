"""CPU checks for verified checkpoint rotation and read-back."""
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "methods/dspark/beverin"))
from checkpoint_layout import list_verified, promote, read_marker
from checkpoint_ring import assign_by_size, verify_directory, verify_optimizer_file, verify_safetensors


def write_safetensors(path, tensors):
    header = {}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        blob = bytes(tensor.contiguous().view(torch.uint8).numpy())
        header[name] = {"dtype": "F32", "shape": list(tensor.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    payload = json.dumps(header).encode()
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(payload)))
        handle.write(payload)
        for blob in blobs:
            handle.write(blob)


def marker(step):
    return {"finite": True, "global_step": step, "epoch": 0, "local_step": step, "learning_rates": [0.001]}


def staging(root, step, weights=b"ok"):
    folder = root / ("staging-%d" % step)
    folder.mkdir()
    (folder / "model.safetensors").write_bytes(weights)
    (folder / "optimizer_state_dict.pt").write_bytes(b"opt")
    (folder / "verified-finite.json").write_text(json.dumps(marker(step)))
    return folder


class RingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_rename_keeps_two_and_a_512_milestone_without_optimizer(self):
        promote(str(self.root), str(staging(self.root, 512)), keep=2, milestone=512)
        promote(str(self.root), str(staging(self.root, 576)), keep=2, milestone=512)
        promote(str(self.root), str(staging(self.root, 640)), keep=2, milestone=512)
        steps = [item[0] for item in list_verified(self.root)]
        self.assertEqual(steps, [576, 640])
        self.assertEqual(read_marker(self.root / "0")["global_step"], 640)
        milestone = self.root / "milestones/step-512"
        self.assertTrue((milestone / "model.safetensors").is_file())
        self.assertFalse((milestone / "optimizer_state_dict.pt").exists())
        self.assertFalse((self.root / "verified/step-512").exists())

    def test_unverified_staging_does_not_replace_zero(self):
        promote(str(self.root), str(staging(self.root, 10)))
        before = (self.root / "0/model.safetensors").read_bytes()
        bad = self.root / "bad"
        bad.mkdir()
        (bad / "model.safetensors").write_bytes(b"corrupt")
        with self.assertRaises(RuntimeError):
            promote(str(self.root), str(bad))
        self.assertEqual((self.root / "0/model.safetensors").read_bytes(), before)

    def test_read_back_rejects_nonfinite_and_splits_work(self):
        path = self.root / "model.safetensors"
        write_safetensors(path, {"w": torch.tensor([1.0, float("nan")])})
        self.assertFalse(verify_safetensors(str(path), torch, 0, 1))
        write_safetensors(path, {"a": torch.ones(4), "b": torch.zeros(8)})
        self.assertTrue(verify_safetensors(str(path), torch, 0, 2))
        self.assertTrue(verify_safetensors(str(path), torch, 1, 2))
        assigned = assign_by_size([("a", 4), ("b", 100), ("c", 4)], 2, 0)
        self.assertIn("b", assigned)

    def test_optimizer_read_back_checks_float_step_and_momentum(self):
        parameter = torch.nn.Parameter(torch.randn(3))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
        path = self.root / "optimizer_state_dict.pt"
        torch.save(optimizer.state_dict(), path)
        self.assertTrue(verify_optimizer_file(str(path), torch, 0, 1))
        state = torch.load(path, weights_only=True)
        state["state"][0]["exp_avg"][0] = float("nan")
        torch.save(state, path)
        self.assertFalse(verify_optimizer_file(str(path), torch, 0, 1))
        finite = {"model.safetensors": torch.ones(2)}
        write_safetensors(self.root / "model.safetensors", finite)
        folder = self.root / "ckpt"
        folder.mkdir()
        write_safetensors(folder / "model.safetensors", {"w": torch.ones(2)})
        torch.save(optimizer.state_dict(), folder / "optimizer_state_dict.pt")
        self.assertTrue(verify_directory(folder, torch, 0, 1))


if __name__ == "__main__":
    unittest.main()
