"""CPU probes of the exact pinned loss source; no model or training job.

Greedy metric helpers are small fixtures; forward loss/confidence code is loaded
unchanged from the captured 7046ac6 files. This is not a ROCm compiler test.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "pinned-source"


def package(name):
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module
    return module


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


package("speculators")
losses = package("speculators.losses")
losses.eager = load("speculators.losses.eager", SOURCES / "losses/eager.py")
utils = load("speculators.losses.utils", SOURCES / "losses/utils.py")
for name in ("LossConfig", "compound_loss", "dflash_loss_decay", "dpace_loss_decay", "tv_loss"):
    setattr(losses, name, getattr(utils, name))
package("speculators.models")
helpers = package("speculators.models.metrics")


def accuracy(pred, target, mask, pos, block):
    correct, total = [], []
    for p in range(block):
        valid = mask.bool() & (pos == p)
        correct.append(((pred == target) & valid).sum().float())
        total.append(valid.sum().float())
    return torch.stack(correct), torch.stack(total)


def length(correct, valid):
    accepted = (correct & valid).float().cumprod(-1).sum(-1) + 1
    return (accepted * valid.any(-1)).sum(), valid.any(-1).sum().float()


helpers.compute_accuracy_multi_step = accuracy
helpers.compute_accepted_length_counts = length
metrics = load("pinned_dspark_metrics", SOURCES / "models/dspark/metrics.py")


def case(dtype, scale, mask_kind, confidence_scale=1., poison=None):
    torch.manual_seed(42)
    logits = (torch.randn(1, 32, 129) * scale).to(dtype)
    targets = (torch.randn_like(logits.float()) * scale).to(dtype)
    confidence = (torch.randn(1, 32) * confidence_scale).to(dtype)
    mask = torch.ones(1, 32, dtype=torch.bool)
    if mask_kind == "empty":
        mask.zero_()
    if mask_kind == "partial":
        mask[:, ::3] = False
    if poison == "target-only":
        targets[0, 0, :] = float("nan")
    elif poison == "masked-shared-output":
        mask[0, 0] = False
        logits[0, 0, :] = float("nan")
        confidence[0, 0] = float("nan")
    logits.requires_grad_()
    confidence.requires_grad_()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        loss, result = metrics.compute_metrics(
            logits, targets, confidence, mask, 8,
            loss_config={"ce": (losses.eager.ce_loss, .1),
                         "tv": (losses.eager.tv_loss, .9)},
            tv_loss_fn=losses.eager.tv_loss)
    loss.backward()
    finite = lambda value: bool(torch.isfinite(value).all())
    record = {"dtype": str(dtype), "logit_scale": scale, "mask": mask_kind,
              "confidence_scale": confidence_scale, "poison": poison,
              "loss_finite": finite(loss), "ce_finite": finite(result["ce_loss_sum"]),
              "tv_finite": finite(result["tv_loss_sum"]),
              "confidence_prediction_finite": finite(result["confidence_pred_mean_sum"]),
              "logit_gradient_finite": finite(logits.grad),
              "confidence_gradient_finite": finite(confidence.grad)}
    if poison is None:
        assert all(record[k] for k in ("loss_finite", "ce_finite", "tv_finite",
                   "confidence_prediction_finite", "logit_gradient_finite",
                   "confidence_gradient_finite")), record
        if mask_kind == "empty":
            assert loss.item() == 0 and torch.count_nonzero(logits.grad) == 0
    elif poison == "target-only":
        assert record["ce_finite"] and record["confidence_prediction_finite"]
        assert not record["tv_finite"] and not record["loss_finite"]
    elif poison == "masked-shared-output":
        assert not record["ce_finite"] and not record["confidence_prediction_finite"]
    return record


records = []
for dtype in (torch.float32, torch.bfloat16):
    for scale in (1., 20., 1000.):
        for mask in ("all", "partial", "empty"):
            records.append(case(dtype, scale, mask, confidence_scale=1000.))
    for poison in ("target-only", "masked-shared-output"):
        records.append(case(dtype, 1., "all", poison=poison))

output = {"torch": torch.__version__, "device": "CPU", "cases": records,
          "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [SOURCES / "losses/eager.py", SOURCES / "losses/utils.py",
                                      SOURCES / "models/dspark/metrics.py"]},
          "limitations": "Host torch CPU; greedy-count helpers are fixtures. Does not test the ROCm compiler, GPU kernels or actual missing failing batch."}
print(json.dumps(output, indent=2))
