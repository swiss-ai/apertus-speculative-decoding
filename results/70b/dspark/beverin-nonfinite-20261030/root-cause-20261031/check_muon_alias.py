"""Test whether the known BF16 workspace alias affects the pinned Nesterov path."""
import ast
import hashlib
import json
import math
from pathlib import Path

import torch


root = Path(__file__).resolve().parents[1]
path = root / "diagnostic/v2/source-audit/image/torch/optim/_muon.py"
source = path.read_text()
tree = ast.parse(source)
names = {"_zeropower_via_newtonschulz", "_adjust_lr", "_single_tensor_muon"}
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(nodes) == 3
namespace = {"torch": torch, "Tensor": torch.Tensor, "math": math, "_to_scalar": lambda x: x}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
results = []
for nesterov in (True, False):
    parameter = torch.ones(4, 8, dtype=torch.bfloat16)
    gradient = torch.full_like(parameter, .05)
    buffer = torch.full_like(parameter, .2)
    expected = buffer.clone().lerp_(gradient, .05)
    namespace["_single_tensor_muon"](
        [parameter], [gradient], [buffer], lr=0.000848528137423857,
        weight_decay=.1, momentum=.95, nesterov=nesterov,
        ns_coefficients=(3.4445, -4.775, 2.0315), ns_steps=5, eps=1e-7,
        adjust_lr_fn="match_rms_adamw", has_complex=False)
    preserved = torch.equal(buffer, expected)
    assert preserved == nesterov
    assert torch.isfinite(parameter).all()
    results.append({"nesterov": nesterov, "momentum_buffer_preserved": preserved,
                    "actual_buffer_value": buffer.flatten()[0].item(),
                    "expected_buffer_value": expected.flatten()[0].item(),
                    "parameters_finite": bool(torch.isfinite(parameter).all())})
print(json.dumps({"torch": torch.__version__, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "source_functions": sorted(names), "results": results,
                  "production_configuration": "build_optimizers does not override the image Muon constructor's nesterov=True default",
                  "limitation": "Exact pinned function bodies with host CPU torch; no ROCm or DTensor kernels tested."}, indent=2))
