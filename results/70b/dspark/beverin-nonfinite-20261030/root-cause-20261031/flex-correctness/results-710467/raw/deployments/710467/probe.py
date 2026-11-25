"""Attention-kernel oracle; no model, optimizer, corpus or training loop.

Use --cpu-oracle for tiny dense-reference checks. GPU mode uses the exact
captured DFlash mask and production Q/KV lengths, GQA heads and head dimension.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask, flex_attention


def mask_module(path):
    spec = importlib.util.spec_from_file_location("captured_mask", path)
    module = importlib.util.module_from_spec(spec)
    # Dynamo imports a callable's defining module when constructing guards.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fixture(context, anchors_count, block, padded, device):
    docs = torch.arange(context, device=device) // (context // 2)
    count = anchors_count // 2 if padded else anchors_count
    positions = torch.linspace(0, context - block - 1, count, device=device).long()
    # Exact select_anchors convention: valid sorted entries, then zero padding.
    anchors = torch.cat([positions, torch.zeros(anchors_count - count, device=device,
                                               dtype=torch.long)])
    starts = torch.maximum(docs[anchors] * (context // 2), anchors - context // 4)
    return docs, anchors, starts


def expected_v_gradient(docs, anchors, starts, block, groups):
    # Production is sliding_window_non_causal=False: query slot s sees s+1
    # synthetic tokens, in addition to the window-limited document prefix.
    denominators = ((anchors - starts)[:, None]
                    + torch.arange(1, block + 1, device=anchors.device)[None, :]).float()
    per_slot = groups / denominators
    weights = per_slot.sum(-1)
    indices = torch.arange(docs.numel(), device=docs.device)
    valid = ((docs[:, None] == docs[anchors][None, :]) &
             (indices[:, None] < anchors[None, :]) &
             (indices[:, None] >= starts[None, :]))
    base = (valid.float() * weights[None, :]).sum(-1)
    synthetic = per_slot.flip(-1).cumsum(-1).flip(-1).flatten()
    return torch.cat([base, synthetic])


def verify_cpu(module):
    records = []
    for padded in (False, True):
        context, count, block, groups = 64, 8, 8, 4
        docs, anchors, starts = fixture(context, count, block, padded, "cpu")
        mod, q_len, kv_len = module.create_anchor_block_mask_mod(
            docs, context, anchors, block, sliding_window=context // 4,
            sliding_window_non_causal=False)
        mask = create_mask(mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device="cpu")[0, 0]
        assert mask.any(-1).all(), "fixture has a wholly masked query"
        reference = (mask.double() / mask.sum(-1, keepdim=True)).sum(0) * groups
        expected = expected_v_gradient(docs, anchors, starts, block, groups)
        torch.testing.assert_close(expected.double(), reference, rtol=1e-6, atol=1e-6)
        # Exercise Dynamo's module imports and the compiled mask builder too;
        # an eager dense-mask oracle alone missed this in GPU job710459.
        compiled_builder = torch.compile(create_block_mask)
        sparse = compiled_builder(mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device="cpu")
        assert sparse.shape[-2] >= q_len and sparse.shape[-1] >= kv_len
        records.append({"padded": padded, "q_len": q_len, "kv_len": kv_len,
                        "zero_attention_columns": int((~mask.any(0)).sum()),
                        "oracle_max_error": float((expected - reference).abs().max()),
                        "compiled_mask_builder_passed": True})
    return records


def run_gpu(module, path):
    torch.cuda.set_device(0)
    torch.backends.cuda.preferred_blas_library("cublas")
    context, count, block, q_heads, kv_heads, dim = 8192, 512, 8, 64, 8, 128
    compiled = torch.compile(flex_attention)
    compiled_mask = torch.compile(create_block_mask)
    result = {"torch": torch.__version__, "torch_git": torch.version.git_version,
              "hip": torch.version.hip, "python": sys.version,
              "device": str(torch.cuda.get_device_properties(0)),
              "argv": sys.argv, "mask_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "configuration": {"context": context, "anchors": count, "block": block,
                                "sliding_window": 2048, "sliding_window_non_causal": False,
                                "compiled_mask_builder": True,
                                "query_heads": q_heads, "kv_heads": kv_heads, "head_dim": dim},
              "cases": [], "status": "running"}
    output = Path(args.output)

    def publish():
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(output)

    publish()
    for padded in (False, True):
        docs, anchors, starts = fixture(context, count, block, padded, "cuda")
        mod, q_len, kv_len = module.create_anchor_block_mask_mod(
            docs, context, anchors, block, sliding_window=context // 4,
            sliding_window_non_causal=False)
        mask = compiled_mask(mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device="cuda")
        for fixture_name in ("uniform", "random-zero-v", "saturated-constant-v"):
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)
            shape_q, shape_kv = (1, q_heads, q_len, dim), (1, kv_heads, kv_len, dim)
            if fixture_name == "uniform":
                q = torch.zeros(shape_q, device="cuda", dtype=torch.bfloat16)
                k = torch.zeros(shape_kv, device="cuda", dtype=torch.bfloat16)
            else:
                scale = 10. if fixture_name.startswith("saturated") else 1.
                q = torch.randn(shape_q, device="cuda", dtype=torch.bfloat16) * scale
                k = torch.randn(shape_kv, device="cuda", dtype=torch.bfloat16) * scale
            v = torch.full(shape_kv, 0. if fixture_name == "random-zero-v" else 1.,
                           device="cuda", dtype=torch.bfloat16)
            q.requires_grad_(); k.requires_grad_(); v.requires_grad_()
            torch.cuda.synchronize()
            started = time.monotonic()
            out = compiled(q, k, v, block_mask=mask, score_mod=None,
                           enable_gqa=True, scale=dim**-.5)
            dq, dk, dv = torch.autograd.grad(out, (q, k, v), torch.ones_like(out))
            torch.cuda.synchronize()
            record = {"padded_anchors": padded, "fixture": fixture_name,
                      "seconds": time.monotonic() - started,
                      "nonfinite": {name: int((~torch.isfinite(value)).sum())
                                    for name, value in (("output", out), ("dq", dq), ("dk", dk), ("dv", dv))}}
            result["cases"].append(record)
            try:
                assert all(value == 0 for value in record["nonfinite"].values()), record
                expected_out = 0. if fixture_name == "random-zero-v" else 1.
                record["max_output_error"] = float((out.float() - expected_out).abs().max())
                assert record["max_output_error"] <= .02, record
                if fixture_name in ("uniform", "random-zero-v"):
                    assert torch.count_nonzero(dq) == 0 and torch.count_nonzero(dk) == 0, record
                # Conservation: sum of dV over keys equals all dOutput heads
                # mapped to this KV head, for each component.
                mass = dv.float().sum(2)
                record["dv_mass_max_relative_error"] = float(
                    (mass / (q_len * (q_heads // kv_heads)) - 1).abs().max())
                assert record["dv_mass_max_relative_error"] <= .02, record
                if fixture_name == "uniform":
                    expected = expected_v_gradient(docs, anchors, starts, block, q_heads // kv_heads)
                    torch.testing.assert_close(dv.float(), expected[None, None, :, None].expand_as(dv),
                                               rtol=.025, atol=.01)
                    assert torch.count_nonzero(dv[:, :, expected == 0, :]) == 0
            except Exception as exc:
                record["error"] = str(exc)
                result["status"] = "failed"
                publish()
                raise
            publish()
            print("FLEX_ORACLE_PASS " + json.dumps(record), flush=True)
            del q, k, v, out, dq, dk, dv
    # Independent small FP32 math oracle exercises nonzero dQ/dK, which the
    # production-size zero/constant-V invariants above do not measure.
    from torch.nn.attention import sdpa_kernel, SDPBackend
    for padded in (False, True):
        torch.manual_seed(43)
        docs, anchors, starts = fixture(256, 32, 8, padded, "cuda")
        mod, q_len, kv_len = module.create_anchor_block_mask_mod(
            docs, 256, anchors, 8, sliding_window=64, sliding_window_non_causal=False)
        sparse = compiled_mask(mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device="cuda")
        dense = create_mask(mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len, device="cuda")
        q = torch.randn(1, 8, q_len, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 2, kv_len, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        reference_inputs = [x.detach().float().requires_grad_() for x in (q, k, v)]
        torch.backends.cuda.matmul.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            reference = torch.nn.functional.scaled_dot_product_attention(
                *reference_inputs, attn_mask=dense, enable_gqa=True, scale=128**-.5)
        reference_gradients = torch.autograd.grad(reference, reference_inputs, torch.ones_like(reference))
        actual = compiled(q, k, v, block_mask=sparse, score_mod=None, enable_gqa=True, scale=128**-.5)
        gradients = torch.autograd.grad(actual, (q, k, v), torch.ones_like(actual))
        record = {"padded_anchors": padded, "fixture": "small-fp32-math-reference",
                  "query_heads": 8, "kv_heads": 2, "q_len": q_len, "kv_len": kv_len,
                  "relative_l2_error": {}, "nonfinite": {}, "reference_nonfinite": {}}
        result["cases"].append(record)
        try:
            for name, value, expected in zip(("output", "dq", "dk", "dv"),
                                             (actual, *gradients), (reference, *reference_gradients)):
                record["nonfinite"][name] = int((~torch.isfinite(value)).sum())
                record["reference_nonfinite"][name] = int((~torch.isfinite(expected)).sum())
                assert record["nonfinite"][name] == 0, record
                assert record["reference_nonfinite"][name] == 0, record
                error = torch.linalg.vector_norm(value.float() - expected) / torch.linalg.vector_norm(expected).clamp_min(1e-12)
                record["relative_l2_error"][name] = float(error)
                assert error <= .03, record
        except Exception as exc:
            record["error"] = str(exc)
            result["status"] = "failed"
            publish()
            raise
        publish()
        print("FLEX_ORACLE_PASS " + json.dumps(record), flush=True)
    result["status"] = "passed"
    publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-source", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cpu-oracle", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    module = mask_module(args.mask_source)
    if args.cpu_oracle:
        Path(args.output).write_text(json.dumps({"torch": torch.__version__,
            "cases": verify_cpu(module), "status": "passed"}, indent=2) + "\n")
    else:
        run_gpu(module, args.mask_source)
