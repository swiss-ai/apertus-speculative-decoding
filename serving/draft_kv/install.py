"""Build a reviewable source overlay against the pinned vLLM tree.

Existing Apertus fixes must be applied before this overlay. No installed package
is modified; the output files are mounted over the serving container sources.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
from pathlib import Path

TRANSFORMS = {
    "vllm/model_executor/models/qwen3_dflash.py": [
        (
            "        attn_output = self.attn(q, k, v)\n",
            "        from vllm.apertus_draft_kv.integration import sparse_layer_attention\n"
            "        attn_output = sparse_layer_attention(self, q, k, v, positions)\n",
        ),
        (
            None,
            "\n\nfrom vllm.apertus_draft_kv.integration import install_model_hooks\n"
            "install_model_hooks(globals())\n",
        ),
    ],
    "vllm/v1/worker/gpu/spec_decode/dspark/utils.py": [
        (
            '    with set_model_tag("dspark_head"):\n',
            "    from vllm.apertus_draft_kv.integration import sparse_draft_config\n"
            "    draft_vllm_config = sparse_draft_config(draft_vllm_config)\n\n"
            '    with set_model_tag("dspark_head"):\n',
        ),
    ],
    "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py": [
        (
            None,
            "\n\nfrom vllm.apertus_draft_kv.integration import install_speculator_hooks\n"
            "install_speculator_hooks(globals())\n",
        ),
    ],
    "vllm/v1/worker/gpu/model_runner.py": [
        (
            "    def _remove_request(self, req_id: str) -> bool:\n",
            "    def _remove_request(self, req_id: str) -> bool:\n"
            "        from vllm.apertus_draft_kv.integration import notify_request_removed\n"
            "        notify_request_removed(self.speculator, req_id)\n",
        ),
        (
            "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n",
            "            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens\n"
            "            from vllm.apertus_draft_kv.integration import attach_release_events\n"
            "            attach_release_events(self.speculator, model_runner_output)\n",
        ),
    ],
    "vllm/v1/core/sched/scheduler.py": [
        (
            "        sampled_token_ids = model_runner_output.sampled_token_ids\n",
            "        from vllm.apertus_draft_kv.integration import apply_release_events\n"
            "        apply_release_events(self.kv_cache_manager, model_runner_output)\n"
            "        sampled_token_ids = model_runner_output.sampled_token_ids\n",
        ),
    ],
    "vllm/v1/outputs.py": [
        ("class ModelRunnerOutput:\n", "class ModelRunnerOutput:\n"),
        (
            "    # [num_reqs, max_num_logprobs + 1]\n",
            "    # Drafter-only physical page ownership after bootstrap compression.\n"
            "    apertus_draft_kv_releases: dict[str, list[int]] | None = None\n"
            "    apertus_draft_kv_trace: dict[str, object] | None = None\n\n"
            "    # [num_reqs, max_num_logprobs + 1]\n",
        ),
    ],
}


def transform(path: str, source: str) -> str:
    if "apertus_draft_kv" in source:
        raise ValueError(f"{path}: sparse overlay already applied")
    for before, after in TRANSFORMS[path]:
        if before is None:
            source += after
        else:
            count = source.count(before)
            # outputs.py has a duplicated upstream comment; only insert once.
            if path.endswith("outputs.py") and before.startswith(
                "    # [num_reqs, max_num_logprobs"
            ):
                if not count:
                    raise ValueError(f"{path}: expected output comment missing")
                source = source.replace(before, after, 1)
            else:
                if count != 1:
                    raise ValueError(f"{path}: expected one source anchor, found{count}")
                source = source.replace(before, after, 1)
    compile(source, path, "exec")
    return source


def build(source_root: Path, output_root: Path, patch_output: Path | None = None):
    manifest = {"source_root": str(source_root), "files": {}, "helper_files": {}}
    patch = []
    for path in TRANSFORMS:
        original = (source_root / path).read_text()
        changed = transform(path, original)
        destination = output_root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(changed)
        manifest["files"][path] = {
            "before_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "after_sha256": hashlib.sha256(changed.encode()).hexdigest(),
        }
        patch.extend(
            difflib.unified_diff(
                original.splitlines(True),
                changed.splitlines(True),
                fromfile="a/workspace/vllm/" + path,
                tofile="b/workspace/vllm/" + path,
            )
        )
    helpers = output_root / "vllm/apertus_draft_kv"
    helpers.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "integration.py", "attention.py", "policy.py", "graphs.py",
                 "tracing.py"):
        source = Path(__file__).parent / name
        shutil.copy2(source, helpers / name)
        manifest["helper_files"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    if patch_output:
        patch_output.parent.mkdir(parents=True, exist_ok=True)
        patch_output.write_text("".join(patch))
    (output_root / "draft-kv-overlay-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    mounts = [f"{output_root / path}:/workspace/vllm/{path}" for path in TRANSFORMS]
    mounts.append(f"{helpers}:/workspace/vllm/vllm/apertus_draft_kv")
    return mounts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--patch-output", type=Path)
    args = parser.parse_args()
    for mount in build(args.source_root, args.output_root, args.patch_output):
        print(mount)


if __name__ == "__main__":
    main()
