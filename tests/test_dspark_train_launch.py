import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("depth", ["1", "5"])
def test_fresh_draft_uses_requested_depth(depth: str) -> None:
    env = {k: v for k, v in os.environ.items() if k != "FROM_PRETRAINED"}
    env["NUM_LAYERS"] = depth
    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; printf "%s\\0" "${draft_init_args[@]}"',
            "bash",
            str(REPO / "methods/dspark/train/draft-init-args.sh"),
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    assert result.stdout.split(b"\0")[:-1] == [b"--num-layers", depth.encode()]


def test_warm_draft_does_not_override_checkpoint_depth() -> None:
    checkpoint = "/scratch/a checkpoint/depth-l1"
    env = {**os.environ, "NUM_LAYERS": "1", "FROM_PRETRAINED": checkpoint}
    result = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1"; printf "%s\\0" "${draft_init_args[@]}"',
            "bash",
            str(REPO / "methods/dspark/train/draft-init-args.sh"),
        ],
        check=True,
        capture_output=True,
        env=env,
    )
    assert result.stdout.split(b"\0")[:-1] == [b"--from-pretrained", checkpoint.encode()]
