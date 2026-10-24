"""speculators' scripts/launch_vllm.py with the Apertus 1.5 config registered.

launch_vllm.py reads the target config with AutoConfig before starting vLLM;
the stock transformers in the ROCm image does not know model_type apertus1p5
(the Clariden image has the Swiss AI transformers fork). vLLM itself picks the
plugin up through its entry point.
  python3 run_launch_vllm.py SPECULATORS_DIR train MODEL ...
"""
import runpy
import sys
from pathlib import Path

import apertus15_text

apertus15_text.register_config()
script = Path(sys.argv[1]) / "scripts" / "launch_vllm.py"
sys.path.insert(0, str(script.parent))
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
