"""Run a Python script or module with the Apertus 1.5 config registered.

speculators reads the target config with AutoConfig (launch_vllm.py before it
starts vLLM, and the trainer for the verifier's sizes); the stock transformers
in the ROCm image does not know model_type apertus1p5 (the Clariden image
carries the Swiss AI transformers fork). vLLM itself loads the plugin through
its entry point.
  python3 apertus15_run.py scripts/launch_vllm.py train MODEL ...
  python3 apertus15_run.py -m speculators.train ...
"""
import os
import runpy
import sys

import apertus15_text

apertus15_text.register_config()
if sys.argv[1] == "-m":
    module = sys.argv[2]
    if module == "speculators.train":
        from train_hooks import install

        install()
    sys.argv = [module, *sys.argv[3:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
else:
    script = os.path.abspath(sys.argv[1])
    sys.path.insert(0, os.path.dirname(script))
    sys.argv = [script, *sys.argv[2:]]
    runpy.run_path(script, run_name="__main__")
