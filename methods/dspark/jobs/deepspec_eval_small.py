"""DeepSpec eval.py with a reduced task list so three drafters fit one short job."""
import importlib, os, sys, torch

def run():
    sys.path.insert(0, os.getcwd())
    ev = importlib.import_module("eval")           # DeepSpec eval.py, run from the repo root
    args = ev.parse_args()
    args.tasks = [("gsm8k", 32), ("humaneval", 16), ("mt-bench", 8)]
    torch.multiprocessing.spawn(ev.main, args=(args,), nprocs=torch.cuda.device_count())

if __name__ == "__main__":
    run()
