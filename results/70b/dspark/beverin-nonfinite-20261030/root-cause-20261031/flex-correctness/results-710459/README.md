# Attention oracle setup failure, job 710459

Both image cells exited before completing any numerical case. Job 710459
failed with exit code 1 after 199 seconds on nid002474 (31 October 2026,
15:58:37-16:01:56 UTC), using immutable runtime
1b5aa6ae0a6482f152cd03dcd56b44e6ae4dccbb.

The common exception was:

```
torch._dynamo.exc.InternalTorchDynamoError:
ModuleNotFoundError: No module named 'captured_mask'
```

probe.mask_module loaded the pinned mask via module_from_spec/exec_module
without registering it in sys.modules. Dynamo's compiled mask builder later
attempted importlib.import_module('captured_mask') and failed. Each result
contains status=running with zero cases; the batch summary correctly declares
both cells failed. This is a harness import/compilation setup failure. It
provides no attention forward/backward numeric result and no NaN reproduction.
No automatic retry was made.

All 27 raw files (97,038 bytes) match collection.json's remote hashes. Raw
sources, complete tracebacks, image hashes, cell exit codes, Slurm job record,
launch provenance and result JSON are retained byte for byte. sacct times
are cluster local UTC+1.
