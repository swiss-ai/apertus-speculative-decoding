# DSpark research pilot

Serve the final 10 epoch DSpark head through the existing SML, OpenTela and
research gateway stack. The default is a development deployment with fixed
k=3, bf16, TP=1, thinking off and a 32,768-token context. The target revision
and image digest are pinned in [dspark-e10.yaml](dspark-e10.yaml).

The [launch audit](../../results/8b/serving-pilot/launch-audit-20261028T151519Z/README.md),
[continuation audit](../../results/8b/serving-pilot/continuation-audit-20261028T154130Z/README.md),
and [gateway audit](../../results/8b/serving-pilot/gateway-audit-20261028T175015Z/README.md)
record the submitted development jobs and their current pre-start inputs.
Check those jobs and controller reports before launching replacements.

Use `serving/research-pilot.sh` from the repository root on Clariden. It uses
the existing `apertus-bench` Python environment. Every launch needs a new
output directory. Commands print their output paths; retain those paths for
the following steps. `plan`, `prepare` and `submit` are separate operations.

## Prepare and submit a development deployment

```bash
./serving/research-pilot.sh plan

./serving/research-pilot.sh prepare \
  --partition debug --time 01:30:00 \
  --output "$PWD/results/8b/serving-pilot/dev-dspark-CHOSEN_ID"

PILOT_RUN="$PWD/results/8b/serving-pilot/dev-dspark-CHOSEN_ID"
bash -n "$PILOT_RUN/rendered/master.sh"
./serving/research-pilot.sh submit "$PILOT_RUN"
./serving/research-pilot.sh status "$PILOT_RUN"
```

Preparation validates small target/head assets, snapshots the source and
validation corpora, creates a unique overlay, fixes the image architecture,
and renders SML's complete job. It preserves the launcher source even if its
checkout has no Git metadata. The image, target weights and head weights are
fully hashed **inside the allocation**, before the engine starts. A mismatch
blocks OpenTela registration and produces `preflight-failed.json`.

The rendered compatibility patch replaces an older SML `vllm serve` entrypoint
with the pinned image's OpenAI API module and creates per-job log directories
before worker startup. The shared SML installation is unchanged. The full
Slurm submission command, including its working directory and log paths, is
saved in the submission journal. An uncertain submission is never retried
automatically; inspect Slurm and the journal before recovering it.

Wait for `preflight.json` to report `passed`, then require a successful smoke
check. A running Slurm job or a model listing alone does not establish readiness.
`status` reports the worker URL once SML discovers it. When needed, tail
`slurm.out`, `slurm.err` and `logs/<job>/replica_0.{out,err}` in the run directory.

The engine serves exactly the full three-segment model ID recorded in
`deployment.json`, for example
`faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3`.
The thinking-off template overrides a request that asks for thinking.
Request logging and usage telemetry are disabled; speculative counters and
ordinary operational statistics remain enabled.

For a queued deployment, `validate` can wait for readiness and run the complete
bounded validation automatically. It reads the appropriate gateway key in
memory, uses the allocation for load generation,
and saves its controller source, logs and final report under `automation/`.
Only one controller may validate a deployment at a time.

```bash
nohup ./serving/research-pilot.sh validate "$PILOT_RUN" --wait-seconds 129600 \
  > "$PILOT_RUN/validation-controller.log" 2>&1 < /dev/null &
```

Use `--smoke-only --stop-after` for the disposable plain calibration control,
or `--stop-after` for other disposable validation deployments. These options
cancel only the deployment's recorded job after collecting its logs. The
manual commands below can also be used individually.

## Validate requests and metrics

```bash
./serving/research-pilot.sh smoke "$PILOT_RUN" --base-url http://WORKER_IP:8080 --limits

# The key is resolved from the selected environment variable or platform store.
./serving/research-pilot.sh smoke "$PILOT_RUN" --gateway \
  --api-key-env CSCS_SERVING_API --metrics-url http://WORKER_IP:8080/metrics --limits
```

Gateway commands first use `--api-key-env` when that variable is set. Otherwise,
development runs read keyring service `apertus-speculative-decoding`, username
`dev-api-key`; research runs read the existing SML `swissai_research_api_key`.
The keyring reader uses the Python environment alongside the profile's `sml_bin`.
Development does not fall back to the research key. Values are held in memory,
are not exported to job steps, and are never written to pilot artifacts.

To save a development key from <https://servingdev.swissai.svc.cscs.ch>, run
these short commands on Clariden and enter the key at the hidden prompt:

```bash
source ~/venvs/sml-apertus/bin/activate
PILOT_SVC=apertus-speculative-decoding
python -m keyring set "$PILOT_SVC" dev-api-key
```

The smoke check retains two greedy captures of the eight synthetic prompts
in [smoke.jsonl](smoke.jsonl), and checks sampled/non-streaming replies,
multi-turn and multilingual behavior, usage, finish reasons, context limits,
disconnect recovery and advancing acceptance counters. Missing or reset
counters fail the DSpark check. A separate HTTP client fetches metrics so
the gateway token never reaches the worker's metrics endpoint.
Context checks use the gateway's `/v1/tokenize` route or the worker's
`/tokenize` route to construct a request close to the advertised limit.

Check the model selector and a streamed reply in the development UI as well.
Tool calls, structured outputs and logprobs are not advertised by this pilot;
add explicit feature tests before advertising them. Prompt/completion
recording remains subject to canvas item 4. Verify that pilot users have no
active content tracing rule; approved synthetic and existing validation
datasets can be tested independently.

Import [dashboard.json](dashboard.json) into the existing Grafana instance
and select its Prometheus data source, model ID and Slurm job ID. It uses
SML's existing vmagent/DCGM remote-write labels. Check that new points arrive
for the selected job. Acceptance is a ratio of counter rates, with separate
accepted-length and per-position panels; plain serving has no acceptance
series. Failure counts remain in the client/k6 result artifacts. Missing
series display no data. The dashboard import does not create a new metrics
service or change the platform gateway.
For the existing Grafana dashboard sidecar, the same dashboard is packaged as
[grafana-configmap.yaml](grafana-configmap.yaml) in the `rob-poc` namespace.

## Measure the bounded comparison

Run load generation on a compute node in the serving allocation. The resolved
environment and source snapshot are already mounted there. The following
command adds the pilot client to the container's Python path:

```bash
PILOT_JOB=JOB_ID_FROM_JOB_JSON
srun --jobid="$PILOT_JOB" --overlap --nodes=1 --ntasks=1 \
  --environment="$PILOT_RUN/environment.toml" \
  bash -c 'export PYTHONPATH="$1/source/src:${PYTHONPATH:-}"; \
    exec python3 -m apertus_bench.research_pilot measure "$1" --base-url "$2"' \
  pilot-client "$PILOT_RUN" http://WORKER_IP:8080

# This command submits k6 as a job step inside the same existing allocation.
./serving/research-pilot.sh open-loop "$PILOT_RUN" --base-url http://WORKER_IP:8080
```

`measure` runs the 64-prompt probe at C=8, then whole prompt passes for chat
and summarization at C=1/8/32, with warm-up and raw metric snapshots. Gateway
cells can be collected with `--gateway --metrics-url ...` and the bearer
token in the environment; compare the same path in both arms.

`open-loop` reuses the snapshotted model-launch k6 script and existing k6
image. It runs 2 arrivals/s for 3 minutes with a 128-token output cap, normal
EOS and at most 32 virtual users. It records the image/script/corpus hashes,
complete scenario, raw samples, native summary and dropped iterations.
The packaged k6 script uses non-streaming requests at temperature 1.0, so
this check measures end-to-end latency and offered load. Use streaming cells
for TTFT. It sends no gateway credential and allocates no additional node.

Prepare and measure a matched `--variant plain` deployment using its own new
directory and the same profile. Keep that plain model ID available for fallback.
Also capture an independently launched plain control with
`--variant plain --model-suffix control`, stopping it after its captures.
These are separate SML jobs; each requests an exclusive four-GPU node while
one engine uses one GPU. Run disposable controls sequentially if capacity is
limited. Record allocated GPU-hours alongside active serving GPUs.

Do not run `serving/loadtest.sh` against a long-lived pilot: that helper owns
and cancels its deployment on exit. The pilot controller leaves servers
running until their allocation expires or `stop` is called.

## Gate correctness and qualify the configuration

```bash
./serving/research-pilot.sh gate \
  --baseline /path/to/plain/checks/TIMESTAMP \
  --candidate /path/to/dspark/checks/TIMESTAMP \
  --control /path/to/independent-plain/checks/TIMESTAMP

./serving/research-pilot.sh qualify \
  --candidate-checks /path/to/dspark/gateway/checks \
  --fallback-checks /path/to/plain/gateway/checks \
  --baseline-measurements /path/to/plain/measurements/TIMESTAMP \
  --candidate-measurements /path/to/dspark/measurements/TIMESTAMP \
  --baseline-open-loop /path/to/plain/open-loop/TIMESTAMP \
  --candidate-open-loop /path/to/dspark/open-loop/TIMESTAMP \
  --gate /path/to/dspark/gates/TIMESTAMP
```

The correctness gate requires independent job IDs, identical control
configurations, identical prompt content and output caps, within-deployment
repeats and a cross-deployment plain control. It applies the repository's
calibrated divergence criterion; an exact-match count alone is not the gate.

Qualification requires successful authenticated gateway checks, a passing
context-limit check, the exact fallback ID, calibrated correctness, complete
matched measurement cells, advancing acceptance counters, useful speedup,
and no p95 latency regression above the profile's 10% allowance. Both arrival
tests must succeed without dropping iterations and use matching workloads,
script, image, sampling and arrival schedule. Failure retains its report.

## Research rollout and operation

```bash
./serving/research-pilot.sh prepare --platform research \
  --qualification /path/to/qualifications/TIMESTAMP/report.json \
  --output "$PWD/results/8b/serving-pilot/research-dspark-NEW_ID"
./serving/research-pilot.sh submit /path/to/research-dspark-NEW_ID
```

A research launch requires a passing development qualification covering the
same target, head, image and engine configuration. All referenced evidence
hashes are rechecked, and runtime preflight verifies the actual weights match
the qualified hashes. Establish the corresponding plain endpoint on the
research mesh, repeat gateway/UI/metrics smoke checks, and test fallback by
selecting its separate model ID. Stop the development DSpark job during the
rollback test; do not stop unrelated deployments. Automatic retry or stream
replay between plain and DSpark is not implemented.

```bash
./serving/research-pilot.sh soak /path/to/research-run --gateway \
  --metrics-url http://WORKER_IP:8080/metrics
./serving/research-pilot.sh collect /path/to/research-run
./serving/research-pilot.sh stop /path/to/research-run
```

The default soak lasts two hours, saves aggregate request timings and raw
metrics, detects counter resets and sustained queue growth, and retains no
live prompt/completion content. Keep it in a durable shell if disconnects
are possible. `--seconds` supports short operational checks; a short check
does not qualify the configuration as stable. `qualify --soak <directory>`
adds the required full observation window to its checks.
Use checks and measurements from the same deployment as that observation.

SML jobs expire. Publish the operator, availability window, fallback ID and
expiry with the endpoint. A replacement deployment gets a new directory and
new submission record. Consecutive job handovers or managed Kubernetes
hosting can follow after the bounded pilot; queueing can interrupt a Slurm
service. `stop` verifies the recorded job's Slurm owner and collects logs
before cancelling only that job.

Keep the complete run tree on the cluster, including failed attempts, source
archives, training provenance, patches, configurations, raw cells and logs.
Fetch compact status and reports for review. Full provenance requirements
remain in [AGENTS.md](../../AGENTS.md), [results/README.md](../../results/README.md)
and [docs/protocol.md](../../docs/protocol.md).
