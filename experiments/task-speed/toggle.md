# Live windows: toggling the DSpark Deployment between DSpark and plain

The live speedup compares the deployed pod with itself: the one Deployment
`vllm-apertus-v1p5-8b-dspark-e10` (Breithorn, namespace `rob-poc`, GitOps repo
`breithorn-poc/rob-poc`) is switched to plain serving for a window and back.
vLLM fixes speculation at engine start, so a toggle is a pod restart (about
5–10 min including the asset hash checks). Each toggle is a Git change: Argo
self-heal reverts manual `kubectl` edits.

| Window | Mode | Round | Model ID served |
|---|---|---|---|
| D1 | DSpark | 1 | `SwissAI-Research/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3` |
| P1 | plain | 1 | `SwissAI-Research/swiss-ai/Apertus-v1.5-8B-plain-matched` |
| P2 | plain | 2 | same as P1 |
| D2 | DSpark | 2 | same as D1 |

Day 1: D1 → toggle → P1 → toggle back. Day 2, another time of day: toggle →
P2 → toggle back → D2. Plain windows run at low-traffic hours. During a plain
window the DSpark ID is not listed; users of it get an error, never plain output.

## Before the first window

1. **Ask the user** whether to announce each plain window (model description or
   Slack) and confirm the window times. Arrange the PR merges (the earlier
   PRs #275/#276 were merged from the user's account; Rob may want to review).
2. Kube access: the user renews the session per `serving/kubernetes/ACCESS.md`
   and places the kubeconfig at
   `/iopsstor/scratch/cscs/$USER/apertus-kubernetes-access/kubeconfig` (mode 600).
   Check: `kubectl --context breithorn-oidc -n rob-poc auth can-i create pods/exec`.
3. Client: runs on Clariden debug nodes with `~/venvs/apertus-bench` and the
   staged aarch64 kubectl; kubectl and the client run there together.
4. Corpus: the controlled run's `corpus.jsonl` (same prompts, same digest).
5. Optional but useful: a PR adding
   `vllm-apertus-v1p5-8b-dspark-e10.rob-poc.svc:8080` to the `vllm-models`
   static targets in `swissai-research-platform/prometheus/values.yaml`, so
   Grafana records real-user acceptance outside the windows.

## One toggle

Render in the task-speed worktree (both modes pin the node the pod runs on):

```bash
.venv/bin/python serving/kubernetes/prepare.py "$SCRATCH/render-<window>" \
  --replicas 1 --variant {plain|dspark} --pin-node nid006597
```

In `/home/projects/rob-poc-dspark-kubernetes`, branch from `origin/main`
(`serving/dspark-e10-task-speed-<window>`), copy **only** `configmap.yaml` and
`deployment.yaml` into `swissai-research-platform/models/vllm-apertus-dspark-e10/`,
add one line to its README (current mode, window, source commit), commit, push,
open the PR with `gh pr create`, and merge once approved. Never touch the PVC
(`Prune=false`) or `service.yaml`.

Wait and verify, recording everything under `<live root>/toggles/<n>-<mode>/`:

```bash
K="kubectl --context breithorn-oidc -n rob-poc"
$K rollout status deploy/vllm-apertus-v1p5-8b-dspark-e10 --timeout=20m
$K get pods -l app=vllm-apertus-v1p5-8b-dspark-e10 -o json > pod.json   # node, uid, image
$K exec <pod> -c vllm -- python3 -c "import urllib.request as u; print(u.urlopen('http://127.0.0.1:8080/v1/models').read().decode())"
$K exec <pod> -c vllm -- sh -c 'ls -t /logs/$POD_UID | head -1'      # start record
$K exec <pod> -c vllm -- sh -c 'cat /logs/$POD_UID/<start>/verification.json /logs/$POD_UID/<start>/gpu.csv'
```

Gates before a window starts:
- Argo Application synced and healthy (or the Deployment generation observed),
  one ready pod, zero restarts, node `nid006597`.
- `verification.json` passed (all asset and runtime hashes).
- The pod's `/v1/models` lists only the mode's ID; through the gateway, an
  authenticated streaming request to that ID succeeds.
- DSpark mode: `vllm:spec_decode_*` counters present and advancing after a
  request. Plain mode: no `vllm:spec_decode_*` series.
- Annotation `apertus.swissai/variant` is `plain` in plain mode and absent in
  DSpark mode (run-live.py refuses a mismatched window).

## One window

From the scratch checkout on a Clariden login node:

```bash
experiments/task-speed/run-live-window.sh D1 dspark 1 \
  results/8b/task-speed/live-<stamp> results/8b/task-speed/controlled-<stamp>/corpus.jsonl
```

The client runs as a debug job (about 70–90 min for the full suite; a job stops
starting cells after 78 min, and starting the window again measures only the
missing cells). It pre-warms
every slice at C=1 and C=128 (any gateway throttling aborts the window), then
runs each slice at C=1/8/32/128. A cell whose engine was busy at its start or
which finished other users' requests moves to `rejected/` and reruns (two
retries). Afterwards, in the same mode, the gateway-overhead cells:

```bash
experiments/task-speed/run-live-window.sh D1 dspark 1 <live root> <corpus> \
  --direct --slices opb chat summarization --concurrencies 1 32
```

`--direct` starts a `kubectl port-forward` to the pod on the client node and
records the route as `direct`; compare its latencies with the gateway cells of
the same window.

## After the last window

The final toggle leaves DSpark serving (D2). Removing the node pin is a
separate PR; **ask the user first**. Report:

```bash
python3 experiments/task-speed/report-task-speed.py \
  --controlled results/8b/task-speed/controlled-<stamp> --live results/8b/task-speed/live-<stamp> \
  --output results/8b/task-speed/report-<stamp>
```
