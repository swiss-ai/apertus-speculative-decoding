# Interactive DSpark e10 examples

This uses the existing DSpark deployment and an already served vanilla model.
It never launches or stops a model. Ten synthetic examples cover explanation,
Python generation/debugging, incident summarization, JSON extraction, arithmetic,
German/French translation, Bosnian writing, multi-turn revision and grounded QA.
The full corpus is in [usage-prompts.yaml](usage-prompts.yaml).

The bounded client waits for the deployment's existing direct/gateway startup
checks to pass, warms each endpoint with one short greeting, then sends 20
streamed requests sequentially (one per prompt and endpoint). Request order
alternates between endpoints. Both use greedy generation, seed 1, normal EOS
and `enable_thinking=false`; each prompt has its own output cap of 384–900 tokens.

Run on Clariden in the isolated usage-demo checkout, with the existing
`apertus-bench` environment. Credentials come from the existing pilot keyring
resolver in memory. Do not put an API key in the command or output directory.

```bash
export PYTHONPATH="$PWD/src"
nohup ~/venvs/apertus-bench/bin/python experiments/serving-pilot/run-usage-demo.py \
  /path/to/existing/deployment \
  --output /path/to/NEW/usage-demo-results \
  > /path/to/controller.log 2>&1 < /dev/null &
```

The report contains full prompts and answers from both endpoints, time to first
token, total request time, completion tokens, finish reasons and available
DSpark counter deltas. `finish_reason=length` flags a truncated answer. Objective
checks cover transport, nonempty final text, reasoning disabled, natural EOS and
exact JSON extraction; other answer quality needs human review against the
listed requirements. No generated code is executed.

Each raw cell retains `metadata.json`, `requests.jsonl`, `summary.json`, and
DSpark metric snapshots when available. Responses include the received streaming
events. The run preserves corpus/source hashes, Git branch/revision, startup
validation, pinned serving assets, rendered Model Launch scripts and the current
gateway model listings. The existing vanilla model's unavailable provenance
fields are explicitly null. A manifest verifies the downloadable result archive.

This is a small usage demonstration, not a controlled speedup or losslessness
test. The existing vanilla worker has different hardware/runtime/template/context
settings and external traffic. Worker-wide acceptance deltas can include other
requests. No paired causal performance or correctness claim follows from this
comparison. Prompts are invented examples, not real production conversations.

`collect-usage-demo.py` can wait locally for the result archive and copy the
verified report/raw cells into this checkout. It depends on valid Clariden SSH
access; the cluster results remain available if collection fails.
