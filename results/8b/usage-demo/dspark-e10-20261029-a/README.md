# DSpark e10 usage demo — awaiting the serving job

Job **3791982** is still queued on Clariden. No answers from this deployment
have been collected yet. A detached client (PID **55064** on `clariden-ln001`)
will run automatically once its existing direct/gateway startup checks pass.
It will send ten realistic synthetic prompts to DSpark and the already served
vanilla endpoint, one request at a time. It launches no models and leaves the
DSpark server running.

The cases cover explanation, Python coding, debugging, incident summarization,
JSON extraction, arithmetic, German/French translation, Bosnian writing,
multi-turn revision and grounded question answering. See the
[exact prompts](../../../../experiments/serving-pilot/usage-prompts.yaml) and
[capture procedure](../../../../experiments/serving-pilot/USAGE-DEMO.md).

The local collector will replace this page with a link to `report.md` after
downloading and verifying the results. That report will contain full prompts
and answers from both endpoints, first-token and total response times, output
token counts, truncation flags and DSpark acceptance-counter deltas.

These are usage examples, not a controlled speedup or losslessness test. The
existing vanilla server has different serving settings and external traffic;
worker-wide counter deltas can include other requests. No new vanilla servers
or independent GPU allocations are created for this demo.

Remote results:
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-usage-code/results/8b/usage-demo/dspark-e10-20261029-a`

Code: branch `experiment/dspark-e10-usage-demo`, revision
`8539a58eea86a3fa05532ba04a57edb0e26d371d`.
The controller and local collector wait at most 36 hours. If local SSH access
expires, the results remain on Clariden and can be retrieved after access is
renewed. The serving job retains its existing 12-hour allocation limit.
