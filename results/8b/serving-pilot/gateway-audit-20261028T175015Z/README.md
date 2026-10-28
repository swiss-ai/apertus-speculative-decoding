# DSpark pilot gateway validation audit

State collected on 28 October 2026 at 17:50 UTC. The implementation is on
`feat/dspark-research-serving-pilot` at `838971f2`. This follows the
[launch audit](../launch-audit-20261028T151519Z/README.md) and
[pre-start continuation audit](../continuation-audit-20261028T154130Z/README.md).

The separate development key is available through keyring service
`apertus-speculative-decoding`, username `dev-api-key`. The authenticated
development request reaches model routing and returns HTTP 400 with a
no-provider response for a deliberately unavailable model. Authentication is
accepted. No live model completion has been tested yet, and the credential's
value is absent from this audit.

The gateway exposes `/v1/tokenize`; its root `/tokenize` returns 404.
The context-limit check now selects the gateway route for authenticated
requests and the root route for direct worker requests. Both paths have
HTTP transport tests. All **244 tests** and lint checks pass.

| Deployment | Existing Slurm job | State | Active validation controller |
| --- | --- | --- | --- |
| DSpark e10, k=3 | 3783300 | PENDING | 285633 |
| Matched plain target | 3783531 | PENDING | 285639 |
| Independent plain control | 3783544 | PENDING | 285695 |

All three controllers are alive, enable authenticated gateway checks, and
allow a 36-hour readiness wait. They run direct checks first. DSpark and the
matched plain target then run the bounded measurements and k6 arrival test;
the independent control retains `--smoke-only --stop-after`.

Only the waiting validation processes were replaced. No serving job was
cancelled, restarted, held or resubmitted during this gateway update. Their
source manifests are unchanged from the pre-start amendment. The controller
journals preserve prior and current process identities, exact commands,
branch, revision and source hashes. All previous audit snapshots remain intact.

- [Authentication check](operations/refresh-gateway-validation-20261028T174942Z/credential-check.json)
- [Gateway tokenizer route checks](operations/refresh-gateway-validation-20261028T174942Z/tokenizer-route-check.json)
- [Controller update journal](operations/refresh-gateway-validation-20261028T174942Z/events.json)
- [Initial gateway enablement](operations/enable-dev-gateway-20261028T174704Z/events.json)
- [Active controller identities and file hashes](snapshot.json)
- [Collection script](snapshot-source.py)

The complete run trees remain on Clariden under
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/`.
Runtime preflight, completed smoke checks, correctness and performance verdicts,
and research qualification remain pending allocation. The gateway credential
blocker is resolved; this snapshot does not establish a live research endpoint.
