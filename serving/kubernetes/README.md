# Persistent epoch-10 DSpark serving on Kubernetes

Status: prepared locally, **not deployed**. Breithorn access is not configured on
this machine or Clariden. No serving pod or replacement Slurm job was submitted.

The GitOps integration checkout is
`/home/projects/rob-poc-dspark-kubernetes`, branch
`serving/dspark-e10-kubernetes`, based on `rob-poc/main` at
`bb9b7f1beaa9f5cef5811ee175a7bb81d149e8de`. Its AGENTS.md requires workload
changes through Git and ArgoCD. Do not apply or patch these resources directly.

The candidate has a dedicated ArgoCD Application, 80Gi retained assets/logs PVC,
staging pod, immutable ConfigMap, Deployment and internal Service. Initial replicas
are zero until assets are staged. The final replicas value is one. Deployment
restartPolicy is Always, with startup/readiness/liveness probes and no
activeDeadlineSeconds. Recreate updates avoid requiring a second GPU. This keeps
the model running until its desired state is changed; it cannot guarantee no
downtime during infrastructure failure or a one-replica update.

## Runtime and discovery

Reuse the exact epoch-10 head and target from the completed controlled benchmark,
including k=3, TP=1, bf16, 32k context, thinking off and prefix caching off. No
training or new performance campaign is needed. The benchmark's four modified
vLLM source files and forced-thinking-off chat template are embedded unchanged.
Every worker start validates the vLLM Git revision and all frozen model, drafter,
binary, template and overlay hashes before serving. A failed check prevents
registration. Each start retains a unique run directory with identities, command,
verification, GPU inventory, OpenTela version, worker log and exit status.

The arm64 OCI child `8eb36e28...` of the existing immutable image index
`634973c5...` has vLLM checkout `a601a9d998ddeb488f0c17e8512874b116aa7658`
in its build history, matching the benchmark. It was inspected from GHCR on
2026-10-31. The SquashFS digest and OCI digest are different artifact formats;
they are not interchangeable. Reuse the pinned benchmark OpenTela binary too.

The model will advertise
`SwissAI-Research/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3`; vLLM accepts both this
ID and its unprefixed form for serving-api routing.

Production bootstrap configuration currently follows the live GitOps source
(`/ip4/148.187.108.178/tcp/43905/p2p/QmbUKJkCfotDzbFE5uoTsXD4GRyPHjzZC1f2yAGLoeBMn9`).
**Its handshake is not verified.** Production's public bootstrap endpoint still
returned an empty array on 2026-10-31. Do not substitute the corrected dev peer ID
on the production IP. Confirm connectivity in-cluster and correct this candidate
before calling discovery fixed.

## Access needed once

Follow [ACCESS.md](ACCESS.md) for exact commands. The user's existing login has
pod read/exec access and can be exported with `export-login.py` to proceed now.
This exports an expiring session; no new account or RoleBinding is needed.
Other live read permissions remain to be verified with the actual credential.
Optionally, an administrator can use
`setup-access.py` to create a dedicated, namespace-scoped account and a long-lived
ServiceAccount token if permitted by cluster policy. This avoids routine expiry;
it is not an automatically renewing OIDC login. The script has not been run
against Breithorn. Upload its private kubeconfig to Clariden at
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-kubernetes-access/dspark-agent.config`; the agent can retrieve it from
there, install the persistent workspace copy, and remove the temporary scratch
copy after staging. The Clariden home directory is currently at quota. Include the
cluster CA; TLS verification remains enabled. Cluster admins manage the
account/RBAC. The serving API key does not grant Kubernetes access.
Use explicit context and namespace in every cluster command. GitOps changes need
GitHub access (already available), cluster inspection needs get/list/watch for
pods, deployments, services, ConfigMaps, PVCs, events, quotas, nodes (for GPU
availability), and Argo Applications; staging needs pods/exec and pods/log.
Do not install privileged credentials or tenant-managed Roles in synced paths.

## Completion sequence after access is configured

1. Inspect the current Application, deployment/pod state, PVCs, GPU availability,
   quotas and existing staging pods. Preserve all unrelated workloads. Check for
   another DSpark worker before starting a replacement.
2. Commit the implementation on its Apertus task branch, then generate manifests:

   ```bash
   .venv/bin/python serving/kubernetes/prepare.py /tmp/dspark-k8s-manifests
   ```

   This is a local renderer only. Copy the YAMLs into the dedicated GitOps
   workload folder and add `applications/vllm-apertus-dspark-e10.yaml`. Stage the
   PVC/helper and zero-replica Deployment through the task branch/PR and ArgoCD.
   Source revisions and content hashes must match the committed implementation.
3. Follow `rob-poc/docs/how-to.md` for a foreground Clariden-to-PVC stream.
   Put the exact `export-assets.py` and rendered `deployment-provenance.json`
   on Clariden and record their hashes. Use a short-lived, namespace-scoped
   kubeconfig there for this transfer if needed; keep it outside the repo.
   The foreground remote pipeline is:

   ```bash
   set -euo pipefail
   python3 export-assets.py deployment-provenance.json |
     kubectl --context breithorn-oidc -n rob-poc exec -i \
       vllm-apertus-v1p5-8b-dspark-e10-stage -- sh -ec '
         test ! -e /assets/target
         mkdir /assets/incoming
         cd /assets/incoming
         tar xf -
         sha256sum -c SHA256SUMS
         mv target drafter opentela training-provenance SHA256SUMS \
           deployment-provenance.json /assets/
         cd /assets
         rmdir incoming
         mkdir runs
       '
   ```

   Do not overwrite a partial/existing transfer or use a detached kubectl
   stream. Inspect it first. Hash verification runs on the staging pod, not on
   the login node. Model files are copied once, not downloaded on each restart.
4. Set replicas to one in Git, remove the helper manifest once transfer is
   verified, and let Argo reconcile. Retain the PVC. Verify Application sync and
   health, pod readiness, GPU, asset hashes, real OpenTela bootstrap connection,
   model listing on production and the serving dashboard, then authenticated
   streaming requests and active draft counters. Record actual pod/node/image
   digest and gateway request artifacts in a new results directory. Until those
   checks pass, this is a candidate, not a completed deployment.
5. To stop permanently, set Deployment replicas to zero **in Git**. Argo self-heal
   would undo a manual kubectl scale. Keep the PVC and provenance for resumption.
