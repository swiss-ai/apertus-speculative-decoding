# One-time Breithorn access setup

Run these steps on the computer where you already administer Breithorn using
kubectl. The website administrator role is separate from Kubernetes API access.
No models or training jobs are launched by this procedure.

## 1. Check your Kubernetes account

```bash
kubectl config get-contexts
kubectl --context breithorn-oidc -n rob-poc auth can-i create roles.rbac.authorization.k8s.io
kubectl --context breithorn-oidc -n rob-poc auth can-i create rolebindings.rbac.authorization.k8s.io
```

Both permission checks should print `yes`. If your Breithorn context has a
different name, use that name in these commands and `--context` below. If the
context is missing, you first need your existing administrator kubeconfig from
the machine where you use Breithorn, or the CSCS Kubernetes administrators. The
setup script cannot create initial administrator access. If either permission
check says `no`, a Kubernetes administrator must run the next step.

## 2. Get the script and create the reusable account

The agent has placed the **script only**, with no credentials, on Clariden for
convenient download. From your laptop/WSL terminal:

```bash
scp -J faruk_zahiragic@ela.cscs.ch \
  faruk_zahiragic@clariden-ln001.cscs.ch:/iopsstor/scratch/cscs/faruk_zahiragic/apertus-kubernetes-access/setup-access.py \
  "$HOME/setup-breithorn-agent-access.py"

python3 "$HOME/setup-breithorn-agent-access.py" \
  --context breithorn-oidc \
  --output "$HOME/breithorn-dspark-agent.kubeconfig" \
  --approve-long-lived-token
```

The administrator must approve use of a long-lived token under cluster policy.
It uses Kubernetes' manually created ServiceAccount-token Secret mechanism:
https://kubernetes.io/docs/tasks/configure-pod-container/configure-service-account/#manually-create-a-long-lived-api-token-for-a-serviceaccount
The token avoids ordinary expiry; an administrator can revoke it by deleting
its Secret or ServiceAccount. The prior suggestion that this would automatically
renew was imprecise: this path uses a long-lived credential instead.

The script checks the server and permissions, creates only these access objects
in `rob-poc`: ServiceAccount, Role, RoleBinding and token Secret, all named
`apertus-dspark-agent` (Secret: `apertus-dspark-agent-token`). It grants namespace
inspection, pod logs and pod exec for staging/diagnostics. It does not grant
cluster-admin, Secret reads, or workload mutations; deployments remain GitOps
managed. It writes a mode-0600 JSON kubeconfig containing the new account's token
and CA, with context `breithorn-oidc`, and never prints credentials. An existing
output file is not overwritten. Do not commit the generated kubeconfig.

If cluster policy does not permit this token type, obtain the cluster's supported
unattended machine-identity configuration from its administrators instead.

## 3. Verify the file locally

```bash
kubectl --kubeconfig "$HOME/breithorn-dspark-agent.kubeconfig" -n rob-poc get pods
kubectl --kubeconfig "$HOME/breithorn-dspark-agent.kubeconfig" -n rob-poc auth can-i create pods/exec
```

Expect a pod list (or `No resources found`) and `yes`.

## 4. Put it on Clariden

Clariden home-directory creation failed with `Disk quota exceeded` on 2026-10-31.
Use the private scratch directory below for credential handoff. The agent will
install a persistent copy in this workspace and can remove the scratch copy
after staging; scratch is not the permanent credential store. Your default
kubeconfig is preserved.

```bash
ssh -J faruk_zahiragic@ela.cscs.ch faruk_zahiragic@clariden-ln001.cscs.ch \
  'mkdir -p /iopsstor/scratch/cscs/faruk_zahiragic/apertus-kubernetes-access && chmod 700 /iopsstor/scratch/cscs/faruk_zahiragic/apertus-kubernetes-access'

scp -J faruk_zahiragic@ela.cscs.ch \
  "$HOME/breithorn-dspark-agent.kubeconfig" \
  faruk_zahiragic@clariden-ln001.cscs.ch:/iopsstor/scratch/cscs/faruk_zahiragic/apertus-kubernetes-access/dspark-agent.config

ssh -J faruk_zahiragic@ela.cscs.ch faruk_zahiragic@clariden-ln001.cscs.ch \
  'chmod 600 /iopsstor/scratch/cscs/faruk_zahiragic/apertus-kubernetes-access/dspark-agent.config && test -s /iopsstor/scratch/cscs/faruk_zahiragic/apertus-kubernetes-access/dspark-agent.config && echo "Kubeconfig installed"'
```

## 5. Tell the agent

Say: `The kubeconfig is on Clariden in apertus-kubernetes-access/dspark-agent.config`.
Do not paste the file or token into chat. The agent can fetch the file, install
it privately in this workspace, verify access on both machines and continue the
prepared Kubernetes deployment. No kubectl installation on Clariden is needed
from you; the agent can handle that once the credential is available.
