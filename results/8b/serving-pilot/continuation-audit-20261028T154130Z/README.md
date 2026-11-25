# DSpark pilot continuation audit

State collected on 28 October 2026 at 15:41 UTC, following the
[original launch audit](../launch-audit-20261028T151519Z/README.md).
Work continues on `feat/dspark-research-serving-pilot`; the implementation
revision at collection was `b8170b9f`.

All three existing deployments remain pending without an allocation:
DSpark **3783300**, plain **3783531**, and independent plain control **3783544**.
No job was restarted or resubmitted. All validation controllers are alive
with a 36-hour allocation wait; they use direct requests while development
gateway authentication is unresolved. The control retains
`--smoke-only --stop-after`.

An inspection of the installed OpenTela CLI showed that its metadata command
is `otela version`; `otela --version` is invalid. Before any job started, each
pending job was held, its frozen preflight source was amended to use the
correct subcommand, its source manifest was updated, and the job was released.
The launch command, resolved environment, weights, engine settings, overlays
and job ID remain unchanged.

Each `source-amendments/` directory below preserves the original and amended
module and source manifest, exact patch, hashes and amendment record. The
original `sources.tar.gz` and a new `sources.after.tar.gz` remain on Clariden;
their hashes are recorded. The original launch audit remains unchanged.
The operation journal includes scheduler records before holding and after
release, along with the exact amendment script.

- [DSpark amendment](dev-dspark-20261028-a/source-amendments/20261028T154001Z/amendment.json)
- [Plain amendment](dev-plain-20261028-b/source-amendments/20261028T154001Z/amendment.json)
- [Independent control amendment](dev-plain-control-20261028-a/source-amendments/20261028T154001Z/amendment.json)
- [Operation and release journal](operations/prestart-opentela-version-20261028T154001Z/amendments.json)
- [Controller replacement journal](operations/extend-waits-20261028T154025Z/events.json)
- [Collected status and file hashes](snapshot.json)
- [Collection script](snapshot-source.py)

The canonical run trees are under
`/iopsstor/scratch/cscs/faruk_zahiragic/apertus-pilot-code/results/8b/serving-pilot/`.
Runtime preflight and all correctness, performance and gateway verdicts remain
pending. This audit records a startup correction and operational state; it
does not establish qualification or a live research endpoint.
