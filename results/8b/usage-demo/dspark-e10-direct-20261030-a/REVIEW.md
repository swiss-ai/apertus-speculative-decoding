# DSpark usage results — 30 October

The requested examples have been captured. Read the
[full prompts, answers and per-request timings](report.md).
This note is a manual review of those saved outputs; no generated code was executed.

| Observation | DSpark | Existing vanilla |
|---|---:|---:|
| Successful requests | 10/10 | 10/10 |
| Truncated answers | 0 | 0 |
| Median first-token time | 25.8 ms | 32.8 ms |
| Median total response time | 433.6 ms | 876.8 ms |
| Draft-token acceptance across observed worker counter windows | 63.2% | Not applicable |

These timings use different request paths: DSpark was queried directly on the
running Clariden worker; vanilla used the research gateway. Hardware, context,
templates and external traffic also differ. The timing difference cannot be
attributed to the drafter. Acceptance is a worker-wide counter delta, not an
isolated request trace. It varied from 35.0% on translation to 97.5% on extraction.

The job is `3791982` on `nid007257`, started at 05:14 Sarajevo time. Its 12-hour
allocation is due to end around 17:14 on 30 October. The target and existing
DSpark e10 head passed runtime preflight and direct request checks. The worker
is **not discoverable on the dev gateway** at the time of this review. The
original automated gateway check failed and stopped the first demo attempt;
it was preserved and this capture used a new result directory with explicit
direct routing. No serving job was restarted, replaced or extended for this capture.

## Answer quality

All responses completed, but that is not a quality pass. Five pairs have exactly
the same final text, including two shared incorrect answers. This is an informal
comparison and does not establish losslessness or a drafter-specific quality effect.

| Example | Review of the saved replies |
|---|---|
| JSON extraction | Both return the exact requested JSON values and missing invoice number as null. |
| Bosnian team message | Both include the URL, deadline and requested feedback fields within the length cap. |
| Multi-turn rewrite | Both preserve retry advice, pilot caveat and Friday deadline under 60 words. |
| Python intervals | Both call `intervals.sort()`, violating the explicit requirement not to mutate the input. The supplied tests do not check that requirement or touching intervals. |
| Mutable-default debugging | Both identify the default-argument bug, but their fix still mutates a caller-supplied list instead of copying it as requested. |
| Worker scheduling arithmetic | Both correctly count 10 jobs by 09:35, but give 09:35 for the 12th completion. The correct time is 09:42: six seven-minute rounds with two completions per round. |
| Incident summary | Both invent a “Release Lead” owner and 25 October deadline for the paused release. Neither follows the requested three-bullet format. |
| Grounded QA | Both claim the notes do not specify a compiled-kernel location, overlooking S2, which explicitly assigns compiled kernels to `/scratch/cache`. |
| Translation | Both change the supplied CEST time-zone label to MEZ in the German answer instead of preserving it. |
| Speculative-decoding explanation | Both incorrectly describe rejecting the whole proposed batch and restarting. Standard speculative verification retains the accepted prefix and handles the first rejected position; the answers also omit draft-computation overhead as a reason acceptance alone cannot specify speedup. |

The most useful conclusion from this small demonstration is that the drafter is
active and produces readable answers, while several instruction-following and
reasoning errors are shared by the existing vanilla model. The raw replies are
available so the user can inspect them independently.

## Reproducibility

The run retains the exact synthetic corpus, generation settings, full responses,
stream events, raw metric snapshots, source hashes, serving launch scripts and
identity metadata. `artifact-manifest.json` verifies the original capture files;
this review and the local README are added derived documents. See `metadata.json`
and `serving-provenance/` for the job, pinned model/checkpoint/image identities and
the actual request routes.
