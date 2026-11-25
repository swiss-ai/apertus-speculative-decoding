# DSpark e10: representative usage examples

Handwritten synthetic usage examples, not a production sample or correctness gate. One sequential request per prompt/model, normal EOS, thinking requested off. The existing vanilla endpoint has different hardware/runtime/template/context settings and uncontrolled concurrent traffic. Timings show the observed user experience; they do not isolate the drafter's speedup. Draft acceptance uses worker-wide counter deltas and can include other traffic; it is not a token-level trace of this request.

DSpark: `faruk_zahiragic/swiss-ai/Apertus-v1.5-8B-dspark-e10-k3`. Existing vanilla: `SwissAI-Research/swiss-ai/Apertus-v1.5-8B-fixed`.

| Prompt | Endpoint | First token (s) | Total (s) | Output tokens | Finish | Draft acceptance* |
|---|---|---:|---:|---:|---|---:|
| explain-speculation | dspark | 0.025 | 0.452 | 117 | stop | 36.3% |
| explain-speculation | vanilla | 0.033 | 0.703 | 116 | stop | unavailable |
| python-intervals | vanilla | 0.033 | 2.077 | 355 | stop | unavailable |
| python-intervals | dspark | 0.040 | 0.809 | 365 | stop | 85.9% |
| debug-python | dspark | 0.025 | 0.323 | 114 | stop | 65.0% |
| debug-python | vanilla | 0.070 | 0.768 | 118 | stop | unavailable |
| incident-summary | vanilla | 0.033 | 1.669 | 282 | stop | unavailable |
| incident-summary | dspark | 0.033 | 0.674 | 282 | stop | 77.3% |
| extract-json | dspark | 0.027 | 0.228 | 105 | stop | 97.5% |
| extract-json | vanilla | 0.059 | 0.674 | 105 | stop | unavailable |
| planning-math | vanilla | 0.026 | 0.985 | 168 | stop | unavailable |
| planning-math | dspark | 0.026 | 0.415 | 168 | stop | 73.7% |
| translate-german-french | dspark | 0.024 | 0.769 | 204 | stop | 35.0% |
| translate-german-french | vanilla | 0.051 | 1.243 | 204 | stop | unavailable |
| bosnian-team-message | vanilla | 0.030 | 0.523 | 86 | stop | unavailable |
| bosnian-team-message | dspark | 0.026 | 0.331 | 86 | stop | 36.6% |
| multi-turn-rewrite | dspark | 0.025 | 0.181 | 48 | stop | 42.9% |
| multi-turn-rewrite | vanilla | 0.029 | 0.298 | 48 | stop | unavailable |
| grounded-qa | vanilla | 0.032 | 1.173 | 198 | stop | unavailable |
| grounded-qa | dspark | 0.027 | 0.477 | 197 | stop | 76.1% |

*Worker-wide metric delta during the DSpark request; other traffic can contribute.

Finish `length` means the answer reached its output cap. Timings include gateway/network overhead. Objective checks below are limited checks, not a quality or losslessness verdict.

## explain-speculation

### Prompt

**user**

Explain speculative decoding to a software engineer who understands ordinary autoregressive generation. Use a small example with three proposed tokens, explain what happens on a rejection, and explain why a 50% draft-token acceptance rate does not by itself tell us the speedup. Keep it under 220 words.

Review points: Explain the target/drafter roles, verification and why acceptance alone does not imply speedup.

### dspark answer

Speculative decoding generates multiple candidate tokens in parallel, unlike autoregressive generation which produces one token at a time. For three tokens, the model might propose T1, T2, T3 simultaneously. If any candidate fails validation (e.g., violates constraints), the entire speculative batch is rejected, forcing a restart. A 50% draft-token acceptance rate doesn't indicate speedup because it ignores batch-level rejection. Rejections nullify prior speculative work, so true speedup depends on the ratio of accepted batches to total speculative attempts, not just individual token success rates.

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/explain-speculation/dspark/requests.jsonl](cells/explain-speculation/dspark/requests.jsonl)
### vanilla answer

Speculative decoding generates multiple candidate tokens in parallel, unlike autoregressive generation which produces one token at a time. For three tokens, it might propose T1, T2, T3 simultaneously. If any candidate fails validation (e.g., violates constraints), the entire speculative batch is rejected, forcing a restart. A 50% draft-token acceptance rate doesn't indicate speedup because it ignores batch-level rejection. Rejections nullify prior speculative work, so true speedup depends on the ratio of accepted batches to total speculative attempts, not just token-level success rates.

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/explain-speculation/vanilla/requests.jsonl](cells/explain-speculation/vanilla/requests.jsonl)

## python-intervals

### Prompt

**user**

Write Python 3 code for merge_intervals(intervals), where intervals is a list of (start, end) integer tuples. Return a sorted list of non-overlapping tuples. Merge touching intervals, support an empty list, reject any interval with start > end using ValueError, and do not mutate the input. Include 5 assert-based tests covering these requirements. Return one Python code block and a brief complexity explanation.

Review points: Valid Python, sorted merged intervals, touching intervals merged, empty input handled, input not mutated, tests included.

### vanilla answer

```python
from typing import List, Tuple

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    # Validate input: each interval must have start <= end
    for start, end in intervals:
        if start > end:
            raise ValueError("Each interval must have start <= end")
    # Sort intervals by start time
    intervals.sort()
    merged = [intervals[0]]
    for current in intervals[1:]:
        last_start, last_end = merged[-1]
        current_start, current_end = current
        if current_start <= last_end:
            # Merge overlapping or touching intervals
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append(current)
    return merged
```

**Complexity Explanation:**  
Time complexity: \(O(n \log n)\) due to sorting the intervals. Space complexity: \(O(n)\) for storing the merged intervals.

**Assert-Based Tests:**
```python
assert merge_intervals([(1, 3), (2, 6), (7, 9)]) == [(1, 6), (7, 9)]
assert merge_intervals([]) == []
assert merge_intervals([(0, 5)]) == [(0, 5)]
assert merge_intervals([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]
try:
    merge_intervals([(2, 1)])
except ValueError as e:
    assert str(e) == "Each interval must have start <= end"
```

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/python-intervals/vanilla/requests.jsonl](cells/python-intervals/vanilla/requests.jsonl)
### dspark answer

```python
from typing import List, Tuple

def merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    # Validate input: each interval must have start <= end
    for start, end in intervals:
        if start > end:
            raise ValueError("Each interval must have start <= end")
    # Sort intervals by start time
    intervals.sort()
    merged = [intervals[0]]
    for current in intervals[1:]:
        last_start, last_end = merged[-1]
        current_start, current_end = current
        if current_start <= last_end:
            # Merge overlapping or adjacent intervals
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append(current)
    return merged
```

**Complexity Explanation:**  
- Time Complexity: \(O(n \log n)\) due to sorting the intervals.  
- Space Complexity: \(O(n)\) for storing the merged intervals.  

**Assert-Based Tests:**  
```python
assert merge_intervals([(1, 3), (2, 6), (7, 9)]) == [(1, 6), (7, 9)]
assert merge_intervals([]) == []
assert merge_intervals([(0, 5)]) == [(0, 5)]
assert merge_intervals([(1, 2), (3, 4)]) == [(1, 2), (3, 4)]
try:
    merge_intervals([(2, 1)])
except ValueError as e:
    assert str(e) == "Each interval must have start <= end"
```

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/python-intervals/dspark/requests.jsonl](cells/python-intervals/dspark/requests.jsonl)

## debug-python

### Prompt

**user**

This function unexpectedly remembers values from earlier calls:
```python
def add_tag(tag, tags=[]):
    tags.append(tag)
    return tags
a = add_tag("alpha")
b = add_tag("beta")
```
Explain the bug, show a corrected implementation that copies a caller-supplied list instead of mutating it, and show what a and b should contain after the fix. Duplicate tags should be preserved. Keep the explanation concise.


Review points: Identify the shared mutable default, preserve duplicate tags, return independent lists, explain the symptom.

### dspark answer

**Bug:** The `tags` list is declared as a mutable default argument (`[]`). It persists across calls, so `add_tag` mutates the same list, causing unexpected shared state.

**Corrected implementation:**
```python
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
```

**Result after fix:**
- `a = add_tag("alpha")` → `["alpha"]`
- `b = add_tag("beta")` → `["beta"]`

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/debug-python/dspark/requests.jsonl](cells/debug-python/dspark/requests.jsonl)
### vanilla answer

**Bug:** The `tags` list is declared as a mutable default argument (`[]`). It persists across calls, so `add_tag` mutates the same list, causing unexpected shared state.

**Corrected implementation:**
```python
def add_tag(tag, tags=None):
    if tags is None:
        tags = []
    tags.append(tag)
    return tags
```

**After the fix:**
- `a = add_tag("alpha")` → `a = ["alpha"]`
- `b = add_tag("beta")` → `b = ["beta"]`

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/debug-python/vanilla/requests.jsonl](cells/debug-python/vanilla/requests.jsonl)

## incident-summary

### Prompt

**user**

Turn this fictional incident note into an update for our team: 3 bullet points followed by an action table (action, owner, due date). Separate confirmed facts from the root-cause hypothesis.

On 22 October, a gateway release started at 09:00 UTC. At 09:07 UTC the 5xx rate rose from 0.1% to 4.8%. Existing streaming requests continued, but some new connections failed. At 09:12 UTC on-call engineer Lejla rolled back the gateway; errors returned to 0.1% at 09:18 UTC. The impact window was 11 minutes. No evidence of data loss was found. The leading hypothesis is a connection-pool limit introduced in the release; it has not been confirmed. Model weights and inference workers did not change. Jonas will reproduce the connection-pool issue by 24 October. Lejla will add an alert for new-connection failures by 23 October. Mira will add a streaming-plus-new-connections rollout test by 25 October. The next release is paused until the reproduction and test are complete.


Review points: Preserve all times, 4.8% error rate, 11-minute impact, recovery action and owners; do not invent a proven root cause.

### vanilla answer

**Incident Update – Gateway Release Rollback (22 Oct)**

**Confirmed Facts**  
- Gateway release initiated at 09:00 UTC on 22 October.  
- 5xx error rate spiked from 0.1% to 4.8% between 09:07–09:12 UTC.  
- On-call engineer Lejla rolled back the release at 09:12 UTC; error rate returned to 0.1% by 09:18 UTC.  
- No data loss was observed during the incident.  
- Model weights and inference workers remained unchanged.  

**Root-Cause Hypothesis**  
- Connection-pool limit introduced in the 22 October release may have caused the failure of new connections.  
- *Note: This hypothesis has not been confirmed.*  

**Action Table**  
| Action | Owner | Due Date |
|--------|-------|----------|
| Reproduce connection-pool issue | Jonas | 24 October |
| Add alert for new-connection failures | Lejla | 23 October |
| Add streaming-plus-new-connections rollout test | Mira | 25 October |
| Pause next release until reproduction & test complete | Release Lead | 25 October |

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/incident-summary/vanilla/requests.jsonl](cells/incident-summary/vanilla/requests.jsonl)
### dspark answer

**Incident Update – Gateway Release Rollback (22 Oct)**

**Confirmed Facts**  
- Gateway release initiated at 09:00 UTC on 22 Oct.  
- 5xx error rate spiked from 0.1% to 4.8% between 09:07–09:12 UTC.  
- On-call engineer Lejla rolled back the release at 09:12 UTC; error rate returned to 0.1% by 09:18 UTC.  
- No data loss was observed during the incident.  
- Model weights and inference workers remained unchanged.  

**Root-Cause Hypothesis**  
- Connection-pool limit introduced in the 22 Oct release may have caused the failure of new connections.  
- *Note: This hypothesis has not been confirmed.*  

**Action Table**  
| Action | Owner | Due Date |
|--------|-------|----------|
| Reproduce connection-pool issue | Jonas | 24 October |
| Add alert for new-connection failures | Lejla | 23 October |
| Add streaming-plus-new-connections rollout test | Mira | 25 October |
| Pause next release until reproduction & test complete | Release Lead | 25 October |

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/incident-summary/dspark/requests.jsonl](cells/incident-summary/dspark/requests.jsonl)

## extract-json

### Prompt

**user**

Extract this fictional purchase note into JSON only, without markdown fences. Use exactly the keys vendor, currency, invoice_number, items, subtotal. Each item needs description, quantity, unit_price. Use numbers for quantities and amounts and null for a missing invoice number. Do not add tax or shipping.
Note: Alpine Lab Supplies charged CHF 8.50 each for 3 notebooks (description "notebook"), and CHF 12.00 each for 2 adapters (description "adapter"). No invoice number is shown. The subtotal is CHF 49.50.


Review points: JSON only; exact strings and numeric types; preserve the missing invoice number as null.

### dspark answer

{
  "vendor": "Alpine Lab Supplies",
  "currency": "CHF",
  "invoice_number": null,
  "items": [
    {
      "description": "notebook",
      "quantity": 3,
      "unit_price": 8.5
    },
    {
      "description": "adapter",
      "quantity": 2,
      "unit_price": 12.0
    }
  ],
  "subtotal": 49.5
}

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass, exact_json_extraction=pass

Raw response: [cells/extract-json/dspark/requests.jsonl](cells/extract-json/dspark/requests.jsonl)
### vanilla answer

{
  "vendor": "Alpine Lab Supplies",
  "currency": "CHF",
  "invoice_number": null,
  "items": [
    {
      "description": "notebook",
      "quantity": 3,
      "unit_price": 8.5
    },
    {
      "description": "adapter",
      "quantity": 2,
      "unit_price": 12.0
    }
  ],
  "subtotal": 49.5
}

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass, exact_json_extraction=pass

Raw response: [cells/extract-json/vanilla/requests.jsonl](cells/extract-json/vanilla/requests.jsonl)

## planning-math

### Prompt

**user**

Two independent workers start at 09:00. Each processes one job at a time and takes exactly 7 minutes per job. There is no setup time and an unlimited queue. How many jobs have finished by 09:35 inclusive, and at what time does the 12th job finish? Explain briefly, then give a JSON object with keys completed_by_0935 and twelfth_job_finished_at.

Review points: Account for two parallel workers and the jobs that finish at the boundary; include reasoning and final JSON.

### vanilla answer

Explanation: Each worker processes one job every 7 minutes. Two workers working simultaneously can finish jobs at a rate of 2 jobs per 7 minutes. From 09:00 to 09:35 (35 minutes), the number of jobs completed is calculated as (35 / 7) * 2 = 10 jobs. The 12th job will be the 11th job completed by the second worker, finishing at 09:35 (since 11 jobs take 11 * 7 = 77 minutes, which ends at 09:35).

JSON object:
{
  "completed_by_0935": 10,
  "twelfth_job_finished_at": "09:35"
}

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/planning-math/vanilla/requests.jsonl](cells/planning-math/vanilla/requests.jsonl)
### dspark answer

Explanation: Each worker processes one job every 7 minutes. Two workers working simultaneously can finish jobs at a rate of 2 jobs per 7 minutes. From 09:00 to 09:35 (35 minutes), the number of jobs completed is calculated as (35 / 7) * 2 = 10 jobs. The 12th job will be the 11th job completed by the second worker, finishing at 09:35 (since 11 jobs take 11 * 7 = 77 minutes, which ends at 09:35).

JSON object:
{
  "completed_by_0935": 10,
  "twelfth_job_finished_at": "09:35"
}

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/planning-math/dspark/requests.jsonl](cells/planning-math/dspark/requests.jsonl)

## translate-german-french

### Prompt

**user**

Translate this service notice into clear German and French, with headings for each language. Preserve the exact times and the time zone: 'Planned maintenance on 24 October, 18:00–18:20 CEST. Requests started before the maintenance window will continue streaming. New requests may fail briefly. Please retry after 18:20 CEST. We will post an update when maintenance is complete.'

Review points: Preserve the maintenance window, time zone and distinction between ongoing streams and new requests in both translations.

### dspark answer

**German**  
**Service Notice**  
Geplante Wartung am 24. Oktober, 18:00–18:20 (MEZ). Anfragen, die vor dem Wartungsfenster eingereicht wurden, werden weiterhin gestreamt. Neue Anfragen können kurzzeitig fehlschlagen. Bitte nach 18:20 (MEZ) erneut versuchen. Wir werden eine Aktualisierung veröffentlichen, sobald die Wartung abgeschlossen ist.  

**French**  
**Avis de Service**  
Maintenance planifiée le 24 octobre, de 18h00 à 18h20 (CEST). Les demandes soumises avant la fenêtre de maintenance continueront de diffuser. De nouvelles demandes peuvent échouer brièvement. Veuillez réessayer après 18h20 (CEST). Nous publierons une mise à jour dès que la maintenance sera terminée.

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/translate-german-french/dspark/requests.jsonl](cells/translate-german-french/dspark/requests.jsonl)
### vanilla answer

**German**  
**Service Notice**  
Geplante Wartung am 24. Oktober, 18:00–18:20 (MEZ). Anfragen, die vor dem Wartungsfenster eingereicht wurden, werden weiterhin gestreamt. Neue Anfragen können kurzzeitig fehlschlagen. Bitte nach 18:20 (MEZ) erneut versuchen. Wir werden eine Aktualisierung veröffentlichen, sobald die Wartung abgeschlossen ist.  

**French**  
**Avis de Service**  
Maintenance planifiée le 24 octobre, de 18h00 à 18h20 (CEST). Les demandes soumises avant la fenêtre de maintenance continueront de diffuser. De nouvelles demandes peuvent échouer brièvement. Veuillez réessayer après 18h20 (CEST). Nous publierons une mise à jour dès que la maintenance sera terminée.

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/translate-german-french/vanilla/requests.jsonl](cells/translate-german-french/vanilla/requests.jsonl)

## bosnian-team-message

### Prompt

**user**

Napiši kratku poruku timu na bosanskom jeziku, latinicom. Nova verzija internog alata je spremna na https://example.org/demo. Zamoli kolege da danas do 16:00 isprobaju jedan svoj uobičajeni zadatak i u odgovoru navedu šta su očekivali, šta su dobili i jesu li primijetili grešku. Ton neka bude prijateljski i profesionalan, bez pretjeranog oduševljenja. Najviše 100 riječi.

Review points: Natural Bosnian in Latin script, warm but concise, mention the link, verification and a 16:00 deadline.

### vanilla answer

Pozdrav tim! Nova verzija internog alata je spremna na https://example.org/demo. Molimo vas da danas do 16:00 isprobate jedan svoj uobičajeni zadatak. U odgovoru navedite šta ste očekivali, šta ste dobili i jesu li primijetili grešku. Hvala na saradnji!

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/bosnian-team-message/vanilla/requests.jsonl](cells/bosnian-team-message/vanilla/requests.jsonl)
### dspark answer

Pozdrav tim! Nova verzija internog alata je spremna na https://example.org/demo. Molimo vas da danas do 16:00 isprobate jedan svoj uobičajeni zadatak. U odgovoru navedite šta ste očekivali, šta ste dobili i jesu li primijetili grešku. Hvala na saradnji!

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/bosnian-team-message/dspark/requests.jsonl](cells/bosnian-team-message/dspark/requests.jsonl)

## multi-turn-rewrite

### Prompt

**user**

Draft a message telling our research team that the demo endpoint is ready to try. This is a pilot, so occasional interruptions are possible. Ask them to share examples of slow or incorrect replies.

**assistant**

The demo endpoint is ready to try. This is a pilot and occasional interruptions are possible. If a request fails, please retry. Please share examples of slow or incorrect replies, including the prompt, the response and the approximate time.

**user**

Revise it to sound less formal, add that feedback is due Friday, and keep it under 60 words. Keep the advice about retrying.

Review points: Respect the latest 60-word limit, mention Friday, preserve retry advice and avoid promising uninterrupted availability.

### dspark answer

The demo endpoint is live—try it out! This is a pilot, so hiccups are normal. If you hit slow or wrong responses, send examples (prompt, response, time) by Friday. Retry if needed.

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/multi-turn-rewrite/dspark/requests.jsonl](cells/multi-turn-rewrite/dspark/requests.jsonl)
### vanilla answer

The demo endpoint is live—try it out! This is a pilot, so hiccups are normal. If you hit slow or wrong responses, send examples (prompt, response, time) by Friday. Retry if needed.

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/multi-turn-rewrite/vanilla/requests.jsonl](cells/multi-turn-rewrite/vanilla/requests.jsonl)

## grounded-qa

### Prompt

**user**

Answer using only these fictional internal notes. Cite the relevant section IDs. If the notes do not answer a question, say so.
[S1] Training exports and final model weights go in /store/models. This directory is backed up nightly. The notes do not specify how many backups are retained.
[S2] Download caches, compiled kernels and temporary benchmark intermediates go in /scratch/cache. Files there may be deleted after 30 days without access; /scratch/cache is not backed up.
[S3] Completed benchmark raw requests, metadata and metric snapshots must be copied to the associated results directory before temporary files are deleted. API keys must never be included in result artifacts.
Questions: Where should I keep final drafter weights and compiled kernels? Can I delete temporary benchmark files immediately after a run? How long are the model backups retained?


Review points: Correctly distinguish permanent data and caches, do not invent backup retention, cite S1/S2/S3.

### vanilla answer

I cannot answer the questions using only the provided internal notes.

For the first question: Where should I keep final drafter weights and compiled kernels?
[S1] states that training exports and final model weights go in /store/models. However, it does not mention compiled kernels. Therefore, the location for compiled kernels is not specified in the notes.

For the second question: Can I delete temporary benchmark files immediately after a run?
[S3] states that completed benchmark raw requests, metadata and metric snapshots must be copied to the associated results directory before temporary files are deleted. This implies that temporary files cannot be deleted immediately after a run; they must wait until the artifacts are copied to the results directory.

For the third question: How long are the model backups retained?
[S1] states that the /store/models directory is backed up nightly, but it does not specify how many backups are retained. Therefore, the retention period is not specified in the notes.

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/grounded-qa/vanilla/requests.jsonl](cells/grounded-qa/vanilla/requests.jsonl)
### dspark answer

I cannot answer the questions using only the provided notes.

For the first question: Where should I keep final drafter weights and compiled kernels?
[S1] states that training exports and final model weights go in /store/models. However, it does not mention compiled kernels. Therefore, the location for compiled kernels is not specified in the notes.

For the second question: Can I delete temporary benchmark files immediately after a run?
[S3] states that completed benchmark raw requests, metadata and metric snapshots must be copied to the associated results directory before temporary files are deleted. This implies that temporary files cannot be deleted immediately after a run; they must wait until the artifacts are copied to the results directory.

For the third question: How long are the model backups retained?
[S1] states that the /store/models directory is backed up nightly, but it does not specify how many backups are retained. Therefore, the retention period is not specified in the notes.

Objective checks: transport_success=pass, natural_finish=pass, nonempty_final_answer=pass, no_reasoning_output=pass

Raw response: [cells/grounded-qa/dspark/requests.jsonl](cells/grounded-qa/dspark/requests.jsonl)

Request routes: DSpark `http://172.28.41.76:8080`, vanilla `https://api.swissai.svc.cscs.ch`.
DSpark was queried directly because it was not discoverable on the dev gateway.
