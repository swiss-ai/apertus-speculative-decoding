"""Send prompts to a running vLLM server, then read spec-decode counters from /metrics."""
import json, re, sys, time, threading, queue, requests, os
url, name, out_path, n_prompts, max_tokens = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
import glob, os
ds = glob.glob(os.path.expandvars("$SPEC/hf/hub/datasets--RedHatAI--speculator_benchmarks/snapshots/*/"))[0]
prompts = []
for f in ["math_reasoning.jsonl", "HumanEval.jsonl"]:
    with open(os.path.join(ds, f)) as fh:
        rows = [json.loads(l) for l in fh][: n_prompts // 2]
    prompts += [(f, r.get("prompt") or r.get("text") or next(v for v in r.values() if isinstance(v, str))) for r in rows]
def m():  # scrape counters
    t = requests.get(url + "/metrics", timeout=30).text
    c = {}
    for line in t.splitlines():
        if line.startswith("vllm:spec_decode"):
            k, v = line.rsplit(" ", 1); c[k] = c.get(k, 0.0) + float(v)
    return c
def sm(c, key): return sum(v for k, v in c.items() if k.startswith(key))
def per_pos(c):
    d = {}
    for k, v in c.items():
        mm = re.search(r'accepted_tokens_per_pos.*position="(\d+)"', k)
        if mm: d[int(mm.group(1))] = d.get(int(mm.group(1)), 0.0) + v
    return d
before = m()
q = queue.Queue(); [q.put(p) for p in prompts]; lat = []
def worker():
    while True:
        try: f, p = q.get_nowait()
        except queue.Empty: return
        t = time.time()
        r = requests.post(url + "/v1/chat/completions", json={"model": os.environ.get("MODEL_NAME","Qwen/Qwen3-8B"), "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": os.environ.get("THINK","false")=="true"},
                          "messages": [{"role": "user", "content": p}]}, timeout=600)
        lat.append((time.time() - t, r.json().get("usage", {}).get("completion_tokens", 0)))
t0 = time.time(); th = [threading.Thread(target=worker) for _ in range(8)]; [t.start() for t in th]; [t.join() for t in th]; wall = time.time() - t0
after = m()
drafts = sm(after, "vllm:spec_decode_num_drafts") - sm(before, "vllm:spec_decode_num_drafts")
dtoks = sm(after, "vllm:spec_decode_num_draft_tokens") - sm(before, "vllm:spec_decode_num_draft_tokens")
acc = sm(after, "vllm:spec_decode_num_accepted_tokens_total") - sm(before, "vllm:spec_decode_num_accepted_tokens_total")
pp_a, pp_b = per_pos(after), per_pos(before); pp = {k: pp_a[k] - pp_b.get(k, 0.0) for k in sorted(pp_a)}
gen = sum(n for _, n in lat)
res = {"name": name, "prompts": len(prompts), "wall_s": wall, "gen_tokens": gen, "tok_per_s": gen / wall if wall else None,
       "drafts": drafts, "draft_tokens": dtoks, "accepted_tokens": acc,
       "accepted_length_incl_bonus": (1 + acc / drafts) if drafts else None,
       "acceptance_rate": (acc / dtoks) if dtoks else None,
       "survival_per_pos": {k: v / drafts for k, v in pp.items()} if drafts else {},
       "conditional_accept_per_pos": {}}
prev = drafts
for k in sorted(pp):
    res["conditional_accept_per_pos"][k] = (pp[k] / prev) if prev else None; prev = pp[k]
json.dump(res, open(out_path, "w"), indent=2)
print(f"[{name}] prompts={len(prompts)} gen_tokens={gen} tok/s={res['tok_per_s']:.1f} drafts={drafts:.0f} "
      f"accepted_len={res['accepted_length_incl_bonus']} accept_rate={res['acceptance_rate']}")
print("  survival per pos   :", {k: round(v, 3) for k, v in res["survival_per_pos"].items()})
print("  conditional per pos:", {k: round(v, 3) for k, v in res["conditional_accept_per_pos"].items() if v is not None})
