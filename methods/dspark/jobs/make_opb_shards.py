# Shuffle Open-PerfectBlend once (seed 0), drop rows without a leading human turn, write N JSONL shards in the
# speculators conversation format (from/value). Nested subsets = prefixes of the shuffled order.
import glob, json, random, collections, os, sys
import pyarrow.parquet as pq
S="/capstor/scratch/cscs/zyu/spec"; N=int(sys.argv[1]) if len(sys.argv)>1 else 4
OUT=S+"/runs/apertus_opb/prompts"; os.makedirs(OUT, exist_ok=True)
files=sorted(glob.glob(S+"/hf/hub/datasets--mlabonne--open-perfectblend/snapshots/*/**/*.parquet", recursive=True))
rows=[]; dropped=collections.Counter()
for f in files:
    d=pq.read_table(f).to_pydict()
    for conv, src in zip(d["conversations"], d["source"]):
        msgs=[{"from": m.get("from"), "value": m.get("value")} for m in conv]
        if len(msgs)<2 or msgs[0]["from"] not in ("human","user") or not msgs[0]["value"]: dropped["short_or_no_human_first"]+=1; continue
        if msgs[0]["from"]=="system": dropped["system_first"]+=1; continue
        rows.append({"source": src, "conversations": msgs})
print("kept", len(rows), "dropped", dict(dropped))
random.Random(0).shuffle(rows)
for i,r in enumerate(rows): r["id"]="opb-%07d" % i
per=(len(rows)+N-1)//N; manifest=[]
for k in range(N):
    part=rows[k*per:(k+1)*per]; p=f"{OUT}/opb_shuffled_seed0_shard{k}of{N}.jsonl"
    with open(p,"w") as f:
        for r in part: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    c=collections.Counter(r["source"] for r in part); manifest.append({"shard": p, "rows": len(part), "sources": c})
    print(p, len(part), "rows")
json.dump({"seed": 0, "shards": N, "total": len(rows), "dropped": dropped, "manifest": manifest}, open(OUT+"/MANIFEST.json","w"), indent=1)
turns=collections.Counter(min(len(r["conversations"]),8) for r in rows); print("messages per conv:", dict(sorted(turns.items())))
print("assistant turns total:", sum(sum(1 for m in r["conversations"] if m["from"] in ("gpt","assistant")) for r in rows))
