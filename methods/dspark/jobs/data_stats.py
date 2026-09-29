import glob, json, collections, statistics as st
import pyarrow.parquet as pq, pyarrow as pa
S="/capstor/scratch/cscs/zyu/spec"
def pct(c, top=12):
    tot=sum(c.values()); return ", ".join("%s %.1f%%" % (k, 100*v/tot) for k,v in c.most_common(top))
# ---- Magpie (parquet from the hub snapshot)
mf=sorted(glob.glob(S+"/hf/hub/datasets--Magpie-Align--Magpie-Llama-3.1-Pro-300K-Filtered/snapshots/*/**/*.parquet", recursive=True))
cols=[c for c in ("task_category","difficulty","language","input_quality","instruction_length","instruction") if c in pq.read_schema(mf[0]).names]
mg=pa.concat_tables([pq.read_table(f, columns=cols) for f in mf])
print("MAGPIE rows:", mg.num_rows, "| columns used:", cols)
sub={c: mg.slice(0,100000).column(c).to_pylist() for c in cols}
for col in ("task_category","difficulty","language","input_quality"):
    if col in sub: print("  %-14s first100k: %s" % (col, pct(collections.Counter(sub[col]), 10)))
il=sub["instruction_length"] if "instruction_length" in sub else [len(x) for x in sub["instruction"]]
print("  instruction chars: median %d, mean %d, p90 %d" % (st.median(il), st.mean(il), sorted(il)[int(0.9*len(il))]))
print("  task_category all300k:", pct(collections.Counter(mg.column("task_category").to_pylist()), 10))
R=[]; multi=0; n=0
for l in open(S+"/runs/apertus_p2/regen/magpie_nothink_100k.jsonl"):
    d=json.loads(l); n+=1; msgs=d.get("conversations") or d.get("messages") or []
    if len(msgs)>2: multi+=1
    for m in msgs:
        if (m.get("from") or m.get("role")) in ("gpt","assistant"): R.append(len(m.get("value") or m.get("content") or ""))
print("  regenerated: rows %d, multi-turn %d, Apertus response chars median %d mean %d p90 %d (~%d tokens mean)" % (n, multi, st.median(R), st.mean(R), sorted(R)[int(0.9*len(R))], st.mean(R)/3.8))
# ---- Open-PerfectBlend
files=sorted(glob.glob(S+"/hf/hub/datasets--mlabonne--open-perfectblend/snapshots/*/**/*.parquet", recursive=True)); print("OPB parquet files:", len(files))
print("  columns:", pq.read_schema(files[0]).names)
src=collections.Counter(); turns=collections.Counter(); plen=[]; rows=0; ex=[]
for f in files:
    t=pq.read_table(f); d=t.to_pydict(); rows+=t.num_rows
    conv_col=[c for c in d if c in ("conversations","messages")][0]
    for i,conv in enumerate(d[conv_col]):
        s=d["source"][i] if "source" in d else "?"; src[s]+=1
        turns[min(len(conv),6)]+=1
        first=next((m.get("value") or m.get("content") or "" for m in conv if (m.get("from") or m.get("role")) in ("human","user")), ""); plen.append(len(first))
        if len(ex)<4 and i%3000==0: ex.append((s, first[:150].replace("\n"," ")))
print("  rows:", rows); print("  source:", pct(src, 12)); print("  messages per conversation (6 = 6+):", pct(turns, 6))
print("  first user prompt chars: median %d, mean %d, p90 %d" % (st.median(plen), st.mean(plen), sorted(plen)[int(0.9*len(plen))]))
for s,e in ex: print("  ex [%s]: %s" % (s, e))
