import glob, json, collections, statistics as st
import pyarrow.parquet as pq
S="/capstor/scratch/cscs/zyu/spec"
snap=glob.glob(S+"/hf/hub/datasets--Magpie-Align--Magpie-Llama-3.1-Pro-300K-Filtered/snapshots/*/")[0]
files=sorted(glob.glob(snap+"**/*.parquet", recursive=True)); print("parquet files:", len(files))
cols=["task_category","difficulty","language","input_quality","instruction_length","response_length","instruction"]
tabs=[pq.read_table(f, columns=[c for c in cols if c in pq.read_schema(f).names]) for f in files]
import pyarrow as pa
t=pa.concat_tables(tabs); print("rows total:", t.num_rows)
N=100000; sub=t.slice(0,N).to_pydict()
def dist(key, top=12):
    c=collections.Counter(sub[key]); tot=sum(c.values())
    return ", ".join("%s %.1f%%" % (k, 100*v/tot) for k,v in c.most_common(top))
print("task_category (first 100k):", dist("task_category"))
print("difficulty:", dist("difficulty"))
print("language:", dist("language", 8))
print("input_quality:", dist("input_quality"))
il=sub["instruction_length"]; print("instruction chars: median %d, mean %d, p90 %d" % (st.median(il), st.mean(il), sorted(il)[int(0.9*len(il))]))
# regenerated responses
R=[]; multi=0; n=0
for l in open(S+"/runs/apertus_p2/regen/magpie_nothink_100k.jsonl"):
    d=json.loads(l); n+=1; msgs=d.get("conversations") or d.get("messages") or []
    if len(msgs)>2: multi+=1
    for m in msgs:
        if (m.get("from") or m.get("role")) in ("gpt","assistant"): R.append(len(m.get("value") or m.get("content") or ""))
print("regenerated rows %d, multi-turn %d, Apertus response chars: median %d, mean %d, p90 %d -> ~%d tokens mean" % (n, multi, st.median(R), st.mean(R), sorted(R)[int(0.9*len(R))], st.mean(R)/3.8))
# first-100k vs whole-set category shares (is the file ordered?)
whole=collections.Counter(t.to_pydict()["task_category"]); tw=sum(whole.values())
print("task_category (all 300k):", ", ".join("%s %.1f%%" % (k, 100*v/tw) for k,v in whole.most_common(12)))
