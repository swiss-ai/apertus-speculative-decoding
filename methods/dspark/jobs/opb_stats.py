import glob, json, collections, statistics as st
import pyarrow.parquet as pq, pyarrow as pa
S="/capstor/scratch/cscs/zyu/spec"
def pct(c, top=12):
    tot=sum(c.values()); return ", ".join("%s %.1f%%" % (k, 100*v/tot) for k,v in c.most_common(top))
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
