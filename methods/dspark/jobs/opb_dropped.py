import glob, collections, json
import pyarrow.parquet as pq
S="/capstor/scratch/cscs/zyu/spec"
files=sorted(glob.glob(S+"/hf/hub/datasets--mlabonne--open-perfectblend/snapshots/*/**/*.parquet", recursive=True))
why=collections.Counter(); bysrc=collections.Counter(); ex=[]; total=0
for f in files:
    d=pq.read_table(f).to_pydict()
    for conv, src in zip(d["conversations"], d["source"]):
        total+=1
        if len(conv)<2: r="only one message"
        elif conv[0].get("from") not in ("human","user"): r="first message is "+str(conv[0].get("from"))
        elif not conv[0].get("value"): r="empty first message"
        else: continue
        why[r]+=1; bysrc[src]+=1
        if len(ex)<4 and r not in [e[0] for e in ex]: ex.append((r, src, json.dumps([{k:(v[:120] if isinstance(v,str) else v) for k,v in m.items()} for m in conv[:2]], ensure_ascii=False)[:400]))
print("total rows", total, "| dropped", sum(why.values())); print("reasons:", dict(why)); print("by source:", dict(bysrc))
for r,s,e in ex: print("  [%s] %s: %s" % (r, s, e))
