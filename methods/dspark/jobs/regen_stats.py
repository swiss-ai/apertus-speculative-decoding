import json, statistics as st, sys
p="/capstor/scratch/cscs/zyu/spec/runs/apertus_p2/regen/magpie_nothink_100k.jsonl"
f=open(p); first=json.loads(f.readline()); f.seek(0)
def shape(v, depth=0):
    if isinstance(v, dict): return "{" + ", ".join("%s: %s" % (k, shape(x, depth+1)) for k,x in list(v.items())[:8]) + "}"
    if isinstance(v, list): return "[%d x %s]" % (len(v), shape(v[0], depth+1) if v else "?")
    if isinstance(v, str): return "str(%d)" % len(v)
    return type(v).__name__
print("row structure:", shape(first)[:700])
# find assistant text generically
def assistant_texts(d):
    out=[]
    for key in ("conversations","messages","conversation","turns"):
        if isinstance(d.get(key), list):
            for m in d[key]:
                if isinstance(m, dict) and (m.get("from") or m.get("role")) in ("gpt","assistant","model"):
                    out.append(m.get("value") or m.get("content") or m.get("text") or "")
    for key in ("response","output","completion","answer"):
        if isinstance(d.get(key), str): out.append(d[key])
    return out
R=[]; n=0; multi=0
for l in f:
    d=json.loads(l); n+=1; t=assistant_texts(d); R.extend(len(x) for x in t)
    if len(t)>1: multi+=1
print("rows %d, rows with >1 assistant turn %d, assistant texts %d" % (n, multi, len(R)))
if R: print("Apertus response chars: median %d, mean %d, p90 %d (~%d tokens mean at 3.8 chars/token)" % (st.median(R), st.mean(R), sorted(R)[int(0.9*len(R))], st.mean(R)/3.8))
print("first row user text:", (next((m.get("value") or m.get("content") for m in (first.get("conversations") or first.get("messages") or []) if (m.get("from") or m.get("role")) in ("human","user")), "?") or "?")[:200].replace("\n"," "))
