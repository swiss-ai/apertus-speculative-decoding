import glob, re, os, statistics as st, sys
O=sys.argv[1] if len(sys.argv)>1 else "/capstor/scratch/cscs/zyu/spec/runs/apertus_dspark_bench"
KEYS=("profile/fetch_ms","profile/fwd_ms","profile/bwd_ms","profile/opt_ms","profile/step_ms","profile/tokens_per_s","profile/fetch_frac","epoch","step")
def records(text):
    toks=re.findall(r"(profile/[a-z_]+|epoch|step)=([0-9.e+-]+)", text); rec={}; out=[]
    for k,v in toks:
        rec[k]=float(v)
        if k=="step": out.append(rec); rec={}
    return [r for r in out if all(k in r for k in KEYS)]
print("%-22s %5s %8s %8s %7s %7s %7s %7s %7s %6s %8s" % ("variant","ranks","tok/s/rk","tok/s","fetch","fwd","bwd","opt","step","steps","val_eal"))
for f in sorted(glob.glob(O+"/train_*.log")):
    s=open(f, errors="ignore").read(); name=os.path.basename(f)[6:-4]
    rows=records(s)
    if not rows: print("%-22s no profile lines" % name); continue
    later=[r for r in rows if r["step"]>=1] or rows
    m=lambda k: st.median(r[k] for r in later)
    rk=f[:-4]+".ranks"; ranks=int(open(rk).read().strip()) if os.path.exists(rk) else 0
    eal=re.findall(r"val/eal_epoch=([0-9.]+)", s)
    print("%-22s %5s %8.0f %8.0f %7.1f %7.1f %7.1f %7.1f %7.1f %6d %8s" % (name, ranks or "?", m("profile/tokens_per_s"), m("profile/tokens_per_s")*(ranks or 1), m("profile/fetch_ms"), m("profile/fwd_ms"), m("profile/bwd_ms"), m("profile/opt_ms"), m("profile/step_ms"), len(rows), ",".join(eal) if eal else "-"))
