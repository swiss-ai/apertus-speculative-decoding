import glob, json, sys
import pyarrow.parquet as pq
D="/capstor/store/cscs/swissai/infra01/datasets"
def show(files, label, n=1, maxlen=900):
    print("==", label, len(files), "files")
    if not files: return
    t=pq.read_table(files[0]); print("  rows in file:", t.num_rows, "| schema:", [(f.name, str(f.type)[:45]) for f in t.schema])
    for r in t.slice(0,n).to_pylist():
        for k,v in r.items():
            s=v if isinstance(v,str) else json.dumps(v, ensure_ascii=False)
            print("   %s: %s" % (k, s[:maxlen].replace(chr(10)," ")))
show(sorted(glob.glob(D+"/Apertus-1.5-SFT-mix-pretrain-v1-by-source/data/toucan-1-5m-7b5ecad0b6d5/**/*.parquet", recursive=True)), "Toucan (T)")
show(sorted(glob.glob(D+"/Apertus-1.5-SFT-mix-pretrain-v1-by-source/data/nvidia-openmathreasoning-810823a3af56/**/*.parquet", recursive=True)), "OpenMathReasoning (R)")
show(sorted(glob.glob(D+"/superior-reasoning-apertus-inner-v1/*.parquet")), "superior-reasoning-apertus-inner-v1")
