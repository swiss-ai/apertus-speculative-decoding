#!/usr/bin/env python3
"""Read only small contiguous BF16 windows; no full checkpoint scan or hash."""
import importlib.util,json,pathlib,subprocess,sys
root=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("watch",str(root.parent/"watch_diagnostic.py"));watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
REMOTE=r'''
import pathlib,struct,json,os,time,math
path=pathlib.Path('/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/opb-thinkoff-32n-nan-debug-20261030-v2/checkpoints/0/model.safetensors');before=path.stat();t=time.monotonic();tensors={};total=bad=read_bytes=0
with path.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n));start=8+n
 for name,info in sorted(header.items()):
  if name=='__metadata__':continue
  assert info['dtype']=='BF16'
  a,b=info['data_offsets'];count=(b-a)//2;values={}
  for offset in sorted({0,count//2,max(0,count-64)}):
   size=min(64,count-offset)*2;raw=os.pread(f.fileno(),size,start+a+2*offset);read_bytes+=len(raw);assert len(raw)==size
   for i,packed in enumerate(struct.iter_unpack('<H',raw)):values[offset+i]=packed[0]
  nonfinite=sum((bits&0x7f80)==0x7f80 for bits in values.values());finite=[struct.unpack('<f',struct.pack('<I',bits<<16))[0] for bits in values.values() if (bits&0x7f80)!=0x7f80];total+=len(values);bad+=nonfinite
  tensors[name]={'shape':info['shape'],'dtype':info['dtype'],'samples':len(values),'nonfinite':nonfinite,'max_abs_sample':max(map(abs,finite)) if finite else None,'index_bf16_bits':sorted(values.items())}
after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
print(json.dumps({'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'path':str(path),'file_size':before.st_size,'file_mtime_ns':before.st_mtime_ns,'tensor_count':len(tensors),'total_sampled':total,'total_nonfinite':bad,'requested_data_bytes':read_bytes,'elapsed_seconds':time.monotonic()-t,'method':'At most64BF16values beginning atstart/midpoint/endofeachtensor; deduplicated. Read incontiguoussmallwindows withpread. Nonfinite exponentmask0x7f80. Sampleonly, nofullscanorwholeweightfilehash.','tensors':tensors,'header':header}))
'''
r=subprocess.run(watch.SSH,input=REMOTE,text=True,capture_output=True,timeout=90)
if r.returncode:raise RuntimeError(r.stderr)
sample=json.loads(r.stdout)
with (root/'checkpoint-sample.json').open('x') as stream:stream.write(json.dumps(sample,indent=2)+'\n')
print(json.dumps({k:sample[k] for k in ('tensor_count','total_sampled','total_nonfinite','requested_data_bytes','elapsed_seconds')}))
