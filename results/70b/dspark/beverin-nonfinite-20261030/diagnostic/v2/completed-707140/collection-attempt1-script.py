#!/usr/bin/env python3
"""Stream completed evidence from explicit Beverin SSH; never write remotely."""
import hashlib,importlib.util,json,pathlib,subprocess,sys,tarfile
root=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("watch",str(root.parent/"watch_diagnostic.py"))
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
REMOTE=r'''
import pathlib,subprocess,json,tarfile,hashlib,io,time,os,struct,math
base=pathlib.Path('/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility');run=base/'runs/opb-thinkoff-32n-nan-debug-20261030-v2';old=base/'runs/opb-thinkoff-32n-1epoch-beverin-20261027';job='707140'
accounting=subprocess.check_output(['sacct','-j',job,'-P','-o','JobIDRaw,State,ElapsedRaw,ExitCode,Start,End,NodeList%1000'],env=dict(os.environ,SLURM_TIME_FORMAT='standard')).decode()
assert '707140|COMPLETED|' in accounting,accounting
files={}
for source,label in [(run/'deployments'/job,'deployment'),(run/'logs','logs'),(run/'launch-provenance','launch-provenance'),(base/'repo-556a464d','runtime-source')]:
 for p in source.rglob('*'):
  if p.is_file() and not p.is_symlink() and '__pycache__' not in p.parts and p.suffix!='.pyc':files[label+'/'+str(p.relative_to(source))]=p
for p in (base/'logs').glob('*707140*'):
 if p.is_file() and not p.is_symlink():files['external-logs/'+p.name]=p
for p in (run/'checkpoints').rglob('*'):
 if p.is_file() and not p.is_symlink() and p.stat().st_size<4*1024*1024:files['checkpoint-metadata/'+str(p.relative_to(run/'checkpoints'))]=p
for p in (old/'deployments/698072').iterdir():
 if p.is_file() and not p.is_symlink() and p.stat().st_size<4*1024*1024:files['original-reference/deployment/'+p.name]=p
for p in (old/'logs').glob('train-698072-node*.log'):
 if p.is_file():files['original-reference/logs/'+p.name]=p
spec=base/'speculators-7046ac6-provenance'
for name in ['train/cli.py','train/config/schema.py','train/distributed_batch_sampler.py','train/__main__.py','models/utils.py']:
 p=spec/'src/speculators'/name;files['speculators-source/'+name]=p
path=run/'checkpoints/0/model.safetensors';before=path.stat()
with path.open('rb') as f:
 n=struct.unpack('<Q',f.read(8))[0];header_raw=f.read(n);header=json.loads(header_raw);start=8+n;tensors={};total=0;bad=0
 for name,info in sorted(header.items()):
  if name=='__metadata__':continue
  assert info['dtype']=='BF16',(name,info['dtype'])
  a,b=info['data_offsets'];count=(b-a)//2;indices=set()
  for offset in [0,count//2,max(0,count-64)]:indices.update(range(offset,min(count,offset+64)))
  values=[]
  for i in sorted(indices):
   f.seek(start+a+2*i);bits=struct.unpack('<H',f.read(2))[0];values.append([i,bits])
  nonfinite=sum(1 for _,bits in values if (bits&0x7f80)==0x7f80)
  finite=[struct.unpack('<f',struct.pack('<I',bits<<16))[0] for _,bits in values if (bits&0x7f80)!=0x7f80]
  tensors[name]={'shape':info['shape'],'dtype':info['dtype'],'samples':len(values),'nonfinite':nonfinite,'max_abs_sample':max(map(abs,finite)) if finite else None,'index_bf16_bits':values}
  total+=len(values);bad+=nonfinite
sample={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'path':str(path),'file_size':before.st_size,'file_mtime_ns':before.st_mtime_ns,'total_sampled':total,'total_nonfinite':bad,'tensor_count':len(tensors),'method':'Up to64BF16values beginning atstart/midpoint/endofeachtensor, deduplicated. Exponent mask0x7f80. Sampleonly; nofullscanorwholeweightfilehash.','tensors':tensors}
assert before.st_size==path.stat().st_size and before.st_mtime_ns==path.stat().st_mtime_ns
extras={'sacct.txt':accounting.encode(),'checkpoint-sample.json':json.dumps(sample,indent=2).encode(),'checkpoint-model-header.json':header_raw,'terminal.json':json.dumps({'failure_marker_exists':(run/'checkpoints/numerical-failure.json').exists(),'capture_files':[str(p) for p in (run/'checkpoints/numerical-failure-batches').glob('*.pt')],'final_training_state':json.loads((run/'checkpoints/0/training_state.json').read_text())}).encode()}
manifest={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'run':str(run),'job_id':job,'remote_mutations':False,'files':[]}
with tarfile.open(fileobj=__import__('sys').stdout.buffer,mode='w|') as archive:
 for name,p in sorted(files.items()):
  raw=p.read_bytes();manifest['files'].append({'archive_path':name,'cluster_path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=int(p.stat().st_mtime);archive.addfile(info,io.BytesIO(raw))
 extras['collection.json']=json.dumps(manifest,indent=2).encode();extras['SOURCE_SHA256SUMS']=''.join(x['sha256']+'  '+x['archive_path']+'\n' for x in manifest['files']).encode()
 for name,raw in sorted(extras.items()):
  info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=int(time.time());archive.addfile(info,io.BytesIO(raw))
'''
archive=pathlib.Path('/tmp/beverin-nan-707140-completed-20261031.tar')
with archive.open("xb") as stream:
    result=subprocess.run(watch.SSH,input=REMOTE.encode(),stdout=stream,stderr=subprocess.PIPE,timeout=120)
if result.returncode:
    print(result.stderr.decode(),file=sys.stderr)
    raise SystemExit(result.returncode)
raw=root/"raw"
raw.mkdir(exist_ok=False)
with tarfile.open(archive) as stream:
    for info in stream.getmembers():
        assert not pathlib.Path(info.name).is_absolute() and ".." not in pathlib.Path(info.name).parts
    stream.extractall(raw,filter="data")
manifest=json.loads((raw/"collection.json").read_text())
for item in manifest["files"]:
    assert hashlib.sha256((raw/item["archive_path"]).read_bytes()).hexdigest()==item["sha256"],item["archive_path"]
print(json.dumps({"files":len(manifest["files"]),"bytes":archive.stat().st_size,"archive":str(archive),"raw":str(raw),"all_source_hashes_verified":True}))
