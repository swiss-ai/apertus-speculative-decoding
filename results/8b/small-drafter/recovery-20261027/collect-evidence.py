from pathlib import Path
import datetime,gzip,hashlib,json,shutil,subprocess,tarfile
base=Path('/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp-data/small-drafter')
repo=Path('/iopsstor/scratch/cscs/faruk_zahiragic/spec-exp/small-drafter')
out=base/'collections'/'recovery-20261027'
if out.exists(): raise SystemExit('collection exists; choose a new version')
out.mkdir(parents=True)
entries=[]
def collect(src,rel):
 if not src.is_file(): return
 dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True)
 h=hashlib.sha256()
 with src.open('rb') as handle:
  while block:=handle.read(1<<20): h.update(block)
 compressed=src.stat().st_size>200000 and src.suffix in ['.log','.out','.csv']
 if compressed:
  dst=Path(str(dst)+'.gz')
  with src.open('rb') as a,gzip.open(dst,'wb') as b:shutil.copyfileobj(a,b)
 else:shutil.copy2(src,dst)
 entries.append({'source':str(src),'artifact':str(dst.relative_to(out)),'source_bytes':src.stat().st_size,'source_sha256':h.hexdigest(),'gzip':compressed})
for name,job in [('depth-l5-scratch','3722176'),('depth-l1-warm','3722177')]:
 root=base/'runs'/name
 for p in root.rglob('*'):
  if p.suffix not in ['.json','.yaml','.txt','.patch','.log','.csv']:continue
  collect(p,Path(name)/p.relative_to(root))
 collect(repo/f'results/8b/dspark/logs/{job}.out',Path(name)/f'slurm-{job}.out')
 for p in (repo/'results/8b/dspark/logs').glob('submit-train-*'):
  if f'RUN_ID={name}\n' in p.read_text():collect(p,Path(name)/p.name)
for name in ['config.json','truncation.json','source-train_command.txt','source-speculators.patch','source-run.yaml','source-val_metrics.json','source-training_state.json']:
 collect(base/'drafters/dspark-e10-trunc-l1'/name,Path('warm-start')/name)
collect(repo/'targets/8b/contract.json',Path('inputs/target-contract.json'))
for p in (base/'data').glob('*'):
 if p.suffix in ['.json','.yaml'] and p.stat().st_size<500000:collect(p,Path('inputs/data')/p.name)
metadata={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Recover immutable evidence for completed five-layer control and failed one-layer warm-start before relaunch','collector_source':__file__,'repo_revision':subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),'slurm_accounting':subprocess.check_output(['sacct','-X','-n','-P','-j','3722176,3722177','--format=JobIDRaw,State,ExitCode,Start,End,Elapsed,NodeList'],text=True),'entries':entries}
(out/'collection-manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
archive=out.with_suffix('.tar.gz')
with tarfile.open(archive,'w:gz') as t:t.add(out,arcname=out.name)
print(json.dumps({'collection':str(out),'archive':str(archive),'bytes':archive.stat().st_size,'files':len(entries)}))
