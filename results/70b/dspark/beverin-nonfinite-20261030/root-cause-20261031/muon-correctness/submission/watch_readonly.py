import importlib.util,pathlib,subprocess,json,time
root=pathlib.Path('/home/projects/apertus-speculative-decoding-beverin-nan')
p=root/'results/70b/dspark/beverin-nonfinite-20261030/diagnostic/v2/watch_diagnostic.py';s=importlib.util.spec_from_file_location('watch',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
out=root/'results/70b/dspark/beverin-nonfinite-20261030/root-cause-20261031/muon-correctness/submission/observations.jsonl'
remote=r'''import pathlib,json,subprocess,datetime
p=pathlib.Path('/iopsstor/scratch/cscs/faruk_zahiragic/beverin-feasibility/runs/muon-correctness-20261031T162338Z')
a=subprocess.run(['sacct','-X','-n','-P','-j','710498','-o','JobIDRaw,State,ElapsedRaw,ExitCode,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=10)
d={'utc':datetime.datetime.utcnow().isoformat()+'Z','sacct':a.stdout.decode(),'cells':{}}
for f in sorted(p.glob('deployments/710498/*/rank*.json')):
 r=json.loads(f.read_text());cases=r['cases'];last=cases[-1] if cases else None
 d['cells'].setdefault(f.parent.name,{})[f.stem]={'status':r['status'],'case_count':len(cases),'last':last,'error':r.get('error'),'worst_relative_error':max([c.get('relative_l2_error',0) for c in cases]+[0])}
for f in sorted(p.glob('deployments/710498/*/exit-code.txt')):
 d['cells'].setdefault(f.parent.name,{})['exit_code']=f.read_text().strip()
if (p/'job-710498.out').exists():d['job_tail']=(p/'job-710498.out').read_text()[-1500:]
for f in p.glob('deployments/710498/*/raw.log'):
 t=f.read_text()[-1800:]
 if any(x in t for x in ('Error:','Exception:','AssertionError')):d['cells'].setdefault(f.parent.name,{})['log_tail']=t
print(json.dumps(d))'''
while True:
 try:
  a=subprocess.run(m.SSH,input=remote,text=True,capture_output=True,timeout=45)
  if a.returncode:raise RuntimeError(str(a.returncode)+' '+a.stderr[-800:])
  d=json.loads(a.stdout)
  with out.open('a') as f:f.write(json.dumps(d)+'\n')
  print(json.dumps(d),flush=True)
  if any(x in d['sacct'] for x in ['|COMPLETED|','|FAILED|','|TIMEOUT|','|CANCELLED|','|OUT_OF_MEMORY|']):break
 except Exception as e:
  print('TRANSPORT_UNCERTAINTY '+str(e),flush=True)
  if any(x in str(e).lower() for x in ['permission denied','authentication failed','certificate invalid']):break
 time.sleep(45)
