import pathlib,json,hashlib,collections
root=pathlib.Path(__file__).resolve().parents[2]
p=root/'comm-correctness/results-710311';collection=json.loads((p/'collection.json').read_text())
for item in collection['files']:
 b=(p/'raw'/item['path']).read_bytes();assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
summary={'raw_files_verified':len(collection['files']),'raw_bytes':sum(i['bytes'] for i in collection['files']),'cells':{}}
for cell in ['upstream-default','upstream-p2p-disabled','alps-default']:
 c=p/'raw/deployments/710311'/cell;rows=[json.loads(f.read_text()) for f in sorted(c.glob('rank*.json'))]
 assert sorted(r['rank'] for r in rows)==list(range(4))
 for r in rows:
  assert r['status']=='passed' and r['total_checks']==1536 and len(r['checks'])==1536
  assert all(x['mismatches']==0 and x['nonfinite']==0 and not x['local_mismatch_detected'] for x in r['checks'])
  assert r['probe_source_sha256']=='667943d8116d8a960089a9983fab4a38bbace79456c2df2c7bb2415cf40271aa'
 counter=collections.Counter()
 for line in (c/'raw.log').read_text().splitlines():
  if 'via ' in line:
   for kind in ['P2P/IPC','SHM','NET/']:
    if kind in line:counter[kind]+=1
 ops=collections.Counter(x['operation'] for r in rows for x in r['checks'])
 summary['cells'][cell]={'rank_count':4,'checks':sum(len(r['checks']) for r in rows),'elements_checked':sum(r['total_elements_checked'] for r in rows),'checks_by_operation':dict(ops),'nonfinite':0,'mismatches':0,'rccl':rows[0]['nccl_or_rccl_version'],'torch':rows[0]['torch'],'torch_git':rows[0]['torch_git_revision'],'p2p_disable':rows[0]['environment'].get('NCCL_P2P_DISABLE'),'transport_log_line_counts':dict(counter),'rank_elapsed_seconds':[r['elapsed_seconds'] for r in rows]}
summary['total_checks']=sum(c['checks'] for c in summary['cells'].values())
summary['total_element_comparisons']=sum(c['elements_checked'] for c in summary['cells'].values())
(p/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
