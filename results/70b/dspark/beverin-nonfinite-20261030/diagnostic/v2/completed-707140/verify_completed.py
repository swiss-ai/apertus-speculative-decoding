#!/usr/bin/env python3
"""Verify collected evidence without importing model code or touching cluster jobs."""
import collections,datetime,hashlib,io,json,math,pathlib,pickle,re,statistics,zipfile
root=pathlib.Path(__file__).resolve().parent
raw=root/'raw'
def records(path):
    result=[]
    for match in re.finditer(r'train/confidence_loss=.*?global_step=(\d+)',path.read_text(),re.S):
        values={key:float(value) for key,value in re.findall(r'((?:train|lr|profile)/[A-Za-z0-9_]+)=([\deE.+-]+|nan|inf)',match.group())}
        result.append(dict(values,step=int(match.group(1))))
    return result
new=records(raw/'logs/train-707140-node0.log');old=records(raw/'original-reference/logs/train-698072-node0.log')
oldmap={r['step']:r for r in old};newmap={r['step']:r for r in new};checks={}
checks['640_contiguous_records']=[r['step'] for r in new]==list(range(640))
checks['all_recorded_values_finite']=all(math.isfinite(v) for r in new for v in r.values())
checks['no_dropped_records']=all(r['train/error_records']==0 for r in new)
terminal=json.loads((raw/'terminal.json').read_text());state=terminal['final_training_state']
checks['checkpoint_0_640_640']=state=={'epoch':0,'local_step':640,'global_step':640}
checks['no_failure_marker_or_captures']=not terminal['failure_marker_exists'] and not terminal['capture_files']
audit=json.loads((raw/'deployment/job-audit.json').read_text());checks['audit_ok_single_attempt']=audit['status']=='ok' and audit['return_code']==0 and not audit['errors'] and len(audit['attempts'])==1
hooks=[];nodes=[]
for path in sorted((raw/'logs').glob('train-707140-node*.log')):
    found=[json.loads(x) for x in re.findall(r'BEVERIN_HOOKS (\{[^\n]+\})',path.read_text())]
    nodes.append({'log':str(path.relative_to(root)),'hook_records':found});hooks.extend(found)
checks['64_guard_hooks_on_32_nodes']=len(nodes)==32 and all(len(x['hook_records'])==2 for x in nodes) and len(hooks)==64 and all(x.get('numerical_guard')=='1' and x['reduce_dtype']=='fp32' and x['blas_backend']=='_BlasBackend.Cublas' and x['profile_steps']==0 and x['segment_steps']==640 for x in hooks)
envs=list((raw/'deployment').glob('node*/resolved-environment.txt'))
checks['activation_capture_flags_all_nodes']=len(envs)==32 and all(all(k+'=1' in p.read_text().splitlines() for k in ['BEVERIN_CHECK_ACTIVATIONS','BEVERIN_CAPTURE_BAD_BATCH','BEVERIN_NUMERICAL_GUARD']) for p in envs)
log=(raw/'logs/train-707140-node0.log').read_text();starts=[json.loads(x) for x in re.findall(r'BEVERIN_SAVE_START (\{[^\n]+\})',log)];ends=[json.loads(x) for x in re.findall(r'BEVERIN_SAVE_END (\{[^\n]+\})',log)]
checks['all_ten_saves_finished']=collections.Counter(x['global_step'] for x in starts)==collections.Counter({step:2 for step in range(64,641,64)})==collections.Counter(x['global_step'] for x in ends)
class Restricted(pickle.Unpickler):
    def find_class(self,module,name):raise ValueError((module,name))
with zipfile.ZipFile(raw/'checkpoint-metadata/0/scheduler_state_dict.pt') as stream:
    name=next(n for n in stream.namelist() if n.endswith('/data.pkl'));schedulers=Restricted(io.BytesIO(stream.read(name))).load()
peak=8.48528137423857e-4;expected=peak*(4077-640)/(4077-40)
checks['two_schedulers_at_640_match_horizon']=len(schedulers)==2 and all(s['last_epoch']==640 and s['_step_count']==641 and s['base_lrs']==[peak] and math.isclose(s['_last_lr'][0],expected,rel_tol=1e-12) for s in schedulers)
sample=json.loads((root/'checkpoint-sample.json').read_text());checks['checkpoint_sample_finite']=sample['tensor_count']==62 and sample['total_sampled']==11073 and sample['total_nonfinite']==0
manifest=json.loads((raw/'collection.json').read_text());checks['raw_file_hashes_match']=all(hashlib.sha256((raw/x['archive_path']).read_bytes()).hexdigest()==x['sha256'] for x in manifest['files'])
revision=(raw/'deployment/repo-revision.txt').read_text().strip();checks['runtime_revision_matches']=revision=='556a464dfbfe7ba56699ad037bf952b15c4e6b4e'
for line in (raw/'launch-provenance/source.sha256').read_text().splitlines():
    sha,name=line.split('  ',1)
    assert hashlib.sha256((raw/'runtime-source'/name).read_bytes()).hexdigest()==sha,name
checks['immutable_runtime_source_hashes_match']=True
warnings={key:[] for key in ['numerical','file_lock_retries','connection_retries','telemetry_quota_errors','shutdown_engine_errors','other_exceptions','compile']}
patterns={'numerical':r'non.?finite|(?:=|:|\s)(?:nan|inf|infinity)(?=$|[,;:\s])|overflow|invalid value','file_lock_retries':r'Hidden-state round trip failed','connection_retries':r'Request aborted','telemetry_quota_errors':r'OSError: \[Errno 122\] Disk quota exceeded','shutdown_engine_errors':r'EngineDeadError:','other_exceptions':r'RuntimeError:|Fatal Python error|Segmentation fault|Bus error','compile':r'graph break|recompil|torch\._dynamo'}
for path in sorted((raw/'logs').glob('*.log')):
    for number,line in enumerate(path.read_text(errors='replace').splitlines(),1):
        for key,pattern in patterns.items():
            if re.search(pattern,line,re.I):warnings[key].append({'path':str(path.relative_to(root)),'line':number,'text':line[:1000]})
checks['no_numerical_warning']=not warnings['numerical']
checks['shutdown_errors_after_final_save']=all('10-30 21:31:' in x['text'] for x in warnings['shutdown_engine_errors'])
(root/'warning-audit.json').write_text(json.dumps(warnings,indent=2)+'\n')
keys=[k for k in new[0] if k.startswith(('train/','lr/'))]
firstdiff=next({'step':step,'differences':{key:[oldmap[step][key],newmap[step][key]] for key in keys if oldmap[step][key]!=newmap[step][key]}} for step in range(519) if any(oldmap[step][key]!=newmap[step][key] for key in keys))
oldpatch=(raw/'original-reference/deployment/speculators.patch').read_text();newpatch=(raw/'deployment/speculators.patch').read_text()
normalize=lambda text:'\n'.join(line for line in text.splitlines() if not line.startswith('# repo:')).rstrip('\n')
oldconfig=raw/'original-reference/deployment/checkpoint-config.json'
configold=json.loads((oldconfig if oldconfig.is_file() else root.parents[2]/'checkpoint-metadata/0/config.json').read_text())
newconfig=json.loads((raw/'checkpoint-metadata/0/config.json').read_text())
comparisons={'container_digest_equal':(raw/'original-reference/deployment/container.sha256').read_bytes()==(raw/'deployment/container.sha256').read_bytes(),'dataset_metadata_hashes_equal':(raw/'original-reference/deployment/dataset-metadata.sha256').read_bytes()==(raw/'deployment/dataset-metadata.sha256').read_bytes(),'speculators_revision_equal':(raw/'original-reference/deployment/speculators-revision.txt').read_bytes()==(raw/'deployment/speculators-revision.txt').read_bytes(),'speculators_patch_equal_after_provenance_header_and_newline_normalization':normalize(oldpatch)==normalize(newpatch),'checkpoint_model_configs_equal':configold==newconfig,'seed_original':re.search(r'seed=(\d+)',(raw/'original-reference/logs/train-698072-node0.log').read_text()).group(1),'seed_diagnostic':re.search(r'seed=(\d+)',log).group(1),'deterministic_original':re.search(r'deterministic_cuda=(\d+)',(raw/'original-reference/logs/train-698072-node0.log').read_text()).group(1),'deterministic_diagnostic':re.search(r'deterministic_cuda=(\d+)',log).group(1),'first_logged_training_difference':firstdiff}
def expand_nodes(value):
    if '[' not in value:return {value}
    prefix,body=value.split('[',1);body=body.rstrip(']');result=set()
    for part in body.split(','):
        if '-' in part:
            begin,end=part.split('-',1);stride=1
            if ':' in end:end,stride=end.split(':');stride=int(stride)
            result.update(prefix+str(n).zfill(len(begin)) for n in range(int(begin),int(end)+1,stride))
        else:result.add(prefix+part)
    return result
oldslurm=(raw/'original-reference/deployment/slurm-job.txt').read_text()
oldnames=expand_nodes(re.search(r'\bNodeList=(\S+)',oldslurm).group(1))
newnames={p.read_text().strip() for p in (raw/'deployment').glob('node*/hostname.txt')}
comparisons['original_node_count']=len(oldnames);comparisons['diagnostic_node_count']=len(newnames);comparisons['shared_node_count']=len(oldnames&newnames)
oldyaml=(root.parents[2]/'checkpoint-metadata/run.yaml').read_text();newyaml=(raw/'deployment/run.yaml').read_text()
comparisons['trainer_yaml_equal_after_output_path_normalization']=oldyaml.replace('opb-thinkoff-32n-1epoch-beverin-20261027','opb-thinkoff-32n-nan-debug-20261030-v2')==newyaml
steady=[r['profile/step_ms']/1000 for r in new if r['step']!=0 and r['step']%64!=0]
clean_record=lambda r:{k:(v if math.isfinite(v) else str(v)) for k,v in r.items()}
result={'job_id':'707140','run_id':'opb-thinkoff-32n-nan-debug-20261030-v2','checks':checks,'failed_checks':[k for k,v in checks.items() if not v],'record_count':len(new),'training_values_checked':sum(sum(k.startswith('train/') for k in r) for r in new),'final_state':state,'audit_status':audit['status'],'schedulers':schedulers,'checkpoint_sample':{k:sample[k] for k in ['tensor_count','total_sampled','total_nonfinite','requested_data_bytes','elapsed_seconds']},'guard_evidence':{'nodes':nodes,'hook_records':len(hooks),'activation_environment_nodes':len(envs),'limitation':'Installation/flags/source verified; per-stage invocation counts are not logged.'},'comparisons':comparisons,'steps':{str(i):{'original':clean_record(oldmap[i]),'diagnostic':clean_record(newmap[i])} for i in [0,1,2,518,519,520,639]},'warning_counts':{k:len(v) for k,v in warnings.items()},'timing':{'steady_samples':len(steady),'steady_mean_s':statistics.mean(steady),'steady_median_s':statistics.median(steady),'last_10_mean_s':statistics.mean(r['profile/step_ms']/1000 for r in new[-10:]),'note':'Diagnostic with activation checks; not a controlled production throughput benchmark.'},'limitations':['No NaN trigger reproduced through640; initial cause remains unproven.','Instrumented fresh run on different nodes with nondeterministic mode and online hidden-state generation is not exact trajectory replay.','Compiler-disabled hooks and additional collectives can change compilation/synchronization.','Checkpoint finite assessment is a bounded sample, not a full saved-weight or optimizer-state scan.','No held-out validation, full epoch or resume test was performed in this bounded run.']}
(root/'verification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
assert not result['failed_checks'],result['failed_checks']
print(json.dumps({k:result[k] for k in ['record_count','training_values_checked','failed_checks','warning_counts','timing','comparisons']},indent=2))
