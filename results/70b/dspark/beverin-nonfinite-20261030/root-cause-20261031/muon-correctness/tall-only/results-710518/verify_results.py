"""Verify native-abort versus finite-reference evidence; this is not a job pass."""
from pathlib import Path
import hashlib,json,math,re,collections
root=Path(__file__).resolve().parent
collection=json.loads((root/'collection.json').read_text())
for item in collection['files']:
    raw=(root/'raw'/item['path']).read_bytes()
    assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
deployment=root/'raw/deployments/710518'
images={'upstream':'b58797bb6a430d7587f1c10ec24ac71f198bd8d590bde98dbbe68d7744a9816f','alps':'416216c9de21d110fbb7cb12cef90923af51e04c955d7d7dffa9786f125a4928'}
fixtures=('zero','tiny-random','random','rank-skewed-random','rank-one-ones','rank-one-signed')
expected={(shape,fixture) for shape in ((80,16),(43008,8192)) for fixture in fixtures}
report={'job':'710518','job_status':'FAILED','classification':'original-native-illegal-instruction_ALPS-finite-pass','historical_nan_reproduced':False,'raw_files_verified':len(collection['files']),'raw_bytes':sum(x['bytes'] for x in collection['files']),'excluded_large_core_files':collection['excluded_files'],'cells':{}}
for name,image in images.items():
    cell=deployment/name
    assert (cell/'image.sha256').read_text().split()[0]==image
    ranks=[json.loads((cell/('rank%d.json'%rank)).read_text()) for rank in (0,1)]
    text=(cell/'raw.log').read_text()
    assert 'CUDA_VISIBLE_DEVICES=2,3' in (cell/'environment.txt').read_text()
    assert 'MUON_ORACLE_SHAPES=80x16 43008x8192' in (cell/'environment.txt').read_text()
    transport=collections.Counter()
    for line in text.splitlines():
        if 'via ' in line:
            for kind in ('P2P/IPC','SHM','NET/'):
                if kind in line:transport[kind]+=1
    cases=[case for result in ranks for case in result['cases']]
    for rank,result in enumerate(ranks):
        assert result['rank']==rank and result['world_size']==2 and result['production_shard_flip']
        assert result['probe_sha256']=='bfd9286b827baaa44cae17cf6e42a976507b6934f880b37c453ed92f70e6ee63'
        assert result['function_sha256']=='99e2359a95e1a914345207e49f072a658866333f4458b7b45faafa35c628f7c4'
        i=result['argv'].index('--shapes');assert result['argv'][i+1:]==['80x16','43008x8192']
        for case in result['cases']:
            assert case['status']=='passed' and case['input_mismatches']==0 and case['input_nonfinite']==0
            assert all(value==0 for value in case['nonfinite'].values())
            assert math.isfinite(case['relative_l2_error']) and case['relative_l2_error']<=.05
            assert case['norm_relative_error']<=.02 and case['clamp_relative_error']<=.02
    row={'rank_count':2,'recorded_cases_per_rank':[len(x['cases']) for x in ranks],
         'torch':ranks[0]['torch'],'torch_git':ranks[0]['torch_git'],'hip':ranks[0]['hip'],
         'recorded_nonfinite':0,'max_recorded_relative_l2_error':max(c['relative_l2_error'] for c in cases),
         'transport_channel_log_counts':dict(transport),'bus_ids':sorted(set(re.findall(r'busId (\S+)',text)))}
    if name=='upstream':
        assert (cell/'exit-code.txt').read_text().strip()=='1'
        assert text.count('HSA_STATUS_ERROR_ILLEGAL_INSTRUCTION')==2 and 'code: 0x2a' in text
        assert all(len(x['cases'])==6 and x['status']=='running' for x in ranks)
        assert all(tuple(c['shape'])==(80,16) for c in cases)
        assert not any('MUON_ORACLE_PASS' in line and '43008' in line for line in text.splitlines())
        row['result']='two-rank SIGABRT; first large zero case incomplete'
        row['failure_is_numerical_assertion']=False
        row['first_uncompleted_case_inferred_from_source']={'shape':[43008,8192],'fixture':'zero'}
    else:
        assert (cell/'exit-code.txt').read_text().strip()=='0'
        assert all(x['status']=='passed' and len(x['cases'])==12 for x in ranks)
        assert all({(tuple(c['shape']),c['fixture']) for c in x['cases']}==expected for x in ranks)
        row['result']='all tall cases passed'
        row['max_large_shape_relative_l2_error']=max(c['relative_l2_error'] for c in cases if c['shape']==[43008,8192])
        row['large_zero_relative_l2_errors']=[c['relative_l2_error'] for c in cases if c['shape']==[43008,8192] and c['fixture']=='zero']
    report['cells'][name]=row
assert (deployment/'REVISION').read_text().strip()=='0030d3123e3bf7b8fa4f6c41e3615ab57b45203c'
assert 'Requeue=0 Restarts=0' in (deployment/'slurm-job.txt').read_text()
summary=json.loads((deployment/'summary.json').read_text())
assert not summary['upstream']['passed'] and summary['alps']['passed']
assert summary['alps']['expected_shapes']==[[80,16],[43008,8192]]
(root/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
